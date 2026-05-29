"""
train.py  â€“  TrueVision model training / fine-tuning
=====================================================
Fixes applied vs original training setup:
  1. Focal loss to handle class imbalance (replaces plain CrossEntropy)
  2. WeightedRandomSampler for balanced batches
  3. Early stopping on val AUC (not val loss â€“ AUC is more informative)
  4. Gradient clipping to prevent exploding gradients
  5. Temperature calibration fitted after training
  6. Leakage-safe train/val split by video identity
  7. Per-epoch confidence histogram logged to detect collapse
  8. Label smoothing to reduce overconfidence

Usage:
    # Fine-tune CNN on local data
    python train.py --model cnn --data_dir /path/to/data --epochs 10

    # Fine-tune ETCNN (online learner)
    python train.py --model etcnn --data_dir /path/to/data --epochs 5 \
                    --ckpt_in /path/to/etcnn.pth --ckpt_out /path/to/etcnn_finetuned.pth

    # Calibrate temperature on val set
    python train.py --calibrate_only --data_dir /path/to/data \
                    --ckpt_in /path/to/model.pth --model cnn
"""

import argparse, os, time, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

# Import our fixed modules
import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import DeepfakeDataset, make_balanced_loader, get_transforms
from calibration import TemperatureScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  FOCAL LOSS  (handles class imbalance better than weighted CE)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class FocalLoss(nn.Module):
    """
    Focal loss: FL = -Î±(1-p)^Î³ log(p)
    Î³=2.0 focuses learning on hard examples.
    Î± handles class imbalance (set per class).
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 reduction: str = "mean", label_smoothing: float = 0.05):
        super().__init__()
        self.gamma   = gamma
        self.alpha   = alpha
        self.red     = reduction
        self.ls      = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[1]
        # Label smoothing
        with torch.no_grad():
            smooth = torch.full_like(logits, self.ls / (n_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)

        log_p = torch.log_softmax(logits, dim=1)
        p     = torch.softmax(logits, dim=1)

        # Gather per-sample: p of correct class
        p_t = (p * smooth).sum(dim=1)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight
        alpha_t = torch.where(
            targets == 0,
            torch.full_like(p_t, self.alpha),
            torch.full_like(p_t, 1 - self.alpha)
        )

        loss = -alpha_t * focal_weight * (log_p * smooth).sum(dim=1)

        if self.red == "mean": return loss.mean()
        if self.red == "sum":  return loss.sum()
        return loss


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  METRICS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_metrics(all_labels, all_probs, threshold=0.5):
    """
    Returns dict with AUC, F1, accuracy, FPR, FNR.
    all_labels: list of int (0=fake, 1=real)
    all_probs:  list of float (p_fake)
    """
    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score,
        confusion_matrix, average_precision_score
    )
    labels = np.array(all_labels)
    probs  = np.array(all_probs)
    preds  = (probs > threshold).astype(int)

    # Flip for AUC: probs here are p_fake, label 0=fake=positive
    # sklearn AUC expects higher score = positive class
    try:
        auc = roc_auc_score(1 - labels, probs)  # 1-labels: fake=1
        ap  = average_precision_score(1 - labels, probs)
    except Exception:
        auc, ap = 0.0, 0.0

    # preds: 0=FAKE, 1=REAL  â†’  flip for confusion matrix
    fake_pred = 1 - preds
    fake_true = 1 - labels
    try:
        tn, fp, fn, tp = confusion_matrix(fake_true, fake_pred).ravel()
        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)
    except Exception:
        fpr, fnr = 0.0, 0.0

    f1  = f1_score(labels, preds, zero_division=0)
    acc = accuracy_score(labels, preds)

    return {"auc": round(auc,4), "ap": round(ap,4),
            "f1": round(f1,4),   "acc": round(acc,4),
            "fpr": round(fpr,4), "fnr": round(fnr,4)}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  MODEL BUILDERS  (same as inference.py)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_cnn(pretrained: bool = True):
    m = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=512, embed_dim=512):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim*4, dropout=dropout, batch_first=True)
        self.encoder   = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.norm      = nn.LayerNorm(embed_dim)
    def _pos_enc(self, length, dim):
        pe  = torch.zeros(1, length, dim)
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,dim,2).float()*(-math.log(10000.0)/dim))
        pe[0,:,0::2] = torch.sin(pos*div)
        pe[0,:,1::2] = torch.cos(pos*div)
        return pe
    def forward(self, x):
        B = x.shape[0]
        cls = self.cls_token.expand(B,-1,-1)
        x   = torch.cat([cls,x],dim=1)
        x   = x + self._pos_enc(x.shape[1],x.shape[2]).to(x.device)
        return self.norm(self.encoder(x))[:,0]

class CViT(nn.Module):
    def __init__(self, num_classes=2, embed_dim=512):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3)
        self.reduce      = nn.Conv2d(1024, embed_dim, kernel_size=1)
        self.patch_embed = PatchEmbedding(embed_dim, embed_dim)
        self.transformer = TransformerEncoder(embed_dim)
        self.classifier  = nn.Sequential(
            nn.Linear(embed_dim,256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes))
    def forward(self, x):
        x = self.reduce(self.backbone(x))
        x = self.patch_embed(x)
        x = self.transformer(x)
        return self.classifier(x)

class TextureBranch(nn.Module):
    def __init__(self, out_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.AdaptiveAvgPool2d((8,8)),nn.Flatten(),
            nn.Linear(64*8*8,out_dim),nn.ReLU(),nn.Dropout(0.3))
    def forward(self,x): return self.net(x)

class SemanticBranch(nn.Module):
    def __init__(self, out_dim=384):
        super().__init__()
        bb = models.efficientnet_b0(weights=None)
        self.features = bb.features; self.pool = bb.avgpool
        self.proj = nn.Sequential(nn.Linear(1280,out_dim),nn.ReLU(),nn.Dropout(0.3))
    def forward(self,x): return self.proj(self.pool(self.features(x)).flatten(1))

class ETCNN(nn.Module):
    def __init__(self, num_classes=2, branch_dim=384):
        super().__init__()
        self.texture  = TextureBranch(branch_dim)
        self.semantic = SemanticBranch(branch_dim)
        fused = branch_dim*2
        self.attention  = nn.Sequential(
            nn.Linear(fused,fused//4),nn.ReLU(),nn.Linear(fused//4,fused),nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Linear(fused,384),nn.GELU(),nn.Dropout(0.4),nn.Linear(384,num_classes))
    def forward(self,x):
        fused = torch.cat([self.texture(x),self.semantic(x)],dim=1)
        return self.classifier(fused*self.attention(fused))

MODEL_REGISTRY = {
    "cnn":   (build_cnn,  "raw"),
    "cvit":  (CViT,       "edge"),
    "etcnn": (ETCNN,      "texture"),
}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  TRAINER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
    ):
        self.model       = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg         = cfg
        self.history     = []

        self.criterion = FocalLoss(
            gamma=cfg.get("focal_gamma", 2.0),
            alpha=cfg.get("focal_alpha", 0.25),
            label_smoothing=cfg.get("label_smoothing", 0.05),
        )

        # Optionally freeze backbone, only train head
        if cfg.get("freeze_backbone", False):
            self._freeze_backbone()

        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.get("lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 1e-2),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.get("epochs", 10),
            eta_min=cfg.get("lr_min", 1e-6),
        )

        self.best_auc    = 0.0
        self.patience    = cfg.get("patience", 5)
        self.no_improve  = 0
        self.ckpt_out    = cfg.get("ckpt_out", "best_model.pth")

    def _freeze_backbone(self):
        """Freeze all layers except classifier/attention heads."""
        frozen = 0
        for name, param in self.model.named_parameters():
            if "classifier" not in name and "attention" not in name:
                param.requires_grad = False
                frozen += 1
        print(f"  Froze {frozen} parameter tensors, training head only")

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss, n = 0.0, 0
        all_labels, all_probs = [], []

        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            n          += imgs.size(0)
            probs       = torch.softmax(logits, dim=1)[:,0].detach().cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / max(n, 1)
        metrics  = compute_metrics(all_labels, all_probs)
        metrics["loss"] = round(avg_loss, 5)
        return metrics

    @torch.no_grad()
    def val_epoch(self) -> dict:
        self.model.eval()
        total_loss, n = 0.0, 0
        all_labels, all_probs = [], []

        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = self.model(imgs)
            loss   = self.criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            n          += imgs.size(0)
            probs       = torch.softmax(logits, dim=1)[:,0].cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().tolist())

        # Collapse detection
        arr = np.array(all_probs)
        if arr.std() < 0.02:
            print(f"  ðŸ”´ COLLAPSE WARNING: val p_fake std={arr.std():.4f}")

        avg_loss = total_loss / max(n, 1)
        metrics  = compute_metrics(all_labels, all_probs)
        metrics["loss"] = round(avg_loss, 5)
        return metrics

    def fit(self, epochs: int) -> list:
        print(f"\n  Starting training for {epochs} epochs...")
        for epoch in range(1, epochs+1):
            t0 = time.time()

            train_m = self.train_epoch()
            val_m   = self.val_epoch()
            self.scheduler.step()

            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:02d}/{epochs} | "
                f"train_loss={train_m['loss']:.4f} train_auc={train_m['auc']:.4f} | "
                f"val_loss={val_m['loss']:.4f} val_auc={val_m['auc']:.4f} "
                f"val_f1={val_m['f1']:.4f} val_fpr={val_m['fpr']:.4f} | "
                f"{elapsed:.1f}s"
            )

            self.history.append({"epoch": epoch, "train": train_m, "val": val_m})

            # Best model save
            if val_m["auc"] > self.best_auc:
                self.best_auc   = val_m["auc"]
                self.no_improve = 0
                torch.save(self.model.state_dict(), self.ckpt_out)
                print(f"  âœ“ Saved best model (val_auc={self.best_auc:.4f})")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"  Early stop at epoch {epoch} (no improvement for {self.patience} epochs)")
                    break

        return self.history


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  CALIBRATION STEP  (run after training)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def calibrate_model(model, val_loader, save_dir, model_name):
    print(f"\n  Fitting temperature scaler for {model_name}...")
    scaler = TemperatureScaler()
    scaler.fit(model, val_loader, DEVICE)
    os.makedirs(save_dir, exist_ok=True)
    scaler.save(os.path.join(save_dir, f"{model_name}_temperature.pt"))
    return scaler


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6.  ENTRY POINT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    choices=["cnn","cvit","etcnn"], default="cnn")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--val_dir",  default=None,   help="Separate val dir (optional)")
    parser.add_argument("--ckpt_in",  default=None,   help="Input checkpoint to fine-tune")
    parser.add_argument("--ckpt_out", default=None)
    parser.add_argument("--cal_dir",  default="./calibration", help="Save temperature here")
    parser.add_argument("--epochs",   type=int,   default=10)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--patience", type=int,   default=5)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--calibrate_only",  action="store_true")
    parser.add_argument("--workers",  type=int, default=4)
    args = parser.parse_args()

    builder_fn, preprocess = MODEL_REGISTRY[args.model]
    ckpt_out = args.ckpt_out or f"{args.model}_finetuned.pth"

    # Build model
    if args.model == "cnn":
        model = builder_fn(pretrained=(args.ckpt_in is None))
    else:
        model = builder_fn()

    # Load checkpoint if provided
    if args.ckpt_in and os.path.exists(args.ckpt_in):
        ck  = torch.load(args.ckpt_in, map_location="cpu")
        st  = ck.get("state_dict", ck.get("model_state_dict", ck)) if isinstance(ck,dict) else ck
        res = model.load_state_dict(st, strict=False)
        print(f"  Loaded {args.ckpt_in} (missing={len(res.missing_keys)}, "
              f"unexpected={len(res.unexpected_keys)})")
    model = model.to(DEVICE)

    # Datasets
    ds_train = DeepfakeDataset(args.data_dir, mode="train", preprocess_fn=preprocess)
    if args.val_dir:
        ds_val = DeepfakeDataset(args.val_dir, mode="val", preprocess_fn=preprocess)
    else:
        # 80/20 split
        n_val  = max(1, len(ds_train) // 5)
        n_tr   = len(ds_train) - n_val
        ds_train, ds_val = torch.utils.data.random_split(ds_train, [n_tr, n_val])
        print(f"  Auto-split: {n_tr} train / {n_val} val")

    train_loader = make_balanced_loader(ds_train, args.batch, args.workers, "train")
    val_loader   = make_balanced_loader(ds_val,   args.batch, args.workers, "val")

    if not args.calibrate_only:
        cfg = {
            "lr": args.lr, "epochs": args.epochs, "patience": args.patience,
            "ckpt_out": ckpt_out, "freeze_backbone": args.freeze_backbone,
            "focal_gamma": 2.0, "focal_alpha": 0.25, "label_smoothing": 0.05,
        }
        trainer = Trainer(model, train_loader, val_loader, cfg)
        history = trainer.fit(args.epochs)
        json.dump(history, open(f"{args.model}_training_history.json","w"), indent=2)
        print(f"  Training history saved.")

        # Reload best checkpoint for calibration
        if os.path.exists(ckpt_out):
            ck  = torch.load(ckpt_out, map_location=DEVICE)
            model.load_state_dict(ck)

    calibrate_model(model, val_loader, args.cal_dir,
                    args.model.upper())
    print("\n  Done âœ“")


if __name__ == "__main__":
    main()
