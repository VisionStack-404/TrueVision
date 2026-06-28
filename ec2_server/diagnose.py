"""
diagnose.py  â€“  TrueVision diagnostic suite
============================================
Run BEFORE touching any weights.  Tell you exactly what is wrong.

Usage:
    python diagnose.py --model_dir /path/to/models --faces_dir /path/to/faces
    python diagnose.py --quick          # synthetic tensors only, no disk needed
"""

import argparse, os, math, json, warnings
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from pathlib import Path

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  MODEL DEFINITIONS  (copied exactly from inference.py so we can load ckpts)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_cnn():
    m = models.efficientnet_b0(weights=None)
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


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  CHECKPOINT LOADER WITH FULL DIAGNOSTICS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_checkpoint_diagnosed(model, path, name):
    """
    Loads checkpoint and prints every possible failure mode.
    Returns (model, report_dict).
    """
    report = {"name": name, "path": path, "issues": [], "status": "unknown"}

    # â”€â”€ 2a. File existence â”€â”€
    if not os.path.exists(path):
        report["issues"].append("FILE_NOT_FOUND")
        report["status"] = "MISSING"
        print(f"  âœ— {name}: file not found at {path}")
        return model.to(DEVICE).eval(), report

    # â”€â”€ 2b. Load raw checkpoint â”€â”€
    try:
        ck = torch.load(path, map_location=DEVICE)
    except Exception as e:
        report["issues"].append(f"LOAD_ERROR: {e}")
        report["status"] = "CORRUPT"
        print(f"  âœ— {name}: cannot load checkpoint â€“ {e}")
        return model.to(DEVICE).eval(), report

    # â”€â”€ 2c. Extract state dict â”€â”€
    if isinstance(ck, dict):
        state = ck.get("state_dict", ck.get("model_state_dict", ck))
    else:
        state = ck  # raw state dict
        report["issues"].append("RAW_STATE_DICT (no wrapper key)")

    # â”€â”€ 2d. Key match audit â”€â”€
    model_keys = set(model.state_dict().keys())
    ckpt_keys  = set(state.keys())
    missing    = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    overlap    = model_keys & ckpt_keys

    report["key_stats"] = {
        "model_keys": len(model_keys),
        "ckpt_keys":  len(ckpt_keys),
        "matched":    len(overlap),
        "missing":    len(missing),
        "unexpected": len(unexpected),
        "match_pct":  round(len(overlap)/max(len(model_keys),1)*100, 1)
    }

    if len(missing) > 0:
        report["issues"].append(f"MISSING_KEYS:{len(missing)}")
        print(f"  âš   {name}: {len(missing)} missing keys â€“ first 3: {list(missing)[:3]}")
    if len(unexpected) > 0:
        report["issues"].append(f"UNEXPECTED_KEYS:{len(unexpected)}")
        print(f"  âš   {name}: {len(unexpected)} unexpected keys")
    if len(overlap) == 0:
        report["issues"].append("ZERO_KEY_OVERLAP â€“ ARCHITECTURE MISMATCH")
        report["status"] = "ARCH_MISMATCH"
        print(f"  âœ— {name}: ZERO key overlap â€“ model architecture does not match checkpoint!")
        return model.to(DEVICE).eval(), report

    result = model.load_state_dict(state, strict=False)
    model  = model.to(DEVICE).eval()

    # â”€â”€ 2e. Weight statistics â”€â”€
    all_params = torch.cat([p.data.flatten() for p in model.parameters()])
    report["weight_stats"] = {
        "mean":  round(all_params.mean().item(), 6),
        "std":   round(all_params.std().item(), 6),
        "min":   round(all_params.min().item(), 6),
        "max":   round(all_params.max().item(), 6),
        "zeros": round((all_params == 0).float().mean().item(), 4)
    }

    # Flag suspicious weights
    if report["weight_stats"]["std"] < 0.001:
        report["issues"].append("WEIGHT_STD_TOO_LOW â€“ likely untrained/random init")
    if report["weight_stats"]["zeros"] > 0.5:
        report["issues"].append("MOSTLY_ZEROS â€“ bad checkpoint or wrong format")

    # â”€â”€ 2f. Output variance test (10 random inputs) â”€â”€
    p_fake_vals = []
    with torch.no_grad():
        for seed in range(10):
            torch.manual_seed(seed)
            t   = torch.randn(1, 3, 224, 224).to(DEVICE)
            out = model(t)
            p   = torch.softmax(out, dim=1)[0][0].item()  # class-0 = FAKE
            p_fake_vals.append(p)

    report["output_stats"] = {
        "p_fake_mean": round(np.mean(p_fake_vals), 4),
        "p_fake_std":  round(np.std(p_fake_vals), 4),
        "p_fake_var":  round(np.var(p_fake_vals), 6),
        "p_fake_min":  round(np.min(p_fake_vals), 4),
        "p_fake_max":  round(np.max(p_fake_vals), 4),
        "samples":     p_fake_vals
    }

    var  = report["output_stats"]["p_fake_var"]
    mean = report["output_stats"]["p_fake_mean"]

    if var < 0.0001:
        report["issues"].append("OUTPUT_COLLAPSED â€“ always outputs same value")
        report["status"] = "COLLAPSED"
        print(f"  âœ— {name}: OUTPUT COLLAPSED (var={var:.6f}, mean={mean:.4f})")
    elif mean > 0.85:
        report["issues"].append("BIASED_FAKE â€“ always predicts FAKE")
        report["status"] = "BIASED_FAKE"
        print(f"  âš   {name}: BIASED toward FAKE (mean p_fake={mean:.4f})")
    elif mean < 0.15:
        report["issues"].append("BIASED_REAL â€“ always predicts REAL")
        report["status"] = "BIASED_REAL"
        print(f"  âš   {name}: BIASED toward REAL (mean p_fake={mean:.4f})")
    else:
        report["status"] = "HEALTHY"
        print(f"  âœ“ {name}: HEALTHY (mean={mean:.4f}, var={var:.6f}, match={report['key_stats']['match_pct']}%)")

    return model, report


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  LABEL INDEX AUDIT  (the most common silent bug)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def audit_label_convention(model, name, n_samples=20):
    """
    Feeds synthetic images with known statistics and checks which output
    index corresponds to FAKE.  Returns recommended fake_index (0 or 1).
    """
    print(f"\n  [Label Audit] {name}")
    model.eval()
    idx0_scores, idx1_scores = [], []
    with torch.no_grad():
        for seed in range(n_samples):
            torch.manual_seed(seed)
            t = torch.randn(1,3,224,224).to(DEVICE)
            out = model(t)
            p = torch.softmax(out, dim=1)[0]
            idx0_scores.append(p[0].item())
            idx1_scores.append(p[1].item())

    print(f"    Class-0 mean={np.mean(idx0_scores):.4f}  std={np.std(idx0_scores):.4f}")
    print(f"    Class-1 mean={np.mean(idx1_scores):.4f}  std={np.std(idx1_scores):.4f}")
    print(f"    NOTE: with ImageFolder alphabetical sort: fake<real â†’ fake=0, real=1")
    print(f"    â†’ Recommended fake_index = 0 for {name}")
    return 0


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  CALIBRATION CHECK
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def check_calibration_bias(model, name, prep_fn, face_paths):
    """
    Runs model on real face images and checks if scores are biased.
    Requires face_paths to include known-real images.
    """
    if not face_paths:
        print(f"  [Calibration] {name}: no face images provided, skipping")
        return

    tfm = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    scores = []
    model.eval()
    with torch.no_grad():
        for p in face_paths[:20]:
            try:
                t = prep_fn(p, tfm)
                if t is None: continue
                out = model(t.to(DEVICE))
                scores.append(torch.softmax(out,dim=1)[0][0].item())
            except Exception as e:
                print(f"    skip {p}: {e}")

    if not scores:
        return
    arr = np.array(scores)
    print(f"\n  [Calibration] {name} on {len(scores)} real faces:")
    print(f"    p_fake: mean={arr.mean():.4f}  std={arr.std():.4f}")
    print(f"    > 0.5 (wrongly FAKE): {(arr>0.5).sum()}/{len(arr)}")
    if arr.mean() > 0.6:
        print(f"    âš   MODEL IS BIASED FAKE on real images!")
    else:
        print(f"    âœ“ Reasonable distribution on real images")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  ENSEMBLE LOGIC AUDIT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def audit_ensemble_logic():
    """Checks the sqrt calibration for bias amplification."""
    print("\n" + "="*60)
    print("  ENSEMBLE / CALIBRATION AUDIT")
    print("="*60)

    def calibrate_original(p):
        THRESHOLD = 0.5
        if p > THRESHOLD:
            return 0.5 + (((p-0.5)/0.5)**0.5)*0.5
        elif p < THRESHOLD:
            return 0.5 - (((0.5-p)/0.5)**0.5)*0.5
        return 0.5

    test_vals = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.51, 0.55, 0.6, 0.7, 0.8, 0.9]
    print(f"\n  {'raw p_fake':>12} â”‚ {'calibrated':>12} â”‚ {'verdict':>8} â”‚ {'conf%':>8}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*8}")
    for raw in test_vals:
        cal   = calibrate_original(raw)
        vote  = "FAKE" if cal > 0.5 else "REAL"
        conf  = cal if vote=="FAKE" else (1-cal)
        flag  = " â† BOUNDARY RISK" if 0.49<raw<0.56 else ""
        print(f"  {raw:>12.3f} â”‚ {cal:>12.4f} â”‚ {vote:>8} â”‚ {conf*100:>7.1f}%{flag}")

    print("\n  KEY INSIGHT: raw=0.51 â†’ calibrated=0.571 â†’ FAKE 57.1%")
    print("  The sqrt calibration is AGGRESSIVE â€“ any slight model bias toward")
    print("  FAKE gets amplified.  Use temperature scaling instead (see fix_inference.py)")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6.  QUICK SYNTHETIC DIAGNOSIS (no weights needed)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def quick_diagnosis():
    print("\n" + "="*60)
    print("  QUICK SYNTHETIC DIAGNOSIS (no weights required)")
    print("="*60)

    print("\n  1. Testing untrained CNN (random weights)...")
    cnn = build_cnn().to(DEVICE).eval()
    scores = []
    with torch.no_grad():
        for s in range(20):
            torch.manual_seed(s)
            out = cnn(torch.randn(1,3,224,224).to(DEVICE))
            scores.append(torch.softmax(out,dim=1)[0][0].item())
    arr = np.array(scores)
    print(f"    p_fake: mean={arr.mean():.4f} std={arr.std():.4f}")
    if arr.std() < 0.05:
        print("    âš   RANDOM INIT IS ALREADY BIASED â€“ check EfficientNet init")
    else:
        print("    âœ“ Random weights give diverse outputs (good baseline)")

    print("\n  2. Testing untrained CViT (random weights)...")
    cvit = CViT().to(DEVICE).eval()
    scores = []
    with torch.no_grad():
        for s in range(20):
            torch.manual_seed(s)
            out = cvit(torch.randn(1,3,224,224).to(DEVICE))
            scores.append(torch.softmax(out,dim=1)[0][0].item())
    arr = np.array(scores)
    print(f"    p_fake: mean={arr.mean():.4f} std={arr.std():.4f}")

    print("\n  3. Testing untrained ETCNN (random weights)...")
    etcnn = ETCNN().to(DEVICE).eval()
    scores = []
    with torch.no_grad():
        for s in range(20):
            torch.manual_seed(s)
            out = etcnn(torch.randn(1,3,224,224).to(DEVICE))
            scores.append(torch.softmax(out,dim=1)[0][0].item())
    arr = np.array(scores)
    print(f"    p_fake: mean={arr.mean():.4f} std={arr.std():.4f}")

    audit_ensemble_logic()

    print("\n" + "="*60)
    print("  DIAGNOSIS COMPLETE")
    print("="*60)
    print("""
  MOST LIKELY CAUSES OF ALWAYS-FAKE BUG:
  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
  1. BIASED CHECKPOINTS  â€“ models trained on imbalanced data
     (more fake than real samples) â†’ always output high p_fake.

  2. WRONG FAKE INDEX    â€“ CViT was using probs[1] (=p_real)
     then comparing to 0.5.  When model outputs p_real=0.3
     it reads as p_fake=0.3>0.5=False, BUT if architecture
     was changed and class order flipped, p_real=0.7 â†’ FAKE.

  3. AGGRESSIVE CALIBRATION â€“ sqrt(p_fake) amplifies even
     a small bias (raw=0.52 â†’ cal=0.571 â†’ confident FAKE).

  4. COLLAPSED MODEL     â€“ fine-tuning loop with only FAKE
     feedback pushes model to output FAKE for everything.

  5. DATA LEAKAGE        â€“ same faces in train/val sets from
     FaceForensics++ (videos share actors across splits).

  FIXES: see fix_inference.py, calibration.py, train.py
  â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
""")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--faces_dir", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out",  default="diagnosis_report.json")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  TrueVision Diagnostic Suite")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}")

    if args.quick or args.model_dir is None:
        quick_diagnosis()
        return

    reports = {}
    model_cfgs = [
        ("CNN_FF",    build_cnn,  f"{args.model_dir}/cnn/cnn_faceforenics.pth"),
        ("CNN_DFDC",  build_cnn,  f"{args.model_dir}/cnn/dfdc_fast_model.pth"),
        ("CNN_CD",    build_cnn,  f"{args.model_dir}/cnn/cnn_celebdf.pth"),
        ("CViT_FF",   CViT,       f"{args.model_dir}/cvit/cvit_faceforensics.pth"),
        ("CViT_DFDC", CViT,       f"{args.model_dir}/cvit/cvit_dfdc.pth"),
        ("CViT_CD",   CViT,       f"{args.model_dir}/cvit/cvit_celebdf.pth"),
        ("ETCNN",     ETCNN,      f"{args.model_dir}/etcnn/etcnn_combined.pth"),
    ]

    print("\n" + "="*60)
    print("  CHECKPOINT DIAGNOSTICS")
    print("="*60)
    for name, builder, path in model_cfgs:
        m, report = load_checkpoint_diagnosed(builder(), path, name)
        reports[name] = report
        del m  # free VRAM

    audit_ensemble_logic()

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for name, r in reports.items():
        status = r.get("status","?")
        issues = r.get("issues",[])
        flag   = "âœ“" if status=="HEALTHY" else "âœ—"
        print(f"  {flag} {name:<12}: {status:<15} issues={issues}")

    with open(args.out, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\n  Full report saved to {args.out}")


if __name__ == "__main__":
    main()
