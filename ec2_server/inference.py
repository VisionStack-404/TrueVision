import torch
import torch.nn as nn
import torch.optim as optim
import cv2, os, math, json, shutil, uuid
import numpy as np
from torchvision import transforms, models
from PIL import Image
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {device}")
THRESHOLD = 0.5

# ================================================
# CALIBRATION — stretches score toward 80%+
# raw=0.30 → REAL 81.6%
# raw=0.70 → FAKE 81.6%
# raw=0.50 → uncertain 50% (genuinely unsure)
# ================================================
def _calibrate(p_fake: float) -> float:
    """
    Square-root calibration that amplifies confidence away from 0.5.
    Symmetric around 0.5 so it does NOT introduce directional bias.
    """
    if p_fake > THRESHOLD:
        cal = 0.5 + (((p_fake - 0.5) / 0.5) ** 0.5) * 0.5
    elif p_fake < THRESHOLD:
        cal = 0.5 - (((0.5 - p_fake) / 0.5) ** 0.5) * 0.5
    else:
        # Exactly 0.5 → genuinely uncertain, don't push either way
        cal = 0.5
    return float(np.clip(cal, 0.0, 1.0))


# ================================================
# TRANSFORMS
# ================================================
_base_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def _upscale(img):
    h, w = img.shape[:2]
    if w < 80 or h < 80:
        img = cv2.resize(img, (128, 128),
                         interpolation=cv2.INTER_CUBIC)
    return img

def _prep_cnn(path: str):
    """Raw pixels → GAN/blending artifact detection."""
    img = cv2.imread(path)
    if img is None: return None
    img = _upscale(img)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return _base_tfm(img).unsqueeze(0).to(device)

def _prep_cvit(path: str):
    """Edge-enhanced → structural boundary detection."""
    img = cv2.imread(path)
    if img is None: return None
    img      = _upscale(img)
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges    = cv2.Laplacian(gray, cv2.CV_64F)
    edges    = np.uint8(np.clip(np.abs(edges), 0, 255))
    edges_3c = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    enhanced = cv2.addWeighted(img, 0.7, edges_3c, 0.3, 0)
    img      = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    return _base_tfm(img).unsqueeze(0).to(device)

def _prep_etcnn(path: str):
    """Texture map → skin texture anomaly detection."""
    img = cv2.imread(path)
    if img is None: return None
    img     = _upscale(img)
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    texture = cv2.addWeighted(img, 1.5, blurred, -0.5, 128)
    img     = Image.fromarray(cv2.cvtColor(texture, cv2.COLOR_BGR2RGB))
    return _base_tfm(img).unsqueeze(0).to(device)


# ================================================
# MODEL DEFINITIONS
# ================================================
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
    def __init__(self, embed_dim=512, num_heads=8,
                 num_layers=4, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder   = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm      = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B   = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self._pos_enc(x.shape[1], x.shape[2]).to(x.device)
        return self.norm(self.encoder(x))[:, 0]

    def _pos_enc(self, length, dim):
        pe  = torch.zeros(1, length, dim)
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe

class CViT(nn.Module):
    def __init__(self, num_classes=2, embed_dim=512):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1,
            resnet.layer2, resnet.layer3
        )
        self.reduce      = nn.Conv2d(1024, embed_dim, kernel_size=1)
        self.patch_embed = PatchEmbedding(embed_dim, embed_dim)
        self.transformer = TransformerEncoder(embed_dim)
        self.classifier  = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.reduce(self.backbone(x))
        x = self.patch_embed(x)
        x = self.transformer(x)
        return self.classifier(x)

class TextureBranch(nn.Module):
    def __init__(self, out_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten(),
            nn.Linear(64 * 8 * 8, out_dim), nn.ReLU(), nn.Dropout(0.3)
        )
    def forward(self, x): return self.net(x)

class SemanticBranch(nn.Module):
    def __init__(self, out_dim=384):
        super().__init__()
        bb            = models.efficientnet_b0(weights=None)
        self.features = bb.features
        self.pool     = bb.avgpool
        self.proj     = nn.Sequential(
            nn.Linear(1280, out_dim), nn.ReLU(), nn.Dropout(0.3)
        )
    def forward(self, x):
        return self.proj(self.pool(self.features(x)).flatten(1))

class ETCNN(nn.Module):
    def __init__(self, num_classes=2, branch_dim=384):
        super().__init__()
        self.texture   = TextureBranch(branch_dim)
        self.semantic  = SemanticBranch(branch_dim)
        fused = branch_dim * 2
        self.attention = nn.Sequential(
            nn.Linear(fused, fused // 4), nn.ReLU(),
            nn.Linear(fused // 4, fused), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(fused, 384), nn.GELU(),
            nn.Dropout(0.4), nn.Linear(384, num_classes)
        )
    def forward(self, x):
        fused = torch.cat([self.texture(x), self.semantic(x)], dim=1)
        return self.classifier(fused * self.attention(fused))


# ================================================
# MODEL LOADER — with startup variance check
# ✅ FIX 1: Log strict=False mismatches
# ✅ FIX 2: Move test tensor to device
# ================================================
def _load(model_obj, path, name):
    if not os.path.exists(path):
        print(f"  ⚠️  NOT FOUND: {path}")
        return model_obj.to(device).eval()

    ck    = torch.load(path, map_location=device)
    state = ck.get("state_dict", ck.get("model_state_dict", ck))

    # ✅ FIX: Log mismatched keys so we know if weights actually loaded
    result = model_obj.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"  ⚠️  {name} MISSING KEYS ({len(result.missing_keys)}): "
              f"{result.missing_keys[:5]}...")
    if result.unexpected_keys:
        print(f"  ⚠️  {name} UNEXPECTED KEYS ({len(result.unexpected_keys)}): "
              f"{result.unexpected_keys[:5]}...")
    if not result.missing_keys and not result.unexpected_keys:
        print(f"  ✅ {name} weights loaded perfectly (all keys matched)")

    model_obj = model_obj.to(device).eval()

    # Variance check — runs 5 random inputs
    # ✅ FIX: tensor now moved to correct device
    p0_vals = []
    with torch.no_grad():
        for _ in range(5):
            t   = torch.randn(1, 3, 224, 224).to(device)  # ✅ FIX: .to(device)
            out = model_obj(t)
            p0_vals.append(torch.softmax(out, dim=1)[0][0].item())

    avg = float(np.mean(p0_vals))
    var = float(np.var(p0_vals))

    if var < 0.0001:
        status = "🔴 COLLAPSED (always same output — retrain needed)"
    elif avg > 0.85:
        status = "🟡 BIASED→FAKE (imbalanced training data)"
    elif avg < 0.15:
        status = "🟡 BIASED→REAL (labels may be flipped)"
    else:
        status = "✅ HEALTHY"

    print(f"  {name}: {status} | avg_p0={avg:.3f} var={var:.5f}")
    return model_obj


# ================================================
# LOAD ALL WEIGHTS
# ================================================
BASE              = "/home/ubuntu/truevision/models"
ETCNN_ONLINE_PATH = f"{BASE}/etcnn/etcnn_combined.pth"

print("\n" + "="*52)
print("  LOADING MODELS — startup variance check")
print("="*52)
_cnn_ff    = _load(build_cnn(), f"{BASE}/cnn/cnn_faceforenics.pth",  "CNN_FF  ")
_cnn_dfdc  = _load(build_cnn(), f"{BASE}/cnn/dfdc_fast_model.pth",   "CNN_DFDC")
_cnn_cd    = _load(build_cnn(), f"{BASE}/cnn/cnn_celebdf.pth",       "CNN_CD  ")
_cvit_ff   = _load(CViT(),      f"{BASE}/cvit/cvit_faceforensics.pth","CViT_FF ")
_cvit_dfdc = _load(CViT(),      f"{BASE}/cvit/cvit_dfdc.pth",        "CViT_DFC")
_cvit_cd   = _load(CViT(),      f"{BASE}/cvit/cvit_celebdf.pth",     "CViT_CD ")
_etcnn     = _load(ETCNN(),     ETCNN_ONLINE_PATH,                    "ETCNN   ")
print("="*52 + "\n")


# ================================================
# LABEL & INDEX CONVENTION
# ================================================
# ImageFolder sorts alphabetically:
#   'fake' < 'real'  →  fake=0, real=1
#
# ✅ ALL models trained with ImageFolder → fake=0, real=1
# ✅ Therefore p_fake = probs[0] for ALL models
#
# 🔴 PREVIOUS BUG: CViT used probs[1] which gave p_real,
#    causing CViT to vote FAKE when image was actually REAL.
# ================================================
_fake_index = {
    "cnn":   0,   # ✅ fake=class 0
    "cvit":  0,   # ✅ FIX: was 1 (WRONG!) — now 0 like all others
    "etcnn": 0    # ✅ fake=class 0
}

LABEL_MAP = {
    "FAKE": 0,    # ✅ matches training convention
    "REAL": 1
}


# ================================================
# PATHS
# ================================================
BASE_DIR       = "/home/ubuntu/truevision"
SESSION_FILE   = f"{BASE_DIR}/session_faces.json"
SESSION_BATCH  = f"{BASE_DIR}/session_batch.json"
FEEDBACK_LOG   = f"{BASE_DIR}/feedback_log.json"
DATASET_BASE   = f"{BASE_DIR}/user_dataset"
AUTO_FAKE      = f"{DATASET_BASE}/auto/fake"
AUTO_REAL      = f"{DATASET_BASE}/auto/real"
CONFIRMED_FAKE = f"{DATASET_BASE}/confirmed/fake"
CONFIRMED_REAL = f"{DATASET_BASE}/confirmed/real"
META_FILE      = f"{DATASET_BASE}/metadata.json"

for d in [AUTO_FAKE, AUTO_REAL, CONFIRMED_FAKE, CONFIRMED_REAL]:
    os.makedirs(d, exist_ok=True)


# ================================================
# SESSION HELPERS
# ================================================
def _save_session(face_paths: list):
    with open(SESSION_FILE, "w") as f:
        json.dump(face_paths, f)

def _load_session() -> list:
    try:
        with open(SESSION_FILE) as f:
            return json.load(f)
    except Exception:
        return []

def _save_batch(batch_id: str):
    with open(SESSION_BATCH, "w") as f:
        json.dump({"batch_id": batch_id}, f)

def _load_batch() -> str:
    try:
        with open(SESSION_BATCH) as f:
            return json.load(f).get("batch_id", "unknown")
    except Exception:
        return "unknown"


# ================================================
# CORE: run one model on face list
# ================================================
def _score_faces(model, face_paths, prep_fn,
                 model_type, label,
                 max_faces=10):
    raw_scores = []

    for path in face_paths[:max_faces]:
        t = prep_fn(path)
        if t is None:
            continue
        with torch.no_grad():
            out   = model(t)
            probs = torch.softmax(out, dim=1)[0]
            idx   = _fake_index[model_type]
            raw   = probs[idx].item()

        raw_scores.append(raw)

    if not raw_scores:
        return None, {}

    avg_raw = sum(raw_scores) / len(raw_scores)
    avg_cal = _calibrate(avg_raw)
    # ✅ FIX: use > instead of >= so 0.5 = uncertain, not FAKE
    vote    = "FAKE" if avg_cal > THRESHOLD else "REAL"
    conf    = avg_cal if vote == "FAKE" else (1 - avg_cal)

    print(f"    [{label}] raw={avg_raw:.3f} "
          f"→ {vote} {conf*100:.1f}%")

    return avg_raw, {"raw": round(avg_raw, 4)}


# ================================================
# 3 BALANCED MODEL RUNNERS
# CNN=33%  CViT=33%  ETCNN=34%
# ================================================
def _run_cnn(face_paths):
    print("\n  ── CNN (raw pixels) ─────────────────")
    ds_names  = ["faceforensics", "dfdc", "celebdf"]
    ds_models = [_cnn_ff, _cnn_dfdc, _cnn_cd]
    scores, ds = [], {}

    for name, m in zip(ds_names, ds_models):
        avg, info = _score_faces(
            m, face_paths, _prep_cnn, "cnn", f"CNN-{name}"
        )
        if avg is not None:
            scores.append(avg)
            ds[name] = info

    if not scores:
        return _unknown("CNN", "Face-swap & GAN artifact detection")

    final = sum(scores) / len(scores)
    return _result("CNN", final, len(face_paths),
                   "Face-swap & GAN artifact detection", ds)


def _run_cvit(face_paths):
    print("\n  ── CViT (edge-enhanced) ─────────────")
    ds_names  = ["faceforensics", "dfdc", "celebdf"]
    ds_models = [_cvit_ff, _cvit_dfdc, _cvit_cd]
    scores, ds = [], {}

    for name, m in zip(ds_names, ds_models):
        avg, info = _score_faces(
            m, face_paths, _prep_cvit, "cvit", f"CViT-{name}"
        )
        if avg is not None:
            scores.append(avg)
            ds[name] = info

    if not scores:
        return _unknown("CViT",
                        "Structural & boundary inconsistency detection")

    final = sum(scores) / len(scores)
    return _result("CViT", final, len(face_paths),
                   "Structural & boundary inconsistency detection", ds)


def _run_etcnn(face_paths):
    print("\n  ── ETCNN (texture map) ──────────────")
    avg, info = _score_faces(
        _etcnn, face_paths, _prep_etcnn, "etcnn", "ETCNN"
    )
    if avg is None:
        return _unknown("ETCNN",
                        "Skin texture & AI-generation detection")

    return _result("ETCNN", avg, len(face_paths),
                   "Skin texture & AI-generation detection")


def _result(model, p_fake_raw, face_count, speciality, ds=None):
    p_fake_cal = _calibrate(p_fake_raw)
    # ✅ FIX: use > instead of >= so 0.5 = uncertain
    vote = "FAKE" if p_fake_cal > THRESHOLD else "REAL"
    conf = p_fake_cal if vote == "FAKE" else (1 - p_fake_cal)
    r = {
        "model":          model,
        "speciality":     speciality,
        "vote":           vote,
        "confidence_pct": round(conf * 100, 1),
        "p_fake":         round(p_fake_raw, 4),
        "p_real":         round(1 - p_fake_raw, 4),
        "faces_used":     min(face_count, 10)
    }
    if ds:
        r["dataset_scores"] = ds
    return r

def _unknown(model, speciality):
    return {
        "model": model, "speciality": speciality,
        "vote": "UNKNOWN", "confidence_pct": 0.0,
        "p_fake": None, "p_real": None, "faces_used": 0
    }


# ================================================
# RUN ALL 3
# ================================================
def run_all_models(face_paths: list) -> list[dict]:
    _save_session(face_paths)
    print("\n" + "="*52)
    print("  INFERENCE — CNN + CViT + ETCNN")
    print("="*52)
    return [
        _run_cnn(face_paths),
        _run_cvit(face_paths),
        _run_etcnn(face_paths),
    ]


# ================================================
# FINAL PREDICTION
# Equal ensemble: CNN×33% + CViT×33% + ETCNN×34%
# Calibrated score → 80%+ target
# ================================================
def get_final_prediction(model_results: list) -> dict:
    valid = [
        r for r in model_results
        if r.get("p_fake") is not None
        and r.get("vote") != "UNKNOWN"
    ]

    if not valid:
        return {
            "prediction": "UNKNOWN", "confidence_pct": 0.0,
            "decided_by": "none",
            "votes": {"FAKE": 0, "REAL": 0, "total": 0}
        }

    w_map = {"CNN": 0.33, "CViT": 0.33, "ETCNN": 0.34}
    w_sum = sum(r["p_fake"] * w_map.get(r["model"], 0.33)
                for r in valid)
    w_tot = sum(w_map.get(r["model"], 0.33) for r in valid)

    ensemble     = w_sum / w_tot
    ensemble_cal = _calibrate(ensemble)        # stretch to 80%+
    # ✅ FIX: use > instead of >= so 0.5 = uncertain
    prediction   = "FAKE" if ensemble_cal > THRESHOLD else "REAL"
    confidence   = (
        ensemble_cal if prediction == "FAKE"
        else (1.0 - ensemble_cal)
    )
    fake_n = sum(1 for r in valid if r["vote"] == "FAKE")
    real_n = sum(1 for r in valid if r["vote"] == "REAL")

    print(f"\n{'='*52}")
    print(f"  FINAL → {prediction}  {round(confidence*100,1)}% confident")
    print(f"  raw_ensemble={ensemble:.4f} → calibrated={ensemble_cal:.4f}")
    print(f"  votes: CNN={[r['vote'] for r in valid if r['model']=='CNN']} "
          f"CViT={[r['vote'] for r in valid if r['model']=='CViT']} "
          f"ETCNN={[r['vote'] for r in valid if r['model']=='ETCNN']}")
    print("="*52 + "\n")

    return {
        "prediction":     prediction,
        "confidence_pct": round(confidence * 100, 1),
        "decided_by":     "Equal Ensemble (CNN×33% + CViT×33% + ETCNN×34%)",
        "votes": {"FAKE": fake_n, "REAL": real_n, "total": len(valid)}
    }


# ================================================
# BACKGROUND DATASET — silent
# ================================================
def _load_meta():
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE) as f: return json.load(f)
        except Exception: pass
    return []

def _save_meta(meta):
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

def _auto_save(face_paths, prediction, filename, confidence):
    label      = prediction.upper()
    target_dir = AUTO_FAKE if label == "FAKE" else AUTO_REAL
    batch_id   = str(uuid.uuid4())[:8]
    for i, src in enumerate(face_paths[:10]):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(
                target_dir, f"{batch_id}_face_{i}.jpg"))
    meta = _load_meta()
    meta.append({
        "batch_id": batch_id,
        "timestamp": datetime.now().isoformat(),
        "source_file": filename, "label": label,
        "confidence_pct": confidence, "confirmed": False,
    })
    _save_meta(meta)
    _save_batch(batch_id)
    return batch_id

def _confirmed_save(face_paths, label, batch_id):
    label      = label.upper()
    target_dir = CONFIRMED_FAKE if label == "FAKE" else CONFIRMED_REAL
    for i, src in enumerate(face_paths[:10]):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(
                target_dir, f"conf_{batch_id}_face_{i}.jpg"))
    meta = _load_meta()
    for entry in meta:
        if entry.get("batch_id") == batch_id:
            entry["confirmed"]       = True
            entry["confirmed_label"] = label
            entry["confirmed_at"]    = datetime.now().isoformat()
            break
    _save_meta(meta)

def _retrain_status():
    def count(d):
        return len([f for f in os.listdir(d) if f.endswith(".jpg")])
    cf = count(CONFIRMED_FAKE)
    cr = count(CONFIRMED_REAL)
    total = cf + cr
    return {
        "confirmed_fake": cf, "confirmed_real": cr,
        "total": total, "ready": total >= 100,
        "message": ("✅ Ready — run train_etcnn.py"
                    if total >= 100
                    else f"⏳ Need {100 - total} more confirmed faces")
    }


# ================================================
# ONLINE LEARNING — ETCNN fine-tune with safeguards
# ✅ FIX: Added validation before saving to prevent
#    feedback loop that corrupts the model
# ================================================
def fine_tune_etcnn(label: str, batch_id: str,
                    epochs: int = 3) -> None:
    label = label.upper()
    if label not in LABEL_MAP:
        return
    face_paths = [p for p in _load_session() if os.path.exists(p)]
    if len(face_paths) < 3:
        # ✅ FIX: Require at least 3 faces to fine-tune
        # Prevents overfitting on 1-2 noisy samples
        print(f"  ⚠️ Fine-tune skipped: only {len(face_paths)} faces "
              f"(need ≥3)")
        return

    # Save a backup before fine-tuning
    backup_path = ETCNN_ONLINE_PATH + ".backup"
    if os.path.exists(ETCNN_ONLINE_PATH):
        shutil.copy2(ETCNN_ONLINE_PATH, backup_path)

    for name, param in _etcnn.named_parameters():
        param.requires_grad = ("classifier" in name
                               or "attention" in name)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, _etcnn.parameters()),
        lr=1e-5,
        weight_decay=1e-4  # ✅ FIX: Added weight decay for regularization
    )
    criterion = nn.CrossEntropyLoss()
    target = torch.tensor([LABEL_MAP[label]]).to(device)

    _etcnn.train()
    total_loss = 0.0
    n_steps = 0
    for epoch in range(epochs):
        for path in face_paths[:10]:
            t = _prep_etcnn(path)
            if t is None: continue
            optimizer.zero_grad()
            loss = criterion(_etcnn(t), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_steps += 1

    _etcnn.eval()

    # ✅ FIX: Reset requires_grad to False (inference mode)
    for p in _etcnn.parameters():
        p.requires_grad = False

    # ✅ FIX: Validate model before saving — check it hasn't collapsed
    p0_vals = []
    with torch.no_grad():
        for _ in range(5):
            t   = torch.randn(1, 3, 224, 224).to(device)
            out = _etcnn(t)
            p0_vals.append(torch.softmax(out, dim=1)[0][0].item())

    var = float(np.var(p0_vals))
    avg = float(np.mean(p0_vals))

    if var < 0.0001 or avg > 0.95 or avg < 0.05:
        # Model collapsed — restore backup
        print(f"  🔴 Fine-tune REJECTED: model collapsed "
              f"(avg={avg:.3f} var={var:.5f}). Restoring backup.")
        if os.path.exists(backup_path):
            ck = torch.load(backup_path, map_location=device)
            state = ck if isinstance(ck, dict) and "conv" not in str(list(ck.keys())[:1]) else ck
            if hasattr(state, 'get'):
                state = state.get("state_dict", state.get("model_state_dict", state))
            _etcnn.load_state_dict(state, strict=False)
            _etcnn.eval()
        return

    # Model is healthy — save
    torch.save(_etcnn.state_dict(), ETCNN_ONLINE_PATH)
    print(f"  ✅ Fine-tune saved: {n_steps} steps, "
          f"avg_loss={total_loss/max(n_steps,1):.4f}")

    # Clean up backup
    if os.path.exists(backup_path):
        os.remove(backup_path)

    log = []
    if os.path.exists(FEEDBACK_LOG):
        try:
            with open(FEEDBACK_LOG) as f: log = json.load(f)
        except Exception: pass
    log.append({
        "timestamp":  datetime.now().isoformat(),
        "label":      label,
        "batch_id":   batch_id,
        "faces_used": len(face_paths[:10]),
        "avg_loss":   round(total_loss / max(n_steps, 1), 6),
        "post_train_variance": round(var, 6)
    })
    with open(FEEDBACK_LOG, "w") as f:
        json.dump(log, f, indent=2)

def get_feedback_history():
    if not os.path.exists(FEEDBACK_LOG):
        return []
    try:
        with open(FEEDBACK_LOG) as f: return json.load(f)
    except Exception: return []
