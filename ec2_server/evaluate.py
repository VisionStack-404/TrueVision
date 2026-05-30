"""
evaluate.py  â€“  TrueVision evaluation & validation protocol
============================================================
Provides:
  1. Per-model evaluation (AUC, F1, FPR, FNR, ECE)
  2. Ensemble evaluation with ablation study
  3. Confidence histograms (ASCII + matplotlib)
  4. Leakage check (file hash overlap between train/val)
  5. Per-threshold sweep to find optimal operating point

Usage:
    python evaluate.py --model_dir /path/to/models \
                       --data_dir /path/to/val_data \
                       --out eval_report.json
"""

import argparse, os, json, math
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# METRICS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def roc_auc(y_true, y_score):
    """Compute AUC without sklearn dependency."""
    pairs = sorted(zip(y_score, y_true), reverse=True)
    tp, fp, n_pos, n_neg = 0, 0, sum(y_true), len(y_true)-sum(y_true)
    auc = 0.0
    prev_fp = 0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            auc += tp * (fp - prev_fp + 1)
            prev_fp = fp
            fp += 1
    return auc / max(n_pos * n_neg, 1)

def compute_all_metrics(y_true, y_score, threshold=0.5):
    """
    y_true:  list of int  (0=FAKE, 1=REAL)
    y_score: list of float (p_fake â€“ higher = more likely FAKE)
    """
    yt = np.array(y_true)
    ys = np.array(y_score)
    yp = (ys > threshold).astype(int)   # 0=predict_FAKE, 1=predict_REAL... wait
    # y_true=0=FAKE, y_score=p_fake â†’ predict FAKE if p_fake > threshold
    # So yp=1 means "predicted FAKE"
    fake_pred = (ys > threshold).astype(int)
    fake_true = (yt == 0).astype(int)

    tp = ((fake_pred==1) & (fake_true==1)).sum()
    tn = ((fake_pred==0) & (fake_true==0)).sum()
    fp = ((fake_pred==1) & (fake_true==0)).sum()
    fn = ((fake_pred==0) & (fake_true==1)).sum()

    precision = tp / max(tp+fp, 1)
    recall    = tp / max(tp+fn, 1)
    f1        = 2*precision*recall / max(precision+recall, 1e-9)
    acc       = (tp+tn) / max(len(yt), 1)
    fpr       = fp / max(fp+tn, 1)   # false alarm rate
    fnr       = fn / max(fn+tp, 1)   # miss rate

    # AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(fake_true, ys)
    except ImportError:
        auc = roc_auc(fake_true.tolist(), ys.tolist())

    # ECE
    ece = _ece(ys, fake_true)

    return {
        "auc": round(auc, 4), "f1": round(f1, 4),
        "acc": round(acc, 4), "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fpr": round(fpr, 4), "fnr": round(fnr, 4),
        "ece": round(ece, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_fake": int(fake_true.sum()), "n_real": int((1-fake_true).sum()),
    }

def _ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    ece  = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0: continue
        conf = probs[mask].mean()
        acc  = labels[mask].mean()
        ece += abs(conf - acc) * (mask.sum() / len(probs))
    return float(ece)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# THRESHOLD SWEEP
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def threshold_sweep(y_true, y_score, steps=20):
    """
    Sweeps threshold from 0.05 to 0.95 and returns
    per-threshold metrics to find optimal operating point.
    """
    results = []
    fake_true = np.array([1 if y==0 else 0 for y in y_true])
    ys = np.array(y_score)

    for t in np.linspace(0.05, 0.95, steps):
        pred = (ys > t).astype(int)
        tp = ((pred==1)&(fake_true==1)).sum()
        tn = ((pred==0)&(fake_true==0)).sum()
        fp = ((pred==1)&(fake_true==0)).sum()
        fn = ((pred==0)&(fake_true==1)).sum()
        f1  = 2*tp / max(2*tp+fp+fn, 1)
        fpr = fp / max(fp+tn, 1)
        fnr = fn / max(fn+tp, 1)
        acc = (tp+tn) / max(len(y_true), 1)
        results.append({
            "threshold": round(float(t),3),
            "f1":  round(float(f1),4),
            "fpr": round(float(fpr),4),
            "fnr": round(float(fnr),4),
            "acc": round(float(acc),4),
        })

    # Best threshold: maximise F1
    best = max(results, key=lambda r: r["f1"])
    return results, best


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONFIDENCE HISTOGRAM (ASCII)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def ascii_histogram(values, n_bins=10, title="", width=40):
    arr  = np.array(values)
    bins = np.linspace(0, 1, n_bins+1)
    counts, _ = np.histogram(arr, bins=bins)
    max_c = max(counts) if max(counts) > 0 else 1
    print(f"\n  {title}")
    for i, c in enumerate(counts):
        lo   = bins[i]
        bar  = "â–ˆ" * int(width * c / max_c)
        print(f"  [{lo:.1f}-{bins[i+1]:.1f}] {bar} {c}")

def confidence_analysis(y_true, y_score, model_name=""):
    """Print confidence breakdown for fake vs real images."""
    fake_scores = [s for s,l in zip(y_score,y_true) if l==0]
    real_scores = [s for s,l in zip(y_score,y_true) if l==1]

    print(f"\n  â”€â”€ Confidence Analysis: {model_name} â”€â”€")
    if real_scores:
        arr = np.array(real_scores)
        print(f"  REAL images (n={len(real_scores)}): "
              f"p_fake mean={arr.mean():.3f} std={arr.std():.3f} "
              f"| wrongly FAKE: {(arr>0.5).sum()}/{len(real_scores)}")
        ascii_histogram(real_scores, title="REAL image p_fake distribution:")
    if fake_scores:
        arr = np.array(fake_scores)
        print(f"  FAKE images (n={len(fake_scores)}): "
              f"p_fake mean={arr.mean():.3f} std={arr.std():.3f} "
              f"| wrongly REAL: {(arr<=0.5).sum()}/{len(fake_scores)}")
        ascii_histogram(fake_scores, title="FAKE image p_fake distribution:")

    # Try matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,4))
        if real_scores: ax.hist(real_scores, bins=20, alpha=0.6, label="REAL", color="green", range=(0,1))
        if fake_scores: ax.hist(fake_scores, bins=20, alpha=0.6, label="FAKE", color="red",   range=(0,1))
        ax.axvline(0.5, color="black", linestyle="--", label="threshold=0.5")
        ax.set_xlabel("p_fake"); ax.set_ylabel("Count")
        ax.set_title(f"{model_name} â€“ p_fake distribution")
        ax.legend()
        plt.tight_layout()
        out = f"confidence_{model_name}.png"
        plt.savefig(out, dpi=150); plt.close()
        print(f"  Saved histogram: {out}")
    except ImportError:
        pass


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# LEAKAGE CHECK
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def check_data_leakage(train_dir: str, val_dir: str) -> bool:
    """
    Checks for file hash overlap between train and val sets.
    Returns True if no leakage found.
    """
    import hashlib, glob
    def hash_file(p):
        h = hashlib.md5()
        with open(p,"rb") as f: h.update(f.read(8192))
        return h.hexdigest()

    print("\n  [Leakage Check]")
    train_hashes = set()
    for ext in ["jpg","jpeg","png"]:
        for f in glob.glob(f"{train_dir}/**/*.{ext}", recursive=True):
            train_hashes.add(hash_file(f))

    val_hashes = {}
    for ext in ["jpg","jpeg","png"]:
        for f in glob.glob(f"{val_dir}/**/*.{ext}", recursive=True):
            h = hash_file(f)
            val_hashes[f] = h

    leaked = [(f,h) for f,h in val_hashes.items() if h in train_hashes]
    if leaked:
        print(f"  âš   LEAKAGE: {len(leaked)} val files found in train set!")
        for f,_ in leaked[:5]:
            print(f"    {f}")
        return False
    else:
        print(f"  âœ“ No leakage detected (train={len(train_hashes)}, val={len(val_hashes)})")
        return True


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ABLATION STUDY
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def ablation_study(
    model_scores: dict,  # {"CNN": [p_fake,...], "CViT": [...], "ETCNN": [...]}
    y_true: list,
    threshold: float = 0.5,
):
    """
    Tests all ensemble combinations to find which models help/hurt.
    model_scores: dict of model_name â†’ list of p_fake scores (same length as y_true)
    """
    from itertools import combinations

    print("\n" + "="*60)
    print("  ABLATION STUDY â€“ which models help?")
    print("="*60)

    results = {}
    names = list(model_scores.keys())

    for r in range(1, len(names)+1):
        for combo in combinations(names, r):
            combo_scores = np.mean([model_scores[n] for n in combo], axis=0)
            m = compute_all_metrics(y_true, combo_scores.tolist(), threshold)
            key = "+".join(combo)
            results[key] = m
            print(f"  {key:<25}: AUC={m['auc']:.4f} F1={m['f1']:.4f} "
                  f"FPR={m['fpr']:.4f} FNR={m['fnr']:.4f}")

    best_combo = max(results.items(), key=lambda x: x[1]["auc"])
    print(f"\n  Best combination: {best_combo[0]} (AUC={best_combo[1]['auc']:.4f})")
    return results


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MAIN EVALUATION LOOP (no-checkpoint synthetic demo)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_synthetic_evaluation():
    """
    Demonstrates the full evaluation pipeline on synthetic data.
    Replace with real model predictions in production.
    """
    print("="*60)
    print("  Synthetic Evaluation Demo (no weights needed)")
    print("="*60)
    np.random.seed(42)
    N = 200  # 100 fake, 100 real

    y_true = [0]*100 + [1]*100  # 0=FAKE, 1=REAL

    # Simulate three models with different biases
    # CNN: slightly biased toward FAKE (common problem)
    cnn_scores = np.concatenate([
        np.random.beta(3, 1.5, 100),   # FAKE images: high p_fake (good)
        np.random.beta(2.5, 1.5, 100), # REAL images: also high p_fake (BAD - biased)
    ])

    # CViT: better calibrated
    cvit_scores = np.concatenate([
        np.random.beta(3, 1.2, 100),   # FAKE
        np.random.beta(1.2, 3, 100),   # REAL
    ])

    # ETCNN: collapsed (always 0.5-0.6 range â€“ common after bad fine-tuning)
    etcnn_scores = np.concatenate([
        np.random.normal(0.55, 0.05, 100).clip(0,1),  # slight FAKE bias
        np.random.normal(0.52, 0.05, 100).clip(0,1),  # even REAL looks FAKE
    ])

    model_scores = {"CNN": cnn_scores, "CViT": cvit_scores, "ETCNN": etcnn_scores}

    print("\n  â”€â”€ Per-model evaluation â”€â”€")
    for name, scores in model_scores.items():
        m = compute_all_metrics(y_true, scores.tolist())
        print(f"  {name:<8}: AUC={m['auc']:.4f}  F1={m['f1']:.4f}  "
              f"FPR={m['fpr']:.4f}  FNR={m['fnr']:.4f}  ECE={m['ece']:.4f}")
        confidence_analysis(y_true, scores.tolist(), name)

    ablation_study(model_scores, y_true)

    print("\n  â”€â”€ Threshold Sweep (CNN) â”€â”€")
    sweep, best_t = threshold_sweep(y_true, cnn_scores.tolist())
    print(f"  Optimal threshold: {best_t['threshold']} "
          f"(F1={best_t['f1']:.4f}, FPR={best_t['fpr']:.4f})")

    print("\n  â”€â”€ Diagnosis â”€â”€")
    cnn_m = compute_all_metrics(y_true, cnn_scores.tolist())
    if cnn_m["fpr"] > 0.3:
        print("  âš   CNN has high FPR (false alarm rate) â€“ model biased FAKE")
        print("     Fix: raise threshold OR retrain with more REAL samples")
    etcnn_m = compute_all_metrics(y_true, etcnn_scores.tolist())
    if etcnn_m["auc"] < 0.65:
        print("  âš   ETCNN has poor AUC â€“ likely collapsed from online learning")
        print("     Fix: restore backup checkpoint, check feedback imbalance")

    return {"cnn": cnn_m, "cvit": compute_all_metrics(y_true, cvit_scores.tolist()),
            "etcnn": etcnn_m}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--data_dir",  default=None)
    parser.add_argument("--train_dir", default=None, help="For leakage check")
    parser.add_argument("--out",       default="eval_report.json")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if args.data_dir and args.model_dir:
        # Real evaluation (requires checkpoints + data)
        print("Real evaluation requires: see docstring. Running synthetic demo instead.")
        results = run_synthetic_evaluation()
    else:
        results = run_synthetic_evaluation()

    if args.train_dir and args.data_dir:
        check_data_leakage(args.train_dir, args.data_dir)

    json.dump(results, open(args.out,"w"), indent=2)
    print(f"\n  Report saved to {args.out}")
