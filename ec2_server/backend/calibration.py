"""
calibration.py  â€“  Temperature Scaling for TrueVision
======================================================
Replaces the current sqrt-based calibration which amplifies bias.

Temperature scaling is a single-parameter post-hoc calibration:
    p_calibrated = softmax(logits / T)

T > 1  â†’ softer (less confident)   [use when model is overconfident]
T < 1  â†’ sharper (more confident)  [rarely needed]
T = 1  â†’ no change

Usage:
    # Fit on a balanced validation set
    calibrator = TemperatureScaler()
    calibrator.fit(model, val_loader, device)
    calibrator.save("temperature.pt")

    # At inference time
    calibrator = TemperatureScaler.load("temperature.pt")
    p_fake = calibrator.predict(logits)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os, json
from pathlib import Path


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# TEMPERATURE SCALER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class TemperatureScaler(nn.Module):
    """
    Single-parameter calibration module.
    Fit once on a held-out balanced validation set.
    """
    def __init__(self, init_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.ones(1) * init_temperature
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns calibrated probabilities."""
        return torch.softmax(logits / self.temperature.clamp(min=0.01), dim=1)

    def predict_fake_prob(self, logits: torch.Tensor) -> float:
        """Returns scalar p_fake after calibration."""
        probs = self.forward(logits)
        return probs[0][0].item()  # class-0 = FAKE (ImageFolder alphabetical)

    def fit(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        max_iter: int = 50,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> float:
        """
        Optimises temperature to minimise NLL on val_loader.
        val_loader must yield (images, labels) where label 0=FAKE, 1=REAL.

        Returns: final temperature value
        """
        model.eval()
        self.to(device)

        # Collect all logits (avoid recomputing model each iteration)
        all_logits, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                logits = model(imgs.to(device))
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        # NLL loss for optimisation
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )

        def eval_loss():
            optimizer.zero_grad()
            scaled = all_logits / self.temperature.clamp(min=0.01)
            loss   = criterion(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        T = self.temperature.item()

        if verbose:
            print(f"  [TemperatureScaler] Fitted T = {T:.4f}")
            if T > 2.0:
                print(f"  âš   T={T:.2f} is high â€“ model is very overconfident (overtrained?)")
            elif T < 0.5:
                print(f"  âš   T={T:.2f} < 1 â€“ model is under-confident (check training)")
            else:
                print(f"  âœ“ T in healthy range [0.5, 2.0]")

        # Report ECE before/after
        ece_before = self._ece(all_logits, all_labels, temperature=1.0)
        ece_after  = self._ece(all_logits, all_labels, temperature=T)
        if verbose:
            print(f"  ECE before calibration: {ece_before:.4f}")
            print(f"  ECE after  calibration: {ece_after:.4f}")

        return T

    @staticmethod
    def _ece(logits, labels, temperature=1.0, n_bins=10):
        """Expected Calibration Error."""
        probs  = torch.softmax(logits / temperature, dim=1)
        conf, preds = probs.max(dim=1)
        correct = preds.eq(labels)

        bins = torch.linspace(0, 1, n_bins+1)
        ece  = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i+1]
            mask   = (conf > lo) & (conf <= hi)
            if mask.sum() == 0:
                continue
            acc  = correct[mask].float().mean().item()
            con  = conf[mask].mean().item()
            ece += abs(acc - con) * (mask.sum().item() / len(labels))
        return ece

    def save(self, path: str):
        torch.save({"temperature": self.temperature.item()}, path)
        print(f"  [TemperatureScaler] Saved T={self.temperature.item():.4f} to {path}")

    @classmethod
    def load(cls, path: str) -> "TemperatureScaler":
        if not os.path.exists(path):
            print(f"  [TemperatureScaler] No saved T found at {path}, using T=1.0")
            return cls(init_temperature=1.0)
        d = torch.load(path, map_location="cpu")
        t = d.get("temperature", 1.0)
        print(f"  [TemperatureScaler] Loaded T={t:.4f} from {path}")
        return cls(init_temperature=float(t))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PER-MODEL CALIBRATION WRAPPER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class PerModelCalibrator:
    """
    Stores one TemperatureScaler per model so each model
    gets independently calibrated confidence.
    """
    def __init__(self, model_names: list):
        self.scalers = {name: TemperatureScaler() for name in model_names}

    def fit_model(self, name, model, val_loader, device):
        print(f"\n  Calibrating {name}...")
        self.scalers[name].fit(model, val_loader, device)

    def predict(self, name, logits):
        return self.scalers[name].predict_fake_prob(logits)

    def save(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        for name, scaler in self.scalers.items():
            scaler.save(os.path.join(dir_path, f"{name}_temperature.pt"))

    @classmethod
    def load(cls, dir_path: str, model_names: list) -> "PerModelCalibrator":
        obj = cls(model_names)
        for name in model_names:
            p = os.path.join(dir_path, f"{name}_temperature.pt")
            obj.scalers[name] = TemperatureScaler.load(p)
        return obj


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONFIDENCE HISTOGRAM UTILITY
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def plot_confidence_histogram(
    p_fake_real: list,
    p_fake_fake: list,
    model_name: str = "Model",
    save_path: str = None,
):
    """
    Prints ASCII confidence histogram.  Use matplotlib if available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(p_fake_real, bins=20, alpha=0.6, label="REAL images", color="green", range=(0,1))
        ax.hist(p_fake_fake, bins=20, alpha=0.6, label="FAKE images", color="red",   range=(0,1))
        ax.axvline(x=0.5, color="black", linestyle="--", label="Threshold 0.5")
        ax.set_xlabel("p_fake score")
        ax.set_ylabel("Count")
        ax.set_title(f"Confidence Distribution â€“ {model_name}")
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"  Saved histogram to {save_path}")
        plt.close()
    except ImportError:
        # ASCII fallback
        print(f"\n  [{model_name}] p_fake distribution (ASCII):")
        print(f"  REAL images: mean={np.mean(p_fake_real):.3f} std={np.std(p_fake_real):.3f}")
        print(f"  FAKE images: mean={np.mean(p_fake_fake):.3f} std={np.std(p_fake_fake):.3f}")
        real_wrong = sum(1 for p in p_fake_real if p > 0.5)
        fake_wrong = sum(1 for p in p_fake_fake if p <= 0.5)
        print(f"  False positives (realâ†’FAKE): {real_wrong}/{len(p_fake_real)}")
        print(f"  False negatives (fakeâ†’REAL): {fake_wrong}/{len(p_fake_fake)}")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STANDALONE CALIBRATION SCRIPT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    import argparse
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    # Demo: show calibration curve without real checkpoints
    print("="*60)
    print("  Calibration Demo â€“ comparing sqrt vs temperature scaling")
    print("="*60)

    def sqrt_calibrate(p):
        if p > 0.5: return 0.5 + (((p-0.5)/0.5)**0.5)*0.5
        elif p < 0.5: return 0.5 - (((0.5-p)/0.5)**0.5)*0.5
        return 0.5

    T = 1.5  # example temperature

    print(f"\n  {'raw':>6} | {'sqrt_cal':>10} | {'temp_cal(T=1.5)':>16} | {'verdict_sqrt':>14} | {'verdict_temp':>14}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*16}-+-{'-'*14}-+-{'-'*14}")
    for raw in [0.1, 0.3, 0.45, 0.48, 0.51, 0.55, 0.7, 0.9]:
        sc  = sqrt_calibrate(raw)
        # temperature scaling is applied to logits, not probs.
        # approximate: logit = log(p/(1-p)), then soften
        logit = np.log(raw / (1 - raw + 1e-9))
        tc    = 1 / (1 + np.exp(-logit / T))
        vs    = "FAKE" if sc > 0.5 else "REAL"
        vt    = "FAKE" if tc > 0.5 else "REAL"
        print(f"  {raw:>6.2f} | {sc:>10.4f} | {tc:>16.4f} | {vs:>14} | {vt:>14}")

    print("""
  KEY DIFFERENCE:
  - sqrt calibration: raw=0.51 â†’ 0.571 (7% jump, always FAKE)
  - temperature(1.5): raw=0.51 â†’ 0.508 (barely moves, stays uncertain)
  Temperature scaling is calibration without introducing directional bias.
    """)
