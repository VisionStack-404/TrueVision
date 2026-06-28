"""
Inspect all model checkpoint to discover their actual architecture.
Run this on EC2:  python inspect_checkpoints.py
"""
import torch
import os

BASE = "/home/ubuntu/truevision/models"

checkpoints = {
    "CNN FaceForensics":  f"{BASE}/cnn/cnn_faceforenics.pth",
    "CNN DFDC":           f"{BASE}/cnn/dfdc_fast_model.pth",
    "CNN CelebDF":        f"{BASE}/cnn/cnn_celebdf.pth",
    "CViT FaceForensics": f"{BASE}/cvit/cvit_faceforensics.pth",
    "CViT DFDC":          f"{BASE}/cvit/cvit_dfdc.pth",
    "CViT CelebDF":       f"{BASE}/cvit/cvit_celebdf.pth",
    "ETCNN Combined":     f"{BASE}/etcnn/etcnn_combined.pth",
}

for name, path in checkpoints.items():
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Path: {path}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print("  ❌ FILE NOT FOUND")
        continue

    ck = torch.load(path, map_location="cpu")

    # Check if it's a wrapper dict or raw state_dict
    if isinstance(ck, dict):
        top_keys = list(ck.keys())
        # Check for common wrapper keys
        if "state_dict" in top_keys:
            print(f"  Wrapper format: has 'state_dict' key")
            print(f"  Other top-level keys: {[k for k in top_keys if k != 'state_dict']}")
            state = ck["state_dict"]
        elif "model_state_dict" in top_keys:
            print(f"  Wrapper format: has 'model_state_dict' key")
            print(f"  Other top-level keys: {[k for k in top_keys if k != 'model_state_dict']}")
            state = ck["model_state_dict"]
        else:
            # Check if keys look like layer names (state_dict) or metadata
            sample_key = top_keys[0] if top_keys else ""
            if isinstance(ck.get(sample_key), torch.Tensor):
                print(f"  Raw state_dict format (no wrapper)")
                state = ck
            else:
                print(f"  ⚠️ Unknown format. Top keys: {top_keys[:10]}")
                state = ck
    else:
        print(f"  ⚠️ Not a dict, type: {type(ck)}")
        continue

    # Print all keys with shapes
    if isinstance(state, dict):
        keys = sorted(state.keys())
        print(f"\n  Total parameters: {len(keys)}")
        print(f"\n  ALL KEYS AND SHAPES:")
        print(f"  {'-'*50}")
        for k in keys:
            if isinstance(state[k], torch.Tensor):
                print(f"    {k}: {list(state[k].shape)}")
            else:
                print(f"    {k}: {type(state[k])}")

        # Detect architecture patterns
        print(f"\n  ARCHITECTURE DETECTION:")
        print(f"  {'-'*50}")

        key_str = " ".join(keys)

        if "conv_stem" in key_str:
            print("  → Uses 'conv_stem' naming → likely timm EfficientNet")
        if "features.0" in key_str:
            print("  → Uses 'features.X' naming → likely torchvision EfficientNet")
        if "backbone" in key_str:
            print("  → Has 'backbone' prefix → custom wrapper model")
        if "lstm" in key_str or "rnn" in key_str:
            print("  → Has LSTM/RNN layers → sequence model (not pure CNN/ViT)")
        if "transformer" in key_str or "cls_token" in key_str:
            print("  → Has transformer/cls_token → Vision Transformer component")
        if "classifier" in key_str or "fc" in key_str or "head" in key_str:
            # Find classifier output size
            for k in keys:
                if any(x in k for x in ["classifier", "fc", "head"]):
                    if isinstance(state[k], torch.Tensor) and state[k].dim() >= 1:
                        print(f"  → Classifier layer '{k}': output_size={state[k].shape[0]}")
        if "attention" in key_str:
            print("  → Has attention layers")
        if "texture" in key_str:
            print("  → Has texture branch")
        if "semantic" in key_str:
            print("  → Has semantic branch")
        if "patch_embed" in key_str:
            print("  → Has patch embedding")
        if "layer1" in key_str and "layer2" in key_str:
            print("  → Uses ResNet-style layer naming")
        if "encoder" in key_str:
            print("  → Has encoder component")

print(f"\n\n{'='*60}")
print("  DONE — Copy this entire output and share it")
print(f"{'='*60}")
