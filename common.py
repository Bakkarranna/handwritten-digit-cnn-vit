"""Shared pieces used by both train.py and app.py.

Keeping the models, constants and the data split in one place is what makes
sure training and the Streamlit app agree on scale, shapes and class mapping.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

IMG_SIZE = 8            # images are 8x8
PATCH = 2               # 2x2 patches
N_PATCHES = (IMG_SIZE // PATCH) ** 2   # 16
PATCH_DIM = PATCH * PATCH              # 4
N_CLASSES = 10
CLASS_NAMES = list(range(10))          # class index i is the digit i
SKLEARN_MAX = 16.0      # sklearn pixels run 0..16 -> divide by 16 for 0..1
PHOTO_MAX = 255.0       # uploaded photo pixels run 0..255 -> divide by 255 for 0..1
SEED = 42


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data():
    """Return images [N,1,8,8] scaled to 0..1 (float32) and labels [N]."""
    d = load_digits()
    x = (d.images.astype(np.float32) / SKLEARN_MAX)[:, None, :, :]
    y = d.target.astype(np.int64)
    return x, y


def make_split(y, seed=SEED):
    """One stratified 60/20/20 train/val/test split of INDICES.

    Both models use the indices returned here, so they see identical data.
    Indices are disjoint, so nothing leaks between the three sets.
    """
    idx = np.arange(len(y))
    train_val, test = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.25, stratify=y[train_val], random_state=seed)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    return train, val, test


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class DigitCNN(nn.Module):
    """Small CNN. Input [N,1,8,8] -> output [N,10] logits."""

    def __init__(self, c1=16, c2=32, hidden=64, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                      # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 4 * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, N_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def to_patches(x):
    """[N,1,8,8] -> [N,16,4]: sixteen non-overlapping 2x2 patches, row by row."""
    n = x.shape[0]
    p = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)   # [N,1,4,4,2,2]
    return p.reshape(n, N_PATCHES, PATCH_DIM)


class DigitViT(nn.Module):
    """Tiny Vision Transformer.

    patches [N,16,4] -> Linear patch embedding -> + learned position embedding
    -> TransformerEncoder (self-attention) -> mean over patches -> 10 logits.
    """

    def __init__(self, d_model=64, heads=4, layers=2, ff=128, dropout=0.1):
        super().__init__()
        self.patch_embed = nn.Linear(PATCH_DIM, d_model)             # trainable projection
        self.pos_embed = nn.Parameter(torch.zeros(1, N_PATCHES, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_CLASSES)

    def forward(self, x):
        t = to_patches(x)                         # [N,16,4]
        t = self.patch_embed(t) + self.pos_embed  # [N,16,d]
        t = self.encoder(t)                       # attention across the 16 patches
        t = self.norm(t).mean(dim=1)              # [N,d]
        return self.head(t)


MODEL_CLASSES = {"cnn": DigitCNN, "vit": DigitViT}


# ----------------------------------------------------------------------------
# Save / restore
# ----------------------------------------------------------------------------
def save_checkpoint(name, model, config, extra=None):
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "name": name,
            "state_dict": model.state_dict(),
            "config": config,                 # constructor settings
            "classes": CLASS_NAMES,
            "sklearn_max": SKLEARN_MAX,
            "extra": extra or {},
        },
        MODEL_DIR / f"{name}.pt",
    )


def load_checkpoint(name):
    """Rebuild a model from its checkpoint file and put it in eval mode."""
    path = MODEL_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}. Run `python train.py` first.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MODEL_CLASSES[name](**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def load_settings():
    return json.loads((MODEL_DIR / "settings.json").read_text())
