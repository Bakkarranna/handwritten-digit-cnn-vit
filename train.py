"""Train the CNN and the ViT on sklearn digits, evaluate both, save everything.

Run:  python train.py
Writes models/cnn.pt, models/vit.pt, models/settings.json and results/*.
The test set is touched exactly once, after training, for the final numbers.
"""
import argparse
import copy
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from common import (CLASS_NAMES, MODEL_CLASSES, RESULT_DIR, SEED, SKLEARN_MAX, MODEL_DIR,
                    load_data, make_split, save_checkpoint, N_PATCHES, PATCH, IMG_SIZE)

CONFIGS = {
    "cnn": dict(config=dict(c1=16, c2=32, hidden=64, dropout=0.25), epochs=80, lr=2e-3, wd=1e-4, patience=20),
    "vit": dict(config=dict(d_model=64, heads=4, layers=2, ff=128, dropout=0.1), epochs=200, lr=1e-3, wd=1e-4, patience=40),
}
BATCH = 64


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x).argmax(1).numpy()


def train_one(name, xtr, ytr, xva, yva):
    cfg = CONFIGS[name]
    torch.manual_seed(SEED)
    model = MODEL_CLASSES[name](**cfg["config"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(SEED)
    best_acc, best_state, best_epoch, bad = -1.0, None, 0, 0
    hist = {"train_loss": [], "val_acc": []}
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=g)
        total = 0.0
        for i in range(0, len(xtr), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xtr[b]), ytr[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        sched.step()
        val_acc = accuracy_score(yva.numpy(), predict(model, xva))
        hist["train_loss"].append(total / len(xtr))
        hist["val_acc"].append(val_acc)
        if val_acc > best_acc:                       # model selection uses VALIDATION only
            best_acc, best_state, best_epoch, bad = val_acc, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
        if ep % 10 == 0:
            print(f"  [{name}] epoch {ep:3d} loss {hist['train_loss'][-1]:.4f} val_acc {val_acc:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    print(f"  [{name}] best val acc {best_acc:.4f} at epoch {best_epoch}")
    return model, hist, best_acc, best_epoch


def top_confusions(cm, k=3):
    pairs = [(int(cm[i, j]), i, j) for i in range(10) for j in range(10) if i != j and cm[i, j] > 0]
    return sorted(pairs, reverse=True)[:k]


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.set_num_threads(1)
    np.random.seed(SEED)
    RESULT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    x, y = load_data()
    tr, va, te = make_split(y)
    np.savez(RESULT_DIR / "split_indices.npz", train=tr, val=va, test=te)
    print(f"split sizes  train {len(tr)}  val {len(va)}  test {len(te)}")
    X = torch.from_numpy(x)
    Y = torch.from_numpy(y)
    xtr, ytr, xva, yva, xte, yte = X[tr], Y[tr], X[va], Y[va], X[te], Y[te]
    print(f"pixel range after /{SKLEARN_MAX:g}: {x.min():.2f} .. {x.max():.2f}")

    results, cms, hists, preds = {}, {}, {}, {}
    for name in ("cnn", "vit"):
        print(f"training {name}")
        model, hist, val_acc, best_ep = train_one(name, xtr, ytr, xva, yva)
        save_checkpoint(name, model, CONFIGS[name]["config"],
                        extra={"best_val_acc": val_acc, "best_epoch": best_ep})
        p = predict(model, xte)                      # TEST set, used once
        results[name] = {
            "val_accuracy": float(val_acc),
            "test_accuracy": float(accuracy_score(yte.numpy(), p)),
            "test_macro_f1": float(f1_score(yte.numpy(), p, average="macro")),
            "best_epoch": best_ep,
            "parameters": int(sum(q.numel() for q in model.parameters())),
        }
        cms[name] = confusion_matrix(yte.numpy(), p, labels=CLASS_NAMES)
        hists[name], preds[name] = hist, p

    # Majority-class baseline: always predict the most common TRAIN class.
    majority = int(np.bincount(ytr.numpy()).argmax())
    pb = np.full(len(yte), majority)
    results["majority_baseline"] = {
        "predicted_class": majority,
        "test_accuracy": float(accuracy_score(yte.numpy(), pb)),
        "test_macro_f1": float(f1_score(yte.numpy(), pb, average="macro", zero_division=0)),
    }
    results["split"] = {"train": len(tr), "val": len(va), "test": len(te), "seed": SEED}
    results["confusion_matrices"] = {k: v.tolist() for k, v in cms.items()}
    (RESULT_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    (MODEL_DIR / "settings.json").write_text(json.dumps({
        "image_size": IMG_SIZE, "patch_size": PATCH, "num_patches": N_PATCHES,
        "train_pixel_divisor": SKLEARN_MAX, "upload_pixel_divisor": 255.0,
        "classes": CLASS_NAMES, "seed": SEED,
        "cnn_config": CONFIGS["cnn"]["config"], "vit_config": CONFIGS["vit"]["config"],
    }, indent=2))

    # ---- figures ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, ("cnn", "vit")):
        ax.imshow(cms[name], cmap="Blues")
        ax.set_title(f"{name.upper()} test confusion matrix\nacc {results[name]['test_accuracy']:.3f}  macro F1 {results[name]['test_macro_f1']:.3f}")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        for i in range(10):
            for j in range(10):
                if cms[name][i, j]:
                    ax.text(j, i, cms[name][i, j], ha="center", va="center",
                            color="white" if cms[name][i, j] > cms[name].max() / 2 else "black", fontsize=8)
    plt.tight_layout(); plt.savefig(RESULT_DIR / "confusion_matrices.png", dpi=130); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("cnn", "vit"):
        axes[0].plot(hists[name]["train_loss"], label=name.upper())
        axes[1].plot(hists[name]["val_acc"], label=name.upper())
    axes[0].set_title("training loss"); axes[1].set_title("validation accuracy")
    for a in axes:
        a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig(RESULT_DIR / "training_curves.png", dpi=130); plt.close()

    # Show a few test mistakes.
    wrong = [(n, i) for n in ("cnn", "vit") for i in np.where(preds[n] != yte.numpy())[0]][:8]
    if wrong:
        fig, axes = plt.subplots(1, len(wrong), figsize=(1.8 * len(wrong), 2.4), squeeze=False)
        for ax, (n, i) in zip(axes[0], wrong):
            ax.imshow(xte[i, 0], cmap="gray_r"); ax.axis("off")
            ax.set_title(f"{n.upper()}\ntrue {int(yte[i])}\npred {int(preds[n][i])}", fontsize=8)
        plt.tight_layout(); plt.savefig(RESULT_DIR / "errors.png", dpi=130); plt.close()

    # ---- written report ----
    lines = ["# Results", "",
             f"Split (stratified, same for both models): train {len(tr)}, validation {len(va)}, test {len(te)}.", "",
             "| Model | Test accuracy | Test macro F1 | Val accuracy | Parameters |", "|---|---|---|---|---|"]
    for n in ("cnn", "vit"):
        r = results[n]
        lines.append(f"| {n.upper()} | {r['test_accuracy']:.4f} | {r['test_macro_f1']:.4f} | {r['val_accuracy']:.4f} | {r['parameters']} |")
    b = results["majority_baseline"]
    lines.append(f"| Majority baseline (always {b['predicted_class']}) | {b['test_accuracy']:.4f} | {b['test_macro_f1']:.4f} | n/a | 0 |")
    lines += ["", "## Mistakes", ""]
    for n in ("cnn", "vit"):
        tc = top_confusions(cms[n])
        wrong_n = int((preds[n] != yte.numpy()).sum())
        if tc:
            txt = ", ".join(f"true {i} predicted as {j} ({c}x)" for c, i, j in tc)
            lines.append(f"- {n.upper()}: {wrong_n} of {len(te)} test images wrong. Most common: {txt}.")
        else:
            lines.append(f"- {n.upper()}: no test mistakes.")
    lines += ["", "Confusion matrices: `confusion_matrices.png`. Example errors: `errors.png`. Curves: `training_curves.png`."]
    (RESULT_DIR / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
