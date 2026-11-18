"""Quick checks for the rubric items. Run: python test_pipeline.py"""
import subprocess
import sys

import torch

from common import (DigitCNN, DigitViT, load_checkpoint, load_data, make_split, to_patches)
from preprocess import ImageProblem, prepare_image


def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond:
        sys.exit(1)


x, y = load_data()
check("dataset: 1797 images, 8x8, labels 0-9", x.shape == (1797, 1, 8, 8) and set(y) == set(range(10)))
check("pixel scale 0..1", x.min() >= 0 and x.max() <= 1.0)
tr, va, te = make_split(y)
check("split: disjoint, covers all", len(set(tr) | set(va) | set(te)) == 1797 and not (set(tr) & set(te)) and not (set(va) & set(te)) and not (set(tr) & set(va)))
check("split: stratified (every class in every part)", all(set(y[s]) == set(range(10)) for s in (tr, va, te)))
check("split: reproducible", all((a == b).all() for a, b in zip((tr, va, te), make_split(y))))

b = torch.from_numpy(x[:5])
check("CNN [N,1,8,8] -> [N,10]", DigitCNN()(b).shape == (5, 10))
check("ViT [N,1,8,8] -> [N,10]", DigitViT()(b).shape == (5, 10))
p = to_patches(b)
check("patches shape [N,16,4]", p.shape == (5, 16, 4))
img = b[0, 0]
check("patch 0 is the top-left 2x2", torch.equal(p[0, 0], img[0:2, 0:2].reshape(4)))
check("patch 5 is row 1, col 1 block", torch.equal(p[0, 5], img[2:4, 2:4].reshape(4)))
check("patches cover every pixel exactly once", torch.isclose(p.sum(), b.sum()))
vit = DigitViT()
check("ViT has trainable patch projection, positions, attention",
      vit.patch_embed.weight.requires_grad and vit.pos_embed.requires_grad
      and any("self_attn" in n for n, _ in vit.named_parameters()))

cnn, _ = load_checkpoint("cnn")
vit, _ = load_checkpoint("vit")
xt = torch.from_numpy(x[te]); yt = y[te]
with torch.no_grad():
    for n, m in (("CNN", cnn), ("ViT", vit)):
        acc = (m(xt).argmax(1).numpy() == yt).mean()
        check(f"{n} restored from file, test acc {acc:.3f}", acc > 0.9)

# fresh Python process: weights reload and predict without training
code = ("import torch;from common import load_checkpoint;m,_=load_checkpoint('vit');"
        "print(int(m(torch.zeros(1,1,8,8)).argmax()))")
r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
check("fresh process loads checkpoint and predicts", r.returncode == 0 and r.stdout.strip().isdigit())

for f in ("samples/sklearn_test_digit_3.png", "samples/photo_digit_7.png", "samples/photo_digit_4_light_on_dark.jpg"):
    a = prepare_image(f)
    check(f"prepare {f.split('/')[-1]}: 8x8, 0..1, strokes bright", a.shape == (8, 8) and 0 <= a.min() and a.max() <= 1 and a.max() > 0.9 and a[0, 0] < 0.5)
for f in ("samples/blank_white.png", "samples/blank_noisy.jpg", "samples/broken.png", "samples/not_an_image.png"):
    try:
        prepare_image(f); ok = False
    except ImageProblem:
        ok = True
    check(f"rejects {f.split('/')[-1]}", ok)
for bad in (None, b"", b"\x00\x01\x02"):
    try:
        prepare_image(bad); ok = False
    except ImageProblem:
        ok = True
    check(f"rejects input {bad!r}", ok)
print("all checks passed")
