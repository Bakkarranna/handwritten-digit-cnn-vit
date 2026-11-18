"""Create demo images in samples/: clear digits, a blank page and a broken file."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from common import load_data, make_split

OUT = "samples"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def font_digit(d, size=400, dark_on_light=True, tilt=0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    bg, fg = (235, 232, 225), (30, 30, 35)
    if not dark_on_light:
        bg, fg = fg, bg
    img = Image.new("RGB", (size, size), bg)
    f = ImageFont.truetype(FONT, int(size * 0.62))
    ImageDraw.Draw(img).text((size // 2, size // 2), str(d), font=f, fill=fg, anchor="mm")
    if tilt:
        img = img.rotate(tilt, fillcolor=bg, resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    a = np.asarray(img).astype(np.float32)
    if noise:
        a += rng.normal(0, noise, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def main():
    os.makedirs(OUT, exist_ok=True)
    # real sklearn test digits, enlarged (bright on dark, like the training data)
    x, y = load_data()
    _, _, te = make_split(y)
    seen = set()
    for i in te:
        if int(y[i]) in seen:
            continue
        seen.add(int(y[i]))
        im = Image.fromarray((x[i, 0] * 255).astype(np.uint8)).resize((320, 320), Image.BICUBIC)
        im.save(f"{OUT}/sklearn_test_digit_{int(y[i])}.png")
    # rendered "photos"
    for d, tilt, noise in [(3, 0, 0), (7, 5, 6), (5, -6, 8), (2, 0, 4), (9, 4, 0)]:
        font_digit(d, tilt=tilt, noise=noise, seed=d).save(f"{OUT}/photo_digit_{d}.png")
    font_digit(4, dark_on_light=False).save(f"{OUT}/photo_digit_4_light_on_dark.jpg", quality=92)
    # bad inputs
    Image.new("RGB", (400, 400), (240, 240, 240)).save(f"{OUT}/blank_white.png")
    rng = np.random.default_rng(1)
    blank = np.clip(rng.normal(238, 3, (400, 400, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(blank).save(f"{OUT}/blank_noisy.jpg", quality=90)
    with open(f"{OUT}/broken.png", "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\nthis is not really a png" + bytes(range(50)))
    with open(f"{OUT}/not_an_image.png", "wb") as f:
        f.write(b"hello, I am a text file renamed to png")


if __name__ == "__main__":
    main()
