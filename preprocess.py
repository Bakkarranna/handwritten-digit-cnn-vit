"""Turn an uploaded photo into the 8x8 input the models were trained on.

Steps: open -> grayscale -> make strokes bright on dark -> find the strokes ->
crop -> centre on a square canvas -> resize to 8x8 -> divide by 255 (0..1).
The sklearn training pixels are divided by 16, so both paths end up in 0..1.
"""
import io

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from common import IMG_SIZE, PHOTO_MAX

MAX_BYTES = 15 * 1024 * 1024
WORK_SIZE = 512            # photos are shrunk to this long side before analysis
MIN_CONTRAST = 40          # 0..255 spread below this means "nothing drawn"
MIN_STROKE_FRACTION = 0.002  # fewer stroke pixels than this means blank
MAX_STROKE_FRACTION = 0.60   # more than this means the image is not one digit
MARGIN = 0.0               # sklearn digits touch the top and bottom edge, so no extra border


class ImageProblem(ValueError):
    """Raised with a message that is safe to show to the user."""


def load_image(source):
    """Open PNG/JPEG from bytes, a file-like object or a path. Raises ImageProblem."""
    if source is None:
        raise ImageProblem("No image was uploaded. Please choose a PNG or JPEG file.")
    try:
        if isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        elif hasattr(source, "getvalue"):
            data = source.getvalue()
        elif hasattr(source, "read"):
            data = source.read()
        else:
            with open(source, "rb") as f:
                data = f.read()
    except OSError:
        raise ImageProblem("The file could not be read.")
    if len(data) == 0:
        raise ImageProblem("The file is empty.")
    if len(data) > MAX_BYTES:
        raise ImageProblem("The file is too large (limit 15 MB).")
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ("PNG", "JPEG"):
            raise ImageProblem("Only PNG and JPEG images are supported.")
        img.load()                       # forces full decode, catches truncated files
    except ImageProblem:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise ImageProblem("This file is not a readable PNG/JPEG image.")
    return ImageOps.exif_transpose(img)  # phone photos carry a rotation flag


def _otsu(gray_u8):
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)
    w_b = 0.0
    sum_b = 0.0
    best, thr = -1.0, 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best, thr = between, t
    return thr


def prepare_image(source, return_steps=False):
    """Return a float32 array [8,8] with values in 0..1 (strokes bright).

    Raises ImageProblem for missing, unreadable or blank input.
    With return_steps=True also returns a dict of intermediate PIL images.
    """
    img = load_image(source)

    # Flatten any transparency onto white, then grayscale.
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    gray = img.convert("L")
    original_gray = gray.copy()

    # Work on a smaller copy and remove salt-and-pepper noise.
    gray.thumbnail((WORK_SIZE, WORK_SIZE), Image.LANCZOS)
    gray = gray.filter(ImageFilter.MedianFilter(3))
    a = np.asarray(gray, dtype=np.uint8)
    h, w = a.shape
    if h < 4 or w < 4:
        raise ImageProblem("The image is too small to contain a digit.")

    # Blank check 1: almost no brightness difference anywhere.
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi - lo < MIN_CONTRAST:
        raise ImageProblem("The image looks blank. Please upload a photo with one clear digit.")

    # Strokes must end up bright on dark. The background is whatever the border looks like.
    border = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
    if np.median(border) > a.mean():
        a = 255 - a                      # bright background -> invert
    # Stretch contrast so the strokes reach the top of the range.
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    a = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1) * 255
    a = a.astype(np.uint8)

    # Find the strokes with an Otsu threshold.
    thr = _otsu(a)
    mask = a > thr
    frac = mask.mean()
    if frac < MIN_STROKE_FRACTION:
        raise ImageProblem("The image looks blank. Please upload a photo with one clear digit.")
    if frac > MAX_STROKE_FRACTION:
        raise ImageProblem("Could not separate a digit from the background. Try a plain background.")

    # Crop to the strokes, ignoring a few stray specks at the extremes.
    rows = np.where(mask.sum(axis=1) > max(1, 0.002 * w))[0]
    cols = np.where(mask.sum(axis=0) > max(1, 0.002 * h))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ImageProblem("The image looks blank. Please upload a photo with one clear digit.")
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    crop = Image.fromarray(a[r0:r1, c0:c1])

    # Centre on a square black canvas with a margin, like the sklearn digits.
    side = int(round(max(crop.size) * (1 + 2 * MARGIN)))
    side = max(side, IMG_SIZE)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(crop, ((side - crop.size[0]) // 2, (side - crop.size[1]) // 2))

    # Area-average down to 8x8 (the sklearn digits were made by block averaging too).
    small = canvas.resize((IMG_SIZE, IMG_SIZE), Image.BOX)
    arr = np.asarray(small, dtype=np.float32) / PHOTO_MAX     # 0..1
    if arr.max() < 0.05:
        raise ImageProblem("The digit is too faint to read. Try a darker pen or better light.")
    # Same idea as the training data: the strongest stroke pixel is full brightness.
    arr = arr / arr.max()

    if return_steps:
        return arr, {"original": original_gray, "cropped": crop, "centered": canvas, "small": small}
    return arr
