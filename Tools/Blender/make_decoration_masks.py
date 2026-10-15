"""Regenerate the black/white decoration masks used by the weapon texture bakes.

    python3 Tools/Blender/make_decoration_masks.py

Needs Pillow and the DejaVu Serif Bold font (fonts-dejavu on Debian/Ubuntu).
White = gold inlay, black = bare steel.
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

WEAPONS = Path(__file__).resolve().parents[2] / "Art" / "Weapons"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"


def sword_inscription(path):
    """'+ IN NOMINE DOMINI +' spread along the fuller (0.22 m x 0.0095 m)."""
    w, h = 4096, 176
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 104)
    chars = list("+ IN  NOMINE  DOMINI +")
    widths = [d.textlength(c, font=font) for c in chars]
    track = (w * 0.92 - sum(widths)) / (len(chars) - 1)
    x = w * 0.04
    for c, cw in zip(chars, widths):
        d.text((x, h / 2), c, font=font, fill=255, anchor="lm")
        x += cw + track
    img.filter(ImageFilter.GaussianBlur(1.2)).save(path)


def axe_braid(path):
    """Braided band between border lines for the axe cheek (0.014 m x 0.060 m)."""
    w, h = 280, 1200
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    lw = 14
    d.line([(lw, 0), (lw, h)], fill=255, width=lw)
    d.line([(w - lw, 0), (w - lw, h)], fill=255, width=lw)
    d.line([(lw, lw // 2), (w - lw, lw // 2)], fill=255, width=lw)
    d.line([(lw, h - lw // 2), (w - lw, h - lw // 2)], fill=255, width=lw)
    amp, cx, period = w * 0.30, w / 2, 300
    for phase in (0, math.pi):
        pts = [(cx + amp * math.sin(2 * math.pi * y / period + phase), y) for y in range(lw, h - lw, 4)]
        d.line(pts, fill=255, width=lw, joint="curve")
    for k in range(int(h / period * 2) + 1):
        y = period / 4 + k * period / 2
        if lw * 3 < y < h - lw * 3:
            d.ellipse([cx - 11, y - 11, cx + 11, y + 11], fill=255)
    img.filter(ImageFilter.GaussianBlur(1.5)).transpose(Image.FLIP_TOP_BOTTOM).save(path)


if __name__ == "__main__":
    sword_inscription(WEAPONS / "SteelSword" / "Source" / "T_SteelSword_EngravingMask.png")
    axe_braid(WEAPONS / "SteelAxe" / "Source" / "T_SteelAxe_InlayMask.png")
