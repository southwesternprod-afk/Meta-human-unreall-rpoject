"""Regenerate the black/white decoration masks used by the weapon texture bakes.

    python3 Tools/Blender/make_decoration_masks.py

Needs Pillow and the DejaVu Serif Bold font (fonts-dejavu on Debian/Ubuntu).
White = inlay / emboss, black = plain surface.
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

WEAPONS = Path(__file__).resolve().parents[2] / "Art" / "Weapons"
CLOTHING = Path(__file__).resolve().parents[2] / "Art" / "Characters" / "Clothing"
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


def bez(p0, p1, p2, p3, n=40):
    out = []
    for i in range(n + 1):
        t = i / n
        a, b, c, d = (1-t)**3, 3*t*(1-t)**2, 3*t*t*(1-t), t**3
        out.append((a*p0[0]+b*p1[0]+c*p2[0]+d*p3[0], a*p0[1]+b*p1[1]+c*p2[1]+d*p3[1]))
    return out

def fleur(size=1024):
    """Heraldic fleur-de-lis embossed on the jerkin's pauldron caps."""
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    s = size / 1024
    def S(pts): return [(x * s, y * s) for x, y in pts]
    # centre petal: pointed top, swelling, narrowing into the band
    right = bez((512, 110), (600, 230), (640, 380), (560, 560))
    left = [(1024 - x, y) for x, y in reversed(right)]
    d.polygon(S(right + left), fill=255)
    # side petals: rise from the band, sweep outward and curl down at the tips
    for sgn in (1, -1):
        def X(x): return 512 + sgn * (x - 512)
        outer = bez((X(560), 600), (X(640), 430), (X(760), 300), (X(850), 360))
        tip = bez((X(850), 360), (X(905), 395), (X(890), 470), (X(830), 480))
        inner = bez((X(830), 480), (X(800), 430), (X(720), 420), (X(600), 590))
        d.polygon(S(outer + tip + inner), fill=255)
    # band across the middle
    d.rounded_rectangle(S([(380, 575), (644, 640)])[0] + S([(380, 575), (644, 640)])[1], radius=20 * s, fill=255)
    # lower part: short centre stem and two small curled feet
    stem_r = bez((512, 640), (560, 700), (560, 780), (512, 900))
    d.polygon(S(stem_r + [(1024 - x, y) for x, y in reversed(stem_r)]), fill=255)
    for sgn in (1, -1):
        def X(x): return 512 + sgn * (x - 512)
        foot = bez((X(540), 640), (X(640), 660), (X(700), 720), (X(650), 790)) + \
               bez((X(650), 790), (X(640), 740), (X(600), 700), (X(530), 700))
        d.polygon(S(foot), fill=255)
    return img.filter(ImageFilter.GaussianBlur(3 * s))


if __name__ == "__main__":
    sword_inscription(WEAPONS / "SteelSword" / "Source" / "T_SteelSword_EngravingMask.png")
    axe_braid(WEAPONS / "SteelAxe" / "Source" / "T_SteelAxe_InlayMask.png")
    fleur().save(CLOTHING / "LeatherJerkin" / "Source" / "T_LeatherJerkin_CrestMask.png")
