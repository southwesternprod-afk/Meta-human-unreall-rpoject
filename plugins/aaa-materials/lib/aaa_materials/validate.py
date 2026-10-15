"""PBR texture-set validator (numpy + Pillow, no Blender needed).

Checks a T_<Name>_{BaseColor,ORM,Normal}.png set against the rules AAA
pipelines enforce (Substance / Unreal / DONTNOD PBR charts):

  * size        power of two, square
  * albedo      non-metals sRGB 30..240 (no pure black / white), metals sRGB >= 180
  * metallic    mostly 0 or 1 - grey metallic only in thin transitions
  * roughness   not flat, not mirror-like everywhere
  * normal      unit length, Z (blue) positive, not empty
  * AO          not crushing to black
  * tiling      (tileable sets) opposite edges continue each other
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Report:
    name: str
    results: list = field(default_factory=list)   # (level, check, message)

    def add(self, level, check, msg):
        self.results.append((level, check, msg))

    @property
    def worst(self):
        order = {"PASS": 0, "INFO": 0, "WARN": 1, "FAIL": 2}
        return max((r[0] for r in self.results), key=lambda l: order[l], default="PASS")

    def text(self):
        lines = [f"== {self.name}: {self.worst}"]
        lines += [f"  [{lvl:4}] {chk:10} {msg}" for lvl, chk, msg in self.results]
        return "\n".join(lines)

    def as_dict(self):
        return {"name": self.name, "verdict": self.worst,
                "results": [dict(level=l, check=c, message=m) for l, c, m in self.results]}


def _load(path, max_side=1024):
    img = Image.open(path).convert("RGB")
    full = img.size
    if max(img.size) > max_side:                      # statistics don't need every pixel
        img = img.resize((max_side, max_side * img.size[1] // img.size[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0, full


def _pct(mask):
    return 100.0 * float(np.mean(mask))


def find_set(directory, name=None):
    d = Path(directory)
    if name is None:
        bases = sorted(d.glob("T_*_BaseColor.png"))
        if not bases:
            raise FileNotFoundError(f"no T_*_BaseColor.png in {d}")
        return [find_set(d, b.name[2:-len("_BaseColor.png")])[0] for b in bases]
    files = {k: d / f"T_{name}_{k}.png" for k in ("BaseColor", "ORM", "Normal")}
    return [(name, files)]


def validate(name, files, tileable=False):
    r = Report(name)
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        r.add("FAIL", "files", f"missing {', '.join(missing)}")
        return r

    base, size = _load(files["BaseColor"])
    orm, osize = _load(files["ORM"])
    nrm, nsize = _load(files["Normal"])

    # --- size
    for label, (w, h) in (("BaseColor", size), ("ORM", osize), ("Normal", nsize)):
        pot = (w & (w - 1) == 0) and (h & (h - 1) == 0)
        if not pot or w != h:
            r.add("WARN", "size", f"{label} is {w}x{h} (use square power-of-two)")
    if len({size, osize, nsize}) == 1:
        r.add("PASS", "size", f"{size[0]}x{size[1]}")
    else:
        r.add("WARN", "size", f"maps differ in size: {size}, {osize}, {nsize}")

    ao, rough, metal = orm[..., 0], orm[..., 1], orm[..., 2]
    is_metal = metal > 0.9
    is_diel = metal < 0.1
    srgb255 = base * 255.0

    # --- albedo
    if is_diel.any():
        d = srgb255[is_diel]
        lum = d @ np.array([0.2126, 0.7152, 0.0722])
        dark, bright = _pct(lum < 30), _pct(lum > 240)
        lvl = "FAIL" if dark > 10 or bright > 10 else "WARN" if dark > 2 or bright > 2 else "PASS"
        r.add(lvl, "albedo", f"non-metal: {dark:.1f}% darker than sRGB 30, {bright:.1f}% brighter than 240 "
                             f"(median {np.median(lum):.0f})")
    if is_metal.any():
        m = srgb255[is_metal].max(axis=1)
        dark = _pct(m < 180)
        lvl = "FAIL" if dark > 25 else "WARN" if dark > 5 else "PASS"
        r.add(lvl, "albedo", f"metal: {dark:.1f}% below sRGB 180 reflectance (median {np.median(m):.0f}); "
                             "dark metals should be a non-metal oxide/grime layer over bright metal")

    # --- metallic
    grey = _pct((metal > 0.15) & (metal < 0.85))
    lvl = "WARN" if grey > 12 else "PASS"
    r.add(lvl, "metallic", f"{grey:.1f}% grey metallic (0.15-0.85); {_pct(is_metal):.0f}% metal, "
                           f"{_pct(is_diel):.0f}% non-metal")

    # --- roughness
    std = float(rough.std())
    glossy = _pct(rough < 0.08)
    lvl = "WARN" if std < 0.02 or glossy > 30 else "PASS"
    r.add(lvl, "roughness", f"mean {rough.mean():.2f}, std {std:.3f}, {glossy:.1f}% below 0.08"
                            + (" - looks flat, add variation" if std < 0.02 else ""))

    # --- normal
    v = nrm * 2 - 1
    length = np.linalg.norm(v, axis=-1)
    bad_len = _pct(np.abs(length - 1) > 0.15)
    neg_z = _pct(v[..., 2] < 0)
    detail = float(v[..., :2].std(axis=(0, 1)).mean())        # spread of surface slopes
    empty = detail < 0.004
    lvl = "FAIL" if neg_z > 1 else "WARN" if bad_len > 5 or empty else "PASS"
    r.add(lvl, "normal", f"{bad_len:.1f}% not unit length, {neg_z:.2f}% negative Z, slope spread {detail:.3f}"
                         + (" - no surface detail" if empty else ""))

    # --- AO
    crushed = _pct(ao < 0.15)
    r.add("WARN" if crushed > 5 else "PASS", "ao", f"mean {ao.mean():.2f}, {crushed:.1f}% below 0.15")

    # --- tiling
    if tileable:
        for label, arr in (("BaseColor", base), ("Normal", nrm)):
            seam = np.abs(arr[:, 0] - arr[:, -1]).mean() + np.abs(arr[0] - arr[-1]).mean()
            inner = np.abs(arr[:, 1] - arr[:, 2]).mean() + np.abs(arr[1] - arr[2]).mean()
            ratio = seam / max(inner, 1e-6)
            r.add("WARN" if ratio > 2.0 else "PASS", "tiling", f"{label} edge/inner difference ratio {ratio:.2f}")
    return r


def validate_dir(directory, name=None, tileable=False):
    return [validate(n, f, tileable) for n, f in find_set(directory, name)]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Validate PBR texture sets (T_<Name>_BaseColor/ORM/Normal.png)")
    ap.add_argument("directory")
    ap.add_argument("--name", help="texture set name (default: every set in the folder)")
    ap.add_argument("--tileable", action="store_true", help="also check seamless tiling")
    ap.add_argument("--json", help="write the report as JSON here")
    args = ap.parse_args(argv)
    reports = validate_dir(args.directory, args.name, args.tileable)
    for rep in reports:
        print(rep.text())
    if args.json:
        Path(args.json).write_text(json.dumps([r.as_dict() for r in reports], indent=2))
    return 2 if any(r.worst == "FAIL" for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
