"""Edge padding for texture sets (numpy + scipy + Pillow, no Blender needed).

Unused texture space left black bleeds into UV seams once the engine builds
mip-maps (dark outlines on the model at a distance). Padding fills every unused
pixel with the nearest island pixel, as texturing tools do on export.
"""

from pathlib import Path

import numpy as np

SUFFIXES = ("BaseColor", "ORM", "Normal", "Normal_OpenGL", "Height")


def empty_mask(base, orm):
    """Pixels no UV island (or bake margin) touched: pure black in both maps."""
    return (base[..., :3].max(-1) == 0) & (orm[..., :3].max(-1) == 0)


def pad(arrays, empty):
    """Fill `empty` pixels of every HxWxC array with the nearest used pixel."""
    if not empty.any() or empty.all():
        return arrays
    from scipy.ndimage import distance_transform_edt
    iy, ix = distance_transform_edt(empty, return_distances=False, return_indices=True)
    return {k: a[iy, ix] for k, a in arrays.items()}


def pad_set(directory, name):
    """Pad T_<name>_*.png in place. Returns the fraction of pixels filled."""
    from PIL import Image
    d = Path(directory)
    files = {s: d / f"T_{name}_{s}.png" for s in SUFFIXES if (d / f"T_{name}_{s}.png").exists()}
    arrays = {s: np.asarray(Image.open(p).convert("RGB")) for s, p in files.items()}
    empty = empty_mask(arrays["BaseColor"], arrays["ORM"])
    padded = pad(arrays, empty)
    for s, p in files.items():
        Image.fromarray(padded[s]).save(p, optimize=False)
    return float(empty.mean())
