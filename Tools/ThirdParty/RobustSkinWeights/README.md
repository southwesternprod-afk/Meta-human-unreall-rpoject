# Robust Skin Weights Transfer (vendored)

`robust_weights.py` is `src/utils.py` from
[rin-23/RobustSkinWeightsTransferCode](https://github.com/rin-23/RobustSkinWeightsTransferCode)
at commit `a60263c17f38625b98a62bea09c91f61f7b865f0`. MIT licensed (see `LICENSE`).

Local changes, for the libigl 2.6 Python API:
- `igl.barycentric_coordinates_tri` -> `igl.barycentric_coordinates`
- `inpaint()`: `igl.min_quad_with_fixed` now returns the solution directly (and raises on
  failure), so the call is wrapped to keep the original `(weights, success)` return value.

Verified with the upstream sphere-to-plane example: 16/256 vertices matched, the rest
inpainted to the exact expected weights.

It implements "Robust Skin Weights Transfer via Weight Inpainting" (Abdrashitov et al.,
SIGGRAPH Asia 2023): copy skin weights from the body only where the garment surface is
close and similarly oriented, then solve for the rest so loose parts (collars, flared
hems, pauldrons) deform smoothly instead of picking up the wrong bones.

Used by the clothing build scripts to skin garments from the MetaHuman body.
Requires `pip install libigl scipy numpy` (see `Tools/requirements.txt`).
