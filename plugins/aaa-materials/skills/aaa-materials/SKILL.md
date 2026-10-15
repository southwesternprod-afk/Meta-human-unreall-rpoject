---
name: aaa-materials
description: Build AAA-quality PBR textures and materials for game assets with headless Blender, then hand them to Unreal. Use when making or improving textures/materials (leather, steel, iron, rust, brass/bronze/gold, chainmail, wool, linen, wood, stone), baking a texture set for a mesh, making a seamless tileable material, checking textures against PBR rules, rendering a material ball, or generating an Unreal master material + material instance.
---

# AAA materials

A procedural material toolkit: smart-material presets with built-in wear and
dirt, a bake pipeline that writes Unreal-ready texture sets, a PBR validator,
material-ball previews and an Unreal import/material-instance generator.

**Toolkit root (`ROOT`)**: `${CLAUDE_PLUGIN_ROOT}` when loaded as a plugin, or
`plugins/aaa-materials` inside the MetaHuman repo. Library: `ROOT/lib/aaa_materials`,
CLI: `ROOT/scripts/aaa.py`, standards: `ROOT/skills/aaa-materials/reference/pbr-standards.md`.

## Setup (once per machine / container)

```bash
pip install bpy==5.2.* numpy pillow          # bpy needs Python 3.13; validate needs only numpy+pillow
apt-get install -y libopengl0 2>/dev/null || true
```
Headless notes: import `bpy` before `bmesh`/`mathutils`; render/bake with Cycles on CPU
(EEVEE/Workbench need a display).

## Pick the job

### 1. A seamless tileable material (floors, fabric, leather panels, trims)
```bash
python3 ROOT/scripts/aaa.py presets                                # list presets + params
python3 ROOT/scripts/aaa.py tile --preset leather --name LeatherDarkBrown --size 0.5 \
    --res 2048 --param color=#3b2618 --param wear=0.6 --out Art/Materials/LeatherDarkBrown
```
`--size` is the real-world width of one tile in metres (leather/metal 0.5, cloth 0.25-0.5,
wood/stone 1-2). The command bakes BaseColor/ORM/Normal(+OpenGL)/Height, makes them
seamless, renders `<Name>_Ball.png`, validates, and writes `Import_<Name>_Unreal.py`.
**Always look at the ball render** (Read the PNG) before calling it done.

### 2. Textures for a specific unwrapped mesh (hero assets)
Use the engine from Python so wear follows the real geometry (edge/cavity/AO are baked first):
```python
import sys; sys.path.insert(0, "ROOT/lib")
import bpy                                      # first
from aaa_materials import engine, library as L

def layers(c):
    part = c.attr("part")                       # any float attribute your build script wrote
    return [(None, L.leather(c, color="#3b2618", wear=0.6)),
            (c.is_value(part, 1), L.steel_blackened(c, wear=0.5)),
            (c.is_value(part, 3), L.chainmail(c, P2=c.attr_vec("pc")))]   # 2D pattern coords in metres

engine.bake(obj, out_dir, "MyAsset", layers, res=4096)   # obj must be UV-unwrapped
engine.assign_baked_material(obj, out_dir, "MyAsset")
```
- `mask_scale` (bake arg) scales the edge/cavity/AO probe distances: 1.0 for props and
  clothing, 3-10 for buildings.
- Weave/ring/quilt presets need 2D surface coords in metres: pass `P2` (a vector attribute)
  or rely on `UV * uv_size`.
- Custom looks: build a `Surface(base, rough, metal, height, bump)` from `c.g` (a node
  `Graph`: `noise`, `voronoi`, `wave`, `warp`, `remap`, `smooth`, `mix_color` ...) and
  `aaa_materials.weathering` masks (`edge_wear`, `cavity_dirt`, `dust`, `stains`,
  `scratches`, `creases`, `pits`, `breakup`).

### 3. Check textures
```bash
python3 ROOT/scripts/aaa.py validate path/to/Textures            # every T_*_BaseColor set in the folder
python3 ROOT/scripts/aaa.py validate path --name X --tileable --json report.json
```
Fix every FAIL; fix WARNs unless there is a stated artistic reason. Exit code 2 = FAIL.

### 4. Unreal
`python3 ROOT/scripts/aaa.py unreal <texdir> <Name> --tiling 2` writes
`Import_<Name>_Unreal.py`. The user runs it in Unreal (File > Execute Python Script).
It imports the textures with correct settings, creates the shared `M_AAA_Master`
once (Tiling, Tint, RoughnessMin/Max, MetallicScale, AOStrength, NormalStrength;
works on skeletal meshes) and `MI_<Name>`. Tell the user: drag `MI_<Name>` onto the mesh.

## AAA quality bar (check every material against this)
Read `reference/pbr-standards.md` for the numbers. In short:
1. **Values** - albedo inside PBR ranges (non-metal sRGB 30-240, metal >= 180); metallic
   is 0 or 1; dark metal = non-metal oxide/grime layer over bright metal.
2. **Three frequencies** - big shapes (panels, stains, tone), mid (creases, dents,
   wear), micro (grain, pores, weave). Missing any one reads as CG.
3. **Roughness tells the story** - it should vary more than colour: worn = smoother,
   grime/dust/rust = rougher.
4. **Wear follows use** - edges, high points and handled areas wear; dirt collects in
   cavities and on top faces. Never uniform noise everywhere.
5. **No visible patterns** - no voronoi webs, no repeating blotches; use `warp` and
   multi-scale `breakup`. Look at the render from 1-2 m and from 20 cm.
6. **Texel density** - characters ~10.24 px/cm (4K for a torso garment), props 5.12 px/cm.
7. **Deliver** BaseColor (sRGB), ORM (linear), Normal DirectX for Unreal, OpenGL copy
   for Blender/glTF, all power-of-two; validator passes; ball or asset render reviewed.
