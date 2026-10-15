# SM_SteelSword

Game-ready steel longsword built by `Tools/Blender/build_steel_sword.py`
(Blender 5.2), with a baked PBR texture set from `Tools/Blender/sword_textures.py`.

Details: fullered blade with a flat ricasso, a secondary cutting bevel and a
gold-inlaid inscription (`+ IN NOMINE DOMINI +`), curved crossguard with a raised
centre block, leather-wrapped grip over three risers, grooved ferrules, and a recessed
wheel pommel with a domed peen. Hard metal edges have micro-bevels with weighted normals.

| | |
|---|---|
| Length | 104 cm (80 cm blade, 17 cm grip) |
| Guard span | 20 cm |
| Mesh | ~15k verts / ~30k tris, one UV channel |
| Material | `M_SteelSword` (single texture set) |
| Textures (2048²) | `T_SteelSword_BaseColor` (sRGB), `T_SteelSword_Normal` (DirectX), `T_SteelSword_ORM` (R AO, G roughness, B metallic) |
| Collision | `UCX_SM_SteelSword_00..02` (blade, guard, handle) |
| Sockets | `SOCKET_BladeBase`, `SOCKET_BladeTip` (for trails / hit traces) |
| Pivot | Centre of the grip; blade points +Z, edges along X |

Texture detail: brushed steel, polished cutting edges, scratches, grime in crevices,
worn bright edges and forging pits on the fittings, leather grain with hand-wear.

## Import into Unreal (easy way)

1. Download this whole `SteelSword` folder (keep the `Textures` folder inside it).
2. In Unreal: **File → Execute Python Script…** and pick `Import_SteelSword_Unreal.py`.
3. The sword appears in `Content/Weapons/SteelSword`, fully textured.

## Import into Unreal (by hand)

1. Drag `SM_SteelSword.fbx` into the Content Browser. Untick **Auto Generate Collision**, click **Import All**.
2. Drag the three PNGs from `Textures/` in (skip `T_SteelSword_Normal_OpenGL.png`, that one is for Blender).
3. Open `T_SteelSword_ORM`: untick **sRGB**, set **Compression Settings** to *Masks*. Save.
4. Create a material `M_SteelSword` with three Texture Sample nodes:
   - BaseColor → **Base Color**
   - ORM: **R → Ambient Occlusion**, **G → Roughness**, **B → Metallic**
   - Normal → **Normal**
5. Open the sword mesh and set its material to `M_SteelSword`.

## Rebuild

```
pip install bpy
python3 Tools/Blender/build_steel_sword.py              # mesh + textures + exports
python3 Tools/Blender/build_steel_sword.py --render     # also preview renders
```

Dimensions are constants at the top of `build_steel_sword.py`; colours, wear amounts
and the inscription placement live in `sword_textures.py`.
