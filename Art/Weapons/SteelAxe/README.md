# SM_SteelAxe

Game-ready bearded battle axe built by `Tools/Blender/build_steel_axe.py`
(Blender 5.2), with a baked PBR texture set from `Tools/Blender/weapon_textures.py`.
Same quality bar and conventions as `../SteelSword`.

Details: forged bearded head with hammer facets, dark mill scale, a bright ground
cutting bevel and a gold-inlaid braided band on both cheeks; flared eye with a
flat poll; oiled ash haft with a swell under the head, an iron wedge in the end grain,
riveted langets, a leather-wrapped grip between grooved collars, and a flared butt cap.
Hard metal edges have micro-bevels with weighted normals.

| | |
|---|---|
| Length | 81 cm (head 21 cm from poll to edge, 21 cm edge-to-beard) |
| Mesh | ~12k verts / ~23k tris, one UV channel |
| Material | `M_SteelAxe` (single texture set) |
| Textures (2048²) | `T_SteelAxe_BaseColor` (sRGB), `T_SteelAxe_Normal` (DirectX), `T_SteelAxe_ORM` (R AO, G roughness, B metallic) |
| Collision | `UCX_SM_SteelAxe_00` (head), `UCX_SM_SteelAxe_01` (haft) |
| Sockets | `SOCKET_BladeBase`, `SOCKET_BladeTip` (same names as the sword) |
| Pivot | Centre of the grip; haft along +Z, blade points +X |

## Import into Unreal (easy way)

1. Download this whole `SteelAxe` folder (keep the `Textures` folder inside it).
2. In Unreal: **File → Execute Python Script…** and pick `Import_SteelAxe_Unreal.py`.
3. The axe appears in `Content/Weapons/SteelAxe`, fully textured.

## Import into Unreal (by hand)

1. Drag `SM_SteelAxe.fbx` into the Content Browser. Untick **Auto Generate Collision**, click **Import All**.
2. Drag the three PNGs from `Textures/` in (skip `T_SteelAxe_Normal_OpenGL.png`, that one is for Blender).
3. Open `T_SteelAxe_ORM`: untick **sRGB**, set **Compression Settings** to *Masks*. Save.
4. Create a material `M_SteelAxe` with three Texture Sample nodes:
   - BaseColor → **Base Color**
   - ORM: **R → Ambient Occlusion**, **G → Roughness**, **B → Metallic**
   - Normal → **Normal**
5. Open the axe mesh and set its material to `M_SteelAxe`.

## Rebuild

```
pip install bpy
python3 Tools/Blender/build_steel_axe.py              # mesh + textures + exports
python3 Tools/Blender/build_steel_axe.py --render     # also preview renders
```
