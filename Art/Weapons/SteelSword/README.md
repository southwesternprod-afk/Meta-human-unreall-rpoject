# SM_SteelSword

Procedural steel longsword built by `Tools/Blender/build_steel_sword.py` (Blender 5.2).

Details: fullered blade with a flat ricasso and a secondary cutting bevel,
curved crossguard with a centre block and langets, leather-wrapped grip over
three risers, grooved ferrules, recessed wheel pommel with a domed peen.

| | |
|---|---|
| Length | 104 cm (80 cm blade, 17 cm grip) |
| Guard span | 20 cm |
| Mesh | ~10.4k verts, one UV channel |
| Materials | `M_SwordBlade_Steel`, `M_SwordFittings_Steel`, `M_SwordGrip_Leather` |
| Collision | `UCX_SM_SteelSword_00..02` (blade, guard, handle) |
| Sockets | `SOCKET_BladeBase`, `SOCKET_BladeTip` (for trails / hit traces) |
| Pivot | Centre of the grip; blade points +Z, edges along X |

## Import into Unreal

1. Drag `SM_SteelSword.fbx` into the Content Browser.
2. In the FBX import dialog, leave *Import Uniform Scale* at 1.0 and keep
   *Auto Generate Collision* off so the `UCX_` hulls are used.
3. Materials import as flat colours. Suggested values for Unreal material instances:
   - Blade steel: Base Color (0.78, 0.78, 0.80), Metallic 1, Roughness 0.16
   - Fittings steel: Base Color (0.48, 0.48, 0.50), Metallic 1, Roughness 0.32
   - Leather: Base Color (0.065, 0.028, 0.013), Metallic 0, Roughness 0.6
4. Generate LODs with Unreal's built-in reduction (Static Mesh Editor → LOD Settings) if needed.

## Rebuild

```
pip install bpy
python3 Tools/Blender/build_steel_sword.py            # mesh + exports
python3 Tools/Blender/build_steel_sword.py --render   # also preview renders
```

Dimensions, curvature and materials are constants at the top of the script.
