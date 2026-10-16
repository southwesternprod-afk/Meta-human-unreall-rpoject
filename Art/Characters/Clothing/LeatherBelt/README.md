# SKM_LeatherBelt

The leather belt for the leather jerkin set: its **own wearable item** (separate skeletal
mesh on the MetaHuman body skeleton), so it can be equipped and removed in game. Built by
`Tools/Blender/build_leather_belt.py` to sit over the jerkin's waist welt and pass under
its four belt loops (both use the same belt shape, so they always fit together).

- 44 mm leather belt, stitched and burnished edges
- forged buckle with a riveted leather chape; punched tongue through a keeper; the end
  tucked under the belt and hanging down the front with an iron strap end
- five leather frogs with iron D-rings: the **attachment slots** for the separate pouches
  and sheaths

| | |
|---|---|
| Mesh | ~10.6k tris, skinned to the MetaHuman body skeleton (buckle, rings and strap end rigid) |
| Material | `M_LeatherBelt` (single texture set) |
| Textures (2048²) | `T_LeatherBelt_BaseColor`, `T_LeatherBelt_Normal` (DirectX), `T_LeatherBelt_ORM` |

## Attachment slots

`BeltSlots.json` lists each D-ring's hook point, outward direction and belt direction:

| Slot | Where | Intended for |
|---|---|---|
| `PouchR` / `PouchL` | front of each hip (±45°) | belt pouches |
| `SheathR` / `SheathL` | side of each hip (±95°) | sword / dagger sheaths |
| `Back` | centre back | satchel, knife, tools |

The pouches and sheaths are made as separate skeletal meshes hung from these rings, so in
game each one is just another piece you equip (or not) - no sockets or re-rigging needed.

## Import into Unreal

1. Download this whole `LeatherBelt` folder (keep `Textures` inside it).
2. In the Content Browser, click your MetaHuman's body mesh (`SKM_John_doe_BodyMesh`).
3. **File → Execute Python Script…** and pick `Import_LeatherBelt_Unreal.py`.

## Put it on / take it off in game

1. In `BP_John_doe`: **Add → Skeletal Mesh Component** as a child of **Body**, name it
   `Belt`, set its mesh to `SKM_LeatherBelt`.
2. In the Event Graph on **Event BeginPlay**: **Set Leader Pose Component** (Target: `Belt`,
   New Leader: `Body`) so it follows the body's animation.
3. To remove / re-equip it: **Set Visibility** on `Belt` (or **Set Skeletal Mesh Asset** to
   none / `SKM_LeatherBelt`).

## Rebuild

```
python3 Tools/Blender/build_leather_belt.py --export --render
```
(Build the jerkin first: the belt is fitted to it and the renders show both.)
