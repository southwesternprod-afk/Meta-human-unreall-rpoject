# SKM_LeatherTrousers

The leather trousers for the leather jerkin set: their **own wearable item** (separate
skeletal mesh on the MetaHuman body skeleton), so they can be equipped and removed in
game. Built by `Tools/Blender/build_leather_trousers.py` to sit under the jerkin's skirt
(checked: no clipping through the skirt or the mail, none into the body).

- dark leather legs, straight cut: they hang from the knee and the calf instead of
  hugging the shin, and stack in soft breaks above the hem (boots go over or under)
- sunken outseams, inseams and rise seams with double top-stitching; a dark wool
  gusset in the crotch; compression folds behind the knees; drag folds from the crotch
- sewn-on waistband with four belt loops and a buttoned fly
- padded boiled-leather knee cops, strapped round the leg above and below the knee
  with iron buckles
- worn paler on the fronts of the thighs; dried mud on the lower legs

| | |
|---|---|
| Mesh | ~55k tris, skinned to the MetaHuman body skeleton (buckles and buttons rigid) |
| Material | `M_LeatherTrousers` (single texture set) |
| Textures (2048²) | `T_LeatherTrousers_BaseColor`, `T_LeatherTrousers_Normal` (DirectX), `T_LeatherTrousers_ORM` |

## Import into Unreal

1. Download this whole `LeatherTrousers` folder (keep `Textures` inside it).
2. In the Content Browser, click your MetaHuman's body mesh (`SKM_John_doe_BodyMesh`).
3. **File → Execute Python Script…** and pick `Import_LeatherTrousers_Unreal.py`.
4. The trousers appear in `Content/Characters/Clothing/LeatherTrousers` (with LODs).

## Put them on / take them off in game

1. In `BP_John_doe`: if it has a **Legs** component, select it and set **Skeletal Mesh
   Asset** to `SKM_LeatherTrousers` (it already follows the body). If not: **Add →
   Skeletal Mesh Component** as a child of **Body**, name it `Trousers`, set its mesh to
   `SKM_LeatherTrousers`, and on **Event BeginPlay** call **Set Leader Pose Component**
   (Target: `Trousers`, New Leader: `Body`).
2. To remove / re-equip them: **Set Visibility** on the component (or **Set Skeletal Mesh
   Asset** to none / `SKM_LeatherTrousers`).

## Rebuild

```
python3 Tools/Blender/build_leather_trousers.py --export --render
```
(The outfit renders also load the jerkin and the belt, so build those first.)
