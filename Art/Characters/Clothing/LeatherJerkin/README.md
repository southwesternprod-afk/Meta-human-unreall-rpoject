# SKM_LeatherJerkin

Medieval fantasy top for the MetaHuman **John_doe**, built by
`Tools/Blender/build_leather_jerkin.py` from the exported body
`Art/Characters/Source/SKM_John_doe_BodyMesh.fbx`.

Pieces (one skeletal mesh, one material):

- leather jerkin: torso with soft folds and sunken panel seams, short skirt (upper thigh,
  made to go over trousers) with front/back slits, stand-up collar, V-neck, waist band,
  centre placket with five buckled closure straps; riveted mail fauld showing through
  the skirt slits
- two buckled shoulder straps holding the pauldrons: each goes over one shoulder,
  across the chest, under the other armpit and up across the back (an X front and back)
- quilted wool gambeson sleeves; riveted mail over the upper arms and under the V-neck
- leather bracers with three buckled straps each
- laminated steel pauldrons (cap + four lames, riveted) with an embossed fleur-de-lis,
  shaped as slightly flattened domes that clear the straps and mail underneath

Gloves, belt and pouches are separate items (not included here).

| | |
|---|---|
| Mesh | ~138k verts, skinned to the MetaHuman body skeleton (341 bones, max 8 influences) |
| Material | `M_LeatherJerkin` (single texture set) |
| Textures (4096²) | `T_LeatherJerkin_BaseColor` (sRGB), `T_LeatherJerkin_Normal` (DirectX), `T_LeatherJerkin_ORM` (R AO, G roughness, B metallic) |

## How it was made

1. The body is welded and its neck hole closed with an estimated neck (the real neck
   belongs to the head mesh).
2. Each layer is an offset shell of the body, relaxed so leather drapes instead of
   hugging muscle, and kept a minimum distance off the skin: gambeson 8 mm,
   jerkin 13.5 mm, mail 14 mm, bracers 16 mm. Pauldrons, straps, collar and skirt are
   generated around the bones / on the jerkin surface.
3. Skin weights come from the body via robust weight transfer with inpainting
   (`Tools/ThirdParty/RobustSkinWeights`); steel plates, buckles and rivets are rigid.
4. Leather pieces are 3-4.5 mm thick with rolled, rounded edges; the jerkin body is
   subdivided and displaced with folds (waist, armpits, back) and seams.
5. Textures are baked with the repo's `aaa-materials` plugin (`plugins/aaa-materials`):
   leather, blackened steel, quilted wool and mail presets with wear and grime, plus the
   jerkin's own seams, stitching, crest and etched borders. Unused texture space is
   edge-padded, and the set is checked with the plugin's PBR validator
   (`python3 plugins/aaa-materials/scripts/aaa.py validate Textures`).
6. Exported onto the untouched skeleton and verified by re-importing next to the body
   (identical bones) and posing both.

## Import into Unreal

1. Download this whole `LeatherJerkin` folder (keep `Textures` inside it).
2. In the Content Browser, click your MetaHuman's body mesh (`SKM_John_doe_BodyMesh`).
3. **File → Execute Python Script…** and pick `Import_LeatherJerkin_Unreal.py`.
4. The top appears in `Content/Characters/Clothing/LeatherJerkin`.
   The script also builds 4 LODs so the top stays cheap at a distance.

## Put it on the character

1. Open the character Blueprint (`BP_John_doe`).
2. If it has a **Torso** component: select it and set **Skeletal Mesh Asset** to
   `SKM_LeatherJerkin`. Done - it already follows the body.
3. If not: **Add → Skeletal Mesh Component** as a child of **Body**, set its mesh to
   `SKM_LeatherJerkin`, then in the Event Graph on **Event BeginPlay** call
   **Set Leader Pose Component** (Target: the new component, New Leader: Body).
   This makes the top copy the body's animation.

## Rebuild

```
pip install -r Tools/requirements.txt
python3 Tools/Blender/build_leather_jerkin.py --export            # mesh, skinning, 4K textures, FBX
python3 Tools/Blender/build_leather_jerkin.py --export --render   # also studio renders
```
