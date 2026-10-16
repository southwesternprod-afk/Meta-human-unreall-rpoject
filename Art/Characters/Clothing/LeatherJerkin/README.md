# SKM_LeatherJerkin

Medieval fantasy top for the MetaHuman **John_doe**, built by
`Tools/Blender/build_leather_jerkin.py` from the exported body
`Art/Characters/Source/SKM_John_doe_BodyMesh.fbx`.

Pieces (one skeletal mesh, one material):

- leather jerkin: torso with soft folds and sunken panel seams, stand-up collar, V-neck,
  centre placket with five buckled closure straps (rounded tips)
- A-line skirt of eight sewn gores (no slits) from the top of the hips, flush with the jerkin at a slim stitched
  waist welt with four belt loops (the separate `LeatherBelt` passes under them); a short
  mail skirt shows below the hem
- two buckled shoulder straps holding the pauldrons, crossing in an X front and back,
  each with its free end through the buckle and held by a keeper
- quilted wool gambeson sleeves; riveted mail over the upper arms and at the throat,
  edged with a border of brass rings
- leather bracers with three buckled straps each
- laminated steel pauldrons (cap + four lames, rolled edges, riveted) with an embossed
  fleur-de-lis, plus an arming strap round each upper arm
- rounded forged-wire buckles throughout

Gloves, belt and pouches are separate items (not included here).

| | |
|---|---|
| Mesh | ~109k tris / ~56k verts at LOD0 (game mesh; detail baked from a ~260k-tri source), skinned to the MetaHuman body skeleton (341 bones, max 8 influences); Unreal builds LOD1-3 on import |
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
4. Leather pieces are 3-4.5 mm thick with rolled, rounded edges. The script builds two
   versions: a bake source (subdivided torso and sleeves with folds) and the game mesh
   (game-resolution torso and sleeves, gambeson hidden under mail and bracers removed).
   The source is baked onto the game mesh (high-to-low), so the folds live in the normal
   map. Hidden inner faces get less texture space; outer surfaces are reported in px/cm.
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
