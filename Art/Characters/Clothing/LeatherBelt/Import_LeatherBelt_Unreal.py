"""One-click import of the leather belt into Unreal Engine 5.

Before running: in the Content Browser, click your MetaHuman's BODY mesh
(the SKM_..._BodyMesh asset) so it is selected. The belt is imported onto
that body's skeleton.

Then: File > Execute Python Script... and pick this file.

It imports SKM_LeatherBelt.fbx and the three textures next to this file into
/Game/Characters/Clothing/LeatherBelt, builds the M_LeatherBelt material and
assigns it.
"""

import os

import unreal

DEST = "/Game/Characters/Clothing/LeatherBelt"
MESH_NAME = "SKM_LeatherBelt"
TEX = "LeatherBelt"

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # older editors don't set __file__ for executed scripts
    HERE = r"C:\PUT\THE\PATH\TO\THE\LeatherBelt\FOLDER\HERE"

tools = unreal.AssetToolsHelpers.get_asset_tools()
assets = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary


def find_skeleton():
    for asset in unreal.EditorUtilityLibrary.get_selected_assets():
        if isinstance(asset, unreal.Skeleton):
            return asset
        if isinstance(asset, unreal.SkeletalMesh):
            return asset.get_editor_property("skeleton")
    return None


def import_file(filename, dest, options=None):
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", os.path.join(HERE, filename))
    task.set_editor_property("destination_path", dest)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    if options is not None:
        task.set_editor_property("options", options)
    tools.import_asset_tasks([task])
    unreal.log("Imported {} -> {}".format(filename, list(task.get_editor_property("imported_object_paths"))))


def import_mesh(skeleton):
    ui = unreal.FbxImportUI()
    ui.set_editor_property("import_mesh", True)
    ui.set_editor_property("import_as_skeletal", True)
    ui.set_editor_property("mesh_type_to_import", unreal.FBXImportType.FBXIT_SKELETAL_MESH)
    ui.set_editor_property("skeleton", skeleton)
    ui.set_editor_property("import_materials", False)
    ui.set_editor_property("import_textures", False)
    ui.set_editor_property("import_animations", False)
    ui.set_editor_property("create_physics_asset", False)
    data = ui.get_editor_property("skeletal_mesh_import_data")
    data.set_editor_property("import_morph_targets", False)
    data.set_editor_property("normal_import_method", unreal.FBXNormalImportMethod.FBXNIM_IMPORT_NORMALS)
    import_file(MESH_NAME + ".fbx", DEST, ui)
    return unreal.load_asset("{}/{}".format(DEST, MESH_NAME))


def import_texture(name, srgb, compression):
    import_file(os.path.join("Textures", name + ".png"), DEST + "/Textures")
    tex = unreal.load_asset("{}/Textures/{}".format(DEST, name))
    tex.set_editor_property("srgb", srgb)
    tex.set_editor_property("compression_settings", compression)
    assets.save_loaded_asset(tex)
    return tex


def texture_node(mat, tex, sampler, x, y):
    node = mel.create_material_expression(mat, unreal.MaterialExpressionTextureSample, x, y)
    node.set_editor_property("texture", tex)
    node.set_editor_property("sampler_type", sampler)
    return node


def build_material(base, normal, orm):
    path = DEST + "/Materials/M_" + TEX
    if assets.does_asset_exist(path):
        assets.delete_asset(path)
    mat = tools.create_asset("M_" + TEX, DEST + "/Materials", unreal.Material, unreal.MaterialFactoryNew())
    mat.set_editor_property("used_with_skeletal_mesh", True)
    n_base = texture_node(mat, base, unreal.MaterialSamplerType.SAMPLERTYPE_COLOR, -500, -250)
    n_orm = texture_node(mat, orm, unreal.MaterialSamplerType.SAMPLERTYPE_MASKS, -500, 50)
    n_nrm = texture_node(mat, normal, unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL, -500, 350)
    mel.connect_material_property(n_base, "RGB", unreal.MaterialProperty.MP_BASE_COLOR)
    mel.connect_material_property(n_orm, "R", unreal.MaterialProperty.MP_AMBIENT_OCCLUSION)
    mel.connect_material_property(n_orm, "G", unreal.MaterialProperty.MP_ROUGHNESS)
    mel.connect_material_property(n_orm, "B", unreal.MaterialProperty.MP_METALLIC)
    mel.connect_material_property(n_nrm, "RGB", unreal.MaterialProperty.MP_NORMAL)
    mel.recompile_material(mat)
    assets.save_loaded_asset(mat)
    return mat


def make_lods(mesh, count=4):
    """Game-ready LODs: Unreal's skeletal mesh reduction builds LOD1-3 from the
    full-detail mesh, so the top stays cheap when the character is far away."""
    try:
        subsystem = unreal.get_editor_subsystem(unreal.SkeletalMeshEditorSubsystem)
        if subsystem.regenerate_lod(mesh, count, True, False):
            unreal.log("Generated {} LODs for {}".format(count, MESH_NAME))
            return
    except Exception as exc:  # older/newer editors without this call
        unreal.log_warning("LOD generation skipped ({}). You can add LODs in the mesh editor: "
                           "LOD Settings > Number of LODs = 4 > Apply Changes.".format(exc))


def main():
    skeleton = find_skeleton()
    if skeleton is None:
        unreal.log_error("Select your MetaHuman's body mesh (SKM_..._BodyMesh) in the Content Browser first, "
                         "then run this script again.")
        return
    unreal.log("Using skeleton {}".format(skeleton.get_path_name()))
    mesh = import_mesh(skeleton)
    if mesh is None:
        unreal.log_error("Could not find {}/{} after import - check the Output Log.".format(DEST, MESH_NAME))
        return

    base = import_texture("T_{}_BaseColor".format(TEX), True, unreal.TextureCompressionSettings.TC_DEFAULT)
    normal = import_texture("T_{}_Normal".format(TEX), False, unreal.TextureCompressionSettings.TC_NORMALMAP)
    orm = import_texture("T_{}_ORM".format(TEX), False, unreal.TextureCompressionSettings.TC_MASKS)

    mat = build_material(base, normal, orm)
    materials = mesh.get_editor_property("materials")
    for slot in materials:
        slot.set_editor_property("material_interface", mat)
    mesh.set_editor_property("materials", materials)
    make_lods(mesh)
    assets.save_loaded_asset(mesh)
    assets.sync_browser_to_objects(["{}/{}".format(DEST, MESH_NAME)])
    unreal.log("Leather belt imported to {}".format(DEST))


main()
