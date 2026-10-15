"""One-click import of the steel axe into Unreal Engine 5.

In the Unreal Editor: File > Execute Python Script... and pick this file.
(Needs the "Python Editor Script Plugin", which is on by default in UE5.)

It imports SM_SteelAxe.fbx and the three textures next to this file into
/Game/Weapons/SteelAxe, builds the M_SteelAxe material from them and
assigns it to the mesh.
"""

import os

import unreal

DEST = "/Game/Weapons/SteelAxe"
MESH_NAME = "SM_SteelAxe"

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # older editors don't set __file__ for executed scripts
    HERE = r"C:\PUT\THE\PATH\TO\THE\SteelAxe\FOLDER\HERE"

tools = unreal.AssetToolsHelpers.get_asset_tools()
assets = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary


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
    paths = task.get_editor_property("imported_object_paths")
    unreal.log("Imported {} -> {}".format(filename, list(paths)))
    return paths


def import_mesh():
    ui = unreal.FbxImportUI()
    ui.set_editor_property("import_mesh", True)
    ui.set_editor_property("import_as_skeletal", False)
    ui.set_editor_property("import_materials", False)
    ui.set_editor_property("import_textures", False)
    ui.set_editor_property("mesh_type_to_import", unreal.FBXImportType.FBXIT_STATIC_MESH)
    data = ui.get_editor_property("static_mesh_import_data")
    data.set_editor_property("auto_generate_collision", False)
    data.set_editor_property("combine_meshes", True)
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
    path = DEST + "/Materials/M_SteelAxe"
    if assets.does_asset_exist(path):
        assets.delete_asset(path)
    mat = tools.create_asset("M_SteelAxe", DEST + "/Materials", unreal.Material,
                             unreal.MaterialFactoryNew())

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


def main():
    mesh = import_mesh()
    if mesh is None:
        unreal.log_error("Could not find {}/{} after import - check the Output Log.".format(DEST, MESH_NAME))
        return

    base = import_texture("T_SteelAxe_BaseColor", True, unreal.TextureCompressionSettings.TC_DEFAULT)
    normal = import_texture("T_SteelAxe_Normal", False, unreal.TextureCompressionSettings.TC_NORMALMAP)
    orm = import_texture("T_SteelAxe_ORM", False, unreal.TextureCompressionSettings.TC_MASKS)

    mat = build_material(base, normal, orm)
    mesh.set_material(0, mat)
    assets.save_loaded_asset(mesh)
    assets.sync_browser_to_objects(["{}/{}".format(DEST, MESH_NAME)])
    unreal.log("Steel axe imported to {}".format(DEST))


main()
