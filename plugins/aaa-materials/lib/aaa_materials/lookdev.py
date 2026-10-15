"""Material-ball renders for checking a baked texture set (Cycles, CPU).

    lookdev.material_ball(tex_dir, "LeatherBrown", out_png, tiling=2.0)

Scene: a sphere and a bevelled block on a neutral floor under a soft key,
fill and rim light, so colour, roughness response and normal detail can all
be judged in one frame.
"""

import math
from pathlib import Path

import bpy
from mathutils import Vector

from . import engine


def _look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _area(name, loc, energy, size, target):
    light = bpy.data.objects.new(name, bpy.data.lights.new(name, "AREA"))
    light.data.energy, light.data.size = energy, size
    light.location = loc
    bpy.context.scene.collection.objects.link(light)
    _look_at(light, target)
    return light


def studio(res=(1200, 800), samples=48):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    try:
        sc.cycles.use_denoising = True
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        sc.cycles.use_denoising = False
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.view_settings.view_transform = "AgX"
    world = bpy.data.worlds.new("Lookdev")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.02, 0.021, 0.023, 1)
    target = (0, 0, 0.35)
    _area("Key", (-1.6, -2.0, 2.2), 160, 1.6, target)
    _area("Fill", (2.2, -1.2, 1.0), 35, 2.5, target)
    _area("Rim", (0.8, 2.2, 1.8), 220, 1.2, target)
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.location = (0.0, -2.4, 1.05)
    cam.data.lens = 55
    _look_at(cam, target)
    floor_mat = bpy.data.materials.new("Floor")
    floor_mat.use_nodes = True
    bsdf = floor_mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.05, 0.05, 0.05, 1)
    bsdf.inputs["Roughness"].default_value = 0.8
    bpy.ops.mesh.primitive_plane_add(size=12)
    bpy.context.object.data.materials.append(floor_mat)


def _props():
    objs = []
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=0.36, location=(-0.42, 0, 0.36))
    sphere = bpy.context.object
    bpy.ops.object.shade_smooth()
    for loop_uv in sphere.data.uv_layers.active.data:      # equal texel aspect around the sphere
        loop_uv.uv.x *= 2.0
    objs.append(sphere)
    bpy.ops.mesh.primitive_cube_add(size=0.56, location=(0.45, 0.05, 0.28))
    block = bpy.context.object
    block.rotation_euler = (0, 0, math.radians(28))
    bev = block.modifiers.new("Bevel", "BEVEL")
    bev.width, bev.segments = 0.03, 4
    bpy.ops.object.modifier_apply(modifier="Bevel")
    bpy.ops.object.shade_smooth()
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.cube_project(cube_size=0.56)
    bpy.ops.object.mode_set(mode="OBJECT")
    objs.append(block)
    return objs


def material_ball(tex_dir, name, out_png, tiling=2.0, res=(1200, 800), samples=48, fresh_scene=True):
    """Render the baked set `name` from `tex_dir` on the lookdev props."""
    if fresh_scene:
        bpy.ops.wm.read_factory_settings(use_empty=True)
    studio(res, samples)
    for obj in _props():
        engine.assign_baked_material(obj, tex_dir, name, tiling)
    sc = bpy.context.scene
    sc.render.filepath = str(out_png)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    return Path(out_png)


def contact_sheet(pngs, out_png, columns=3, label=True):
    """Tile several material-ball renders into one image (Pillow)."""
    from PIL import Image, ImageDraw
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    w, h = imgs[0].size
    rows = math.ceil(len(imgs) / columns)
    sheet = Image.new("RGB", (w * columns, h * rows), (20, 20, 22))
    for i, (img, p) in enumerate(zip(imgs, pngs)):
        x, y = (i % columns) * w, (i // columns) * h
        sheet.paste(img, (x, y))
        if label:
            ImageDraw.Draw(sheet).text((x + 16, y + 12), Path(p).stem.replace("_Ball", ""), fill=(230, 230, 230))
    sheet.save(out_png)
    return Path(out_png)
