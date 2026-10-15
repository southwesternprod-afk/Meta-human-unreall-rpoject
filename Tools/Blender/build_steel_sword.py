"""Build a steel longsword procedurally in Blender and export it for Unreal.

Run headless with the `bpy` module (pip install bpy) or inside Blender:

    python3 Tools/Blender/build_steel_sword.py            # build + export
    python3 Tools/Blender/build_steel_sword.py --render   # also render previews

Outputs go to Art/Weapons/SteelSword/:
    SM_SteelSword.fbx   - static mesh + UCX_ collision + SOCKET_ empties (Unreal)
    SM_SteelSword.glb   - mesh only (web preview / other engines)
    SM_SteelSword.blend - source scene
    Renders/*.png       - preview renders (with --render)

Conventions: metres in Blender (exported so 1 m = 100 uu in Unreal), Z up,
blade points +Z, blade edges along X, flat of the blade faces +/-Y,
origin at the centre of the grip (where the hand holds it).
"""

import math
import sys
from pathlib import Path

import bpy  # must come before bmesh when running as a module
import bmesh
from mathutils import Vector

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "Art" / "Weapons" / "SteelSword"
ASSET_NAME = "SM_SteelSword"

# --- Dimensions (metres) ---------------------------------------------------
GRIP_HALF = 0.085            # grip runs z = -0.085 .. +0.085
GUARD_Z = 0.095              # centre of the crossguard
GUARD_HALF_SPAN = 0.10       # quillon tip to tip = 20 cm
BLADE_START = 0.098          # blade root sits just inside the guard
BLADE_LEN = 0.80
BLADE_W0 = 0.052             # width at the root
BLADE_W1 = 0.040             # width where the tip curve begins
BLADE_T0 = 0.0075            # thickness at the root
BLADE_T1 = 0.0045            # thickness near the tip
FULLER_DEPTH = 0.0016
FULLER_END = 0.68            # fraction of blade length where the fuller ends
RICASSO_END = 0.055          # fraction of blade length left flat and unsharpened
POMMEL_Z = -0.115


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


# --- Mesh helpers -----------------------------------------------------------
def loft(name, rings, cap_start=True, cap_end=True, tip=None, sharp_cols=()):
    """Skin a list of equal-length vertex rings into a closed mesh object.

    sharp_cols: ring indices whose lengthwise edges get a hard crease.
    """
    bm = bmesh.new()
    vrings = [[bm.verts.new(p) for p in ring] for ring in rings]
    n = len(rings[0])
    for a, b in zip(vrings, vrings[1:]):
        for j in range(n):
            k = (j + 1) % n
            bm.faces.new((a[j], a[k], b[k], b[j]))
        for j in sharp_cols:
            bm.edges.get((a[j], b[j])).smooth = False
    if cap_start:
        bm.faces.new(list(reversed(vrings[0])))
    if tip is not None:
        tv = bm.verts.new(tip)
        last = vrings[-1]
        for j in range(n):
            bm.faces.new((last[j], last[(j + 1) % n], tv))
    elif cap_end:
        bm.faces.new(vrings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def hull_object(name, points):
    bm = bmesh.new()
    for p in points:
        bm.verts.new(p)
    bmesh.ops.convex_hull(bm, input=bm.verts)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def make_material(name, color, metallic, roughness):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    mat.diffuse_color = (*color, 1.0)
    return mat


# --- Parts ------------------------------------------------------------------
BEVEL = 0.44   # where the cutting bevel starts, as a fraction of blade width
FULLER_HALF = 0.135
BLADE_XS = [0.5, 0.47, BEVEL, 0.40, 0.34, 0.28, 0.22, 0.18, FULLER_HALF, 0.10, 0.05, 0.0]


def blade_section(w, t, fuller, ricasso):
    """Blade cross-section: a lens-shaped main grind, a steeper cutting bevel
    at each edge and a rounded fuller groove. `ricasso` (0..1) blends towards
    the flat-sided, unsharpened section next to the guard."""
    top = []
    for xf in BLADE_XS + [-x for x in reversed(BLADE_XS[:-1])]:
        a = abs(xf)
        if a <= BEVEL:
            lens = 0.5 * t * (0.30 + 0.70 * (1 - (a / BEVEL) ** 1.8))
        else:
            lens = 0.5 * t * 0.30 * (0.5 - a) / (0.5 - BEVEL)
        flat = 0.5 * t if a <= BEVEL else 0.5 * t * (0.5 - a) / (0.5 - BEVEL)
        y = lerp(lens, flat, ricasso)
        if a < FULLER_HALF:
            y -= fuller * math.cos(math.pi * a / (2 * FULLER_HALF))
        top.append((xf * w, y))
    # top runs +x edge -> -x edge; bottom mirrors it back (edges shared)
    bottom = [(x, -y) for x, y in reversed(top[1:-1])]
    return top + bottom


def blade_sharp_cols():
    top = BLADE_XS + [-x for x in reversed(BLADE_XS[:-1])]
    n_top = len(top)
    cols = [i for i, xf in enumerate(top) if abs(xf) in (BEVEL, FULLER_HALF)]
    # bottom side is the top side mirrored, skipping the two shared edge verts
    cols += [n_top + (n_top - 2 - i) for i in cols]
    return cols


def build_blade():
    rings = []
    straight = 0.78
    steps_body, steps_tip = 64, 26
    zs = sorted(set([i / steps_body * straight for i in range(steps_body + 1)]
                    + [RICASSO_END + (i - 6) * 0.004 for i in range(14)]))
    zs += [straight + (1 - straight) * (i / steps_tip) ** 0.85 * 0.985
           for i in range(1, steps_tip + 1)]
    for s in zs:
        if s <= straight:
            w = lerp(BLADE_W0, BLADE_W1, s / straight)
            t = lerp(BLADE_T0, BLADE_T1, s / straight)
        else:
            u = (s - straight) / (1 - straight)
            shape = (1 - u ** 1.7) ** 0.55          # ogival point
            w = BLADE_W1 * shape
            t = BLADE_T1 * (0.25 + 0.75 * shape)
        fuller = FULLER_DEPTH * (1 - smoothstep((s - (FULLER_END - 0.06)) / 0.06))
        fuller *= smoothstep((s - RICASSO_END) / 0.035)   # fuller starts past the ricasso
        ricasso = 1 - smoothstep((s - RICASSO_END) / 0.012)
        z = BLADE_START + s * BLADE_LEN
        rings.append([Vector((x, y, z)) for x, y in blade_section(w, t, fuller, ricasso)])
    tip = Vector((0, 0, BLADE_START + BLADE_LEN))
    return loft("Blade", rings, cap_start=True, tip=tip, sharp_cols=blade_sharp_cols())


OCTAGON = [(1, 0.45), (0.45, 1), (-0.45, 1), (-1, 0.45),
           (-1, -0.45), (-0.45, -1), (0.45, -1), (1, -0.45)]


def build_guard():
    rings = []
    steps = 48
    for i in range(steps + 1):
        x = lerp(-GUARD_HALF_SPAN, GUARD_HALF_SPAN, i / steps)
        u = abs(x) / GUARD_HALF_SPAN
        hy = lerp(0.0125, 0.0070, u ** 0.8)
        hz = lerp(0.0105, 0.0060, u ** 0.8)
        knob = math.exp(-((u - 0.93) / 0.055) ** 2)   # flared terminals
        hy += 0.0040 * knob
        hz += 0.0035 * knob
        if u > 0.985:                                  # round off the ends
            hy *= 0.6
            hz *= 0.6
        zc = GUARD_Z + 0.009 * u ** 2                  # quillons sweep toward the blade
        rings.append([Vector((x, oy * hy, zc + oz * hz)) for oy, oz in OCTAGON])
    return loft("Guard", rings)


def build_guard_block():
    """Raised centre block (escutcheon) where the blade passes through the guard."""
    hx, hy, hz, c = 0.017, 0.0148, 0.0118, 0.004
    section = [(hx, hy - c), (hx - c, hy), (-hx + c, hy), (-hx, hy - c),
               (-hx, -hy + c), (-hx + c, -hy), (hx - c, -hy), (hx, -hy + c)]
    rings = []
    for z, k in ((-hz, 0.6), (-hz + c, 1.0), (hz - c, 1.0), (hz, 0.6)):
        rings.append([Vector((x * k, y * k, GUARD_Z + z)) for x, y in section])
    return loft("GuardBlock", rings)


def build_grip():
    rings = []
    seg, steps = 32, 120
    pitch = 0.012
    for i in range(steps + 1):
        z = lerp(-GRIP_HALF, GRIP_HALF, i / steps)
        s = (z + GRIP_HALF) / (2 * GRIP_HALF)
        swell = 1 + 0.08 * math.sin(math.pi * s)      # slight barrel shape
        ring = []
        for j in range(seg):
            a = 2 * math.pi * j / seg
            wrap = max(0.0, math.cos(2 * math.pi * z / pitch + a)) ** 10
            risers = sum(math.exp(-((s - c) / 0.022) ** 2) for c in (0.25, 0.5, 0.75))
            r = swell * (1 + 0.055 * risers) * (1 - 0.055 * wrap)   # risers under a spiral wrap
            ring.append(Vector((0.0155 * r * math.cos(a), 0.0125 * r * math.sin(a), z)))
        rings.append(ring)
    return loft("Grip", rings)


def build_ring(name, z0, z1, rx, ry, seg=32, bevel=0.0012, groove=0.0):
    rings = []
    zm = (z0 + z1) / 2
    profile = [(z0, -bevel), (z0 + bevel, 0)]
    if groove:
        profile += [(zm - 0.0012, 0), (zm - 0.0005, -groove), (zm + 0.0005, -groove), (zm + 0.0012, 0)]
    profile += [(z1 - bevel, 0), (z1, -bevel)]
    for z, k in profile:
        rings.append([Vector(((rx + k) * math.cos(2 * math.pi * j / seg),
                              (ry + k) * math.sin(2 * math.pi * j / seg), z))
                      for j in range(seg)])
    return loft(name, rings)


def build_pommel():
    # Wheel pommel: a bevelled disc whose axis runs along Y.
    # Each face has a recessed field with a raised central boss.
    half = [(-0.0124, 0.0030), (-0.0122, 0.0058), (-0.0114, 0.0080),
            (-0.0090, 0.0092), (-0.0088, 0.0138), (-0.0106, 0.0160),
            (-0.0112, 0.0186), (-0.0096, 0.0228), (-0.0066, 0.0250)]
    profile = half + [(-y, r) for y, r in reversed(half)]
    seg = 48
    rings = [[Vector((r * math.cos(2 * math.pi * j / seg), y,
                      POMMEL_Z + r * math.sin(2 * math.pi * j / seg)))
              for j in range(seg)] for y, r in profile]
    return loft("Pommel", rings)


def build_peen():
    z_top = POMMEL_Z - 0.0245
    seg = 20
    profile = [(z_top, 0.0058), (z_top - 0.0030, 0.0058), (z_top - 0.0046, 0.0050),
               (z_top - 0.0058, 0.0034)]
    rings = [[Vector((r * math.cos(2 * math.pi * j / seg), r * math.sin(2 * math.pi * j / seg), z))
              for j in range(seg)] for z, r in profile]
    return loft("Peen", rings, tip=Vector((0, 0, z_top - 0.0064)))


# --- Assembly ---------------------------------------------------------------
def join(objs, name):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    obj.data.name = name
    return obj


def finish_shading(obj, angle_deg=30):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    limit = math.radians(angle_deg)
    for f in bm.faces:
        f.smooth = True
    for e in bm.edges:
        if not e.is_manifold or e.calc_face_angle(0) > limit:
            e.smooth = False
    bm.to_mesh(obj.data)
    bm.free()


def unwrap(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.01)
    bpy.ops.object.mode_set(mode="OBJECT")


def build_sword():
    bpy.ops.wm.read_factory_settings(use_empty=True)

    steel = make_material("M_SwordBlade_Steel", (0.78, 0.78, 0.80), 1.0, 0.16)
    fittings = make_material("M_SwordFittings_Steel", (0.48, 0.48, 0.50), 1.0, 0.32)
    leather = make_material("M_SwordGrip_Leather", (0.065, 0.028, 0.013), 0.0, 0.6)

    parts = [
        (build_blade(), steel),
        (build_guard(), fittings),
        (build_guard_block(), fittings),
        (build_grip(), leather),
        (build_ring("FerruleTop", GRIP_HALF - 0.004, GRIP_HALF + 0.006, 0.0172, 0.0142, groove=0.0006), fittings),
        (build_ring("FerruleBottom", -GRIP_HALF - 0.006, -GRIP_HALF + 0.004, 0.0172, 0.0142, groove=0.0006), fittings),
        (build_pommel(), fittings),
        (build_peen(), fittings),
    ]
    for obj, mat in parts:
        obj.data.materials.append(mat)

    sword = join([p[0] for p in parts], ASSET_NAME)
    finish_shading(sword)
    unwrap(sword)

    # Simple convex collision: blade, guard, handle (Unreal UCX_ convention).
    blade_top = BLADE_START + BLADE_LEN
    collisions = [
        hull_object(f"UCX_{ASSET_NAME}_00", [
            Vector((sx * BLADE_W0 / 2, sy * BLADE_T0 / 2, GUARD_Z + 0.01))
            for sx in (-1, 1) for sy in (-1, 1)] + [
            Vector((sx * BLADE_W1 / 2, sy * BLADE_T1 / 2, BLADE_START + 0.78 * BLADE_LEN))
            for sx in (-1, 1) for sy in (-1, 1)] + [Vector((0, 0, blade_top))]),
        hull_object(f"UCX_{ASSET_NAME}_01", [
            Vector((sx * (GUARD_HALF_SPAN + 0.002), sy * 0.016, z))
            for sx in (-1, 1) for sy in (-1, 1) for z in (GUARD_Z - 0.011, GUARD_Z + 0.020)]),
        hull_object(f"UCX_{ASSET_NAME}_02", [
            Vector((sx * 0.026, sy * 0.014, z))
            for sx in (-1, 1) for sy in (-1, 1)
            for z in (GRIP_HALF + 0.006, POMMEL_Z - 0.031)]),
    ]
    for c in collisions:
        c.parent = sword
        c.display_type = "WIRE"
        c.hide_render = True

    # Sockets for weapon trails / attachment.
    for name, z in (("SOCKET_BladeBase", GUARD_Z + 0.015), ("SOCKET_BladeTip", blade_top - 0.005)):
        empty = bpy.data.objects.new(name, None)
        empty.empty_display_size = 0.03
        empty.location = (0, 0, z)
        empty.parent = sword
        bpy.context.scene.collection.objects.link(empty)

    return sword, collisions


def export(sword, collisions):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    extras = collisions + [c for c in sword.children if c.name.startswith("SOCKET_")]

    bpy.ops.object.select_all(action="DESELECT")
    for o in [sword] + extras:
        o.select_set(True)
    bpy.context.view_layer.objects.active = sword
    bpy.ops.export_scene.fbx(
        filepath=str(OUT_DIR / f"{ASSET_NAME}.fbx"),
        use_selection=True,
        object_types={"MESH", "EMPTY"},
        apply_scale_options="FBX_SCALE_UNITS",
        mesh_smooth_type="FACE",
        add_leaf_bones=False,
        bake_anim=False,
    )

    bpy.ops.object.select_all(action="DESELECT")
    sword.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(OUT_DIR / f"{ASSET_NAME}.glb"),
        export_format="GLB",
        use_selection=True,
    )

    bpy.ops.wm.save_as_mainfile(filepath=str(OUT_DIR / f"{ASSET_NAME}.blend"), compress=True)


# --- Preview renders ---------------------------------------------------------
def setup_studio():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 900
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -0.9

    world = bpy.data.worlds.new("Studio")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nodes, links = nt.nodes, nt.links
    bg = nodes.get("Background")
    coords = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[0].color = (0.015, 0.015, 0.017, 1)
    ramp.color_ramp.elements[1].position = 0.9
    ramp.color_ramp.elements[1].color = (0.55, 0.57, 0.62, 1)
    maprange = nodes.new("ShaderNodeMapRange")
    maprange.inputs["From Min"].default_value = -1
    maprange.inputs["From Max"].default_value = 1
    links.new(coords.outputs["Generated"], sep.inputs[0])
    links.new(sep.outputs["Z"], maprange.inputs["Value"])
    links.new(maprange.outputs["Result"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.8

    floor_mat = make_material("M_Floor", (0.05, 0.05, 0.055), 0.0, 0.55)
    bpy.ops.mesh.primitive_plane_add(size=8, location=(0, 0, 0))
    bpy.context.active_object.data.materials.append(floor_mat)

    def area(name, loc, size, energy, color=(1, 1, 1)):
        data = bpy.data.lights.new(name, "AREA")
        data.size = size
        data.energy = energy
        data.color = color
        obj = bpy.data.objects.new(name, data)
        obj.location = loc
        scene.collection.objects.link(obj)
        track(obj, Vector((0, 0, 0)))

    area("Key", (-0.9, -1.2, 1.6), 1.2, 260, (1.0, 0.96, 0.9))
    area("Rim", (1.2, 1.4, 0.7), 0.8, 220, (0.85, 0.9, 1.0))
    area("Top", (0.0, 0.0, 2.2), 2.5, 120)


def track(obj, target):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render_view(path, cam_loc, target, lens):
    scene = bpy.context.scene
    cam = scene.camera
    if cam is None:
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        scene.collection.objects.link(cam)
        scene.camera = cam
    cam.location = cam_loc
    cam.data.lens = lens
    track(cam, Vector(target))
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print("Rendered", path)


def render_previews(sword):
    renders = OUT_DIR / "Renders"
    renders.mkdir(parents=True, exist_ok=True)
    for c in sword.children:
        c.hide_render = True
    # Lay the sword flat on the floor, diagonally across the frame.
    sword.rotation_euler = (math.pi / 2, 0, math.radians(-28))
    sword.location = (0.0, 0.0, 0.0145)
    bpy.context.view_layer.update()
    setup_studio()

    blade_mid = sword.matrix_world @ Vector((0, 0, 0.40))
    hilt = sword.matrix_world @ Vector((0, 0, -0.01))
    render_view(renders / "SteelSword_Hero.png", Vector((0.15, -1.05, 0.95)), blade_mid - Vector((0.08, 0, 0)), 42)
    render_view(renders / "SteelSword_Hilt.png", hilt + Vector((0.12, -0.32, 0.22)), hilt + Vector((0.0, 0.0, 0.0)), 70)
    guard = sword.matrix_world @ Vector((0, 0, 0.12))
    render_view(renders / "SteelSword_Guard.png", guard + Vector((-0.20, -0.22, 0.20)), guard, 70)


if __name__ == "__main__":
    sword_obj, ucx = build_sword()
    export(sword_obj, ucx)
    me = sword_obj.data
    print(f"{ASSET_NAME}: {len(me.vertices)} verts, {len(me.polygons)} faces, "
          f"dimensions {tuple(round(d, 3) for d in sword_obj.dimensions)} m")
    if "--render" in sys.argv:
        render_previews(sword_obj)
