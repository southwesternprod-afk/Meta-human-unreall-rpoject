"""Build a steel bearded battle axe procedurally in Blender and export it for Unreal.

Same pipeline and conventions as build_steel_sword.py:

    python3 Tools/Blender/build_steel_axe.py              # build, bake textures, export
    python3 Tools/Blender/build_steel_axe.py --render     # also render previews
    python3 Tools/Blender/build_steel_axe.py --no-bake    # reuse existing textures

Outputs go to Art/Weapons/SteelAxe/. Metres in Blender, Z up, the haft runs
along +Z, the blade points +X, the cheeks face +/-Y, and the origin is the
centre of the leather grip (where the hand holds it).
"""

import math
import sys
from pathlib import Path

import bpy  # must come before bmesh when running as a module
import bmesh
import numpy as np
from mathutils import Vector

import build_steel_sword as kit
import weapon_textures as wt
from build_steel_sword import lerp, loft, set_point_attr

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "Art" / "Weapons" / "SteelAxe"
TEX_DIR = OUT_DIR / "Textures"
INLAY = OUT_DIR / "Source" / "T_SteelAxe_InlayMask.png"
ASSET_NAME = "SM_SteelAxe"
TEX_NAME = "SteelAxe"

# --- Dimensions (metres) ---------------------------------------------------
GRIP_Z0, GRIP_Z1 = -0.090, 0.085     # leather wrap
BUTT_Z = -0.130                      # bottom of the butt cap
EYE_Z0, EYE_Z1 = 0.556, 0.660        # socket the haft passes through
EYE_X = -0.004                       # eye centre (slightly behind the haft axis)
HAFT_TOP = 0.6625                    # haft end grain shows just above the eye
EDGE_BEVEL_U = 0.80                  # where the ground cutting bevel starts

# Part IDs for the texture bake.
HEAD, FITTINGS, LEATHER, WOOD = 0.0, 1.0, 2.0, 3.0


# --- Head ----------------------------------------------------------------------
def bezier(p0, c, p2, u):
    return tuple((1 - u) ** 2 * a + 2 * u * (1 - u) * b + u ** 2 * d for a, b, d in zip(p0, c, p2))


def head_outline(u, v):
    """Side-view (X, Z) position on the blade. u: eye (0) -> edge (1);
    v: top (0) -> bottom of the beard (1)."""
    top = (lerp(0.012, 0.168, u), 0.650 + 0.022 * u ** 1.6)
    bottom = bezier((0.012, 0.570), (0.075, 0.594), (0.136, 0.462), u)
    x = lerp(top[0], bottom[0], v) + 0.010 * math.sin(math.pi * v) * u ** 3   # convex edge
    z = lerp(top[1], bottom[1], v)
    return x, z


HEAD_THICKNESS = [(0.0, 0.020), (0.15, 0.013), (0.5, 0.0085), (EDGE_BEVEL_U, 0.0055), (1.0, 0.0006)]
HEAD_VS = [0.0, 0.012, 0.04, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.96, 0.988, 1.0]
HEAD_V_ROUND = {0.0: 0.55, 0.012: 0.85, 0.988: 0.85, 1.0: 0.55}   # chamfered spine and beard


def build_head_blade():
    us = sorted(set([i / 30 * EDGE_BEVEL_U for i in range(31)] +
                    [EDGE_BEVEL_U + (1 - EDGE_BEVEL_U) * i / 8 for i in range(9)]))
    tu, tt = zip(*HEAD_THICKNESS)
    rings, edgeu = [], []
    for u in us:
        half_t = float(np.interp(u, tu, tt)) / 2
        front, back = [], []
        for v in HEAD_VS:
            x, z = head_outline(u, v)
            y = half_t * HEAD_V_ROUND.get(v, 1.0) * (1 - 0.12 * (2 * v - 1) ** 2)
            front.append(Vector((x, y, z)))
            back.append(Vector((x, -y, z)))
        rings.append(front + list(reversed(back)))
        edgeu += [u] * (2 * len(HEAD_VS))
    blade = loft("HeadBlade", rings, sharp_rings=[us.index(EDGE_BEVEL_U)])
    set_point_attr(blade, "edgeu", edgeu)
    return blade


def superellipse(rx, ry, n=3.2, seg=40, cx=0.0):
    pts = []
    for j in range(seg):
        a = 2 * math.pi * j / seg
        c, s = math.cos(a), math.sin(a)
        pts.append((cx + rx * math.copysign(abs(c) ** (2 / n), c),
                    ry * math.copysign(abs(s) ** (2 / n), s)))
    return pts


def build_eye():
    """Socket around the haft with flared rims, and the flat poll behind it."""
    profile = [(EYE_Z0, 0.94), (EYE_Z0 + 0.003, 1.05), (EYE_Z0 + 0.010, 1.0),
               (EYE_Z1 - 0.010, 1.0), (EYE_Z1 - 0.003, 1.05), (EYE_Z1, 0.94)]
    rings = [[Vector((EYE_X + (x - EYE_X) * k, y * k, z))
              for x, y in superellipse(0.027, 0.0175, cx=EYE_X)] for z, k in profile]
    return loft("Eye", rings)


def chamfer_box(name, x_in, x_out, hy, hz, zc, c):
    """Box lofted along X from x_in (buried) to x_out, with chamfered long
    edges and a bevelled outer end."""
    sect = [(hy, hz - c), (hy - c, hz), (-hy + c, hz), (-hy, hz - c),
            (-hy, -hz + c), (-hy + c, -hz), (hy - c, -hz), (hy, -hz + c)]
    step = math.copysign(c, x_out - x_in)
    rings = []
    for x, k in ((x_in, 1.0), (x_out - step, 1.0), (x_out, 0.75)):
        rings.append([Vector((x, y * k, zc + z * k)) for y, z in sect])
    return loft(name, rings)


def build_poll():
    return chamfer_box("Poll", EYE_X - 0.020, EYE_X - 0.034, 0.0135, 0.026, 0.608, 0.003)


# --- Haft and fittings -----------------------------------------------------------
HAFT_PROFILE = [(BUTT_Z + 0.006, 1.00), (GRIP_Z0, 0.97), (0.10, 0.95),
                (0.45, 0.98), (0.54, 1.04), (HAFT_TOP, 1.0)]


def haft_scale(z):
    zs, ks = zip(*HAFT_PROFILE)
    return float(np.interp(z, zs, ks))


def build_haft():
    seg = 28
    z0, z1 = HAFT_PROFILE[0][0], HAFT_TOP
    rings = []
    for i in range(81):
        z = lerp(z0, z1, i / 80)
        k = haft_scale(z)
        rings.append([Vector((0.0165 * k * math.cos(2 * math.pi * j / seg),
                              0.0125 * k * math.sin(2 * math.pi * j / seg), z)) for j in range(seg)])
    return loft("Haft", rings)


def build_wrap():
    rings = []
    seg, steps, pitch = 32, 110, 0.014
    for i in range(steps + 1):
        z = lerp(GRIP_Z0, GRIP_Z1, i / steps)
        k = haft_scale(z)
        ring = []
        for j in range(seg):
            a = 2 * math.pi * j / seg
            wrap = max(0.0, math.cos(2 * math.pi * z / pitch + a)) ** 10
            r = k * (1 - 0.05 * wrap)
            ring.append(Vector((0.0185 * r * math.cos(a), 0.0145 * r * math.sin(a), z)))
        rings.append(ring)
    return loft("Wrap", rings)


def build_butt_cap():
    seg = 28
    profile = [(GRIP_Z0 - 0.004, 1.00), (BUTT_Z + 0.020, 1.02), (BUTT_Z + 0.010, 1.12),
               (BUTT_Z + 0.004, 1.13), (BUTT_Z, 1.02)]
    rings = [[Vector((0.0180 * k * math.cos(2 * math.pi * j / seg),
                      0.0140 * k * math.sin(2 * math.pi * j / seg), z)) for j in range(seg)]
             for z, k in profile]
    return loft("ButtCap", rings, tip=Vector((0, 0, BUTT_Z - 0.004)))


def build_langet(side):
    """Steel strip riveted down each cheek-side of the haft below the head."""
    z_top, z_bot = EYE_Z0 + 0.004, 0.425
    rings = []
    for i in range(13):
        u = i / 12
        z = lerp(z_top, z_bot, u)
        y0 = side * (0.0125 * haft_scale(z) - 0.0008)
        hw = lerp(0.0055, 0.0030, u)
        th = 0.0018
        y1 = y0 + side * th
        rings.append([Vector((x, y, z)) for x, y in
                      ((hw, y0), (hw * 0.8, y1), (-hw * 0.8, y1), (-hw, y0))])
    tip = Vector((0, side * (0.0125 * haft_scale(z_bot) + 0.0005), z_bot - 0.012))
    return loft(f"Langet{'F' if side > 0 else 'B'}", rings, tip=tip)


def build_rivet(side, z):
    y0 = side * (0.0125 * haft_scale(z) + 0.0008)
    seg = 12
    rings = []
    for dy, r in ((0.0, 0.0030), (0.0006, 0.0029), (0.0012, 0.0023)):
        rings.append([Vector((r * math.cos(2 * math.pi * j / seg), y0 + side * dy,
                              z + r * math.sin(2 * math.pi * j / seg))) for j in range(seg)])
    return loft(f"Rivet{side}_{z:.3f}", rings, tip=Vector((0, y0 + side * 0.0017, z)))


def build_wedge():
    rings = []
    hy = 0.0012
    for x, k in ((-0.012, 0.8), (-0.011, 1.0), (0.011, 1.0), (0.012, 0.8)):
        rings.append([Vector((x, y * k, z)) for y, z in
                      ((hy, HAFT_TOP - 0.004), (hy, HAFT_TOP + 0.0008),
                       (-hy, HAFT_TOP + 0.0008), (-hy, HAFT_TOP - 0.004))])
    return loft("Wedge", rings)


# --- UVs -----------------------------------------------------------------------
def head_and_haft_uvs(obj):
    """Planar-project the blade cheeks (one island per side) and cut the long
    haft into lengthwise bands so everything packs densely into one square."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    uv = bm.loops.layers.uv.active
    part = bm.verts.layers.float["part"]
    edgeu = bm.verts.layers.float["edgeu"]
    for f in bm.faces:
        p = f.verts[0][part]
        if abs(p - HEAD) < 0.5 and abs(f.normal.y) > 0.5 and any(v[edgeu] > 0.0 for v in f.verts):
            side = 1 if f.normal.y >= 0 else -1
            for loop in f.loops:
                co = loop.vert.co
                loop[uv].uv = (side * co.x + (0.0 if side > 0 else 0.5), co.z)
        elif abs(p - WOOD) < 0.5:
            band = min(2, max(0, int((f.calc_center_median().z - BUTT_Z) // 0.27)))
            for loop in f.loops:
                loop[uv].uv = (loop[uv].uv.x + 3.0 * (band + 1), loop[uv].uv.y)
    bm.to_mesh(obj.data)
    bm.free()


# --- Assembly ------------------------------------------------------------------
def build_axe():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    parts = [
        (build_head_blade(), HEAD),
        (build_eye(), HEAD),
        (build_poll(), HEAD),
        (build_haft(), WOOD),
        (build_wrap(), LEATHER),
        (kit.build_ring("CollarTop", GRIP_Z1 - 0.002, GRIP_Z1 + 0.008, 0.0192, 0.0152, groove=0.0006), FITTINGS),
        (kit.build_ring("CollarBottom", GRIP_Z0 - 0.008, GRIP_Z0 + 0.002, 0.0192, 0.0152, groove=0.0006), FITTINGS),
        (build_butt_cap(), FITTINGS),
        (build_langet(1), FITTINGS),
        (build_langet(-1), FITTINGS),
        (build_wedge(), FITTINGS),
    ] + [(build_rivet(side, z), FITTINGS) for side in (1, -1) for z in (0.535, 0.495, 0.455)]

    for obj, part_id in parts:
        n = len(obj.data.vertices)
        set_point_attr(obj, "part", [part_id] * n)
        if "edgeu" not in obj.data.attributes:
            set_point_attr(obj, "edgeu", [0.0] * n)

    axe = kit.join([p[0] for p in parts], ASSET_NAME)
    kit.finish_shading(axe)
    kit.bevel_and_normals(axe)
    kit.unwrap(axe, planar=head_and_haft_uvs)

    collisions = [
        kit.hull_object(f"UCX_{ASSET_NAME}_00", [
            Vector((x, sy * y, z)) for sy in (-1, 1) for x, y, z in (
                (EYE_X - 0.036, 0.018, EYE_Z0), (EYE_X - 0.036, 0.018, EYE_Z1),
                (0.030, 0.018, EYE_Z0), (0.030, 0.018, EYE_Z1),
                (0.170, 0.004, 0.674), (0.182, 0.004, 0.570), (0.137, 0.004, 0.460))]),
        kit.hull_object(f"UCX_{ASSET_NAME}_01", [
            Vector((sx * 0.021, sy * 0.017, z))
            for sx in (-1, 1) for sy in (-1, 1) for z in (BUTT_Z - 0.004, EYE_Z0)]),
    ]
    for c in collisions:
        c.parent = axe
        c.display_type = "WIRE"
        c.hide_render = True

    # Same socket names as the sword, so weapon trail / trace code can be shared.
    for name, loc in (("SOCKET_BladeBase", (0.030, 0, 0.605)), ("SOCKET_BladeTip", (0.176, 0, 0.570))):
        empty = bpy.data.objects.new(name, None)
        empty.empty_display_size = 0.03
        empty.location = loc
        empty.parent = axe
        bpy.context.scene.collection.objects.link(empty)
    return axe, collisions


# --- Texture recipe --------------------------------------------------------------
def axe_layers(c):
    """part 0 = forged head with a ground edge, 1 = steel fittings,
    2 = leather wrap, 3 = oiled ash haft."""
    g = c.g
    rgb = wt.rgb
    is_head = c.is_part(HEAD)
    scratch = c.scratches()
    pits = c.pits()

    # Forged head: hammer facets, dark mill scale, bright ground bevel, gold inlay.
    hammer = g.remap(g.voronoi(c.P, 90.0), 0.0, 0.55)
    tone = g.remap(g.noise(c.P, 40.0, 4.0), 0.4, 0.65)
    scale_patch = g.remap(g.noise(c.P, 9.0, 3.0), 0.55, 0.7)
    polish = g.math("MULTIPLY", g.remap(c.edgeu, EDGE_BEVEL_U - 0.01, EDGE_BEVEL_U + 0.01), is_head)
    grind = c.streaks((6, 900, 900))
    inlay = c.projected_mask(INLAY, "X", (0.031, 0.045), "Z", (0.583, 0.643), "X",
                             g.math("MULTIPLY", is_head, g.math("GREATER_THAN", c.edgeu, 0.0)))

    base = g.mix_color(rgb(0.13, 0.13, 0.14), rgb(0.21, 0.21, 0.22), tone)
    base = g.mix_color(base, rgb(0.10, 0.10, 0.11), g.math("MULTIPLY", hammer, 0.4))
    base = g.mix_color(base, rgb(0.04, 0.04, 0.045),
                       g.math("ADD", g.math("MULTIPLY", c.cavity, 0.85),
                              g.math("MULTIPLY", scale_patch, 0.5), clamp=True))
    base = g.mix_color(base, rgb(0.55, 0.55, 0.56), c.edge)
    base = g.mix_color(base, rgb(0.45, 0.45, 0.46), g.math("MULTIPLY", scratch, 0.3))
    base = g.mix_color(base, g.mix_color(rgb(0.62, 0.62, 0.63), rgb(0.77, 0.77, 0.78), grind), polish)
    base = g.mix_color(base, rgb(0.92, 0.70, 0.36), inlay)

    rough = g.math("ADD", 0.50, g.math("MULTIPLY", tone, 0.08))
    rough = g.mix_float(rough, 0.66, g.math("MULTIPLY", scale_patch, 0.6))
    rough = g.mix_float(rough, 0.26, c.edge)
    rough = g.mix_float(rough, 0.45, g.math("MULTIPLY", scratch, 0.5))
    rough = g.mix_float(rough, 0.72, c.cavity)
    rough = g.mix_float(rough, g.math("ADD", 0.12, g.math("MULTIPLY", grind, 0.08)), polish)
    rough = g.mix_float(rough, 0.30, inlay)

    unpolished = g.math("SUBTRACT", 1.0, polish)
    height = g.math("MULTIPLY", g.math("SUBTRACT", g.math("MULTIPLY", hammer, 0.7),
                                       g.math("MULTIPLY", pits, 0.6)), unpolished)
    height = g.math("SUBTRACT", height, g.math("ADD", g.math("MULTIPLY", scratch, 0.5), inlay))
    height = g.math("ADD", height, g.math("MULTIPLY", grind, g.math("MULTIPLY", polish, 0.05)))
    head = wt.Surface(base, rough, wt.metal_with_grime(c), height, 0.0001)

    # Oiled ash: long grain along Z, darker grain lines and pores, dents, grime.
    grain = g.remap(g.noise(g.mapping(c.P, (250, 250, 4)), 1.0, 6.0, 0.55), 0.3, 0.7)
    lines = g.remap(g.noise(g.mapping(c.P, (520, 520, 6)), 1.0, 3.0, 0.5), 0.5, 0.56)
    pores = g.remap(g.noise(g.mapping(c.P, (2600, 2600, 260)), 1.0, 2.0, 0.5), 0.62, 0.74)
    wbase = g.mix_color(rgb(0.10, 0.052, 0.022), rgb(0.19, 0.10, 0.045), grain)
    wbase = g.mix_color(wbase, rgb(0.055, 0.028, 0.012), g.math("MULTIPLY", lines, 0.7))
    wbase = g.mix_color(wbase, rgb(0.04, 0.02, 0.009), g.math("MULTIPLY", pores, 0.6))
    wbase = g.mix_color(wbase, rgb(0.26, 0.16, 0.08), g.math("MULTIPLY", scratch, 0.3))
    wbase = g.mix_color(wbase, rgb(0.05, 0.03, 0.015), g.math("MULTIPLY", c.cavity, 0.85))
    wrough = g.math("ADD", 0.52, g.math("MULTIPLY", grain, 0.10))
    wrough = g.mix_float(wrough, 0.75, pores)
    wrough = g.mix_float(wrough, 0.70, g.math("MULTIPLY", scratch, 0.7))
    wrough = g.mix_float(wrough, 0.82, c.cavity)
    wheight = g.math("SUBTRACT", g.math("ADD", g.math("MULTIPLY", grain, 0.3),
                                        g.math("MULTIPLY", lines, 0.4)),
                     g.math("ADD", g.math("MULTIPLY", pores, 0.6), g.math("MULTIPLY", scratch, 0.6)))
    wood = wt.Surface(wbase, wrough, 0.0, wheight, 0.00012)

    return [(None, wt.fittings_surface(c)),
            (is_head, head),
            (c.is_part(LEATHER), wt.leather_surface(c)),
            (c.is_part(WOOD), wood)]


# --- Preview renders -------------------------------------------------------------
def render_previews(axe):
    renders = OUT_DIR / "Renders"
    renders.mkdir(parents=True, exist_ok=True)
    for c in axe.children:
        c.hide_render = True
    axe.rotation_euler = (math.pi / 2, 0, math.radians(-28))
    axe.location = (0.0, 0.0, 0.0185)
    bpy.context.view_layer.update()
    kit.setup_studio()

    def at(x, z):
        return axe.matrix_world @ Vector((x, 0, z))

    mid = at(0.07, 0.30)
    kit.render_view(renders / "SteelAxe_Hero.png", mid + Vector((0.10, -0.85, 0.80)), mid, 40)
    head = at(0.07, 0.575)
    kit.render_view(renders / "SteelAxe_Head.png", head + Vector((0.12, -0.30, 0.34)), head, 55)
    grip = at(0.0, 0.0)
    kit.render_view(renders / "SteelAxe_Grip.png", grip + Vector((0.14, -0.30, 0.24)), grip, 60)
    langets = at(0.0, 0.50)
    kit.render_view(renders / "SteelAxe_Langets.png", langets + Vector((-0.16, -0.14, 0.18)), langets, 70)


if __name__ == "__main__":
    axe_obj, ucx = build_axe()
    wt.texture_weapon(axe_obj, TEX_DIR, TEX_NAME, axe_layers, bake="--no-bake" not in sys.argv)
    kit.export(axe_obj, ucx, OUT_DIR, ASSET_NAME)
    me = axe_obj.data
    print(f"{ASSET_NAME}: {len(me.vertices)} verts, {len(me.polygons)} faces, "
          f"dimensions {tuple(round(d, 3) for d in axe_obj.dimensions)} m")
    if "--render" in sys.argv:
        render_previews(axe_obj)
