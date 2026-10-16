"""Build the leather belt that goes over the leather jerkin, as its own wearable item.

    python3 Tools/Blender/build_leather_belt.py [--export] [--render] [--res 2048]

A separate skeletal mesh on the MetaHuman skeleton (equip / remove it in game like
the jerkin). It is fitted over the jerkin's waist welt and passes under its belt
loops. Pieces (one mesh, one texture set):
  - 36 mm belt (3.2 mm leather) with stitched, burnished edges, cinched onto the jerkin
  - forged buckle with a leather chape, the tongue through a keeper, its end
    tied over the belt (medieval knot) and hanging down the front, weighted by a
    forged strap end with an engraved border and a knop
  - five leather frogs with iron D-rings: the attachment slots for the separate
    pouches and sheaths (positions written to BeltSlots.json)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy  # must come before bmesh/mathutils when running as a module
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

import build_leather_jerkin as bj
import clothing_kit as ck
from build_leather_jerkin import (LEATHER, STEEL, STRAP, WAIST_Z, FRONT, BELT_W, BELT_T, BELT_Z, BELT_KS,
                                  Piece, aaa, L, outermost)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "Art" / "Characters" / "Clothing" / "LeatherBelt"
ASSET_NAME = "SKM_LeatherBelt"
TEX = "LeatherBelt"

TONGUE = 3.0              # tone id: the punched tongue
STRAP_END = 4.0           # tone id: the iron strap end (engraved)
BUCKLE_DEG = -12.0        # around from the centre front (+ = character's left)
KEEPER_DEG = -1.0
TAIL_DEG = 11.5           # where the end loops over the belt and hangs down (a little tongue shows)
TAIL_LEN = 0.11           # how far it hangs below the belt, strap end included
TONGUE_LIFT, TONGUE_T = 0.0029, 0.0026
TAIL_T = 0.0028
END_W = BELT_W * 0.9 + 0.0024     # strap end: a forged sheath over the leather tip
END_LEN = 0.036                   # straight part + ogee taper (the knop hangs below)
END_T = TAIL_T + 0.002            # a 1 mm plate either side of the leather
# attachment slots for the separate pouches / sheaths (degrees from the centre front)
SLOTS = {"PouchR": -45.0, "PouchL": 45.0, "SheathR": -95.0, "SheathL": 95.0, "Back": 180.0}
LEATHER_BELT = "#3a2a1f"
BELT_C = (0.0, 0.0)       # belt axis (x, y), set by build_all; the fittings' wear faces away from it
STITCH = "#7d6a52"


# --- helpers ------------------------------------------------------------------------------
def sweep_tube(points, radius, closed, up, sides=8):
    """Round wire swept along `points` (D-rings, rings)."""
    bm = bmesh.new()
    m = len(points)
    rings = []
    for i, q in enumerate(points):
        t = (points[(i + 1) % m] if closed or i < m - 1 else q) - (points[i - 1] if closed or i > 0 else q)
        t = t.normalized()
        a = (up - t * up.dot(t)).normalized()
        b = t.cross(a)
        rings.append([bm.verts.new(q + (a * math.cos(2 * math.pi * k / sides) +
                                        b * math.sin(2 * math.pi * k / sides)) * radius) for k in range(sides)])
    pairs = list(zip(rings, rings[1:])) + ([(rings[-1], rings[0])] if closed else [])
    for r0, r1 in pairs:
        for k in range(sides):
            bm.faces.new((r0[k], r0[(k + 1) % sides], r1[(k + 1) % sides], r1[k]))
    bj.orient_outward(bm)
    return bm


class BeltFrame:
    """The belt band's outer surface as a function of (angle, across)."""

    def __init__(self, c, angles, radius):
        self.c, self.angles, self.r = c, angles, radius

    def dirn(self, deg):
        a = FRONT + math.radians(deg)
        return Vector((math.cos(a), math.sin(a), 0.0))

    def radius(self, deg, k=0.0):
        th = math.radians(deg) % (2 * math.pi)
        x = np.append(self.angles, 2 * math.pi)
        cols = [float(np.interp(th, x, np.append(self.r[:, c], self.r[0, c]))) for c in range(len(BELT_KS))]
        return float(np.interp(k, BELT_KS, cols))

    def point(self, deg, k=0.0, lift=0.0):
        """Point on the belt's outer face; `k` -1..1 from the lower to the upper edge."""
        d = self.dirn(deg)
        return Vector((self.c[0], self.c[1], BELT_Z + k * BELT_W / 2)) + d * (self.radius(deg, k) + lift)

    def normal(self, deg):
        d = self.dirn(deg)
        up = (self.point(deg, 1) - self.point(deg, -1)).normalized()
        side = Vector((0, 0, 1)).cross(d)
        return side.cross(up).normalized() if side.cross(up).dot(d) > 0 else -side.cross(up).normalized()

    def tangent(self, deg):
        return (self.point(deg + 0.5) - self.point(deg - 0.5)).normalized()


# --- the belt -----------------------------------------------------------------------------
def build_band(bvhs, c):
    """The belt band, cinched onto the jerkin (see bj.belt_profile, shared with the
    jerkin's belt loops so the two always fit)."""
    angles, radius = bj.belt_profile(bvhs, c)
    ks = BELT_KS
    frame = BeltFrame(c, angles, radius)

    rows = []
    for k in ks:
        rows.append([frame.point(math.degrees(th), k, 0.0004 * (1 - k * k)) for th in angles])
    bm, vr = bj.loft_grid(rows)
    bj.orient_outward(bm)
    piece = Piece("Belt", bm, LEATHER, STRAP, thickness=BELT_T)
    r = float(np.mean(radius))
    for i, row in enumerate(vr):
        for j, v in enumerate(row):
            piece.pc[v] = (angles[j] * r, ks[i] * BELT_W / 2)
    return piece, frame


def band_ribbon(frame, deg0, deg1, lift, width, name, tone, thickness, steps=24, tips=(0.0, 0.0)):
    """A strap lying on the belt's face between two angles."""
    samples = [(frame.point(d, 0.0, lift), frame.normal(d)) for d in np.linspace(deg0, deg1, steps)]
    return bj.surface_ribbon(samples, width, 0.0, name, tone, thickness=thickness, tips=tips)


def build_buckle_set(frame, bvhs_under, c):
    """Buckle, chape, punched tongue, keeper and the tucked hanging end."""
    pieces = []
    nb, tb = frame.normal(BUCKLE_DEG), frame.tangent(BUCKLE_DEG)
    pieces.append(bj.buckle(frame.point(BUCKLE_DEG, 0.0, 0.0003), nb, tb, BELT_W + 0.009, 0.034, bar=0.0042))
    # chape: the leather folded round the buckle bar, riveted
    chape, _ = band_ribbon(frame, BUCKLE_DEG - 2.0, BUCKLE_DEG - 13.0, 0.0027, BELT_W * 0.94, "Chape", STRAP,
                           0.0024, steps=16, tips=(0.0, 0.0))
    pieces.append(chape)
    for k in (-0.5, 0.5):
        q = frame.point(BUCKLE_DEG - 9.5, k, 0.0027)
        pieces.append(bj.rivet(q, frame.normal(BUCKLE_DEG - 9.5), r=0.0029))
    # tongue: through the buckle and the keeper; its end runs on under the loop below
    tongue, _ = band_ribbon(frame, BUCKLE_DEG, TAIL_DEG + 2.5, TONGUE_LIFT, BELT_W * 0.9, "Tongue", TONGUE,
                            TONGUE_T, steps=24)
    pieces.append(tongue)
    pieces.append(bj.keeper(frame.point(KEEPER_DEG, 0.0, 0.0002), frame.normal(KEEPER_DEG),
                            frame.tangent(KEEPER_DEG), BELT_W, 0.0050, "BeltKeeper"))
    return pieces + build_tail(frame, bvhs_under, c)


def build_tail(frame, bvhs_under, c):
    """The end after the keeper, tied the medieval way: tucked behind the belt,
    brought up over its top edge and down the front over the tongue, then hanging
    free over the skirt, weighted by a forged strap end. One continuous strap."""
    d, nrm, side = frame.dirn(TAIL_DEG), frame.normal(TAIL_DEG), frame.tangent(TAIL_DEG)
    up = (frame.point(TAIL_DEG, 1.0) - frame.point(TAIL_DEG, -1.0)).normalized()
    stack = TONGUE_LIFT + 0.0004 + 0.0006           # the tongue (+ its crown) and a little air
    lift = stack + TAIL_T                           # the end's outer face over the belt
    l0 = (stack - BELT_T) / 2                       # middle of the belt + tongue stack
    centre = frame.point(TAIL_DEG, 1.0) + nrm * l0
    samples = []
    for phi in np.linspace(math.radians(112), 0.0, 15, endpoint=False):   # over the top edge
        nn = nrm * math.cos(phi) + up * math.sin(phi)
        p = centre + nn * (lift - l0)
        if not samples:                             # start behind the belt only where there is room
            clear = True
            for q in (p, p - nn * TAIL_T):
                for o in (-0.5, 0.0, 0.5):
                    qo = q + side * (o * BELT_W * 0.9)
                    dq = Vector((qo.x - c[0], qo.y - c[1], 0.0))
                    clear &= dq.length > outermost(bvhs_under, Vector((c[0], c[1], qo.z)), dq.normalized()) + 0.0006
            if not clear:
                continue
        samples.append((p, nn))
    for k in np.linspace(1.0, -1.0, 9):                                     # down the front
        samples.append((frame.point(TAIL_DEG, k) + nrm * lift, nrm))
    # below the belt it hangs plumb from the lower edge, resting on the skirt only
    # where the skirt flares out further (a drape: the convex hull of both)
    bottom = samples[-1][0]
    z_end = bottom.z - TAIL_LEN
    z_sheath = z_end + 0.0052 + END_LEN             # top of the strap end (knop below END_LEN)
    zs = np.arange(bottom.z - 0.003, z_end + 0.0052 - 1e-6, -0.003)
    r0 = (bottom - Vector((c[0], c[1], bottom.z))).dot(d)
    rs = np.array([max(r0, outermost(bvhs_under, Vector((c[0], c[1], z)), d)
                       + (0.0012 if z > z_sheath + 0.002 else 0.0022) + TAIL_T) for z in zs])
    for _ in range(30):
        rs[1:-1] = np.maximum(rs[1:-1], (rs[:-2] + rs[2:]) / 2)
    hang = [Vector((c[0], c[1], z)) + d * r for z, r in zip(zs, rs)]

    def hang_normal(i):
        t = hang[min(i + 1, len(hang) - 1)] - hang[max(i - 1, 0)]
        n = t.cross(side).normalized()
        return n if n.dot(d) > 0 else -n

    leather_end = z_sheath - 0.008                  # the leather runs 8 mm into the strap end
    samples += [(p, hang_normal(i)) for i, p in enumerate(hang) if p.z >= leather_end]
    tail, _ = bj.surface_ribbon(samples, BELT_W * 0.9, 0.0, "BeltEnd", STRAP, thickness=TAIL_T)
    pieces = [tail]

    # forged strap end: a sheath with an ogee point and a knop, two rivets and an
    # engraved border (see belt_layers)
    i_top = min(range(len(hang)), key=lambda i: abs(hang[i].z - z_sheath))
    i_bot = min(range(len(hang)), key=lambda i: abs(hang[i].z - (z_sheath - END_LEN)))
    down = (hang[i_bot] - hang[i_top]).normalized()
    n_s = down.cross(side).normalized()
    n_s = n_s if n_s.dot(d) > 0 else -n_s
    across = down.cross(n_s).normalized()            # (across, down, n_s) is right-handed
    origin = hang[i_top] + n_s * 0.001

    def half_width(v):
        return END_W / 2 + (0.0026 - END_W / 2) * bj.smooth01((v - 0.020) / 0.016)

    # clear the skirt with the back plate
    push = 0.0
    for v in np.linspace(0.0, END_LEN, 7):
        for k in (-1.0, 0.0, 1.0):
            q = origin + down * v + across * (k * half_width(v)) - n_s * END_T
            dq = Vector((q.x - c[0], q.y - c[1], 0.0))
            rq = dq.length
            dq.normalize()
            need = outermost(bvhs_under, Vector((c[0], c[1], q.z)), dq) + 0.0008
            push = max(push, (need - rq) / max(dq.dot(n_s), 0.5))
    origin = origin + n_s * push
    rows, uv = [], []
    for v in np.linspace(0.0, END_LEN, 19):
        hw = half_width(v)
        rows.append([origin + down * v + across * (k * hw) + n_s * (0.0005 * (1 - k * k))
                     for k in np.linspace(-1.0, 1.0, 9)])
        uv.append([(v, k * hw) for k in np.linspace(-1.0, 1.0, 9)])
    bm, vr = bj.loft_grid(rows, closed=False)
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    if bm.faces[0].normal.dot(n_s) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    end = Piece("StrapEnd", bm, STEEL, STRAP_END, thickness=END_T, rigid=True)
    for row, uvrow in zip(vr, uv):
        for vert, p in zip(row, uvrow):
            end.pc[vert] = p
    pieces.append(end)
    knop = bmesh.new()
    bmesh.ops.create_uvsphere(knop, u_segments=12, v_segments=8, radius=1.0)
    kc = origin - n_s * (END_T / 2) + down * (END_LEN + 0.0016)
    for vert in knop.verts:
        x, y, z = vert.co
        vert.co = kc + across * (x * 0.0034) + down * (y * 0.0036) + n_s * (z * 0.0030)
    knop.normal_update()
    if sum((f.calc_center_median() - kc).dot(f.normal) for f in knop.faces) < 0:
        bmesh.ops.reverse_faces(knop, faces=knop.faces)
    pieces.append(Piece("StrapEndKnop", knop, STEEL, STRAP_END, rigid=True))
    for u in (-0.0085, 0.0085):
        pieces.append(bj.rivet(origin + down * 0.0065 + across * u + n_s * 0.0003, n_s, r=0.0021))
    return pieces


def build_slots(frame, bvhs_under, c):
    """Leather frogs wrapped over the belt, each carrying an iron D-ring: where the
    separate pouches and sheaths hang. Returns pieces and slot records."""
    pieces, slots = [], {}
    for name, deg in SLOTS.items():
        d = frame.dirn(deg)
        zs = np.linspace(BELT_Z + BELT_W / 2 - 0.001, BELT_Z - BELT_W / 2 - 0.016, 16)   # from the top edge
        side = Vector((0, 0, 1)).cross(d).normalized()

        def under(z):
            # the widest of the frog's centre and edges (it may sit on a sunken skirt seam)
            r = 0.0
            for off in (-0.0085, 0.0, 0.0085):
                o = Vector((c[0], c[1], z)) + side * off
                r = max(r, outermost(bvhs_under, o, d))
            if abs(z - BELT_Z) <= BELT_W / 2 + 0.002:
                r = max(r, frame.radius(deg, max(-1.0, min(1.0, (z - BELT_Z) / (BELT_W / 2)))))
            return r

        rs = np.array([under(z) for z in zs])
        for _ in range(20):                                 # drape over the belt's top edge
            rs[1:-1] = np.maximum(rs[1:-1], (rs[:-2] + rs[2:]) / 2)
        samples = [(Vector((c[0], c[1], z)) + d * (r + 0.0018 + 0.0024), d) for z, r in zip(zs, rs)]
        frog, _ = bj.surface_ribbon(samples, 0.017, 0.0, f"Frog_{name}", STRAP, thickness=0.0024)
        pieces.append(frog)
        pieces.append(bj.rivet(samples[8][0] + d * 0.0003, d, r=0.0026))
        # D-ring hanging from the frog's lower end, lying against the skirt
        top = samples[-1][0] + Vector((0, 0, 0.004))
        tangent = Vector((0, 0, 1)).cross(d).normalized()
        w, h, wire = 0.026, 0.022, 0.0023
        pts = [top + tangent * (w / 2 - w * i / 6) for i in range(7)]                 # straight bar
        pts += [top + tangent * (-w / 2 * math.cos(p)) - Vector((0, 0, h * math.sin(p)))
                for p in np.linspace(0, math.pi, 14)[1:-1]]                              # round belly
        # push the ring out until every point of it clears the skirt (it hangs against it)
        push = 0.0
        for q in pts:
            dq = Vector((q.x - c[0], q.y - c[1], 0.0))
            rq = dq.length
            dq.normalize()
            need = outermost(bvhs_under, Vector((c[0], c[1], q.z)), dq) + wire + 0.0012
            push = max(push, (need - rq) / max(dq.dot(d), 0.5))
        pts = [q + d * push for q in pts]
        ring = Piece(f"DRing_{name}", sweep_tube(pts, wire, True, d), STEEL, rigid=True)
        pieces.append(ring)
        hook = top + d * push - Vector((0, 0, h))                                        # bottom of the D
        slots[name] = {"degrees_from_front": deg,
                       "hook_m": [round(x, 4) for x in hook],
                       "outward": [round(x, 4) for x in d],
                       "along_belt": [round(x, 4) for x in tangent]}
    return pieces, slots


def build_all(body):
    torso = bj.build_torso(body, high=True)           # the jerkin is placed on this...
    skirt, _ = bj.build_skirt(body, torso)
    welt = bj.build_waist_band(torso, skirt.bm)
    game_torso = bj.build_torso(body, high=False)     # ...but the belt sits on the game mesh
    c = np.mean([v.co[:2] for v in torso.verts if abs(v.co.z - WAIST_Z) < 0.02], axis=0)
    global BELT_C
    BELT_C = (float(c[0]), float(c[1]))
    placket = bj.build_front_details(torso)[0]        # the centre placket runs down under the belt
    under = [BVHTree.FromBMesh(b) for b in (game_torso, skirt.bm, welt.bm, placket.bm)]
    band, frame = build_band(under, c)
    pieces = [band] + build_buckle_set(frame, under, c)
    slot_pieces, slots = build_slots(frame, under, c)
    return pieces + slot_pieces, slots


# --- texture recipe ------------------------------------------------------------------------
def belt_layers(c):
    """Belt leather (punched holes on the tongue), forged iron fittings."""
    g = c.g
    srgb = aaa.engine.srgb
    tone, part, ed = c.attr("tone"), c.attr("part"), c.attr("edgedist")
    pc = g.xyz(c.attr_vec("pc"))

    def near(x, target, width):
        return g.remap(g.math("ABSOLUTE", g.sub(x, target)), 0.0, width, 1.0, 0.0)

    stitch_dots = g.remap(g.voronoi(g.mapping(c.P, (1 / 0.0042,) * 3), 1.0), 0.16, 0.30, 1.0, 0.0)
    stitch = g.mul(near(ed, 0.0032, 0.0007), stitch_dots)
    burnish = g.remap(ed, 0.0, 0.0025, 1.0, 0.0)
    # punched holes every 25 mm along the tongue's centre line
    is_tongue = c.is_value(tone, TONGUE)
    along = g.math("FRACT", g.math("DIVIDE", pc["X"], 0.025))
    hole_d = g.math("SQRT", g.add(g.math("POWER", g.mul(g.sub(along, 0.5), 0.025), 2.0),
                                  g.math("POWER", pc["Y"], 2.0)))
    holes = g.mul(g.mul(g.remap(hole_d, 0.0017, 0.0022, 1.0, 0.0), is_tongue),
                  g.math("LESS_THAN", pc["X"], 0.075))
    hide_v = g.node("ShaderNodeTexWhiteNoise", {"W": c.attr("hide")}, noise_dimensions="1D").outputs["Value"]
    hide_k = g.add(0.90, g.mul(hide_v, 0.2))

    cl = c.with_masks(edge=g.maxf(g.remap(ed, 0.0, 0.005, 1.0, 0.0), g.mul(c.edge, 0.3)))
    s = L.leather(cl, color=LEATHER_BELT, wear=0.6, dirt=0.5, grain_amount=0.22, scale=1.6)
    col = srgb(LEATHER_BELT)
    dark = tuple(v * 0.4 for v in col[:3]) + (1.0,)
    base = g.tint(s.base, g.node("ShaderNodeCombineColor", {"Red": hide_k, "Green": hide_k, "Blue": hide_k}).outputs[0])
    base = g.mix_color(base, dark, g.mul(burnish, 0.6))
    base = g.mix_color(base, srgb(STITCH), stitch)
    base = g.mix_color(base, srgb("#120d0a"), holes)
    rough = g.mix_float(g.mix_float(s.rough, 0.36, g.mul(burnish, 0.7)), 0.82, stitch)
    height = g.sub(g.add(s.height, g.mul(stitch, 0.7)), g.add(g.mul(holes, 3.0), g.mul(near(ed, 0.0012, 0.0012), 0.8)))
    leather = aaa.engine.Surface(base, rough, 0.0, height, s.bump)

    # fittings wear where the leather and the hands rub them: the outward crown of
    # the wire (buckle frame, prong, D-rings, knop) and a thin line round plate rims;
    # everywhere else keeps its black oxide
    xyz = g.xyz(c.P)
    radial = g.node("ShaderNodeCombineXYZ", {"X": g.sub(xyz["X"], BELT_C[0]), "Y": g.sub(xyz["Y"], BELT_C[1]),
                                             "Z": 0.0}).outputs[0]
    radial = g.node("ShaderNodeVectorMath", {0: radial}, operation="NORMALIZE").outputs["Vector"]
    facing = g.node("ShaderNodeVectorMath", {0: c.N, 1: radial}, operation="DOT_PRODUCT").outputs["Value"]
    crown = g.mul(g.remap(facing, 0.8, 0.98), g.add(0.55, g.mul(aaa.weathering.breakup(c, 30.0, 0.35, 0.65), 0.45)))
    has_rim = g.math("LESS_THAN", ed, 0.049)          # plates (strap end) have rims; wire does not
    rim = g.mul(g.mul(g.remap(ed, 0.0, 0.0016, 1.0, 0.0), g.remap(facing, 0.3, 0.8)),     # front edge, not the sides
                g.add(0.7, g.mul(aaa.weathering.breakup(c, 40.0, 0.35, 0.6), 0.3)))
    edge = g.mix_float(g.maxf(g.mul(crown, 0.85), g.mul(c.edge, 0.25)), rim, has_rim)   # plates: rims only
    iron = L.steel_blackened(c.with_masks(edge=edge), wear=0.5, dirt=0.45, rust=0.3, rub=0.0)
    # strap end: an engraved border and a cross line, black with packed grime
    is_end = c.is_value(tone, STRAP_END)
    engrave = g.mul(is_end, g.maxf(near(ed, 0.0024, 0.0005), near(pc["X"], 0.0125, 0.0006)))
    iron = aaa.engine.Surface(g.mix_color(iron.base, srgb("#1c1b1a"), engrave), g.mix_float(iron.rough, 0.7, engrave),
                              g.mul(iron.metal, g.inv(engrave)), g.sub(iron.height, g.mul(engrave, 0.9)), iron.bump)
    return [(None, leather), (c.is_value(part, STEEL), iron)]


# --- output -------------------------------------------------------------------------------
def write_import_script():
    """The jerkin's one-click Unreal import, retargeted to the belt."""
    src = (bj.OUT_DIR / "Import_LeatherJerkin_Unreal.py").read_text()
    text = (src.replace("leather jerkin top", "leather belt")
               .replace("Leather jerkin imported", "Leather belt imported")
               .replace("LeatherJerkin", TEX)
               .replace("The top is imported onto", "The belt is imported onto"))
    (OUT_DIR / f"Import_{TEX}_Unreal.py").write_text(text)


def render_previews(body_fbx):
    """Belt worn over the jerkin, loaded the way Unreal would load them."""
    import build_steel_sword as kit
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(body_fbx))
    for o in list(bpy.context.scene.objects):
        if o.type == "MESH" and not o.name.endswith("LOD0"):
            bpy.data.objects.remove(o)
    body_mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH")
    for fbx, tex_dir, name in ((bj.OUT_DIR / f"{bj.ASSET_NAME}.fbx", bj.OUT_DIR / "Textures", "LeatherJerkin"),
                               (OUT_DIR / f"{ASSET_NAME}.fbx", OUT_DIR / "Textures", TEX)):
        before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.fbx(filepath=str(fbx))
        mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH" and o not in before)
        aaa.engine.assign_baked_material(mesh, tex_dir, name)
    clay = kit.make_material("M_Clay", (0.50, 0.45, 0.42), 0.0, 0.55)
    body_mesh.data.materials.clear()
    body_mesh.data.materials.append(clay)
    kit.setup_studio()
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = 1400, 1100
    scene.view_settings.exposure = -0.9
    for light, loc, energy in (("Key", (-1.6, -2.4, 2.6), 380), ("Rim", (1.8, 2.2, 2.0), 300),
                               ("Top", (0.0, 0.0, 3.4), 150)):
        obj = bpy.data.objects[light]
        obj.location, obj.data.energy, obj.data.size = loc, energy, 2.0
        kit.track(obj, Vector((0, 0, 1.0)))
    renders = OUT_DIR / "Renders"
    renders.mkdir(parents=True, exist_ok=True)
    z = BELT_Z
    kit.render_view(renders / "LeatherBelt_Front.png", Vector((0.55, -1.55, z + 0.2)), (0.0, 0.0, z - 0.03), 50)
    kit.render_view(renders / "LeatherBelt_Buckle.png", Vector((0.1, -0.6, z + 0.06)), (0.0, -0.12, z - 0.02), 70)
    kit.render_view(renders / "LeatherBelt_Side.png", Vector((1.55, -0.55, z + 0.15)), (0.0, 0.0, z - 0.03), 50)
    kit.render_view(renders / "LeatherBelt_Back.png", Vector((-0.6, 1.55, z + 0.2)), (0.0, 0.0, z - 0.03), 50)
    kit.render_view(renders / "LeatherBelt_Low.png", Vector((0.7, -0.9, z - 0.45)), (0.05, 0.0, z), 55)
    kit.render_view(renders / "LeatherBelt_High.png", Vector((0.55, -0.75, z + 0.6)), (0.05, 0.0, z), 55)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default=str(REPO_ROOT / "Art/Characters/Source/SKM_John_doe_BodyMesh.fbx"))
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--render", action="store_true")
    args, _ = ap.parse_known_args(sys.argv[1:])
    body = ck.Body(args.body)
    pieces, slots = build_all(body)
    obj = bj.pieces_to_object(pieces, ASSET_NAME)
    tris = sum(len(f.vertices) - 2 for f in obj.data.polygons)
    print(f"{ASSET_NAME}: {len(obj.data.vertices)} verts, {tris} tris")
    if args.export:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "BeltSlots.json").write_text(json.dumps({
            "units": "metres, Blender world space (Z up, character faces -Y); Unreal: x100 cm, Y flipped",
            "skeleton": "MetaHuman body (same as SKM_John_doe_BodyMesh)",
            "slots": slots}, indent=2))
        bj.skin(body, obj)
        bj.unwrap(obj, game=True, res=args.res)
        tex_dir = OUT_DIR / "Textures"
        aaa.engine.bake(obj, tex_dir, TEX, belt_layers, res=args.res, ao_distance=0.02, export_height=False)
        aaa.engine.assign_baked_material(obj, tex_dir, TEX)
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT_DIR / f"{ASSET_NAME}.blend"), compress=True)
        ck.export_skeletal(body, obj, OUT_DIR / f"{ASSET_NAME}.fbx")
        write_import_script()
        print("Exported", OUT_DIR / f"{ASSET_NAME}.fbx")
    if args.render:
        render_previews(args.body)
    return body, obj


if __name__ == "__main__":
    main()
