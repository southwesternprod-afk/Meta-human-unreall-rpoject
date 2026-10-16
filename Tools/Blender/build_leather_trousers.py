"""Build the leather trousers that match the leather jerkin, as their own wearable item.

    python3 Tools/Blender/build_leather_trousers.py [--export] [--render] [--res 2048]

A separate skeletal mesh on the MetaHuman skeleton (equip / remove it in game like the
jerkin and the belt), worn under the jerkin's skirt. Pieces (one mesh, one texture set):
  - dark leather trousers: straight legs that hang from the knee and calf instead of
    hugging the shin, sunken outseams, inseams and rise seams with double top-stitching,
    a dark wool crotch gusset, turned hems stacking above the boot, compression folds
    behind the knee and drag folds from the crotch
  - a sewn-on waistband with four belt loops and a buttoned fly
  - padded boiled-leather knee cops, each strapped round the leg above and below the
    knee with an iron buckle
Fitted to the body (and under the jerkin's skirt), skinned from the body with robust
weight transfer and exported onto the untouched MetaHuman skeleton. High-to-low: the
folds and seams of a detailed bake source are baked into the game mesh's normal map.
"""

import argparse
import math
import sys
from pathlib import Path

import bpy  # must come before bmesh/mathutils when running as a module
import bmesh
import numpy as np
from mathutils import Vector
from mathutils import noise as mnoise
from mathutils.bvhtree import BVHTree

import build_leather_jerkin as bj
import clothing_kit as ck
from build_leather_jerkin import (LEATHER, STEEL, MAIN, STRAP, FRONT, Piece, aaa, L, outermost,
                                  smooth01, subdivide, orient_outward, loft_grid, surface_ribbon, circular_smooth)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "Art" / "Characters" / "Clothing" / "LeatherTrousers"
ASSET_NAME = "SKM_LeatherTrousers"
TEX = "LeatherTrousers"

KNEE = 2.0                                  # leather tone of the knee cops (MAIN 0, STRAP 1)
WAIST_FRONT_Z, WAIST_BACK_Z = 0.952, 0.972  # top of the trousers: at the hips, higher at the back
WAIST_FRONT_Y, WAIST_BACK_Y = -0.12, 0.10
HEM_FRONT_Z, HEM_BACK_Z = 0.120, 0.106      # hems break over the boot in front
HEM_FRONT_Y, HEM_BACK_Y = -0.06, 0.06
EASE = 0.0065                               # outer surface of the trousers off the skin
LEATHER_T = 0.0022                          # trouser leather
INNER_KEEP = 0.045                          # inside facing kept at the hems and the waist
GAME_RATIO = 0.5                            # game mesh: decimate the once-subdivided shell to this
TAPER = 0.05                                # below the knee the legs narrow 5 mm per 10 cm
BAND_H = 0.036                              # waistband height
GUSSET_W = 0.030                            # half-width of the wool gusset at the crotch
GUSSET_LOW_Z = 0.60                         # where the gusset runs out down the inseam
FLY_X = 0.004                               # the fly flap overlaps to the wearer's left
LEATHER_TROUSERS = "#3d2d22"                # darker than the jerkin, like the reference
LEATHER_KNEE = "#4b372a"
LEATHER_STRAP = bj.LEATHER_STRAP
WOOL_GUSSET = "#37382f"
MUD = "#6f5f4b"


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def waist_z(y):
    return WAIST_FRONT_Z + (WAIST_BACK_Z - WAIST_FRONT_Z) * (y - WAIST_FRONT_Y) / (WAIST_BACK_Y - WAIST_FRONT_Y)


class Leg:
    """One leg's joint line (hip, knee, ankle) and a cylindrical frame round it:
    phi 0 at the front, +90 deg on the outside, 180 at the back, -90 on the inside.
    The legs are near vertical, so the frame works in horizontal slices."""

    def __init__(self, body, s):
        self.s = s
        self.hip, self.knee, self.ankle = (Vector(body.bone_head(f"{b}_{s}")) for b in ("thigh", "calf", "foot"))
        self.sign = 1.0 if self.hip.x > 0 else -1.0
        self.fwd = Vector((0.0, -1.0, 0.0))
        self.lat = Vector((self.sign, 0.0, 0.0))

    def axis(self, z):
        a, b = (self.hip, self.knee) if z >= self.knee.z else (self.knee, self.ankle)
        t = max(-0.25, min(1.25, (a.z - z) / (a.z - b.z)))
        q = a.lerp(b, t)
        return Vector((q.x, q.y, z))

    def dirn(self, phi):
        return self.fwd * math.cos(phi) + self.lat * math.sin(phi)

    def polar(self, p):
        o = self.axis(p.z)
        d = Vector((p.x - o.x, p.y - o.y, 0.0))
        return math.atan2(d.dot(self.lat), d.dot(self.fwd)), d.length


def leg_of(legs, p):
    return legs[0] if p.x * legs[0].sign >= 0 else legs[1]


def crotch_height(body):
    sel = (np.abs(body.co[:, 0]) < 0.01) & (body.co[:, 2] > 0.6) & (body.co[:, 2] < 1.0)
    return float(body.co[sel, 2].min())


# --- the trouser shell ----------------------------------------------------------------------
class ClosedBody:
    """Signed distance (m) to the body with its narrow creases closed (the crotch, the
    groin and the folds round the genitals), on a grid round the pelvis. Trousers bridge
    these creases, as leather does; pushing out along the nearest skin normal can't,
    because in a crease the nearest skin faces sideways (into the other side)."""
    LO, HI, H = np.array((-0.22, -0.20, 0.64)), np.array((0.22, 0.16, 1.03)), 0.0025
    CLOSE = 4                                   # voxels: creases narrower than ~20 mm are filled
    MARGIN = 0.001                              # extra clearance in the field (it is sampled)

    def __init__(self, body):
        import igl
        from scipy import ndimage
        tris = []
        for f in body.bm.faces:
            ids = [v.index for v in f.verts]
            tris += [(ids[0], ids[k], ids[k + 1]) for k in range(1, len(ids) - 1)]
        self.V, self.F = body.co.astype(np.float64), np.array(tris, np.int64)
        self.shape = tuple(int(round(n)) + 1 for n in (self.HI - self.LO) / self.H)
        axes = [self.LO[i] + self.H * np.arange(self.shape[i]) for i in range(3)]
        q = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        inside = (igl.fast_winding_number(self.V, self.F, q) > 0.5).reshape(self.shape)
        r = self.CLOSE
        k = np.arange(-r, r + 1) ** 2
        ball = np.add.outer(np.add.outer(k, k), k) <= r * r
        pad = r + 1
        closed = ndimage.binary_closing(np.pad(inside, pad, mode="edge"), structure=ball)[pad:-pad, pad:-pad, pad:-pad]
        # distances between voxel centres: the surface lies half a voxel short of them
        self.sdf = ((ndimage.distance_transform_edt(~closed) - 0.5) - (ndimage.distance_transform_edt(closed) - 0.5)) * self.H
        self.grad = np.gradient(self.sdf, self.H)

    def zone(self, p):
        """Where the field is used: the crotch and groin (well inside the grid)."""
        return abs(p.x) < 0.09 and 0.70 < p.z < 0.98 and self.LO[1] + 0.03 < p.y < self.HI[1] - 0.03

    def _sample(self, grid, pts):
        from scipy import ndimage
        return ndimage.map_coordinates(grid, ((pts - self.LO) / self.H).T, order=1, mode="nearest")

    def push(self, pts, min_offset):
        min_offset += self.MARGIN
        for _ in range(4):
            s = self._sample(self.sdf, pts)
            need = s < min_offset
            if not need.any():
                break
            g = np.stack([self._sample(gr, pts[need]) for gr in self.grad], 1)
            g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
            pts[need] += g * (min_offset - s[need])[:, None]
        return pts

    def inside_body(self, pts):
        """Exact inside test against the real body (winding number)."""
        import igl
        return igl.fast_winding_number(self.V, self.F, np.asarray(pts, np.float64)) > 0.5


def push_out_robust(body, field, verts, min_offset):
    """ck.push_out, except in the crotch and groin, where vertices are pushed out of
    the crease-closed body along its distance gradient."""
    zone = [v for v in verts if field.zone(v.co)]
    zs = set(zone)
    ck.push_out(body, [v for v in verts if v not in zs], min_offset)
    if zone:
        pts = field.push(np.array([v.co[:] for v in zone]), min_offset)
        for v, p in zip(zone, pts):
            v.co = Vector(p)


def relax_shell(body, field, bm, iters, factor, min_offset):
    """ck.relax with the robust push-out."""
    boundary = {v for e in bm.edges if e.is_boundary for v in e.verts}
    for _ in range(iters):
        new = {}
        for v in bm.verts:
            nb = [e.other_vert(v) for e in v.link_edges if (e.is_boundary or v not in boundary)]
            if nb:
                new[v] = v.co.lerp(sum((u.co for u in nb), Vector()) / len(nb), factor)
        for v, co in new.items():
            v.co = co
        push_out_robust(body, field, bm.verts, min_offset)
    bm.normal_update()


def drape(bm, legs):
    """Below the knee, trousers hang from the knee cap and the calf (narrowing a little
    towards the hem) instead of hugging the shin and the ankle."""
    bvh = BVHTree.FromBMesh(bm)
    phis = np.linspace(-math.pi, math.pi, 72, endpoint=False)
    for leg in legs:
        z_top = leg.knee.z + 0.03
        zs = np.arange(z_top, HEM_BACK_Z - 0.04, -0.01)
        r = np.array([[outermost([bvh], leg.axis(z), leg.dirn(ph), reach=0.16) for ph in phis] for z in zs])
        rd = r.copy()
        for i in range(1, len(zs)):
            rd[i] = np.maximum(r[i], rd[i - 1] - TAPER * 0.01)
        rd = np.array([circular_smooth(row, k=2, passes=2) for row in rd])
        for _ in range(2):
            rd[1:-1] = (rd[:-2] + 2 * rd[1:-1] + rd[2:]) / 4
        rd += np.array([0.003 * smooth01((0.19 - z) / 0.07) for z in zs])[:, None]   # the hem flares a touch
        for v in bm.verts:
            p = v.co
            if p.z >= z_top or leg_of(legs, p) is not leg:
                continue
            phi, rv = leg.polar(p)
            x = (phi + math.pi) / (2 * math.pi) * len(phis)
            j0 = int(math.floor(x)) % len(phis)
            j1, fj = (j0 + 1) % len(phis), x - math.floor(x)
            t0 = float(np.interp(-p.z, -zs, rd[:, j0]))
            t1 = float(np.interp(-p.z, -zs, rd[:, j1]))
            target = t0 + (t1 - t0) * fj
            if target > rv:
                w = smooth01((z_top - p.z) / 0.05)
                v.co = leg.axis(p.z) + leg.dirn(phi) * (rv + (target - rv) * w)
    bm.normal_update()


def cut(bm, co, normal, keep_below):
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=geom, plane_co=co, plane_no=normal,
                           clear_outer=keep_below, clear_inner=not keep_below)


def shell_base(body, field, legs):
    """The trousers at the body's resolution: an eased shell over the legs and hips,
    relaxed over muscle detail, hanging straight below the knee, cut at the waist and
    the hems."""
    arms = body.family(bj.ARMS)
    z = body.co[:, 2]
    mask = (arms < 0.3) & (z > HEM_BACK_Z - 0.02) & (z < WAIST_BACK_Z + 0.035)
    bm = ck.shell_from_faces(body, body.region_faces(mask), offset=EASE, smooth_iters=0)
    relax_shell(body, field, bm, 40, 0.5, EASE * 0.85)
    drape(bm, legs)
    relax_shell(body, field, bm, 4, 0.3, EASE * 0.85)
    slope = (WAIST_BACK_Z - WAIST_FRONT_Z) / (WAIST_BACK_Y - WAIST_FRONT_Y)
    cut(bm, Vector((0, WAIST_FRONT_Y, WAIST_FRONT_Z)), Vector((0, -slope, 1)).normalized(), keep_below=True)
    slope = (HEM_BACK_Z - HEM_FRONT_Z) / (HEM_BACK_Y - HEM_FRONT_Y)
    cut(bm, Vector((0, HEM_FRONT_Y, HEM_FRONT_Z)), Vector((0, -slope, 1)).normalized(), keep_below=False)
    bm.normal_update()
    return bm


def seam_fields(p, legs, crotch_z):
    """Distance to the nearest sewn seam (outseam, inseam, centre front/back rise, the
    gusset's edge) and the signed distance into the wool gusset (> 0 inside)."""
    leg = leg_of(legs, p)
    phi, r = leg.polar(p)
    sd = r * abs(wrap(phi - math.pi / 2))                                  # outseam
    g = -1.0
    if p.z < crotch_z + 0.015:
        inseam = r * abs(wrap(phi + math.pi / 2))
        sd = min(sd, inseam)
        g = GUSSET_W * smooth01((p.z - GUSSET_LOW_Z) / (crotch_z - GUSSET_LOW_Z)) - inseam
    if p.z > crotch_z - 0.015:
        sd = min(sd, abs(p.x))                                             # rise seams
        g = max(g, GUSSET_W * (1 - smooth01((p.z - crotch_z) / 0.07)) - abs(p.x))
    return min(sd, abs(g)), g, phi, r


def soften_trousers(body, field, bm, legs, crotch_z):
    """Displace the shell so it reads as worn leather trousers: soft undulation,
    compression folds behind the knee, fabric stacking above the hems, drag folds
    fanning from the crotch, sunken seams with the panels pillowing between them."""
    bm.normal_update()
    disp = []
    for v in bm.verts:
        p = v.co
        d = 0.0011 * mnoise.noise(p * 9.0) + 0.0004 * mnoise.noise(p * 26.0 + Vector((3, 1, 7)))
        sd, g, phi, r = seam_fields(p, legs, crotch_z)
        leg = leg_of(legs, p)
        zk = p.z - leg.knee.z
        back = smooth01((abs(phi) - 2.0) / 0.5)
        # compression folds behind the knee, broken into clusters
        kb = smooth01((0.075 - abs(zk - 0.005)) / 0.045) * back
        if kb > 0:
            cl = smooth01((mnoise.noise(p * 7.0 + Vector((2, 5, 1))) + 0.6) / 0.9)
            d += 0.0027 * kb * cl * math.sin(2 * math.pi * (zk + 0.006 * mnoise.noise(p * 18.0)) / 0.021)
        # the leg stacks in soft rings above the hem
        zh = p.z - (HEM_FRONT_Z + HEM_BACK_Z) / 2
        stack = smooth01((0.14 - zh) / 0.08) * smooth01((zh - 0.012) / 0.014)
        if stack > 0:
            slant = zh + 0.018 * math.sin(phi + 0.6) + 0.02 * mnoise.noise(p * Vector((7, 7, 2)))
            wave = math.sin(2 * math.pi * slant / 0.045 + 2.4 * mnoise.noise(p * 3.0))
            amp = max(0.0, 0.25 + 0.9 * mnoise.noise(p * 5.0 + Vector((1, 4, 2))))
            d += 0.0036 * stack * wave * amp * (0.65 + 0.35 * abs(math.cos(phi)))
        # drag folds fanning from the crotch up and out over the front of the thighs
        dx, dz = abs(p.x) - 0.02, p.z - crotch_z
        dist = math.hypot(dx, dz)
        front = smooth01((0.7 - abs(phi)) / 0.5) if phi < 0.4 else smooth01((1.1 - phi) / 0.5)
        fan = smooth01((dist - 0.03) / 0.04) * smooth01((0.17 - dist) / 0.08) * front * (dz > -0.03)
        if fan > 0:
            psi = math.atan2(dz, max(dx, 1e-4))
            d += 0.0013 * fan * math.sin(9.0 * psi + 1.6 * mnoise.noise(p * 4.0))
        # seams sink, panels pillow between them
        d += -0.0012 * math.exp(-(sd / 0.0035) ** 2) + 0.0005 * smooth01((sd - 0.004) / 0.03)
        disp.append(d)
    for v, d in zip(bm.verts, disp):
        v.co += v.normal * d
    push_out_robust(body, field, bm.verts, EASE * 0.8)
    bm.normal_update()


def decimate(bm, ratio):
    """Collapse-decimate (via a temporary object): keeps the folds and the hems,
    thins out the plain panels."""
    obj = ck.bm_to_object(bm, "_decimate")
    mod = obj.modifiers.new("Decimate", "DECIMATE")
    mod.decimate_type, mod.ratio = "COLLAPSE", ratio
    out = bmesh.new()
    out.from_object(obj, bpy.context.evaluated_depsgraph_get())
    mesh = obj.data
    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(mesh)
    bm.free()
    out.normal_update()
    return out


def build_shell(body, field, base, legs, crotch_z, levels, ratio=1.0):
    bm = subdivide(base.copy(), levels)
    soften_trousers(body, field, bm, legs, crotch_z)
    if ratio < 1.0:
        bm = decimate(bm, ratio)
        push_out_robust(body, field, bm.verts, EASE * 0.8)  # collapses can pull vertices into creases
        bm.normal_update()
    piece = Piece("Trousers", bm, LEATHER, MAIN, thickness=LEATHER_T)
    piece.inner_keep = INNER_KEEP
    piece.extra["gusset"] = {}
    for v in bm.verts:
        sd, g, phi, r = seam_fields(v.co, legs, crotch_z)
        piece.seamdist[v] = sd
        piece.extra["gusset"][v] = g
        piece.pc[v] = (r * phi, v.co.z)
        piece.hide[v] = (0.0 if v.co.x * legs[0].sign >= 0 else 0.41) + (0.27 if g > 0 else 0.0)
    return piece


# --- waistband, loops, fly ---------------------------------------------------------------------
def build_waistband(shell_bm):
    """A sewn-on band round the top, a hair proud of the trousers."""
    bvh = BVHTree.FromBMesh(shell_bm)
    top = [v.co for v in shell_bm.verts if abs(v.co.z - waist_z(v.co.y)) < 0.004]
    c = Vector((0.0, float(np.mean([p.y for p in top])), 0.0))
    n = 168
    rows = []
    for f in (0.0, 0.15, 0.5, 0.85, 1.0):                  # top edge down
        row = []
        for j in range(n):
            a = FRONT + 2 * math.pi * j / n
            d = Vector((math.cos(a), math.sin(a), 0.0))
            r, y = 0.15, c.y + d.y * 0.15
            for _ in range(3):                             # the cut height depends on y
                zt = waist_z(y)
                z = zt + 0.0015 - BAND_H * f
                r = outermost([bvh], Vector((c.x, c.y, min(z, zt - 0.0025))), d, reach=0.35)
                y = c.y + d.y * r
            row.append(Vector((c.x, c.y, z)) + d * (r + 0.0004 + 0.0003 * (1 - (2 * f - 1) ** 2)))
        rows.append(row)
    bm, vr = loft_grid(rows)
    orient_outward(bm)
    piece = Piece("Waistband", bm, LEATHER, STRAP, thickness=0.0024)
    rm = float(np.mean([(p - Vector((c.x, c.y, p.z))).length for p in rows[2]]))
    for row in vr:
        for j, v in enumerate(row):
            piece.pc[v] = (2 * math.pi * j / n * rm, v.co.z)
    return piece, c


def build_belt_loops(shell_bm, band, c):
    """Four belt loops, sewn below the band and folded over its top edge."""
    bvhs = [BVHTree.FromBMesh(shell_bm), BVHTree.FromBMesh(band.bm)]
    pieces = []
    for deg in (52, 138, -138, -52):
        a = FRONT + math.radians(deg)
        d = Vector((math.cos(a), math.sin(a), 0.0))
        side = Vector((0, 0, 1)).cross(d).normalized()
        zt = waist_z(c.y + d.y * 0.15)
        zs = np.linspace(zt - BAND_H - 0.012, zt - 0.004, 22)
        rs = np.array([max(outermost(bvhs, Vector((c.x, c.y, z)) + side * o, d, reach=0.35)
                           for o in (-0.005, 0.0, 0.005)) for z in zs]) + 0.0002 + 0.0015
        for _ in range(20):                                 # bridge the band's lower edge
            rs[1:-1] = np.maximum(rs[1:-1], (rs[:-2] + rs[2:]) / 2)
        samples = [(Vector((c.x, c.y, z)) + d * r, d) for z, r in zip(zs, rs)]
        loop, _ = surface_ribbon(samples, 0.011, 0.0, f"TrouserLoop_{deg}", STRAP, thickness=0.0015, crown=0.0002)
        pieces.append(loop)
    return pieces


def build_fly(shell_bm, crotch_z):
    """Buttoned fly flap down the centre front, under the band."""
    bvh = BVHTree.FromBMesh(shell_bm)
    z_top = waist_z(-0.12) - BAND_H + 0.004
    z_bot = crotch_z + 0.085
    samples = [h for z in np.linspace(z_top, z_bot, 20) if (h := bj.project_front(bvh, FLY_X, z))]
    flap, _ = surface_ribbon(samples, 0.026, 0.0004, "Fly", MAIN, thickness=0.0018, crown=0.0002, tips=(0.0, 0.013))
    pieces = [flap]
    for z in np.arange(z_top - 0.018, z_bot + 0.02, -0.031):
        hit = bj.project_front(bvh, FLY_X, z)
        if hit:
            loc, n = hit
            pieces.append(bj.rivet(loc + n * 0.0031, n, r=0.0042))
    band_hit = bj.project_front(bvh, FLY_X, waist_z(-0.12) - BAND_H / 2)
    if band_hit:
        loc, n = band_hit
        pieces.append(bj.rivet(loc + n * 0.0034, n, r=0.0045))
    return pieces


# --- knee cops -------------------------------------------------------------------------------
COP_H = 0.066                    # half-height
COP_PHI = math.radians(64)       # half-width round the knee
COP_ROT = math.radians(6)        # turned a little to the outside of the knee
COP_DOME = 0.0065                # padding at the centre
COP_T = 0.0045
KNEE_REACH = 0.12                # rays start this far out: past the knee, short of the other leg


def build_knee(shell_bm, leg):
    """A padded, boiled-leather knee cop (an oval dome with a rolled rim) and two
    straps round the leg, riveted on the inside and buckled on the outside."""
    bvh = BVHTree.FromBMesh(shell_bm)
    zc = leg.knee.z + 0.014
    na, nb = 15, 17
    grid = []
    for b in np.linspace(-1.0, 1.0, nb):
        row = []
        for a in np.linspace(-1.0, 1.0, na):
            u = a * math.sqrt(max(0.0, 1 - b * b / 2))     # square grid -> oval (elliptical map)
            v = b * math.sqrt(max(0.0, 1 - a * a / 2))
            z, phi = zc + v * COP_H, COP_ROT + u * COP_PHI
            row.append((u, v, z, phi, outermost([bvh], leg.axis(z), leg.dirn(phi), reach=KNEE_REACH)))
        grid.append(row)
    r = np.array([[q[4] for q in row] for row in grid])
    rs = r.copy()
    for _ in range(3):                                      # a stiff cop bridges small wrinkles
        rs[1:-1, 1:-1] = (rs[:-2, 1:-1] + rs[2:, 1:-1] + rs[1:-1, :-2] + rs[1:-1, 2:] + 4 * rs[1:-1, 1:-1]) / 8
        rs = np.maximum(rs, r)
    rows, uvs = [], []
    for i, row in enumerate(grid):
        pts, uv = [], []
        for j, (u, v, z, phi, _) in enumerate(row):
            rho = min(1.0, math.hypot(u, v))
            dome = COP_DOME * (1 - rho * rho) ** 0.6
            pts.append(leg.axis(z) + leg.dirn(phi) * (rs[i, j] + 0.0010 + COP_T + dome))
            uv.append((u * COP_PHI * float(np.mean(r)), v * COP_H))
        rows.append(pts)
        uvs.append(uv)
    bm, vr = loft_grid(rows, closed=False)
    bm.normal_update()
    bm.faces.ensure_lookup_table()
    centre = bm.faces[len(bm.faces) // 2]
    if centre.normal.dot(leg.dirn(COP_ROT)) < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
    cop = Piece(f"KneeCop_{leg.s}", bm, LEATHER, KNEE, thickness=COP_T)
    for row, uvrow in zip(vr, uvs):
        for vert, p in zip(row, uvrow):
            cop.pc[vert] = p
    pieces = [cop]

    # straps above and below the knee
    cop_probe = bm.copy()
    bj.thicken(Piece("_", cop_probe, LEATHER, thickness=COP_T))
    bvhs = [bvh, BVHTree.FromBMesh(cop_probe)]
    for vz in (0.64, -0.64):
        z = zc + vz * COP_H
        reach = COP_PHI * math.sqrt(1 - vz * vz / 2) * 0.80
        p_in, p_out = COP_ROT - reach, COP_ROT + reach       # strap ends, on the cop
        phis = np.linspace(p_in, p_out - 2 * math.pi, 56)    # the long way: round the back
        rs = np.array([outermost(bvhs, leg.axis(z), leg.dirn(ph), reach=KNEE_REACH) for ph in phis]) + 0.0006 + 0.0025
        for _ in range(25):
            rs[1:-1] = np.maximum(rs[1:-1], (rs[:-2] + rs[2:]) / 2)
        samples = [(leg.axis(z) + leg.dirn(ph) * rr, leg.dirn(ph)) for ph, rr in zip(phis, rs)]
        strap, centers = surface_ribbon(samples, 0.016, 0.0, f"KneeStrap_{leg.s}_{vz}", STRAP,
                                        thickness=0.0025, tips=(0.007, 0.008))
        pieces.append(strap)
        loc, n = samples[2]
        pieces.append(bj.rivet(loc + n * 0.0003, n, r=0.0030))
        # buckle at the outer edge of the cop; the strap's end runs on through it
        k = min(range(len(phis)), key=lambda i: abs(wrap(phis[i] - (COP_ROT + COP_PHI * 0.93))))
        loc, n = samples[k]
        along = centers[min(k + 1, len(centers) - 1)] - centers[max(k - 1, 0)]
        pieces.append(bj.buckle(loc + n * 0.0003, n, along, 0.022, 0.021, bar=0.0026))
    return pieces


def build_all(body, high=True):
    """Every piece of the trousers. `high` is the bake source (shell subdivided twice);
    the game mesh's shell is subdivided once. Details are placed on the bake source."""
    legs = [Leg(body, "l"), Leg(body, "r")]
    crotch_z = crotch_height(body)
    field = ClosedBody(body)
    base = shell_base(body, field, legs)
    shell_ref = build_shell(body, field, base, legs, crotch_z, 2)
    shell = shell_ref if high else build_shell(body, field, base, legs, crotch_z, 1, GAME_RATIO)
    band, c = build_waistband(shell_ref.bm)
    pieces = [shell, band]
    pieces += build_belt_loops(shell_ref.bm, band, c)
    pieces += build_fly(shell_ref.bm, crotch_z)
    for leg in legs:
        pieces += build_knee(shell_ref.bm, leg)
    return pieces


def to_object(pieces, name):
    return bj.pieces_to_object(pieces, name, extra_defaults={"gusset": -1.0})


# --- texture recipe ----------------------------------------------------------------------------
def trousers_layers(c):
    """Dark leather trousers (tone 0), straps and band (1), knee cops (2), the wool
    gusset (inside the 'gusset' distance field) and blackened iron fittings."""
    g = c.g
    srgb = aaa.engine.srgb
    tone, part, ed, sd = c.attr("tone"), c.attr("part"), c.attr("edgedist"), c.attr("seamdist")
    gus = c.attr("gusset")
    P = g.xyz(c.P)
    N = g.xyz(c.N)

    def near(x, target, width):
        return g.remap(g.math("ABSOLUTE", g.sub(x, target)), 0.0, width, 1.0, 0.0)

    def stitch_dots(scale):
        return g.remap(g.voronoi(g.mapping(c.P, (1 / scale,) * 3), 1.0), 0.16, 0.30, 1.0, 0.0)

    is_strap, is_knee = c.is_value(tone, STRAP), c.is_value(tone, KNEE)
    dots = stitch_dots(0.0042)
    seam = near(sd, 0.0, 0.0011)
    piping = near(sd, 0.0, 0.0042)
    seam_stitch = g.mul(near(sd, 0.0062, 0.0008), dots)                         # double top-stitching
    edge_stitch = g.mul(near(ed, g.mix_float(0.0065, 0.0032, is_strap), 0.0008), dots)
    hem_stitch = g.mul(g.mul(near(ed, 0.014, 0.0008), dots), g.math("LESS_THAN", P["Z"], 0.3))
    stitch = g.maxf(g.maxf(seam_stitch, edge_stitch), hem_stitch)
    hide_v = g.node("ShaderNodeTexWhiteNoise", {"W": c.attr("hide")}, noise_dimensions="1D").outputs["Value"]
    hide_k = g.add(0.90, g.mul(hide_v, 0.2))
    hide_tint = g.node("ShaderNodeCombineColor", {"Red": g.mul(hide_k, 1.02), "Green": hide_k,
                                                  "Blue": g.mul(hide_k, 0.97)}).outputs["Color"]
    burnish = g.remap(ed, 0.0, 0.0035, 1.0, 0.0)
    rim_wear = g.mul(g.remap(ed, 0.0, 0.007, 1.0, 0.0), aaa.weathering.breakup(c, 40.0, 0.4, 0.62))

    def tailor(s, color):
        col = srgb(color)
        dark = tuple(v * 0.45 for v in col[:3]) + (1.0,)
        light = tuple(min(1.0, v * 2.0) for v in col[:3]) + (1.0,)
        base = g.tint(s.base, hide_tint)
        base = g.mix_color(base, light, g.mul(rim_wear, 0.5))
        base = g.mix_color(base, dark, g.mul(burnish, 0.6))
        base = g.mix_color(base, dark, seam)
        base = g.mix_color(base, srgb(bj.STITCH), stitch)
        rough = g.mix_float(s.rough, 0.38, g.mul(burnish, 0.7))
        rough = g.mix_float(rough, 0.82, stitch)
        height = g.add(s.height, g.add(g.mul(piping, 1.0), g.mul(stitch, 0.7)))
        height = g.sub(height, g.add(g.mul(seam, 1.5), g.mul(near(ed, 0.0012, 0.0012), 0.8)))
        return aaa.engine.Surface(base, rough, s.metal, height, s.bump)

    # leather wears at its edges and seams (the convexity mask would also catch fold crests)
    cl = c.with_masks(edge=g.maxf(g.remap(ed, 0.0, 0.006, 1.0, 0.0), g.maxf(g.mul(piping, 0.5), g.mul(c.edge, 0.3))))
    main = tailor(L.leather(cl, color=LEATHER_TROUSERS, wear=0.5, dirt=0.6, grain_amount=0.4), LEATHER_TROUSERS)
    # the fronts of the thighs are rubbed paler and smoother; the lower legs carry dried mud
    rubbed = g.mul(g.mul(g.remap(N["Y"], -0.35, -0.85), g.mul(g.remap(P["Z"], 0.52, 0.60), g.remap(P["Z"], 0.84, 0.74))),
                   aaa.weathering.breakup(c, 7.0, 0.38, 0.62))
    mud = g.mul(g.remap(P["Z"], 0.36, 0.13), aaa.weathering.breakup(c, 11.0, 0.36, 0.6))
    splat = g.mul(g.mul(g.remap(g.voronoi(g.mapping(c.P, (90, 90, 90)), 1.0), 0.0, 0.11, 1.0, 0.0),
                        g.remap(g.noise(c.P, 30.0, 2.0), 0.55, 0.62)), g.remap(P["Z"], 0.48, 0.2))
    mud = g.clamp01(g.maxf(mud, splat))
    col = srgb(LEATHER_TROUSERS)
    base = g.mix_color(main.base, tuple(min(1.0, v * 1.45) for v in col[:3]) + (1.0,), g.mul(rubbed, 0.55))
    base = g.mix_color(base, srgb(MUD), g.mul(mud, 0.85))
    rough = g.mix_float(g.mix_float(main.rough, 0.40, g.mul(rubbed, 0.6)), 0.93, mud)
    height = g.add(main.height, g.mul(mud, 0.35))
    main = aaa.engine.Surface(base, rough, 0.0, height, main.bump)

    strap = tailor(L.leather(cl, color=LEATHER_STRAP, wear=0.6, dirt=0.5, grain_amount=0.35, scale=1.15), LEATHER_STRAP)
    knee = tailor(L.leather(cl, color=LEATHER_KNEE, wear=0.7, dirt=0.5, grain_amount=0.45), LEATHER_KNEE)
    wool = L.wool(c, color=WOOL_GUSSET, quilted=False, wear=0.35, dirt=0.6)
    in_gusset = g.math("GREATER_THAN", gus, 0.0)
    wool = aaa.engine.Surface(g.mix_color(wool.base, srgb(bj.STITCH), seam_stitch), wool.rough,
                              wool.metal, g.sub(wool.height, g.mul(seam, 1.2)), wool.bump)

    steel_edge = g.maxf(g.mul(c.edge, 0.5), g.mul(g.remap(ed, 0.0, 0.0016, 1.0, 0.0), 0.8))
    steel = L.steel_blackened(c.with_masks(edge=steel_edge), wear=0.5, dirt=0.5, rust=0.35, rub=0.0)
    return [(None, main), (is_strap, strap), (is_knee, knee), (in_gusset, wool), (c.is_value(part, STEEL), steel)]


# --- output ------------------------------------------------------------------------------------
def write_import_script():
    """The jerkin's one-click Unreal import, retargeted to the trousers."""
    src = (bj.OUT_DIR / "Import_LeatherJerkin_Unreal.py").read_text()
    text = (src.replace("leather jerkin top", "leather trousers")
               .replace("Leather jerkin imported", "Leather trousers imported")
               .replace("LeatherJerkin", TEX)
               .replace("The top is imported onto", "The trousers are imported onto"))
    (OUT_DIR / f"Import_{TEX}_Unreal.py").write_text(text)


def render_previews(body_fbx):
    """The trousers on their own, then the whole outfit (jerkin, belt, trousers), loaded
    the way Unreal would load them."""
    import build_leather_belt as bb
    import build_steel_sword as kit
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(body_fbx))
    for o in list(bpy.context.scene.objects):
        if o.type == "MESH" and not o.name.endswith("LOD0"):
            bpy.data.objects.remove(o)
    body_mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH")

    def load(fbx, tex_dir, name):
        before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.fbx(filepath=str(fbx))
        mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH" and o not in before)
        aaa.engine.assign_baked_material(mesh, tex_dir, name)
        return mesh

    load(OUT_DIR / f"{ASSET_NAME}.fbx", OUT_DIR / "Textures", TEX)
    clay = kit.make_material("M_Clay", (0.50, 0.45, 0.42), 0.0, 0.55)
    body_mesh.data.materials.clear()
    body_mesh.data.materials.append(clay)
    kit.setup_studio()
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = 1200, 1600
    scene.view_settings.exposure = -0.9
    for light, loc, energy in (("Key", (-1.6, -2.4, 2.2), 380), ("Rim", (1.8, 2.2, 1.6), 300),
                               ("Top", (0.0, 0.0, 3.0), 150)):
        obj = bpy.data.objects[light]
        obj.location, obj.data.energy, obj.data.size = loc, energy, 2.0
        kit.track(obj, Vector((0, 0, 0.6)))
    renders = OUT_DIR / "Renders"
    renders.mkdir(parents=True, exist_ok=True)
    kit.render_view(renders / "LeatherTrousers_Front.png", Vector((0.75, -2.35, 0.85)), (0.0, 0.0, 0.53), 50)
    kit.render_view(renders / "LeatherTrousers_Back.png", Vector((-0.75, 2.35, 0.85)), (0.0, 0.0, 0.53), 50)
    kit.render_view(renders / "LeatherTrousers_Side.png", Vector((2.4, -0.45, 0.8)), (0.0, 0.0, 0.53), 50)
    kit.render_view(renders / "LeatherTrousers_Knee.png", Vector((0.55, -0.72, 0.66)), (0.13, -0.06, 0.50), 70)
    kit.render_view(renders / "LeatherTrousers_Hem.png", Vector((0.62, -0.78, 0.36)), (0.12, -0.02, 0.2), 70)
    # the whole outfit
    load(bj.OUT_DIR / f"{bj.ASSET_NAME}.fbx", bj.OUT_DIR / "Textures", "LeatherJerkin")
    load(bb.OUT_DIR / f"{bb.ASSET_NAME}.fbx", bb.OUT_DIR / "Textures", bb.TEX)
    scene.render.resolution_x, scene.render.resolution_y = 1100, 1700
    for light, loc in (("Key", (-1.6, -2.4, 2.6)), ("Rim", (1.8, 2.2, 2.0)), ("Top", (0.0, 0.0, 3.4))):
        obj = bpy.data.objects[light]
        obj.location = loc
        kit.track(obj, Vector((0, 0, 0.95)))
    kit.render_view(renders / "Outfit_Front.png", Vector((1.15, -3.5, 1.25)), (0.0, 0.0, 0.92), 50)
    kit.render_view(renders / "Outfit_Back.png", Vector((-1.2, 3.45, 1.3)), (0.0, 0.0, 0.92), 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default=str(REPO_ROOT / "Art/Characters/Source/SKM_John_doe_BodyMesh.fbx"))
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--render", action="store_true")
    args, _ = ap.parse_known_args(sys.argv[1:])
    body = ck.Body(args.body)
    print("Building the game mesh (low)...")
    obj = to_object(build_all(body, high=False), ASSET_NAME)
    high = None
    if args.export:
        print("Building the bake source (high)...")
        high = to_object(build_all(body, high=True), ASSET_NAME + "_High")
    for o in (high, obj):
        if o is not None:
            tris = sum(len(f.vertices) - 2 for f in o.data.polygons)
            print(f"{o.name}: {len(o.data.vertices)} verts, {tris} tris")
    if args.export:
        print("Skinning...")
        bj.skin(body, obj)
        print("Unwrapping...")
        bj.unwrap(high)
        bj.unwrap(obj, game=True, res=args.res)
        tex_dir = OUT_DIR / "Textures"
        aaa.engine.bake(high, tex_dir, TEX, trousers_layers, res=args.res, ao_distance=0.03,
                        export_height=False, low=obj, ray=0.008)
        aaa.engine.assign_baked_material(obj, tex_dir, TEX)
        bpy.data.objects.remove(high)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT_DIR / f"{ASSET_NAME}.blend"), compress=True)
        ck.export_skeletal(body, obj, OUT_DIR / f"{ASSET_NAME}.fbx")
        write_import_script()
        print("Exported", OUT_DIR / f"{ASSET_NAME}.fbx")
    if args.render:
        render_previews(args.body)
    return body, obj


if __name__ == "__main__":
    main()
