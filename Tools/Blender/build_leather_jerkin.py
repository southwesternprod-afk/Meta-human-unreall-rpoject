"""Build the medieval leather jerkin top for a MetaHuman and export it.

    python3 Tools/Blender/build_leather_jerkin.py --body Art/Characters/Source/SKM_John_doe_BodyMesh.fbx
        [--no-bake] [--render] [--preview]

Pieces (all one skeletal mesh, one texture set):
  - leather jerkin: torso shell, flared skirt with front/back slits, stand-up collar,
    waist seam band, centre placket with five buckled closure straps, cross strap
  - quilted gambeson sleeves, riveted mail over the upper arms and at the throat
  - leather bracers with three buckled straps
  - laminated steel pauldrons (cap + four lames) with an embossed fleur-de-lis
Everything is fitted to the exported body, skinned from it with robust weight
transfer, and exported onto the untouched MetaHuman skeleton.
"""

import argparse
import math
import sys
from pathlib import Path

import bpy  # must come before bmesh/mathutils when running as a module
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

import clothing_kit as ck
import weapon_textures as wt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "Art" / "Characters" / "Clothing" / "LeatherJerkin"
ASSET_NAME = "SKM_LeatherJerkin"

# Part IDs (drive the texture recipe)
LEATHER, STEEL, CLOTH, MAIL = 0.0, 1.0, 2.0, 3.0
# Leather tones
MAIN, STRAP, BRACER = 0.0, 1.0, 2.0

WAIST_Z = 0.955          # jerkin body / skirt seam
V_BOTTOM_Z = 1.505       # bottom of the V-neck
V_HALF_WIDTH = 0.075     # half-width of the V-neck where it meets the collar
HEM_Z = 0.70             # bottom of the skirt
FRONT = -math.pi / 2     # the character faces -Y


# --- small helpers -------------------------------------------------------------
class Piece:
    """A bmesh plus per-vertex tags, merged into the final mesh at the end."""

    def __init__(self, name, bm, part, tone=0.0, thickness=0.0, rigid=False):
        self.name, self.bm, self.part, self.tone = name, bm, part, tone
        self.thickness, self.rigid = thickness, rigid
        self.pc = {}          # vert -> (u, v) pattern coordinates in metres


def loft_grid(rows, closed=True, bm=None):
    """rows: list of lists of Vectors (same length). Returns (bm, vert rows)."""
    bm = bm or bmesh.new()
    vr = [[bm.verts.new(p) for p in row] for row in rows]
    n = len(rows[0])
    span = n if closed else n - 1
    for a, b in zip(vr, vr[1:]):
        for j in range(span):
            k = (j + 1) % n
            bm.faces.new((a[j], a[k], b[k], b[j]))
    return bm, vr


def resample_loop(points, n):
    """Resample a closed polyline to n points evenly by arc length."""
    pts = np.asarray(points + [points[0]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return [Vector(np.array([np.interp(x, s, pts[:, i]) for i in range(3)])) for x in t]


def angle_of(p, c):
    return math.atan2(p[1] - c[1], p[0] - c[0])


def circular_smooth(values, k=3, passes=2):
    v = np.asarray(values, float)
    for _ in range(passes):
        v = sum(np.roll(v, s) for s in range(-k, k + 1)) / (2 * k + 1)
    return v


def orient_outward(bm):
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)


# --- jerkin body ------------------------------------------------------------------
def build_torso(body):
    arm = body.family(r"(upperarm|lowerarm|hand|thumb_|index_|middle_|ring_|pinky_)")
    z = body.co[:, 2]
    mask = (arm < 0.30) & (z > WAIST_Z - 0.03) & (z < ck.NECK_BASE_Z + 0.002)
    bm = ck.shell_from_faces(body, body.region_faces(mask), offset=0.012, smooth_iters=75,
                             smooth_factor=0.5, min_offset=0.0095)
    # clean horizontal cut at the waist seam
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, WAIST_Z), plane_no=(0, 0, 1),
                           clear_inner=True)
    # V-neck at the front, showing the mail underneath: split the front of the
    # shell along both edges of the V, then delete the faces inside it.
    slope = (ck.NECK_BASE_Z + 0.01 - V_BOTTOM_Z) / V_HALF_WIDTH
    for sx in (1, -1):
        normal = Vector((-sx * slope, 0, 1)).normalized()
        front = [f for f in bm.faces if f.calc_center_median().y < -0.03]
        geom = list({e for f in front for e in f.edges}) + list({v for f in front for v in f.verts}) + front
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0, 0, V_BOTTOM_Z), plane_no=normal)
    inside = [f for f in bm.faces
              if (c := f.calc_center_median()).y < -0.03 and c.z > V_BOTTOM_Z + slope * abs(c.x)]
    bmesh.ops.delete(bm, geom=inside, context="FACES")
    bm.normal_update()
    return bm


def bottom_and_top_loops(bm):
    loops = ck.ordered_boundary_loops(bm)
    zs = [np.mean([v.co.z for v in L]) for L in loops]
    bottom = loops[int(np.argmin(zs))]
    top = loops[int(np.argmax(zs))]
    return bottom, top


def build_skirt(body, torso_bm):
    bottom, _ = bottom_and_top_loops(torso_bm)
    pts = [v.co.copy() for v in bottom]
    c = np.mean([p[:2] for p in pts], axis=0)
    n = 112
    ring = resample_loop([Vector((p.x, p.y, WAIST_Z + 0.012)) for p in pts], n)
    # order by angle starting at the front, counter-clockwise
    ring.sort(key=lambda p: (angle_of(p, c) - FRONT) % (2 * math.pi))
    angles = np.array([(angle_of(p, c) - FRONT) % (2 * math.pi) for p in ring])
    r_top = np.array([math.hypot(p.x - c[0], p.y - c[1]) for p in ring]) - 0.002

    arms = body.family(r"(upperarm|lowerarm|hand|thumb_|index_|middle_|ring_|pinky_)")
    legs = body.co[(body.co[:, 2] < WAIST_Z) & (body.co[:, 2] > HEM_Z - 0.05) & (arms < 0.2)]
    leg_ang = (np.arctan2(legs[:, 1] - c[1], legs[:, 0] - c[0]) - FRONT) % (2 * math.pi)
    leg_r = np.hypot(legs[:, 0] - c[0], legs[:, 1] - c[1])

    zs = np.linspace(WAIST_Z + 0.012, HEM_Z, 26)
    radii = []
    prev = r_top
    for i, zr in enumerate(zs):
        t = i / (len(zs) - 1)
        base = r_top * (1 + 0.30 * t ** 1.2)
        sel = np.abs(legs[:, 2] - zr) < 0.012
        req = np.zeros(n)
        for j, a in enumerate(angles):
            d = np.abs((leg_ang[sel] - a + math.pi) % (2 * math.pi) - math.pi)
            near = leg_r[sel][d < math.radians(5)]
            req[j] = near.max() + 0.028 if len(near) else 0.0
        r = circular_smooth(np.maximum(base, req), k=4, passes=3)
        r = np.maximum(r, prev)           # skirts only flare outwards going down
        radii.append(r)
        prev = r

    def point(j, i, gap=0.0):
        a = angles[j] + FRONT
        r = radii[i][j]
        tangent = Vector((-math.sin(a), math.cos(a), 0))
        return Vector((c[0] + r * math.cos(a), c[1] + r * math.sin(a), zs[i])) + tangent * gap

    # Two panels (left: front->back through +X, right: back->front through -X)
    # so the front and back seams can open into slits near the hem.
    front_slit_top, back_slit_top = 0.845, 0.80
    half = n // 2

    def slit_gap(z, top):
        return 0.0 if z >= top else 0.004 + 0.022 * (top - z) / (top - HEM_Z)

    bm = bmesh.new()
    panels = []
    for cols, start_slit, end_slit in ((range(0, half + 1), front_slit_top, back_slit_top),
                                       (range(half, n + 1), back_slit_top, front_slit_top)):
        cols = list(cols)
        rows = []
        for i in range(len(zs)):
            row = []
            for k, j in enumerate(cols):
                g = 0.0
                if k == 0:
                    g = slit_gap(zs[i], start_slit)
                elif k == len(cols) - 1:
                    g = -slit_gap(zs[i], end_slit)
                row.append(point(j % n, i, g))
            rows.append(row)
        _, vr = loft_grid(rows, closed=False, bm=bm)
        panels.append(vr)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)   # closes the seams above the slits
    orient_outward(bm)
    piece = Piece("Skirt", bm, LEATHER, MAIN, thickness=0.003)
    return piece


def build_collar(body, torso_bm):
    """Stand-up collar rising from the jerkin neckline, open at the throat."""
    _, top = bottom_and_top_loops(torso_bm)
    c = np.mean([v.co[:2] for v in top], axis=0)
    loop = sorted([v.co.copy() for v in top], key=lambda p: (angle_of(p, c) - FRONT) % (2 * math.pi))
    n = 96
    base = resample_loop(loop, n)
    base.sort(key=lambda p: (angle_of(p, c) - FRONT) % (2 * math.pi))
    gap = math.radians(26)
    rows = []
    steps = 9
    keep = [p for p in base if gap < (angle_of(p, c) - FRONT) % (2 * math.pi) < 2 * math.pi - gap]
    for i in range(steps + 1):
        t = i / steps
        row = []
        for p in keep:
            a = (angle_of(p, c) - FRONT) % (2 * math.pi)
            frontness = math.cos(a / 2) ** 2          # 1 at the opening, 0 at the back
            height = (0.068 - 0.022 * frontness) * t
            flare = 1.0 + 0.10 * t ** 1.5
            q = Vector((c[0] + (p.x - c[0]) * flare, c[1] + (p.y - c[1]) * flare, p.z - 0.004 + height))
            row.append(q)
        rows.append(row)
    bm, vr = loft_grid(rows, closed=False)
    orient_outward(bm)
    return Piece("Collar", bm, LEATHER, MAIN, thickness=0.003)


def build_mail_gorget(body):
    """Mail over the upper chest and around the neck, seen through the V-neck
    and above the collar. Sits under the leather everywhere else."""
    arm = body.family(r"(upperarm|lowerarm|hand|thumb_|index_|middle_|ring_|pinky_)")
    z = body.co[:, 2]
    mask = (arm < 0.3) & (z > V_BOTTOM_Z - 0.06) & (z < ck.NECK_BASE_Z + 0.048)
    bm = ck.shell_from_faces(body, body.region_faces(mask), offset=0.0055, smooth_iters=10,
                             smooth_factor=0.5, min_offset=0.0045)
    orient_outward(bm)
    piece = Piece("MailGorget", bm, MAIL)
    for v in bm.verts:
        a = math.atan2(v.co.y - ck.NECK_CENTER_Y, v.co.x)
        piece.pc[v] = (a * 0.08, v.co.z)
    return piece


# --- surface details (straps, placket, waist band) -------------------------------------
def project_front(bvh, x, z):
    hit = bvh.ray_cast(Vector((x, -1.0, z)), Vector((0, 1, 0)), 2.0)
    if hit[0] is None:
        return None
    n = hit[1] if hit[1].y < 0 else -hit[1]
    return hit[0], n


def surface_ribbon(samples, width, lift, name, tone, part=LEATHER, thickness=0.0025):
    """Strap lying on a surface. samples: [(location, normal), ...] along its length."""
    centers = [loc + n * lift for loc, n in samples]
    bm = bmesh.new()
    rows, along = [], 0.0
    piece = Piece(name, bm, part, tone, thickness=thickness)
    for i, (c, (loc, n)) in enumerate(zip(centers, samples)):
        t = centers[min(i + 1, len(centers) - 1)] - centers[max(i - 1, 0)]
        side = n.cross(t).normalized()
        if i:
            along += (c - centers[i - 1]).length
        row = []
        for k in (-1, 1):
            v = bm.verts.new(c + side * (k * width / 2))
            piece.pc[v] = (along, k * width / 2)
            row.append(v)
        rows.append(row)
    for a, b in zip(rows, rows[1:]):
        bm.faces.new((a[0], a[1], b[1], b[0]))
    orient_outward(bm)
    return piece, centers


def front_line(bvh, p0, p1, steps):
    out = []
    for t in np.linspace(0, 1, steps):
        x = p0[0] + (p1[0] - p0[0]) * t
        z = p0[1] + (p1[1] - p0[1]) * t
        hit = project_front(bvh, x, z)
        if hit:
            out.append(hit)
    return out


def build_front_details(torso_bm):
    bvh = BVHTree.FromBMesh(torso_bm)
    pieces = []
    # centre placket from the V down to the waist
    placket, _ = surface_ribbon(front_line(bvh, (0.0, V_BOTTOM_Z - 0.004), (0.0, WAIST_Z + 0.004), 40),
                                0.046, 0.0012, "Placket", MAIN)
    pieces.append(placket)
    # five buckled closure straps across the front
    for z in (1.44, 1.345, 1.25, 1.155, 1.06):
        samples = front_line(bvh, (-0.058, z), (0.062, z), 16)
        strap, centers = surface_ribbon(samples, 0.021, 0.0042, f"Closure_{z}", STRAP)
        pieces.append(strap)
        loc, n = samples[-4]
        pieces.append(buckle(loc + n * 0.003, n, centers[-1] - centers[-6], 0.027, 0.03))
    # diagonal cross strap from the left shoulder to the right hip
    samples = front_line(bvh, (0.135, 1.535), (-0.175, 0.975), 70)
    strap, centers = surface_ribbon(samples, 0.034, 0.0055, "CrossStrap", STRAP, thickness=0.003)
    pieces.append(strap)
    i = len(samples) // 3
    loc, n = samples[i]
    pieces.append(buckle(loc + n * 0.0065, n, centers[i + 2] - centers[i - 2], 0.042, 0.046, bar=0.0034))
    return pieces


def build_waist_band(torso_bm, skirt_bm):
    """Leather band over the seam between the jerkin body and the skirt."""
    bvhs = [BVHTree.FromBMesh(torso_bm), BVHTree.FromBMesh(skirt_bm)]
    c = np.mean([v.co[:2] for v in torso_bm.verts if abs(v.co.z - WAIST_Z) < 0.02], axis=0)
    seg = 128
    rows = []
    for z, lift in ((WAIST_Z - 0.022, 0.0015), (WAIST_Z - 0.018, 0.0055), (WAIST_Z + 0.026, 0.0055),
                    (WAIST_Z + 0.030, 0.0015)):
        row = []
        for j in range(seg):
            a = 2 * math.pi * j / seg
            d = Vector((math.cos(a), math.sin(a), 0))
            o = Vector((c[0], c[1], z))
            r = 0.0
            for bvh in bvhs:
                # cast inwards from outside so we land on the outer surface
                hit = bvh.ray_cast(o + d * 0.6, -d, 0.6)
                if hit[0] is not None:
                    r = max(r, (hit[0] - o).length)
            row.append(o + d * (r + lift))
        rows.append(row)
    bm, vr = loft_grid(rows)
    orient_outward(bm)
    piece = Piece("WaistBand", bm, LEATHER, STRAP, thickness=0.003)
    for i, row in enumerate(vr):
        for j, v in enumerate(row):
            piece.pc[v] = (2 * math.pi * j / seg * 0.2, v.co.z)
    return piece


# --- arms --------------------------------------------------------------------------
def cut_plane(bm, co, normal):
    """Remove everything on the +normal side of the plane."""
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=geom, plane_co=co, plane_no=normal, clear_outer=True)


def arm_axes(body, side):
    s = "l" if side > 0 else "r"
    shoulder = Vector(body.bone_head(f"upperarm_{s}"))
    elbow = Vector(body.bone_head(f"lowerarm_{s}"))
    wrist = Vector(body.bone_head(f"hand_{s}"))
    return shoulder, elbow, wrist


def cylinder_pc(piece, a, b):
    """Pattern coords around/along the a->b axis (metres)."""
    axis = (b - a).normalized()
    ref = Vector((0, -1, 0))
    ref = (ref - axis * ref.dot(axis)).normalized()
    side = axis.cross(ref)
    for v in piece.bm.verts:
        d = v.co - a
        along = d.dot(axis)
        radial = d - axis * along
        ang = math.atan2(radial.dot(side), radial.dot(ref))
        piece.pc[v] = (ang * 0.05, along)


def build_sleeves(body):
    armw = body.family(r"(upperarm|lowerarm|hand|thumb_|index_|middle_|ring_|pinky_)")
    hand = body.family(r"(hand|thumb_|index_|middle_|ring_|pinky_)")
    mask = (armw > 0.22) & (hand < 0.15)
    bm = ck.shell_from_faces(body, body.region_faces(mask), offset=0.008, smooth_iters=12,
                             smooth_factor=0.5, min_offset=0.0065)
    pieces = []
    for side in (1, -1):
        sh, el, wr = arm_axes(body, side)
        d = (wr - el).normalized()
        part = bm.copy()
        # keep one arm, cut at the wrist
        cut_plane(part, (0, 0, 0), (-side, 0, 0))
        cut_plane(part, wr - d * 0.03, d)
        orient_outward(part)
        p = Piece(f"Sleeve_{side}", part, CLOTH)
        cylinder_pc(p, sh, wr)
        pieces.append(p)
    bm.free()
    return pieces


def build_mail_sleeves(body):
    up = body.family(r"upperarm")
    low = body.family(r"(lowerarm|hand)")
    bm = ck.shell_from_faces(body, body.region_faces((up > 0.35) & (low < 0.35)), offset=0.0145,
                             smooth_iters=10, smooth_factor=0.5, min_offset=0.0125)
    pieces = []
    for side in (1, -1):
        sh, el, wr = arm_axes(body, side)
        a = (el - sh).normalized()
        part = bm.copy()
        cut_plane(part, (0, 0, 0), (-side, 0, 0))
        cut_plane(part, el - a * 0.035, a)
        orient_outward(part)
        p = Piece(f"MailSleeve_{side}", part, MAIL)
        cylinder_pc(p, sh, el)
        pieces.append(p)
    bm.free()
    return pieces


def build_bracers(body):
    pieces = []
    for side in (1, -1):
        sh, el, wr = arm_axes(body, side)
        axis = wr - el
        length = axis.length
        d = axis.normalized()
        ref = Vector((0, -1, 0))
        ref = (ref - d * ref.dot(d)).normalized()
        side_v = d.cross(ref)
        seg, ts = 40, np.linspace(0.30, 0.93, 26)
        radii = []
        for t in ts:
            o = el + axis * t
            row = []
            for j in range(seg):
                a = 2 * math.pi * j / seg
                direction = ref * math.cos(a) + side_v * math.sin(a)
                hit = body.bvh.ray_cast(o, direction, 0.2)
                row.append((hit[3] if hit[0] is not None else 0.04))
            radii.append(circular_smooth(row, k=2, passes=2))
        radii = np.array(radii)
        # bracers are stiff: smooth along the arm too, then add clearance and flared ends
        for _ in range(4):
            radii[1:-1] = (radii[:-2] + radii[1:-1] * 2 + radii[2:]) / 4
        rows = []
        for i, t in enumerate(ts):
            u = (t - ts[0]) / (ts[-1] - ts[0])
            flare = 0.004 * max(0, (u - 0.85) / 0.15) ** 2 + 0.003 * max(0, (0.1 - u) / 0.1) ** 2
            o = el + axis * t
            rows.append([o + (ref * math.cos(2 * math.pi * j / seg) + side_v * math.sin(2 * math.pi * j / seg))
                         * (radii[i][j] + 0.0165 + flare) for j in range(seg)])
        bm, vr = loft_grid(rows)
        orient_outward(bm)
        p = Piece(f"Bracer_{side}", bm, LEATHER, BRACER, thickness=0.003)
        cylinder_pc(p, el, wr)
        pieces.append(p)

        # three strap bands
        for t0 in (0.40, 0.62, 0.84):
            band_rows = []
            for t in (t0 - 0.03, t0 - 0.026, t0 + 0.026, t0 + 0.03):
                i = int(np.argmin(np.abs(ts - t)))
                lift = 0.0028 if abs(t - t0) < 0.028 else 0.0008
                o = el + axis * t
                band_rows.append([o + (ref * math.cos(2 * math.pi * j / seg) + side_v * math.sin(2 * math.pi * j / seg))
                                  * (radii[i][j] + 0.0165 + lift) for j in range(seg)])
            bbm, _ = loft_grid(band_rows)
            orient_outward(bbm)
            sp = Piece(f"BracerStrap_{side}_{t0}", bbm, LEATHER, STRAP)
            cylinder_pc(sp, el, wr)
            pieces.append(sp)
            pieces.append(buckle(o_on_surface(el + axis * t0, side_v * side, radii, ts, t0, 0.0165 + 0.0028),
                                 side_v * side, d, 0.022, 0.026))
    return pieces


def o_on_surface(o, direction, radii, ts, t0, extra):
    i = int(np.argmin(np.abs(ts - t0)))
    return o + direction.normalized() * (float(np.mean(radii[i])) + extra)


def buckle(center, normal, along, width, height, bar=0.0028):
    """Rectangular iron buckle frame lying on a surface."""
    n = normal.normalized()
    u = along.normalized()
    u = (u - n * u.dot(n)).normalized()
    w = n.cross(u)
    bm = bmesh.new()
    outer = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

    def ring(scale_w, scale_h, lift):
        return [bm.verts.new(center + n * lift + u * (sx * height / 2 * scale_h) + w * (sy * width / 2 * scale_w))
                for sx, sy in outer]

    o0 = ring(1.0, 1.0, 0.0015)
    i0 = ring(1 - 2 * bar / width, 1 - 2 * bar / height, 0.0015)
    o1 = ring(1.0, 1.0, 0.0045)
    i1 = ring(1 - 2 * bar / width, 1 - 2 * bar / height, 0.0045)
    for a, b in ((o0, o1), (o1, i1), (i1, i0)):
        for j in range(4):
            k = (j + 1) % 4
            bm.faces.new((a[j], a[k], b[k], b[j]))
    # prong across the middle
    pr = [center + n * 0.004 + u * (s * height * 0.42) + w * (t * 0.0012) for s in (-1, 1) for t in (-1, 1)]
    pv = [bm.verts.new(p) for p in pr]
    bm.faces.new((pv[0], pv[2], pv[3], pv[1]))
    orient_outward(bm)
    return Piece("Buckle", bm, STEEL, rigid=True)


# --- pauldrons -----------------------------------------------------------------------
def build_pauldrons(body):
    pieces = []
    for side in (1, -1):
        sh, el, wr = arm_axes(body, side)
        A = (el - sh).normalized()                 # down the arm
        out = Vector((side, 0, 0))
        e2 = (out - A * out.dot(A)).normalized()   # outward from the shoulder
        e3 = A.cross(e2)
        C = sh + e2 * 0.004 - A * 0.012

        def direction(alpha, phi):
            return (-A * math.cos(alpha) * 1.0 + (e2 * math.cos(phi) + e3 * math.sin(phi)) * math.sin(alpha))

        def stretch(alpha):
            return 1.0 + 0.45 * max(0.0, math.cos(alpha) * -1)  # longer down the arm

        # required radius: clear the body (mail sleeve + jerkin) everywhere under the plate
        req = []
        for alpha in np.radians(np.linspace(5, 115, 24)):
            for phi in np.radians(np.linspace(-115, 115, 24)):
                dvec = direction(alpha, phi).normalized()
                hit = body.bvh.ray_cast(C, dvec, 0.4)
                if hit[0] is not None:
                    req.append(hit[3] / (dvec.length * 1.0))
        R = float(np.percentile(req, 97)) + 0.026
        lames = [("Cap", 0, 54, 0.012), ("Lame1", 50, 70, 0.009), ("Lame2", 66, 86, 0.006),
                 ("Lame3", 82, 102, 0.003), ("Lame4", 98, 116, 0.0)]
        for name, a0, a1, extra in lames:
            rows = []
            alphas = np.radians(np.linspace(a0, a1, 14 if name == "Cap" else 7))
            phis = np.radians(np.linspace(-118, 118, 48))
            for k, alpha in enumerate(alphas):
                roll = 0.0025 if k == len(alphas) - 1 else 0.0
                rows.append([C + direction(alpha, phi) * (R + extra + roll) for phi in phis])
            bm, vr = loft_grid(rows, closed=False)
            orient_outward(bm)
            p = Piece(f"Pauldron{name}_{side}", bm, STEEL, thickness=0.002, rigid=True)
            for i, row in enumerate(vr):
                for j, v in enumerate(row):
                    p.pc[v] = (phis[j] * (R + extra), alphas[i] * (R + extra))
            pieces.append(p)
            # rivets at both ends of each overlap
            if name != "Cap":
                for phi in np.radians((-108, 108)):
                    pos = C + direction(math.radians(a0 + 3), phi) * (R + extra + 0.0035)
                    pieces.append(rivet(pos, direction(math.radians(a0 + 3), phi)))
    return pieces


def rivet(center, normal, r=0.0038):
    n = normal.normalized()
    u = n.orthogonal().normalized()
    w = n.cross(u)
    bm = bmesh.new()
    seg = 10
    rings = []
    for h, k in ((0.0, 1.0), (0.0012, 0.85), (0.002, 0.5)):
        rings.append([bm.verts.new(center + n * h + (u * math.cos(2 * math.pi * j / seg) + w * math.sin(2 * math.pi * j / seg)) * r * k)
                      for j in range(seg)])
    for a, b in zip(rings, rings[1:]):
        for j in range(seg):
            k = (j + 1) % seg
            bm.faces.new((a[j], a[k], b[k], b[j]))
    tip = bm.verts.new(center + n * 0.0024)
    for j in range(seg):
        bm.faces.new((rings[-1][j], rings[-1][(j + 1) % seg], tip))
    orient_outward(bm)
    return Piece("Rivet", bm, STEEL, rigid=True)


# --- assembly --------------------------------------------------------------------------
def build_all(body):
    torso_bm = build_torso(body)
    torso = Piece("Torso", torso_bm, LEATHER, MAIN, thickness=0.003)
    skirt = build_skirt(body, torso_bm)
    pieces = [torso, skirt, build_collar(body, torso_bm), build_mail_gorget(body),
              build_waist_band(torso_bm, skirt.bm)]
    pieces += build_front_details(torso_bm)
    pieces += build_sleeves(body)
    pieces += build_mail_sleeves(body)
    pieces += build_bracers(body)
    pieces += build_pauldrons(body)
    return pieces


def thicken(piece):
    """Give a surface piece real thickness: an inner copy pushed in along the
    vertex normals plus rim faces along the open edges (hems, cuffs, slits)."""
    bm = piece.bm
    bm.normal_update()
    outer_faces = list(bm.faces)
    boundary = [e for e in bm.edges if e.is_boundary]
    inner = {}
    for v in list(bm.verts):
        n = v.normal if v.normal.length > 1e-6 else Vector((0, 0, 0))
        inner[v] = bm.verts.new(v.co - n * piece.thickness)
        if v in piece.pc:
            piece.pc[inner[v]] = piece.pc[v]
    for f in outer_faces:
        bm.faces.new([inner[v] for v in reversed(f.verts)])
    for e in boundary:
        f = e.link_faces[0]
        a, b = e.verts
        # keep the rim winding consistent with the outer face that owns the edge
        loop = next(l for l in f.loops if l.edge is e)
        if loop.vert is not a:
            a, b = b, a
        bm.faces.new((b, a, inner[a], inner[b]))
    bm.normal_update()


def pieces_to_object(pieces, name):
    """Merge pieces into one mesh object with part/tone/pc/rigid_id attributes."""
    bm = bmesh.new()
    part_l, tone_l, pc_l, rid_l = [], [], [], []
    for rid, p in enumerate(pieces):
        src = p.bm
        if p.thickness:
            thicken(p)
        mapping = {}
        for v in src.verts:
            nv = bm.verts.new(v.co)
            mapping[v] = nv
            part_l.append(p.part)
            tone_l.append(p.tone)
            pc_l.append(p.pc.get(v, (0.0, 0.0)))
            rid_l.append(rid if p.rigid else -1)
        for f in src.faces:
            try:
                bm.faces.new([mapping[v] for v in f.verts])
            except ValueError:
                pass
    for f in bm.faces:
        f.smooth = True
    obj = ck.bm_to_object(bm, name)
    ck.set_attr(obj, "part", part_l)
    ck.set_attr(obj, "tone", tone_l)
    ck.set_attr(obj, "rigid_id", rid_l)
    ck.set_attr(obj, "pc", pc_l, kind="FLOAT2")
    return obj


# --- texture recipe -----------------------------------------------------------------------
CREST = OUT_DIR / "Source" / "T_LeatherJerkin_CrestMask.png"


def jerkin_layers(c):
    """part 0 leather (tone 0 jerkin, 1 straps/band, 2 bracers), 1 steel,
    2 quilted wool gambeson, 3 riveted mail."""
    g = c.g
    rgb = wt.rgb
    pc = g.node("ShaderNodeSeparateXYZ", {
        "Vector": g.node("ShaderNodeAttribute", attribute_name="pc").outputs["Vector"]}).outputs
    tone = g.node("ShaderNodeAttribute", attribute_name="tone").outputs["Fac"]
    P = g.node("ShaderNodeSeparateXYZ", {"Vector": c.P}).outputs
    scratch = c.scratches()

    def near(x, target, width):
        return g.remap(g.math("ABSOLUTE", g.math("SUBTRACT", x, target)), 0.0, width, 1.0, 0.0)

    def dashes(x, period, duty=0.55):
        return g.math("GREATER_THAN", g.math("SINE", g.math("MULTIPLY", x, 2 * math.pi / period)),
                      1.0 - 2.0 * duty)

    # ---- leather ---------------------------------------------------------------------
    grain = g.remap(g.math("ADD", g.math("MULTIPLY", g.voronoi(c.P, 1400.0), 0.7),
                           g.math("MULTIPLY", g.noise(c.P, 450.0, 4.0), 0.5)), 0.2, 0.8)
    mottle = g.remap(g.noise(c.P, 9.0, 4.0), 0.35, 0.68)
    crease_noise = g.noise(g.mapping(c.P, (60, 60, 14)), 1.0, 6.0, 0.6)
    creases = g.math("MULTIPLY", g.remap(g.math("ABSOLUTE", g.math("SUBTRACT", crease_noise, 0.5)), 0.0, 0.035, 1.0, 0.0),
                     g.remap(g.noise(c.P, 4.0, 2.0), 0.5, 0.66))
    # panel seams down the front and back of the jerkin, with stitching either side
    is_body = g.math("MULTIPLY", g.math("LESS_THAN", P["Z"], 1.47), g.math("LESS_THAN", tone, 0.5))
    ax = g.math("ABSOLUTE", P["X"])
    seam_x = g.mix_float(0.085, 0.07, g.math("GREATER_THAN", P["Y"], 0.0))
    seam = g.math("MULTIPLY", near(ax, seam_x, 0.0013), is_body)
    stitch_rows = g.math("MAXIMUM", near(ax, g.math("ADD", seam_x, 0.0045), 0.0008),
                         near(ax, g.math("SUBTRACT", seam_x, 0.0045), 0.0008))
    seam_stitch = g.math("MULTIPLY", g.math("MULTIPLY", stitch_rows, dashes(P["Z"], 0.0065)), is_body)
    # straps and placket: stitch rows a fixed distance from the strap centre line
    ay = g.math("ABSOLUTE", pc["Y"])
    has_pc = g.math("GREATER_THAN", g.math("ABSOLUTE", pc["X"]), 0.0001)
    strap_rows = g.math("MAXIMUM", near(ay, 0.0075, 0.0007), near(ay, 0.019, 0.0007))
    strap_stitch = g.math("MULTIPLY", g.math("MULTIPLY", strap_rows, dashes(pc["X"], 0.006)), has_pc)
    stitch = g.math("MAXIMUM", seam_stitch, strap_stitch)

    is_strap = g.math("LESS_THAN", g.math("ABSOLUTE", g.math("SUBTRACT", tone, STRAP)), 0.5)
    is_bracer = g.math("GREATER_THAN", tone, 1.5)

    def by_tone(main, strap, bracer):
        return g.mix_color(g.mix_color(main, strap, is_strap), bracer, is_bracer)

    lbase = by_tone(g.mix_color(rgb(0.075, 0.040, 0.021), rgb(0.135, 0.075, 0.040), mottle),
                    g.mix_color(rgb(0.042, 0.022, 0.011), rgb(0.070, 0.038, 0.019), mottle),
                    g.mix_color(rgb(0.095, 0.052, 0.027), rgb(0.150, 0.085, 0.045), mottle))
    lbase = g.mix_color(lbase, rgb(0.05, 0.027, 0.014), g.math("MULTIPLY", grain, 0.35))
    lbase = g.mix_color(lbase, rgb(0.035, 0.018, 0.009), g.math("MULTIPLY", creases, 0.3))
    lbase = g.mix_color(lbase, rgb(0.20, 0.13, 0.08), g.math("MULTIPLY", scratch, 0.45))
    lbase = g.mix_color(lbase, rgb(0.19, 0.12, 0.07), g.math("MULTIPLY", c.edge, 0.55))
    lbase = g.mix_color(lbase, rgb(0.02, 0.011, 0.006), g.math("MULTIPLY", c.cavity, 0.85))
    lbase = g.mix_color(lbase, rgb(0.025, 0.014, 0.008), seam)
    lbase = g.mix_color(lbase, rgb(0.17, 0.14, 0.10), stitch)
    lrough = g.math("ADD", 0.52, g.math("MULTIPLY", grain, 0.14))
    lrough = g.mix_float(lrough, 0.38, g.math("MULTIPLY", c.edge, 0.7))
    lrough = g.mix_float(lrough, 0.70, g.math("MULTIPLY", scratch, 0.6))
    lrough = g.mix_float(lrough, 0.80, c.cavity)
    lrough = g.mix_float(lrough, 0.75, stitch)
    lheight = g.math("SUBTRACT", g.math("ADD", g.math("MULTIPLY", grain, 0.35), g.math("MULTIPLY", stitch, 0.6)),
                     g.math("ADD", g.math("MULTIPLY", creases, 0.5),
                            g.math("ADD", g.math("MULTIPLY", seam, 1.2), g.math("MULTIPLY", scratch, 0.4))))
    leather = wt.Surface(lbase, lrough, 0.0, lheight, 0.00022)

    # ---- steel: pauldrons, buckles, rivets -----------------------------------------------
    hammer = g.remap(g.voronoi(c.P, 70.0), 0.0, 0.55)
    tone_s = g.remap(g.noise(c.P, 30.0, 4.0), 0.4, 0.65)
    # fleur-de-lis on the pauldron caps, placed in the plate's own (phi, alpha) coordinates
    crest_uv = g.node("ShaderNodeCombineXYZ", {
        "X": g.math("ADD", g.math("DIVIDE", pc["X"], 0.075), 0.5),
        "Y": g.math("DIVIDE", g.math("SUBTRACT", pc["Y"], 0.004), 0.075)}).outputs["Vector"]
    crest_img = bpy.data.images.load(str(CREST), check_existing=True)
    crest_img.colorspace_settings.name = "Non-Color"
    crest = g.math("MULTIPLY", g.image(crest_img, crest_uv, "CLIP").outputs["Color"], c.is_part(STEEL))
    sbase = g.mix_color(rgb(0.15, 0.15, 0.16), rgb(0.24, 0.24, 0.25), tone_s)
    sbase = g.mix_color(sbase, rgb(0.18, 0.18, 0.19), g.math("MULTIPLY", hammer, 0.3))
    sbase = g.mix_color(sbase, rgb(0.50, 0.50, 0.51), c.edge)
    sbase = g.mix_color(sbase, rgb(0.55, 0.55, 0.56), g.math("MULTIPLY", scratch, 0.5))
    sbase = g.mix_color(sbase, rgb(0.55, 0.55, 0.56), g.math("MULTIPLY", crest, 0.85))
    sbase = g.mix_color(sbase, rgb(0.05, 0.045, 0.04), g.math("MULTIPLY", c.cavity, 0.85))
    srough = g.math("ADD", 0.42, g.math("MULTIPLY", tone_s, 0.12))
    srough = g.mix_float(srough, 0.22, c.edge)
    srough = g.mix_float(srough, 0.42, g.math("MULTIPLY", scratch, 0.6))
    srough = g.mix_float(srough, 0.20, crest)
    srough = g.mix_float(srough, 0.70, c.cavity)
    sheight = g.math("ADD", g.math("MULTIPLY", hammer, 0.4),
                     g.math("SUBTRACT", g.math("MULTIPLY", crest, 1.6), g.math("MULTIPLY", scratch, 0.4)))
    steel = wt.Surface(sbase, srough, wt.metal_with_grime(c), sheight, 0.00012)

    # ---- quilted wool gambeson -----------------------------------------------------------
    channel = g.math("ABSOLUTE", g.math("SINE", g.math("MULTIPLY", pc["X"], math.pi / 0.026)))
    quilt_line = g.remap(channel, 0.0, 0.16, 1.0, 0.0)
    weave = g.remap(g.noise(g.mapping(c.P, (2600, 2600, 2600)), 1.0, 2.0), 0.35, 0.65)
    wool_var = g.remap(g.noise(c.P, 14.0, 3.0), 0.35, 0.65)
    cbase = g.mix_color(rgb(0.075, 0.080, 0.055), rgb(0.12, 0.125, 0.09), wool_var)
    cbase = g.mix_color(cbase, rgb(0.05, 0.052, 0.036), g.math("MULTIPLY", weave, 0.3))
    cbase = g.mix_color(cbase, rgb(0.035, 0.036, 0.025), g.math("MULTIPLY", quilt_line, 0.8))
    cbase = g.mix_color(cbase, rgb(0.03, 0.03, 0.022), g.math("MULTIPLY", c.cavity, 0.8))
    crough = g.mix_float(0.86, 0.95, quilt_line)
    cheight = g.math("ADD", g.math("POWER", channel, 0.5), g.math("MULTIPLY", weave, 0.15))
    cloth = wt.Surface(cbase, crough, 0.0, cheight, 0.0006)

    # ---- riveted mail -------------------------------------------------------------------
    def ring_grid(offset):
        m = g.node("ShaderNodeMapping", {"Vector": g.node("ShaderNodeCombineXYZ", {"X": pc["X"], "Y": pc["Y"]}).outputs["Vector"],
                                         "Scale": (1 / 0.0075, 1 / 0.0075, 1.0), "Location": (offset, offset, 0.0)})
        vor = g.node("ShaderNodeTexVoronoi", {"Vector": m.outputs["Vector"], "Scale": 1.0, "Randomness": 0.0},
                     voronoi_dimensions="2D")
        d = vor.outputs["Distance"]
        ring = g.remap(g.math("ABSOLUTE", g.math("SUBTRACT", d, 0.44)), 0.0, 0.16, 1.0, 0.0)
        return g.math("POWER", ring, 0.6)            # rounded wire profile

    rings = g.math("MAXIMUM", ring_grid(0.0), ring_grid(0.5))
    rust = g.math("MULTIPLY", g.remap(g.noise(c.P, 40.0, 3.0), 0.55, 0.7), g.math("SUBTRACT", 1.0, rings))
    mbase = g.mix_color(rgb(0.006, 0.006, 0.006), rgb(0.36, 0.36, 0.37), rings)
    mbase = g.mix_color(mbase, rgb(0.07, 0.035, 0.015), g.math("MULTIPLY", rust, 0.5))
    mbase = g.mix_color(mbase, rgb(0.02, 0.02, 0.02), g.math("MULTIPLY", c.cavity, 0.6))
    mrough = g.mix_float(0.75, 0.32, rings)
    mmetal = g.math("MULTIPLY", rings, 1.0)
    mail = wt.Surface(mbase, mrough, mmetal, rings, 0.0012)

    return [(None, leather),
            (c.is_part(STEEL), steel),
            (c.is_part(CLOTH), cloth),
            (c.is_part(MAIL), mail)]


# --- skinning, UVs, export -------------------------------------------------------------------
def skin(body, obj):
    W = ck.transfer_weights(body, obj, max_dist=0.05, max_angle=40.0, max_influences=8)
    rid = np.zeros(len(obj.data.vertices))
    obj.data.attributes["rigid_id"].data.foreach_get("value", rid)
    # rigid pieces (steel plates, buckles, rivets) move as one: average their weights
    for r in np.unique(rid[rid >= 0]):
        ids = rid == r
        W[ids] = W[ids].mean(0)
    W = ck.limit_influences(W, 8)
    ck.apply_weights(body, obj, W)


def unwrap(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.002)
    bpy.ops.uv.pack_islands(rotate=True, rotate_method="AXIS_ALIGNED", margin=0.002)
    bpy.ops.object.mode_set(mode="OBJECT")


def render_previews(body_fbx):
    """Studio renders of the exported FBX on the body, loaded the way Unreal
    would load them (fresh import, untouched skeleton)."""
    import build_steel_sword as kit
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(body_fbx))
    for o in list(bpy.context.scene.objects):
        if o.type == "MESH" and not o.name.endswith("LOD0"):
            bpy.data.objects.remove(o)
    body_mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH")
    bpy.ops.import_scene.fbx(filepath=str(OUT_DIR / f"{ASSET_NAME}.fbx"))
    garment = next(o for o in bpy.context.scene.objects if o.type == "MESH" and o is not body_mesh)
    wt.assign_final_material(garment, OUT_DIR / "Textures", "LeatherJerkin")
    clay = kit.make_material("M_Clay", (0.50, 0.45, 0.42), 0.0, 0.55)
    body_mesh.data.materials.clear()
    body_mesh.data.materials.append(clay)

    kit.setup_studio()
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = 1400, 1600
    scene.view_settings.exposure = -0.9
    for light, loc, energy in (("Key", (-1.6, -2.4, 2.6), 380), ("Rim", (1.8, 2.2, 2.0), 300),
                               ("Top", (0.0, 0.0, 3.4), 150)):
        obj = bpy.data.objects[light]
        obj.location = loc
        obj.data.energy = energy
        obj.data.size = 2.0
        kit.track(obj, Vector((0, 0, 1.15)))
    renders = OUT_DIR / "Renders"
    renders.mkdir(parents=True, exist_ok=True)
    kit.render_view(renders / "LeatherJerkin_Front.png", Vector((1.25, -2.55, 1.45)), (0.0, 0.0, 1.13), 50)
    kit.render_view(renders / "LeatherJerkin_Back.png", Vector((-1.35, 2.45, 1.5)), (0.0, 0.0, 1.13), 50)
    kit.render_view(renders / "LeatherJerkin_Detail.png", Vector((0.75, -0.95, 1.62)), (0.10, -0.02, 1.42), 70)
    kit.render_view(renders / "LeatherJerkin_Pauldron.png", Vector((0.95, -0.35, 1.85)), (0.22, 0.0, 1.50), 85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default=str(REPO_ROOT / "Art/Characters/Source/SKM_John_doe_BodyMesh.fbx"))
    ap.add_argument("--no-bake", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--res", type=int, default=4096)
    ap.add_argument("--render", action="store_true")
    args, _ = ap.parse_known_args(sys.argv[1:])
    body = ck.Body(args.body)
    pieces = build_all(body)
    obj = pieces_to_object(pieces, ASSET_NAME)
    print(f"{ASSET_NAME}: {len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces")
    if args.export:
        print("Skinning...")
        skin(body, obj)
        print("Unwrapping...")
        unwrap(obj)
        wt.TEX_RES = args.res
        wt.texture_weapon(obj, OUT_DIR / "Textures", "LeatherJerkin", jerkin_layers, bake=not args.no_bake)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT_DIR / f"{ASSET_NAME}.blend"), compress=True)
        ck.export_skeletal(body, obj, OUT_DIR / f"{ASSET_NAME}.fbx")
        print("Exported", OUT_DIR / f"{ASSET_NAME}.fbx")
    if args.render:
        render_previews(args.body)
    return body, obj


if __name__ == "__main__":
    main()
