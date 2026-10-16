"""Build the medieval leather jerkin top for a MetaHuman and export it.

    python3 Tools/Blender/build_leather_jerkin.py --body Art/Characters/Source/SKM_John_doe_BodyMesh.fbx
        [--no-bake] [--render] [--preview]

Pieces (all one skeletal mesh, one texture set):
  - leather jerkin: torso shell, flared skirt with front/back slits, stand-up collar,
    waist seam band, centre placket with five buckled closure straps, two shoulder
    straps that cross in an X on the chest and the back
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
from mathutils import noise as mnoise
from mathutils.bvhtree import BVHTree

import clothing_kit as ck

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "plugins" / "aaa-materials" / "lib"))
import aaa_materials as aaa  # noqa: E402  (the repo's aaa-materials plugin)
from aaa_materials import library as L  # noqa: E402

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
HEM_Z = 0.79             # bottom of the skirt (upper thigh, worn over trousers)
CROSS_Z = 1.405          # where the two shoulder straps cross, front and back
CROSS_SLOPE = 0.82       # rise of each shoulder strap per metre across the chest
ARMS = r"(upperarm|lowerarm|hand|thumb_|index_|middle_|ring_|pinky_)"
FRONT = -math.pi / 2     # the character faces -Y


# --- small helpers -------------------------------------------------------------
class Piece:
    """A bmesh plus per-vertex tags, merged into the final mesh at the end."""

    def __init__(self, name, bm, part, tone=0.0, thickness=0.0, rigid=False):
        self.name, self.bm, self.part, self.tone = name, bm, part, tone
        self.thickness, self.rigid = thickness, rigid
        self.pc = {}          # vert -> (u, v) pattern coordinates in metres
        self.edgedist = {}    # vert -> distance to the nearest open edge (metres)


EDGE_CAP = 0.05


def edge_distance(bm):
    """Geodesic-ish distance from every vertex to the nearest boundary edge,
    used for stitched trims and worn edges along hems, cuffs and openings."""
    import scipy.sparse as sps
    from scipy.sparse.csgraph import dijkstra
    bm.verts.index_update()
    boundary = sorted({v.index for e in bm.edges if e.is_boundary for v in e.verts})
    if not boundary:
        return {v: EDGE_CAP for v in bm.verts}
    rows, cols, w = [], [], []
    for e in bm.edges:
        a, b = e.verts
        rows += [a.index, b.index]
        cols += [b.index, a.index]
        w += [e.calc_length()] * 2
    graph = sps.csr_matrix((w, (rows, cols)), shape=(len(bm.verts), len(bm.verts)))
    d = dijkstra(graph, indices=boundary, min_only=True, limit=EDGE_CAP)
    d = np.where(np.isfinite(d), d, EDGE_CAP)
    return {v: float(d[v.index]) for v in bm.verts}


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


def smooth01(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def subdivide(bm, levels=1):
    """Catmull-Clark subdivision (via a temporary object) for a softer, denser surface."""
    obj = ck.bm_to_object(bm, "_subdiv")
    mod = obj.modifiers.new("Subsurf", "SUBSURF")
    mod.levels = mod.render_levels = levels
    mod.boundary_smooth = "PRESERVE_CORNERS"
    out = bmesh.new()
    out.from_object(obj, bpy.context.evaluated_depsgraph_get())
    mesh = obj.data
    bpy.data.objects.remove(obj)
    bpy.data.meshes.remove(mesh)
    bm.free()
    out.normal_update()
    return out


def soften_leather(body, bm):
    """Displace the jerkin body so it reads as heavy, worn leather rather than a
    rigid shell: soft undulation, folds bunching above the waist seam, folds
    radiating from the armpits, and sunken seams with the panels pillowing
    between them."""
    bm.normal_update()
    armpits = [Vector((s * 0.165, 0.0, 1.33)) for s in (1, -1)]
    disp = []
    for v in bm.verts:
        p = v.co
        d = 0.0018 * mnoise.noise(p * 9.0) + 0.0006 * mnoise.noise(p * 26.0 + Vector((3, 1, 7)))
        # compression folds above the waist seam, broken into clusters; flatter
        # in front where the closure straps hold the leather taut
        zr = p.z - WAIST_Z
        band = smooth01((0.17 - zr) / 0.12) * smooth01(zr / 0.02)
        if band > 0:
            cluster = smooth01((mnoise.noise(p * 7.0 + Vector((5, 2, 0))) + 0.15) / 0.5)
            taut = 0.35 if (p.y < 0 and abs(p.x) < 0.08) else 1.0
            wave = math.sin(2 * math.pi * (zr + 0.010 * mnoise.noise(p * 14.0)) / 0.034)
            d += 0.0050 * band * cluster * taut * wave
        # folds fanning down and inwards from each armpit
        for a in armpits:
            dz = p.z - a.z
            h = math.hypot(p.x - a.x, p.y - a.y)
            r = math.hypot(h, dz)
            if r < 0.18:
                fan = smooth01((r - 0.025) / 0.03) * smooth01((0.18 - r) / 0.07) * smooth01((0.04 - dz) / 0.04)
                d += 0.0042 * fan * math.sin(9 * math.atan2(dz, h) + 2.0 * mnoise.noise(p * 5.0))
        # the back hangs in soft vertical folds below the shoulder blades
        if p.y > 0.02:
            hang = smooth01((1.30 - p.z) / 0.15) * smooth01((p.z - WAIST_Z) / 0.05) * smooth01((p.y - 0.02) / 0.04)
            d += 0.0045 * hang * math.sin(2 * math.pi * p.x / 0.085 + 2.5 * mnoise.noise(p * Vector((4, 4, 1.5))))
        # seams between the centre and side panels (same lines as the texture)
        if p.z < 1.47:
            dist = abs(abs(p.x) - (0.085 if p.y < 0 else 0.07))
            d += -0.0016 * math.exp(-(dist / 0.004) ** 2) + 0.0008 * smooth01((dist - 0.004) / 0.03)
        disp.append(d)
    for v, d in zip(bm.verts, disp):
        v.co += v.normal * d
    ck.push_out(body, bm.verts, 0.0085)   # leather (4 mm thick) never touches the skin
    bm.normal_update()


# --- jerkin body ------------------------------------------------------------------
def build_torso(body):
    arm = body.family(r"(upperarm|lowerarm|hand|thumb_|index_|middle_|ring_|pinky_)")
    z = body.co[:, 2]
    mask = (arm < 0.30) & (z > WAIST_Z - 0.03) & (z < ck.NECK_BASE_Z + 0.002)
    bm = ck.shell_from_faces(body, body.region_faces(mask), offset=0.0135, smooth_iters=75,
                             smooth_factor=0.5, min_offset=0.011)
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
    bm = subdivide(bm)
    soften_leather(body, bm)
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
    n = 200
    ring = resample_loop([Vector((p.x, p.y, WAIST_Z + 0.012)) for p in pts], n)
    # order by angle starting at the front, counter-clockwise
    ring.sort(key=lambda p: (angle_of(p, c) - FRONT) % (2 * math.pi))
    angles = np.array([(angle_of(p, c) - FRONT) % (2 * math.pi) for p in ring])
    r_top = np.array([math.hypot(p.x - c[0], p.y - c[1]) for p in ring]) - 0.002

    arms = body.family(ARMS)
    legs = body.co[(body.co[:, 2] < WAIST_Z) & (body.co[:, 2] > HEM_Z - 0.05) & (arms < 0.2)]
    leg_ang = (np.arctan2(legs[:, 1] - c[1], legs[:, 0] - c[0]) - FRONT) % (2 * math.pi)
    leg_r = np.hypot(legs[:, 0] - c[0], legs[:, 1] - c[1])

    zs = np.linspace(WAIST_Z + 0.012, HEM_Z, 25)
    radii = []
    prev = r_top
    for i, zr in enumerate(zs):
        t = i / (len(zs) - 1)
        base = r_top * (1 + 0.18 * t ** 1.2)
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
    # soft drape folds growing towards the hem, and a slightly uneven hem
    folds = np.array([0.55 * math.sin(7 * a + 1.3) + 0.35 * math.sin(12 * a + 0.4) for a in angles])
    hem_wave = np.array([0.004 * math.sin(5 * a + 0.7) + 0.002 * math.sin(11 * a + 2.1) for a in angles])
    for i in range(len(zs)):
        t = i / (len(zs) - 1)
        wobble = np.array([mnoise.noise(Vector((2 * math.cos(a), 2 * math.sin(a), 1.5 * t))) for a in angles])
        radii[i] = radii[i] + (0.0008 + 0.0060 * t ** 1.2) * (folds + 0.3 * wobble)

    def point(j, i, gap=0.0, flare=0.0):
        a = angles[j] + FRONT
        t = i / (len(zs) - 1)
        r = radii[i][j] + flare
        tangent = Vector((-math.sin(a), math.cos(a), 0))
        z = zs[i] + hem_wave[j] * t * t
        return Vector((c[0] + r * math.cos(a), c[1] + r * math.sin(a), z)) + tangent * gap

    # Two panels (left: front->back through +X, right: back->front through -X)
    # so the front and back seams can open into slits near the hem.
    front_slit_top, back_slit_top = 0.875, 0.855
    half = n // 2

    def slit_gap(z, top):
        return 0.0 if z >= top else 0.004 + 0.016 * (top - z) / (top - HEM_Z)

    bm = bmesh.new()
    panels = []
    for cols, start_slit, end_slit in ((range(0, half + 1), front_slit_top, back_slit_top),
                                       (range(half, n + 1), back_slit_top, front_slit_top)):
        cols = list(cols)
        rows = []
        for i in range(len(zs)):
            row = []
            for k, j in enumerate(cols):
                g, flare = 0.0, 0.0
                # the panel edges along a slit part and curl slightly outwards
                for dk, curl in ((0, 1.0), (1, 0.5), (2, 0.2)):
                    if k == dk:
                        g = slit_gap(zs[i], start_slit) if dk == 0 else 0.0
                        flare = 0.35 * curl * slit_gap(zs[i], start_slit) if zs[i] < start_slit else 0.0
                    elif k == len(cols) - 1 - dk:
                        g = -slit_gap(zs[i], end_slit) if dk == 0 else 0.0
                        flare = 0.35 * curl * slit_gap(zs[i], end_slit) if zs[i] < end_slit else 0.0
                row.append(point(j % n, i, g, flare))
            rows.append(row)
        _, vr = loft_grid(rows, closed=False, bm=bm)
        panels.append(vr)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)   # closes the seams above the slits
    orient_outward(bm)
    piece = Piece("Skirt", bm, LEATHER, MAIN, thickness=0.0045)

    # mail skirt hanging inside the leather one, seen through the slits
    mail_rows = []
    for i in range(len(zs)):
        z = zs[i] - 0.012 * i / (len(zs) - 1)
        mail_rows.append([Vector((c[0] + (radii[i][j] - 0.0075) * math.cos(angles[j] + FRONT),
                                  c[1] + (radii[i][j] - 0.0075) * math.sin(angles[j] + FRONT), z))
                          for j in range(n)])
    mbm, mvr = loft_grid(mail_rows)
    orient_outward(mbm)
    fauld = Piece("MailFauld", mbm, MAIL)
    for i, row in enumerate(mvr):
        for j, v in enumerate(row):
            fauld.pc[v] = (angles[j] * float(np.mean(radii[i])), v.co.z)
    return piece, fauld


def build_collar(body, torso_bm):
    """Stand-up collar rising from the jerkin neckline, open at the throat."""
    _, top = bottom_and_top_loops(torso_bm)
    c = np.mean([v.co[:2] for v in top], axis=0)
    loop = sorted([v.co.copy() for v in top], key=lambda p: (angle_of(p, c) - FRONT) % (2 * math.pi))
    n = 160
    base = resample_loop(loop, n)
    base.sort(key=lambda p: (angle_of(p, c) - FRONT) % (2 * math.pi))
    gap = math.radians(26)
    rows = []
    steps = 12
    keep = [p for p in base if gap < (angle_of(p, c) - FRONT) % (2 * math.pi) < 2 * math.pi - gap]
    for i in range(steps + 1):
        t = i / steps
        row = []
        for p in keep:
            a = (angle_of(p, c) - FRONT) % (2 * math.pi)
            frontness = math.cos(a / 2) ** 2          # 1 at the opening, 0 at the back
            height = (0.068 - 0.022 * frontness) * t
            flare = 1.0 + 0.10 * t ** 1.5 + 0.012 * t * mnoise.noise(Vector((p.x * 12, p.y * 12, 0.3)))
            q = Vector((c[0] + (p.x - c[0]) * flare, c[1] + (p.y - c[1]) * flare, p.z - 0.004 + height))
            row.append(q)
        rows.append(row)
    bm, vr = loft_grid(rows, closed=False)
    orient_outward(bm)
    return Piece("Collar", bm, LEATHER, MAIN, thickness=0.004)


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


ACROSS = (-1.0, -0.55, 0.0, 0.55, 1.0)   # strap cross-section columns


def strap_from_rows(name, tone, thickness, rows, along, closed=False):
    """rows: per sample along the strap, [(position, across offset), ...]."""
    bm = bmesh.new()
    piece = Piece(name, bm, LEATHER, tone, thickness=thickness)
    vrows = []
    for row, a in zip(rows, along):
        vr = []
        for pos, k in row:
            v = bm.verts.new(pos)
            piece.pc[v] = (a, k)
            vr.append(v)
        vrows.append(vr)
    pairs = list(zip(vrows, vrows[1:])) + ([(vrows[-1], vrows[0])] if closed else [])
    for a, b in pairs:
        for j in range(len(a) - 1):
            bm.faces.new((a[j], a[j + 1], b[j + 1], b[j]))
    orient_outward(bm)
    return piece


def surface_ribbon(samples, width, lift, name, tone, thickness=0.003, crown=0.0004):
    """Strap lying on a surface. samples: [(location, normal), ...] along its length.
    Five columns across with a slight crown, so it shades like thick leather."""
    centers = [loc + n * lift for loc, n in samples]
    rows, along = [], []
    for i, (c, (loc, n)) in enumerate(zip(centers, samples)):
        t = centers[min(i + 1, len(centers) - 1)] - centers[max(i - 1, 0)]
        side = n.cross(t).normalized()
        along.append(along[-1] + (c - centers[i - 1]).length if i else 0.0)
        rows.append([(c + side * (k * width / 2) + n * (crown * (1 - k * k)), k * width / 2) for k in ACROSS])
    return strap_from_rows(name, tone, thickness, rows, along), centers


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
    placket, _ = surface_ribbon(front_line(bvh, (0.0, V_BOTTOM_Z - 0.004), (0.0, WAIST_Z + 0.004), 60),
                                0.046, 0.0018, "Placket", MAIN, crown=0.0008)
    pieces.append(placket)
    # five buckled closure straps across the front
    # (the shoulder straps cross between the first and second one)
    for z in (1.487, 1.31, 1.22, 1.13, 1.04):
        samples = front_line(bvh, (-0.058, z), (0.062, z), 24)
        strap, centers = surface_ribbon(samples, 0.021, 0.0045, f"Closure_{z}", STRAP)
        pieces.append(strap)
        loc, n = samples[-4]
        pieces.append(buckle(loc + n * 0.0035, n, centers[-1] - centers[-6], 0.027, 0.03))
        pieces.append(rivet(samples[1][0] + samples[1][1] * 0.0048, samples[1][1], r=0.0032))
    return pieces


def build_shoulder_straps(body, torso_bm):
    """The two straps that hold the pauldrons. Each loops over one shoulder
    (under the pauldron), runs diagonally down across the chest, passes under the
    opposite armpit and climbs back across the back, so they cross in an X on
    the chest and again on the back."""
    torso_bvh = BVHTree.FromBMesh(torso_bm)
    body.bm.faces.ensure_lookup_table()
    armw = body.family(ARMS)
    width, n = 0.034, 280
    Y = Vector((0, 1, 0))
    pieces = []
    # (shoulder side, base lift, extra lift over the closures/placket); the +X
    # strap lies on top of the other one where they cross
    for s, lift_base, lift_front in ((1, 0.0072, 0.0063), (-1, 0.0040, 0.0045)):
        # slicing plane through the crossing point, containing the front-back axis
        D = Vector((-s, 0.0, -CROSS_SLOPE)).normalized()
        o = Vector((0.0, -0.01, CROSS_Z)) + D * 0.02

        def clearance_surface(origin, direction):
            """Distance along a ray from inside the torso to where the strap's
            underside may sit: on the jerkin, or clear of the skin/mail."""
            r = 0.0
            hit = body.bvh.ray_cast(origin, direction, 1.0)
            if hit[0] is not None:
                aw = float(np.mean([armw[v.index] for v in body.bm.faces[hit[2]].verts]))
                r = hit[3] + 0.014 + 0.005 * aw
            hit = torso_bvh.ray_cast(origin, direction, 1.0)
            if hit[0] is not None:
                r = max(r, hit[3] + 0.0005)
            return r

        dirs = [D * math.cos(2 * math.pi * j / n) + Y * math.sin(2 * math.pi * j / n) for j in range(n)]
        rs = np.array([clearance_surface(o, d) for d in dirs])
        for _ in range(60):                     # a strap is taut: it bridges hollows
            rs = np.maximum(rs, circular_smooth(rs, k=2, passes=1))
        centers = [o + d * r for d, r in zip(dirs, rs)]

        # cross-section columns, each lifted onto the surface beneath it and kept taut
        frames, heights = [], []
        for j, (cpos, d) in enumerate(zip(centers, dirs)):
            loc, nrm, _, dist = torso_bvh.find_nearest(cpos)
            if loc is None or dist > 0.02:
                loc, nrm, _ = body.nearest(cpos)
            if nrm.dot(d) < 0:
                nrm = -nrm
            t = (centers[(j + 1) % n] - centers[j - 1]).normalized()
            nrm = (nrm - t * nrm.dot(t)).normalized()
            side = nrm.cross(t).normalized()
            frames.append((nrm, side))
            hs = []
            for k in ACROSS:
                q = cpos + side * (k * width / 2)
                start = q + nrm * 0.04
                h = -0.02
                for bvh, extra in ((torso_bvh, 0.0005), (body.bvh, 0.014)):
                    hit = bvh.ray_cast(start, -nrm, 0.08)
                    if hit[0] is not None:
                        h = max(h, 0.04 - hit[3] + extra)
                hs.append(h)
            heights.append(hs)
        # a thick strap stays flat across its width: fit a tilted line over the
        # columns that clears all of them, then keep it taut and smooth along the strap
        heights = np.array(heights)
        ks = np.array(ACROSS)
        tilt = (heights[:, -1] - heights[:, 0]) / 2
        tilt = circular_smooth(tilt, k=6, passes=4)
        base = (heights - tilt[:, None] * ks[None, :]).max(1)
        for _ in range(150):
            base = np.maximum(base, circular_smooth(base, k=3, passes=1))
        base = circular_smooth(base, k=3, passes=2)
        need = (heights - tilt[:, None] * ks[None, :]).max(1)
        base = np.maximum(base, need - 0.0008)
        base = circular_smooth(base, k=2, passes=1)
        heights = base[:, None] + tilt[:, None] * ks[None, :]

        rows, along, tops = [], [], []
        for j, (cpos, (nrm, side)) in enumerate(zip(centers, frames)):
            front = 1.0 - smooth01((abs(cpos.x) - 0.065) / 0.035) if cpos.y < 0 else 0.0
            lift = lift_base + lift_front * front
            row = [(cpos + side * (k * width / 2) + nrm * (heights[j][c] + lift + 0.0005 * (1 - k * k)), k * width / 2)
                   for c, k in enumerate(ACROSS)]
            rows.append(row)
            tops.append(row[2][0])
            along.append(along[-1] + (tops[-1] - tops[-2]).length if j else 0.0)
        pieces.append(strap_from_rows(f"ShoulderStrap_{s}", STRAP, 0.0035, rows, along, closed=True))

        # buckle on the chest on the armpit side, with the tongue riveted behind it
        j = min((j for j in range(n) if centers[j].y < 0), key=lambda j: abs(centers[j].x + s * 0.10))
        nrm, side = frames[j]
        t = tops[(j + 1) % n] - tops[j - 1]
        pieces.append(buckle(tops[j] + nrm * 0.0002, nrm, t, 0.042, 0.046, bar=0.0034))
        step = 1 if tops[(j + 1) % n].z > tops[j].z else -1     # towards the shoulder
        k = (j + step * 13) % n
        for sgn in (-1, 1):
            pieces.append(rivet(tops[k] + frames[k][1] * sgn * 0.009, frames[k][0], r=0.0033))
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
    piece = Piece("WaistBand", bm, LEATHER, STRAP, thickness=0.0035)
    for i, row in enumerate(vr):
        for j, v in enumerate(row):
            piece.pc[v] = (2 * math.pi * j / seg * 0.2, v.co.z)
    # decorative rivets around the band, in pairs, seated on its surface
    band_bvh = BVHTree.FromBMesh(bm)
    rivets = []
    mid = [(a.co + b.co) / 2 for a, b in zip(vr[1], vr[2])]
    for j in range(0, seg, 8):
        out = Vector((mid[j].x - c[0], mid[j].y - c[1], 0)).normalized()
        for dz in (-0.009, 0.013):
            hit = band_bvh.ray_cast(mid[j] + Vector((0, 0, dz)) + out * 0.05, -out, 0.1)
            if hit[0] is not None:
                nrm = hit[1] if hit[1].dot(out) > 0 else -hit[1]
                rivets.append(rivet(hit[0] + nrm * 0.0002, nrm, r=0.0034))
    return piece, rivets


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
    bm = ck.shell_from_faces(body, body.region_faces(mask), offset=0.0085, smooth_iters=12,
                             smooth_factor=0.5, min_offset=0.0065)
    bm = subdivide(bm)
    pieces = []
    for side in (1, -1):
        sh, el, wr = arm_axes(body, side)
        d = (wr - el).normalized()
        part = bm.copy()
        # keep one arm, cut at the wrist
        cut_plane(part, (0, 0, 0), (-side, 0, 0))
        cut_plane(part, wr - d * 0.03, d)
        orient_outward(part)
        quilt_sleeve(body, part, sh, el, wr)
        p = Piece(f"Sleeve_{side}", part, CLOTH)
        cylinder_pc(p, sh, wr)
        pieces.append(p)
    bm.free()
    return pieces


def quilt_sleeve(body, bm, sh, el, wr):
    """Sunken quilting channels running down the arm (same spacing as the texture)
    and the padded sleeve bunching up at the elbow and above the bracer."""
    axis = (wr - sh).normalized()
    ref = Vector((0, -1, 0))
    ref = (ref - axis * ref.dot(axis)).normalized()
    side = axis.cross(ref)
    upper = (el - sh).length
    fore = (wr - el).length
    bm.normal_update()
    disp = []
    for v in bm.verts:
        dv = v.co - sh
        along = dv.dot(axis)
        radial = dv - axis * along
        ang = math.atan2(radial.dot(side), radial.dot(ref))
        channel = math.sin(ang * 0.05 * math.pi / 0.026)
        d = -0.0013 * math.exp(-channel * channel / 0.03) + 0.0006 * min(1.0, abs(channel) * 3)
        # bunching: around the elbow and on the forearm just above the bracer
        e = along - upper
        bunch = math.exp(-(e / 0.06) ** 2) + math.exp(-((e - fore * 0.24) / 0.035) ** 2)
        wave = math.sin(2 * math.pi * along / 0.028 + 1.8 * mnoise.noise(v.co * 9.0) + ang * 0.6)
        d += 0.0030 * bunch * wave + 0.0010 * mnoise.noise(v.co * 14.0)
        disp.append(d)
    for v, d in zip(bm.verts, disp):
        v.co += v.normal * d
    ck.push_out(body, bm.verts, 0.0045)
    bm.normal_update()


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
        seg, ts = 64, np.linspace(0.30, 0.93, 32)
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
        p = Piece(f"Bracer_{side}", bm, LEATHER, BRACER, thickness=0.004)
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
# The cap faces mostly upwards, so the crest sits on its front-facing slope where it
# is seen from the front, like the emblem in the reference.
CREST_ALPHA = math.radians(32)   # crest centre, measured down from the top of the cap
CREST_PHI = math.radians(62)     # ... and around it towards the front
CREST_V = 0.090                  # pattern-space v of the crest centre (see jerkin_layers)
CREST_SIZE = 0.07                # crest height/width on the plate (metres)
PAULDRON_ALPHA = 116     # how far down the arm the lames reach (degrees from the top)
PAULDRON_PHI = 108       # how far the plates wrap around the front and back (degrees)
PAULDRON_SCALE = 0.90    # outward size relative to the original, roomier design
PAULDRON_GAP = 0.008     # air between the garment underneath and the plates


def pieces_bvh(pieces):
    """One BVH over the outer surfaces of several (not yet thickened) pieces."""
    verts, polys = [], []
    for p in pieces:
        p.bm.verts.index_update()
        off = len(verts)
        verts += [v.co.copy() for v in p.bm.verts]
        polys += [[off + v.index for v in f.verts] for f in p.bm.faces]
    return BVHTree.FromPolygons(verts, polys)


def build_pauldrons(body, garment):
    """Laminated pauldrons (cap + four lames). Each is a slightly flattened dome
    about the shoulder joint: PAULDRON_SCALE x the original size over the outside
    of the shoulder, easing out towards the front and back just enough to clear
    the jerkin and straps underneath (`garment` BVH) by PAULDRON_GAP."""
    pieces = []
    amax, pmax = math.radians(PAULDRON_ALPHA), math.radians(PAULDRON_PHI)
    for side in (1, -1):
        sh, el, wr = arm_axes(body, side)
        A = (el - sh).normalized()                 # down the arm
        out = Vector((side, 0, 0))
        e2 = (out - A * out.dot(A)).normalized()   # outward from the shoulder
        e3 = A.cross(e2)
        C = sh + e2 * 0.004 - A * 0.012

        def plain(alpha, phi):
            return -A * math.cos(alpha) + (e2 * math.cos(phi) + e3 * math.sin(phi)) * math.sin(alpha)

        def direction(alpha, phi):
            # lames sweep down towards their front and back ends (the pole stays put)
            return plain(alpha + math.radians(8) * (phi / pmax) ** 2 * (alpha / amax), phi)

        # original sizing: clear the skin + mail by 26 mm in 97% of directions
        req = []
        for alpha in np.radians(np.linspace(5, 115, 24)):
            for phi in np.radians(np.linspace(-115, 115, 24)):
                hit = body.bvh.ray_cast(C, plain(alpha, phi).normalized(), 0.4)
                if hit[0] is not None:
                    req.append(hit[3])
        r_out = (float(np.percentile(req, 97)) + 0.026) * PAULDRON_SCALE

        def blend(alpha, phi):
            s = min(1.0, math.sin(alpha) / math.sin(math.radians(40)))
            return math.sin(phi) ** 2 * s * s * (3 - 2 * s)

        # front/back radius: the least that clears what is underneath (outermost
        # surface within reach of the plate), with the outer radius fixed
        r_side = r_out
        for alpha in np.radians(np.linspace(0, PAULDRON_ALPHA + 3, 24)):
            for phi in np.radians(np.linspace(-PAULDRON_PHI - 3, PAULDRON_PHI + 3, 40)):
                dv = direction(alpha, phi).normalized()
                hit = garment.ray_cast(C + dv * 0.17, -dv, 0.17)
                w = blend(alpha, phi)
                if hit[0] is not None and w > 0.15:
                    need = 0.17 - hit[3] + PAULDRON_GAP
                    r_side = max(r_side, r_out + (need - r_out) / w)
        r_side = min(r_side, r_out / PAULDRON_SCALE * 1.12)

        def radius(alpha, phi):
            return r_out + (r_side - r_out) * blend(alpha, phi)

        print(f"  pauldron {side:+d}: outer radius {r_out:.3f}, front/back {r_side:.3f}")
        lames = [("Cap", 0, 54, 0.012), ("Lame1", 50, 70, 0.009), ("Lame2", 66, 86, 0.006),
                 ("Lame3", 82, 102, 0.003), ("Lame4", 98, PAULDRON_ALPHA, 0.0)]
        r_crest = radius(CREST_ALPHA, side * CREST_PHI)
        for name, a0, a1, extra in lames:
            rows = []
            alphas = np.radians(np.linspace(a0, a1, 14 if name == "Cap" else 7))
            phis = np.radians(np.linspace(-PAULDRON_PHI, PAULDRON_PHI, 48))
            for k, alpha in enumerate(alphas):
                roll = 0.0025 if k == len(alphas) - 1 else 0.0
                rows.append([C + direction(alpha, phi) * (radius(alpha, phi) + extra + roll) for phi in phis])
            bm, vr = loft_grid(rows, closed=False)
            for f in bm.faces:                     # face away from the shoulder joint
                if f.normal.dot(f.calc_center_median() - C) < 0:
                    f.normal_flip()
            bm.normal_update()
            p = Piece(f"Pauldron{name}_{side}", bm, STEEL, thickness=0.002, rigid=True)
            for i, row in enumerate(vr):
                for j, v in enumerate(row):
                    # v is anchored so the crest (centred at v = CREST_V) sits at CREST_ALPHA,
                    # the outward face of the cap; u is arc length around the cap from the
                    # crest centre (positive towards the front on both shoulders)
                    p.pc[v] = ((side * phis[j] - CREST_PHI) * (r_crest + extra) * math.sin(CREST_ALPHA),
                               (alphas[i] - CREST_ALPHA) * (r_crest + extra) + CREST_V)
            pieces.append(p)
            # rivets at both ends of each overlap
            if name != "Cap":
                a = math.radians(a0 + 3)
                for phi in np.radians((-PAULDRON_PHI + 10, PAULDRON_PHI - 10)):
                    pos = C + direction(a, phi) * (radius(a, phi) + extra + 0.0035)
                    pieces.append(rivet(pos, direction(a, phi)))
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
    torso = Piece("Torso", torso_bm, LEATHER, MAIN, thickness=0.004)
    skirt, mail_fauld = build_skirt(body, torso_bm)
    band, band_rivets = build_waist_band(torso_bm, skirt.bm)
    collar = build_collar(body, torso_bm)
    pieces = [torso, skirt, mail_fauld, collar, build_mail_gorget(body), band]
    pieces += band_rivets
    pieces += build_front_details(torso_bm)
    straps = build_shoulder_straps(body, torso_bm)
    sleeves = build_sleeves(body)
    mail_sleeves = build_mail_sleeves(body)
    pieces += straps + sleeves + mail_sleeves
    pieces += build_bracers(body)
    under = [p for p in [torso] + straps + sleeves + mail_sleeves if not p.rigid]
    pieces += build_pauldrons(body, pieces_bvh(under))
    return pieces


def thicken(piece):
    """Give a surface piece real thickness: an inner copy pushed in along the
    vertex normals plus a rounded, rolled rim along the open edges (hems, cuffs,
    slits, strap edges) so the edges catch the light like thick leather."""
    bm = piece.bm
    bm.normal_update()
    outer_faces = list(bm.faces)
    boundary = [e for e in bm.edges if e.is_boundary]
    outward = {}
    for e in boundary:
        f = e.link_faces[0]
        a, b = e.verts
        d = (b.co - a.co).cross(f.normal)
        if d.dot((a.co + b.co) / 2 - f.calc_center_median()) < 0:
            d = -d
        d = d.normalized()
        for v in (a, b):
            outward[v] = outward.get(v, Vector()) + d
    inner, mid = {}, {}
    for v in list(bm.verts):
        n = v.normal if v.normal.length > 1e-6 else Vector((0, 0, 0))
        inner[v] = bm.verts.new(v.co - n * piece.thickness)
        if v in outward:
            o = outward[v].normalized() if outward[v].length > 1e-9 else Vector()
            mid[v] = bm.verts.new(v.co - n * piece.thickness * 0.5 + o * piece.thickness * 0.32)
        for w in (inner[v], mid.get(v)):
            if w is None:
                continue
            if v in piece.pc:
                piece.pc[w] = piece.pc[v]
            piece.edgedist[w] = 0.0 if w is mid.get(v) else piece.edgedist.get(v, EDGE_CAP)
    for f in outer_faces:
        bm.faces.new([inner[v] for v in reversed(f.verts)])
    for e in boundary:
        f = e.link_faces[0]
        a, b = e.verts
        # keep the rim winding consistent with the outer face that owns the edge
        loop = next(l for l in f.loops if l.edge is e)
        if loop.vert is not a:
            a, b = b, a
        bm.faces.new((b, a, mid[a], mid[b]))
        bm.faces.new((mid[b], mid[a], inner[a], inner[b]))
    bm.normal_update()


def pieces_to_object(pieces, name):
    """Merge pieces into one mesh object with part/tone/pc/rigid_id attributes."""
    bm = bmesh.new()
    part_l, tone_l, pc_l, rid_l, ed_l = [], [], [], [], []
    for rid, p in enumerate(pieces):
        src = p.bm
        p.edgedist = edge_distance(src)
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
            ed_l.append(p.edgedist.get(v, EDGE_CAP))
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
    ck.set_attr(obj, "edgedist", ed_l)
    return obj


# --- texture recipe -----------------------------------------------------------------------
# Built from the aaa-materials presets (plugins/aaa-materials) plus the jerkin's own
# construction details. Leather dyes are sRGB picks matched to the reference.
CREST = OUT_DIR / "Source" / "T_LeatherJerkin_CrestMask.png"
LEATHER_MAIN = "#4f3b2c"     # jerkin body, skirt, collar
LEATHER_STRAP = "#3e2d21"    # straps, waist band, placket closures
LEATHER_BRACER = "#564231"   # bracers
WOOL = "#4a4632"             # gambeson sleeves
STITCH = "#7d6a52"           # waxed linen thread


def jerkin_layers(c):
    """part 0 leather (tone 0 jerkin, 1 straps/band, 2 bracers), 1 steel,
    2 quilted wool gambeson, 3 riveted mail."""
    g = c.g
    srgb = aaa.engine.srgb
    tone, part, ed = c.attr("tone"), c.attr("part"), c.attr("edgedist")
    pc_vec = c.attr_vec("pc")
    pc = g.xyz(pc_vec)
    P = g.xyz(c.P)

    def near(x, target, width):
        return g.remap(g.math("ABSOLUTE", g.sub(x, target)), 0.0, width, 1.0, 0.0)

    def dashes(x, period, duty=0.55):
        return g.math("GREATER_THAN", g.math("SINE", g.mul(x, 2 * math.pi / period)), 1.0 - 2.0 * duty)

    def stitch_dots(scale):
        """Dotted running stitches that follow any band (3D cells, so no UV needed)."""
        return g.remap(g.voronoi(g.mapping(c.P, (1 / scale,) * 3), 1.0), 0.16, 0.30, 1.0, 0.0)

    # ---- leather: one preset, three dyes -----------------------------------------------
    is_strap, is_bracer = c.is_value(tone, STRAP), c.is_value(tone, BRACER)
    is_body = g.mul(g.math("LESS_THAN", P["Z"], 1.47), c.is_value(tone, MAIN))
    ax = g.math("ABSOLUTE", P["X"])
    seam_x = g.mix_float(0.085, 0.07, g.math("GREATER_THAN", P["Y"], 0.0))
    side_panel = g.mul(g.remap(g.sub(ax, seam_x), 0.0, 0.002), is_body)
    seam = g.mul(near(ax, seam_x, 0.0012), is_body)
    piping = g.mul(near(ax, seam_x, 0.0045), is_body)
    seam_stitch = g.mul(g.mul(g.maxf(near(ax, g.add(seam_x, 0.0065), 0.0008),
                                     near(ax, g.sub(seam_x, 0.0065), 0.0008)), dashes(P["Z"], 0.0065)), is_body)
    edge_stitch = g.mul(near(ed, g.mix_float(0.0068, 0.0032, is_strap), 0.0008), stitch_dots(0.0042))
    stitch = g.maxf(seam_stitch, edge_stitch)
    burnish = g.remap(ed, 0.0, 0.0035, 1.0, 0.0)               # dark, glossy cut edges
    rim_wear = g.mul(g.remap(ed, 0.0, 0.007, 1.0, 0.0), aaa.weathering.breakup(c, 40.0, 0.4, 0.62))

    def tailor(s, color, side_dark):
        """Seams, piping, stitching and burnished edges on a leather surface."""
        col = srgb(color)
        dark = tuple(v * 0.45 for v in col[:3]) + (1.0,)
        light = tuple(min(1.0, v * 2.0) for v in col[:3]) + (1.0,)
        base = s.base
        if side_dark:
            base = g.mix_color(base, tuple(v * 0.6 for v in col[:3]) + (1.0,), g.mul(side_panel, 0.7))
        base = g.mix_color(base, light, g.mul(rim_wear, 0.5))
        base = g.mix_color(base, dark, g.mul(burnish, 0.6))
        base = g.mix_color(base, dark, seam)
        base = g.mix_color(base, srgb(STITCH), stitch)
        rough = g.mix_float(s.rough, 0.38, g.mul(burnish, 0.7))
        rough = g.mix_float(rough, 0.82, stitch)
        height = g.add(s.height, g.add(g.mul(piping, 1.2), g.mul(stitch, 0.7)))
        height = g.sub(height, g.add(g.mul(seam, 1.5), g.mul(near(ed, 0.0012, 0.0012), 0.8)))
        return aaa.engine.Surface(base, rough, s.metal, height, s.bump)

    main = tailor(L.leather(c, color=LEATHER_MAIN, wear=0.55, dirt=0.55, grain_amount=0.4), LEATHER_MAIN, True)
    strap = tailor(L.leather(c, color=LEATHER_STRAP, wear=0.6, dirt=0.5, grain_amount=0.35, scale=1.15),
                   LEATHER_STRAP, False)
    bracer = tailor(L.leather(c, color=LEATHER_BRACER, wear=0.65, dirt=0.5, grain_amount=0.4), LEATHER_BRACER, False)

    # ---- blackened steel: pauldrons, buckles, rivets ----------------------------------------
    # Plates are only 2 mm thick, so the AO-based edge mask would flag them everywhere;
    # their worn edges come from the distance to the plate rim instead.
    steel_edge = g.mul(g.remap(ed, 0.0, 0.005, 1.0, 0.0), aaa.weathering.breakup(c, 30.0, 0.35, 0.6))
    steel = L.steel_blackened(c.with_masks(edge=steel_edge), wear=0.55, dirt=0.45, rust=0.3)
    crest_uv = g.node("ShaderNodeCombineXYZ", {
        "X": g.add(g.math("DIVIDE", pc["X"], CREST_SIZE), 0.5),
        "Y": g.math("DIVIDE", g.sub(CREST_V + CREST_SIZE / 2, pc["Y"]), CREST_SIZE)}).outputs["Vector"]
    crest_img = bpy.data.images.load(str(CREST), check_existing=True)
    crest_img.colorspace_settings.name = "Non-Color"
    crest = g.mul(g.image(crest_img, crest_uv, "CLIP").outputs["Color"], c.is_value(part, STEEL))
    outline = g.clamp01(g.mul(g.mul(crest, g.inv(crest)), 4.0))
    etched = g.mul(near(ed, 0.009, 0.0011), c.is_value(part, STEEL))      # etched border
    inlay = g.maxf(outline, etched)
    sbase = g.mix_color(steel.base, srgb("#c4c6c9"), g.mul(crest, 0.9))     # polished embossed crest
    sbase = g.mix_color(sbase, srgb("#2b2826"), inlay)                      # blackened engraving
    srough = g.mix_float(g.mix_float(steel.rough, 0.22, crest), 0.6, inlay)
    smetal = g.mul(g.maxf(steel.metal, g.smooth(crest, 0.4, 0.6)), g.inv(inlay))
    sheight = g.sub(g.add(steel.height, g.mul(crest, 1.6)), g.mul(inlay, 0.8))
    steel = aaa.engine.Surface(sbase, srough, smetal, sheight, steel.bump)

    # ---- quilted wool gambeson and riveted mail (2D pattern coords from the build) ---------
    cloth = L.wool(c, color=WOOL, quilted=True, channel=0.026, channel_axis="X", P2=pc_vec, wear=0.4, dirt=0.55)
    mail = L.chainmail(c, ring=0.0085, rust=0.45, dirt=0.55, oxide=0.7, P2=pc_vec)

    return [(None, main), (is_strap, strap), (is_bracer, bracer),
            (c.is_value(part, STEEL), steel), (c.is_value(part, CLOTH), cloth), (c.is_value(part, MAIL), mail)]


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
    aaa.engine.assign_baked_material(garment, OUT_DIR / "Textures", "LeatherJerkin")
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
        tex_dir = OUT_DIR / "Textures"
        if not args.no_bake or not (tex_dir / "T_LeatherJerkin_BaseColor.png").exists():
            # 2.5 cm contact shadows: enough for straps and folds, without the
            # hidden inner layers of the garment going black
            aaa.engine.bake(obj, tex_dir, "LeatherJerkin", jerkin_layers, res=args.res,
                            ao_distance=0.025, export_height=False)
        aaa.engine.assign_baked_material(obj, tex_dir, "LeatherJerkin")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT_DIR / f"{ASSET_NAME}.blend"), compress=True)
        ck.export_skeletal(body, obj, OUT_DIR / f"{ASSET_NAME}.fbx")
        print("Exported", OUT_DIR / f"{ASSET_NAME}.fbx")
    if args.render:
        render_previews(args.body)
    return body, obj


if __name__ == "__main__":
    main()
