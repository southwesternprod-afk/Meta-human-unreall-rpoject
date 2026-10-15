"""Shared helpers for building MetaHuman clothing in headless Blender.

The garment scripts fit their pieces to a MetaHuman body exported from
Unreal (SKM_*_BodyMesh.fbx), skin them to the same skeleton and export an
FBX that imports onto the character's existing skeleton.

Coordinates while building are metres, Z up, character facing -Y (Blender's
FBX import of an Unreal skeletal mesh).
"""

import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import bpy  # must come before bmesh/mathutils when running as a module
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Tools" / "ThirdParty" / "RobustSkinWeights"))

# Neck proxy: the MetaHuman body mesh stops at a wide neckline and the neck
# itself belongs to the head mesh, so the hole is closed with an estimated neck.
NECK_RX, NECK_RY = 0.062, 0.068
NECK_CENTER_Y = 0.022
NECK_BASE_Z = 1.585
NECK_TOP_Z = 1.665


class Body:
    """The MetaHuman body: welded world-space mesh, skin weights and skeleton."""

    def __init__(self, fbx_path):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=str(fbx_path))
        self.armature = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
        meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        self.mesh_obj = next(o for o in meshes if o.name.endswith("LOD0"))
        for o in meshes:
            if o is not self.mesh_obj:
                bpy.data.objects.remove(o)
        self.group_names = [g.name for g in self.mesh_obj.vertex_groups]

        bm = bmesh.new()
        bm.from_mesh(self.mesh_obj.data)
        bm.transform(self.mesh_obj.matrix_world)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
        # Round-trip through a mesh so the welded bmesh has no freed slots;
        # otherwise vertices added later reuse them and shift the vertex order.
        tmp = bpy.data.meshes.new("_welded")
        bm.to_mesh(tmp)
        bm.free()
        bm = bmesh.new()
        bm.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
        bm.verts.index_update()
        deform = bm.verts.layers.deform.active
        W = np.zeros((len(bm.verts), len(self.group_names)), np.float32)
        for v in bm.verts:
            for gi, w in v[deform].items():
                W[v.index, gi] = w
        self._add_neck(bm, W)
        bm.verts.index_update()
        bm.faces.index_update()
        bm.normal_update()
        self.bm = bm
        self.co = np.array([v.co[:] for v in bm.verts])
        self.normals = np.array([v.normal[:] for v in bm.verts])
        self.W = self._weights
        self.bvh = BVHTree.FromBMesh(bm)

    # -- skeleton --------------------------------------------------------------
    def bone_head(self, name):
        return np.array((self.armature.matrix_world @ self.armature.data.bones[name].head_local)[:])

    def family(self, pattern):
        idx = [i for i, n in enumerate(self.group_names) if re.match(pattern, n)]
        return self.W[:, idx].sum(1)

    def gid(self, name):
        return self.group_names.index(name)

    # -- neck proxy ------------------------------------------------------------
    def _boundary_loop(self, bm):
        edges = [e for e in bm.edges if e.is_boundary]
        adj = defaultdict(list)
        for e in edges:
            a, b = e.verts
            adj[a].append(b)
            adj[b].append(a)
        start = max(adj, key=lambda v: v.co.z)
        loop, prev, cur = [start], None, start
        while True:
            nxt = next(v for v in adj[cur] if v is not prev)
            if nxt is start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        # orient counter-clockwise seen from above
        pts = np.array([v.co[:2] for v in loop])
        area = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
        return loop if area > 0 else loop[::-1]

    def _add_neck(self, bm, W):
        """Loft from the body's neckline up to an estimated neck tube.
        New vertices blend the neckline weights into neck_01."""
        loop = self._boundary_loop(bm)
        if "neck_01" not in self.group_names:   # the body is not skinned to the neck
            self.group_names.append("neck_01")
            W = np.hstack([W, np.zeros((len(W), 1), W.dtype)])
        neck_gid = self.group_names.index("neck_01")
        center = Vector((0.0, NECK_CENTER_Y, 0.0))
        rows = [loop]
        weights = [W[[v.index for v in loop]]]
        steps = 7
        for k in range(1, steps + 1):
            t = k / steps
            row = []
            for v in loop:
                d = Vector((v.co.x - center.x, v.co.y - center.y, 0.0))
                a = math.atan2(d.y, d.x)
                neck = Vector((center.x + NECK_RX * math.cos(a), center.y + NECK_RY * math.sin(a), NECK_BASE_Z))
                p = v.co.lerp(neck, t)
                p.z = v.co.z + (NECK_BASE_Z - v.co.z) * t * t   # trapezius: rises steeply near the neck
                row.append(bm.verts.new(p))
            rows.append(row)
            w = weights[0] * (1 - t)
            w[:, neck_gid] += t
            weights.append(w)
        for z in np.linspace(NECK_BASE_Z, NECK_TOP_Z, 5)[1:]:
            row = [bm.verts.new((r.co.x, r.co.y, z)) for r in rows[-1]]
            rows.append(row)
            w = np.zeros_like(weights[0])
            w[:, neck_gid] = 1.0
            weights.append(w)
        n = len(loop)
        for a, b in zip(rows, rows[1:]):
            for j in range(n):
                k = (j + 1) % n
                bm.faces.new((a[j], a[k], b[k], b[j]))
        self._weights = np.concatenate([W] + weights[1:], axis=0)

    # -- queries ---------------------------------------------------------------
    def nearest(self, p):
        loc, normal, idx, dist = self.bvh.find_nearest(Vector(p))
        return loc, normal, dist

    def region_faces(self, vert_mask):
        return [f for f in self.bm.faces if all(vert_mask[v.index] for v in f.verts)]


# --- Garment shells ------------------------------------------------------------
def shell_from_faces(body, faces, offset, smooth_iters=12, smooth_factor=0.5, min_offset=None):
    """Copy body faces into a new bmesh, push them out along the body normal
    and relax them so the garment drapes over muscle detail instead of
    hugging it. Vertices never end up closer to the body than `min_offset`."""
    min_offset = offset * 0.8 if min_offset is None else min_offset
    bm = bmesh.new()
    vmap = {}
    for f in faces:
        for v in f.verts:
            if v.index not in vmap:
                vmap[v.index] = bm.verts.new(v.co + v.normal * offset)
        bm.faces.new([vmap[v.index] for v in f.verts])
    bm.normal_update()
    relax(body, bm, smooth_iters, smooth_factor, min_offset)
    return bm


def relax(body, bm, iters, factor, min_offset):
    boundary = {v for e in bm.edges if e.is_boundary for v in e.verts}
    for _ in range(iters):
        new = {}
        for v in bm.verts:
            if v in boundary:
                nb = [e.other_vert(v) for e in v.link_edges if e.is_boundary]
            else:
                nb = [e.other_vert(v) for e in v.link_edges]
            if nb:
                avg = sum((u.co for u in nb), Vector()) / len(nb)
                new[v] = v.co.lerp(avg, factor)
        for v, co in new.items():
            v.co = co
        push_out(body, bm.verts, min_offset)
    bm.normal_update()


def push_out(body, verts, min_offset):
    for v in verts:
        loc, normal, dist = body.nearest(v.co)
        if loc is None:
            continue
        d = (v.co - loc).dot(normal)
        if d < min_offset:
            v.co = v.co + normal * (min_offset - d)


def bm_to_object(bm, name):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def set_attr(obj, name, values, domain="POINT", kind="FLOAT"):
    attr = obj.data.attributes.get(name) or obj.data.attributes.new(name, kind, domain)
    if kind == "FLOAT2":
        attr.data.foreach_set("vector", np.asarray(values, np.float32).ravel())
    else:
        attr.data.foreach_set("value", np.asarray(values, np.float32))


def ordered_boundary_loops(bm):
    edges = [e for e in bm.edges if e.is_boundary]
    adj = defaultdict(list)
    for e in edges:
        a, b = e.verts
        adj[a].append(b)
        adj[b].append(a)
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [v for v in adj[cur] if v is not prev]
            if not nxt or nxt[0] is start:
                break
            prev, cur = cur, nxt[0]
            if cur in seen:
                break
            loop.append(cur)
            seen.add(cur)
        loops.append(loop)
    return loops


# --- Skinning --------------------------------------------------------------------
def transfer_weights(body, obj, max_dist=0.04, max_angle=35.0, max_influences=8):
    """Robust skin-weight transfer (Abdrashitov et al. 2023) from the body to a
    garment object; returns a (verts x bones) weight matrix."""
    import igl
    from robust_weights import find_matches_closest_surface, inpaint, smooth

    tris = [f for f in body.bm.faces]
    V1 = body.co.astype(np.float64)
    F1 = []
    for f in tris:
        ids = [v.index for v in f.verts]
        for k in range(1, len(ids) - 1):
            F1.append((ids[0], ids[k], ids[k + 1]))
    F1 = np.array(F1, np.int64)
    N1 = igl.per_vertex_normals(V1, F1)

    me = obj.data
    me.calc_loop_triangles()
    V2 = np.array([v.co[:] for v in me.vertices], np.float64)
    F2 = np.array([t.vertices[:] for t in me.loop_triangles], np.int64)
    N2 = igl.per_vertex_normals(V2, F2)

    used = np.where(body.W.max(0) > 0)[0]
    W1 = body.W[:, used].astype(np.float64)
    matched, Wi = find_matches_closest_surface(V1, F1, N1, V2, F2, N2, W1, max_dist ** 2, max_angle)

    # Inpainting needs at least one matched vertex per separate piece; pieces
    # with none (e.g. a buckle standing off the surface) take closest-point weights.
    import scipy.sparse as sps
    from scipy.sparse.csgraph import connected_components
    rows = np.concatenate([F2[:, 0], F2[:, 1], F2[:, 2]])
    cols = np.concatenate([F2[:, 1], F2[:, 2], F2[:, 0]])
    adj = sps.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(V2), len(V2)))
    n_comp, labels = connected_components(adj, directed=False)
    for comp in range(n_comp):
        ids = labels == comp
        if not matched[ids].any():
            matched[ids] = True
    # Zero-area triangles (slivers from cuts, collapsed tips) break the cotangent
    # Laplacian, so the solve uses only proper triangles; vertices they leave
    # unconnected keep their closest-point weights.
    e1 = V2[F2[:, 1]] - V2[F2[:, 0]]
    e2 = V2[F2[:, 2]] - V2[F2[:, 0]]
    good = np.linalg.norm(np.cross(e1, e2), axis=1) > 1e-10
    Fs = F2[good]
    orphan = np.ones(len(V2), bool)
    orphan[np.unique(Fs)] = False
    matched = matched | orphan
    Wp, ok = inpaint(V2, Fs, Wi, matched)
    if not ok:
        raise RuntimeError("weight inpainting failed")
    Ws, _ = smooth(V2, Fs, Wp, matched, max_dist, num_smooth_iter_steps=10, smooth_alpha=0.2)
    broken = ~np.isfinite(Ws).all(1)
    Ws[broken] = Wi[broken]
    if broken.any():
        print(f"  weights: {int(broken.sum())} verts fell back to closest-point weights")
    full = np.zeros((len(V2), len(body.group_names)))
    full[:, used] = np.clip(Ws, 0, None)
    print(f"  weights: {int(matched.sum())}/{len(matched)} verts matched directly, rest inpainted")
    return limit_influences(full, max_influences)


def limit_influences(W, k):
    W = W.copy()
    if W.shape[1] > k:
        cut = np.argpartition(-W, k, axis=1)[:, k:]
        np.put_along_axis(W, cut, 0.0, axis=1)
    W[W < 0.01] = 0.0
    s = W.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return W / s


def apply_weights(body, obj, W):
    obj.vertex_groups.clear()
    for gi, name in enumerate(body.group_names):
        col = W[:, gi]
        nz = np.nonzero(col)[0]
        if len(nz) == 0:
            continue
        vg = obj.vertex_groups.new(name=name)
        for vi in nz:
            vg.add([int(vi)], float(col[vi]), "REPLACE")


# --- Export ----------------------------------------------------------------------
def export_skeletal(body, garment, path):
    """Export the garment skinned to the untouched MetaHuman skeleton.

    The FBX import put the armature under a 0.01-scaled null (Unreal's
    centimetres); the garment is built in metres. The garment is moved into
    the armature's centimetre space so the armature exports exactly as it
    came in."""
    arm = body.armature
    to_arm = arm.matrix_world.inverted()
    garment.data.transform(to_arm)
    garment.matrix_world = Matrix.Identity(4)
    garment.parent = arm
    garment.matrix_parent_inverse = Matrix.Identity(4)
    mod = garment.modifiers.new("Armature", "ARMATURE")
    mod.object = arm

    root_null = arm.parent
    arm.parent = None
    arm.matrix_world = Matrix.Identity(4)
    if root_null is not None:
        bpy.data.objects.remove(root_null)
    body.mesh_obj.parent = None
    bpy.data.objects.remove(body.mesh_obj)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 0.01

    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    garment.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_scene.fbx(
        filepath=str(path),
        use_selection=True,
        object_types={"ARMATURE", "MESH"},
        apply_scale_options="FBX_SCALE_UNITS",
        add_leaf_bones=False,
        use_armature_deform_only=False,
        bake_anim=False,
        mesh_smooth_type="FACE",
        path_mode="STRIP",
    )
