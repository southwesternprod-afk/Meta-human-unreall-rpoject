"""Procedural PBR material engine for headless Blender (bpy).

A material is a list of layers ``[(mask, Surface), ...]`` built from shader
nodes. The engine bakes it into a game-ready texture set:

    T_<Name>_BaseColor.png      sRGB
    T_<Name>_Normal.png         tangent space, DirectX (-Y) - Unreal
    T_<Name>_Normal_OpenGL.png  tangent space, OpenGL (+Y) - Blender / glTF / Unity
    T_<Name>_ORM.png            R ambient occlusion, G roughness, B metallic
    T_<Name>_Height.png         (tileable sets) 0.5 = surface, for parallax / displacement

Two kinds of bake:
  * asset bake - textures for one unwrapped mesh. Edge, cavity and AO masks are
    baked from the geometry first, so wear and grime follow the real shape.
  * tileable bake - a seamless square texture for any surface (floors, cloth,
    leather panels...). Baked on a plane and made seamless in image space.
"""

from dataclasses import dataclass
from pathlib import Path

import bpy  # noqa: F401  (must be imported before mathutils/bmesh users)
import numpy as np

from .padding import empty_mask, pad

BAKE_MARGIN = 16


def texture_names(name):
    return {
        "base": (f"T_{name}_BaseColor", "sRGB"),
        "orm": (f"T_{name}_ORM", "Non-Color"),
        "normal_gl": (f"T_{name}_Normal_OpenGL", "Non-Color"),
        "normal_dx": (f"T_{name}_Normal", "Non-Color"),
        "height": (f"T_{name}_Height", "Non-Color"),
    }


def rgb(r, g, b):
    return (r, g, b, 1.0)


def srgb(hex_or_tuple):
    """Colour picked in sRGB (0-255 tuple or '#rrggbb') -> linear RGBA for nodes."""
    if isinstance(hex_or_tuple, str):
        h = hex_or_tuple.lstrip("#")
        hex_or_tuple = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    out = []
    for v in hex_or_tuple[:3]:
        c = v / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return (*out, 1.0)


class Graph:
    """Small helper for wiring shader node trees in code."""

    def __init__(self, mat):
        self.nt = mat.node_tree
        self.nodes = self.nt.nodes
        self.links = self.nt.links

    def node(self, kind, inputs=None, **props):
        nd = self.nodes.new(kind)
        for key, value in props.items():
            setattr(nd, key, value)
        for key, value in (inputs or {}).items():
            self.set(nd.inputs[key], value)
        return nd

    def set(self, socket, value):
        if isinstance(value, bpy.types.NodeSocket):
            self.links.new(value, socket)
        else:
            socket.default_value = value

    def value(self, v):
        nd = self.nodes.new("ShaderNodeValue")
        nd.outputs[0].default_value = v
        return nd.outputs[0]

    def math(self, op, a, b=0.0, clamp=False):
        nd = self.nodes.new("ShaderNodeMath")
        nd.operation = op
        nd.use_clamp = clamp
        self.set(nd.inputs[0], a)
        self.set(nd.inputs[1], b)
        return nd.outputs[0]

    # shorthands
    def add(self, a, b):
        return self.math("ADD", a, b)

    def mul(self, a, b):
        return self.math("MULTIPLY", a, b)

    def sub(self, a, b):
        return self.math("SUBTRACT", a, b)

    def inv(self, a):
        return self.math("SUBTRACT", 1.0, a)

    def maxf(self, a, b):
        return self.math("MAXIMUM", a, b)

    def minf(self, a, b):
        return self.math("MINIMUM", a, b)

    def clamp01(self, a):
        return self.math("ADD", a, 0.0, clamp=True)

    def remap(self, value, a, b, to_a=0.0, to_b=1.0):
        """Linear remap of a..b onto to_a..to_b, clamped."""
        nd = self.node("ShaderNodeMapRange", {"Value": value, "From Min": a, "From Max": b,
                                              "To Min": to_a, "To Max": to_b})
        nd.clamp = True
        return nd.outputs["Result"]

    def smooth(self, value, a, b):
        """Smoothstep remap of a..b onto 0..1."""
        nd = self.node("ShaderNodeMapRange", {"Value": value, "From Min": a, "From Max": b},
                       interpolation_type="SMOOTHSTEP")
        nd.clamp = True
        return nd.outputs["Result"]

    def _mix(self, kind, a, b, t):
        nd = self.nodes.new("ShaderNodeMix")
        nd.data_type = kind
        nd.clamp_factor = True
        suffix = {"FLOAT": "Float", "RGBA": "Color", "VECTOR": "Vector"}[kind]
        self.set(self._sock(nd.inputs, "Factor_Float"), t)
        self.set(self._sock(nd.inputs, f"A_{suffix}"), a)
        self.set(self._sock(nd.inputs, f"B_{suffix}"), b)
        return self._sock(nd.outputs, f"Result_{suffix}")

    def mix_float(self, a, b, t):
        return self._mix("FLOAT", a, b, t)

    def mix_color(self, a, b, t):
        return self._mix("RGBA", a, b, t)

    def mix_vector(self, a, b, t):
        return self._mix("VECTOR", a, b, t)

    def tint(self, color, factor):
        """Multiply a colour by an RGB(A) factor."""
        nd = self.nodes.new("ShaderNodeMix")
        nd.data_type = "RGBA"
        nd.blend_type = "MULTIPLY"
        self.set(self._sock(nd.inputs, "Factor_Float"), 1.0)
        self.set(self._sock(nd.inputs, "A_Color"), color)
        self.set(self._sock(nd.inputs, "B_Color"), factor)
        return self._sock(nd.outputs, "Result_Color")

    @staticmethod
    def _sock(sockets, identifier):
        return next(s for s in sockets if s.identifier == identifier)

    def xyz(self, vec):
        return self.node("ShaderNodeSeparateXYZ", {"Vector": vec}).outputs

    def mapping(self, vec, scale=(1, 1, 1), rot=(0, 0, 0), loc=(0, 0, 0)):
        return self.node("ShaderNodeMapping", {"Vector": vec, "Scale": scale, "Rotation": rot,
                                               "Location": loc}).outputs["Vector"]

    def noise(self, vec, scale, detail=4.0, rough=0.5, distortion=0.0):
        return self.node("ShaderNodeTexNoise", {"Vector": vec, "Scale": scale, "Detail": detail,
                                                "Roughness": rough, "Distortion": distortion}).outputs["Fac"]

    def warp(self, vec, scale, amount):
        """Domain warp: offset `vec` by low-frequency noise (organic, non-grid shapes)."""
        col = self.node("ShaderNodeTexNoise", {"Vector": vec, "Scale": scale, "Detail": 2.0}).outputs["Color"]
        off = self.node("ShaderNodeVectorMath", {0: col, 1: (0.5, 0.5, 0.5)}, operation="SUBTRACT").outputs[0]
        off = self.node("ShaderNodeVectorMath", {0: off, "Scale": amount}, operation="SCALE").outputs[0]
        return self.node("ShaderNodeVectorMath", {0: vec, 1: off}, operation="ADD").outputs[0]

    def voronoi(self, vec, scale, feature="F1", randomness=1.0, dims="3D"):
        return self.node("ShaderNodeTexVoronoi", {"Vector": vec, "Scale": scale, "Randomness": randomness},
                         feature=feature, voronoi_dimensions=dims).outputs["Distance"]

    def wave(self, vec, scale, distortion=0.0, detail=2.0, kind="BANDS", direction="X", profile="SIN"):
        return self.node("ShaderNodeTexWave", {"Vector": vec, "Scale": scale, "Distortion": distortion,
                                               "Detail": detail},
                         wave_type=kind, bands_direction=direction, wave_profile=profile).outputs["Fac"]

    def image(self, img, vec=None, extension="REPEAT"):
        nd = self.node("ShaderNodeTexImage", image=img, extension=extension)
        if vec is not None:
            self.set(nd.inputs["Vector"], vec)
        return nd

    def ao(self, distance, inside=False, samples=8):
        return self.node("ShaderNodeAmbientOcclusion", {"Distance": distance},
                         inside=inside, only_local=True, samples=samples).outputs["AO"]


def new_image(name, colorspace, res, float_buffer=False):
    img = bpy.data.images.new(name, res, res, alpha=False, float_buffer=float_buffer)
    img.colorspace_settings.name = colorspace
    return img


@dataclass
class Surface:
    """One material layer. ``height`` is unitless (about 0..1); ``bump`` is the
    real-world depth in metres that a height change of 1.0 represents."""
    base: object
    rough: object
    metal: object
    height: object
    bump: float


class Context:
    """Inputs shared by every layer of a material.

    P      object-space position in metres (use for most patterns)
    UV     UV coordinates (use for patterns that must follow a layout: weave, rings)
    N      object-space normal
    edge   convex edges (0..1)     - baked from the mesh, 0 in tileable mode
    cavity concave creases (0..1)  - baked from the mesh, 0 in tileable mode
    ao     ambient occlusion (1 = open)
    up     how much the surface faces world up (0..1) - dust / rain settle here
    attr(name)  any float mesh attribute (e.g. a "part" id written by a build script)
    """

    def __init__(self, g, name, res, tileable=False, mask_scale=1.0, ao_distance=None):
        self.g, self.name, self.res, self.tileable = g, name, res, tileable
        tc = g.node("ShaderNodeTexCoord")
        self.P = tc.outputs["Object"]
        self.UV = tc.outputs["UV"]
        self.N = tc.outputs["Normal"]
        geo = g.node("ShaderNodeNewGeometry")
        if tileable:
            # A flat tile has no real edges or creases, so wear and grime come
            # from soft "grunge" masks instead (what tileable materials ship with).
            self.masks_emit = None
            self.masks_img = None
            P = self.P
            self.edge = g.mul(g.remap(g.noise(g.mapping(P, loc=(3.1, 7.7, 1.3)), 10.0, 6.0, 0.62), 0.64, 0.76),
                              g.remap(g.noise(P, 60.0, 4.0, 0.6), 0.4, 0.6))
            self.cavity = g.remap(g.noise(g.mapping(P, loc=(9.4, 2.2, 5.6)), 8.0, 6.0, 0.6), 0.6, 0.74)
            self.ao = g.value(1.0)
            self.up = g.value(0.25)
        else:
            self.up = g.remap(g.xyz(geo.outputs["Normal"])["Z"], 0.2, 0.9)
            s = mask_scale
            edge = g.remap(g.math("SUBTRACT", 1.0, g.ao(0.0025 * s, inside=True)), 0.12, 0.55)
            cavity = g.remap(g.math("SUBTRACT", 1.0, g.ao(0.005 * s)), 0.08, 0.5)
            occlusion = g.ao(0.05 * s if ao_distance is None else ao_distance)
            self.masks_emit = g.node("ShaderNodeEmission", {
                "Color": g.node("ShaderNodeCombineColor", {"Red": edge, "Green": cavity,
                                                           "Blue": occlusion}).outputs["Color"]})
            self.masks_img = new_image(f"T_{name}_Masks", "Non-Color", res)
            sep = g.node("ShaderNodeSeparateColor",
                         {"Color": g.image(self.masks_img, self.UV).outputs["Color"]})
            self.edge = sep.outputs["Red"]
            self.cavity = sep.outputs["Green"]
            self.ao = sep.outputs["Blue"]

    def with_masks(self, edge=None, cavity=None, ao=None, up=None):
        """A copy of this context with some masks replaced - e.g. an edge mask from
        a distance-to-rim attribute for plates too thin for the baked AO edge."""
        import copy
        c = copy.copy(self)
        for key, value in (("edge", edge), ("cavity", cavity), ("ao", ao), ("up", up)):
            if value is not None:
                setattr(c, key, value)
        return c

    def attr(self, name):
        return self.g.node("ShaderNodeAttribute", attribute_name=name).outputs["Fac"]

    def attr_vec(self, name):
        return self.g.node("ShaderNodeAttribute", attribute_name=name).outputs["Vector"]

    def is_value(self, attr_output, value):
        g = self.g
        return g.math("LESS_THAN", g.math("ABSOLUTE", g.math("SUBTRACT", attr_output, value)), 0.5)


# --- material assembly ----------------------------------------------------------
# Linear luminance limits matching validate.py: non-metal sRGB 30..240, metal >= ~175.
SAFE_DIELECTRIC = (0.0135, 0.85)
SAFE_METAL_MIN = 0.47


def pbr_safe(g, base, metal):
    """Rescale albedo brightness (keeping hue) into the PBR-valid range for its
    metalness, like a 'PBR safe colour' filter in texturing tools."""
    lum = g.node("ShaderNodeRGBToBW", {"Color": base}).outputs[0]
    lum = g.maxf(lum, 1e-4)
    lo = g.mix_float(SAFE_DIELECTRIC[0], SAFE_METAL_MIN, metal)
    k = g.maxf(1.0, g.math("DIVIDE", lo, lum))                       # lift too-dark values
    hi = g.mix_float(SAFE_DIELECTRIC[1], 1.0, metal)                 # metals may be brighter
    k = g.minf(k, g.math("DIVIDE", hi, lum))                         # tame too-bright ones
    scale = g.node("ShaderNodeCombineColor", {"Red": k, "Green": k, "Blue": k}).outputs["Color"]
    return g.tint(base, scale)


def combine(g, c, layers, safe=True):
    """Blend ``[(mask, Surface), ...]`` (first mask None) into bake shaders.
    ``safe`` keeps the final albedo inside PBR ranges."""
    first = layers[0][1]
    base, rough, metal, height = first.base, first.rough, first.metal, first.height

    def bump(s):
        return g.node("ShaderNodeBump", {"Height": s.height, "Distance": s.bump,
                                         "Strength": 1.0}).outputs["Normal"]

    normal = bump(first)
    hscaled = g.mul(first.height, first.bump)
    for mask, s in layers[1:]:
        base = g.mix_color(base, s.base, mask)
        rough = g.mix_float(rough, s.rough, mask)
        metal = g.mix_float(metal, s.metal, mask)
        normal = g.mix_vector(normal, bump(s), mask)
        hscaled = g.mix_float(hscaled, g.mul(s.height, s.bump), mask)
    normal = g.node("ShaderNodeVectorMath", {0: normal}, operation="NORMALIZE").outputs["Vector"]
    if safe:
        base = pbr_safe(g, base, g.clamp01(metal))
    hcm = g.add(g.mul(hscaled, 100.0), 0.5)
    return {
        "masks": c.masks_emit,
        "base": g.node("ShaderNodeEmission", {"Color": base}),
        "orm": g.node("ShaderNodeEmission", {"Color": g.node("ShaderNodeCombineColor", {
            "Red": c.ao, "Green": g.clamp01(rough), "Blue": g.clamp01(metal)}).outputs["Color"]}),
        "normal": g.node("ShaderNodeBsdfPrincipled", {"Normal": normal}),
        # height in centimetres + 0.5 (kept positive for the emission bake),
        # baked to a float image and normalised afterwards
        "height": g.node("ShaderNodeEmission", {"Color": g.node("ShaderNodeCombineColor", {
            "Red": hcm, "Green": hcm, "Blue": hcm}).outputs["Color"]}),
    }


def new_bake_material(obj, name, res, tileable=False, mask_scale=1.0, ao_distance=None):
    mat = bpy.data.materials.new(f"M_{name}_Bake")
    mat.use_nodes = True
    g = Graph(mat)
    g.nodes.clear()
    out = g.node("ShaderNodeOutputMaterial")
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return g, out, Context(g, name, res, tileable, mask_scale, ao_distance)


def make_cage(low, high, margin=0.0008, limit=0.006):
    """Per-vertex bake cage: the low mesh pushed out along its normals only as far
    as the high mesh actually sticks out at that spot (+ `margin`). A uniform cage
    would start rays inside nearby layers (straps, overlapping plates) and catch
    their undersides; this one hugs the surface like a hand-tuned cage."""
    from mathutils.bvhtree import BVHTree
    deps = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(high, deps)
    me = low.data.copy()
    cage = bpy.data.objects.new(low.name + "_Cage", me)
    bpy.context.scene.collection.objects.link(cage)
    cage.matrix_world = low.matrix_world.copy()
    pushes = []
    for v in me.vertices:
        loc, _, _, _ = bvh.find_nearest(v.co)
        out = (loc - v.co).dot(v.normal) if loc is not None else 0.0
        pushes.append(min(limit, max(0.0, out) + margin))
    for v, e in zip(me.vertices, pushes):
        v.co = v.co + v.normal * e
    print(f"[aaa] cage: push p50 {1000 * float(np.median(pushes)):.1f} mm, max {1000 * max(pushes):.1f} mm")
    return cage


def bake_pass(g, out, shader_node, target_img, bake_type, samples, target_graph=None, cage=None, ray=None):
    """Bake `shader_node` of graph `g` into `target_img`. With `target_graph` (the
    material of a selected-to-active low-poly target) the bake is projected from
    the high-poly source onto the target's UVs."""
    g.links.new(shader_node.outputs[0], out.inputs["Surface"])
    tg = target_graph or g
    target = tg.node("ShaderNodeTexImage", image=target_img)
    for nd in tg.nodes:
        nd.select = False
    target.select = True
    tg.nodes.active = target
    bpy.context.scene.cycles.samples = samples
    extra = {}
    if target_graph is not None:
        extra = dict(use_selected_to_active=True, max_ray_distance=ray)
        if isinstance(cage, bpy.types.Object):
            extra.update(use_cage=True, cage_object=cage.name)
        else:
            extra.update(cage_extrusion=cage)
    bpy.ops.object.bake(type=bake_type, margin=BAKE_MARGIN, use_clear=True, normal_space="TANGENT", **extra)
    tg.nodes.remove(target)


def _select(active, *others):
    bpy.ops.object.select_all(action="DESELECT")
    for o in (active,) + others:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active


def pixels(img):
    px = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(img.size[1], img.size[0], 4)


def save_array(arr, name, colorspace, path, res):
    """Write an HxWx4 float array (0..1, already in the image's colour space) to PNG."""
    img = new_image(name, colorspace, res)
    img.pixels.foreach_set(np.ascontiguousarray(arr, dtype=np.float32).ravel())
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()
    return img


def make_seamless(arr, band=0.25):
    """Image-space seamless tiling: blend the image with a half-offset copy of
    itself. The copy fills a cross-shaped band over the borders, so opposite
    edges match exactly while the centre keeps the original detail."""
    h, w = arr.shape[:2]

    def ramp(n):
        x = np.abs(np.linspace(-1, 1, n))              # 0 centre, 1 at the edges
        t = np.clip((x - (1 - band * 2)) / (band * 2), 0, 1)
        return t * t * (3 - 2 * t)

    wy, wx = ramp(h)[:, None], ramp(w)[None, :]
    weight = np.maximum(wy, wx)[..., None]            # 1 near any border
    shifted = np.roll(np.roll(arr, h // 2, axis=0), w // 2, axis=1)
    return arr * (1 - weight) + shifted * weight


def bake(obj, out_dir, name, layers_fn, res=2048, tileable=False, mask_scale=1.0,
         samples=4, export_height=None, ao_distance=None, safe=True, low=None, cage="auto", ray=0.008):
    """Bake ``layers_fn(context) -> [(mask, Surface), ...]`` on ``obj`` (unwrapped).

    With ``low`` (a game mesh with its own UVs) the material is evaluated on the
    high-poly ``obj`` and projected onto ``low`` (high-to-low bake): geometric
    detail that only exists on ``obj`` - folds, subdivision - ends up in the
    low mesh's normal map. ``cage`` is "auto" (a per-vertex cage, see make_cage) or a
    uniform extrusion in metres; ``ray`` is the maximum ray distance.
    Returns {key: Path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.render.bake.margin_type = "EXTEND"
    _select(obj)

    g, out, c = new_bake_material(obj, name, res, tileable, mask_scale, ao_distance)
    shaders = combine(g, c, layers_fn(c), safe)
    names = texture_names(name)
    export_height = tileable if export_height is None else export_height

    if shaders["masks"] is not None:
        print(f"[aaa] {name}: baking edge/cavity/AO masks")
        bake_pass(g, out, shaders["masks"], c.masks_img, "EMIT", 16)

    target = {}
    if low is not None:
        tmat = bpy.data.materials.new(f"M_{name}_BakeTarget")
        tmat.use_nodes = True
        low.data.materials.clear()
        low.data.materials.append(tmat)
        cage_obj = make_cage(low, obj) if cage == "auto" else cage
        target = dict(target_graph=Graph(tmat), cage=cage_obj, ray=ray)
        _select(low, obj)
        print(f"[aaa] {name}: projecting high ({len(obj.data.polygons)} faces) -> low ({len(low.data.polygons)} faces)")

    raw = {}
    for key, kind, colorspace, float_buf in (("base", "EMIT", "sRGB", False),
                                             ("orm", "EMIT", "Non-Color", False),
                                             ("normal_gl", "NORMAL", "Non-Color", False),
                                             ("height", "EMIT", "Non-Color", True)):
        if key == "height" and not export_height:
            continue
        print(f"[aaa] {name}: baking {key}")
        img = new_image(names[key][0] + "_raw", colorspace, res, float_buffer=float_buf)
        shader = shaders["normal" if key == "normal_gl" else key]
        bake_pass(g, out, shader, img, kind, samples, **target)
        raw[key] = pixels(img)

    if low is not None and isinstance(target.get("cage"), bpy.types.Object):
        cage_mesh = target["cage"].data
        bpy.data.objects.remove(target["cage"])
        bpy.data.meshes.remove(cage_mesh)
    if tileable:
        for key in raw:
            raw[key] = make_seamless(raw[key])
    else:
        raw = pad(raw, empty_mask(raw["base"], raw["orm"]))
        n = raw["normal_gl"][..., :3] * 2 - 1
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        raw["normal_gl"][..., :3] = n * 0.5 + 0.5

    # Write the normal map the way the engine reads it: BC5 keeps X/Y and rebuilds
    # Z = sqrt(1 - x^2 - y^2), so Z is always positive and the vector unit length.
    n = raw["normal_gl"][..., :3] * 2 - 1
    xy = n[..., :2]
    ln = np.linalg.norm(xy, axis=-1, keepdims=True)
    xy = np.where(ln > 0.995, xy * (0.995 / np.maximum(ln, 1e-6)), xy)
    z = np.sqrt(np.clip(1 - (xy ** 2).sum(-1, keepdims=True), 0, 1))
    raw["normal_gl"][..., :3] = np.concatenate([xy, z], -1) * 0.5 + 0.5

    paths = {}
    for key in ("base", "orm", "normal_gl"):
        paths[key] = out_dir / f"{names[key][0]}.png"
        save_array(raw[key], names[key][0], names[key][1], paths[key], res)
    dx = raw["normal_gl"].copy()
    dx[..., 1] = 1.0 - dx[..., 1]                        # DirectX convention for Unreal
    paths["normal_dx"] = out_dir / f"{names['normal_dx'][0]}.png"
    save_array(dx, names["normal_dx"][0], "Non-Color", paths["normal_dx"], res)
    if export_height:
        h = (raw["height"][..., 0] - 0.5) / 100.0          # back to metres
        lo, hi = np.percentile(h, 0.5), np.percentile(h, 99.5)
        hn = np.clip((h - lo) / max(hi - lo, 1e-9), 0, 1)
        arr = np.dstack([hn, hn, hn, np.ones_like(hn)])
        paths["height"] = out_dir / f"{names['height'][0]}.png"
        save_array(arr, names["height"][0], "Non-Color", paths["height"], res)
        print(f"[aaa] {name}: height range {1000 * (hi - lo):.2f} mm")
    print(f"[aaa] {name}: wrote {len(paths)} textures to {out_dir}")
    return paths


def assign_baked_material(obj, tex_dir, name, tiling=1.0):
    """Principled material driven by a baked set (what Unreal will match)."""
    names = texture_names(name)
    mat = bpy.data.materials.new(f"M_{name}")
    mat.use_nodes = True
    g = Graph(mat)
    bsdf = g.nodes.get("Principled BSDF")
    uv = g.mapping(g.node("ShaderNodeTexCoord").outputs["UV"], (tiling, tiling, 1))

    def load(key):
        img = bpy.data.images.load(str(Path(tex_dir) / f"{names[key][0]}.png"), check_existing=True)
        img.colorspace_settings.name = names[key][1]
        return g.image(img, uv)

    g.set(bsdf.inputs["Base Color"], load("base").outputs["Color"])
    orm = g.node("ShaderNodeSeparateColor", {"Color": load("orm").outputs["Color"]})
    g.set(bsdf.inputs["Roughness"], orm.outputs["Green"])
    g.set(bsdf.inputs["Metallic"], orm.outputs["Blue"])
    nmap = g.node("ShaderNodeNormalMap", {"Color": load("normal_gl").outputs["Color"]})
    g.set(bsdf.inputs["Normal"], nmap.outputs["Normal"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return mat
