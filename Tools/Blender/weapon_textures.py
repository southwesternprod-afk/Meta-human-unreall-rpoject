"""Bake game-ready PBR texture sets for the procedural weapons.

Everything is procedural: shader networks driven by object-space position,
per-vertex attributes written by the build scripts ("part", "edgeu") and
baked curvature/cavity masks. Each weapon gets one texture set in the
layout Unreal expects (NAME is e.g. SteelSword or SteelAxe):

    T_NAME_BaseColor.png      sRGB
    T_NAME_Normal.png         tangent space, DirectX (-Y) for Unreal
    T_NAME_Normal_OpenGL.png  tangent space, OpenGL (+Y) for Blender / glTF
    T_NAME_ORM.png            R = ambient occlusion, G = roughness, B = metallic
"""

from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np

TEX_RES = 2048
BAKE_MARGIN = 16


def texture_names(name):
    return {
        "base": (f"T_{name}_BaseColor", "sRGB"),
        "orm": (f"T_{name}_ORM", "Non-Color"),
        "normal_gl": (f"T_{name}_Normal_OpenGL", "Non-Color"),
        "normal_dx": (f"T_{name}_Normal", "Non-Color"),
    }


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

    def math(self, op, a, b=0.0, clamp=False):
        nd = self.nodes.new("ShaderNodeMath")
        nd.operation = op
        nd.use_clamp = clamp
        self.set(nd.inputs[0], a)
        self.set(nd.inputs[1], b)
        return nd.outputs[0]

    def remap(self, value, a, b, to_a=0.0, to_b=1.0):
        """Linear remap of a..b onto to_a..to_b, clamped."""
        nd = self.node("ShaderNodeMapRange", {"Value": value, "From Min": a, "From Max": b,
                                              "To Min": to_a, "To Max": to_b})
        nd.clamp = True
        return nd.outputs["Result"]

    def mix_float(self, a, b, t):
        nd = self.nodes.new("ShaderNodeMix")
        nd.data_type = "FLOAT"
        nd.clamp_factor = True
        self.set(self._sock(nd.inputs, "Factor_Float"), t)
        self.set(self._sock(nd.inputs, "A_Float"), a)
        self.set(self._sock(nd.inputs, "B_Float"), b)
        return self._sock(nd.outputs, "Result_Float")

    def mix_color(self, a, b, t):
        nd = self.nodes.new("ShaderNodeMix")
        nd.data_type = "RGBA"
        nd.clamp_factor = True
        self.set(self._sock(nd.inputs, "Factor_Float"), t)
        self.set(self._sock(nd.inputs, "A_Color"), a)
        self.set(self._sock(nd.inputs, "B_Color"), b)
        return self._sock(nd.outputs, "Result_Color")

    def mix_vector(self, a, b, t):
        nd = self.nodes.new("ShaderNodeMix")
        nd.data_type = "VECTOR"
        nd.clamp_factor = True
        self.set(self._sock(nd.inputs, "Factor_Float"), t)
        self.set(self._sock(nd.inputs, "A_Vector"), a)
        self.set(self._sock(nd.inputs, "B_Vector"), b)
        return self._sock(nd.outputs, "Result_Vector")

    @staticmethod
    def _sock(sockets, identifier):
        return next(s for s in sockets if s.identifier == identifier)

    def mapping(self, vec, scale=(1, 1, 1), rot=(0, 0, 0)):
        return self.node("ShaderNodeMapping", {"Vector": vec, "Scale": scale,
                                               "Rotation": rot}).outputs["Vector"]

    def noise(self, vec, scale, detail=4.0, rough=0.5):
        return self.node("ShaderNodeTexNoise", {"Vector": vec, "Scale": scale, "Detail": detail,
                                                "Roughness": rough}).outputs["Fac"]

    def voronoi(self, vec, scale, feature="F1"):
        return self.node("ShaderNodeTexVoronoi", {"Vector": vec, "Scale": scale},
                         feature=feature).outputs["Distance"]

    def image(self, img, vec=None, extension="REPEAT"):
        nd = self.node("ShaderNodeTexImage", image=img, extension=extension)
        if vec is not None:
            self.set(nd.inputs["Vector"], vec)
        return nd

    def ao(self, distance, inside=False, samples=8):
        return self.node("ShaderNodeAmbientOcclusion", {"Distance": distance},
                         inside=inside, only_local=True, samples=samples).outputs["AO"]




def rgb(r, g, b):
    return (r, g, b, 1.0)


def new_image(name, colorspace):
    img = bpy.data.images.new(name, TEX_RES, TEX_RES, alpha=False, float_buffer=False)
    img.colorspace_settings.name = colorspace
    return img


# --- Shared building blocks --------------------------------------------------
@dataclass
class Surface:
    """One material layer: colour, roughness, metallic and a bump height."""
    base: object
    rough: object
    metal: object
    height: object
    bump_distance: float


class Context:
    """Common inputs for a weapon bake material: coordinates, attributes,
    the baked edge/cavity/AO masks and a few shared noise patterns."""

    def __init__(self, g, name):
        self.g = g
        tc = g.node("ShaderNodeTexCoord")
        self.P = tc.outputs["Object"]
        self.UV = tc.outputs["UV"]
        self.N = tc.outputs["Normal"]
        self.part = g.node("ShaderNodeAttribute", attribute_name="part").outputs["Fac"]
        self.edgeu = g.node("ShaderNodeAttribute", attribute_name="edgeu").outputs["Fac"]

        # Pass 1 shader: convex edges, cavities and broad AO.
        edge = g.remap(g.math("SUBTRACT", 1.0, g.ao(0.0025, inside=True)), 0.12, 0.55)
        cavity = g.remap(g.math("SUBTRACT", 1.0, g.ao(0.005)), 0.08, 0.5)
        occlusion = g.ao(0.05)
        self.masks_emit = g.node("ShaderNodeEmission", {
            "Color": g.node("ShaderNodeCombineColor", {"Red": edge, "Green": cavity,
                                                       "Blue": occlusion}).outputs["Color"]})
        self.masks_img = new_image(f"T_{name}_Masks", "Non-Color")
        sep = g.node("ShaderNodeSeparateColor",
                     {"Color": g.image(self.masks_img, self.UV).outputs["Color"]})
        self.edge = sep.outputs["Red"]
        self.cavity = sep.outputs["Green"]
        self.ao = sep.outputs["Blue"]

        P = self.P
        self.patch = g.remap(g.noise(P, 6.0, 2.0, 0.5), 0.45, 0.62)
        self.smudge = g.remap(g.noise(P, 14.0, 3.0, 0.5), 0.5, 0.72)

    def is_part(self, value):
        g = self.g
        return g.math("LESS_THAN", g.math("ABSOLUTE", g.math("SUBTRACT", self.part, value)), 0.5)

    def streaks(self, scale):
        """Directional grinding/brushing streaks; `scale` stretches the noise."""
        g = self.g
        return g.remap(g.noise(g.mapping(self.P, scale), 1.0, 8.0, 0.6), 0.3, 0.7)

    def scratches(self):
        """Sparse scratch lines, mostly lying in the weapon's flat (XZ) plane."""
        g = self.g

        def layer(scale, rot, width):
            d = g.voronoi(g.mapping(self.P, scale, rot), 1.0, "DISTANCE_TO_EDGE")
            return g.remap(d, 0.0, width, 1.0, 0.0)

        s1 = layer((22, 22, 2.2), (0, 0.35, 0), 0.020)
        s2 = layer((38, 38, 3.5), (0, -0.55, 0.2), 0.016)
        s3 = g.math("MULTIPLY", layer((110, 110, 10), (0, 1.15, 0), 0.03), 0.45)
        return g.math("MULTIPLY", g.math("MAXIMUM", g.math("MAXIMUM", s1, s2), s3), self.patch)

    def pits(self, scale=650.0):
        g = self.g
        return g.math("MULTIPLY", g.remap(g.voronoi(self.P, scale), 0.0, 0.12, 0.7, 0.0),
                      g.remap(g.noise(self.P, 25.0, 2.0), 0.52, 0.62))

    def projected_mask(self, img_path, u_axis, u_range, v_axis, v_range, mirror_axis, gate):
        """Project a black/white image onto both flat faces of the weapon in
        object space. `mirror_axis` is flipped on the back (-Y) face so the
        image reads the same way from either side."""
        g = self.g
        xyz = g.node("ShaderNodeSeparateXYZ", {"Vector": self.P}).outputs
        side = g.math("SIGN", g.node("ShaderNodeSeparateXYZ", {"Vector": self.N}).outputs["Y"])

        def coord(axis, lo, hi):
            t = g.math("DIVIDE", g.math("SUBTRACT", xyz[axis], lo), hi - lo)
            if axis == mirror_axis:   # reflect within the range on the back face
                t = g.math("ADD", 0.5, g.math("MULTIPLY", g.math("SUBTRACT", t, 0.5), side))
            return t

        img = bpy.data.images.load(str(img_path), check_existing=True)
        img.colorspace_settings.name = "Non-Color"
        uv = g.node("ShaderNodeCombineXYZ", {"X": coord(u_axis, *u_range),
                                             "Y": coord(v_axis, *v_range)}).outputs["Vector"]
        return g.math("MULTIPLY", g.image(img, uv, "CLIP").outputs["Color"], gate)


def fittings_surface(c, streak_scale=(8, 700, 700)):
    """Forged steel fittings: file marks, pits, bright worn edges, grime."""
    g = c.g
    streak = c.streaks(streak_scale)
    pits = c.pits()
    scratch = c.scratches()

    base = g.mix_color(rgb(0.36, 0.36, 0.38), rgb(0.45, 0.45, 0.47), streak)
    base = g.mix_color(base, rgb(0.27, 0.27, 0.28), pits)
    base = g.mix_color(base, rgb(0.72, 0.72, 0.73), c.edge)
    base = g.mix_color(base, rgb(0.70, 0.70, 0.71), g.math("MULTIPLY", scratch, 0.5))
    base = g.mix_color(base, rgb(0.08, 0.07, 0.06), g.math("MULTIPLY", c.cavity, 0.9))

    rough = g.math("ADD", 0.30, g.math("MULTIPLY", streak, 0.12))
    rough = g.mix_float(rough, 0.18, c.edge)
    rough = g.mix_float(rough, 0.60, pits)
    rough = g.mix_float(rough, 0.45, g.math("MULTIPLY", scratch, 0.7))
    rough = g.mix_float(rough, 0.68, c.cavity)

    height = g.math("SUBTRACT", g.math("MULTIPLY", streak, 0.3),
                    g.math("ADD", g.math("MULTIPLY", pits, 0.9), g.math("MULTIPLY", scratch, 0.5)))
    return Surface(base, rough, metal_with_grime(c), height, 0.00008)


def leather_surface(c):
    """Dark brown leather with grain, hand-wear on raised areas, dirty creases."""
    g = c.g
    grain = g.remap(g.math("ADD", g.math("MULTIPLY", g.voronoi(c.P, 1700.0), 0.7),
                           g.math("MULTIPLY", g.noise(c.P, 500.0, 4.0), 0.5)), 0.2, 0.8)
    worn = g.math("MULTIPLY", g.math("SUBTRACT", 1.0, c.cavity),
                  g.remap(g.noise(c.P, 30.0, 3.0), 0.45, 0.65))

    base = g.mix_color(rgb(0.045, 0.020, 0.009), rgb(0.090, 0.042, 0.019), grain)
    base = g.mix_color(base, rgb(0.130, 0.066, 0.032), g.math("MULTIPLY", worn, 0.65))
    base = g.mix_color(base, rgb(0.022, 0.011, 0.005), g.math("MULTIPLY", c.cavity, 0.8))

    rough = g.math("ADD", 0.56, g.math("MULTIPLY", grain, 0.16))
    rough = g.mix_float(rough, 0.40, g.math("MULTIPLY", worn, 0.7))
    rough = g.mix_float(rough, 0.82, c.cavity)
    return Surface(base, rough, 0.0, grain, 0.00025)


def metal_with_grime(c):
    g = c.g
    return g.math("SUBTRACT", 1.0, g.math("MULTIPLY", c.cavity, 0.35))


def finish_bake_material(g, c, layers):
    """Blend layers by mask (first layer is the default) and build the
    shaders for each bake pass."""
    base, rough, metal = layers[0][1].base, layers[0][1].rough, layers[0][1].metal

    def bump(s):
        return g.node("ShaderNodeBump", {"Height": s.height, "Distance": s.bump_distance,
                                         "Strength": 1.0}).outputs["Normal"]

    normal = bump(layers[0][1])
    for mask, s in layers[1:]:
        base = g.mix_color(base, s.base, mask)
        rough = g.mix_float(rough, s.rough, mask)
        metal = g.mix_float(metal, s.metal, mask)
        normal = g.mix_vector(normal, bump(s), mask)
    normal = g.node("ShaderNodeVectorMath", {0: normal}, operation="NORMALIZE").outputs["Vector"]

    return {
        "masks": c.masks_emit,
        "base": g.node("ShaderNodeEmission", {"Color": base}),
        "orm": g.node("ShaderNodeEmission", {"Color": g.node("ShaderNodeCombineColor", {
            "Red": c.ao, "Green": rough, "Blue": metal}).outputs["Color"]}),
        "normal": g.node("ShaderNodeBsdfPrincipled", {"Normal": normal}),
    }


def new_bake_material(obj, name):
    mat = bpy.data.materials.new(f"M_{name}_Bake")
    mat.use_nodes = True
    g = Graph(mat)
    g.nodes.clear()
    out = g.node("ShaderNodeOutputMaterial")
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return g, out, Context(g, name)


# --- Sword ---------------------------------------------------------------------
# Inscription placement inside the fuller (object space, metres).
ENGRAVE_Z0 = 0.19
ENGRAVE_LEN = 0.22
ENGRAVE_HEIGHT = 0.0095


def sword_layers(c, engraving_path):
    """part 0 = blade, 1 = fittings, 2 = leather grip."""
    g = c.g
    is_blade = c.is_part(0)
    streak = c.streaks((900, 900, 5))
    scratch = c.scratches()

    polish = g.math("MULTIPLY", g.remap(c.edgeu, 0.872, 0.892), is_blade)
    in_fuller = g.remap(c.edgeu, 0.255, 0.235)
    # Inscription runs along the blade (Z) and across the fuller (X).
    engr = c.projected_mask(engraving_path, "Z", (ENGRAVE_Z0, ENGRAVE_Z0 + ENGRAVE_LEN),
                            "X", (-ENGRAVE_HEIGHT / 2, ENGRAVE_HEIGHT / 2), "X",
                            g.math("MULTIPLY", in_fuller, is_blade))

    base = g.mix_color(rgb(0.50, 0.50, 0.51), rgb(0.62, 0.62, 0.63), streak)
    base = g.mix_color(base, rgb(0.78, 0.78, 0.79), polish)
    base = g.mix_color(base, rgb(0.80, 0.80, 0.81), g.math("MULTIPLY", scratch, 0.55))
    base = g.mix_color(base, rgb(0.20, 0.18, 0.15),
                       g.math("ADD", g.math("MULTIPLY", c.cavity, 0.65),
                              g.math("MULTIPLY", c.smudge, 0.12), clamp=True))
    base = g.mix_color(base, rgb(0.92, 0.70, 0.36), engr)

    rough = g.math("ADD", g.math("ADD", 0.20, g.math("MULTIPLY", streak, 0.06)),
                   g.math("MULTIPLY", c.smudge, 0.16))
    rough = g.mix_float(rough, 0.10, polish)
    rough = g.mix_float(rough, 0.40, g.math("MULTIPLY", scratch, 0.8))
    rough = g.mix_float(rough, 0.55, c.cavity)
    rough = g.mix_float(rough, 0.30, engr)

    height = g.math("SUBTRACT", g.math("MULTIPLY", streak, 0.06),
                    g.math("ADD", g.math("MULTIPLY", scratch, 0.6), engr))
    blade = Surface(base, rough, metal_with_grime(c), height, 0.00005)

    return [(None, fittings_surface(c)),
            (is_blade, blade),
            (c.is_part(2), leather_surface(c))]


# --- Baking --------------------------------------------------------------------
def bake_pass(g, out, shader_node, target_img, bake_type, samples):
    g.links.new(shader_node.outputs[0], out.inputs["Surface"])
    target = g.node("ShaderNodeTexImage", image=target_img)
    for nd in g.nodes:
        nd.select = False
    target.select = True
    g.nodes.active = target
    bpy.context.scene.cycles.samples = samples
    bpy.ops.object.bake(type=bake_type, margin=BAKE_MARGIN, use_clear=True,
                        normal_space="TANGENT")
    g.nodes.remove(target)


def save_png(img, path):
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()


def bake_textures(obj, tex_dir, name, layers_fn):
    """layers_fn(context) -> [(mask, Surface), ...] with mask None for the base layer."""
    tex_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.render.bake.margin_type = "EXTEND"

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    g, out, c = new_bake_material(obj, name)
    shaders = finish_bake_material(g, c, layers_fn(c))
    names = texture_names(name)

    print("Baking masks...")
    bake_pass(g, out, shaders["masks"], c.masks_img, "EMIT", 16)

    imgs = {key: new_image(n, cs) for key, (n, cs) in names.items() if key != "normal_dx"}
    print("Baking base colour...")
    bake_pass(g, out, shaders["base"], imgs["base"], "EMIT", 4)
    print("Baking ORM...")
    bake_pass(g, out, shaders["orm"], imgs["orm"], "EMIT", 4)
    print("Baking normal...")
    bake_pass(g, out, shaders["normal"], imgs["normal_gl"], "NORMAL", 4)

    for key, img in imgs.items():
        save_png(img, tex_dir / f"{names[key][0]}.png")

    # Unreal expects DirectX-style normal maps: flip the green channel.
    px = np.empty(TEX_RES * TEX_RES * 4, dtype=np.float32)
    imgs["normal_gl"].pixels.foreach_get(px)
    px = px.reshape(-1, 4)
    px[:, 1] = 1.0 - px[:, 1]
    dx = new_image(names["normal_dx"][0], "Non-Color")
    dx.pixels.foreach_set(px.ravel())
    save_png(dx, tex_dir / f"{names['normal_dx'][0]}.png")


def assign_final_material(obj, tex_dir, name):
    """Game material driven by the baked textures (what Unreal will match)."""
    names = texture_names(name)
    mat = bpy.data.materials.new(f"M_{name}")
    mat.use_nodes = True
    g = Graph(mat)
    bsdf = g.nodes.get("Principled BSDF")

    def load(key):
        img = bpy.data.images.load(str(tex_dir / f"{names[key][0]}.png"), check_existing=True)
        img.colorspace_settings.name = names[key][1]
        return g.image(img)

    g.set(bsdf.inputs["Base Color"], load("base").outputs["Color"])
    orm = g.node("ShaderNodeSeparateColor", {"Color": load("orm").outputs["Color"]})
    g.set(bsdf.inputs["Roughness"], orm.outputs["Green"])
    g.set(bsdf.inputs["Metallic"], orm.outputs["Blue"])
    nmap = g.node("ShaderNodeNormalMap", {"Color": load("normal_gl").outputs["Color"]})
    g.set(bsdf.inputs["Normal"], nmap.outputs["Normal"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return mat


def texture_weapon(obj, tex_dir: Path, name, layers_fn, bake=True):
    if bake or not (tex_dir / f"{texture_names(name)['base'][0]}.png").exists():
        bake_textures(obj, tex_dir, name, layers_fn)
    return assign_final_material(obj, tex_dir, name)


def texture_sword(sword, tex_dir: Path, engraving_path: Path, bake=True):
    return texture_weapon(sword, tex_dir, "SteelSword",
                          lambda c: sword_layers(c, engraving_path), bake)
