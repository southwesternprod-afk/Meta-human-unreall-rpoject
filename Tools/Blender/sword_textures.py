"""Bake a game-ready PBR texture set for the steel sword.

Everything is procedural: shader networks driven by object-space position,
per-vertex attributes written by build_steel_sword.py ("part", "edgeu") and
baked curvature/cavity masks. The result is one texture set in the layout
Unreal expects:

    T_SteelSword_BaseColor.png      sRGB
    T_SteelSword_Normal.png         tangent space, DirectX (-Y) for Unreal
    T_SteelSword_Normal_OpenGL.png  tangent space, OpenGL (+Y) for Blender / glTF
    T_SteelSword_ORM.png            R = ambient occlusion, G = roughness, B = metallic
"""

from pathlib import Path

import bpy
import numpy as np

TEX_RES = 2048
BAKE_MARGIN = 16

# Inscription placement inside the fuller (object space, metres).
ENGRAVE_Z0 = 0.19
ENGRAVE_LEN = 0.22
ENGRAVE_HEIGHT = 0.0095

TEXTURES = {
    "base": ("T_SteelSword_BaseColor", "sRGB"),
    "orm": ("T_SteelSword_ORM", "Non-Color"),
    "normal_gl": ("T_SteelSword_Normal_OpenGL", "Non-Color"),
    "normal_dx": ("T_SteelSword_Normal", "Non-Color"),
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


def new_image(name, colorspace):
    img = bpy.data.images.new(name, TEX_RES, TEX_RES, alpha=False, float_buffer=False)
    img.colorspace_settings.name = colorspace
    return img


def build_bake_material(sword, engraving_path):
    mat = bpy.data.materials.new("M_SteelSword_Bake")
    mat.use_nodes = True
    g = Graph(mat)
    g.nodes.clear()
    out = g.node("ShaderNodeOutputMaterial")

    tc = g.node("ShaderNodeTexCoord")
    P = tc.outputs["Object"]
    UV = tc.outputs["UV"]

    part = g.node("ShaderNodeAttribute", attribute_name="part").outputs["Fac"]
    edgeu = g.node("ShaderNodeAttribute", attribute_name="edgeu").outputs["Fac"]
    is_blade = g.math("LESS_THAN", part, 0.5)
    is_leather = g.math("GREATER_THAN", part, 1.5)

    # ---- Pass 1: curvature / cavity / AO masks ------------------------------
    edge = g.remap(g.math("SUBTRACT", 1.0, g.ao(0.0025, inside=True)), 0.12, 0.55)
    cavity = g.remap(g.math("SUBTRACT", 1.0, g.ao(0.005)), 0.08, 0.5)
    occlusion = g.ao(0.05)
    masks_emit = g.node("ShaderNodeEmission", {
        "Color": g.node("ShaderNodeCombineColor", {"Red": edge, "Green": cavity,
                                                   "Blue": occlusion}).outputs["Color"]})

    masks_img = new_image("T_SteelSword_Masks", "Non-Color")
    masks_sep = g.node("ShaderNodeSeparateColor",
                       {"Color": g.image(masks_img, UV).outputs["Color"]})
    edge_m = masks_sep.outputs["Red"]
    cavity_m = masks_sep.outputs["Green"]
    ao_m = masks_sep.outputs["Blue"]

    # ---- Pattern library ----------------------------------------------------
    streak = g.remap(g.noise(g.mapping(P, (900, 900, 5)), 1.0, 8.0, 0.6), 0.3, 0.7)
    streak_f = g.remap(g.noise(g.mapping(P, (8, 700, 700)), 1.0, 8.0, 0.6), 0.3, 0.7)
    smudge = g.remap(g.noise(P, 14.0, 3.0, 0.5), 0.5, 0.72)
    patch = g.remap(g.noise(P, 6.0, 2.0, 0.5), 0.45, 0.62)

    def scratch_layer(scale, rot, width):
        d = g.voronoi(g.mapping(P, scale, rot), 1.0, "DISTANCE_TO_EDGE")
        return g.remap(d, 0.0, width, 1.0, 0.0)

    s1 = scratch_layer((22, 22, 2.2), (0, 0.35, 0), 0.020)
    s2 = scratch_layer((38, 38, 3.5), (0, -0.55, 0.2), 0.016)
    s3 = g.math("MULTIPLY", scratch_layer((110, 110, 10), (0, 1.15, 0), 0.03), 0.45)
    scratch = g.math("MULTIPLY", g.math("MAXIMUM", g.math("MAXIMUM", s1, s2), s3), patch)

    pits = g.math("MULTIPLY", g.remap(g.voronoi(P, 650.0), 0.0, 0.16, 1.0, 0.0),
                  g.remap(g.noise(P, 25.0, 2.0), 0.48, 0.6))
    grain = g.remap(g.math("ADD", g.math("MULTIPLY", g.voronoi(P, 1700.0), 0.7),
                           g.math("MULTIPLY", g.noise(P, 500.0, 4.0), 0.5)), 0.2, 0.8)
    worn = g.math("MULTIPLY", g.math("SUBTRACT", 1.0, cavity_m),
                  g.remap(g.noise(P, 30.0, 3.0), 0.45, 0.65))

    polish = g.math("MULTIPLY", g.remap(edgeu, 0.872, 0.892), is_blade)
    in_fuller = g.remap(edgeu, 0.255, 0.235)

    # Inscription: object-space projection into the fuller, both faces.
    xyz = g.node("ShaderNodeSeparateXYZ", {"Vector": P}).outputs
    u = g.math("DIVIDE", g.math("SUBTRACT", xyz["Z"], ENGRAVE_Z0), ENGRAVE_LEN)
    v = g.math("ADD", g.math("DIVIDE", xyz["X"], ENGRAVE_HEIGHT), 0.5)
    engr_img = bpy.data.images.load(str(engraving_path), check_existing=True)
    engr_img.colorspace_settings.name = "Non-Color"
    engr_uv = g.node("ShaderNodeCombineXYZ", {"X": u, "Y": v}).outputs["Vector"]
    engr = g.math("MULTIPLY", g.image(engr_img, engr_uv, "CLIP").outputs["Color"],
                  g.math("MULTIPLY", in_fuller, is_blade))

    # ---- Base colour --------------------------------------------------------
    def rgb(r, gr, b):
        return (r, gr, b, 1.0)

    blade = g.mix_color(rgb(0.50, 0.50, 0.51), rgb(0.62, 0.62, 0.63), streak)
    blade = g.mix_color(blade, rgb(0.78, 0.78, 0.79), polish)
    blade = g.mix_color(blade, rgb(0.80, 0.80, 0.81), g.math("MULTIPLY", scratch, 0.55))
    blade = g.mix_color(blade, rgb(0.20, 0.18, 0.15),
                        g.math("ADD", g.math("MULTIPLY", cavity_m, 0.65),
                               g.math("MULTIPLY", smudge, 0.12), clamp=True))
    blade = g.mix_color(blade, rgb(0.92, 0.70, 0.36), engr)

    fit = g.mix_color(rgb(0.36, 0.36, 0.38), rgb(0.45, 0.45, 0.47), streak_f)
    fit = g.mix_color(fit, rgb(0.20, 0.20, 0.21), pits)
    fit = g.mix_color(fit, rgb(0.72, 0.72, 0.73), edge_m)
    fit = g.mix_color(fit, rgb(0.70, 0.70, 0.71), g.math("MULTIPLY", scratch, 0.5))
    fit = g.mix_color(fit, rgb(0.08, 0.07, 0.06), g.math("MULTIPLY", cavity_m, 0.9))

    leather = g.mix_color(rgb(0.045, 0.020, 0.009), rgb(0.090, 0.042, 0.019), grain)
    leather = g.mix_color(leather, rgb(0.130, 0.066, 0.032), g.math("MULTIPLY", worn, 0.65))
    leather = g.mix_color(leather, rgb(0.022, 0.011, 0.005), g.math("MULTIPLY", cavity_m, 0.8))

    base = g.mix_color(g.mix_color(fit, blade, is_blade), leather, is_leather)

    # ---- Roughness / metallic ----------------------------------------------
    r_blade = g.math("ADD", g.math("ADD", 0.19, g.math("MULTIPLY", streak, 0.10)),
                     g.math("MULTIPLY", smudge, 0.16))
    r_blade = g.mix_float(r_blade, 0.10, polish)
    r_blade = g.mix_float(r_blade, 0.40, g.math("MULTIPLY", scratch, 0.8))
    r_blade = g.mix_float(r_blade, 0.55, cavity_m)
    r_blade = g.mix_float(r_blade, 0.30, engr)

    r_fit = g.math("ADD", 0.30, g.math("MULTIPLY", streak_f, 0.12))
    r_fit = g.mix_float(r_fit, 0.18, edge_m)
    r_fit = g.mix_float(r_fit, 0.60, pits)
    r_fit = g.mix_float(r_fit, 0.45, g.math("MULTIPLY", scratch, 0.7))
    r_fit = g.mix_float(r_fit, 0.68, cavity_m)

    r_leather = g.math("ADD", 0.56, g.math("MULTIPLY", grain, 0.16))
    r_leather = g.mix_float(r_leather, 0.40, g.math("MULTIPLY", worn, 0.7))
    r_leather = g.mix_float(r_leather, 0.82, cavity_m)

    rough = g.mix_float(g.mix_float(r_fit, r_blade, is_blade), r_leather, is_leather)
    metal_metal = g.math("SUBTRACT", 1.0, g.math("MULTIPLY", cavity_m, 0.35))
    metal = g.mix_float(metal_metal, 0.0, is_leather)

    orm_emit = g.node("ShaderNodeEmission", {
        "Color": g.node("ShaderNodeCombineColor", {"Red": ao_m, "Green": rough,
                                                   "Blue": metal}).outputs["Color"]})
    base_emit = g.node("ShaderNodeEmission", {"Color": base})

    # ---- Surface detail (normal map) ---------------------------------------
    h_blade = g.math("SUBTRACT", g.math("MULTIPLY", streak, 0.15),
                     g.math("ADD", g.math("MULTIPLY", scratch, 0.6), engr))
    h_fit = g.math("SUBTRACT", g.math("MULTIPLY", streak_f, 0.3),
                   g.math("ADD", g.math("MULTIPLY", pits, 0.9), g.math("MULTIPLY", scratch, 0.5)))

    def bump(height, distance):
        return g.node("ShaderNodeBump", {"Height": height, "Distance": distance,
                                         "Strength": 1.0}).outputs["Normal"]

    n = g.mix_vector(bump(h_fit, 0.00008), bump(h_blade, 0.00005), is_blade)
    n = g.mix_vector(n, bump(grain, 0.00025), is_leather)
    n = g.node("ShaderNodeVectorMath", {0: n}, operation="NORMALIZE").outputs["Vector"]
    normal_bsdf = g.node("ShaderNodeBsdfPrincipled", {"Normal": n})

    sword.data.materials.clear()
    sword.data.materials.append(mat)
    return g, out, {"masks": masks_emit, "base": base_emit, "orm": orm_emit,
                    "normal": normal_bsdf}, masks_img


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


def bake_textures(sword, tex_dir, engraving_path):
    tex_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.render.bake.margin_type = "EXTEND"

    bpy.ops.object.select_all(action="DESELECT")
    sword.select_set(True)
    bpy.context.view_layer.objects.active = sword

    g, out, shaders, masks_img = build_bake_material(sword, engraving_path)

    print("Baking masks...")
    bake_pass(g, out, shaders["masks"], masks_img, "EMIT", 16)

    imgs = {key: new_image(name, cs) for key, (name, cs) in TEXTURES.items() if key != "normal_dx"}
    print("Baking base colour...")
    bake_pass(g, out, shaders["base"], imgs["base"], "EMIT", 4)
    print("Baking ORM...")
    bake_pass(g, out, shaders["orm"], imgs["orm"], "EMIT", 4)
    print("Baking normal...")
    bake_pass(g, out, shaders["normal"], imgs["normal_gl"], "NORMAL", 4)

    for key, img in imgs.items():
        save_png(img, tex_dir / f"{TEXTURES[key][0]}.png")

    # Unreal expects DirectX-style normal maps: flip the green channel.
    px = np.empty(TEX_RES * TEX_RES * 4, dtype=np.float32)
    imgs["normal_gl"].pixels.foreach_get(px)
    px = px.reshape(-1, 4)
    px[:, 1] = 1.0 - px[:, 1]
    dx = new_image(TEXTURES["normal_dx"][0], "Non-Color")
    dx.pixels.foreach_set(px.ravel())
    save_png(dx, tex_dir / f"{TEXTURES['normal_dx'][0]}.png")


def assign_final_material(sword, tex_dir):
    """Game material driven by the baked textures (what Unreal will match)."""
    mat = bpy.data.materials.new("M_SteelSword")
    mat.use_nodes = True
    g = Graph(mat)
    bsdf = g.nodes.get("Principled BSDF")

    def load(key):
        img = bpy.data.images.load(str(tex_dir / f"{TEXTURES[key][0]}.png"), check_existing=True)
        img.colorspace_settings.name = TEXTURES[key][1]
        return g.image(img)

    g.set(bsdf.inputs["Base Color"], load("base").outputs["Color"])
    orm = g.node("ShaderNodeSeparateColor", {"Color": load("orm").outputs["Color"]})
    g.set(bsdf.inputs["Roughness"], orm.outputs["Green"])
    g.set(bsdf.inputs["Metallic"], orm.outputs["Blue"])
    nmap = g.node("ShaderNodeNormalMap", {"Color": load("normal_gl").outputs["Color"]})
    g.set(bsdf.inputs["Normal"], nmap.outputs["Normal"])

    sword.data.materials.clear()
    sword.data.materials.append(mat)
    return mat


def texture_sword(sword, tex_dir: Path, engraving_path: Path, bake=True):
    if bake or not (tex_dir / f"{TEXTURES['base'][0]}.png").exists():
        bake_textures(sword, tex_dir, engraving_path)
    return assign_final_material(sword, tex_dir)
