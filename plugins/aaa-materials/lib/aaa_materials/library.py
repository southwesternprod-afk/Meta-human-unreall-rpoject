"""Smart materials: physically based presets with built-in wear and dirt.

Every preset is ``preset(c, **params) -> Surface`` and works for both asset
bakes (wear follows the baked edge/cavity masks) and tileable bakes.

Common params
    wear   0..1  edge wear / burnish / chipping
    dirt   0..1  cavity grime, stains, dust
    color  base colour as sRGB hex ('#4a2f1f') - stays inside PBR-safe ranges
    scale  multiplies every pattern frequency (2.0 = features half the size)
    P      coordinates for 3D patterns (default: object-space metres)
    P2     2D surface coordinates in metres for woven / ring patterns
           (default: UV * uv_size)

Albedo and roughness values follow common PBR charts (non-metals sRGB ~30-240,
raw metals sRGB 180-255 / linear >= 0.45). Oxide, rust, patina and grime are
modelled as non-metal layers on top of bright bare metal, which is what keeps
dark metals correct under Unreal's lighting.
"""

import math

from . import weathering as w
from .engine import Surface, rgb, srgb

PRESETS = {}


def preset(fn):
    PRESETS[fn.__name__] = fn
    return fn


def _P(c, P):
    return c.P if P is None else P


def _P2(c, P2, uv_size):
    if P2 is not None:
        return P2
    return c.g.mapping(c.UV, (uv_size, uv_size, 1.0))


def _snap(period, c, P2, uv_size):
    """For tileable bakes (UV-driven coords) fit a whole number of repeats in the tile."""
    if P2 is not None or not c.tileable:
        return period
    return uv_size / max(1, round(uv_size / period))


def _shade(color, k):
    """Colour multiplied by k (k < 1 darker)."""
    return tuple(min(1.0, v * k) for v in color[:3]) + (1.0,)


# --- leather ---------------------------------------------------------------------------
@preset
def leather(c, color="#4a2f1f", wear=0.5, dirt=0.5, scale=1.0, P=None, P2=None, uv_size=1.0):
    """Vegetable-tanned leather: pebbled grain, mottled dye, creases, scuffs,
    burnished worn areas and dark grime in the creases."""
    g = c.g
    P = _P(c, P)
    base_col = srgb(color)
    # pebbled grain: ~2 mm cells separated by fine valleys, plus pore noise
    cells = g.voronoi(g.mapping(P, (1, 1, 1), loc=(0.37, 0.11, 0.83)), 520.0 * scale, "DISTANCE_TO_EDGE")
    pebble = g.smooth(cells, 0.0, 0.16)
    pores = g.remap(g.noise(P, 1800.0 * scale, 2.0), 0.35, 0.65)
    grain = g.clamp01(g.add(g.mul(pebble, 0.8), g.mul(pores, 0.25)))
    mottle = g.remap(g.noise(P, 9.0 * scale, 4.0), 0.30, 0.72)
    crease = w.creases(c, 0.5, P)
    scuff = g.mul(g.remap(g.noise(P, 22.0 * scale, 6.0, 0.7), 0.56, 0.72), g.inv(c.cavity))
    stain = w.stains(c, dirt, 3.5 * scale, P)
    worn = w.edge_wear(c, wear, P)
    grime = w.cavity_dirt(c, dirt, P)
    dust = w.dust(c, dirt * 0.5, P)
    scratch = w.scratches(c, scale, P)

    base = g.mix_color(_shade(base_col, 0.55), _shade(base_col, 1.25), mottle)
    base = g.mix_color(base, _shade(base_col, 0.6), g.mul(grain, 0.35))
    base = g.mix_color(base, _shade(base_col, 0.45), g.mul(stain, 0.9))
    base = g.mix_color(base, _shade(base_col, 0.40), g.mul(crease, 0.5))
    base = g.mix_color(base, _shade(base_col, 2.0), g.mul(scuff, 0.5 * wear * 2))
    base = g.mix_color(base, _shade(base_col, 1.9), g.mul(scratch, 0.25 * wear * 2))
    base = g.mix_color(base, _shade(base_col, 2.1), g.mul(worn, 0.7))
    base = g.mix_color(base, _shade(base_col, 0.30), g.mul(grime, 0.9))
    base = g.mix_color(base, srgb("#8a7d6b"), g.mul(dust, 0.4))

    rough = g.add(0.72, g.mul(pebble, -0.20))           # valleys rough, pebble tops satin
    rough = g.mix_float(rough, 0.42, g.mul(stain, 0.6))
    rough = g.mix_float(rough, 0.36, g.mul(worn, 0.8))
    rough = g.mix_float(rough, 0.78, g.mul(scuff, wear))
    rough = g.mix_float(rough, 0.86, grime)
    rough = g.mix_float(rough, 0.90, g.mul(dust, 0.7))
    height = g.sub(g.mul(grain, 0.5), g.add(g.mul(crease, 0.8), g.mul(scratch, 0.4)))
    return Surface(base, rough, 0.0, height, 0.00035)


# --- metals ----------------------------------------------------------------------------
@preset
def steel(c, finish="brushed", wear=0.5, dirt=0.4, rust=0.15, scale=1.0, direction=(4, 700, 700), P=None,
          P2=None, uv_size=1.0):
    """Bare steel. finish: 'polished' (blade), 'brushed' (grinding streaks along
    `direction`, the stretch of the streak noise) or 'forged' (hammer dents,
    darker scale in the hollows)."""
    g = c.g
    P = _P(c, P)
    streak = g.remap(g.noise(g.mapping(P, direction), 1.0, 8.0, 0.6), 0.3, 0.7)
    dents = g.remap(g.noise(P, 55.0 * scale, 2.0), 0.3, 0.7)
    scratch = w.scratches(c, scale, P)
    smudge = g.remap(g.noise(P, 14.0 * scale, 3.0), 0.5, 0.72)
    pit = w.pits(c, 650.0 * scale, rust, P)
    rustm = g.mul(g.remap(g.noise(P, 18.0 * scale, 5.0, 0.65), 0.62, 0.74),
                  g.add(g.mul(c.cavity, 0.8), 0.25))
    rustm = g.mul(rustm, rust * 2.0)
    grime = w.cavity_dirt(c, dirt, P)
    edges = w.edge_wear(c, wear, P)

    rough0 = {"polished": 0.16, "brushed": 0.28, "forged": 0.42}[finish]
    base = g.mix_color(srgb("#b4b6b8"), srgb("#c8cacc"), streak)
    if finish == "forged":
        base = g.mix_color(base, srgb("#8e9092"), g.mul(dents, 0.6))
    base = g.mix_color(base, srgb("#dcdddf"), edges)
    base = g.mix_color(base, srgb("#d2d3d5"), g.mul(scratch, 0.5))
    rough = g.add(rough0, g.mul(streak, 0.08))
    rough = g.add(rough, g.mul(smudge, 0.08))
    rough = g.mix_float(rough, rough0 * 0.6, edges)
    rough = g.mix_float(rough, rough0 + 0.18, g.mul(scratch, 0.8))
    metal = g.value(1.0)
    # non-metal layers on top: pits, rust, grime
    base = g.mix_color(base, srgb("#3a2a20"), pit)
    base = g.mix_color(base, srgb("#6e3418"), rustm)
    base = g.mix_color(base, srgb("#2a2622"), g.mul(grime, 0.9))
    rough = g.mix_float(rough, 0.85, g.maxf(rustm, pit))
    rough = g.mix_float(rough, 0.75, grime)
    metal = g.sub(1.0, g.clamp01(g.add(g.add(g.mul(rustm, 0.95), pit), g.mul(grime, 0.8))))
    height = g.sub(g.add(g.mul(streak, 0.06), g.mul(dents, 0.3 if finish == "forged" else 0.0)),
                   g.add(g.mul(scratch, 0.5), g.mul(pit, 0.8)))
    return Surface(base, rough, metal, height, 0.00018)


@preset
def steel_blackened(c, wear=0.5, dirt=0.4, rust=0.25, scale=1.0, P=None, P2=None, uv_size=1.0):
    """Blackened / oil-quenched armour steel: a dark oxide skin (non-metal) worn
    through to bright steel on the edges and high points, soft hammer dents,
    rust and grime in the hollows."""
    g = c.g
    P = _P(c, P)
    dents = g.remap(g.noise(P, 55.0 * scale, 2.0), 0.3, 0.7)
    tone = g.remap(g.noise(P, 30.0 * scale, 4.0), 0.4, 0.65)
    rubbed = g.mul(g.remap(g.noise(P, 12.0 * scale, 4.0), 0.64, 0.76), wear * 0.6)   # broad polished patches
    bare = g.clamp01(g.add(w.edge_wear(c, wear, P), rubbed))
    rustm = g.mul(g.mul(g.remap(g.noise(P, 18.0 * scale, 5.0, 0.65), 0.6, 0.74),
                        g.add(g.mul(c.cavity, 0.8), 0.25)), rust * 2.0)
    grime = w.cavity_dirt(c, dirt, P)

    oxide = g.mix_color(srgb("#2a2b2e"), srgb("#3a3b3f"), tone)
    oxide = g.mix_color(oxide, srgb("#232426"), g.mul(dents, 0.4))
    base = g.mix_color(oxide, srgb("#b9bbbe"), bare)
    base = g.mix_color(base, srgb("#6b3317"), rustm)
    base = g.mix_color(base, srgb("#26221e"), g.mul(grime, 0.8))
    rough = g.mix_float(g.add(0.38, g.mul(tone, 0.14)), 0.24, bare)
    rough = g.mix_float(rough, 0.86, rustm)
    rough = g.mix_float(rough, 0.72, grime)
    # the oxide skin is a dark dielectric; only the worn-through steel is metal
    metal = bare
    metal = g.mul(metal, g.sub(1.0, g.clamp01(g.add(rustm, g.mul(grime, 0.7)))))
    forge = g.remap(g.noise(P, 320.0 * scale, 3.0, 0.6), 0.3, 0.7)      # fine forging texture
    height = g.sub(g.add(g.mul(dents, 0.5), g.mul(forge, 0.12)), g.mul(rustm, 0.2))
    return Surface(base, rough, metal, height, 0.0005)


@preset
def iron_rusted(c, rust=0.7, dirt=0.4, scale=1.0, P=None, P2=None, uv_size=1.0):
    """Heavily rusted wrought iron: layered flaking rust over pitted metal."""
    g = c.g
    P = _P(c, P)
    cover = g.remap(g.noise(P, 6.0 * scale, 6.0, 0.6), 0.62 - 0.35 * rust, 0.635 - 0.35 * rust)
    flakes = g.remap(g.voronoi(P, 90.0 * scale, "F2"), 0.2, 0.9)
    tone = g.remap(g.noise(P, 40.0 * scale, 6.0, 0.7), 0.3, 0.7)
    pit = w.pits(c, 400.0 * scale, 0.8, P)
    edges = w.edge_wear(c, 0.4, P)
    rust_col = g.mix_color(srgb("#5a2410"), srgb("#8c4a22"), tone)
    rust_col = g.mix_color(rust_col, srgb("#3a1a0c"), g.mul(flakes, 0.5))
    base = g.mix_color(srgb("#c6c8ca"), rust_col, g.clamp01(g.sub(cover, g.mul(edges, 0.5))))
    base = g.mix_color(base, srgb("#2a1a12"), g.mul(pit, 0.8))
    grime = w.cavity_dirt(c, dirt, P)
    base = g.mix_color(base, srgb("#2a2420"), g.mul(grime, 0.7))
    rough = g.mix_float(0.45, g.add(0.82, g.mul(tone, 0.12)), cover)
    metal = g.math("LESS_THAN", g.add(g.add(cover, g.mul(pit, 2.0)), grime), 0.25)
    height = g.sub(g.add(g.mul(cover, 0.5), g.mul(flakes, g.mul(cover, 0.4))), g.mul(pit, 0.6))
    return Surface(base, rough, metal, height, 0.0012)


@preset
def precious_metal(c, kind="brass", wear=0.5, dirt=0.4, patina=0.4, scale=1.0, P=None, P2=None, uv_size=1.0):
    """Brass, bronze, gold, silver or copper with tarnish/patina in the hollows
    and polished high points."""
    g = c.g
    P = _P(c, P)
    cols = {"brass": ("#e3c27a", "#c9a45c", "#4f7a62"), "bronze": ("#c99a6a", "#a8774c", "#3f7d6a"),
            "gold": ("#ffd77a", "#f2c45a", "#7a5a2a"), "silver": ("#f2f2f0", "#d9d9d6", "#3a3a38"),
            "copper": ("#f3a985", "#d98a63", "#4f8f7a")}[kind]
    tone = g.remap(g.noise(P, 20.0 * scale, 4.0), 0.4, 0.65)
    tarnish = g.clamp01(g.mul(g.add(c.cavity, g.mul(g.remap(c.ao, 0.95, 0.5), 0.5)), patina * 2.0))
    tarnish = g.mul(tarnish, w.breakup(c, 7.0, 0.3, 0.7, P))
    scratch = w.scratches(c, scale, P)
    edges = w.edge_wear(c, wear, P)
    base = g.mix_color(srgb(cols[1]), srgb(cols[0]), tone)
    base = g.mix_color(base, srgb(cols[0]), edges)
    base = g.mix_color(base, srgb(cols[2]), tarnish)
    base = g.mix_color(base, srgb("#2a2420"), g.mul(w.cavity_dirt(c, dirt, P), 0.6))
    rough = g.mix_float(g.add(0.26, g.mul(tone, 0.1)), 0.14, edges)
    rough = g.mix_float(rough, 0.4, g.mul(scratch, 0.7))
    rough = g.mix_float(rough, 0.8, tarnish)
    metal = g.sub(1.0, g.mul(tarnish, 0.95))
    height = g.sub(g.mul(g.remap(g.noise(P, 80.0 * scale, 3.0), 0.3, 0.7), 0.4), g.mul(scratch, 0.5))
    return Surface(base, rough, metal, height, 0.0002)


@preset
def chainmail(c, ring=0.0085, rust=0.4, dirt=0.5, P=None, P2=None, uv_size=1.0):
    """Riveted mail: interlocking rings on a diagonal grid (2D pattern coords),
    dark gaps, per-ring variation, rust and grime between the rings."""
    g = c.g
    P = _P(c, P)
    p2 = _P2(c, P2, uv_size)
    ring = _snap(ring, c, P2, uv_size)

    def grid(offset):
        m = g.mapping(p2, (1 / ring, 1 / ring, 1.0), loc=(offset, offset, 0.0))
        d = g.voronoi(m, 1.0, randomness=0.0, dims="2D")
        r = g.remap(g.math("ABSOLUTE", g.sub(d, 0.44)), 0.0, 0.16, 1.0, 0.0)
        return g.math("POWER", r, 0.6)

    rings = g.maxf(grid(0.0), grid(0.5))
    var = g.remap(g.noise(P, 160.0, 1.0), 0.3, 0.7)
    rustm = g.mul(g.mul(g.remap(g.noise(P, 30.0, 4.0), 0.5, 0.68), g.sub(1.15, rings)), rust * 2.0)
    grime = g.mul(g.inv(rings), dirt)
    metal_col = g.mix_color(srgb("#b4b6b9"), srgb("#d0d2d5"), var)
    solid = g.smooth(rings, 0.25, 0.45)
    base = g.mix_color(srgb("#262422"), metal_col, solid)
    base = g.mix_color(base, srgb("#5e2c12"), g.clamp01(rustm))
    base = g.mix_color(base, srgb("#262220"), g.mul(grime, 0.6))
    rough = g.mix_float(g.mix_float(0.85, 0.36, solid), 0.88, g.clamp01(rustm))
    metal = g.mul(solid, g.math("LESS_THAN", g.add(rustm, g.mul(grime, 1.2)), 0.35))
    return Surface(base, rough, metal, rings, 0.0012)


# --- textiles --------------------------------------------------------------------------
@preset
def wool(c, color="#4b4a33", quilted=True, channel=0.026, wear=0.4, dirt=0.5, P=None, P2=None, uv_size=1.0,
         channel_axis="X"):
    """Heavy felted wool / padded gambeson: twill weave, fuzz, optional sewn
    quilting channels running along the 2D coordinate `channel_axis`."""
    g = c.g
    P = _P(c, P)
    p2 = g.xyz(_P2(c, P2, uv_size))
    col = srgb(color)
    channel = _snap(channel, c, P2, uv_size)
    weave = g.remap(g.wave(g.mapping(P, (1, 1, 1), (0, 0, math.radians(45))), 900.0, 2.0), 0.3, 0.7)
    fuzz = g.remap(g.noise(P, 2600.0, 2.0), 0.35, 0.65)
    var = g.remap(g.noise(P, 14.0, 3.0), 0.35, 0.65)
    height = g.add(g.mul(weave, 0.25), g.mul(fuzz, 0.15))
    base = g.mix_color(_shade(col, 0.75), _shade(col, 1.2), var)
    base = g.mix_color(base, _shade(col, 0.6), g.mul(fuzz, 0.3))
    if quilted:
        ch = g.math("ABSOLUTE", g.math("SINE", g.mul(p2[channel_axis], math.pi / channel)))
        line = g.remap(ch, 0.0, 0.16, 1.0, 0.0)
        base = g.mix_color(base, _shade(col, 0.4), g.mul(line, 0.8))
        height = g.add(height, g.math("POWER", ch, 0.5))
    base = g.mix_color(base, _shade(col, 0.35), g.mul(w.cavity_dirt(c, dirt, P), 0.85))
    base = g.mix_color(base, _shade(col, 1.6), g.mul(w.edge_wear(c, wear, P), 0.5))
    rough = g.add(g.mix_float(0.80, 0.95, fuzz), g.mul(g.sub(var, 0.5), 0.1))
    return Surface(base, rough, 0.0, height, 0.0006)


@preset
def linen(c, color="#b9ab8c", thread=0.0011, wear=0.3, dirt=0.5, P=None, P2=None, uv_size=1.0):
    """Plain-weave linen / canvas: over-under threads with slub irregularity."""
    g = c.g
    P = _P(c, P)
    p2 = g.xyz(_P2(c, P2, uv_size))
    col = srgb(color)
    thread = _snap(thread, c, P2, uv_size)
    slub = g.mul(g.noise(P, 300.0, 2.0), 0.6)
    wx = g.math("SINE", g.mul(g.add(p2["X"], g.mul(slub, thread)), math.pi / thread))
    wy = g.math("SINE", g.mul(g.add(p2["Y"], g.mul(slub, thread)), math.pi / thread))
    over = g.math("GREATER_THAN", g.mul(wx, wy), 0.0)
    threads = g.mix_float(g.math("ABSOLUTE", wy), g.math("ABSOLUTE", wx), over)
    var = g.remap(g.noise(P, 12.0, 3.0), 0.35, 0.65)
    base = g.mix_color(_shade(col, 0.8), _shade(col, 1.1), var)
    base = g.mix_color(_shade(col, 0.55), base, threads)
    base = g.mix_color(base, _shade(col, 0.45), w.stains(c, dirt, 4.0, P))
    base = g.mix_color(base, _shade(col, 0.4), g.mul(w.cavity_dirt(c, dirt, P), 0.8))
    rough = g.mix_float(0.92, 0.82, threads)
    return Surface(base, rough, 0.0, threads, 0.00025)


# --- wood ------------------------------------------------------------------------------
@preset
def wood(c, species="oak", axis="Z", wear=0.5, dirt=0.4, varnish=0.0, scale=1.0, P=None, P2=None, uv_size=1.0):
    """Sawn hardwood with growth rings, open pores and handled wear. `axis` is
    the grain direction in object space; `varnish` 0..1 adds a satin finish."""
    g = c.g
    P = _P(c, P)
    cols = {"oak": ("#6b4a2c", "#a07a50"), "ash": ("#9c8466", "#c8b08c"),
            "walnut": ("#3d2a1c", "#6a4a32"), "pine": ("#a0784a", "#d0a870")}[species]
    stretch = {"X": (2, 40, 40), "Y": (40, 2, 40), "Z": (40, 40, 2)}[axis]
    q = g.mapping(P, tuple(s * scale for s in stretch))
    rings = g.wave(q, 1.2, 8.0, 3.0, "RINGS", profile="SAW")
    pores = g.remap(g.noise(g.mapping(P, tuple(s * 18 * scale for s in stretch)), 1.0, 2.0), 0.6, 0.75)
    var = g.remap(g.noise(P, 5.0 * scale, 3.0), 0.35, 0.65)
    edges = w.edge_wear(c, wear, P)
    base = g.mix_color(srgb(cols[0]), srgb(cols[1]), g.mul(rings, 0.8))
    base = g.mix_color(base, _shade(srgb(cols[0]), 0.7), g.mul(var, 0.5))
    base = g.mix_color(base, _shade(srgb(cols[0]), 0.5), g.mul(pores, 0.7))
    base = g.mix_color(base, _shade(srgb(cols[1]), 1.15), g.mul(edges, 0.6))
    base = g.mix_color(base, srgb("#221a14"), g.mul(w.cavity_dirt(c, dirt, P), 0.85))
    rough = g.mix_float(g.add(g.add(0.56, g.mul(pores, 0.2)), g.mul(rings, 0.12)), 0.38, varnish)
    rough = g.mix_float(rough, 0.42, g.mul(edges, 0.7))
    height = g.sub(g.mul(rings, 0.25), g.mul(pores, 0.6))
    return Surface(base, rough, 0.0, height, 0.0003)


# --- stone -----------------------------------------------------------------------------
@preset
def stone(c, color="#7d776d", wear=0.4, dirt=0.5, moss=0.0, scale=1.0, P=None, P2=None, uv_size=1.0):
    """Weathered sandstone/limestone: grains, cracks, soot, optional moss on top faces."""
    g = c.g
    P = _P(c, P)
    col = srgb(color)
    grain = g.remap(g.noise(P, 300.0 * scale, 4.0), 0.3, 0.7)
    blot = g.remap(g.noise(P, 4.0 * scale, 5.0, 0.6), 0.35, 0.7)
    crack = g.remap(g.voronoi(g.warp(P, 6.0 * scale, 0.06), 3.5 * scale, "DISTANCE_TO_EDGE"), 0.0, 0.01, 1.0, 0.0)
    crack = g.mul(crack, g.remap(g.noise(P, 2.5, 3.0), 0.52, 0.62))
    mossm = g.mul(g.mul(c.up, g.remap(g.noise(P, 8.0, 5.0, 0.6), 0.5, 0.6)), moss * 2.0)
    base = g.mix_color(_shade(col, 0.8), _shade(col, 1.15), blot)
    base = g.mix_color(base, _shade(col, 0.7), g.mul(grain, 0.4))
    base = g.mix_color(base, _shade(col, 0.3), crack)
    base = g.mix_color(base, _shade(col, 0.45), g.mul(w.cavity_dirt(c, dirt, P), 0.8))
    base = g.mix_color(base, srgb("#4a5a2a"), g.clamp01(mossm))
    rough = g.add(0.78, g.mul(grain, 0.15))
    height = g.sub(g.add(g.mul(grain, 0.3), g.mul(blot, 0.4)), g.mul(crack, 1.2))
    return Surface(base, rough, 0.0, height, 0.0015)


def get(name):
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from None
