"""Weathering masks shared by the smart materials. Every function returns a
0..1 shader socket. ``c`` is an engine.Context; ``P`` overrides the coordinates
(object-space metres by default). Masks combine the baked geometry masks
(edge, cavity, AO, up) with noise so wear never looks uniform."""


def _P(c, P):
    return c.P if P is None else P


def breakup(c, scale=8.0, lo=0.42, hi=0.62, P=None):
    """Large soft blotches used to break up any other mask."""
    return c.g.remap(c.g.noise(_P(c, P), scale, 4.0, 0.55), lo, hi)


def edge_wear(c, amount=0.5, P=None):
    """Worn convex edges: paint / oxide rubbed off, leather burnished."""
    g = c.g
    chip = g.remap(g.noise(_P(c, P), 60.0, 6.0, 0.7), 0.35, 0.65)
    m = g.math("SUBTRACT", g.mul(c.edge, g.add(0.6, g.mul(chip, 0.8))), g.sub(1.0, amount))
    return g.clamp01(g.mul(m, 2.0))


def cavity_dirt(c, amount=0.5, P=None):
    """Grime packed into creases and corners, plus a little in broad occlusion."""
    g = c.g
    occl = g.remap(c.ao, 0.9, 0.35)
    m = g.maxf(c.cavity, g.mul(occl, 0.6))
    return g.clamp01(g.mul(g.mul(m, breakup(c, 5.0, 0.3, 0.7, P)), amount * 2.0))


def dust(c, amount=0.3, P=None):
    """Dust settling on upward-facing surfaces."""
    g = c.g
    return g.clamp01(g.mul(g.mul(c.up, breakup(c, 6.0, 0.35, 0.7, P)), amount * 1.6))


def stains(c, amount=0.4, scale=3.5, P=None):
    """Large darker stains (water, oil, sweat)."""
    g = c.g
    return g.mul(g.remap(g.noise(_P(c, P), scale, 5.0, 0.65), 0.56, 0.70), amount)


def scratches(c, density=1.0, P=None):
    """Thin scratch lines in three directions, gated so they come in clusters."""
    g = c.g
    P = _P(c, P)

    def layer(scale, rot, width):
        d = g.voronoi(g.mapping(P, scale, rot), 1.0, "DISTANCE_TO_EDGE")
        return g.remap(d, 0.0, width, 1.0, 0.0)

    s = g.maxf(layer((30 * density, 30 * density, 3), (0.3, 0.35, 0), 0.02),
               layer((52 * density, 52 * density, 5), (0.2, -0.55, 0.2), 0.016))
    s = g.maxf(s, g.mul(layer((140 * density, 140 * density, 12), (0, 1.15, 0.4), 0.03), 0.45))
    return g.mul(s, breakup(c, 9.0, 0.45, 0.62, P))


def creases(c, amount=0.5, P=None, stretch=(60, 60, 14)):
    """Soft crease lines for leather and heavy cloth."""
    g = c.g
    n = g.noise(g.mapping(_P(c, P), stretch), 1.0, 6.0, 0.6)
    line = g.remap(g.math("ABSOLUTE", g.sub(n, 0.5)), 0.0, 0.035, 1.0, 0.0)
    return g.mul(g.mul(line, breakup(c, 4.0, 0.5, 0.66, P)), amount * 2.0)


def pits(c, scale=650.0, amount=0.5, P=None):
    """Small corrosion pits / pores."""
    g = c.g
    P = _P(c, P)
    return g.mul(g.mul(g.remap(g.voronoi(P, scale), 0.0, 0.12, 1.0, 0.0),
                       g.remap(g.noise(P, 25.0, 2.0), 0.5, 0.62)), amount * 2.0)
