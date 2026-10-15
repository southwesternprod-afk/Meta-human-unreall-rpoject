# PBR standards used by aaa-materials

Values are for the metal/roughness workflow used by Unreal Engine 5. "sRGB" means
the 0-255 value stored in the BaseColor texture.

## Albedo (BaseColor)

| Material | sRGB (approx.) | Notes |
|---|---|---|
| Darkest allowed non-metal (charcoal, black leather) | 30-50 | below 30 looks like a hole |
| Brown / vegetable-tanned leather | 45-110 | dye mottling +-25% |
| Wool, felt (dyed) | 50-120 | |
| Linen, canvas (undyed) | 150-190 | |
| Oak, walnut, ash | 60-200 | rings darker than early wood |
| Stone, concrete | 100-180 | |
| Rust | R 90-140, G 40-70, B 20-35 | non-metal |
| Verdigris / patina | ~ (95,143,122) | non-metal |
| Brightest allowed non-metal (fresh snow) | 240 | |
| Iron / steel (bare) | 180-200 neutral | metal |
| Silver | 240-250 | metal |
| Gold | (255,215,120) | metal |
| Brass | (227,194,122) | metal |
| Copper | (243,169,133) | metal |

Dark "blackened" or oxidised steel is a non-metal oxide layer (sRGB 35-60,
metallic 0) with bright bare metal showing through on worn edges.

## Metallic
0 or 1. Grey values only in thin anti-aliased transitions or under partial
dirt; large grey areas look like plastic-coated metal.

## Roughness (linear, ORM green)
| Surface | Roughness |
|---|---|
| Polished steel / gold | 0.10-0.25 |
| Brushed / worn steel | 0.25-0.45 |
| Oiled leather, burnished edges | 0.35-0.50 |
| Matte leather | 0.55-0.75 |
| Wood (raw) | 0.55-0.80 |
| Wool, linen | 0.80-0.95 |
| Rust, dust, dry dirt | 0.80-0.95 |

Variation inside one material (std > 0.03) matters more than the mean.

## Normal maps
* Unreal: DirectX convention (green = -Y). Blender / glTF / Unity: OpenGL (+Y).
* Unit length, blue >= 0.5 everywhere.
* Bake height detail with real-world depth: leather grain 0.2-0.5 mm, metal
  dents 0.2-0.6 mm, rust flakes 1 mm, stone 1-3 mm.

## Ambient occlusion (ORM red)
Cavity-scale only (a few cm); large-scale occlusion is the engine's job. Avoid
values below 0.15.

## Texel density targets
| Asset | px/cm | Example |
|---|---|---|
| Hero character, close-up | 10.24-20.48 | 4K on a torso garment |
| Character, third person | 10.24 | |
| Weapons (first person) | 20.48+ | 2K on a 1 m sword |
| Props | 5.12 | |
| Environment | 5.12 | tileables: 2K per 2 m |

## Texture set
`T_<Name>_BaseColor` (sRGB, Default compression), `T_<Name>_ORM` (linear, Masks),
`T_<Name>_Normal` (linear, Normalmap, DirectX), optional `T_<Name>_Height` (linear,
for parallax / Nanite displacement). Square, power of two.
