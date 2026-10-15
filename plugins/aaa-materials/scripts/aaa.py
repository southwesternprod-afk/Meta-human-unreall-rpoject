"""aaa-materials command line.

    python3 aaa.py presets
    python3 aaa.py tile --preset leather --name LeatherBrown --size 0.5 \
        --param color=#4a2f1f --param wear=0.6 --out Art/Materials/LeatherBrown
    python3 aaa.py validate Art/Materials/LeatherBrown --tileable
    python3 aaa.py ball Art/Materials/LeatherBrown LeatherBrown --tiling 2
    python3 aaa.py unreal Art/Materials/LeatherBrown LeatherBrown --tiling 2
    python3 aaa.py sheet out.png a_Ball.png b_Ball.png ...

`tile` bakes a seamless texture set from a preset, renders a material ball,
validates the set and writes the Unreal import script - one finished material.
Needs: pip install bpy==5.2.* numpy pillow (validate needs only numpy + pillow).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))


def parse_params(items):
    params = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if value.lower() in ("true", "false"):
            params[key] = value.lower() == "true"
            continue
        try:
            params[key] = float(value)
        except ValueError:
            params[key] = value
    return params


def cmd_presets(args):
    import inspect
    from aaa_materials import library
    for name, fn in sorted(library.PRESETS.items()):
        sig = ", ".join(f"{p.name}={p.default!r}" for p in list(inspect.signature(fn).parameters.values())[1:]
                        if p.default is not inspect.Parameter.empty and p.name not in ("P", "P2"))
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"{name:16} {doc}\n{'':16} {sig}")


def make_plane(size):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_plane_add(size=size)
    plane = bpy.context.object
    # the plane's own UVs run 0..1 across it: P2 = UV * size gives metres
    return plane


def cmd_tile(args):
    from aaa_materials import engine, library, lookdev, unreal, validate
    out = Path(args.out or f"Art/Materials/{args.name}")
    params = parse_params(args.param)
    fn = library.get(args.preset)
    plane = make_plane(args.size)
    engine.bake(plane, out, args.name, lambda c: [(None, fn(c, uv_size=args.size, **params))],
                res=args.res, tileable=True, samples=args.samples)
    script = unreal.write_import_script(out / f"Import_{args.name}_Unreal.py", args.name, tiling=args.ue_tiling)
    print("Unreal import script:", script)
    if not args.no_ball:
        ball = lookdev.material_ball(out, args.name, out / f"{args.name}_Ball.png", tiling=args.ball_tiling,
                                     samples=args.ball_samples)
        print("Material ball:", ball)
    rep = validate.validate_dir(out, args.name, tileable=True)[0]
    print(rep.text())
    return 2 if rep.worst == "FAIL" else 0


def cmd_validate(args):
    from aaa_materials import validate
    argv = [args.directory] + (["--name", args.name] if args.name else []) + \
           (["--tileable"] if args.tileable else []) + (["--json", args.json] if args.json else [])
    return validate.main(argv)


def cmd_ball(args):
    from aaa_materials import lookdev
    out = Path(args.out or Path(args.directory) / f"{args.name}_Ball.png")
    print(lookdev.material_ball(args.directory, args.name, out, tiling=args.tiling, samples=args.samples))


def cmd_unreal(args):
    from aaa_materials import unreal
    print(unreal.write_import_script(Path(args.directory) / f"Import_{args.name}_Unreal.py", args.name,
                                     dest=args.dest, tiling=args.tiling))


def cmd_sheet(args):
    from aaa_materials import lookdev
    print(lookdev.contact_sheet(args.images, args.out, columns=args.columns))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("presets", help="list smart-material presets and their parameters")

    t = sub.add_parser("tile", help="bake a seamless texture set from a preset")
    t.add_argument("--preset", required=True)
    t.add_argument("--name", required=True, help="texture set name, e.g. LeatherBrown")
    t.add_argument("--size", type=float, default=1.0, help="real-world size of one tile in metres")
    t.add_argument("--res", type=int, default=2048)
    t.add_argument("--samples", type=int, default=4)
    t.add_argument("--param", action="append", help="preset parameter key=value (repeatable)")
    t.add_argument("--out")
    t.add_argument("--ue-tiling", type=float, default=1.0, help="default Tiling in the Unreal instance")
    t.add_argument("--no-ball", action="store_true")
    t.add_argument("--ball-tiling", type=float, default=1.0)
    t.add_argument("--ball-samples", type=int, default=48)

    v = sub.add_parser("validate", help="check texture sets against PBR rules")
    v.add_argument("directory")
    v.add_argument("--name")
    v.add_argument("--tileable", action="store_true")
    v.add_argument("--json")

    b = sub.add_parser("ball", help="render a material ball for a baked set")
    b.add_argument("directory")
    b.add_argument("name")
    b.add_argument("--out")
    b.add_argument("--tiling", type=float, default=1.0)
    b.add_argument("--samples", type=int, default=48)

    u = sub.add_parser("unreal", help="write the Unreal import + material instance script")
    u.add_argument("directory")
    u.add_argument("name")
    u.add_argument("--dest")
    u.add_argument("--tiling", type=float, default=1.0)

    s = sub.add_parser("sheet", help="tile material-ball renders into one image")
    s.add_argument("out")
    s.add_argument("images", nargs="+")
    s.add_argument("--columns", type=int, default=3)

    args = ap.parse_args(argv)
    return {"presets": cmd_presets, "tile": cmd_tile, "validate": cmd_validate, "ball": cmd_ball,
            "unreal": cmd_unreal, "sheet": cmd_sheet}[args.cmd](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
