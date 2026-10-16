"""aaa_materials - procedural AAA PBR materials for headless Blender + Unreal.

    import aaa_materials as aaa
    aaa.engine.bake(obj, out_dir, "LeatherBrown",
                    lambda c: [(None, aaa.library.leather(c, color="#4a2f1f", wear=0.6))])

Modules: engine (bake pipeline), library (smart materials), weathering (masks),
lookdev (material-ball renders), unreal (UE import + master material script),
validate (PBR checks, no Blender needed).
"""

import importlib

__all__ = ["engine", "library", "weathering", "lookdev", "unreal", "validate", "padding"]


def __getattr__(name):            # lazy: validate/unreal work without bpy installed
    if name in __all__:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(name)
