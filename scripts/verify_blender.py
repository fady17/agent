# type: ignore

"""
scripts/verify_blender.py

Smoke test for the blender optional dependencies.
Run after: uv sync --extra blender

    uv run python scripts/verify_blender.py

Note: bpy installed from PyPI is the headless standalone module.
It can run scripts and access most bpy.data APIs but has no
viewport, operators (bpy.ops), or render engine support.
For full operator support the script must run inside Blender itself
via blender_server.py.
"""

import sys


def check(label: str, import_path: str) -> bool:
    try:
        parts = import_path.split(".")
        mod = __import__(parts[0])
        for part in parts[1:]:
            mod = getattr(mod, part)
        version = getattr(mod, "__version__", None) or getattr(mod, "version", "?")
        print(f"  [ok]  {label:<20} {version}")
        return True
    except Exception as e:
        print(f"  [!!]  {label:<20} MISSING — {e}")
        return False


def main() -> None:
    print("\nBlender optional dependencies")
    print("─" * 50)

    results = [
        check("bpy",       "bpy"),
        check("bmesh",     "bmesh"),
        check("mathutils", "mathutils"),
    ]

    print()

    # Sanity check: can we access bpy.data?
    try:
        import bpy
        scene_count = len(bpy.data.scenes)
        print(f"  [ok]  bpy.data.scenes accessible ({scene_count} scene(s) in default blend)")
        results.append(True)
    except Exception as e:
        print(f"  [!!]  bpy.data check failed — {e}")
        results.append(False)

    # mathutils vector sanity
    try:
        from mathutils import Vector
        v = Vector((1.0, 0.0, 0.0))
        assert abs(v.length - 1.0) < 1e-6
        print("  [ok]  mathutils.Vector works")
        results.append(True)
    except Exception as e:
        print(f"  [!!]  mathutils.Vector failed — {e}")
        results.append(False)

    print("\n" + "─" * 50)
    if all(results):
        print("  All blender deps OK.\n")
        print("  Note: bpy from PyPI is headless.")
        print("  bpy.ops (operators) require running inside Blender.\n")
    else:
        failed = results.count(False)
        print(f"  {failed} check(s) failed.\n")
        print("  Install with: uv sync --extra blender\n")
        sys.exit(1)


if __name__ == "__main__":
    main()