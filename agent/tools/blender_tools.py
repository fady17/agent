"""
agent/tools/blender_tools.py

Core 3D tools — four operations sent through the Blender socket bridge.

The agent side never imports bpy. Each tool:
    1. Builds a Python script string
    2. Sends it through BlenderBridge.send_script()
    3. Parses the JSON output into a typed result dataclass

The scripts themselves run inside Blender's interpreter (Python 3.11 + bpy).
They print JSON to stdout which the server captures and returns.

Tools:
    get_scene_info  — objects, meshes, materials, render settings
    export_glb      — export scene or selection as GLB
    fix_ngons       — triangulate n-gon faces across all mesh objects
    generate_lod    — create LOD variants at configurable decimate ratios
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agent.core.logger import get_logger
from agent.tools.blender_bridge import BlenderBridge, ScriptResult

log = get_logger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SceneInfo:
    object_count:   int
    mesh_count:     int
    material_count: int
    light_count:    int
    camera_count:   int
    render_engine:  str
    objects:        list[dict] = field(default_factory=list)
    success:        bool = True
    error:          str  = ""


@dataclass(frozen=True)
class ExportResult:
    path:         str
    object_count: int
    success:      bool
    error:        str = ""


@dataclass(frozen=True)
class NgonResult:
    objects_checked:  int
    objects_fixed:    int
    faces_triangulated: int
    success:          bool
    error:            str = ""


@dataclass(frozen=True)
class LodResult:
    source_object:  str
    lods_created:   list[dict] = field(default_factory=list)
    success:        bool = True
    error:          str  = ""


# ── Scripts (run inside Blender) ──────────────────────────────────────────────
# Each script prints a single JSON line to stdout.
# The server captures stdout and returns it as the result string.

_GET_SCENE_INFO_SCRIPT = """
import bpy, json

objects = list(bpy.data.objects)
result = {
    "object_count":   len(objects),
    "mesh_count":     len(bpy.data.meshes),
    "material_count": len(bpy.data.materials),
    "light_count":    sum(1 for o in objects if o.type == "LIGHT"),
    "camera_count":   sum(1 for o in objects if o.type == "CAMERA"),
    "render_engine":  bpy.context.scene.render.engine,
    "objects": [
        {
            "name":     o.name,
            "type":     o.type,
            "visible":  o.visible_get(),
            "poly_count": len(o.data.polygons) if o.type == "MESH" and o.data else 0,
        }
        for o in objects
    ],
}
print(json.dumps(result))
"""

_FIX_NGONS_SCRIPT = """
import bpy, bmesh, json

objects_checked = 0
objects_fixed = 0
total_faces_triangulated = 0

for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    objects_checked += 1
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    ngons = [f for f in bm.faces if len(f.verts) > 4]
    if ngons:
        bmesh.ops.triangulate(bm, faces=ngons, quad_method="BEAUTY", ngon_method="BEAUTY")
        bm.to_mesh(mesh)
        mesh.update()
        objects_fixed += 1
        total_faces_triangulated += len(ngons)

    bm.free()

print(json.dumps({
    "objects_checked":    objects_checked,
    "objects_fixed":      objects_fixed,
    "faces_triangulated": total_faces_triangulated,
}))
"""


def _export_glb_script(
    output_path: str,
    selected_only: bool,
    use_draco: bool,
) -> str:
    return f"""
import bpy, json

output_path = {output_path!r}
selected_only = {selected_only!r}

export_kwargs = dict(
    filepath=output_path,
    use_selection=selected_only,
    export_apply=True,
    export_yup=True,
)

if {use_draco!r}:
    export_kwargs["export_draco_mesh_compression_enable"] = True
    export_kwargs["export_draco_mesh_compression_level"] = 6

bpy.ops.export_scene.gltf(**export_kwargs)

exported = [
    o for o in bpy.data.objects
    if o.type == "MESH" and (not selected_only or o.select_get())
]
print(json.dumps({{
    "path":         output_path,
    "object_count": len(exported),
}}))
"""


def _generate_lod_script(object_name: str, ratios: list[float]) -> str:
    ratios_repr = repr(ratios)
    return f"""
import bpy, json

source_name = {object_name!r}
ratios = {ratios_repr}

source = bpy.data.objects.get(source_name)
if source is None:
    print(json.dumps({{"error": f"Object {{source_name!r}} not found"}}))
    raise SystemExit(1)

if source.type != "MESH":
    print(json.dumps({{"error": f"Object {{source_name!r}} is not a mesh (type={{source.type}})"}}))
    raise SystemExit(1)

lods = []
original_poly_count = len(source.data.polygons)

for i, ratio in enumerate(ratios):
    lod_name = f"{{source_name}}_LOD{{i+1}}"

    # Duplicate the source object
    new_mesh = source.data.copy()
    new_obj  = source.copy()
    new_obj.data = new_mesh
    new_obj.name = lod_name
    bpy.context.collection.objects.link(new_obj)

    # Apply decimate modifier
    mod = new_obj.modifiers.new(name="Decimate_LOD", type="DECIMATE")
    mod.ratio = ratio
    bpy.context.view_layer.objects.active = new_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    poly_count = len(new_obj.data.polygons)
    lods.append({{
        "name":       lod_name,
        "ratio":      ratio,
        "poly_count": poly_count,
        "reduction":  round(1.0 - poly_count / max(original_poly_count, 1), 3),
    }})

print(json.dumps({{
    "source_object":         source_name,
    "original_poly_count":   original_poly_count,
    "lods":                  lods,
}}))
"""


# ── Tool functions ────────────────────────────────────────────────────────────

async def get_scene_info(bridge: BlenderBridge) -> SceneInfo:
    """
    Return structured information about the current Blender scene.
    Safe read-only operation — does not modify anything.
    """
    log.debug("blender_tools.get_scene_info")
    result = await bridge.send_script(_GET_SCENE_INFO_SCRIPT)
    return _parse_scene_info(result)


async def export_glb(
    bridge: BlenderBridge,
    output_path: Path | str,
    *,
    selected_only: bool = False,
    use_draco: bool = False,
    timeout: float = 60.0,
) -> ExportResult:
    """
    Export the scene (or selection) as a GLB file.

    Args:
        bridge:        Active BlenderBridge connection.
        output_path:   Destination .glb file path (inside Blender's filesystem).
        selected_only: Export only selected objects. Default False (full scene).
        use_draco:     Enable Draco mesh compression. Default False.
        timeout:       Export timeout in seconds (large scenes may be slow).
    """
    path_str = str(output_path)
    script   = _export_glb_script(path_str, selected_only, use_draco)
    log.info("blender_tools.export_glb", path=path_str, draco=use_draco)
    result = await bridge.send_script(script, timeout=timeout)
    return _parse_export_result(result, path_str)


async def fix_ngons(bridge: BlenderBridge) -> NgonResult:
    """
    Find and triangulate all n-gon faces (>4 vertices) across every mesh.

    N-gons cause issues in game engines and real-time renderers.
    Uses bmesh BEAUTY triangulation for best results.
    """
    log.info("blender_tools.fix_ngons")
    result = await bridge.send_script(_FIX_NGONS_SCRIPT, timeout=60.0)
    return _parse_ngon_result(result)


async def generate_lod(
    bridge: BlenderBridge,
    object_name: str,
    *,
    ratios: list[float] | None = None,
    timeout: float = 120.0,
) -> LodResult:
    """
    Generate LOD (Level of Detail) variants of a mesh object.

    Creates duplicates with Decimate modifiers applied at each ratio.
    LODs are named {object_name}_LOD1, _LOD2, etc.

    Args:
        bridge:      Active BlenderBridge connection.
        object_name: Name of the source mesh object in Blender.
        ratios:      Decimate ratios for each LOD level.
                     Default: [0.5, 0.25, 0.1] → 50%, 25%, 10% of original.
        timeout:     Generation timeout for complex meshes.
    """
    effective_ratios = [0.5, 0.25, 0.1] if ratios is None else ratios


    if not effective_ratios:
        return LodResult(source_object=object_name, success=False, error="ratios list is empty")

    for r in effective_ratios:
        if not (0.0 < r < 1.0):
            return LodResult(
                source_object=object_name,
                success=False,
                error=f"Invalid ratio {r} — must be between 0 and 1 (exclusive)",
            )

    script = _generate_lod_script(object_name, effective_ratios)
    log.info("blender_tools.generate_lod", object=object_name, ratios=effective_ratios)
    result = await bridge.send_script(script, timeout=timeout)
    return _parse_lod_result(result, object_name)


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_json_result(result: ScriptResult) -> tuple[bool, dict, str]:
    """
    Parse a ScriptResult containing JSON stdout into (success, data, error).
    """
    if not result.success:
        return False, {}, result.error

    output = result.output.strip()
    if not output:
        return False, {}, "Empty response from Blender"

    # Take the last non-empty line — scripts may print debug before final JSON
    lines = [l for l in output.splitlines() if l.strip()]  # noqa: E741
    if not lines:
        return False, {}, "No output lines from Blender"

    try:
        data = json.loads(lines[-1])
        if "error" in data:
            return False, data, data["error"]
        return True, data, ""
    except json.JSONDecodeError as exc:
        return False, {}, f"JSON parse failed: {exc} | output={output[:200]!r}"


def _parse_scene_info(result: ScriptResult) -> SceneInfo:
    ok, data, error = _parse_json_result(result)
    if not ok:
        log.warning("blender_tools.scene_info_failed", error=error)
        return SceneInfo(0, 0, 0, 0, 0, "", success=False, error=error)
    return SceneInfo(
        object_count=data.get("object_count", 0),
        mesh_count=data.get("mesh_count", 0),
        material_count=data.get("material_count", 0),
        light_count=data.get("light_count", 0),
        camera_count=data.get("camera_count", 0),
        render_engine=data.get("render_engine", ""),
        objects=data.get("objects", []),
        success=True,
    )


def _parse_export_result(result: ScriptResult, path: str) -> ExportResult:
    ok, data, error = _parse_json_result(result)
    if not ok:
        log.warning("blender_tools.export_failed", path=path, error=error)
        return ExportResult(path=path, object_count=0, success=False, error=error)
    return ExportResult(
        path=data.get("path", path),
        object_count=data.get("object_count", 0),
        success=True,
    )


def _parse_ngon_result(result: ScriptResult) -> NgonResult:
    ok, data, error = _parse_json_result(result)
    if not ok:
        log.warning("blender_tools.ngon_failed", error=error)
        return NgonResult(0, 0, 0, success=False, error=error)
    return NgonResult(
        objects_checked=data.get("objects_checked", 0),
        objects_fixed=data.get("objects_fixed", 0),
        faces_triangulated=data.get("faces_triangulated", 0),
        success=True,
    )


def _parse_lod_result(result: ScriptResult, object_name: str) -> LodResult:
    ok, data, error = _parse_json_result(result)
    if not ok:
        log.warning("blender_tools.lod_failed", object=object_name, error=error)
        return LodResult(source_object=object_name, success=False, error=error)
    return LodResult(
        source_object=data.get("source_object", object_name),
        lods_created=data.get("lods", []),
        success=True,
    )