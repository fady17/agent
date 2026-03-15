# type: ignore
"""
tests/test_blender_tools.py

Tests for blender_tools.py — all bridge calls are mocked.
No Blender installation or bpy import required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.tools.blender_bridge import BlenderBridge, ScriptResult
from agent.tools.blender_tools import (
    ExportResult,
    LodResult,
    NgonResult,
    SceneInfo,
    export_glb,
    fix_ngons,
    generate_lod,
    get_scene_info,
    _parse_json_result,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_bridge(output: str, success: bool = True) -> BlenderBridge:
    bridge = MagicMock(spec=BlenderBridge)
    bridge.send_script = AsyncMock(
        return_value=ScriptResult(
            success=success,
            output=output,
            error="" if success else output,
            request_id="req-test",
        )
    )
    return bridge


def json_output(data: dict) -> str:
    return json.dumps(data)


# ── _parse_json_result ────────────────────────────────────────────────────────

def test_parse_json_valid() -> None:
    result = ScriptResult(success=True, output='{"key": "value"}', error="", request_id="r")
    ok, data, error = _parse_json_result(result)
    assert ok is True
    assert data == {"key": "value"}
    assert error == ""


def test_parse_json_script_failed() -> None:
    result = ScriptResult(success=False, output="", error="NameError: x", request_id="r")
    ok, data, error = _parse_json_result(result)
    assert ok is False
    assert "NameError" in error


def test_parse_json_empty_output() -> None:
    result = ScriptResult(success=True, output="", error="", request_id="r")
    ok, data, error = _parse_json_result(result)
    assert ok is False
    assert "Empty" in error


def test_parse_json_invalid_json() -> None:
    result = ScriptResult(success=True, output="not json", error="", request_id="r")
    ok, data, error = _parse_json_result(result)
    assert ok is False
    assert "JSON parse failed" in error


def test_parse_json_takes_last_line() -> None:
    """Scripts may print debug lines before final JSON."""
    result = ScriptResult(
        success=True,
        output="Processing...\nDone.\n{\"count\": 5}",
        error="",
        request_id="r",
    )
    ok, data, error = _parse_json_result(result)
    assert ok is True
    assert data["count"] == 5


def test_parse_json_data_has_error_key() -> None:
    result = ScriptResult(
        success=True,
        output='{"error": "Object not found"}',
        error="",
        request_id="r",
    )
    ok, data, error = _parse_json_result(result)
    assert ok is False
    assert "Object not found" in error


# ── get_scene_info ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_scene_info_success() -> None:
    data = {
        "object_count": 5, "mesh_count": 3, "material_count": 4,
        "light_count": 1, "camera_count": 1, "render_engine": "CYCLES",
        "objects": [{"name": "Cube", "type": "MESH", "visible": True, "poly_count": 6}],
    }
    bridge = make_bridge(json_output(data))
    result = await get_scene_info(bridge)
    assert isinstance(result, SceneInfo)
    assert result.success is True
    assert result.object_count == 5
    assert result.mesh_count == 3
    assert result.material_count == 4
    assert result.light_count == 1
    assert result.camera_count == 1
    assert result.render_engine == "CYCLES"
    assert len(result.objects) == 1
    assert result.objects[0]["name"] == "Cube"


@pytest.mark.asyncio
async def test_get_scene_info_bridge_failure() -> None:
    bridge = make_bridge("NameError: bpy not found", success=False)
    result = await get_scene_info(bridge)
    assert result.success is False
    assert result.error != ""


@pytest.mark.asyncio
async def test_get_scene_info_empty_scene() -> None:
    data = {
        "object_count": 0, "mesh_count": 0, "material_count": 0,
        "light_count": 0, "camera_count": 0, "render_engine": "EEVEE",
        "objects": [],
    }
    bridge = make_bridge(json_output(data))
    result = await get_scene_info(bridge)
    assert result.success is True
    assert result.object_count == 0
    assert result.objects == []


@pytest.mark.asyncio
async def test_get_scene_info_calls_bridge() -> None:
    bridge = make_bridge(json_output({"object_count": 1, "mesh_count": 1,
        "material_count": 0, "light_count": 0, "camera_count": 0,
        "render_engine": "EEVEE", "objects": []}))
    await get_scene_info(bridge)
    bridge.send_script.assert_called_once()


# ── export_glb ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_export_glb_success() -> None:
    data = {"path": "/tmp/scene.glb", "object_count": 3}
    bridge = make_bridge(json_output(data))
    result = await export_glb(bridge, "/tmp/scene.glb")
    assert isinstance(result, ExportResult)
    assert result.success is True
    assert result.path == "/tmp/scene.glb"
    assert result.object_count == 3


@pytest.mark.asyncio
async def test_export_glb_failure() -> None:
    bridge = make_bridge("RuntimeError: export failed", success=False)
    result = await export_glb(bridge, "/tmp/scene.glb")
    assert result.success is False
    assert result.error != ""


@pytest.mark.asyncio
async def test_export_glb_passes_path_to_script() -> None:
    bridge = make_bridge(json_output({"path": "/out/model.glb", "object_count": 1}))
    await export_glb(bridge, "/out/model.glb")
    script_sent = bridge.send_script.call_args[0][0]
    assert "/out/model.glb" in script_sent


@pytest.mark.asyncio
async def test_export_glb_selected_only_in_script() -> None:
    bridge = make_bridge(json_output({"path": "/out/sel.glb", "object_count": 1}))
    await export_glb(bridge, "/out/sel.glb", selected_only=True)
    script_sent = bridge.send_script.call_args[0][0]
    assert "True" in script_sent   # selected_only=True in the script


@pytest.mark.asyncio
async def test_export_glb_draco_in_script() -> None:
    bridge = make_bridge(json_output({"path": "/out/d.glb", "object_count": 1}))
    await export_glb(bridge, "/out/d.glb", use_draco=True)
    script_sent = bridge.send_script.call_args[0][0]
    assert "draco" in script_sent.lower()


@pytest.mark.asyncio
async def test_export_glb_accepts_path_object() -> None:
    bridge = make_bridge(json_output({"path": "/out/p.glb", "object_count": 0}))
    result = await export_glb(bridge, Path("/out/p.glb"))
    assert result.success is True


# ── fix_ngons ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fix_ngons_success() -> None:
    data = {"objects_checked": 5, "objects_fixed": 2, "faces_triangulated": 14}
    bridge = make_bridge(json_output(data))
    result = await fix_ngons(bridge)
    assert isinstance(result, NgonResult)
    assert result.success is True
    assert result.objects_checked == 5
    assert result.objects_fixed == 2
    assert result.faces_triangulated == 14


@pytest.mark.asyncio
async def test_fix_ngons_none_found() -> None:
    data = {"objects_checked": 3, "objects_fixed": 0, "faces_triangulated": 0}
    bridge = make_bridge(json_output(data))
    result = await fix_ngons(bridge)
    assert result.success is True
    assert result.objects_fixed == 0


@pytest.mark.asyncio
async def test_fix_ngons_failure() -> None:
    bridge = make_bridge("RuntimeError: bmesh error", success=False)
    result = await fix_ngons(bridge)
    assert result.success is False


@pytest.mark.asyncio
async def test_fix_ngons_uses_bmesh_script() -> None:
    bridge = make_bridge(json_output({"objects_checked": 0, "objects_fixed": 0, "faces_triangulated": 0}))
    await fix_ngons(bridge)
    script = bridge.send_script.call_args[0][0]
    assert "bmesh" in script
    assert "triangulate" in script


# ── generate_lod ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_lod_success() -> None:
    data = {
        "source_object": "PlantModel",
        "original_poly_count": 48000,
        "lods": [
            {"name": "PlantModel_LOD1", "ratio": 0.5, "poly_count": 24000, "reduction": 0.5},
            {"name": "PlantModel_LOD2", "ratio": 0.25, "poly_count": 12000, "reduction": 0.75},
            {"name": "PlantModel_LOD3", "ratio": 0.1, "poly_count": 4800, "reduction": 0.9},
        ],
    }
    bridge = make_bridge(json_output(data))
    result = await generate_lod(bridge, "PlantModel")
    assert isinstance(result, LodResult)
    assert result.success is True
    assert result.source_object == "PlantModel"
    assert len(result.lods_created) == 3
    assert result.lods_created[0]["name"] == "PlantModel_LOD1"


@pytest.mark.asyncio
async def test_generate_lod_custom_ratios() -> None:
    data = {
        "source_object": "Tree",
        "original_poly_count": 10000,
        "lods": [
            {"name": "Tree_LOD1", "ratio": 0.7, "poly_count": 7000, "reduction": 0.3},
        ],
    }
    bridge = make_bridge(json_output(data))
    result = await generate_lod(bridge, "Tree", ratios=[0.7])
    assert result.success is True
    script = bridge.send_script.call_args[0][0]
    assert "0.7" in script


@pytest.mark.asyncio
async def test_generate_lod_default_ratios_in_script() -> None:
    bridge = make_bridge(json_output({"source_object": "X", "original_poly_count": 100, "lods": []}))
    await generate_lod(bridge, "X")
    script = bridge.send_script.call_args[0][0]
    assert "0.5" in script
    assert "0.25" in script
    assert "0.1" in script


@pytest.mark.asyncio
async def test_generate_lod_invalid_ratio_returns_error() -> None:
    bridge = make_bridge("")
    result = await generate_lod(bridge, "Obj", ratios=[0.5, 1.5])
    assert result.success is False
    assert "Invalid ratio" in result.error
    bridge.send_script.assert_not_called()


@pytest.mark.asyncio
async def test_generate_lod_empty_ratios_returns_error() -> None:
    bridge = make_bridge("")
    result = await generate_lod(bridge, "Obj", ratios=[])
    assert result.success is False
    bridge.send_script.assert_not_called() 


@pytest.mark.asyncio
async def test_generate_lod_object_not_found() -> None:
    data = {"error": "Object 'Missing' not found"}
    bridge = make_bridge(json_output(data))
    result = await generate_lod(bridge, "Missing")
    assert result.success is False
    assert "not found" in result.error


@pytest.mark.asyncio
async def test_generate_lod_bridge_failure() -> None:
    bridge = make_bridge("RuntimeError: decimate failed", success=False)
    result = await generate_lod(bridge, "PlantModel")
    assert result.success is False