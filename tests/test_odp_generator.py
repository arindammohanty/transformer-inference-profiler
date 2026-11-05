"""
Unit tests for OpenDocument Presentation synthesis and MCP tools.
"""

import os
import zipfile
import pytest
from mcp_impress.odp_generator import ODPPresentationBuilder
from mcp_impress.server import (
    handle_mcp_request,
    tool_add_slide,
    tool_apply_animation,
    tool_create_presentation,
    tool_export_presentation,
    tool_set_slide_transition,
)


def test_odp_builder_lifecycle(tmp_path):
    builder = ODPPresentationBuilder(title="Test Deck", theme="cyber_dark")
    s1 = builder.add_slide(
        title="Slide 1",
        subtitle="Subtitle 1",
        bullets=["Point A: Explanation A", "Point B: Explanation B"],
        callout_boxes=[{"title": "KPI", "value": "99.9%", "desc": "Uptime"}],
        transition_type="push",
        transition_speed="fast",
    )
    assert len(builder.slides) == 1

    builder.apply_animation(0, "content_frame_0", "appear", "click")
    assert len(s1.animations) == 1

    builder.set_slide_transition(0, "fade", "medium")
    assert s1.transition_type == "fade"
    assert s1.transition_speed == "medium"

    output_path = str(tmp_path / "test.odp")
    builder.save_odp(output_path)
    assert os.path.exists(output_path)

    # Verify zip content
    with zipfile.ZipFile(output_path, "r") as z:
        names = z.namelist()
        assert "mimetype" in names
        assert "content.xml" in names
        assert "styles.xml" in names
        assert "META-INF/manifest.xml" in names

        # Verify mimetype is uncompressed
        info = z.getinfo("mimetype")
        assert info.compress_type == zipfile.ZIP_STORED


def test_mcp_tools():
    res_create = tool_create_presentation("MCP Test Deck")
    assert res_create["status"] == "success"

    res_slide = tool_add_slide(
        title="Slide MCP",
        bullets=["Alpha", "Beta"],
        transition_type="wipe",
    )
    assert res_slide["status"] == "success"
    assert res_slide["slide_index"] == 0

    res_trans = tool_set_slide_transition(0, "fade", "fast")
    assert res_trans["status"] == "success"

    res_anim = tool_apply_animation(0, "shape_1", "appear", "click")
    assert res_anim["status"] == "success"


def test_mcp_jsonrpc_protocol():
    # Test initialize
    init_req = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    init_resp = handle_mcp_request(init_req)
    assert init_resp["result"]["serverInfo"]["name"] == "mcp-libreoffice-impress"

    # Test tools/list
    list_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    list_resp = handle_mcp_request(list_req)
    tools = list_resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "create_presentation" in tool_names
    assert "add_slide" in tool_names
    assert "apply_animation" in tool_names
    assert "set_slide_transition" in tool_names
    assert "export_presentation" in tool_names

    # Test tools/call create_presentation
    call_req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "create_presentation",
            "arguments": {"title": "JSONRPC Deck"},
        },
    }
    call_resp = handle_mcp_request(call_req)
    assert "result" in call_resp
    assert "content" in call_resp["result"]
