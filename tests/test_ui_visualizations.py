"""
Unit tests for extended UI components, visualization builders, and prompt presets.
"""

import importlib
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
from core.profiler import StepTelemetry

app_module = importlib.import_module("ui.app")
PROMPT_PRESETS = app_module.PROMPT_PRESETS
update_prompt_preset = app_module.update_prompt_preset
from ui.components import (
    build_confidence_margin_chart,
    build_latency_distribution_chart,
    build_temperature_sensitivity_chart,
    build_telemetry_table,
)


def _create_mock_records():
    records = []
    for i in range(10):
        records.append(
            StepTelemetry(
                step=i,
                token_id=100 + i,
                token_str=f"tok_{i}",
                entropy=0.5 + 0.1 * i,
                norm_entropy=(0.5 + 0.1 * i) / 4.0,
                margin_p1_p2=0.4 - 0.03 * i,
                top1_confidence=0.8 - 0.05 * i,
                top_candidates=[
                    {"rank": 1, "token": f"tok_{i}", "prob": 0.6, "logit": 12.0},
                    {"rank": 2, "token": f"alt_{i}", "prob": 0.25, "logit": 10.5},
                    {"rank": 3, "token": f"cand3", "prob": 0.1, "logit": 8.0},
                    {"rank": 4, "token": f"cand4", "prob": 0.03, "logit": 5.0},
                    {"rank": 5, "token": f"cand5", "prob": 0.02, "logit": 4.0},
                ],
                kv_cache_bytes_analytical=1024 * 1024 * (i + 1),
                kv_cache_mb_analytical=(i + 1),
                cuda_allocated_bytes=2048 * 1024 * (i + 1),
                cuda_allocated_mb=2 * (i + 1),
                cuda_reserved_bytes=4096 * 1024 * (i + 1),
                cuda_reserved_mb=4 * (i + 1),
                cuda_fragmentation_ratio=0.15,
                step_latency_ms=25.0 + 2.0 * i,
                cumulative_ms=(25.0 + 2.0 * i) * (i + 1),
                is_prefill=(i == 0),
            )
        )
    return records


def test_confidence_margin_chart():
    records = _create_mock_records()
    df = pd.DataFrame([r.to_dict() for r in records])
    fig = build_confidence_margin_chart(df)
    assert fig is not None
    assert len(fig.data) >= 2  # Top-1 prob and Margin traces


def test_latency_distribution_chart():
    records = _create_mock_records()
    df = pd.DataFrame([r.to_dict() for r in records])
    fig = build_latency_distribution_chart(df)
    assert fig is not None
    assert len(fig.data) >= 1  # Histogram trace


def test_temperature_sensitivity_chart():
    records = _create_mock_records()
    fig = build_temperature_sensitivity_chart(records[3], current_temp=0.8)
    assert fig is not None
    assert len(fig.data) >= 4  # 3 candidate traces + 1 entropy trace


def test_telemetry_table():
    records = _create_mock_records()
    table = build_telemetry_table(records)
    assert table is not None


def test_prompt_presets_and_callback():
    assert "math" in PROMPT_PRESETS
    assert "triton" in PROMPT_PRESETS
    assert "entropy" in PROMPT_PRESETS

    mock_ctx = MagicMock()
    mock_ctx.triggered_id = "preset-btn-math"
    with patch.object(app_module, "ctx", mock_ctx):
        res = update_prompt_preset(1, None, None)
        assert res == PROMPT_PRESETS["math"]
        assert "FLOPs/byte" in res

    mock_ctx.triggered_id = "preset-btn-triton"
    with patch.object(app_module, "ctx", mock_ctx):
        res = update_prompt_preset(None, 1, None)
        assert res == PROMPT_PRESETS["triton"]
        assert "Triton" in res


def test_dynamic_token_canvas_modes():
    from ui.components import build_token_canvas_elements, get_canvas_text_summary
    records = _create_mock_records()

    # 1. Flow mode
    flow_elems = build_token_canvas_elements(records, view_mode="flow")
    assert len(flow_elems) == len(records)
    assert "token-flow-span" in flow_elems[0].className

    # 2. Pills mode
    pill_elems = build_token_canvas_elements(records, view_mode="pills")
    assert len(pill_elems) == len(records)
    assert "token-pill" in pill_elems[0].className

    # 3. High entropy filter
    filtered_elems = build_token_canvas_elements(records, filter_mode="high_entropy")
    assert len(filtered_elems) == len(records)

    # 4. Canvas text summary
    summary = get_canvas_text_summary(records)
    assert summary["token_count"] == len(records)
    assert summary["word_count"] > 0
    assert len(summary["full_text"]) > 0


def test_dashboard_telemetry_lifecycle():
    update_dashboard_telemetry = app_module.update_dashboard_telemetry
    worker_state = app_module.worker_state
    telemetry_buffer = app_module.telemetry_buffer

    telemetry_buffer.clear()
    records = _create_mock_records()

    # 1. Test while running
    worker_state["is_running"] = True
    worker_state["last_error"] = None
    worker_state["status_message"] = ("running", "Active Profiling...")
    for r in records[:3]:
        telemetry_buffer.add_step(r)

    out = update_dashboard_telemetry(1, 0)
    assert len(out) == 20
    status_text, status_color, alert = out[17], out[18], out[19]
    assert "Streaming Token" in status_text
    assert status_color == "primary"
    assert alert.color == "info"
    assert "Active Profiling" in str(alert.children)

    # 2. Test when finished
    worker_state["is_running"] = False
    out = update_dashboard_telemetry(2, 0)
    status_text, status_color, alert = out[17], out[18], out[19]
    assert "Finished" in status_text
    assert status_color == "success"
    assert alert.color == "success"
    assert "Profiling Complete" in str(alert.children)

    # 3. Clean up
    telemetry_buffer.clear()
    worker_state["status_message"] = None


def test_token_selection_preserves_canvas_dom():
    """Verify that clicking a token pill returns dash.no_update for canvas container, preventing scroll reset."""
    import dash
    update_dashboard_telemetry = app_module.update_dashboard_telemetry
    worker_state = app_module.worker_state
    telemetry_buffer = app_module.telemetry_buffer

    telemetry_buffer.clear()
    records = _create_mock_records()
    for r in records:
        telemetry_buffer.add_step(r)

    mock_ctx = MagicMock()
    mock_ctx.triggered_id = "selected-step-store"
    with patch.object(app_module, "ctx", mock_ctx):
        out = update_dashboard_telemetry(0, 3)
        assert len(out) == 20
        canvas_children = out[6]
        canvas_class = out[7]
        candidate_fig = out[10]
        temp_fig = out[11]

        # Canvas DOM is preserved (no_update), preventing scroll jumps and selection reset!
        assert canvas_children is dash.no_update
        assert canvas_class is dash.no_update
        # Inspection charts are actively updated
        assert candidate_fig is not dash.no_update
        assert temp_fig is not dash.no_update


def test_ram_telemetry_logger():
    """Verify RamTelemetryLogger stores logs in memory without writing to disk."""
    ram_logger = app_module.ram_logger
    ram_logger.log("INFO", "Test telemetry event in RAM")
    entries = ram_logger.get_entries()
    assert len(entries) >= 1
    assert entries[-1]["message"] == "Test telemetry event in RAM"
    assert entries[-1]["level"] == "INFO"


def test_dynamic_softmax_logits_variation():
    """Verify synthetic benchmark produces dynamic logits that shift with confidence."""
    from core.profiler import TransformerInferenceProfiler
    prof = TransformerInferenceProfiler()
    steps = list(prof.profile_synthetic_benchmark(total_steps=10))
    assert len(steps) == 10
    
    # Check that candidate logits are not all identical constants
    all_step_logits = [[c["logit"] for c in s.top_candidates] for s in steps]
    # Verify that different steps have different top-1 logits and logit differences
    top1_logits = [logits[0] for logits in all_step_logits]
    assert len(set(top1_logits)) > 1, "Logits must vary across steps for dynamic temperature sensitivity!"


def test_max_tokens_handling_and_sentence_completion():
    """Verify arbitrary token limits (e.g. 4000), unconstrained EOS (0), and graceful sentence completion."""
    from core.profiler import TransformerInferenceProfiler
    prof = TransformerInferenceProfiler()

    # 1. Verify complete_sentences allows finishing the sentence cleanly with a period
    steps_with_grace = list(prof.profile_synthetic_benchmark(total_steps=10, complete_sentences=True))
    assert len(steps_with_grace) >= 10
    # The last token emitted must be a period '.'
    assert steps_with_grace[-1].token_str.strip() == "."

    # 2. Verify handle_controls max_tokens parsing
    handle_controls = app_module.handle_controls
    worker_state = app_module.worker_state

    # Stop any background worker
    worker_state["is_running"] = False

    mock_ctx = MagicMock()
    mock_ctx.triggered_id = "run-button"
    with patch.object(app_module, "ctx", mock_ctx):
        with patch.object(app_module.threading, "Thread") as mock_thread:
            # Case A: User specifies 4000 tokens
            handle_controls(1, 0, 0, 0, "synthetic", "Llama-3-8B", "test prompt", 4000, 0.7, "standard", False)
            assert mock_thread.called
            args = mock_thread.call_args[1]["args"]
            assert args[3] == 4000, f"Expected 4000 max tokens, got {args[3]}"

            # Case B: User enters string "4000"
            handle_controls(1, 0, 0, 0, "synthetic", "Llama-3-8B", "test prompt", "4000", 0.7, "standard", False)
            args = mock_thread.call_args[1]["args"]
            assert args[3] == 4000, f"Expected 4000 max tokens, got {args[3]}"

            # Case C: User enters 0 for unconstrained EOS
            handle_controls(1, 0, 0, 0, "synthetic", "Llama-3-8B", "test prompt", 0, 0.7, "standard", False)
            args = mock_thread.call_args[1]["args"]
            assert args[3] == 0, f"Expected 0 (unconstrained EOS), got {args[3]}"

            # Case D: Empty string or None defaults safely to 2048 (NOT 256)
            handle_controls(1, 0, 0, 0, "synthetic", "Llama-3-8B", "test prompt", "", 0.7, "standard", False)
            args = mock_thread.call_args[1]["args"]
            assert args[3] == 2048, f"Expected default 2048, got {args[3]}"

