"""
Unit tests for core profiler, hybrid CUDA telemetry engine, and synthetic benchmarks.
"""

import pytest
import torch
from core.profiler import (
    ARCHITECTURE_PRESETS,
    CudaTelemetryEngine,
    TransformerInferenceProfiler,
)


def test_cuda_telemetry_engine_simulation():
    """Verify hybrid CUDA allocator simulation."""
    engine = CudaTelemetryEngine(force_simulation=True, base_model_weight_mb=2000.0)
    alloc_0, res_0, frag_0 = engine.get_telemetry(current_kv_cache_bytes=0)
    assert alloc_0 >= 2000 * 1024 * 1024
    assert res_0 >= alloc_0
    assert 0.0 <= frag_0 <= 1.0

    # Increase KV cache bytes significantly to force arena expansion
    large_kv = 500 * 1024 * 1024  # 500 MB
    alloc_1, res_1, frag_1 = engine.get_telemetry(current_kv_cache_bytes=large_kv)
    assert alloc_1 > alloc_0
    assert res_1 >= alloc_1


def test_synthetic_benchmark_generation():
    """Verify synthetic benchmark generator streams valid telemetry."""
    profiler = TransformerInferenceProfiler()
    steps = list(
        profiler.profile_synthetic_benchmark(
            preset_name="Llama-3-8B",
            total_steps=10,
            initial_context_len=128,
            scenario="standard",
        )
    )
    assert len(steps) == 10
    step0 = steps[0]
    assert step0.is_prefill is True
    assert step0.step == 0
    assert step0.entropy > 0.0
    assert step0.kv_cache_bytes_analytical > 0
    assert len(step0.top_candidates) == 5

    # Step 1 should be autoregressive ITL
    step1 = steps[1]
    assert step1.is_prefill is False
    assert step1.step == 1
    assert step1.step_latency_ms > 0.0
    assert step1.kv_cache_bytes_analytical > step0.kv_cache_bytes_analytical


def test_high_entropy_branching_scenario():
    """Verify high entropy scenario produces elevated entropy in target window."""
    profiler = TransformerInferenceProfiler()
    steps = list(
        profiler.profile_synthetic_benchmark(
            preset_name="Qwen-2.5-14B",
            total_steps=25,
            initial_context_len=256,
            scenario="high_entropy_branching",
        )
    )
    assert len(steps) == 25
    # Steps 15-24 should have high entropy
    high_ent_steps = [s.entropy for s in steps[15:25]]
    assert sum(high_ent_steps) / len(high_ent_steps) > 2.0


def test_system_environment_query():
    """Verify system environment introspection."""
    profiler = TransformerInferenceProfiler()
    env = profiler.get_system_environment()
    assert "cuda_available" in env
    assert "device_name" in env
    assert "pytorch_version" in env
    assert "presets" in env
    assert "Llama-3-8B" in env["presets"]


def test_ollama_ram_estimation_and_safeguards():
    """Verify RAM estimation categorizes 30B/14B/7B models correctly to prevent OOM."""
    from core.ollama_client import OllamaClient
    client = OllamaClient()

    # Test fallback heuristics and catalog lookups
    ram_30b = client.estimate_model_ram_gb("qwen3-coder:30b")
    assert ram_30b >= 20.0, f"Expected 30B model >= 20GB, got {ram_30b}"

    ram_14b = client.estimate_model_ram_gb("qwen2.5-coder:14b")
    assert 8.0 <= ram_14b <= 16.0, f"Expected 14B model ~10-14GB, got {ram_14b}"

    ram_7b = client.estimate_model_ram_gb("gemma:7b")
    assert ram_7b < 8.0, f"Expected 7B model < 8GB, got {ram_7b}"


def test_ollama_model_eviction_methods():
    """Verify unload_model and unload_all_models execute without crashing."""
    from core.ollama_client import OllamaClient
    client = OllamaClient()

    # Calling unload on non-existent or dummy model should safely return bool
    res = client.unload_model("nonexistent_model:dummy")
    assert isinstance(res, bool)

    # Calling unload_all_models should return list of unloaded model strings
    unloaded = client.unload_all_models()
    assert isinstance(unloaded, list)


def test_ui_ram_safety_and_emergency_controls():
    """Verify UI safety callbacks and emergency eviction logic."""
    import importlib
    from unittest.mock import MagicMock, patch

    app_mod = importlib.import_module("ui.app")
    check_ram_safety = app_mod.check_ram_safety
    handle_controls = app_mod.handle_controls

    # 1. Verify 30B model triggers danger alert and disables run button
    is_open, children, color, style, disabled = check_ram_safety("ollama", "qwen3-coder:30b", False)
    assert is_open is True
    assert color == "danger"
    assert disabled is True
    assert style == {"display": "block"}

    # 2. Verify override enables run button
    _, _, _, _, disabled_override = check_ram_safety("ollama", "qwen3-coder:30b", True)
    assert disabled_override is False

    # 3. Verify safe 7B model does not disable run button when sufficient RAM is available
    mock_mem = MagicMock(available=int(10.0 * (1024**3)), total=int(16.0 * (1024**3)))
    with patch.object(app_mod.psutil, "virtual_memory", return_value=mock_mem):
        _, _, _, _, disabled_7b = check_ram_safety("ollama", "gemma:7b", False)
        assert disabled_7b is False

    # 4. Verify emergency button eviction handling
    mock_ctx = MagicMock()
    mock_ctx.triggered_id = "emergency-button"
    with patch.object(app_mod, "ctx", mock_ctx):
        alert = handle_controls(None, None, None, 1, "ollama", "gemma:7b", "prompt", 20, 0.7, "standard", False)
        assert alert.color == "success"
        assert "EMERGENCY EVICTION EXECUTED" in str(alert.children)


