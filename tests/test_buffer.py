"""
Unit tests for in-memory telemetry buffer.
"""

import pytest
from core.profiler import StepTelemetry
from storage.buffer import TelemetryBuffer


def test_telemetry_buffer_operations():
    buf = TelemetryBuffer()
    assert buf.count() == 0

    step0 = StepTelemetry(
        step=0,
        token_id=42,
        token_str="Hello",
        entropy=1.25,
        norm_entropy=0.12,
        margin_p1_p2=0.45,
        top1_confidence=0.70,
        top_candidates=[{"token": "Hello", "prob": 0.7, "logit": 12.0, "token_id": 42}],
        kv_cache_bytes_analytical=1024 * 1024,
        kv_cache_mb_analytical=1.0,
        cuda_allocated_bytes=4000 * 1024 * 1024,
        cuda_allocated_mb=4000.0,
        cuda_reserved_bytes=4500 * 1024 * 1024,
        cuda_reserved_mb=4500.0,
        cuda_fragmentation_ratio=0.11,
        step_latency_ms=120.0,
        cumulative_ms=120.0,
        is_prefill=True,
    )

    buf.add_step(step0)
    assert buf.count() == 1

    fetched = buf.get_step(0)
    assert fetched is not None
    assert fetched.token_str == "Hello"
    assert fetched.is_prefill is True

    # Add second step
    step1 = StepTelemetry(
        step=1,
        token_id=88,
        token_str=" World",
        entropy=0.85,
        norm_entropy=0.08,
        margin_p1_p2=0.62,
        top1_confidence=0.82,
        top_candidates=[{"token": " World", "prob": 0.82, "logit": 14.0, "token_id": 88}],
        kv_cache_bytes_analytical=2 * 1024 * 1024,
        kv_cache_mb_analytical=2.0,
        cuda_allocated_bytes=4001 * 1024 * 1024,
        cuda_allocated_mb=4001.0,
        cuda_reserved_bytes=4500 * 1024 * 1024,
        cuda_reserved_mb=4500.0,
        cuda_fragmentation_ratio=0.11,
        step_latency_ms=25.0,
        cumulative_ms=145.0,
        is_prefill=False,
    )
    buf.add_step(step1)
    assert buf.count() == 2

    # Verify DataFrame
    df = buf.get_dataframe()
    assert len(df) == 2
    assert "token_str" in df.columns
    assert "kv_cache_mb_analytical" in df.columns

    # Verify Summary Stats
    stats = buf.get_summary_stats()
    assert stats["total_tokens"] == 2
    assert stats["ttft_ms"] == 120.0
    assert stats["mean_itl_ms"] == 25.0
    assert stats["peak_kv_cache_mb"] == 2.0

    # Verify Clear
    buf.clear()
    assert buf.count() == 0
    assert len(buf.get_dataframe()) == 0
