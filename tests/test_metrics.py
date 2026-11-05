"""
Unit tests for mathematical and telemetry metrics.
"""

import math
import pytest
import torch
import numpy as np

from core.metrics import (
    calculate_analytical_kv_cache_bytes,
    calculate_kv_cache_mb,
    compare_attention_architectures,
    compute_latency_telemetry,
    compute_top_k_candidates,
    normalized_entropy,
    shannon_entropy,
)


def test_shannon_entropy_deterministic():
    """Deterministic distribution should have zero entropy."""
    probs = [1.0, 0.0, 0.0, 0.0]
    ent = shannon_entropy(probs)
    assert math.isclose(ent, 0.0, abs_tol=1e-5)

    probs_t = torch.tensor([1.0, 0.0, 0.0])
    ent_t = shannon_entropy(probs_t)
    assert math.isclose(ent_t, 0.0, abs_tol=1e-5)


def test_shannon_entropy_uniform():
    """Uniform distribution over N elements should yield exactly log2(N) bits."""
    n = 8
    probs = [1.0 / n] * n
    expected_bits = math.log2(n)  # 3.0
    ent = shannon_entropy(probs)
    assert math.isclose(ent, expected_bits, rel_tol=1e-4)


def test_normalized_entropy_bounds():
    """Normalized entropy must remain bounded in [0.0, 1.0]."""
    assert normalized_entropy(0.0, 32000) == 0.0
    max_ent = math.log2(32000)
    assert math.isclose(normalized_entropy(max_ent, 32000), 1.0, rel_tol=1e-4)
    assert 0.0 <= normalized_entropy(2.5, 32000) <= 1.0


def test_top_k_candidates():
    """Verify top-k candidate extraction and margins."""
    logits = torch.tensor([2.0, 5.0, 1.0, 0.5, 3.0, 0.1])
    res = compute_top_k_candidates(logits, k=3, is_logits=True)
    assert len(res["top_indices"]) == 3
    assert res["top_indices"][0] == 1  # logit 5.0
    assert res["top_indices"][1] == 4  # logit 3.0
    assert res["top_indices"][2] == 0  # logit 2.0
    assert res["margin_p1_p2"] > 0
    assert math.isclose(res["margin_z1_z2"], 5.0 - 3.0, rel_tol=1e-4)


def test_analytical_kv_cache_calculation():
    """
    Test KV-cache analytical formula:
    Batch=1, SeqLen=1024, Layers=32, KV_Heads=8, HeadDim=128, FP16 (2 bytes)
    Bytes = 2 * 1 * 1024 * 32 * 8 * 128 * 2 = 134,217,728 bytes = 128 MB (MiB)
    """
    bytes_val = calculate_analytical_kv_cache_bytes(
        batch_size=1,
        seq_len=1024,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        dtype_bytes=2,
    )
    assert bytes_val == 134217728
    mb_val = calculate_kv_cache_mb(
        batch_size=1,
        seq_len=1024,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        dtype_bytes=2,
    )
    assert math.isclose(mb_val, 128.0, rel_tol=1e-4)


def test_compare_attention_architectures():
    """Verify GQA and MQA footprint reduction factors."""
    res = compare_attention_architectures(
        seq_len=2048,
        num_layers=32,
        num_attention_heads=32,
        head_dim=128,
        batch_size=1,
        dtype_bytes=2,
    )
    # MHA: 32 heads
    # GQA-4: 8 heads -> should be 1/4 of MHA
    # GQA-8: 4 heads -> should be 1/8 of MHA
    assert math.isclose(res["GQA_4"] * 4, res["MHA"], rel_tol=0.01)
    assert math.isclose(res["GQA_8"] * 8, res["MHA"], rel_tol=0.01)
    assert res["MQA"] < res["GQA_8"]


def test_latency_telemetry():
    """Test TTFT and ITL statistics."""
    latencies = [150.0, 20.0, 25.0, 22.0, 19.0, 24.0]
    stats = compute_latency_telemetry(latencies)
    assert stats["ttft_ms"] == 150.0
    assert 20.0 <= stats["mean_itl_ms"] <= 25.0
    assert stats["tokens_per_sec"] > 0.0
