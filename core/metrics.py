"""
Mathematical and telemetry metrics for autoregressive Transformer inference profiling.
Implements token-level Shannon entropy, logit margins, KV-cache scaling laws,
and inter-token latency (ITL) statistics.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import numpy as np


def shannon_entropy(
    probs: Union[torch.Tensor, np.ndarray, Sequence[float]],
    base: float = 2.0,
    eps: float = 1e-12,
) -> float:
    """
    Computes Shannon entropy: H(P_t) = - \sum_{i=1}^{|V|} p_i * log_b(p_i).
    
    Args:
        probs: Probability distribution vector over vocabulary.
        base: Logarithm base (default 2.0 for bits/shannons).
        eps: Small epsilon to prevent log(0) numerical instability.
        
    Returns:
        Entropy value in shannons (bits).
    """
    if isinstance(probs, torch.Tensor):
        p = probs.detach().float()
        # Clamp to avoid numerical issues
        p = torch.clamp(p, min=eps)
        # Normalize in case probabilities don't sum strictly to 1.0
        p = p / p.sum()
        log_p = torch.log(p) / math.log(base)
        ent = -(p * log_p).sum().item()
        return float(max(0.0, ent))
    
    arr = np.asarray(probs, dtype=np.float64)
    arr = np.clip(arr, eps, 1.0)
    arr = arr / np.sum(arr)
    ent = -np.sum(arr * (np.log(arr) / math.log(base)))
    return float(max(0.0, ent))


def normalized_entropy(
    entropy: float,
    vocab_size: int,
    base: float = 2.0,
) -> float:
    """
    Normalizes Shannon entropy to [0.0, 1.0] by dividing by the theoretical maximum H_max = log_b(|V|).
    
    A value close to 0 indicates deterministic certainty (single dominant candidate).
    A value close to 1 indicates uniform uncertainty (branching or hallucination risk).
    """
    if vocab_size <= 1:
        return 0.0
    max_entropy = math.log(vocab_size) / math.log(base)
    if max_entropy <= 0.0:
        return 0.0
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def compute_top_k_candidates(
    logits_or_probs: Union[torch.Tensor, np.ndarray, Sequence[float]],
    k: int = 5,
    is_logits: bool = True,
    tokenizer_decode_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Extracts top-k candidates, probabilities, logit values, and margins.
    
    Returns:
        Dictionary containing:
            - top_indices: list of token IDs
            - top_probs: list of probabilities
            - top_tokens: decoded strings if tokenizer provided
            - margin_p1_p2: probability margin p(1) - p(2)
            - top1_confidence: p(1)
            - top_logits: list of raw logit values
    """
    if isinstance(logits_or_probs, (list, tuple, np.ndarray)):
        tensor = torch.tensor(logits_or_probs, dtype=torch.float32)
    else:
        tensor = logits_or_probs.detach().float()

    if tensor.ndim > 1:
        tensor = tensor.squeeze(0)

    if is_logits:
        logits = tensor
        probs = torch.softmax(logits, dim=-1)
    else:
        probs = tensor / tensor.sum()
        logits = torch.log(torch.clamp(probs, min=1e-12))

    k = min(k, probs.size(-1))
    top_probs_t, top_indices_t = torch.topk(probs, k=k)
    top_logits_t = logits[top_indices_t]

    top_indices = top_indices_t.cpu().tolist()
    top_probs = [float(x) for x in top_probs_t.cpu().tolist()]
    top_logits = [float(x) for x in top_logits_t.cpu().tolist()]

    # Probability and logit margins
    p1 = top_probs[0] if len(top_probs) > 0 else 0.0
    p2 = top_probs[1] if len(top_probs) > 1 else 0.0
    margin_p = float(p1 - p2)

    z1 = top_logits[0] if len(top_logits) > 0 else 0.0
    z2 = top_logits[1] if len(top_logits) > 1 else 0.0
    margin_z = float(z1 - z2)

    top_tokens: List[str] = []
    if tokenizer_decode_fn:
        for idx in top_indices:
            try:
                decoded = tokenizer_decode_fn(idx)
                # Escape non-printable or newline
                top_tokens.append(repr(decoded)[1:-1])
            except Exception:
                top_tokens.append(f"token_{idx}")
    else:
        top_tokens = [f"token_{idx}" for idx in top_indices]

    return {
        "top_indices": top_indices,
        "top_probs": top_probs,
        "top_tokens": top_tokens,
        "top_logits": top_logits,
        "top1_confidence": p1,
        "margin_p1_p2": margin_p,
        "margin_z1_z2": margin_z,
    }


def calculate_analytical_kv_cache_bytes(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,  # FP16 / BF16 = 2 bytes, FP8 = 1 byte
) -> int:
    """
    Computes exact theoretical KV-cache allocation footprint in bytes:
    
        KV_bytes = 2 * batch_size * seq_len * num_layers * num_kv_heads * head_dim * dtype_bytes
        
    Factor of 2 accounts for both Keys and Values.
    """
    return int(2 * batch_size * seq_len * num_layers * num_kv_heads * head_dim * dtype_bytes)


def calculate_kv_cache_mb(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> float:
    """Computes exact analytical KV cache in Megabytes (MiB = 1024^2 bytes)."""
    bytes_val = calculate_analytical_kv_cache_bytes(
        batch_size=batch_size,
        seq_len=seq_len,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_bytes=dtype_bytes,
    )
    return float(bytes_val / (1024.0 * 1024.0))


def compare_attention_architectures(
    seq_len: int,
    num_layers: int = 32,
    num_attention_heads: int = 32,
    head_dim: int = 128,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> Dict[str, float]:
    """
    Compares KV cache footprint (in MB) across different attention paradigms:
    - Multi-Head Attention (MHA): num_kv_heads = num_attention_heads
    - Grouped-Query Attention (GQA-4): num_kv_heads = num_attention_heads // 4
    - Grouped-Query Attention (GQA-8): num_kv_heads = num_attention_heads // 8
    - Multi-Query Attention (MQA): num_kv_heads = 1
    - Multi-Head Latent Attention (MLA): DeepSeek-style compressed latent KV (latent dim = 512)
    """
    mha_mb = calculate_kv_cache_mb(
        batch_size, seq_len, num_layers, num_attention_heads, head_dim, dtype_bytes
    )
    gqa4_heads = max(1, num_attention_heads // 4)
    gqa4_mb = calculate_kv_cache_mb(
        batch_size, seq_len, num_layers, gqa4_heads, head_dim, dtype_bytes
    )
    gqa8_heads = max(1, num_attention_heads // 8)
    gqa8_mb = calculate_kv_cache_mb(
        batch_size, seq_len, num_layers, gqa8_heads, head_dim, dtype_bytes
    )
    mqa_mb = calculate_kv_cache_mb(
        batch_size, seq_len, num_layers, 1, head_dim, dtype_bytes
    )
    # MLA compresses KV into a single compressed latent vector d_c (e.g. 512) + decoupled RoPE key (e.g. 64)
    mla_latent_dim = 512 + 64
    mla_bytes = batch_size * seq_len * num_layers * mla_latent_dim * dtype_bytes
    mla_mb = float(mla_bytes / (1024.0 * 1024.0))

    return {
        "MHA": round(mha_mb, 2),
        "GQA_4": round(gqa4_mb, 2),
        "GQA_8": round(gqa8_mb, 2),
        "MQA": round(mqa_mb, 2),
        "MLA_DeepSeek": round(mla_mb, 2),
    }


def compute_latency_telemetry(
    step_latencies_ms: List[float],
) -> Dict[str, float]:
    """
    Computes TTFT (Time-to-First-Token) and ITL (Inter-Token Latency) statistical dynamics.
    
    Args:
        step_latencies_ms: List of step durations in milliseconds.
            step_latencies_ms[0] is typically TTFT (pre-fill phase).
            step_latencies_ms[1:] are autoregressive generation steps (ITL).
    """
    if not step_latencies_ms:
        return {
            "ttft_ms": 0.0,
            "mean_itl_ms": 0.0,
            "median_itl_ms": 0.0,
            "p95_itl_ms": 0.0,
            "p99_itl_ms": 0.0,
            "jitter_itl_ms": 0.0,
            "tokens_per_sec": 0.0,
            "total_gen_time_ms": 0.0,
        }

    ttft_ms = float(step_latencies_ms[0])
    itl_steps = step_latencies_ms[1:] if len(step_latencies_ms) > 1 else [step_latencies_ms[0]]
    arr = np.array(itl_steps, dtype=np.float64)

    mean_itl = float(np.mean(arr))
    median_itl = float(np.median(arr))
    p95_itl = float(np.percentile(arr, 95))
    p99_itl = float(np.percentile(arr, 99))
    jitter_itl = float(np.std(arr))

    total_gen_time_ms = float(sum(step_latencies_ms))
    tokens_per_sec = (len(step_latencies_ms) / (total_gen_time_ms / 1000.0)) if total_gen_time_ms > 0 else 0.0

    return {
        "ttft_ms": round(ttft_ms, 2),
        "mean_itl_ms": round(mean_itl, 2),
        "median_itl_ms": round(median_itl, 2),
        "p95_itl_ms": round(p95_itl, 2),
        "p99_itl_ms": round(p99_itl, 2),
        "jitter_itl_ms": round(jitter_itl, 2),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "total_gen_time_ms": round(total_gen_time_ms, 2),
    }
