"""
Core LLM Inference Profiler & Telemetry Package
"""
from core.metrics import (
    shannon_entropy,
    normalized_entropy,
    compute_top_k_candidates,
    calculate_analytical_kv_cache_bytes,
    calculate_kv_cache_mb,
    compare_attention_architectures,
    compute_latency_telemetry,
)
from core.profiler import (
    StepTelemetry,
    CudaTelemetryEngine,
    TransformerInferenceProfiler,
    ARCHITECTURE_PRESETS,
)
from core.ollama_client import OllamaClient

__all__ = [
    "shannon_entropy",
    "normalized_entropy",
    "compute_top_k_candidates",
    "calculate_analytical_kv_cache_bytes",
    "calculate_kv_cache_mb",
    "compare_attention_architectures",
    "compute_latency_telemetry",
    "StepTelemetry",
    "CudaTelemetryEngine",
    "TransformerInferenceProfiler",
    "ARCHITECTURE_PRESETS",
    "OllamaClient",
]
