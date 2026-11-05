"""
Ollama Server Client for Dynamic Model Discovery, Architecture Inspection,
and Real-Time Streaming Telemetry Interception.
"""

from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional
import requests

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if local Ollama daemon is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """Fetch list of downloaded Ollama models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("models", [])
        except Exception as e:
            logger.warning(f"Failed to query Ollama models: {e}")
        return []

    def get_running_models(self) -> List[Dict[str, Any]]:
        """Fetch list of currently loaded/running models in Ollama VRAM/RAM."""
        try:
            resp = requests.get(f"{self.base_url}/api/ps", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("models", [])
        except Exception as e:
            logger.warning(f"Failed to query running Ollama models: {e}")
        return []

    def unload_model(self, model_name: str) -> bool:
        """Immediately evicts a model from RAM/VRAM by setting keep_alive to 0."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=10.0,
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"Failed to unload model {model_name}: {e}")
            return False

    def unload_all_models(self) -> List[str]:
        """Evicts ALL currently running models from Ollama memory to return to clean idle state."""
        unloaded = []
        running = self.get_running_models()
        for item in running:
            m_name = item.get("name") or item.get("model")
            if m_name:
                if self.unload_model(m_name):
                    unloaded.append(m_name)
                    logger.info(f"Evicted model '{m_name}' from memory.")
        return unloaded

    def estimate_model_ram_gb(self, model_name: str) -> float:
        """
        Calculates estimated total RAM/VRAM footprint in GB required to run the model.
        Accounts for model weight size plus context working memory and allocator overhead.
        """
        try:
            models = self.list_models()
            for m in models:
                if m.get("name") == model_name or m.get("model") == model_name:
                    size_bytes = m.get("size", 0)
                    if size_bytes > 0:
                        # 1.25x disk size for GGUF runtime buffers and context
                        return round((size_bytes * 1.25) / (1024**3), 1)
        except Exception:
            pass

        # Heuristic fallback based on name
        name_lower = model_name.lower()
        if "30b" in name_lower or "32b" in name_lower:
            return 25.0
        elif "14b" in name_lower:
            return 12.5
        elif "7b" in name_lower or "8b" in name_lower:
            return 7.5
        elif "12b" in name_lower:
            return 10.5
        elif "3b" in name_lower or "e4b" in name_lower:
            return 4.5
        elif "e2b" in name_lower or "1b" in name_lower:
            return 2.5
        return 6.0

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Query Ollama /api/show to inspect model architecture parameters,
        GGUF parameters, context length, and attention head counts.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                details = data.get("details", {})
                model_info = data.get("model_info", {})

                # Try to parse key attention parameters from model_info
                # Common GGUF keys:
                # {family}.block_count -> layers
                # {family}.attention.head_count -> Q heads
                # {family}.attention.head_count_kv -> KV heads
                # {family}.embedding_length -> hidden_size
                # {family}.context_length -> context_len
                family = details.get("family", "llama")
                
                num_layers = (
                    model_info.get(f"{family}.block_count")
                    or model_info.get("llama.block_count")
                    or model_info.get("qwen2.block_count")
                    or 32
                )
                num_q_heads = (
                    model_info.get(f"{family}.attention.head_count")
                    or model_info.get("llama.attention.head_count")
                    or model_info.get("qwen2.attention.head_count")
                    or 32
                )
                num_kv_heads = (
                    model_info.get(f"{family}.attention.head_count_kv")
                    or model_info.get("llama.attention.head_count_kv")
                    or model_info.get("qwen2.attention.head_count_kv")
                    or 8
                )
                hidden_size = (
                    model_info.get(f"{family}.embedding_length")
                    or details.get("embedding_length")
                    or 4096
                )
                context_length = (
                    model_info.get(f"{family}.context_length")
                    or details.get("context_length")
                    or 8192
                )
                head_dim = (hidden_size // num_q_heads) if num_q_heads > 0 else 128

                return {
                    "model": model_name,
                    "family": family,
                    "parameter_size": details.get("parameter_size", "Unknown"),
                    "quantization_level": details.get("quantization_level", "Unknown"),
                    "context_length": context_length,
                    "num_layers": int(num_layers),
                    "num_q_heads": int(num_q_heads),
                    "num_kv_heads": int(num_kv_heads),
                    "hidden_size": int(hidden_size),
                    "head_dim": int(head_dim),
                    "raw_details": details,
                }
        except Exception as e:
            logger.warning(f"Failed to fetch model info for {model_name}: {e}")

        # Fallback heuristic for known models
        return self._heuristic_model_info(model_name)

    def _heuristic_model_info(self, model_name: str) -> Dict[str, Any]:
        """Provides accurate architectural constants for standard LLMs if API inspection fails."""
        name_lower = model_name.lower()
        if "30b" in name_lower or "32b" in name_lower:
            return {
                "model": model_name,
                "family": "qwen2",
                "parameter_size": "32B",
                "quantization_level": "Q4_K_M",
                "context_length": 32768,
                "num_layers": 64,
                "num_q_heads": 40,
                "num_kv_heads": 8,
                "hidden_size": 5120,
                "head_dim": 128,
            }
        elif "14b" in name_lower:
            return {
                "model": model_name,
                "family": "qwen2",
                "parameter_size": "14.8B",
                "quantization_level": "Q4_K_M",
                "context_length": 32768,
                "num_layers": 48,
                "num_q_heads": 40,
                "num_kv_heads": 8,
                "hidden_size": 5120,
                "head_dim": 128,
            }
        elif "7b" in name_lower or "8b" in name_lower:
            return {
                "model": model_name,
                "family": "llama3",
                "parameter_size": "8.0B",
                "quantization_level": "Q4_0",
                "context_length": 8192,
                "num_layers": 32,
                "num_q_heads": 32,
                "num_kv_heads": 8,
                "hidden_size": 4096,
                "head_dim": 128,
            }
        elif "12b" in name_lower:
            return {
                "model": model_name,
                "family": "gemma",
                "parameter_size": "11.9B",
                "quantization_level": "Q4_K_M",
                "context_length": 8192,
                "num_layers": 40,
                "num_q_heads": 32,
                "num_kv_heads": 16,
                "hidden_size": 3840,
                "head_dim": 120,
            }
        return {
            "model": model_name,
            "family": "transformer",
            "parameter_size": "7B",
            "quantization_level": "Q4",
            "context_length": 8192,
            "num_layers": 32,
            "num_q_heads": 32,
            "num_kv_heads": 8,
            "hidden_size": 4096,
            "head_dim": 128,
        }

    def stream_generate(
        self,
        model_name: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 128,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streams inference tokens from Ollama server while measuring exact step-level latency.
        Yields per-step telemetry payloads.
        """
        options: Dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens if max_tokens > 0 else -1,
        }
        if max_tokens > 0:
            options["num_ctx"] = max(4096, min(131072, max_tokens + 2048))
        else:
            options["num_ctx"] = 32768

        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": options,
        }
        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.perf_counter()
        prev_step_time = start_time
        step_idx = 0

        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=600.0,
            ) as resp:
                if resp.status_code != 200:
                    yield {
                        "error": f"Ollama HTTP error {resp.status_code}: {resp.text}",
                        "done": True,
                    }
                    return

                for line in resp.iter_lines():
                    if not line:
                        continue
                    now = time.perf_counter()
                    step_latency_ms = (now - prev_step_time) * 1000.0
                    prev_step_time = now

                    data = json.loads(line.decode("utf-8"))
                    token_str = data.get("response", "")
                    done = data.get("done", False)

                    payload_out: Dict[str, Any] = {
                        "step": step_idx,
                        "token_str": token_str,
                        "step_latency_ms": step_latency_ms,
                        "cumulative_ms": (now - start_time) * 1000.0,
                        "done": done,
                    }

                    if done:
                        # Append Ollama's native high-resolution timing
                        # Durations are provided in nanoseconds
                        payload_out["total_duration_ms"] = data.get("total_duration", 0) / 1e6
                        payload_out["load_duration_ms"] = data.get("load_duration", 0) / 1e6
                        payload_out["prompt_eval_count"] = data.get("prompt_eval_count", 0)
                        payload_out["prompt_eval_duration_ms"] = (
                            data.get("prompt_eval_duration", 0) / 1e6
                        )
                        payload_out["eval_count"] = data.get("eval_count", step_idx)
                        payload_out["eval_duration_ms"] = data.get("eval_duration", 0) / 1e6

                    yield payload_out
                    step_idx += 1
                    if done:
                        break

        except Exception as e:
            logger.error(f"Error in stream_generate: {e}")
            yield {
                "error": str(e),
                "done": True,
            }
