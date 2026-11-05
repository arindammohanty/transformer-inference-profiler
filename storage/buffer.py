"""
High-throughput in-memory telemetry buffer.
Thread-safe ingestion of step-by-step profiling records with instant DataFrame queries,
aggregate summaries, and token step inspection.
"""

from __future__ import annotations
import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional
import pandas as pd

from core.profiler import StepTelemetry


class TelemetryBuffer:
    """
    In-memory high-throughput telemetry storage.
    Uses SQLite in-memory tables + cached list for microsecond step lookups
    and seamless pandas DataFrame conversion.
    """

    def __init__(self, max_records: int = 100000):
        self.max_records = max_records
        self.lock = threading.Lock()
        self._records: List[StepTelemetry] = []
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry (
                    step INTEGER PRIMARY KEY,
                    token_id INTEGER,
                    token_str TEXT,
                    entropy REAL,
                    norm_entropy REAL,
                    margin_p1_p2 REAL,
                    top1_confidence REAL,
                    top_candidates_json TEXT,
                    kv_cache_bytes_analytical INTEGER,
                    kv_cache_mb_analytical REAL,
                    cuda_allocated_bytes INTEGER,
                    cuda_allocated_mb REAL,
                    cuda_reserved_bytes INTEGER,
                    cuda_reserved_mb REAL,
                    cuda_fragmentation_ratio REAL,
                    step_latency_ms REAL,
                    cumulative_ms REAL,
                    is_prefill INTEGER
                )
                """
            )

    def clear(self):
        """Reset all stored records."""
        with self.lock:
            self._records.clear()
            with self._conn:
                self._conn.execute("DELETE FROM telemetry")

    def add_step(self, step: StepTelemetry):
        """Append single step telemetry record."""
        with self.lock:
            self._records.append(step)
            candidates_json = json.dumps(step.top_candidates)
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO telemetry VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        step.step,
                        step.token_id,
                        step.token_str,
                        step.entropy,
                        step.norm_entropy,
                        step.margin_p1_p2,
                        step.top1_confidence,
                        candidates_json,
                        step.kv_cache_bytes_analytical,
                        step.kv_cache_mb_analytical,
                        step.cuda_allocated_bytes,
                        step.cuda_allocated_mb,
                        step.cuda_reserved_bytes,
                        step.cuda_reserved_mb,
                        step.cuda_fragmentation_ratio,
                        step.step_latency_ms,
                        step.cumulative_ms,
                        1 if step.is_prefill else 0,
                    ),
                )

    def add_batch(self, steps: List[StepTelemetry]):
        """Append batch of step telemetry records."""
        with self.lock:
            self._records.extend(steps)
            rows = [
                (
                    s.step,
                    s.token_id,
                    s.token_str,
                    s.entropy,
                    s.norm_entropy,
                    s.margin_p1_p2,
                    s.top1_confidence,
                    json.dumps(s.top_candidates),
                    s.kv_cache_bytes_analytical,
                    s.kv_cache_mb_analytical,
                    s.cuda_allocated_bytes,
                    s.cuda_allocated_mb,
                    s.cuda_reserved_bytes,
                    s.cuda_reserved_mb,
                    s.cuda_fragmentation_ratio,
                    s.step_latency_ms,
                    s.cumulative_ms,
                    1 if s.is_prefill else 0,
                )
                for s in steps
            ]
            with self._conn:
                self._conn.executemany(
                    """
                    INSERT OR REPLACE INTO telemetry VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    rows,
                )

    def count(self) -> int:
        with self.lock:
            return len(self._records)

    def get_step(self, step_idx: int) -> Optional[StepTelemetry]:
        """Fetch exact step telemetry by index."""
        with self.lock:
            if 0 <= step_idx < len(self._records):
                return self._records[step_idx]
        return None

    def get_dataframe(self) -> pd.DataFrame:
        """Export current buffer as pandas DataFrame."""
        with self.lock:
            if not self._records:
                return pd.DataFrame(
                    columns=[
                        "step",
                        "token_id",
                        "token_str",
                        "entropy",
                        "norm_entropy",
                        "margin_p1_p2",
                        "top1_confidence",
                        "kv_cache_mb_analytical",
                        "cuda_allocated_mb",
                        "cuda_reserved_mb",
                        "cuda_fragmentation_ratio",
                        "step_latency_ms",
                        "cumulative_ms",
                        "is_prefill",
                    ]
                )
            df = pd.read_sql_query(
                """
                SELECT step, token_id, token_str, entropy, norm_entropy,
                       margin_p1_p2, top1_confidence, kv_cache_mb_analytical,
                       cuda_allocated_mb, cuda_reserved_mb, cuda_fragmentation_ratio,
                       step_latency_ms, cumulative_ms, is_prefill
                FROM telemetry ORDER BY step ASC
                """,
                self._conn,
            )
            return df

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates instantaneous summary statistics across the active run."""
        with self.lock:
            if not self._records:
                return {
                    "total_tokens": 0,
                    "ttft_ms": 0.0,
                    "mean_itl_ms": 0.0,
                    "jitter_ms": 0.0,
                    "tokens_per_sec": 0.0,
                    "peak_kv_cache_mb": 0.0,
                    "peak_cuda_alloc_mb": 0.0,
                    "peak_cuda_res_mb": 0.0,
                    "mean_entropy": 0.0,
                    "min_margin": 0.0,
                }

            latencies = [r.step_latency_ms for r in self._records]
            ttft = latencies[0]
            itl = latencies[1:] if len(latencies) > 1 else [latencies[0]]
            
            mean_itl = float(sum(itl) / len(itl))
            variance = sum((x - mean_itl) ** 2 for x in itl) / len(itl)
            jitter = float(variance ** 0.5)

            total_time_s = float(self._records[-1].cumulative_ms / 1000.0)
            tok_per_sec = (len(self._records) / total_time_s) if total_time_s > 0 else 0.0

            peak_kv = max(r.kv_cache_mb_analytical for r in self._records)
            peak_alloc = max(r.cuda_allocated_mb for r in self._records)
            peak_res = max(r.cuda_reserved_mb for r in self._records)
            mean_ent = sum(r.entropy for r in self._records) / len(self._records)
            min_marg = min(r.margin_p1_p2 for r in self._records)

            return {
                "total_tokens": len(self._records),
                "total_time_s": round(total_time_s, 2),
                "ttft_ms": round(ttft, 2),
                "mean_itl_ms": round(mean_itl, 2),
                "jitter_ms": round(jitter, 2),
                "tokens_per_sec": round(tok_per_sec, 2),
                "peak_kv_cache_mb": round(peak_kv, 3),
                "peak_cuda_alloc_mb": round(peak_alloc, 2),
                "peak_cuda_res_mb": round(peak_res, 2),
                "mean_entropy": round(mean_ent, 3),
                "min_margin": round(min_marg, 3),
            }
