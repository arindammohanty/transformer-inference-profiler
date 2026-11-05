"""
Main Plotly Dash Application: Transformer Inference Profiler & KV-Cache/VRAM Inspector.
Integrates live PyTorch hooks, dynamic Ollama model streaming, and synthetic architectural telemetry.
"""

from __future__ import annotations
import gc
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import psutil

import collections
import logging

# Silence Werkzeug request access logs (eliminates continuous stdout disk writes)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("werkzeug").disabled = True


class RamTelemetryLogger:
    """Stores recent telemetry events and error logs purely in memory (RAM)."""

    def __init__(self, max_entries: int = 2000):
        self._entries = collections.deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def log(self, level: str, message: str):
        with self._lock:
            self._entries.append(
                {
                    "timestamp": time.time(),
                    "level": level,
                    "message": message,
                }
            )

    def get_entries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._entries)


ram_logger = RamTelemetryLogger()

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import dash
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

from core.profiler import (
    ARCHITECTURE_PRESETS,
    StepTelemetry,
    TransformerInferenceProfiler,
)
from storage.buffer import TelemetryBuffer
from ui.components import (
    build_attention_scaling_figure,
    build_candidate_competition_chart,
    build_confidence_margin_chart,
    build_latency_distribution_chart,
    build_latency_entropy_chart,
    build_temperature_sensitivity_chart,
    build_telemetry_table,
    build_token_canvas_elements,
    build_vram_telemetry_chart,
    get_canvas_text_summary,
    render_kpi_card,
)

# High-Intensity Stress-Test Prompt Presets
PROMPT_PRESETS = {
    "math": (
        "Provide a rigorous mathematical and architectural breakdown of the memory-wall bottleneck in large transformer inference:\n\n"
        "1. Formulate the exact arithmetic intensity (FLOPs/byte) of the pre-fill phase versus autoregressive token generation for a 70B model with GQA (Grouped-Query Attention: 64 Q-heads, 8 KV-heads, hidden dimension 8192, 80 layers) across a 32,768-token context window in FP16 vs FP8.\n"
        "2. Derive the exact KV-cache memory footprints (in GiB) comparing Multi-Head Attention (MHA), Grouped-Query Attention (GQA-8), Multi-Query Attention (MQA), and DeepSeek Multi-Head Latent Attention (MLA with low-rank compression dim 512).\n"
        "3. Write a self-contained, high-performance PyTorch custom C++/CUDA or Triton kernel implementing PagedAttention block table lookup with dynamic virtual memory translation to eliminate internal fragmentation.\n"
        "4. Analyze the information-theoretic implications of top-k logit entropy spikes during chain-of-thought reasoning: explain why high-entropy steps predict branching perplexity and speculative decoding rejection rates."
    ),
    "triton": (
        "Implement a production-ready Triton JIT kernel for fused PagedAttention with causal masking and online softmax rescaling:\n\n"
        "1. Write the Triton kernel `@triton.jit` function accepting Q, K_cache, V_cache, block_tables, and context_lens tensors.\n"
        "2. Calculate optimal block dimensions (BLOCK_M, BLOCK_N) for Hopper/Ada Lovelace tensor cores and detail how shared memory bank conflicts are avoided.\n"
        "3. Provide theoretical derivations for memory bandwidth saturation, achieved memory throughput (GB/s), and arithmetic intensity relative to the Roofline model."
    ),
    "entropy": (
        "Investigate the mechanistic relationship between vocabulary logit entropy, hallucination branching, and speculative decoding verification:\n\n"
        "1. Derive the continuous Shannon entropy formula H(P_t) = -sum(p_i * log2(p_i)) and prove the upper bound H_max = log2(|V|).\n"
        "2. Formulate the mathematical conditions under which entropy spikes correlate with semantic hallucination and multi-token branching divergence.\n"
        "3. In speculative decoding (Medusa, EAGLE), prove how draft tree acceptance probabilities scale as a direct function of the entropy difference between target and draft distributions."
    ),
}

# Global Profiler and Buffer Singletons
profiler = TransformerInferenceProfiler()
telemetry_buffer = TelemetryBuffer()

# Background Worker State
worker_state = {
    "is_running": False,
    "selected_step": None,
    "last_error": None,
    "run_mode": "idle",
    "status_message": None,
    "max_tokens": 2048,
    "temperature": 0.7,
}


def run_profiler_background(
    engine_type: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    scenario: str,
):
    """Executes inference generation in background thread and streams records to buffer."""
    worker_state["is_running"] = True
    worker_state["last_error"] = None
    worker_state["max_tokens"] = max_tokens
    worker_state["temperature"] = temperature
    telemetry_buffer.clear()

    try:
        if engine_type == "ollama":
            generator = profiler.profile_ollama_generation(
                model_name=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        elif engine_type == "pytorch":
            generator = profiler.profile_pytorch_generation(
                prompt=prompt,
                model_name=model_name if model_name else "gpt2",
                max_new_tokens=max_tokens if max_tokens > 0 else 2048,
                temperature=temperature,
            )
        else:  # synthetic
            generator = profiler.profile_synthetic_benchmark(
                preset_name=model_name if model_name else "Llama-3-8B",
                total_steps=max_tokens if max_tokens > 0 else 120,
                initial_context_len=512,
                scenario=scenario,
                complete_sentences=True,
            )

        for step_record in generator:
            telemetry_buffer.add_step(step_record)
            if not worker_state["is_running"]:
                break
            if engine_type == "synthetic":
                time.sleep(0.004)

    except Exception as e:
        worker_state["last_error"] = str(e)
    finally:
        worker_state["is_running"] = False


# Initialize Dash Application
app = dash.Dash(
    __name__,
    title="Transformer Inference Profiler & KV-Cache/VRAM Inspector",
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)

# Query system status
sys_env = profiler.get_system_environment()
ollama_models = sys_env.get("ollama_models", [])
default_mode = "ollama" if ollama_models else "synthetic"

# Default to a safe, low-RAM model to prevent laptop freezing
default_model = "Llama-3-8B"
if ollama_models:
    safe_candidates = [m for m in ollama_models if profiler.ollama.estimate_model_ram_gb(m) < 8.0]
    default_model = safe_candidates[0] if safe_candidates else ollama_models[0]

# App Layout
app.layout = html.Div(
    [
        # Telemetry update interval (runs at steady 1 Hz during active generation)
        dcc.Interval(id="telemetry-interval", interval=1000, n_intervals=0),
        # Dedicated Host RAM update interval (independent slow poll every 3.5s)
        dcc.Interval(id="host-ram-interval", interval=3500, n_intervals=0),
        # Store for selected token step inspection
        dcc.Store(id="selected-step-store", data=0),
        
        # Header & System Environment Bar
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H3(
                                    [
                                        html.Span("⚡ TRANSFORMER ", style={"color": "#38bdf8", "fontWeight": "800"}),
                                        "INFERENCE PROFILER",
                                    ],
                                    style={"margin": "0", "letterSpacing": "-0.5px"},
                                ),
                                html.Div(
                                    "Real-Time Autoregressive Decoding Telemetry, Shannon Entropy & KV-Cache / VRAM Dynamics",
                                    style={"color": "#94a3b8", "fontSize": "0.85rem", "marginTop": "0.2rem"},
                                ),
                            ],
                            md=8,
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    [
                                        dbc.Badge(
                                            f"Device: {sys_env['device_name']}",
                                            color="info",
                                            className="me-2 p-2",
                                        ),
                                        dbc.Badge(
                                            f"PyTorch {sys_env['pytorch_version']}",
                                            color="secondary",
                                            className="me-2 p-2",
                                        ),
                                        dbc.Badge(
                                            f"Ollama: {'ONLINE (' + str(len(ollama_models)) + ' models)' if sys_env['ollama_active'] else 'OFFLINE'}",
                                            color="success" if sys_env["ollama_active"] else "warning",
                                            className="me-2 p-2",
                                        ),
                                        dbc.Badge(
                                            id="host-ram-badge",
                                            children="Host RAM: Detecting...",
                                            color="info",
                                            className="me-2 p-2",
                                        ),
                                        dbc.Badge(
                                            id="engine-status-badge",
                                            children="Status: Idle",
                                            color="secondary",
                                            className="p-2",
                                        ),
                                    ],
                                    className="d-flex justify-content-md-end align-items-center h-100 flex-wrap",
                                ),
                            ],
                            md=4,
                        ),
                    ]
                )
            ],
            className="profiler-header",
        ),

        # Main Container
        dbc.Container(
            [
                # Control & Parameter Panel
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Label("Engine Mode", className="fw-bold text-light mb-1 small"),
                                        dcc.Dropdown(
                                            id="engine-mode-select",
                                            options=[
                                                {
                                                    "label": f"🦙 Ollama Local Models ({len(ollama_models)} installed)",
                                                    "value": "ollama",
                                                    "disabled": not sys_env["ollama_active"],
                                                },
                                                {
                                                    "label": "🔥 PyTorch Local CausalLM (gpt2 / HF)",
                                                    "value": "pytorch",
                                                },
                                                {
                                                    "label": "📊 Synthetic Scale Benchmark (Llama-3, Qwen-2.5, DeepSeek)",
                                                    "value": "synthetic",
                                                },
                                            ],
                                            value=default_mode,
                                            clearable=False,
                                            className="custom-select",
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Target Architecture / Checkpoint", className="fw-bold text-light mb-1 small"),
                                        dcc.Dropdown(
                                            id="model-target-select",
                                            options=[{"label": m, "value": m} for m in ollama_models]
                                            if ollama_models
                                            else [{"label": k, "value": k} for k in ARCHITECTURE_PRESETS.keys()],
                                            value=default_model,
                                            clearable=False,
                                            className="custom-select",
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Label("Benchmark Scenario (for Synthetic)", className="fw-bold text-light mb-1 small"),
                                        dcc.Dropdown(
                                            id="scenario-select",
                                            options=[
                                                {"label": "Standard Inference", "value": "standard"},
                                                {"label": "High-Entropy Branching / Hallucination", "value": "high_entropy_branching"},
                                                {"label": "Memory Pressure Burst", "value": "memory_pressure"},
                                            ],
                                            value="standard",
                                            clearable=False,
                                            className="custom-select",
                                        ),
                                    ],
                                    md=4,
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.Label("Inference Prompt (Stress-Test Presets):", className="fw-bold text-light mb-1 small"),
                                                html.Div(
                                                    [
                                                        dbc.Button("🧠 Memory Wall & GQA Math", id="preset-btn-math", size="sm", color="secondary", outline=True, className="me-1 py-0 px-2 small"),
                                                        dbc.Button("⚡ Triton PagedAttention", id="preset-btn-triton", size="sm", color="secondary", outline=True, className="me-1 py-0 px-2 small"),
                                                        dbc.Button("🔬 Shannon Entropy & Branching", id="preset-btn-entropy", size="sm", color="secondary", outline=True, className="py-0 px-2 small"),
                                                    ],
                                                    className="d-flex align-items-center mb-1 flex-wrap",
                                                ),
                                            ],
                                            className="d-flex justify-content-between align-items-center flex-wrap",
                                        ),
                                        dbc.Textarea(
                                            id="prompt-input",
                                            placeholder="Enter prompt for autoregressive profiling...",
                                            value=PROMPT_PRESETS["math"],
                                            rows=3,
                                            className="custom-input",
                                        ),
                                    ],
                                    md=7,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Max Tokens", className="fw-bold text-light mb-1 small"),
                                                        dbc.Input(
                                                            id="max-tokens-input",
                                                            type="number",
                                                            value=2048,
                                                            min=0,
                                                            max=10000000,
                                                            step=1,
                                                            placeholder="Tokens (e.g. 2048, 4000, or 0 for EOS)",
                                                            className="custom-input",
                                                        ),
                                                        dbc.FormText(
                                                            "Limit (e.g. 4000) or 0 for unconstrained EOS",
                                                            className="text-muted small",
                                                            style={"fontSize": "0.72rem"},
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Temperature", className="fw-bold text-light mb-1 small"),
                                                        dbc.Input(
                                                            id="temperature-input",
                                                            type="number",
                                                            value=0.7,
                                                            min=0.1,
                                                            max=1.8,
                                                            step=0.1,
                                                            className="custom-input",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "⚡ Run Profiler",
                                                    id="run-button",
                                                    color="primary",
                                                    className="me-2 mt-2 px-3 fw-bold",
                                                ),
                                                dbc.Button(
                                                    "⏹ Stop",
                                                    id="stop-button",
                                                    color="danger",
                                                    outline=True,
                                                    className="me-2 mt-2 px-3",
                                                ),
                                                dbc.Button(
                                                    "🗑 Clear",
                                                    id="clear-button",
                                                    color="secondary",
                                                    outline=True,
                                                    className="me-2 mt-2 px-3",
                                                ),
                                                dbc.Button(
                                                    "🚨 Emergency Evict RAM",
                                                    id="emergency-button",
                                                    color="warning",
                                                    className="mt-2 px-3 fw-bold",
                                                ),
                                            ],
                                            className="d-flex align-items-center mt-1 flex-wrap",
                                        ),
                                    ],
                                    md=5,
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                dbc.Alert(id="ram-warning-alert", is_open=False, color="danger", className="py-2 mb-2"),
                                dbc.Checkbox(
                                    id="ram-override-checkbox",
                                    label="⚠️ Override Safety Lock: I acknowledge high memory consumption and accept the risk to run this model",
                                    value=False,
                                    className="fw-bold text-warning mb-2",
                                    style={"display": "none"},
                                ),
                            ],
                            id="ram-safety-wrapper",
                            className="mt-2",
                        ),
                        html.Div(id="status-alert", className="mt-2"),
                    ],
                    className="dashboard-panel",
                ),

                # KPI Metrics Cards Row
                dbc.Row(
                    [
                        dbc.Col(html.Div(id="kpi-ttft"), md=2),
                        dbc.Col(html.Div(id="kpi-itl"), md=2),
                        dbc.Col(html.Div(id="kpi-throughput"), md=2),
                        dbc.Col(html.Div(id="kpi-kv-cache"), md=2),
                        dbc.Col(html.Div(id="kpi-vram"), md=2),
                        dbc.Col(html.Div(id="kpi-entropy"), md=2),
                    ],
                    className="mb-4 g-3",
                ),

                # Token Entropy Canvas Panel
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span("🔤 Token Entropy Canvas", className="panel-title mb-0"),
                                        html.Span(id="canvas-stats-tokens", className="text-info small ms-2 fw-normal"),
                                    ],
                                    className="d-flex align-items-center flex-wrap",
                                ),
                                html.Div(
                                    [
                                        # View Mode
                                        html.Span("View:", className="small text-muted me-1"),
                                        dbc.RadioItems(
                                            id="canvas-view-mode",
                                            options=[
                                                {"label": "📄 Paragraph Flow", "value": "flow"},
                                                {"label": "🏷️ Pills", "value": "pills"},
                                            ],
                                            value="flow",
                                            inline=True,
                                            className="me-3 small text-light",
                                        ),
                                        # Filter Mode
                                        html.Span("Filter:", className="small text-muted me-1"),
                                        dcc.Dropdown(
                                            id="canvas-filter-mode",
                                            options=[
                                                {"label": "🔍 All Tokens", "value": "all"},
                                                {"label": "🔥 High Entropy (H > 2.0)", "value": "high_entropy"},
                                                {"label": "⚖️ Uncertain (Top-1 < 50%)", "value": "uncertain"},
                                            ],
                                            value="all",
                                            clearable=False,
                                            style={"width": "180px"},
                                            className="custom-select me-2 small",
                                        ),
                                        # Canvas Sizing
                                        html.Span("Height:", className="small text-muted me-1"),
                                        dcc.Dropdown(
                                            id="canvas-height-mode",
                                            options=[
                                                {"label": "Standard (360px)", "value": "height-standard"},
                                                {"label": "Expanded (600px)", "value": "height-expanded"},
                                                {"label": "Fit All (Auto)", "value": "height-auto"},
                                            ],
                                            value="height-standard",
                                            clearable=False,
                                            style={"width": "160px"},
                                            className="custom-select me-2 small",
                                        ),
                                        # Copy Button
                                        dcc.Clipboard(
                                            id="canvas-copy-btn",
                                            target_id="canvas-copy-target",
                                            title="Copy full paragraph text to clipboard",
                                            className="btn btn-sm btn-outline-info py-0 px-2 me-2",
                                        ),
                                        html.Textarea(id="canvas-copy-target", style={"display": "none"}),
                                        # Certainty Gradient
                                        html.Div(
                                            [
                                                html.Span("Certainty ", style={"fontSize": "0.72rem", "color": "#10b981"}),
                                                html.Span(
                                                    style={
                                                        "display": "inline-block",
                                                        "width": "45px",
                                                        "height": "6px",
                                                        "background": "linear-gradient(90deg, #10b981, #06b6d4, #f59e0b, #f43f5e)",
                                                        "borderRadius": "3px",
                                                        "margin": "0 4px",
                                                    }
                                                ),
                                                html.Span(" Branching", style={"fontSize": "0.72rem", "color": "#f43f5e"}),
                                            ],
                                            className="d-flex align-items-center",
                                        ),
                                    ],
                                    className="d-flex align-items-center flex-wrap mt-2 mt-md-0",
                                ),
                            ],
                            className="d-flex justify-content-between align-items-center flex-wrap mb-2",
                        ),
                        html.Div(
                            id="token-canvas-container",
                            className="token-canvas-container height-standard mode-flow",
                        ),
                        html.Div(
                            "💡 Tip: Click on any word/token in the canvas above to inspect candidate logits and simulate softmax temperature sensitivity below. Use '📄 Paragraph Flow' for long essays/reports or 'Fit All (Auto)' to view the entire response without scrollbars.",
                            style={"fontSize": "0.75rem", "color": "#64748b", "marginTop": "0.5rem"},
                        ),
                    ],
                    className="dashboard-panel",
                ),

                # Advanced Telemetry & Mechanistic Interpretability Tabbed Visualizations
                html.Div(
                    [
                        dbc.Tabs(
                            [
                                # Tab 1: Hardware & Memory Telemetry
                                dbc.Tab(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                id="vram-telemetry-chart",
                                                                config={"displayModeBar": False},
                                                            )
                                                        ],
                                                        className="dashboard-panel",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                id="attention-scaling-chart",
                                                                figure=build_attention_scaling_figure(),
                                                                config={"displayModeBar": False},
                                                            )
                                                        ],
                                                        className="dashboard-panel",
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="g-3 mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                id="latency-entropy-chart",
                                                                config={"displayModeBar": False},
                                                            )
                                                        ],
                                                        className="dashboard-panel",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                id="latency-distribution-chart",
                                                                config={"displayModeBar": False},
                                                            )
                                                        ],
                                                        className="dashboard-panel",
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="g-3 mb-2",
                                        ),
                                    ],
                                    label="⚡ Telemetry & Hardware Dynamics",
                                    tab_id="tab-telemetry",
                                    tab_class_name="fw-bold",
                                    className="pt-3",
                                ),

                                # Tab 2: Mechanistic Interpretability & Logit Space
                                dbc.Tab(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                id="candidate-competition-chart",
                                                                config={"displayModeBar": False},
                                                            )
                                                        ],
                                                        className="dashboard-panel",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                id="temperature-sensitivity-chart",
                                                                config={"displayModeBar": False},
                                                            )
                                                        ],
                                                        className="dashboard-panel",
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="g-3 mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            dcc.Graph(
                                                                id="confidence-margin-chart",
                                                                config={"displayModeBar": False},
                                                            )
                                                        ],
                                                        className="dashboard-panel",
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        [
                                                            html.Div(
                                                                [
                                                                    html.Span("📋 Step Telemetry Audit Table (Scrollable Detailed Telemetry)", className="panel-title mb-2"),
                                                                    html.Div(id="telemetry-table-container"),
                                                                ]
                                                            )
                                                        ],
                                                        className="dashboard-panel",
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="g-3 mb-2",
                                        ),
                                    ],
                                    label="🧠 Mechanistic Interpretability & Logit Space",
                                    tab_id="tab-mechanistic",
                                    tab_class_name="fw-bold",
                                    className="pt-3",
                                ),
                            ],
                            active_tab="tab-telemetry",
                        )
                    ],
                    className="mb-4",
                ),
            ],
            fluid=True,
            style={"maxWidth": "1500px"},
        ),
    ]
)


# Callback: Update model choices based on selected engine mode
@app.callback(
    Output("model-target-select", "options"),
    Output("model-target-select", "value"),
    Input("engine-mode-select", "value"),
)
def update_model_dropdown(engine_mode):
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024**3)

    if engine_mode == "ollama":
        models = profiler.ollama.list_models()
        opts = []
        safe_default = None
        for m in models:
            m_name = m.get("name")
            est_ram = profiler.ollama.estimate_model_ram_gb(m_name)
            if est_ram > avail_gb or est_ram >= 15.0:
                tag = f"🔴 {m_name} [DANGER: ~{est_ram:.1f}GB RAM]"
            elif est_ram >= 8.0:
                tag = f"🟡 {m_name} [HEAVY: ~{est_ram:.1f}GB RAM]"
            else:
                tag = f"🟢 {m_name} [SAFE: ~{est_ram:.1f}GB RAM]"
                if safe_default is None:
                    safe_default = m_name
            opts.append({"label": tag, "value": m_name})

        val = safe_default or (opts[0]["value"] if opts else "")
        return opts, val
    elif engine_mode == "pytorch":
        opts = [
            {"label": "🟢 gpt2 (Lightweight local CausalLM, ~0.5GB)", "value": "gpt2"},
            {"label": "🟢 distilgpt2 (Ultra-fast, ~0.3GB)", "value": "distilgpt2"},
        ]
        return opts, "gpt2"
    else:  # synthetic
        opts = [
            {"label": f"📊 {k} ({v['params']} - {v['attn_type']})", "value": k}
            for k, v in ARCHITECTURE_PRESETS.items()
        ]
        return opts, "Llama-3-8B"


# Callback: Switch Inference Prompt Presets
@app.callback(
    Output("prompt-input", "value"),
    Input("preset-btn-math", "n_clicks"),
    Input("preset-btn-triton", "n_clicks"),
    Input("preset-btn-entropy", "n_clicks"),
    prevent_initial_call=True,
)
def update_prompt_preset(btn_math, btn_triton, btn_entropy):
    trigger = ctx.triggered_id
    if trigger == "preset-btn-math":
        return PROMPT_PRESETS["math"]
    elif trigger == "preset-btn-triton":
        return PROMPT_PRESETS["triton"]
    elif trigger == "preset-btn-entropy":
        return PROMPT_PRESETS["entropy"]
    return dash.no_update


# Callback: Check RAM Safety & Enforce Override Lock
@app.callback(
    Output("ram-warning-alert", "is_open"),
    Output("ram-warning-alert", "children"),
    Output("ram-warning-alert", "color"),
    Output("ram-override-checkbox", "style"),
    Output("run-button", "disabled"),
    Input("engine-mode-select", "value"),
    Input("model-target-select", "value"),
    Input("ram-override-checkbox", "value"),
)
def check_ram_safety(engine_mode, model_name, override_checked):
    if engine_mode != "ollama" or not model_name:
        return False, "", "info", {"display": "none"}, False

    est_ram = profiler.ollama.estimate_model_ram_gb(model_name)
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024**3)
    total_gb = mem.total / (1024**3)

    if est_ram > avail_gb or est_ram >= 15.0:
        warning_children = [
            html.B("🚨 HIGH-RAM HAZARD & SAFETY LOCK ACTIVATED: "),
            f"Selected model '{model_name}' requires approximately {est_ram:.1f} GB RAM, but only {avail_gb:.1f} GB is currently available ({total_gb:.1f} GB total physical RAM). ",
            html.Br(),
            html.Span(
                "Running this model will exhaust physical memory, heavily saturate swap, and risks freezing your system! The 'Run Profiler' button is locked. If you have sufficient swap and wish to continue, check the override box below.",
                style={"color": "#fca5a5"},
            ),
        ]
        checkbox_style = {"display": "block"}
        button_disabled = not bool(override_checked)
        return True, warning_children, "danger", checkbox_style, button_disabled

    elif est_ram >= 8.0 or est_ram > (avail_gb * 0.7):
        warning_children = [
            html.B("⚠️ MODERATE/HIGH RAM NOTICE: "),
            f"Model '{model_name}' will consume approximately {est_ram:.1f} GB RAM ({avail_gb:.1f} GB available). Ensure no other heavy applications are running.",
        ]
        return True, warning_children, "warning", {"display": "none"}, False

    else:
        return False, "", "success", {"display": "none"}, False


# Callback: Handle Action Buttons (Run, Stop, Clear, Emergency Evict)
@app.callback(
    Output("status-alert", "children", allow_duplicate=True),
    Input("run-button", "n_clicks"),
    Input("stop-button", "n_clicks"),
    Input("clear-button", "n_clicks"),
    Input("emergency-button", "n_clicks"),
    State("engine-mode-select", "value"),
    State("model-target-select", "value"),
    State("prompt-input", "value"),
    State("max-tokens-input", "value"),
    State("temperature-input", "value"),
    State("scenario-select", "value"),
    State("ram-override-checkbox", "value"),
    prevent_initial_call=True,
)
def handle_controls(run_btn, stop_btn, clear_btn, emerg_btn, mode, model, prompt, max_tokens, temp, scenario, ram_override):
    trigger = ctx.triggered_id
    if trigger == "emergency-button":
        worker_state["is_running"] = False
        worker_state["run_mode"] = "needs_render"
        worker_state["last_rendered_token_count"] = -1
        unloaded = profiler.ollama.unload_all_models()
        telemetry_buffer.clear()
        gc.collect()
        models_str = ", ".join(unloaded) if unloaded else "none running"
        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024**3)
        worker_state["status_message"] = ("emergency", f"Evicted active models ({models_str}) from RAM. Caches purged. Available RAM restored to {avail_gb:.1f} GB.")
        return dbc.Alert(
            [
                html.B("🚨 EMERGENCY EVICTION EXECUTED: "),
                f"Evicted active models ({models_str}) from RAM. Caches purged. Available RAM restored to {avail_gb:.1f} GB.",
            ],
            color="success",
            dismissable=True,
            className="py-2",
        )

    elif trigger == "run-button":
        if worker_state["is_running"]:
            return dbc.Alert("Profiling already active!", color="warning", dismissable=True)

        if mode == "ollama" and model:
            est_ram = profiler.ollama.estimate_model_ram_gb(model)
            avail_gb = psutil.virtual_memory().available / (1024**3)
            if (est_ram > avail_gb or est_ram >= 15.0) and not ram_override:
                return dbc.Alert(
                    f"🚫 EXECUTION BLOCKED: Model '{model}' requires ~{est_ram:.1f} GB RAM, which exceeds safe thresholds. You must check the override safety lock to run.",
                    color="danger",
                    dismissable=True,
                    className="py-2",
                )

        try:
            if max_tokens is None or str(max_tokens).strip() == "":
                raw_tokens = 2048
            else:
                raw_tokens = int(float(str(max_tokens).strip()))
        except (ValueError, TypeError):
            raw_tokens = 2048

        if raw_tokens <= 0:
            effective_max_tokens = 0  # 0 indicates unconstrained generation until natural EOS
        elif raw_tokens >= 100000:
            effective_max_tokens = 65536
        else:
            effective_max_tokens = max(5, raw_tokens)

        temp = max(0.1, min(float(temp or 0.7), 2.0))

        worker_state["run_mode"] = "running"
        worker_state["last_rendered_token_count"] = -1
        worker_state["status_message"] = ("running", f"Active Profiling: Engine '{mode}', Model '{model}'...")
        thread = threading.Thread(
            target=run_profiler_background,
            args=(mode, model, prompt, effective_max_tokens, temp, scenario),
            daemon=True,
        )
        thread.start()
        return dbc.Alert(
            [
                dbc.Spinner(size="sm", spinner_class_name="me-2"),
                f"Active Profiling: Engine '{mode}', Model '{model}'...",
            ],
            color="info",
            className="py-1",
        )

    elif trigger == "stop-button":
        worker_state["is_running"] = False
        worker_state["run_mode"] = "needs_render"
        worker_state["status_message"] = ("stopped", "Profiling stopped by user.")
        return dbc.Alert("Profiling stopped by user.", color="secondary", dismissable=True, className="py-1")

    elif trigger == "clear-button":
        worker_state["is_running"] = False
        worker_state["run_mode"] = "needs_render"
        worker_state["last_rendered_token_count"] = -1
        telemetry_buffer.clear()
        worker_state["selected_step"] = None
        worker_state["status_message"] = ("cleared", "Telemetry buffer cleared.")
        return dbc.Alert("Telemetry buffer cleared.", color="secondary", dismissable=True, className="py-1")

    return dash.no_update


# Callback: Select token step when user clicks any token pill in the canvas
@app.callback(
    Output("selected-step-store", "data"),
    Input({"type": "token-pill", "index": ALL}, "n_clicks"),
    State("selected-step-store", "data"),
    prevent_initial_call=True,
)
def handle_token_click(n_clicks_list, current_selected):
    try:
        tid = ctx.triggered_id
        if isinstance(tid, dict) and "index" in tid:
            selected_idx = int(tid["index"])
            worker_state["selected_step"] = selected_idx
            return selected_idx
        if ctx.triggered:
            triggered_prop = ctx.triggered[0]["prop_id"]
            dict_part = triggered_prop.split(".")[0]
            parsed = json.loads(dict_part)
            selected_idx = int(parsed.get("index", current_selected or 0))
            worker_state["selected_step"] = selected_idx
            return selected_idx
    except Exception:
        pass
    return current_selected


# Host RAM Real-Time Telemetry Callback
@app.callback(
    Output("host-ram-badge", "children"),
    Output("host-ram-badge", "color"),
    Input("host-ram-interval", "n_intervals"),
)
def update_host_ram_telemetry(n_intervals):
    mem = psutil.virtual_memory()
    used_gb = (mem.total - mem.available) / (1024**3)
    total_gb = mem.total / (1024**3)
    pct = mem.percent
    host_ram_text = f"Host RAM: {used_gb:.1f}/{total_gb:.1f} GB ({pct:.0f}%)"
    if pct > 85:
        host_ram_color = "danger"
    elif pct > 70:
        host_ram_color = "warning"
    else:
        host_ram_color = "info"
    return host_ram_text, host_ram_color


# Main Real-Time Telemetry Update Callback
@app.callback(
    Output("kpi-ttft", "children"),
    Output("kpi-itl", "children"),
    Output("kpi-throughput", "children"),
    Output("kpi-kv-cache", "children"),
    Output("kpi-vram", "children"),
    Output("kpi-entropy", "children"),
    Output("token-canvas-container", "children"),
    Output("token-canvas-container", "className"),
    Output("canvas-stats-tokens", "children"),
    Output("canvas-copy-target", "value"),
    Output("candidate-competition-chart", "figure"),
    Output("temperature-sensitivity-chart", "figure"),
    Output("confidence-margin-chart", "figure"),
    Output("telemetry-table-container", "children"),
    Output("vram-telemetry-chart", "figure"),
    Output("latency-entropy-chart", "figure"),
    Output("latency-distribution-chart", "figure"),
    Output("engine-status-badge", "children"),
    Output("engine-status-badge", "color"),
    Output("status-alert", "children", allow_duplicate=True),
    Input("telemetry-interval", "n_intervals"),
    Input("selected-step-store", "data"),
    Input("canvas-view-mode", "value"),
    Input("canvas-filter-mode", "value"),
    Input("canvas-height-mode", "value"),
    prevent_initial_call="initial_duplicate",
)
def update_dashboard_telemetry(
    n_intervals,
    selected_step,
    view_mode="flow",
    filter_mode="all",
    height_mode="height-standard",
):
    # Guard against unnecessary DOM re-renders and UI flicker when idle
    try:
        trigger = ctx.triggered_id
    except Exception:
        trigger = None

    is_running = bool(worker_state.get("is_running", False))
    run_mode = worker_state.get("run_mode", "idle")

    # If interval triggered while completely idle without new state changes, skip DOM re-render
    if trigger == "telemetry-interval" and not is_running and run_mode == "idle":
        return (dash.no_update,) * 20

    # If this was the final render after finishing or clearing, transition to idle
    if not is_running and run_mode in ("running", "needs_render"):
        worker_state["run_mode"] = "idle"

    # If streaming, check if new tokens arrived since last render; if unchanged, skip heavy redraws
    current_token_count = telemetry_buffer.count()
    if trigger == "telemetry-interval" and is_running and current_token_count == worker_state.get("last_rendered_token_count", -1):
        return (dash.no_update,) * 20
    worker_state["last_rendered_token_count"] = current_token_count

    # If triggered purely by token click inspection, update ONLY candidate & temperature figures!
    # This completely eliminates canvas DOM thrashing, preserves text selection, and prevents scroll jumping.
    if trigger == "selected-step-store":
        with telemetry_buffer.lock:
            records = list(telemetry_buffer._records)
        target_record = None
        if records:
            sel_idx = selected_step if (selected_step is not None and 0 <= selected_step < len(records)) else len(records) - 1
            target_record = records[sel_idx]
        candidate_fig = build_candidate_competition_chart(target_record)
        current_temp = float(worker_state.get("temperature", 0.7))
        temp_fig = build_temperature_sensitivity_chart(target_record, current_temp=current_temp)

        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            candidate_fig,
            temp_fig,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
            dash.no_update,
        )

    stats = telemetry_buffer.get_summary_stats()
    df = telemetry_buffer.get_dataframe()
    with telemetry_buffer.lock:
        records = list(telemetry_buffer._records)

    # 1. Status Badge & Alert Banner Logic
    alert_children = None
    if worker_state.get("last_error"):
        status_text = f"❌ Error: {worker_state['last_error'][:30]}"
        status_color = "danger"
        alert_children = dbc.Alert(
            [html.B("❌ Error during profiling: "), worker_state["last_error"]],
            color="danger",
            dismissable=True,
            className="py-2",
        )
    elif worker_state.get("is_running"):
        status_text = f"⚡ Streaming Token #{len(records)}..."
        status_color = "primary"
        alert_children = dbc.Alert(
            [
                dbc.Spinner(size="sm", spinner_class_name="me-2"),
                html.Span(f"⚡ Active Profiling: Streaming token #{len(records)} ({stats['tokens_per_sec']:.1f} tok/s)...", className="fw-semibold"),
            ],
            color="info",
            className="py-2",
        )
    elif records:
        dur = stats.get("total_time_s", 0.0)
        status_text = f"✅ Finished ({len(records)} tokens)"
        status_color = "success"
        if worker_state.get("status_message") and worker_state["status_message"][0] == "stopped":
            alert_children = dbc.Alert("⏹ Profiling stopped by user.", color="secondary", dismissable=True, className="py-2")
        elif worker_state.get("status_message") and worker_state["status_message"][0] == "emergency":
            alert_children = dbc.Alert([html.B("🚨 EMERGENCY EVICTION: "), worker_state["status_message"][1]], color="success", dismissable=True, className="py-2")
        else:
            alert_children = dbc.Alert(
                [
                    html.B("✅ Profiling Complete: "),
                    f"Generated {len(records)} tokens in {dur:.2f}s ({stats['tokens_per_sec']:.1f} tok/s). Mean Entropy: {stats['mean_entropy']:.2f} bits, Peak KV-Cache: {stats['peak_kv_cache_mb']:.2f} MB.",
                ],
                color="success",
                dismissable=True,
                className="py-2",
            )
    else:
        status_text = "Status: Idle"
        status_color = "secondary"
        if worker_state.get("status_message") and worker_state["status_message"][0] == "cleared":
            alert_children = dbc.Alert("Telemetry buffer cleared.", color="secondary", dismissable=True, className="py-2")
        elif worker_state.get("status_message") and worker_state["status_message"][0] == "emergency":
            alert_children = dbc.Alert([html.B("🚨 EMERGENCY EVICTION: "), worker_state["status_message"][1]], color="success", dismissable=True, className="py-2")
        else:
            alert_children = None

    # 2. KPI Cards
    kpi_ttft = render_kpi_card(
        "Time-to-First-Token",
        f"{stats['ttft_ms']:.1f}",
        "ms",
        accent_color="#f43f5e",
        subtitle="Pre-fill Phase Latency",
    )
    kpi_itl = render_kpi_card(
        "Inter-Token Latency",
        f"{stats['mean_itl_ms']:.1f}",
        "ms",
        accent_color="#38bdf8",
        subtitle=f"Jitter: ±{stats['jitter_ms']:.1f} ms",
    )
    kpi_throughput = render_kpi_card(
        "Throughput",
        f"{stats['tokens_per_sec']:.1f}",
        "tok/s",
        accent_color="#10b981",
        subtitle=f"Total: {stats['total_tokens']} tokens",
    )
    kpi_kv = render_kpi_card(
        "Peak KV-Cache",
        f"{stats['peak_kv_cache_mb']:.2f}",
        "MB",
        accent_color="#06b6d4",
        subtitle="Analytical Exact Footprint",
    )
    kpi_vram = render_kpi_card(
        "Peak GPU VRAM",
        f"{stats['peak_cuda_alloc_mb']:.0f}",
        "MB",
        accent_color="#a855f7",
        subtitle=f"Reserved: {stats['peak_cuda_res_mb']:.0f} MB",
    )
    kpi_entropy = render_kpi_card(
        "Mean Shannon Entropy",
        f"{stats['mean_entropy']:.2f}",
        "bits",
        accent_color="#f59e0b",
        subtitle=f"Min Margin: {stats['min_margin']:.2f}",
    )

    # 3. Token Canvas & Dynamic Long-Paragraph Handling
    summary = get_canvas_text_summary(records)
    canvas_class = f"token-canvas-container {height_mode or 'height-standard'} mode-{view_mode or 'flow'}"
    canvas_stats = (
        f"({summary['token_count']} tokens | {summary['word_count']} words | Avg H: {summary['mean_entropy']:.2f} bits | {summary['branching_count']} branching)"
        if records
        else ""
    )
    copy_text = summary["full_text"]
    canvas_elements = build_token_canvas_elements(
        records,
        selected_step=selected_step,
        view_mode=view_mode or "flow",
        filter_mode=filter_mode or "all",
    )

    # 4. Target step record for detailed inspection
    target_record = None
    if records:
        sel_idx = selected_step if (selected_step is not None and 0 <= selected_step < len(records)) else len(records) - 1
        target_record = records[sel_idx]

    # 5. Charts & Visualizations
    candidate_fig = build_candidate_competition_chart(target_record)
    current_temp = float(worker_state.get("temperature", 0.7))
    temp_fig = build_temperature_sensitivity_chart(target_record, current_temp=current_temp)
    margin_fig = build_confidence_margin_chart(df)
    telemetry_table = build_telemetry_table(records)
    vram_fig = build_vram_telemetry_chart(df)
    latency_fig = build_latency_entropy_chart(df)
    latency_dist_fig = build_latency_distribution_chart(df)

    return (
        kpi_ttft,
        kpi_itl,
        kpi_throughput,
        kpi_kv,
        kpi_vram,
        kpi_entropy,
        canvas_elements,
        canvas_class,
        canvas_stats,
        copy_text,
        candidate_fig,
        temp_fig,
        margin_fig,
        telemetry_table,
        vram_fig,
        latency_fig,
        latency_dist_fig,
        status_text,
        status_color,
        alert_children,
    )


# Application entrypoint
server = app.server

if __name__ == "__main__":
    print("Starting Transformer Inference Profiler on http://127.0.0.1:8050 ...")
    app.run(host="0.0.0.0", port=8050, debug=False)
