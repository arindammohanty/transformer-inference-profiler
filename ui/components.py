"""
UI Components, Visual Widgets, and Plotly WebGL Figure Builders
for the Transformer Inference Profiler Dashboard.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from core.metrics import compare_attention_architectures
from core.profiler import StepTelemetry


def entropy_to_color(norm_entropy: float) -> str:
    """
    Interpolates normalized entropy [0.0, 1.0] to RGBA background color:
    - 0.0 -> Cool Emerald (Certainty / Low Entropy)
    - 0.3 -> Cyan
    - 0.6 -> Amber (Decision branching)
    - 1.0 -> Vibrant Crimson (High uncertainty / Hallucination risk)
    """
    e = max(0.0, min(1.0, norm_entropy))
    if e < 0.35:
        t = e / 0.35
        r = int(16 * (1 - t) + 6 * t)
        g = int(185 * (1 - t) + 182 * t)
        b = int(129 * (1 - t) + 212 * t)
        alpha = 0.35 + 0.15 * t
    elif e < 0.7:
        t = (e - 0.35) / 0.35
        r = int(6 * (1 - t) + 245 * t)
        g = int(182 * (1 - t) + 158 * t)
        b = int(212 * (1 - t) + 11 * t)
        alpha = 0.45 + 0.15 * t
    else:
        t = (e - 0.7) / 0.3
        r = int(245 * (1 - t) + 244 * t)
        g = int(158 * (1 - t) + 63 * t)
        b = int(11 * (1 - t) + 94 * t)
        alpha = 0.6 + 0.25 * t

    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


def render_kpi_card(
    title: str,
    value: str,
    unit: str = "",
    accent_color: str = "#38bdf8",
    subtitle: Optional[str] = None,
) -> dbc.Card:
    """Render a styled KPI metric widget."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="kpi-title"),
                html.Div(
                    [
                        html.Span(value, className="kpi-value", style={"color": accent_color}),
                        html.Span(unit, className="kpi-unit") if unit else None,
                    ]
                ),
                html.Div(
                    subtitle,
                    style={"fontSize": "0.75rem", "color": "#64748b", "marginTop": "0.2rem"},
                )
                if subtitle
                else None,
            ],
            className="p-3",
        ),
        className="kpi-card",
    )


def get_canvas_text_summary(records: List[StepTelemetry]) -> Dict[str, Any]:
    """Generates reading statistics and text aggregation for long paragraphs."""
    if not records:
        return {
            "full_text": "",
            "token_count": 0,
            "word_count": 0,
            "char_count": 0,
            "mean_entropy": 0.0,
            "branching_count": 0,
        }
    full_text = "".join(r.token_str for r in records)
    word_count = len(full_text.split())
    char_count = len(full_text)
    mean_entropy = sum(r.entropy for r in records) / len(records)
    branching_count = sum(1 for r in records if r.entropy >= 2.0)
    return {
        "full_text": full_text,
        "token_count": len(records),
        "word_count": word_count,
        "char_count": char_count,
        "mean_entropy": mean_entropy,
        "branching_count": branching_count,
    }


def build_token_canvas_elements(
    records: List[StepTelemetry],
    selected_step: Optional[int] = None,
    view_mode: str = "flow",
    filter_mode: str = "all",
) -> List[Any]:
    """
    Renders the Token Entropy Canvas:
    Supports dynamic modes:
    - 'flow': Natural Paragraph Flow for seamless, readable paragraphs of any length.
    - 'pills': Classic discrete badge pills.
    Supports filtering: 'all', 'high_entropy' (H >= 2.0), 'uncertain' (p1 < 0.50).
    """
    if not records:
        return [
            html.Div(
                "Inference idle. Enter a prompt and click '⚡ Run Profiler' to stream tokens and inspect live entropy.",
                style={"color": "#64748b", "fontStyle": "italic", "padding": "1rem 0"},
            )
        ]

    elements: List[Any] = []
    # Window to latest 1000 tokens when buffer is massive to keep DOM responsive
    display_records = records[-1000:] if len(records) > 1000 else records

    for r in display_records:
        bg_color = entropy_to_color(r.norm_entropy)
        is_selected = (selected_step is not None and r.step == selected_step)

        # Check if dimmed by filter
        dimmed = False
        if filter_mode == "high_entropy" and r.entropy < 2.0:
            dimmed = True
        elif filter_mode == "uncertain" and r.top1_confidence >= 0.50:
            dimmed = True

        tooltip_text = f"Step #{r.step}: {repr(r.token_str)} | H={r.entropy:.2f}b ({r.norm_entropy:.0%}) | Top-1={r.top1_confidence:.0%} | {r.step_latency_ms:.1f}ms"

        if view_mode == "flow":
            raw_str = r.token_str
            style: Dict[str, Any] = {}
            if dimmed:
                style["backgroundColor"] = "transparent"
                style["color"] = "rgba(148, 163, 184, 0.35)"
                style["borderBottom"] = f"2px solid {bg_color}"
            else:
                style["backgroundColor"] = bg_color

            if is_selected:
                style["outline"] = "2px solid #38bdf8"
                style["boxShadow"] = "0 0 10px rgba(56, 189, 248, 0.9)"
                style["fontWeight"] = "bold"

            elements.append(
                html.Span(
                    raw_str,
                    id={"type": "token-pill", "index": r.step},
                    className="token-flow-span selected" if is_selected else "token-flow-span",
                    style=style,
                    title=tooltip_text,
                    tabIndex=0,
                )
            )
        else:
            token_display = r.token_str.replace(" ", "␣") if r.token_str == " " else r.token_str
            style = {}
            if dimmed:
                style["backgroundColor"] = "rgba(15, 23, 42, 0.4)"
                style["color"] = "rgba(148, 163, 184, 0.35)"
                style["opacity"] = 0.35
            else:
                style["backgroundColor"] = bg_color

            if is_selected:
                style["outline"] = "2px solid #38bdf8"
                style["boxShadow"] = "0 0 12px rgba(56, 189, 248, 0.8)"
                style["fontWeight"] = "bold"

            elements.append(
                html.Span(
                    token_display,
                    id={"type": "token-pill", "index": r.step},
                    className="token-pill selected" if is_selected else "token-pill",
                    style=style,
                    title=tooltip_text,
                    tabIndex=0,
                )
            )

    return elements


def build_candidate_competition_chart(
    step_record: Optional[StepTelemetry],
) -> go.Figure:
    """
    Builds the Candidate Competition View:
    Displays top-5 candidate tokens, probabilities, and logit values for the selected decoding step.
    """
    fig = go.Figure()
    if step_record is None or not step_record.top_candidates:
        fig.add_annotation(
            text="Click any token in the Entropy Canvas to inspect its candidate logits.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=13, color="#64748b"),
        )
        fig.update_layout(
            paper_bgcolor="#111726",
            plot_bgcolor="#111726",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    candidates = step_record.top_candidates
    tokens = [c["token"] for c in candidates][::-1]
    probs = [c["prob"] for c in candidates][::-1]
    logits = [c["logit"] for c in candidates][::-1]

    colors = ["#10b981" if i == len(candidates) - 1 else "#38bdf8" for i in range(len(candidates))]

    fig.add_trace(
        go.Bar(
            y=tokens,
            x=probs,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.2)", width=1),
            ),
            text=[f"{p:.1%}  (logit: {l:.1f})" for p, l in zip(probs, logits)],
            textposition="auto",
            textfont=dict(color="#ffffff", size=11, family="monospace"),
            hovertemplate="Candidate: <b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Step #{step_record.step} Candidate Probabilities & Logit Margins (Margin: {step_record.margin_p1_p2:.3f})",
            font=dict(size=13, color="#e2e8f0"),
            x=0.01,
            xanchor="left",
            y=0.97,
        ),
        paper_bgcolor="#111726",
        plot_bgcolor="#111726",
        height=340,
        margin=dict(l=90, r=40, t=50, b=40),
        xaxis=dict(
            title="Probability",
            range=[0, max(1.0, max(probs) * 1.15 if probs else 1.0)],
            tickformat=".0%",
            color="#94a3b8",
            gridcolor="#1e293b",
            automargin=True,
        ),
        yaxis=dict(
            color="#e2e8f0",
            tickfont=dict(family="monospace", size=12),
            automargin=True,
        ),
    )
    return fig


def build_vram_telemetry_chart(df: pd.DataFrame) -> go.Figure:
    """
    Builds the Dual-Axis VRAM & Cache Allocation Telemetry chart:
    - Primary Y-axis: Analytical KV-cache memory in MB (O(N) growth).
    - Secondary Y-axis: CUDA Allocated vs Reserved Memory in MB.
    Legend placed at bottom to prevent title collision.
    """
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(
            text="Awaiting telemetry data...",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#64748b"),
        )
        fig.update_layout(
            paper_bgcolor="#111726",
            plot_bgcolor="#111726",
            height=340,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    # Downsample large datasets (> 250 points) to maintain 60 FPS rendering
    if len(df) > 250:
        stride = max(1, len(df) // 250)
        sampled_indices = list(range(0, len(df), stride))
        if (len(df) - 1) not in sampled_indices:
            sampled_indices.append(len(df) - 1)
        plot_df = df.iloc[sampled_indices]
    else:
        plot_df = df

    steps = plot_df["step"]
    use_markers = len(plot_df) <= 60
    scatter_mode = "lines+markers" if use_markers else "lines"

    # 1. Analytical KV Cache footprint (MB)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=plot_df["kv_cache_mb_analytical"],
            name="Analytical KV-Cache (MB)",
            mode=scatter_mode,
            line=dict(color="#06b6d4", width=2.5),
            marker=dict(size=4) if use_markers else None,
            fill="tozeroy",
            fillcolor="rgba(6, 182, 212, 0.15)",
            yaxis="y1",
            hovertemplate="Step %{x}<br>Analytical KV: %{y:.2f} MB<extra></extra>",
        )
    )

    # 2. CUDA Allocated Memory (MB)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=plot_df["cuda_allocated_mb"],
            name="CUDA Allocated (MB)",
            mode="lines",
            line=dict(color="#3b82f6", width=2),
            yaxis="y2",
            hovertemplate="Step %{x}<br>CUDA Allocated: %{y:.1f} MB<extra></extra>",
        )
    )

    # 3. CUDA Reserved Arena (MB)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=plot_df["cuda_reserved_mb"],
            name="CUDA Reserved Pool (MB)",
            mode="lines",
            line=dict(color="#a855f7", width=2, dash="dash"),
            yaxis="y2",
            hovertemplate="Step %{x}<br>CUDA Reserved: %{y:.1f} MB<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="VRAM Allocation Dynamics: KV-Cache vs. CUDA Allocator",
            font=dict(size=13, color="#f8fafc"),
            x=0.01,
            xanchor="left",
            y=0.97,
        ),
        paper_bgcolor="#111726",
        plot_bgcolor="#111726",
        height=340,
        margin=dict(l=65, r=75, t=45, b=65),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color="#cbd5e1", size=10.5),
        ),
        xaxis=dict(
            title="Decoding Step (t)",
            color="#94a3b8",
            gridcolor="#1e293b",
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text="Analytical KV Cache (MB)", font=dict(color="#06b6d4")),
            tickfont=dict(color="#06b6d4"),
            gridcolor="#1e293b",
            automargin=True,
        ),
        yaxis2=dict(
            title=dict(text="Total GPU VRAM (MB)", font=dict(color="#a855f7")),
            tickfont=dict(color="#a855f7"),
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            automargin=True,
        ),
    )
    return fig


def build_latency_entropy_chart(df: pd.DataFrame) -> go.Figure:
    """
    Builds the Step Latency & Token Shannon Entropy dual plot.
    Legend placed at bottom to prevent title collision.
    """
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(
            text="Awaiting latency and entropy trajectory...",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#64748b"),
        )
        fig.update_layout(
            paper_bgcolor="#111726",
            plot_bgcolor="#111726",
            height=340,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    # Downsample large datasets (> 200 points) to maintain 60 FPS rendering
    if len(df) > 200:
        stride = max(1, len(df) // 200)
        sampled_indices = list(range(0, len(df), stride))
        if (len(df) - 1) not in sampled_indices:
            sampled_indices.append(len(df) - 1)
        plot_df = df.iloc[sampled_indices]
    else:
        plot_df = df

    steps = plot_df["step"]
    use_markers = len(plot_df) <= 60
    scatter_mode = "lines+markers" if use_markers else "lines"

    # Entropy trace
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=plot_df["entropy"],
            name="Shannon Entropy H(Pt) [bits]",
            mode=scatter_mode,
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=5, color="#f59e0b") if use_markers else None,
            yaxis="y1",
            hovertemplate="Step %{x}<br>Entropy: %{y:.3f} bits<extra></extra>",
        )
    )

    # Latency trace (highlighting TTFT) - vectorized color computation
    bar_colors = np.where(plot_df["is_prefill"] == 1, "#f43f5e", "#38bdf8").tolist()
    fig.add_trace(
        go.Bar(
            x=steps,
            y=plot_df["step_latency_ms"],
            name="Latency (ms)",
            marker=dict(color=bar_colors),
            opacity=0.6,
            yaxis="y2",
            hovertemplate="Step %{x}<br>Latency: %{y:.1f} ms<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Token Entropy Trajectory & Inter-Token Latency Profile",
            font=dict(size=13, color="#f8fafc"),
            x=0.01,
            xanchor="left",
            y=0.97,
        ),
        paper_bgcolor="#111726",
        plot_bgcolor="#111726",
        height=340,
        margin=dict(l=65, r=75, t=45, b=65),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color="#cbd5e1", size=10.5),
        ),
        xaxis=dict(
            title="Decoding Step (t)",
            color="#94a3b8",
            gridcolor="#1e293b",
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text="Shannon Entropy (bits)", font=dict(color="#f59e0b")),
            tickfont=dict(color="#f59e0b"),
            gridcolor="#1e293b",
            automargin=True,
        ),
        yaxis2=dict(
            title=dict(text="Step Latency (ms)", font=dict(color="#38bdf8")),
            tickfont=dict(color="#38bdf8"),
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            automargin=True,
        ),
    )
    return fig


def build_attention_scaling_figure(context_len: int = 4096) -> go.Figure:
    """
    Renders analytical KV-cache memory scaling across different attention paradigms
    (MHA, GQA-4, GQA-8, MQA, MLA) up to 32k context length.
    Legend placed at bottom to prevent title collision.
    """
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    data: Dict[str, List[float]] = {
        "MHA (32 heads)": [],
        "GQA-4 (8 KV heads)": [],
        "GQA-8 (4 KV heads)": [],
        "MQA (1 KV head)": [],
        "MLA (DeepSeek Latent)": [],
    }

    for s in seq_lengths:
        res = compare_attention_architectures(
            seq_len=s,
            num_layers=32,
            num_attention_heads=32,
            head_dim=128,
            batch_size=1,
            dtype_bytes=2,
        )
        data["MHA (32 heads)"].append(res["MHA"])
        data["GQA-4 (8 KV heads)"].append(res["GQA_4"])
        data["GQA-8 (4 KV heads)"].append(res["GQA_8"])
        data["MQA (1 KV head)"].append(res["MQA"])
        data["MLA (DeepSeek Latent)"].append(res["MLA_DeepSeek"])

    colors = {
        "MHA (32 heads)": "#f43f5e",
        "GQA-4 (8 KV heads)": "#38bdf8",
        "GQA-8 (4 KV heads)": "#10b981",
        "MQA (1 KV head)": "#a855f7",
        "MLA (DeepSeek Latent)": "#f59e0b",
    }

    fig = go.Figure()
    for name, values in data.items():
        fig.add_trace(
            go.Scatter(
                x=[f"{s//1024}k" if s >= 1024 else f"{s}" for s in seq_lengths],
                y=values,
                name=name,
                mode="lines+markers",
                line=dict(color=colors[name], width=2.5),
                marker=dict(size=6),
                hovertemplate=f"{name}: %{{y:.1f}} MB<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="KV-Cache Memory Wall: Architecture Scaling (32 Layers)",
            font=dict(size=13, color="#f8fafc"),
            x=0.01,
            xanchor="left",
            y=0.97,
        ),
        paper_bgcolor="#111726",
        plot_bgcolor="#111726",
        height=340,
        margin=dict(l=65, r=40, t=45, b=65),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color="#cbd5e1", size=10),
        ),
        xaxis=dict(
            title="Sequence Length (tokens)",
            color="#94a3b8",
            gridcolor="#1e293b",
            automargin=True,
        ),
        yaxis=dict(
            title="KV-Cache Size (MB)",
            color="#94a3b8",
            gridcolor="#1e293b",
            automargin=True,
        ),
    )
    return fig


def build_confidence_margin_chart(df: pd.DataFrame) -> go.Figure:
    """
    Renders the Candidate Confidence Trajectory & Probability Margin chart:
    - Top-1 Confidence: p(1) (winning token probability)
    - Runner-Up Margin: delta_p = p(1) - p(2)
    - Critical Decision Threshold line at delta_p = 0.20
    """
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(
            text="Awaiting probability margin telemetry...",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#64748b"),
        )
        fig.update_layout(
            paper_bgcolor="#111726",
            plot_bgcolor="#111726",
            height=340,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    # Downsample large datasets (> 250 points) to maintain 60 FPS rendering
    if len(df) > 250:
        stride = max(1, len(df) // 250)
        sampled_indices = list(range(0, len(df), stride))
        if (len(df) - 1) not in sampled_indices:
            sampled_indices.append(len(df) - 1)
        plot_df = df.iloc[sampled_indices]
    else:
        plot_df = df

    steps = plot_df["step"]
    use_markers = len(plot_df) <= 60
    scatter_mode = "lines+markers" if use_markers else "lines"

    # 1. Top-1 Candidate Confidence
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=plot_df["top1_confidence"],
            name="Top-1 Confidence p(1)",
            mode=scatter_mode,
            line=dict(color="#10b981", width=2),
            marker=dict(size=4) if use_markers else None,
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.08)",
            hovertemplate="Step %{x}<br>Top-1 Prob: %{y:.1%}<extra></extra>",
        )
    )

    # 2. Probability Margin (p1 - p2)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=plot_df["margin_p1_p2"],
            name="Margin Δp = p(1) - p(2)",
            mode=scatter_mode,
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=4) if use_markers else None,
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.15)",
            hovertemplate="Step %{x}<br>Candidate Margin: %{y:.3f}<extra></extra>",
        )
    )

    # 3. Critical Branching Ambiguity Threshold line
    fig.add_hline(
        y=0.20,
        line_dash="dash",
        line_color="#f43f5e",
        line_width=1.5,
        annotation_text="Ambiguity Threshold (Δp < 0.2)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#f43f5e"),
    )

    fig.update_layout(
        title=dict(
            text="Candidate Competition: Top-1 Confidence vs. Runner-Up Margin Δp",
            font=dict(size=13, color="#f8fafc"),
            x=0.01,
            xanchor="left",
            y=0.97,
        ),
        paper_bgcolor="#111726",
        plot_bgcolor="#111726",
        height=340,
        margin=dict(l=65, r=40, t=45, b=65),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color="#cbd5e1", size=10.5),
        ),
        xaxis=dict(
            title="Decoding Step (t)",
            color="#94a3b8",
            gridcolor="#1e293b",
            automargin=True,
        ),
        yaxis=dict(
            title="Probability / Margin",
            range=[0, 1.05],
            tickformat=".0%",
            color="#94a3b8",
            gridcolor="#1e293b",
            automargin=True,
        ),
    )
    return fig


def build_latency_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Renders the Inter-Token Latency (ITL) Distribution & Tail Jitter chart.
    Analyzes generation steps (excluding step 0 TTFT) and displays percentiles (p50, p95, p99).
    """
    fig = go.Figure()
    if df.empty or len(df) < 2:
        fig.add_annotation(
            text="Awaiting multiple tokens for latency distribution...",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#64748b"),
        )
        fig.update_layout(
            paper_bgcolor="#111726",
            plot_bgcolor="#111726",
            height=340,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    # Filter out pre-fill (TTFT)
    gen_df = df[df["is_prefill"] == 0] if (df["is_prefill"] == 0).any() else df[1:]
    if gen_df.empty:
        gen_df = df

    lats = gen_df["step_latency_ms"]
    p50 = float(lats.quantile(0.50))
    p95 = float(lats.quantile(0.95))
    p99 = float(lats.quantile(0.99))
    mean_lat = float(lats.mean())
    std_lat = float(lats.std()) if len(lats) > 1 else 0.0

    fig.add_trace(
        go.Histogram(
            x=lats,
            nbinsx=max(8, min(25, len(lats) // 2)),
            marker=dict(
                color="#38bdf8",
                line=dict(color="rgba(255,255,255,0.2)", width=1),
            ),
            opacity=0.75,
            name="Token Frequency",
            hovertemplate="Latency Bucket: %{x:.1f} ms<br>Count: %{y} tokens<extra></extra>",
        )
    )

    fig.add_vline(
        x=p50,
        line_dash="dash",
        line_color="#10b981",
        line_width=2,
        annotation_text=f"p50: {p50:.1f}ms",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#10b981"),
    )
    fig.add_vline(
        x=p95,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=2,
        annotation_text=f"p95: {p95:.1f}ms",
        annotation_position="top",
        annotation_font=dict(size=10, color="#f59e0b"),
    )
    if p99 > p95:
        fig.add_vline(
            x=p99,
            line_dash="dash",
            line_color="#f43f5e",
            line_width=2,
            annotation_text=f"p99: {p99:.1f}ms",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#f43f5e"),
        )

    fig.update_layout(
        title=dict(
            text=f"ITL Distribution & Tail Jitter (Mean: {mean_lat:.1f} ms ± {std_lat:.1f} ms)",
            font=dict(size=13, color="#f8fafc"),
            x=0.01,
            xanchor="left",
            y=0.97,
        ),
        paper_bgcolor="#111726",
        plot_bgcolor="#111726",
        height=340,
        margin=dict(l=65, r=45, t=45, b=50),
        xaxis=dict(
            title="Step Latency (ms)",
            color="#94a3b8",
            gridcolor="#1e293b",
            automargin=True,
        ),
        yaxis=dict(
            title="Frequency (Tokens)",
            color="#94a3b8",
            gridcolor="#1e293b",
            automargin=True,
        ),
        showlegend=False,
    )
    return fig


def build_temperature_sensitivity_chart(
    step_record: Optional[StepTelemetry],
    current_temp: float = 0.7,
) -> go.Figure:
    """
    Simulates Softmax thermodynamic scaling over temperature T in [0.1, 2.0]
    for the selected decoding step's candidate logits:
    P_i(T) = exp((z_i - max(z)) / T) / sum_j exp((z_j - max(z)) / T)
    H(T) = -sum P_i(T) * log2(P_i(T))
    """
    fig = go.Figure()
    if step_record is None or not step_record.top_candidates:
        fig.add_annotation(
            text="Click any token in the Entropy Canvas to simulate temperature sensitivity.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=13, color="#64748b"),
        )
        fig.update_layout(
            paper_bgcolor="#111726",
            plot_bgcolor="#111726",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    candidates = step_record.top_candidates
    logits = [float(c["logit"]) for c in candidates]
    tokens = [c["token"] for c in candidates]
    max_logit = max(logits)

    temps = [round(0.1 + i * 0.05, 2) for i in range(39)]
    candidate_probs: Dict[int, List[float]] = {i: [] for i in range(len(candidates))}
    entropies: List[float] = []

    for t in temps:
        exp_z = [math.exp(max(-50.0, min(50.0, (z - max_logit) / t))) for z in logits]
        denom = sum(exp_z)
        probs = [ez / denom for ez in exp_z]
        for i, p in enumerate(probs):
            candidate_probs[i].append(p)
        ent = -sum(p * math.log2(max(1e-12, p)) for p in probs)
        entropies.append(ent)

    palette = ["#10b981", "#38bdf8", "#a855f7", "#f59e0b", "#94a3b8"]
    for i, tok in enumerate(tokens[:3]):
        fig.add_trace(
            go.Scatter(
                x=temps,
                y=candidate_probs[i],
                name=f"'{tok.strip()}' Prob",
                mode="lines",
                line=dict(color=palette[i % len(palette)], width=2.2),
                yaxis="y1",
                hovertemplate=f"T=%{{x}}<br>Prob '{tok}': %{{y:.1%}}<extra></extra>",
            )
        )

    # Entropy trace on secondary axis
    fig.add_trace(
        go.Scatter(
            x=temps,
            y=entropies,
            name="Entropy H(T) [bits]",
            mode="lines",
            line=dict(color="#f59e0b", width=2, dash="dot"),
            yaxis="y2",
            hovertemplate="T=%{x}<br>Simulated Entropy: %{y:.2f} bits<extra></extra>",
        )
    )

    # Vertical line at current temperature
    fig.add_vline(
        x=current_temp,
        line_dash="dash",
        line_color="#e2e8f0",
        line_width=1.5,
        annotation_text=f"Active T={current_temp}",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#e2e8f0"),
    )

    tok_clean = repr(step_record.token_str)[1:-1] if step_record.token_str else f"#{step_record.step}"
    fig.update_layout(
        title=dict(
            text=f"Step #{step_record.step} ('{tok_clean}') Softmax Thermodynamic Scaling (H={step_record.entropy:.2f}b, Δp={step_record.margin_p1_p2:.2f})",
            font=dict(size=13, color="#e2e8f0"),
            x=0.01,
            xanchor="left",
            y=0.97,
        ),
        paper_bgcolor="#111726",
        plot_bgcolor="#111726",
        height=340,
        margin=dict(l=65, r=70, t=45, b=65),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color="#cbd5e1", size=10),
        ),
        xaxis=dict(
            title="Softmax Temperature (T)",
            color="#94a3b8",
            gridcolor="#1e293b",
            automargin=True,
        ),
        yaxis=dict(
            title="Candidate Probability",
            range=[0, 1.05],
            tickformat=".0%",
            color="#94a3b8",
            gridcolor="#1e293b",
            automargin=True,
        ),
        yaxis2=dict(
            title=dict(text="Simulated Entropy (bits)", font=dict(color="#f59e0b")),
            tickfont=dict(color="#f59e0b"),
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            automargin=True,
        ),
    )
    return fig


def build_telemetry_table(records: List[StepTelemetry]) -> Any:
    """
    Renders a compact, smoothly scrollable step-by-step telemetry audit table.
    Displays up to 200 decoding steps in chronological sequence with sticky header.
    """
    if not records:
        return html.Div(
            "No telemetry records collected yet.",
            style={"color": "#64748b", "fontStyle": "italic", "padding": "1rem"},
        )

    rows = []
    display_records = records[-80:] if len(records) > 80 else records
    for r in display_records:
        token_clean = r.token_str.replace(" ", "␣") if r.token_str == " " else r.token_str
        rows.append(
            html.Tr(
                [
                    html.Td(f"#{r.step}", style={"fontWeight": "bold", "color": "#94a3b8"}),
                    html.Td("Prefill" if r.is_prefill else "Decode", style={"color": "#f43f5e" if r.is_prefill else "#38bdf8"}),
                    html.Td(html.Code(token_clean, style={"color": "#f8fafc", "backgroundColor": "rgba(255,255,255,0.06)", "padding": "2px 4px", "borderRadius": "3px"})),
                    html.Td(f"{r.entropy:.3f}", style={"color": "#f59e0b"}),
                    html.Td(f"{r.norm_entropy:.1%}", style={"color": "#e2e8f0"}),
                    html.Td(f"{r.top1_confidence:.1%}", style={"color": "#10b981"}),
                    html.Td(f"{r.margin_p1_p2:.3f}", style={"color": "#06b6d4"}),
                    html.Td(f"{r.kv_cache_mb_analytical:.2f}", style={"color": "#cbd5e1"}),
                    html.Td(f"{r.step_latency_ms:.1f}", style={"color": "#38bdf8"}),
                ]
            )
        )

    table_header = html.Thead(
        html.Tr(
            [
                html.Th("Step", style={"color": "#94a3b8"}),
                html.Th("Phase", style={"color": "#94a3b8"}),
                html.Th("Token", style={"color": "#94a3b8"}),
                html.Th("Entropy H", style={"color": "#94a3b8"}),
                html.Th("Norm H", style={"color": "#94a3b8"}),
                html.Th("Top-1 p", style={"color": "#94a3b8"}),
                html.Th("Margin Δp", style={"color": "#94a3b8"}),
                html.Th("KV (MB)", style={"color": "#94a3b8"}),
                html.Th("ITL (ms)", style={"color": "#94a3b8"}),
            ]
        ),
        style={"backgroundColor": "#0d131f", "position": "sticky", "top": "0", "zIndex": "10"},
    )

    return html.Div(
        dbc.Table(
            [table_header, html.Tbody(rows)],
            responsive=True,
            striped=True,
            hover=True,
            size="sm",
            style={
                "color": "#f8fafc",
                "fontSize": "0.82rem",
                "fontFamily": "JetBrains Mono, monospace",
                "marginBottom": "0",
            },
        ),
        className="telemetry-table-scroll",
        style={
            "maxHeight": "360px",
            "overflowY": "auto",
            "overflowX": "auto",
            "borderRadius": "6px",
            "border": "1px solid #1e293b",
            "scrollbarWidth": "thin",
        },
    )
