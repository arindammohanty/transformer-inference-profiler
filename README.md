# Transformer Inference Profiler & KV-Cache/VRAM Inspector

[![Tests](https://img.shields.io/badge/Tests-29%2F29%20Passed-emerald)](tests)
[![Python](https://img.shields.io/badge/Python-3.13-blue)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13%2B-orange)](core/profiler.py)
[![LaTeX](https://img.shields.io/badge/LaTeX-pdflatex%20%7C%20TikZ-red)](report/main.pdf)
[![LibreOffice](https://img.shields.io/badge/LibreOffice-MCP%20%7C%20ODP-purple)](slides/presentation.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, high-resolution diagnostic telemetry suite, mechanistic interpretability dashboard, and memory profiler for autoregressive Large Language Model (LLM) inference.

---

## 🚨 Memory Protection & Emergency Stop System

To protect consumer laptops and workstations from out-of-memory (OOM) freezing or catastrophic swap thrashing (e.g. running 30B/32B models on 16GB systems):

1. **Real-Time Host RAM Telemetry Badge**:
   - Continuously displays physical memory utilization (`Host RAM: X.X / Y.Y GB (Z%)`) in the navbar.
2. **Model RAM Estimation & Safety Badges**:
   - Every Ollama model in the dropdown is automatically introspected and tagged:
     - 🟢 **SAFE**: `< 8GB RAM required` (e.g., `llama3:latest`, `gemma:7b`).
     - 🟡 **HEAVY**: `8GB - 15GB RAM required` (e.g., `gemma4:12b`, `coder:latest`).
     - 🔴 **DANGER**: `> 15GB RAM required` (e.g., `qwen3-coder:30b`, `qwen2.5vl:32b`).
3. **Interactive Safety Lock**:
   - If a selected model exceeds available host RAM, the dashboard locks the **"⚡ Run Profiler"** button and displays a high-visibility hazard warning.
   - Execution is strictly blocked unless the user explicitly checks the override acknowledgment: *"⚠️ Override Safety Lock: I acknowledge high memory consumption..."*.
4. **Auto-Eviction of Prior Models**:
   - Profiling a new model automatically evicts any previously active models in Ollama via `keep_alive: 0`, preventing cumulative memory stacking.
5. **Multi-Tier Emergency Stop**:
   - **In UI**: Click the **"🚨 Emergency Evict RAM"** button at any time to abort inference, unload all Ollama models, purge buffers, and run garbage collection.
   - **In Terminal**: Run `./run_dashboard.sh --emergency-stop` or `./emergency_stop.sh` to terminate any rogue processes, evict all models, and restore memory to idle.
   - **On Exit**: Pressing `Ctrl+C` in the dashboard terminal automatically traps the signal and cleans up all memory before exit.

---

## 🚀 Key Highlights & Capabilities

1. **Information-Theoretic Telemetry**:
   - Computes token-level **Shannon entropy** $H(P_t) = -\sum p_i \log_2(p_i)$ across vocabulary logits.
   - Calculates **Normalized Shannon Entropy** $H_{\text{norm}}(P_t) = \frac{H(P_t)}{\log_2(|V|)}$ and Top-1 vs Top-2 candidate probability margins $\Delta p_t = p_{(1)} - p_{(2)}$.
2. **Analytical KV-Cache Scaling Engine**:
   - Exact byte calculations for Key-Value tensor footprints:
     $$\mathcal{M}_{\text{KV}} = 2 \cdot B \cdot S \cdot L \cdot N_{\text{kv}} \cdot d_{\text{head}} \cdot b_{\text{dtype}}$$
   - Comparative architectural scaling across Multi-Head Attention (MHA), Grouped-Query Attention (GQA-4, GQA-8), Multi-Query Attention (MQA), and DeepSeek Multi-Head Latent Attention (MLA).
3. **Hybrid CUDA Telemetry Engine**:
   - Binds to live `torch.cuda` memory counters (`memory_allocated`, `memory_reserved`).
   - Includes a deterministic PyTorch Caching Allocator simulator when executing on CPU, modeling 512B/2MB block rounding, 20MB arena expansions, and internal fragmentation ratio $\Phi_{\text{frag}}$.
4. **Dynamic Dual-Engine Runtime**:
   - Direct PyTorch causal decoding loop under `torch.inference_mode()`.
   - Local **Ollama server integration** (`http://localhost:11434`) with automatic model discovery (`llama3:latest`, `qwen2.5-coder:14b`, `gemma:7b`, etc.) and GGUF architectural parameter introspection (`/api/show`).
   - Synthetic long-context scale-benchmarking presets (`Llama-3-8B`, `Qwen-2.5-14B`, `Mistral-7B`, `DeepSeek-V3`).
5. **Real-Time Interactive Diagnostic Dashboard (Plotly Dash)**:
   - **Token Entropy Canvas**: Live generated tokens rendered as clickable pills with continuous background color interpolation from Emerald (certainty) to Crimson (uncertainty/branching).
   - **Candidate Competition View**: Horizontal bar charts displaying Top-5 probabilities and logit margins for the inspected token step.
   - **Dual-Axis VRAM Telemetry**: Compares analytical linear KV growth against the stepwise reserved pool of the CUDA caching allocator.
   - **Latency Waterfall & Jitter**: Pre-fill Time-to-First-Token (TTFT) and steady-state Inter-Token Latency (ITL) dynamics.
6. **Publication-Grade LaTeX Technical Report**:
   - Multi-page academic paper typeset natively with TikZ dataflow architecture diagrams, empirical tables, and formal derivations.
7. **Model Context Protocol (MCP) LibreOffice Impress Automation**:
   - OpenDocument Presentation (`.odp`) XML compilation engine with native SMIL transitions (`fade`, `push`, `wipe`) and sequential build animations (`<anim:seq>`, `<anim:par>`), exported directly to PDF via headless LibreOffice.

---

## 📂 Repository Structure

```
.
├── Makefile                     # Build automation (test, dashboard, report, slides)
├── README.md                    # Comprehensive documentation and runbook
├── run_dashboard.sh             # Launch Plotly Dash interactive profiler
├── compile_report.sh            # Compile LaTeX technical report to PDF
├── build_slides.sh              # Generate animated LibreOffice deck via MCP
├── core/
│   ├── __init__.py
│   ├── metrics.py               # Shannon entropy, margins, GQA formulas, ITL stats
│   ├── profiler.py              # Custom decoding loop, PyTorch hooks, hybrid CUDA simulator
│   └── ollama_client.py         # Ollama dynamic discovery and streaming telemetry
├── storage/
│   ├── __init__.py
│   └── buffer.py                # In-memory thread-safe SQLite telemetry store & Pandas exporter
├── ui/
│   ├── __init__.py
│   ├── app.py                   # Plotly Dash application with streaming callbacks
│   ├── components.py            # Token canvas, candidate bar charts, dual-axis VRAM plots
│   └── assets/
│       └── custom.css           # Glassmorphic dark-mode styling
├── mcp_impress/
│   ├── __init__.py
│   ├── server.py                # Model Context Protocol (MCP) server & deck generator
│   ├── odp_generator.py         # OpenDocument XML synthesis engine with transitions/animations
│   └── uno_bridge.py            # Python-UNO headless LibreOffice bridge
├── report/
│   ├── main.tex                 # Publication-grade technical report
│   ├── references.bib           # Academic bibliography
│   ├── compile.sh               # LaTeX compilation script
│   └── main.pdf                 # Compiled 5-page PDF report
├── slides/
│   ├── presentation.odp         # 8-slide transition-animated presentation
│   └── presentation.pdf         # Exported PDF presentation
└── tests/
    ├── test_metrics.py          # Unit tests for entropy, margins, and KV scaling
    ├── test_buffer.py           # Unit tests for in-memory SQLite telemetry buffer
    ├── test_profiler.py         # Unit tests for profiler and CUDA simulation
    └── test_odp_generator.py    # Unit tests for ODP generator and MCP JSON-RPC protocol
```

---

## 🛠️ Prerequisites & Installation

### System Dependencies
- **Linux** (tested on Ubuntu / Debian / Fedora)
- **Python**: 3.10+ (tested on Python 3.13)
- **LaTeX**: `pdflatex` and `bibtex` (`texlive-latex-base`, `texlive-latex-extra`, `texlive-science`)
- **LibreOffice**: `libreoffice-impress` (tested on LibreOffice 25.2)

### Python Dependencies
Install required packages via pip:
```bash
pip install torch transformers dash dash-bootstrap-components plotly pandas requests pytest
```

---

## ⚡ Execution Guide

### 1. Launching the Interactive Profiler Dashboard
Execute the launcher script:
```bash
./run_dashboard.sh
```
Or run directly via Python:
```bash
python3 ui/app.py
```
Open **[http://127.0.0.1:8050](http://127.0.0.1:8050)** in any modern web browser.
- Select your **Engine Mode**:
  - `🦙 Ollama Local Models`: dynamically populates with models currently pulled into your local Ollama daemon (`llama3:latest`, `qwen2.5-coder:14b`, etc.).
  - `🔥 PyTorch Local CausalLM`: runs local Hugging Face checkpoints (e.g., `gpt2`) with raw PyTorch logit interception.
  - `📊 Synthetic Scale Benchmark`: simulates long-context memory profiles for `Llama-3-8B`, `Qwen-2.5-14B`, `Mistral-7B`, or `DeepSeek-V3-MLA`.
- Click **"⚡ Run Profiler"** to stream tokens in real-time.
- **Click any token chip** in the Token Entropy Canvas to immediately inspect its candidate competition probabilities and logit margins.

### 2. Compiling the Technical Report
To recompile the LaTeX technical report with full TikZ diagrams and citations:
```bash
./compile_report.sh
```
The compiled publication-grade PDF is available at `report/main.pdf`.

### 3. Generating the Presentation Slides (MCP Tool Integration)
To generate the 8-slide technical deck with active slide transitions (`fade`, `push`, `wipe`) and object build animations:
```bash
./build_slides.sh
```
The slide deck is saved as both:
- OpenDocument Presentation: `slides/presentation.odp`
- Exported PDF: `slides/presentation.pdf`

To run the MCP server in JSON-RPC stdio mode for Model Context Protocol clients:
```bash
python3 -m mcp_impress.server
```

### 4. Running the Test Suite
Run all unit tests:
```bash
make test
# Or directly:
python3 -m pytest tests/ -v
```

---

## 🔬 Mathematical Formulas Reference

| Metric | Mathematical Derivation | Description |
|---|---|---|
| **Shannon Entropy** | $H(P_t) = -\sum_{i=1}^{\|V\|} p_i \log_2(p_i)$ | Uncertainty over next-token distribution |
| **Normalized Entropy** | $H_{\text{norm}}(P_t) = \frac{H(P_t)}{\log_2(\|V\|)}$ | Scale-independent uncertainty in $[0, 1]$ |
| **Probability Margin** | $\Delta p_t = p_{(1)} - p_{(2)}$ | Confidence separation between top-2 candidates |
| **Logit Margin** | $\Delta z_t = z_{(1)} - z_{(2)}$ | Unnormalized logit gap between top-2 candidates |
| **Analytical KV-Cache** | $2 \cdot B \cdot S \cdot L \cdot N_{\text{kv}} \cdot d \cdot b$ | Exact physical memory occupied by keys and values |
| **Allocator Fragmentation** | $\Phi_{\text{frag}} = 1 - \frac{M_{\text{allocated}}}{M_{\text{reserved}}}$ | Fraction of mapped VRAM unusable due to fragmentation |
| **Memory Bandwidth Demand** | $B_{\text{mem}} = \frac{2P + \mathcal{M}_{\text{KV}}}{\Delta t}$ | Minimum HBM bandwidth required per generation step |

---

## 📜 License
MIT License. Developed for advanced LLM telemetry, mechanistic interpretability, and hardware memory dynamics.
