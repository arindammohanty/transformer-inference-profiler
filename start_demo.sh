#!/usr/bin/env bash
# ==============================================================================
# ⚡ REAL-TIME TRANSFORMER INFERENCE PROFILER & KV-CACHE / VRAM INSPECTOR
# Live Demonstration Startup & Orchestration Script (start_demo.sh)
#
# Presenter: Arindam Mohanty (250301120059)
# Guided By: Assistant Prof. Sanjib Kumar Naik
# Centurion University of Technology and Management (CUTM)
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$PROJECT_DIR"

# Default Configuration
PORT="${PORT:-8050}"
HOST="127.0.0.1"
URL="http://${HOST}:${PORT}"
RAM_LOG_DIR="/dev/shm/davp_profiler"
OPEN_BROWSER=true
RUN_TESTS=false
REGEN_PDF=false
STARTED_OLLAMA_PID=""
DASH_PID=""

# ANSI Colors for Presenter Console
C_RESET="\033[0m"
C_BOLD="\033[1m"
C_CYAN="\033[1;36m"
C_GREEN="\033[1;32m"
C_AMBER="\033[1;33m"
C_RED="\033[1;31m"
C_SLATE="\033[0;37m"

print_banner() {
    echo -e "${C_CYAN}==========================================================================${C_RESET}"
    echo -e "${C_BOLD}⚡ REAL-TIME TRANSFORMER INFERENCE PROFILER & KV-CACHE / VRAM INSPECTOR${C_RESET}"
    echo -e "${C_SLATE}   Mechanistic Logit Telemetry, Shannon Entropy & Hardware Memory Diagnostics${C_RESET}"
    echo -e "${C_CYAN}==========================================================================${C_RESET}"
}

show_help() {
    print_banner
    echo -e "${C_BOLD}Usage:${C_RESET} ./start_demo.sh [OPTIONS]"
    echo ""
    echo -e "${C_BOLD}Options:${C_RESET}"
    echo "  (no args)           Launch full live demonstration & open browser automatically"
    echo "  --no-browser        Launch server without auto-opening a browser window"
    echo "  --test              Run the 29-unit-test pytest suite before launching"
    echo "  --pdf               Regenerate Arindam_Mohanty_PythonProject.pdf before launching"
    echo "  --port <PORT>       Specify custom port (default: 8050)"
    echo "  --stop, -e          Emergency stop: kill server, evict all Ollama models & free RAM"
    echo "  --help, -h          Show this help message"
    echo ""
}

# ==============================================================================
# Emergency Stop & Memory Recovery Routine
# ==============================================================================
emergency_stop() {
    echo ""
    echo -e "${C_RED}==========================================================================${C_RESET}"
    echo -e "${C_RED}🚨 EXECUTING EMERGENCY STOP & SYSTEM RAM / VRAM RECOVERY${C_RESET}"
    echo -e "${C_RED}==========================================================================${C_RESET}"

    echo -e "${C_AMBER}[1/4] Terminating profiler dashboard processes on port ${PORT}...${C_RESET}"
    if [ -n "$DASH_PID" ] && kill -0 "$DASH_PID" 2>/dev/null; then
        kill -9 "$DASH_PID" 2>/dev/null || true
    fi
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    ps aux | grep -E "ui/app\.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true

    echo -e "${C_AMBER}[2/4] Evicting ALL loaded LLM weights from Ollama RAM/VRAM (keep_alive: 0)...${C_RESET}"
    ACTIVE_MODELS=$(curl -s --max-time 2 http://127.0.0.1:11434/api/ps 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4 || true)
    if [ -n "$ACTIVE_MODELS" ]; then
        for model in $ACTIVE_MODELS; do
            echo -e "      -> Unloading active model: ${C_BOLD}${model}${C_RESET}"
            curl -s --max-time 3 http://127.0.0.1:11434/api/generate -d "{\"model\": \"$model\", \"keep_alive\": 0}" >/dev/null 2>&1 || true
        done
    else
        echo "      -> Zero active models currently held in Ollama memory."
    fi

    if [ -n "$STARTED_OLLAMA_PID" ] && kill -0 "$STARTED_OLLAMA_PID" 2>/dev/null; then
        echo "      -> Stopping background Ollama daemon started by demo (PID: $STARTED_OLLAMA_PID)..."
        kill "$STARTED_OLLAMA_PID" 2>/dev/null || true
    fi

    echo -e "${C_AMBER}[3/4] Flushing in-memory SQLite (:memory:) buffer & /dev/shm tmpfs logs...${C_RESET}"
    PYTHONDONTWRITEBYTECODE=1 python3 -c "from storage.buffer import TelemetryBuffer; TelemetryBuffer().clear(); import gc; gc.collect()" 2>/dev/null || true
    rm -rf "$RAM_LOG_DIR" 2>/dev/null || true

    echo -e "${C_AMBER}[4/4] Host Memory Status After Recovery:${C_RESET}"
    free -h | awk 'NR==1 {print "      " $0} NR==2 {print "      " $0}'

    echo -e "${C_GREEN}==========================================================================${C_RESET}"
    echo -e "${C_GREEN}✅ RECOVERY COMPLETE: System RAM & GPU VRAM restored to clean idle state.${C_RESET}"
    echo -e "${C_GREEN}==========================================================================${C_RESET}"
}

# Parse CLI Arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stop|--emergency-stop|-e|stop)
            emergency_stop
            exit 0
            ;;
        --no-browser)
            OPEN_BROWSER=false
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --pdf)
            REGEN_PDF=true
            shift
            ;;
        --port)
            PORT="$2"
            URL="http://${HOST}:${PORT}"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo -e "${C_RED}Unknown option: $1${C_RESET}"
            show_help
            exit 1
            ;;
    esac
done

# Trap Ctrl+C (SIGINT) and SIGTERM for clean demo exit
trap_cleanup() {
    echo ""
    echo -e "${C_AMBER}Demo session concluded. Running automatic memory & tmpfs cleanup...${C_RESET}"
    emergency_stop
    exit 0
}
trap trap_cleanup SIGINT SIGTERM

print_banner

# ==============================================================================
# Step 1: Configure Zero-Disk-Write RAM Environment
# ==============================================================================
echo -e "${C_CYAN}[Step 1/5] Initializing Zero-Disk-Write RAM Workspace (/dev/shm)...${C_RESET}"
mkdir -p "$RAM_LOG_DIR" 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
echo -e "           -> ${C_GREEN}PYTHONDONTWRITEBYTECODE=1${C_RESET} (Zero .pyc disk writes)"
echo -e "           -> ${C_GREEN}SQLite Engine: :memory:${C_RESET} (100% RAM-resident telemetry)"
echo -e "           -> ${C_GREEN}Tmpfs Log Dir: ${RAM_LOG_DIR}${C_RESET}"

# ==============================================================================
# Step 2: Verify Python Environment & Optional Pre-Flight Suites
# ==============================================================================
echo -e "${C_CYAN}[Step 2/5] Verifying Python Runtime & Core Telemetry Libraries...${C_RESET}"
PYTHON_INFO=$(python3 -c "
import sys, torch, dash, plotly, pandas, numpy, psutil
dev = f'CUDA ({torch.cuda.get_device_name(0)})' if torch.cuda.is_available() else 'CPU + Hybrid CUDA Allocator Simulator'
print(f'Python {sys.version.split()[0]} | PyTorch {torch.__version__} | Dash {dash.__version__} | Device: {dev}')
")
echo -e "           -> ${C_GREEN}${PYTHON_INFO}${C_RESET}"

if [ "$RUN_TESTS" = true ]; then
    echo -e "${C_CYAN}           -> Running 29-test verification suite (pytest)...${C_RESET}"
    pytest -q
    echo -e "           -> ${C_GREEN}All unit tests passed!${C_RESET}"
fi

if [ "$REGEN_PDF" = true ]; then
    echo -e "${C_CYAN}           -> Regenerating Academic Project Report PDF...${C_RESET}"
    python3 "$PROJECT_DIR/generate_report_pdf.py"
fi

# ==============================================================================
# Step 3: Check Ollama Daemon & Pre-Clear Stale Model Weights
# ==============================================================================
echo -e "${C_CYAN}[Step 3/5] Checking Local Ollama LLM Runtime & Host RAM Headroom...${C_RESET}"
if curl -s --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    MODEL_COUNT=$(curl -s http://127.0.0.1:11434/api/tags | grep -o '"name":' | wc -l)
    echo -e "           -> ${C_GREEN}Ollama Status: ONLINE (${MODEL_COUNT} local GGUF models available)${C_RESET}"
elif command -v ollama >/dev/null 2>&1; then
    echo -e "           -> ${C_AMBER}Ollama daemon not running. Starting 'ollama serve' in background...${C_RESET}"
    ollama serve >"${RAM_LOG_DIR}/ollama_daemon.log" 2>&1 &
    STARTED_OLLAMA_PID=$!
    sleep 2
    if curl -s --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        MODEL_COUNT=$(curl -s http://127.0.0.1:11434/api/tags | grep -o '"name":' | wc -l)
        echo -e "           -> ${C_GREEN}Ollama Status: ONLINE (${MODEL_COUNT} local GGUF models available)${C_RESET}"
    else
        echo -e "           -> ${C_AMBER}Ollama Status: OFFLINE (Synthetic & PyTorch engines remain 100% available)${C_RESET}"
    fi
else
    echo -e "           -> ${C_AMBER}Ollama Status: Not installed (Synthetic & PyTorch engines active)${C_RESET}"
fi

# Release any leftover loaded Ollama models before demo starts if RAM is tight
TOTAL_RAM=$(free -m | awk '/Mem:/ {print $2}')
AVAIL_RAM=$(free -m | awk '/Mem:/ {print $7}')
SWAP_TOTAL=$(free -m | awk '/Swap:/ {print $2}')
echo -e "           -> ${C_GREEN}Host Memory: ${AVAIL_RAM} MB Available / ${TOTAL_RAM} MB Total (Swap: ${SWAP_TOTAL} MB)${C_RESET}"

# ==============================================================================
# Step 4: Clear Port & Launch Live Plotly Dash Server
# ==============================================================================
echo -e "${C_CYAN}[Step 4/5] Starting Live Diagnostic Server on ${URL} ...${C_RESET}"
fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
ps aux | grep -E "ui/app\.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
sleep 0.5

python3 "$PROJECT_DIR/ui/app.py" >"${RAM_LOG_DIR}/dash_server.log" 2>&1 &
DASH_PID=$!

# Wait for HTTP 200 readiness
READY=false
for _ in {1..30}; do
    if curl -s -o /dev/null --max-time 1 "${URL}"; then
        READY=true
        break
    fi
    sleep 0.3
done

if [ "$READY" = false ]; then
    echo -e "${C_RED}❌ Server failed to bind to ${URL}. Check log output:${C_RESET}"
    cat "${RAM_LOG_DIR}/dash_server.log" || true
    exit 1
fi

echo -e "           -> ${C_GREEN}Server is LIVE and responding on ${URL} (PID: ${DASH_PID})${C_RESET}"

# ==============================================================================
# Step 5: Open Browser & Display Live Demonstration Walkthrough Cheat-Sheet
# ==============================================================================
echo -e "${C_CYAN}[Step 5/5] Launching Demonstration Interface...${C_RESET}"
if [ "$OPEN_BROWSER" = true ] && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${URL}" >/dev/null 2>&1 &
    echo -e "           -> ${C_GREEN}Opened ${URL} in default web browser.${C_RESET}"
else
    echo -e "           -> ${C_AMBER}Open ${C_BOLD}${URL}${C_RESET}${C_AMBER} in your browser to view the live dashboard.${C_RESET}"
fi

echo ""
echo -e "${C_CYAN}==========================================================================${C_RESET}"
echo -e "${C_GREEN}🎯 LIVE DEMONSTRATION GUIDE (PRESENTER CHEAT-SHEET)${C_RESET}"
echo -e "${C_CYAN}==========================================================================${C_RESET}"
echo -e "  ${C_BOLD}🌐 Dashboard URL:${C_RESET}   ${C_CYAN}${URL}${C_RESET}"
echo -e "  ${C_BOLD}📄 Project Report:${C_RESET}  ${PROJECT_DIR}/Arindam_Mohanty_PythonProject.pdf"
echo ""
echo -e "  ${C_BOLD}1. Demonstrate Real-Time Entropy & Branching (Synthetic Engine):${C_RESET}"
echo -e "     • Keep Engine Mode on ${C_CYAN}Synthetic Scale Benchmark${C_RESET} (${C_BOLD}Llama-3-8B${C_RESET})."
echo -e "     • Click preset button ${C_AMBER}[Adversarial Entropy & Branching]${C_RESET} -> Click ${C_GREEN}[⚡ Run Profiler]${C_RESET}."
echo -e "     • Watch the 6 KPI Cards, Token Entropy Canvas, and VRAM curves update at 60 FPS."
echo ""
echo -e "  ${C_BOLD}2. Demonstrate Dual-Mode Token Canvas & Mechanistic Logit Space:${C_RESET}"
echo -e "     • Toggle ${C_CYAN}Paragraph Flow${C_RESET} <-> ${C_CYAN}Pills${C_RESET} in the Token Entropy Canvas."
echo -e "     • Click any amber/red token chip to inspect its ${C_BOLD}Top-5 Candidate Logits${C_RESET} and"
echo -e "       ${C_BOLD}Softmax Thermodynamic Temperature Scaling${C_RESET} curve."
echo ""
echo -e "  ${C_BOLD}3. Demonstrate Proactive Host RAM Safety Lock & Live Ollama Models:${C_RESET}"
echo -e "     • Switch Engine Mode to ${C_CYAN}Ollama Local Models${C_RESET} and select ${C_RED}qwen3-coder:30b${C_RESET}"
echo -e "       to showcase the automatic OOM Hazard Lock banner."
echo -e "     • Select a lightweight model (e.g., ${C_GREEN}llama3.2:1b${C_RESET} or ${C_GREEN}qwen2.5:3b${C_RESET}) for live local LLM profiling."
echo -e "     • Click ${C_RED}[🚨 Emergency Evict RAM]${C_RESET} anytime to unload model weights in 1 click."
echo -e "${C_CYAN}==========================================================================${C_RESET}"
echo -e "${C_AMBER}Press Ctrl+C at any time to stop the server and automatically clean up RAM.${C_RESET}"
echo ""

# Stream live server log while waiting for user Ctrl+C
wait "$DASH_PID"
