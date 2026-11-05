"""
LibreOffice Impress Model Context Protocol (MCP) Server.
Exposes JSON-RPC 2.0 tool endpoints for programmatic slide deck creation,
shape animations, slide transitions, and automated headless PDF exporting.
"""

from __future__ import annotations
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from mcp_impress.odp_generator import ODPPresentationBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mcp_impress")

# Global active builder instance
_active_builder: Optional[ODPPresentationBuilder] = None


def get_builder() -> ODPPresentationBuilder:
    global _active_builder
    if _active_builder is None:
        _active_builder = ODPPresentationBuilder(
            title="Transformer Inference Profiler", theme="cyber_dark"
        )
    return _active_builder


# Tool Implementations
def tool_create_presentation(title: str, theme: str = "cyber_dark") -> Dict[str, Any]:
    """Initializes a new presentation with the specified title and theme."""
    global _active_builder
    _active_builder = ODPPresentationBuilder(title=title, theme=theme)
    return {
        "status": "success",
        "message": f"Created new presentation: '{title}' with theme '{theme}'",
        "slides_count": 0,
    }


def tool_add_slide(
    title: str,
    subtitle: str = "",
    bullets: Optional[List[str]] = None,
    callout_boxes: Optional[List[Dict[str, str]]] = None,
    transition_type: str = "fade",
    transition_speed: str = "fast",
    transition_direction: str = "from-right",
) -> Dict[str, Any]:
    """Appends a new slide with bullets, callouts, and transition configuration."""
    builder = get_builder()
    slide = builder.add_slide(
        title=title,
        subtitle=subtitle,
        bullets=bullets or [],
        callout_boxes=callout_boxes or [],
        transition_type=transition_type,
        transition_speed=transition_speed,
        transition_direction=transition_direction,
    )
    idx = len(builder.slides) - 1
    return {
        "status": "success",
        "slide_index": idx,
        "title": title,
        "total_slides": len(builder.slides),
    }


def tool_apply_animation(
    slide_index: int,
    shape_id: str,
    animation_type: str = "appear",
    trigger_event: str = "click",
    delay: float = 0.0,
) -> Dict[str, Any]:
    """Applies a sequential build animation to a shape on a specific slide."""
    builder = get_builder()
    builder.apply_animation(
        slide_index=slide_index,
        shape_id=shape_id,
        animation_type=animation_type,
        trigger_event=trigger_event,
        delay=delay,
    )
    return {
        "status": "success",
        "slide_index": slide_index,
        "shape_id": shape_id,
        "animation": animation_type,
    }


def tool_set_slide_transition(
    slide_index: int,
    transition_type: str = "push",
    speed: str = "fast",
    direction: str = "from-right",
) -> Dict[str, Any]:
    """Configures the slide transition type (fade, push, wipe) and speed."""
    builder = get_builder()
    builder.set_slide_transition(
        slide_index=slide_index,
        transition_type=transition_type,
        speed=speed,
        direction=direction,
    )
    return {
        "status": "success",
        "slide_index": slide_index,
        "transition": transition_type,
        "speed": speed,
    }


def tool_export_presentation(
    output_odp_path: str = "slides/presentation.odp",
    export_pdf: bool = True,
) -> Dict[str, Any]:
    """Packages the active presentation into an ODP archive and exports to PDF via LibreOffice."""
    builder = get_builder()
    abs_odp = os.path.abspath(output_odp_path)
    builder.save_odp(abs_odp)

    pdf_path = None
    if export_pdf:
        pdf_path = builder.export_to_pdf(abs_odp)

    return {
        "status": "success",
        "odp_path": abs_odp,
        "pdf_path": pdf_path,
        "total_slides": len(builder.slides),
    }


def build_full_technical_deck(output_dir: str = "slides") -> Dict[str, str]:
    """
    Constructs the comprehensive 8-slide technical presentation deck:
    Transformer Inference Profiler & KV-Cache/VRAM Inspector.
    """
    os.makedirs(output_dir, exist_ok=True)
    builder = ODPPresentationBuilder(
        title="Transformer Inference Profiler & KV-Cache/VRAM Inspector",
        theme="cyber_dark",
    )

    # Slide 1: Title & Abstract
    builder.add_slide(
        title="Transformer Inference Profiler & KV-Cache / VRAM Inspector",
        subtitle="Real-Time Telemetry, Entropy Dynamics, and Hardware Allocator Mechanics",
        bullets=[
            "Diagnostic Architecture: Real-time interception of autoregressive decoding steps without GPU-to-CPU stalls.",
            "Unified Observability: Bridges information-theoretic predictive uncertainty with physical hardware memory dynamics.",
            "Multi-Engine Integration: Seamless profiling across PyTorch native CausalLM and local Ollama GGUF endpoints.",
            "Interactive Dashboard: Plotly Dash interface featuring Token Entropy Canvas and Candidate Competition View.",
        ],
        callout_boxes=[
            {"title": "TELEMETRY METRICS", "value": "H(Pt) & KV", "desc": "Shannon entropy, logit margins, analytical KV bytes, and CUDA VRAM fragmentation."}
        ],
        transition_type="fade",
        transition_speed="fast",
    )

    # Slide 2: The GPU Memory Wall
    builder.add_slide(
        title="The Memory Wall in Autoregressive LLM Inference",
        subtitle="Compute-Bound Pre-Fill vs. Memory-Bandwidth-Bound Token Generation",
        bullets=[
            "Operational Asymmetry: Prompt pre-fill is compute-bound (GEMM), while sequential generation is strictly memory-bandwidth bound (GEMV).",
            "Hardware Saturation: Each generated token requires streaming the full parameter weights and expanding KV-cache activations through GPU SRAM.",
            "Memory Bandwidth Equation: Bandwidth demand scales as B_mem = (2*P + M_KV) / Latency.",
            "Context Length Cliff: Memory required for long-context KV activations rapidly eclipses static model weights.",
        ],
        callout_boxes=[
            {"title": "THE 32k CLIFF", "value": "16.4 GB", "desc": "Standard MHA on 7B model requires 16.4 GB solely for KV cache at 32k context."}
        ],
        transition_type="push",
        transition_direction="from-right",
    )

    # Slide 3: Profiler Architecture & Zero-Disk Storage
    builder.add_slide(
        title="Profiler Architecture & Telemetry Interception",
        subtitle="Non-Blocking Step Hooks, In-Memory Storage, and Zero-Disk Telemetry",
        bullets=[
            "PyTorch Hooks: Intercepts unnormalized logits via torch.inference_mode() with zero synchronization stalls.",
            "Dynamic Ollama Client: GGUF model introspection via /api/show and streaming latency telemetry via /api/generate.",
            "In-RAM Zero-Disk Architecture: /dev/shm tmpfs logs, circular RAM buffers, and in-memory SQLite buffer with zero physical disk wear.",
            "Dynamic Token Scaling: Unconstrained EOS generation and context scaling up to 131k tokens without sentence truncation.",
        ],
        callout_boxes=[
            {"title": "DISK WRITES", "value": "0.00 MB/s", "desc": "Pure in-RAM circular logging and tmpfs eliminate all physical drive wear."}
        ],
        transition_type="wipe",
        transition_direction="from-left",
    )

    # Slide 4: KV-Cache Allocation: Math vs Allocator
    builder.add_slide(
        title="KV-Cache Allocation: Analytical Math vs. PyTorch Reserved Pool",
        subtitle="Attention Scaling Laws and Dynamic Caching Allocator Fragmentation",
        bullets=[
            "Exact Analytical Footprint: M_KV = 2 * Batch * SeqLen * Layers * KV_Heads * HeadDim * DtypeBytes.",
            "Grouped-Query Attention (GQA): 8 KV heads vs 32 query heads achieves a 75% memory reduction over standard MHA.",
            "Allocator Arena Dynamics: PyTorch rounds allocations to 512B/2MB blocks and expands reserved pools in 20MB chunks.",
            "Fragmentation Telemetry: Tracks Phi_frag = 1 - (M_allocated / M_reserved) to identify out-of-memory risks before failure.",
        ],
        callout_boxes=[
            {"title": "GQA SAVINGS", "value": "75%", "desc": "Llama-3-8B reduces KV-cache from 16.4GB to 4.1GB at 32k tokens."}
        ],
        transition_type="push",
        transition_direction="from-bottom",
    )

    # Slide 5: Shannon Entropy & Thermodynamic Softmax
    builder.add_slide(
        title="Token Entropy & Thermodynamic Softmax Scaling",
        subtitle="Information-Theoretic Telemetry and Softmax Temperature Perturbation",
        bullets=[
            "Shannon Entropy: Quantifies model predictive uncertainty: H(P_t) = -sum(p * log2(p)).",
            "Thermodynamic Logit Inversion: Reconstructs candidate logits dynamically via z_i = z_1 + T_eff * ln(p_i / p_1).",
            "Probability Margins: Delta_p = p(1) - p(2) reveals intense competition between alternative completion candidates.",
            "Token Entropy Canvas: Dual paragraph flow and pills view with cyan text selection and non-destructive inspection.",
        ],
        callout_boxes=[
            {"title": "THERMODYNAMICS", "value": "T in [0.1, 2]", "desc": "Simulates candidate probability redistribution across temperatures in real-time."}
        ],
        transition_type="fade",
        transition_speed="medium",
    )

    # Slide 6: Latency Profiles & 60 FPS Visualizations
    builder.add_slide(
        title="Latency Profiles & 60 FPS Visualizations",
        subtitle="Pre-Fill vs Generation Latency with Adaptive Downsampling",
        bullets=[
            "Time-to-First-Token (TTFT): Reflects prompt pre-fill latency across matrix-matrix attention kernels.",
            "Inter-Token Latency (ITL): Steady-state autoregressive decoding time with real-time jitter tracking.",
            "Adaptive Downsampling: Compresses dense series (>250 points) for 18.5x faster rendering (4.4ms vs 82ms).",
            "Host RAM Safety: Proactive memory estimation and 1-click emergency model eviction prevent laptop freezes.",
        ],
        callout_boxes=[
            {"title": "CHART SPEEDUP", "value": "18.5x", "desc": "Vectorized operations and downsampling deliver buttery-smooth 60 FPS rendering."}
        ],
        transition_type="push",
        transition_direction="from-right",
    )

    # Slide 7: Real-World Case Studies
    builder.add_slide(
        title="Real-World Architectural Case Studies",
        subtitle="Empirical Telemetry across Llama-3, Qwen-2.5, and DeepSeek-V3",
        bullets=[
            "Llama-3-8B: GQA (8 KV heads) exhibits linear KV growth reaching 1.02 GB at 8k context and 4.10 GB at 32k context.",
            "Qwen-2.5-14B: 48 layers with GQA requires 6.14 GB at 32k context, maintaining steady ITL of ~45 ms.",
            "DeepSeek-V3 (MLA): Multi-Head Latent Attention compresses KV representations into a 512-dim latent vector.",
            "Compression Superiority: MLA achieves an 86% memory reduction relative to standard MHA, scaling easily to 128k context.",
        ],
        callout_boxes=[
            {"title": "MLA ADVANTAGE", "value": "2.2 GB", "desc": "DeepSeek-V3 requires only 2.2 GB for 32k context vs 16.4 GB for MHA."}
        ],
        transition_type="wipe",
        transition_direction="from-right",
    )

    # Slide 8: Roadmap & Extensibility
    builder.add_slide(
        title="Roadmap & Future Extensions",
        subtitle="Advanced PagedAttention Telemetry and Distributed Multi-GPU Serving",
        bullets=[
            "PagedAttention Integration: Visualizing virtual block tables, page-level fragmentation, and memory sharing across parallel requests.",
            "vLLM & SGLang Interceptors: Direct hooks into production inference engine memory managers.",
            "Multi-GPU Telemetry: Profiling tensor-parallelism communication bubbles, NCCL collective overhead, and TP-rank memory skew.",
            "Automated Speculative Decoding: Tracking draft model acceptance rates and verification step latency savings.",
        ],
        callout_boxes=[
            {"title": "NEXT FRONTIER", "value": "Distributed", "desc": "Extending telemetry to multi-node Tensor Parallelism and Speculative Decoding."}
        ],
        transition_type="fade",
        transition_speed="fast",
    )

    odp_file = os.path.join(output_dir, "presentation.odp")
    builder.save_odp(odp_file)
    logger.info(f"Saved OpenDocument presentation to: {odp_file}")

    pdf_file = builder.export_to_pdf(odp_file, output_dir=output_dir)
    logger.info(f"Exported presentation PDF to: {pdf_file}")

    return {"odp": odp_file, "pdf": pdf_file}


# MCP Server Dispatcher & JSON-RPC Protocol Loop
MCP_TOOLS_SPEC = [
    {
        "name": "create_presentation",
        "description": "Initialize a new presentation with title and theme",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "theme": {"type": "string", "default": "cyber_dark"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "add_slide",
        "description": "Add a new slide with bullets, callouts, and transition",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "subtitle": {"type": "string"},
                "bullets": {"type": "array", "items": {"type": "string"}},
                "transition_type": {"type": "string", "enum": ["fade", "push", "wipe"]},
                "transition_speed": {"type": "string", "enum": ["fast", "medium", "slow"]},
            },
            "required": ["title"],
        },
    },
    {
        "name": "apply_animation",
        "description": "Apply sequential build animation to a shape on a slide",
        "parameters": {
            "type": "object",
            "properties": {
                "slide_index": {"type": "integer"},
                "shape_id": {"type": "string"},
                "animation_type": {"type": "string", "default": "appear"},
                "trigger_event": {"type": "string", "default": "click"},
            },
            "required": ["slide_index", "shape_id"],
        },
    },
    {
        "name": "set_slide_transition",
        "description": "Set transition properties for a slide",
        "parameters": {
            "type": "object",
            "properties": {
                "slide_index": {"type": "integer"},
                "transition_type": {"type": "string"},
                "speed": {"type": "string"},
            },
            "required": ["slide_index", "transition_type"],
        },
    },
    {
        "name": "export_presentation",
        "description": "Save ODP presentation and export to PDF via LibreOffice",
        "parameters": {
            "type": "object",
            "properties": {
                "output_odp_path": {"type": "string", "default": "slides/presentation.odp"},
                "export_pdf": {"type": "boolean", "default": True},
            },
        },
    },
]


def handle_mcp_request(req: Dict[str, Any]) -> Dict[str, Any]:
    """Handles an incoming Model Context Protocol (MCP) JSON-RPC request."""
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": MCP_TOOLS_SPEC},
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        try:
            if tool_name == "create_presentation":
                res = tool_create_presentation(**args)
            elif tool_name == "add_slide":
                res = tool_add_slide(**args)
            elif tool_name == "apply_animation":
                res = tool_apply_animation(**args)
            elif tool_name == "set_slide_transition":
                res = tool_set_slide_transition(**args)
            elif tool_name == "export_presentation":
                res = tool_export_presentation(**args)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": json.dumps(res, indent=2)}]},
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)},
            }

    elif method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "mcp-libreoffice-impress",
                    "version": "1.0.0",
                },
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not supported: {method}"},
    }


def run_stdio_server():
    """Starts the standard I/O loop for MCP clients."""
    logger.info("Starting LibreOffice Impress MCP stdio server...")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_mcp_request(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    if "--build-deck" in sys.argv:
        print("Generating 8-slide technical deck...")
        paths = build_full_technical_deck("slides")
        print(f"Deck generated successfully!")
        print(f"ODP: {paths['odp']}")
        print(f"PDF: {paths['pdf']}")
    else:
        # Default to building deck if running directly, or starting stdio server if piped
        if sys.stdin.isatty():
            print("Interactive execution detected. Generating technical deck...")
            paths = build_full_technical_deck("slides")
            print(f"Deck generation complete: {paths['pdf']}")
        else:
            run_stdio_server()
