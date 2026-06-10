"""
Gradio UI for the Qwen2.5-0.5B Visual Step-Through Explorer.

Phase 3 — Two-panel layout: interactive SVG flowchart (left) +
detail panel (right) replaces the old tabbed layout.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import gradio as gr
import plotly.graph_objects as go

from data_structures import StepSnapshot
from engine import QwenExplorer
from flowchart import (
    FlowchartNode,
    build_qwen_node_tree,
    render_flowchart_svg,
)
from plots import (
    _empty_fig,
    plot_attention,
    plot_attention_scores,
    plot_embedding_slice,
    plot_histogram,
    plot_layer_contributions,
    plot_mlp_activation,
    plot_qkv_vector,
    plot_residual_delta,
    plot_rope_comparison,
    plot_rope_rotation,
    plot_rmsnorm_comparison,
    plot_silu_scatter,
    plot_top_logits,
    plot_topk,
    plot_vector_bar,
    plot_weight_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Controller — holds state between callbacks
# ─────────────────────────────────────────────────────────────────────────────


class UIController:
    """Holds the explorer instance, node tree, and UI state across callbacks."""

    def __init__(self):
        self.explorer: Optional[QwenExplorer] = None
        self.ready = False
        self.node_tree: dict[str, FlowchartNode] = {}
        self.active_node_id: Optional[str] = None
        self.expanded_override: set[str] = set()
        self.thumbnails: dict[str, str] = {}

    @property
    def current_svg(self) -> str:
        """Render the SVG flowchart using current UI state."""
        return render_flowchart_svg(
            nodes=self.node_tree,
            active_node_id=self.active_node_id,
            thumbnails=self.thumbnails,
            expanded_override=self.expanded_override,
        )

    @property
    def latest_step_idx(self) -> int:
        """Index of the most recent step, or 0."""
        if self.explorer is not None:
            return max(0, self.explorer.num_steps() - 1)
        return 0

    def get_snapshot(self, step_idx: int | None = None) -> StepSnapshot | None:
        """Return a StepSnapshot (latest by default)."""
        if self.explorer is None or self.explorer.num_steps() == 0:
            return None
        idx = step_idx if step_idx is not None else self.latest_step_idx
        return self.explorer.get_step(idx)


controller = UIController()


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

_LAYER_ID_RE = re.compile(r"layer_(\d+)")


def _parse_layer_idx(node_id: str) -> int | None:
    """Extract the layer index from a node id like ``layer_12_q_proj``."""
    m = _LAYER_ID_RE.search(node_id)
    return int(m.group(1)) if m else None


def _get_token_labels() -> list[str]:
    """Return decoded token strings from the current explorer state."""
    if controller.explorer is not None:
        return controller.explorer.get_token_labels()
    return []


# ─────────────────────────────────────────────────────────────────────────────
#  Node descriptions
# ─────────────────────────────────────────────────────────────────────────────

NODE_DESCRIPTIONS: dict[str, str] = {
    "root": "Overview of the full Qwen2.5-0.5B transformer — 24 layers, each with attention + MLP.",
    "embedding": "Token Embedding — maps each token ID → 896-dimensional vector.",
    "rmsnorm": "RMS Layer Normalization — normalises activations by their RMS value.",
    "attention": "Multi-Head Self-Attention — Q/K/V projections with RoPE, scaled dot-product, softmax.",
    "mlp": "MLP (Feed-Forward) — SwiGLU: Gate×Up → SiLU activation → Down projection.",
    "residual": "Residual Connection — adds the sublayer output back into the residual stream.",
    "linear": "Linear Projection — weight-matrix multiplication (no bias).",
    "activation": "SiLU (SwiGLU) Activation — σ(x)·x element-wise nonlinearity.",
    "softmax": "Softmax — normalises attention scores into a probability distribution.",
    "output": "LM Head — final linear projection from hidden dim → vocabulary logits.",
    "layer_group": "One of 24 identical transformer layers.",
    "component_group": "Collapsible group of related components within a layer.",
}


def _get_node_description(node_id: str) -> str:
    """Return a human-readable description for the given node.

    When a generation snapshot is available, enriches descriptions with
    actual tensor shapes relevant to the node type.
    """
    if not controller.node_tree:
        return "Loading model…"
    node = controller.node_tree.get(node_id)
    if node is None:
        return ""
    base = NODE_DESCRIPTIONS.get(node.node_type, f"{node.label} — explore internals.")

    # Attach tensor shapes from the current snapshot (if any)
    snapshot = controller.get_snapshot()
    if snapshot is None:
        return base

    shape_info = _tensor_shapes_for_node(node_id, node.node_type, snapshot)
    if shape_info:
        return base + "\n\n" + shape_info
    return base


def _tensor_shapes_for_node(node_id: str, node_type: str, snapshot: StepSnapshot) -> str:
    """Return a formatted string of relevant tensor shapes for *node_type*."""
    # If a specific layer is targeted, extract its snapshot
    layer_idx = _layer_index_from_node_id(node_id)
    layer = snapshot.layers.get(layer_idx) if layer_idx is not None else None

    if node_type == "root":
        n = len(snapshot.layers)
        h = snapshot.input_embeds.shape[-1] if snapshot.input_embeds is not None else "?"
        return f"• **Layers:** {n}  • **Hidden dim:** {h}"

    if node_type == "embedding" and snapshot.input_embeds is not None:
        return f"• **Shape:** `{list(snapshot.input_embeds.shape)}`  (seq_len × hidden_dim)"

    if node_type == "output" and snapshot.logits is not None:
        return f"• **Logits shape:** `{list(snapshot.logits.shape)}`  (vocab_size,)"

    if layer is None:
        return ""

    if node_type == "rmsnorm" and layer.input_layernorm_output is not None:
        return f"• **Shape:** `{list(layer.input_layernorm_output.shape)}`"

    if node_type == "attention":
        parts = []
        if layer.q is not None:
            parts.append(f"Q `{list(layer.q.shape)}`")
        if layer.k is not None:
            parts.append(f"K `{list(layer.k.shape)}`")
        if layer.v is not None:
            parts.append(f"V `{list(layer.v.shape)}`")
        if layer.attn_output is not None:
            parts.append(f"out `{list(layer.attn_output.shape)}`")
        return "• " + "  • ".join(parts) if parts else ""

    if node_type == "mlp":
        parts = []
        if layer.mlp_gate_raw is not None:
            parts.append(f"gate `{list(layer.mlp_gate_raw.shape)}`")
        if layer.mlp_up is not None:
            parts.append(f"up `{list(layer.mlp_up.shape)}`")
        if layer.mlp_output is not None:
            parts.append(f"out `{list(layer.mlp_output.shape)}`")
        return "• " + "  • ".join(parts) if parts else ""

    if node_type == "residual":
        parts = []
        if layer.residual_pre_attn is not None:
            parts.append(f"pre-attn `{list(layer.residual_pre_attn.shape)}`")
        if layer.residual_post_attn is not None:
            parts.append(f"post-attn `{list(layer.residual_post_attn.shape)}`")
        if layer.residual_post_mlp is not None:
            parts.append(f"post-mlp `{list(layer.residual_post_mlp.shape)}`")
        return "• " + "  • ".join(parts) if parts else ""

    if node_type == "activation" and layer.mlp_gate_silu is not None:
        return f"• **Shape:** `{list(layer.mlp_gate_silu.shape)}`"

    if node_type == "softmax" and layer.attn_probs is not None:
        return f"• **Shape:** `{list(layer.attn_probs.shape)}`  (num_heads × seq_len)"

    if node_type == "linear":
        # Show weight matrix shape for the relevant linear layer
        parts = []
        for name, arr in [("Q", layer.q_weight), ("K", layer.k_weight),
                           ("V", layer.v_weight), ("O", layer.o_weight),
                           ("gate", layer.gate_weight), ("up", layer.up_weight),
                           ("down", layer.down_weight)]:
            if arr is not None:
                parts.append(f"{name} `{list(arr.shape)}`")
        return "• " + "  • ".join(parts) if parts else ""

    return ""


def _layer_index_from_node_id(node_id: str) -> int | None:
    """Parse layer index from a node ID (delegates to _parse_layer_idx)."""
    if not node_id:
        return None
    return _parse_layer_idx(node_id)


def _get_node_title(node_id: str) -> str:
    """Return a display title (with icon) for the given node."""
    if not controller.node_tree:
        return "Qwen2.5-0.5B"
    node = controller.node_tree.get(node_id)
    if node is None:
        return "Qwen2.5-0.5B"
    icon = node.icon or ""
    return f"{icon} {node.label}"


# ─────────────────────────────────────────────────────────────────────────────
#  Plot dispatch  (Task 3.4, fixed: extract tensors per node type)
# ─────────────────────────────────────────────────────────────────────────────


def _dispatch_plot(node_id: str) -> go.Figure:
    """
    Map a node ID to the appropriate plot function, extracting the right
    tensors from the current StepSnapshot.

    Each node type / node_id pattern maps to a specific plot call with the
    correct arguments extracted from the snapshot.
    """
    node = controller.node_tree.get(node_id)
    if not node:
        return _empty_fig("Unknown component.")

    snapshot = controller.get_snapshot()
    labels = _get_token_labels()

    # ── Helper: get a single layer's snapshot ──
    def _layer(lidx: int) -> Optional:
        if snapshot is None:
            return None
        return snapshot.layers.get(lidx)

    try:
        # ── Root → layer contributions overview ──
        if node_id == "root":
            if snapshot is not None and snapshot.layers:
                contribs: List[float] = []
                for lidx in sorted(snapshot.layers.keys()):
                    layer = snapshot.layers[lidx]
                    if layer.residual_pre_attn is not None and layer.residual_post_mlp is not None:
                        delta = float(np.linalg.norm(
                            layer.residual_post_mlp - layer.residual_pre_attn
                        ))
                        contribs.append(delta)
                    else:
                        contribs.append(0.0)
                return plot_layer_contributions(contribs)
            return _empty_fig("Press Generate to start.")

        # ── Token Embedding ──
        if node_id == "token_embedding":
            if snapshot is not None and snapshot.token_embeddings is not None:
                return plot_embedding_slice(
                    snapshot.token_embeddings,
                    token_id=snapshot.token_id,
                )
            return _empty_fig("Token embedding — no data (generate first).")

        # ── Layer-based dispatch ──
        lidx = _parse_layer_idx(node_id)
        layer = _layer(lidx) if lidx is not None else None

        # ── Attention weight matrices ──
        if node_id.endswith("_q_proj"):
            if layer and layer.q_weight is not None:
                return plot_weight_matrix(layer.q_weight, "Q Projection")
            return _empty_fig("Q weight — static; not yet captured.")
        if node_id.endswith("_k_proj"):
            if layer and layer.k_weight is not None:
                return plot_weight_matrix(layer.k_weight, "K Projection")
            return _empty_fig("K weight — static; not yet captured.")
        if node_id.endswith("_v_proj"):
            if layer and layer.v_weight is not None:
                return plot_weight_matrix(layer.v_weight, "V Projection")
            return _empty_fig("V weight — static; not yet captured.")
        if node_id.endswith("_o_proj"):
            if layer and layer.o_weight is not None:
                return plot_weight_matrix(layer.o_weight, "O Projection")
            return _empty_fig("O weight — static; not yet captured.")

        # ── RoPE comparison (Q/K pre vs post) ──
        if "_rope" in node_id and "rope_comparison" not in node_id:
            if layer is not None and layer.q_pre_rope is not None:
                return plot_rope_comparison(
                    layer.q_pre_rope,
                    layer.k_pre_rope,
                    layer.q_post_rope,
                    layer.k_post_rope,
                )
            return _empty_fig("RoPE — no data (step through a token).")

        # ── Attention scores (pre-softmax) ──
        if node_id.endswith("_attn_scores"):
            if layer is not None and layer.attn_scores is not None:
                return plot_attention_scores(layer.attn_scores, labels, lidx, 0)
            return _empty_fig("Attention scores — no data.")

        # ── Attention weights (post-softmax) ──
        if node_id.endswith("_attn_weights"):
            if layer is not None and layer.attn_probs is not None:
                return plot_attention(layer.attn_probs, labels, lidx, 0)
            return _empty_fig("Attention weights — no data.")

        # ── RMSNorm ──
        if node_id.endswith("_pre_attn_rmsnorm"):
            if layer is not None and layer.residual_pre_attn is not None:
                return plot_rmsnorm_comparison(
                    layer.residual_pre_attn, layer.input_layernorm_output
                )
            return _empty_fig("Pre-attention RMSNorm — no data.")
        if node_id.endswith("_pre_mlp_rmsnorm"):
            if layer is not None and layer.residual_post_attn is not None:
                return plot_rmsnorm_comparison(
                    layer.residual_post_attn, layer.post_attention_layernorm_output
                )
            return _empty_fig("Pre-MLP RMSNorm — no data.")

        # ── Residual connections ──
        if node_id.endswith("_post_attn_residual"):
            if layer is not None:
                return plot_residual_delta(
                    layer.residual_pre_attn, layer.residual_post_attn,
                    title="Post-Attention Residual",
                )
            return _empty_fig("Post-attention residual — no data.")
        if node_id.endswith("_post_mlp_residual"):
            if layer is not None:
                return plot_residual_delta(
                    layer.residual_post_attn, layer.residual_post_mlp,
                    title="Post-MLP Residual",
                )
            return _empty_fig("Post-MLP residual — no data.")

        # ── MLP sub-nodes ──
        if node_id.endswith("_gate_proj"):
            if layer is not None and layer.mlp_gate_raw is not None:
                return plot_histogram(layer.mlp_gate_raw, "Gate Projection (pre-SiLU)")
            return _empty_fig("Gate projection — no data.")
        if node_id.endswith("_silu"):
            if layer is not None and layer.mlp_gate_raw is not None:
                return plot_silu_scatter(layer.mlp_gate_raw, layer.mlp_gate_silu)
            return _empty_fig("SiLU activation — no data.")
        if node_id.endswith("_up_proj"):
            if layer and layer.up_weight is not None:
                return plot_weight_matrix(layer.up_weight, "Up Projection")
            return _empty_fig("Up weight — static; not yet captured.")
        if node_id.endswith("_down_proj"):
            if layer and layer.down_weight is not None:
                return plot_weight_matrix(layer.down_weight, "Down Projection")
            return _empty_fig("Down weight — static; not yet captured.")
        if node_id.endswith("_mlp_out"):
            if layer is not None and layer.mlp_output is not None:
                return plot_mlp_activation(layer.mlp_output, "MLP Output")
            return _empty_fig("MLP output — no data.")

        # ── Final RMSNorm ──
        if node_id == "final_rmsnorm":
            if snapshot is not None and snapshot.final_norm_output is not None:
                # Compare last layer's post-mlp residual with final norm output
                last_lid = max(snapshot.layers.keys(), default=0)
                last_layer = snapshot.layers.get(last_lid)
                if last_layer and last_layer.residual_post_mlp is not None:
                    return plot_rmsnorm_comparison(
                        last_layer.residual_post_mlp,
                        snapshot.final_norm_output,
                        title="Final RMSNorm",
                    )
                return plot_vector_bar(snapshot.final_norm_output, "Final RMSNorm Output")
            return _empty_fig("Final RMSNorm — no data.")

        # ── LM Head / Logits ──
        if node_id == "lm_head":
            if snapshot is not None:
                return plot_topk(snapshot.topk_tokens, snapshot.topk_probs)
            return _empty_fig("LM Head — no data.")

        # ── Fallback by node_type ──
        if node.node_type == "linear" and lidx is not None and layer is not None:
            # Generic linear: try to find any weight
            for attr in ("q_weight", "k_weight", "v_weight", "o_weight",
                         "gate_weight", "up_weight", "down_weight"):
                w = getattr(layer, attr, None)
                if w is not None:
                    return plot_weight_matrix(w, node.label)
        if node.node_type == "residual" and lidx is not None and layer is not None:
            return plot_residual_delta(
                layer.residual_pre_attn, layer.residual_post_mlp,
                title=node.label,
            )

    except Exception as e:
        return _empty_fig(f"Plot error: {e}")

    return _empty_fig("No plot available for this component.")


# ─────────────────────────────────────────────────────────────────────────────
#  Lazy-loading
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_loaded():
    """Lazy-load the model and build the node tree on first use."""
    if controller.explorer is None:
        controller.explorer = QwenExplorer()
        controller.explorer.load()
        controller.ready = True
        controller.node_tree = build_qwen_node_tree()


# ─────────────────────────────────────────────────────────────────────────────
#  Output helpers
# ─────────────────────────────────────────────────────────────────────────────

_NUM_OUTPUTS = 4  # flowchart_html, detail_plot, detail_title, detail_description

_O_SVG = 0
_O_PLOT = 1
_O_TITLE = 2
_O_DESC = 3


def _build_response() -> tuple:
    """Build the 4-part output tuple for the current UI state."""
    svg = controller.current_svg
    snapshot = controller.get_snapshot()

    # Active node selected → detail view
    if controller.active_node_id and snapshot:
        fig = _dispatch_plot(controller.active_node_id)
        title = _get_node_title(controller.active_node_id)
        desc = _get_node_description(controller.active_node_id)
        return (svg, fig, f"### {title}", desc)

    # No active node → overview (delegate to dispatch which has correct extraction)
    if snapshot:
        fig = _dispatch_plot("root")
        return (svg, fig, "### 🏗️ Qwen2.5-0.5B", "Click a component in the flowchart to inspect.")
    else:
        return (svg, _empty_fig("Press Generate to start."), "### 🏗️ Qwen2.5-0.5B", "Enter a prompt and click **Generate**.")


def _empty_response() -> tuple:
    """Return outputs for the empty / no-data state."""
    return (
        controller.current_svg if controller.node_tree else "<svg></svg>",
        _empty_fig("Press Generate to start."),
        "### 🏗️ Qwen2.5-0.5B",
        "Enter a prompt and click **Generate** to explore.",
    )


def _error_response(msg: str) -> tuple:
    """Return error-state outputs."""
    return (
        controller.current_svg,
        _empty_fig(f"⚠️ {msg}"),
        "### ⚠️ Error",
        msg,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Callbacks  (Task 3.6)
# ─────────────────────────────────────────────────────────────────────────────


def on_generate(prompt: str, max_new: int = 20, temperature: float = 0.8) -> tuple:
    """Full auto-generation."""
    _ensure_loaded()
    explorer = controller.explorer
    try:
        explorer.generate_full(prompt, max_new_tokens=max_new, temperature=temperature)
    except Exception as e:
        return _error_response(str(e))
    controller.active_node_id = None
    return _build_response()


def on_step(prompt: str, temperature: float = 0.8) -> tuple:
    """Generate one more token."""
    _ensure_loaded()
    explorer = controller.explorer
    try:
        if explorer.num_steps() == 0:
            explorer.generate_full(prompt, max_new_tokens=1, temperature=temperature)
        else:
            explorer.step_forward(temperature=temperature)
    except Exception as e:
        return _error_response(str(e))
    return _build_response()


def on_node_select(node_id: str) -> tuple:
    """Handle node selection from a flowchart click event."""
    if not node_id:
        return _build_response()
    _ensure_loaded()
    controller.active_node_id = node_id

    # Auto-expand ancestors so the selected node is visible
    node_node = controller.node_tree.get(node_id)
    while node_node and node_node.parent_id:
        parent = controller.node_tree.get(node_node.parent_id)
        if parent and parent.collapsed:
            controller.expanded_override.add(node_node.parent_id)
        node_node = parent

    return _build_response()


def on_collapse_toggle(node_id: str) -> str:
    """Toggle collapse state for a node group (called from JS bridge)."""
    if not node_id:
        return controller.current_svg
    if node_id in controller.expanded_override:
        controller.expanded_override.discard(node_id)
    else:
        controller.expanded_override.add(node_id)
    return controller.current_svg


def on_reset() -> tuple:
    """Reset the entire explorer state."""
    controller.explorer = None
    controller.ready = False
    controller.node_tree = {}
    controller.active_node_id = None
    controller.expanded_override = set()
    controller.thumbnails = {}
    return _empty_response()


# ─────────────────────────────────────────────────────────────────────────────
#  UI Builder  (Task 3.1 + 3.7 + 3.8)
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* Phase 3 — Two-panel layout styles */
#flowchart-container {
    overflow-y: auto !important;
    height: calc(100vh - 180px) !important;
    min-height: 500px;
    background: #1e1e1e;
    border-radius: 8px;
    border: 1px solid #333;
    padding: 8px;
}
#flowchart-container svg {
    width: 100%;
    height: auto;
}
#detail-panel {
    height: calc(100vh - 180px) !important;
    min-height: 500px;
    overflow-y: auto;
    padding: 0 4px;
}
#detail-title {
    font-size: 1.1em;
    font-weight: 600;
    color: #e0e0e0;
    margin-bottom: 4px;
}
#detail-description {
    font-size: 0.9em;
    color: #aaa;
    margin-bottom: 12px;
}
#generate-btn {
    background: #4a9eff;
    border: none;
    color: white;
    font-weight: 600;
}
#generate-btn:hover {
    background: #6ab0ff;
}
#prompt-input textarea {
    font-size: 14px;
}
"""

_JS_BRIDGE = """
<script type="text/javascript">
// Bridge between SVG flowchart CustomEvents and Gradio hidden textboxes
(function() {
    function setNativeValue(element, value) {
        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(
            window.HTMLTextAreaElement.prototype, 'value'
        ).set;
        nativeInputValueSetter.call(element, value);
        element.dispatchEvent(new Event('input', { bubbles: true }));
    }

    document.addEventListener('flowchart-node-select', function(e) {
        var nodeId = e.detail ? e.detail.nodeId : null;
        if (nodeId) {
            var input = document.getElementById('node-select-input');
            if (input) setNativeValue(input, nodeId);
        }
    });

    document.addEventListener('flowchart-collapse-toggle', function(e) {
        var nodeId = e.detail ? e.detail.nodeId : null;
        if (nodeId) {
            var input = document.getElementById('collapse-toggle-input');
            if (input) setNativeValue(input, nodeId);
        }
    });
})();
</script>
"""


def create_ui() -> gr.Blocks:
    """Build the Phase 3 two-panel Gradio interface.

    Left panel: Interactive SVG flowchart.
    Right panel: Detail view (title, description, plot).

    Returns:
        A gr.Blocks instance ready to launch.
    """
    # Build node tree upfront (before UI renders)
    if not controller.node_tree:
        controller.node_tree = build_qwen_node_tree()

    with gr.Blocks(
        title="LLM Visual Step-Through Explorer",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="gray"),
        css=CUSTOM_CSS,
    ) as demo:
        # ── Top bar: prompt + generate ──
        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                prompt_input = gr.Textbox(
                    value="The meaning of life is",
                    label="Prompt",
                    placeholder="Enter a prompt…",
                    elem_id="prompt-input",
                )
            with gr.Column(scale=1, min_width=100):
                temperature_input = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.8, step=0.05,
                    label="Temperature",
                )
            with gr.Column(scale=1, min_width=80):
                max_new_input = gr.Slider(
                    minimum=1, maximum=100, value=20, step=1,
                    label="Max tokens",
                )
            with gr.Column(scale=1, min_width=80):
                step_btn = gr.Button(
                    "▶ Step",
                    elem_id="step-btn",
                )
            with gr.Column(scale=1, min_width=100):
                generate_btn = gr.Button(
                    "⚡ Generate",
                    variant="primary",
                    elem_id="generate-btn",
                )
            with gr.Column(scale=1, min_width=80):
                reset_btn = gr.Button(
                    "⟲ Reset",
                    elem_id="reset-btn",
                )

        with gr.Row(equal_height=False):
            # ── LEFT: Flowchart ──
            with gr.Column(scale=35, min_width=350):
                flowchart_html = gr.HTML(
                    value=controller.current_svg,
                    elem_id="flowchart-container",
                )

            # ── RIGHT: Detail panel ──
            with gr.Column(scale=65, min_width=450):
                with gr.Column(elem_id="detail-panel"):
                    detail_title = gr.Markdown(
                        value="### 🏗️ Qwen2.5-0.5B",
                        elem_id="detail-title",
                    )
                    detail_description = gr.Markdown(
                        value="Click a component in the flowchart to inspect its internals.",
                        elem_id="detail-description",
                    )
                    detail_plot = gr.Plot(
                        value=_empty_fig("Press Generate."),
                        label="Detail View",
                        elem_id="detail-plot",
                    )

        # Hidden textboxes for JS → Python communication
        node_select_input = gr.Textbox(
            visible=False, value="", elem_id="node-select-input"
        )
        collapse_toggle_input = gr.Textbox(
            visible=False, value="", elem_id="collapse-toggle-input"
        )

        # JS bridge script (invisible HTML component at the bottom)
        gr.HTML(value=_JS_BRIDGE, visible=False)

        # ── Wire up events ──

        # Generate button → on_generate(prompt, max_new, temperature)
        generate_btn.click(
            fn=on_generate,
            inputs=[prompt_input, max_new_input, temperature_input],
            outputs=[flowchart_html, detail_plot, detail_title, detail_description],
        )

        # Node selection from JS bridge
        node_select_input.change(
            fn=on_node_select,
            inputs=[node_select_input],
            outputs=[flowchart_html, detail_plot, detail_title, detail_description],
        )

        # Collapse toggle from JS bridge
        collapse_toggle_input.change(
            fn=on_collapse_toggle,
            inputs=[collapse_toggle_input],
            outputs=[flowchart_html],
        )

        # Step button → on_step(prompt, temperature)
        step_btn.click(
            fn=on_step,
            inputs=[prompt_input, temperature_input],
            outputs=[flowchart_html, detail_plot, detail_title, detail_description],
        )

        # Reset button → on_reset()
        reset_btn.click(
            fn=on_reset,
            outputs=[flowchart_html, detail_plot, detail_title, detail_description],
        )

    return demo
