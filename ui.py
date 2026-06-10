"""
Gradio UI for the Qwen2.5-0.5B Visual Step-Through Explorer.

Phase 3 — Two-panel layout: interactive SVG flowchart (left) +
detail panel (right) replaces the old tabbed layout.
"""

from __future__ import annotations

from typing import Optional

import gradio as gr
import plotly.graph_objects as go

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
    plot_residual_evolution,
    plot_rope_comparison,
    plot_rope_rotation,
    plot_rmsnorm_comparison,
    plot_silu_scatter,
    plot_topk,
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
        self.collapsed: set[str] = set()

    @property
    def current_svg(self) -> str:
        """Render the SVG flowchart using current UI state."""
        return render_flowchart_svg(
            nodes=self.node_tree,
            active_node_id=self.active_node_id,
            collapsed_override=self.collapsed,
        )


controller = UIController()


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
    """Return a human-readable description for the given node."""
    if not controller.node_tree:
        return "Loading model…"
    node = controller.node_tree.get(node_id)
    if node is None:
        return ""
    return NODE_DESCRIPTIONS.get(node.node_type, f"{node.label} — explore internals.")


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
#  Plot dispatch  (Task 3.4)
# ─────────────────────────────────────────────────────────────────────────────

# Map plot_fn names (from FlowchartNode) → plot functions
PLOT_DISPATCH: dict[str, callable] = {
    "embedding": plot_embedding_slice,
    "rmsnorm": plot_rmsnorm_comparison,
    "attention": plot_attention_scores,
    "attention_scores": plot_attention_scores,
    "attention_probs": plot_attention,
    "mlp": plot_mlp_activation,
    "residual": plot_residual_delta,
    "linear": plot_weight_matrix,
    "activation": plot_silu_scatter,
    "softmax": plot_attention_scores,
    "output": plot_topk,
    "qkv": plot_qkv_vector,
    "rope": plot_rope_rotation,
    "logits": plot_topk,
    "histogram": plot_histogram,
    "silu_scatter": plot_silu_scatter,
    "layer_contributions": plot_layer_contributions,
    "residual_evolution": plot_residual_evolution,
    "rope_comparison": plot_rope_comparison,
}

# Fallback by node_type
NODE_TYPE_PLOT: dict[str, callable] = {
    "embedding": plot_embedding_slice,
    "rmsnorm": plot_rmsnorm_comparison,
    "attention": plot_attention_scores,
    "mlp": plot_mlp_activation,
    "residual": plot_residual_delta,
    "linear": plot_weight_matrix,
    "activation": plot_silu_scatter,
    "softmax": plot_attention_scores,
    "output": plot_topk,
}


def _dispatch_plot(node_id: str, snapshot) -> go.Figure:
    """Map a node ID to the appropriate plot function.

    Priority:
      1. plot_fn name (from FlowchartNode) into PLOT_DISPATCH
      2. node_type into NODE_TYPE_PLOT
      3. Empty figure fallback.
    """
    node = controller.node_tree.get(node_id)
    if not node:
        return _empty_fig("Unknown component.")

    # Try plot_fn first
    if node.plot_fn and node.plot_fn in PLOT_DISPATCH:
        fn = PLOT_DISPATCH[node.plot_fn]
        try:
            fig = fn(snapshot)
            if fig is not None:
                return fig
        except Exception:
            pass

    # Fallback by node_type
    if node.node_type in NODE_TYPE_PLOT:
        fn = NODE_TYPE_PLOT[node.node_type]
        try:
            fig = fn(snapshot)
            if fig is not None:
                return fig
        except Exception:
            pass

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

# Output indices
_O_SVG = 0
_O_PLOT = 1
_O_TITLE = 2
_O_DESC = 3


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


def _build_response(token_idx: int = 0) -> tuple:
    """Build the 4-part output tuple for the current UI state."""
    svg = controller.current_svg
    explorer = controller.explorer
    snapshot = explorer.get_step(int(token_idx)) if explorer and explorer.num_steps() > 0 else None

    # Active node selected → detail view
    if controller.active_node_id and snapshot:
        fig = _dispatch_plot(controller.active_node_id, snapshot)
        title = _get_node_title(controller.active_node_id)
        desc = _get_node_description(controller.active_node_id)
        return (svg, fig, f"### {title}", desc)

    # No active node → overview
    if snapshot:
        fig = plot_layer_contributions(snapshot)
        return (svg, fig, "### 🏗️ Qwen2.5-0.5B", "Click a component in the flowchart to inspect.")
    else:
        return (svg, _empty_fig("Press Generate to start."), "### 🏗️ Qwen2.5-0.5B", "Enter a prompt and click **Generate**.")


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
    return _build_response(explorer.num_steps() - 1)


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
    return _build_response(explorer.num_steps() - 1)


def on_node_select(node_id: str) -> tuple:
    """Handle node selection from a flowchart click event."""
    _ensure_loaded()
    controller.active_node_id = node_id
    return _build_response()


def on_collapse_toggle(node_id: str) -> str:
    """Toggle collapse state for a node group (called from JS bridge)."""
    if node_id in controller.collapsed:
        controller.collapsed.discard(node_id)
    else:
        controller.collapsed.add(node_id)
    return controller.current_svg


def on_reset() -> tuple:
    """Reset the entire explorer state."""
    controller.explorer = None
    controller.ready = False
    controller.node_tree = {}
    controller.active_node_id = None
    controller.collapsed = set()
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
            with gr.Column(scale=1, min_width=120):
                generate_btn = gr.Button(
                    "⚡ Generate",
                    variant="primary",
                    elem_id="generate-btn",
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
        node_select_input = gr.Textbox(visible=False, value="", elem_id="node-select-input")
        collapse_toggle_input = gr.Textbox(visible=False, value="", elem_id="collapse-toggle-input")

        # ── Wire up events ──

        # Generate button
        generate_btn.click(
            fn=on_generate,
            inputs=[prompt_input],
            outputs=[flowchart_html, detail_plot, detail_title, detail_description],
        )

        # JS bridge: node selection
        def _handle_node_select(node_id: str) -> tuple:
            if not node_id:
                return _build_response(0)
            return on_node_select(node_id)

        node_select_input.change(
            fn=_handle_node_select,
            inputs=[node_select_input],
            outputs=[flowchart_html, detail_plot, detail_title, detail_description],
        )

        # JS bridge: collapse toggle
        collapse_toggle_input.change(
            fn=on_collapse_toggle,
            inputs=[collapse_toggle_input],
            outputs=[flowchart_html],
        )

        # ── JavaScript injection ──
        # Listen for CustomEvent('flowchart-node-select') dispatched by the
        # SVG's selectNode() and toggleCollapse() JS functions, then forward
        # the node ID into the hidden Gradio textboxes.
        flowchart_html.change(
            fn=None,
            inputs=None,
            outputs=None,
            _js="""
            () => {
                const nodeInput = document.getElementById('node-select-input');
                const collapseInput = document.getElementById('collapse-toggle-input');

                function setNativeValue(element, value) {
                    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                        window.HTMLTextAreaElement.prototype, 'value'
                    ).set;
                    nativeInputValueSetter.call(element, value);
                    element.dispatchEvent(new Event('input', { bubbles: true }));
                }

                document.addEventListener('flowchart-node-select', (e) => {
                    const nodeId = e.detail ? e.detail.nodeId : null;
                    if (nodeId && nodeInput) setNativeValue(nodeInput, nodeId);
                });

                document.addEventListener('flowchart-collapse-toggle', (e) => {
                    const nodeId = e.detail ? e.detail.nodeId : null;
                    if (nodeId && collapseInput) setNativeValue(collapseInput, nodeId);
                });
                return [];
            }
            """,
        )

    return demo