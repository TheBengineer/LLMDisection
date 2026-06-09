"""
Gradio UI for the Qwen2.5-0.5B Visual Step-Through Explorer.

Contains the UIController, all Gradio callbacks, and the create_ui() builder.

This module depends on engine.py and plots.py.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import gradio as gr
import numpy as np

from engine import QwenExplorer
from plots import (
    _empty_fig,
    plot_attention,
    plot_attention_scores,
    plot_histogram,
    plot_layer_contributions,
    plot_logits_sampled,
    plot_mlp_activation,
    plot_qkv_vector,
    plot_residual_delta,
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
    """Holds the explorer instance and ready flag across Gradio calls."""

    def __init__(self):
        self.explorer: Optional[QwenExplorer] = None
        self.ready = False


controller = UIController()


def _ensure_loaded():
    """Lazy-load the model on first use."""
    if controller.explorer is None:
        controller.explorer = QwenExplorer()
        controller.explorer.load()
        controller.ready = True


# ─────────────────────────────────────────────────────────────────────────────
#  Output tuple helpers
# ─────────────────────────────────────────────────────────────────────────────
# The UI has many output components. We define a fixed-length tuple so
# callbacks always return exactly the right number of values.

_NUM_OUTPUTS = 26  # total gr outputs

# Output indices (for readability)
_O_TEXT = 0
_O_COUNTER = 1
_O_TOKEN_SEL = 2
_O_ATTN_LAYER = 3
_O_ATTN_HEAD = 4
_O_QKV_LAYER = 5
_O_MLP_LAYER = 6
# Plots
_O_ATTN = 7
_O_ATTN_SCORES = 8
_O_Q = 9
_O_K = 10
_O_V = 11
_O_Q_WEIGHT = 12
_O_K_WEIGHT = 13
_O_V_WEIGHT = 14
_O_GATE = 15
_O_GATE_RAW = 16
_O_SILU = 17
_O_MLP_OUT = 18
_O_NORM_IN = 19
_O_NORM_OUT = 20
_O_RESIDUAL = 21
_O_TOPK = 22
_O_LOGITS = 23
_O_EMBED = 24
_O_STATE = 25


def _default_outputs() -> Tuple:
    """Return all outputs in their 'no data' state."""
    empty = _empty_fig("Press Generate")
    return (
        "No generation yet",          # text
        "0 tokens",                   # counter
        gr.Slider(value=0, maximum=0),  # token_selector
        gr.Slider(value=0, maximum=0),  # attn_layer
        gr.Slider(value=0, maximum=0),  # attn_head
        gr.Slider(value=0, maximum=0),  # qkv_layer
        gr.Slider(value=0, maximum=0),  # mlp_layer
    ) + tuple([empty] * 18) + (gr.State(False),)


# ─────────────────────────────────────────────────────────────────────────────
#  Callbacks
# ─────────────────────────────────────────────────────────────────────────────


def on_generate(prompt: str, max_new: int, temperature: float) -> Tuple:
    """Full auto-generation."""
    _ensure_loaded()
    explorer = controller.explorer

    try:
        explorer.generate_full(prompt, max_new_tokens=max_new, temperature=temperature)
    except Exception as e:
        return _make_error_outputs(f"Error: {e}")

    return _build_all_outputs(explorer.num_steps() - 1)


def on_step(prompt: str, temperature: float) -> Tuple:
    """Generate one more token."""
    _ensure_loaded()
    explorer = controller.explorer

    try:
        if explorer.num_steps() == 0:
            explorer.generate_full(prompt, max_new_tokens=1, temperature=temperature)
        else:
            explorer.step_forward(temperature=temperature)
    except Exception as e:
        return _make_error_outputs(f"Error: {e}")

    return _build_all_outputs(explorer.num_steps() - 1)


def on_select_token(token_idx: int, layer_idx: int, head_idx: int) -> Tuple:
    """User selected a different token position."""
    _ensure_loaded()
    return _build_all_outputs(token_idx, layer_idx, head_idx)


def on_change_layer(token_idx: int, layer_idx: int, head_idx: int) -> Tuple:
    """Layer or head slider changed in attention tab."""
    return _build_all_outputs(token_idx, layer_idx, head_idx)


def on_change_qkv_layer(token_idx: int, layer_idx: int) -> Tuple:
    """QKV tab layer changed."""
    return _build_qkv_outputs(token_idx, layer_idx)


def on_change_mlp_layer(token_idx: int, layer_idx: int) -> Tuple:
    """MLP tab layer changed."""
    return _build_mlp_outputs(token_idx, layer_idx)


def on_reset() -> Tuple:
    """Reset everything."""
    controller.explorer = None
    controller.ready = False
    return _default_outputs()


# ─────────────────────────────────────────────────────────────────────────────
#  Output builders
# ─────────────────────────────────────────────────────────────────────────────


def _make_error_outputs(msg: str) -> Tuple:
    """Return a full output tuple with an error message."""
    out = list(_default_outputs())
    out[_O_TEXT] = msg
    out[_O_STATE] = gr.State(False)
    return tuple(out)


def _build_all_outputs(
    token_idx: int, layer_idx: int = 0, head_idx: int = 0
) -> Tuple:
    """Return all Gradio output updates for the given token position."""
    explorer = controller.explorer
    if explorer is None or not controller.ready:
        return _default_outputs()

    step = explorer.get_step(int(token_idx))
    if step is None:
        return _default_outputs()

    token_labels = explorer.get_token_labels()
    full_text = explorer.get_full_text()
    num_steps = explorer.num_steps()
    max_layer = explorer.num_layers - 1
    max_head = explorer.num_heads - 1

    # Clamp
    layer_idx = min(int(layer_idx), max_layer)
    head_idx = min(int(head_idx), max_head)

    layer = step.layers.get(layer_idx)
    head_dim = explorer.head_dim

    # ── Attention plots ──
    attn_fig = (
        plot_attention(layer.attn_probs, token_labels, layer_idx, head_idx)
        if layer and layer.attn_probs is not None
        else _empty_fig("Attention (run Generate first)")
    )
    attn_scores_fig = (
        plot_attention_scores(layer.attn_scores, token_labels, layer_idx, head_idx)
        if layer and layer.attn_scores is not None
        else _empty_fig("Attention scores (run Generate first)")
    )

    # ── Q/K/V plots ──
    q_fig = (
        plot_qkv_vector(layer.q, f"Q — Layer {layer_idx}", head_dim)
        if layer and layer.q is not None
        else _empty_fig("Q (not captured)")
    )
    k_fig = (
        plot_qkv_vector(layer.k, f"K — Layer {layer_idx}", head_dim)
        if layer and layer.k is not None
        else _empty_fig("K (not captured)")
    )
    v_fig = (
        plot_qkv_vector(layer.v, f"V — Layer {layer_idx}", head_dim)
        if layer and layer.v is not None
        else _empty_fig("V (not captured)")
    )

    # ── Weight matrix plots ──
    q_w_fig = (
        plot_weight_matrix(layer.q_weight, f"Q Weight — Layer {layer_idx}")
        if layer and layer.q_weight is not None
        else _empty_fig("Q weight (not captured)")
    )
    k_w_fig = (
        plot_weight_matrix(layer.k_weight, f"K Weight — Layer {layer_idx}")
        if layer and layer.k_weight is not None
        else _empty_fig("K weight (not captured)")
    )
    v_w_fig = (
        plot_weight_matrix(layer.v_weight, f"V Weight — Layer {layer_idx}")
        if layer and layer.v_weight is not None
        else _empty_fig("V weight (not captured)")
    )

    # ── MLP plots ──
    gate_fig = (
        plot_mlp_activation(
            layer.mlp_gate_silu, f"Gate (SiLU) — Layer {layer_idx}"
        )
        if layer and layer.mlp_gate_silu is not None
        else _empty_fig("Gate (SiLU) (run Generate first)")
    )
    gate_raw_fig = (
        plot_histogram(
            layer.mlp_gate_raw, f"Gate Pre-Activation — Layer {layer_idx}"
        )
        if layer and layer.mlp_gate_raw is not None
        else _empty_fig("Gate raw (run Generate first)")
    )
    silu_fig = (
        plot_silu_scatter(
            layer.mlp_gate_raw, layer.mlp_gate_silu,
            f"SiLU Activation — Layer {layer_idx}",
        )
        if layer and layer.mlp_gate_raw is not None and layer.mlp_gate_silu is not None
        else _empty_fig("SiLU (run Generate first)")
    )
    mlp_out_fig = (
        plot_mlp_activation(
            layer.mlp_output, f"MLP Output — Layer {layer_idx}"
        )
        if layer and layer.mlp_output is not None
        else _empty_fig("MLP output (run Generate first)")
    )

    # ── RMSNorm plots ──
    norm_in_fig = (
        plot_rmsnorm_comparison(
            layer.residual_pre_attn,
            layer.input_layernorm_output,
            f"Pre-Attention RMSNorm — Layer {layer_idx}",
        )
        if layer and layer.residual_pre_attn is not None
        and layer.input_layernorm_output is not None
        else _empty_fig("RMSNorm (run Generate first)")
    )
    norm_out_fig = (
        plot_rmsnorm_comparison(
            layer.residual_post_attn,
            layer.post_attention_layernorm_output,
            f"Post-Attention RMSNorm — Layer {layer_idx}",
        )
        if layer and layer.residual_post_attn is not None
        and layer.post_attention_layernorm_output is not None
        else _empty_fig("RMSNorm (run Generate first)")
    )

    # ── Residual plot ──
    residual_fig = (
        plot_residual_delta(
            layer.residual_pre_attn,
            layer.residual_post_mlp,
            f"Residual Delta — Layer {layer_idx}",
        )
        if layer and layer.residual_pre_attn is not None
        and layer.residual_post_mlp is not None
        else _empty_fig("Residual (run Generate first)")
    )

    # ── Top-K and logits ──
    topk_fig = (
        plot_topk(step.topk_tokens, step.topk_probs)
        if step.topk_tokens
        else _empty_fig("Top-K (no data)")
    )
    logit_fig = (
        plot_logits_sampled(step.logits)
        if step.logits is not None
        else _empty_fig("Logits (no data)")
    )

    # ── Embedding ──
    embed_fig = (
        plot_vector_bar(step.input_embeds, "Token Embedding (last token)")
        if step.input_embeds is not None
        else _empty_fig("Embedding (no data)")
    )

    # Build output tuple
    return (
        full_text,                        # 0: text
        f"{num_steps} tokens",            # 1: counter
        gr.Slider(maximum=max(0, num_steps - 1), value=int(token_idx)),  # 2
        gr.Slider(maximum=max_layer, value=layer_idx),                    # 3
        gr.Slider(maximum=max_head, value=head_idx),                      # 4
        gr.Slider(maximum=max_layer, value=layer_idx),                    # 5
        gr.Slider(maximum=max_layer, value=layer_idx),                    # 6
        attn_fig,                # 7
        attn_scores_fig,         # 8
        q_fig,                   # 9
        k_fig,                   # 10
        v_fig,                   # 11
        q_w_fig,                 # 12
        k_w_fig,                 # 13
        v_w_fig,                 # 14
        gate_fig,                # 15
        gate_raw_fig,            # 16
        silu_fig,                # 17
        mlp_out_fig,             # 18
        norm_in_fig,             # 19
        norm_out_fig,            # 20
        residual_fig,            # 21
        topk_fig,                # 22
        logit_fig,               # 23
        embed_fig,               # 24
        gr.State(True),          # 25: state
    )


def _build_qkv_outputs(token_idx: int, layer_idx: int) -> Tuple:
    """Return only Q/K/V plot updates (3 plots)."""
    explorer = controller.explorer
    if explorer is None or not controller.ready:
        return _empty_fig("Q"), _empty_fig("K"), _empty_fig("V")

    step = explorer.get_step(int(token_idx))
    if step is None:
        return _empty_fig("Q"), _empty_fig("K"), _empty_fig("V")

    layer_idx = min(int(layer_idx), explorer.num_layers - 1)
    layer = step.layers.get(layer_idx)
    head_dim = explorer.head_dim

    q_fig = (
        plot_qkv_vector(layer.q, f"Q — Layer {layer_idx}", head_dim)
        if layer and layer.q is not None
        else _empty_fig("Q (not captured)")
    )
    k_fig = (
        plot_qkv_vector(layer.k, f"K — Layer {layer_idx}", head_dim)
        if layer and layer.k is not None
        else _empty_fig("K (not captured)")
    )
    v_fig = (
        plot_qkv_vector(layer.v, f"V — Layer {layer_idx}", head_dim)
        if layer and layer.v is not None
        else _empty_fig("V (not captured)")
    )
    return q_fig, k_fig, v_fig


def _build_mlp_outputs(token_idx: int, layer_idx: int) -> Tuple:
    """Return only MLP plot updates (3 plots)."""
    explorer = controller.explorer
    if explorer is None or not controller.ready:
        return _empty_fig("Gate"), _empty_fig("Gate raw"), _empty_fig("Output")

    step = explorer.get_step(int(token_idx))
    if step is None:
        return _empty_fig("Gate"), _empty_fig("Gate raw"), _empty_fig("Output")

    layer_idx = min(int(layer_idx), explorer.num_layers - 1)
    layer = step.layers.get(layer_idx)

    gate_fig = (
        plot_mlp_activation(layer.mlp_gate_silu, f"Gate (SiLU) — Layer {layer_idx}")
        if layer and layer.mlp_gate_silu is not None
        else _empty_fig("Gate (not captured)")
    )
    gate_raw_fig = (
        plot_histogram(layer.mlp_gate_raw, f"Gate Pre-Activation — Layer {layer_idx}")
        if layer and layer.mlp_gate_raw is not None
        else _empty_fig("Gate raw (not captured)")
    )
    out_fig = (
        plot_mlp_activation(layer.mlp_output, f"MLP Output — Layer {layer_idx}")
        if layer and layer.mlp_output is not None
        else _empty_fig("Output (not captured)")
    )
    return gate_fig, gate_raw_fig, out_fig


# ─────────────────────────────────────────────────────────────────────────────
#  UI Builder
# ─────────────────────────────────────────────────────────────────────────────


def create_ui():
    """Build the Gradio Blocks interface with expanded tab layout."""
    with gr.Blocks(
        title="Qwen2.5-0.5B Visual Explorer",
        theme=gr.themes.Soft(),
        css="""
        .gr-slider label { font-size: 0.9em !important; }
        .plot-container { min-height: 150px; }
        """,
    ) as ui:
        gr.Markdown(
            "# 🧠 Qwen2.5-0.5B Step-Through Explorer\n"
            "Inspect attention patterns, Q/K/V vectors, MLP activations, "
            "residual streams, and token probabilities — layer by layer, token by token."
        )

        # ── Input row ──
        with gr.Row():
            prompt_input = gr.Textbox(
                label="Prompt Input",
                placeholder="Enter a prompt…",
                value="The meaning of life is",
                lines=2,
                scale=3,
            )
            with gr.Column(scale=1, min_width=180):
                max_new_slider = gr.Slider(1, 100, value=20, step=1, label="Max new tokens")
                temp_slider = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")

        # ── Buttons ──
        with gr.Row():
            generate_btn = gr.Button("🚀 Generate", variant="primary", scale=2)
            step_btn = gr.Button("⏭️ One Step", scale=1)
            reset_btn = gr.Button("🔄 Reset", scale=1)

        # ── Counter ──
        token_counter = gr.Textbox(label="Token Counter", value="0 tokens", interactive=False)

        # ── Main layout: text left + tabs right ──
        with gr.Row():
            with gr.Column(scale=2):
                generated_text = gr.Textbox(label="Generated Text So Far", lines=10, interactive=False)
                token_selector = gr.Slider(0, 1, value=0, step=1, label="◀ Select Token Step ▶")

            with gr.Column(scale=4):
                with gr.Tabs():
                    # ── Attention tab ──
                    with gr.TabItem("🔍 Attention"):
                        with gr.Row():
                            attn_layer_slider = gr.Slider(0, 1, value=0, step=1, label="Layer", scale=1)
                            attn_head_slider = gr.Slider(0, 1, value=0, step=1, label="Head", scale=1)
                        with gr.Row():
                            attn_plot = gr.Plot(label="Attention (post-softmax)", scale=1)
                            attn_scores_plot = gr.Plot(label="Attention Scores (pre-softmax)", scale=1)

                    # ── QKV tab ──
                    with gr.TabItem("📊 Q / K / V"):
                        qkv_layer_slider = gr.Slider(0, 1, value=0, step=1, label="Layer")
                        with gr.Row():
                            q_plot = gr.Plot(label="Q")
                            k_plot = gr.Plot(label="K")
                            v_plot = gr.Plot(label="V")
                        gr.Markdown("**Weight Matrices**")
                        with gr.Row():
                            q_weight_plot = gr.Plot(label="Q Weight")
                            k_weight_plot = gr.Plot(label="K Weight")
                            v_weight_plot = gr.Plot(label="V Weight")

                    # ── MLP tab ──
                    with gr.TabItem("⚙️ MLP"):
                        mlp_layer_slider = gr.Slider(0, 1, value=0, step=1, label="Layer")
                        with gr.Row():
                            gate_plot = gr.Plot(label="Gate (SiLU)")
                            gate_raw_plot = gr.Plot(label="Gate Pre-Activation Histogram")
                        with gr.Row():
                            silu_plot = gr.Plot(label="SiLU Scatter (raw vs activated)")
                            mlp_out_plot = gr.Plot(label="MLP Output")

                    # ── Normalisation tab ──
                    with gr.TabItem("🧮 RMSNorm"):
                        with gr.Row():
                            norm_in_plot = gr.Plot(label="Pre-Attention Norm (input vs output)")
                            norm_out_plot = gr.Plot(label="Post-Attention Norm (input vs output)")

                    # ── Residual tab ──
                    with gr.TabItem("🔄 Residual"):
                        residual_plot = gr.Plot(label="Residual Stream Delta")

                    # ── Embedding tab ──
                    with gr.TabItem("🔤 Embedding"):
                        embed_plot = gr.Plot(label="Token Embedding")

        # ── Bottom row: probabilities ──
        with gr.Row():
            topk_plot = gr.Plot(label="Top-K Token Probabilities", scale=1)
            logit_plot = gr.Plot(label="Logit Distribution", scale=1)

        # ── Hidden state ──
        state = gr.State(False)

        # ── Output lists for convenience ──
        all_outputs = [
            generated_text, token_counter, token_selector,
            attn_layer_slider, attn_head_slider,
            qkv_layer_slider, mlp_layer_slider,
            attn_plot, attn_scores_plot,
            q_plot, k_plot, v_plot,
            q_weight_plot, k_weight_plot, v_weight_plot,
            gate_plot, gate_raw_plot, silu_plot, mlp_out_plot,
            norm_in_plot, norm_out_plot,
            residual_plot,
            topk_plot, logit_plot,
            embed_plot,
            state,
        ]

        # ── Wire events ──
        generate_btn.click(
            fn=on_generate,
            inputs=[prompt_input, max_new_slider, temp_slider],
            outputs=all_outputs,
        )

        step_btn.click(
            fn=on_step,
            inputs=[prompt_input, temp_slider],
            outputs=all_outputs,
        )

        token_selector.change(
            fn=on_select_token,
            inputs=[token_selector, attn_layer_slider, attn_head_slider],
            outputs=all_outputs,
        )

        attn_layer_slider.change(
            fn=on_change_layer,
            inputs=[token_selector, attn_layer_slider, attn_head_slider],
            outputs=all_outputs,
        )

        attn_head_slider.change(
            fn=on_change_layer,
            inputs=[token_selector, attn_layer_slider, attn_head_slider],
            outputs=all_outputs,
        )

        qkv_layer_slider.change(
            fn=on_change_qkv_layer,
            inputs=[token_selector, qkv_layer_slider],
            outputs=[q_plot, k_plot, v_plot],
        )

        mlp_layer_slider.change(
            fn=on_change_mlp_layer,
            inputs=[token_selector, mlp_layer_slider],
            outputs=[gate_plot, gate_raw_plot, mlp_out_plot],
        )

        reset_btn.click(
            fn=on_reset,
            inputs=[],
            outputs=all_outputs,
        )

    return ui
