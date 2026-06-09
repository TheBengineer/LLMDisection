"""
Plotly visualisation functions for all captured transformer internals.

Every function takes a numpy array (or None) and returns a Plotly Figure
suitable for embedding in Gradio.

Phase 1: image/plot generators for every component.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def _empty_fig(msg: str = "No data") -> go.Figure:
    """Small placeholder figure for missing data."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
    )
    fig.update_layout(height=150, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ── Attention ────────────────────────────────────────────────────────────────

def plot_attention(
    attn_probs: Optional[np.ndarray],
    token_labels: List[str],
    layer: int,
    head: int,
) -> go.Figure:
    """Post-softmax attention heatmap (last query → all keys)."""
    if attn_probs is None or attn_probs.size == 0:
        return _empty_fig("Attention (no data)")

    if head >= attn_probs.shape[0]:
        head = 0

    attn = attn_probs[head]  # [seq_len]

    fig = go.Figure(
        data=go.Heatmap(
            z=attn.reshape(1, -1),
            x=token_labels,
            y=[f"L{layer} H{head}"],
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            hovertemplate="Key: %{x}<br>Weight: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Attention — Layer {layer}, Head {head}",
        xaxis_title="Key tokens",
        height=160,
        margin=dict(l=20, r=20, t=40, b=50),
        xaxis=dict(tickangle=45),
    )
    return fig


def plot_attention_scores(
    scores: Optional[np.ndarray],
    token_labels: List[str],
    layer: int,
    head: int,
) -> go.Figure:
    """Pre-softmax attention scores heatmap (last query → all keys)."""
    if scores is None or scores.size == 0:
        return _empty_fig("Attention scores (no data)")

    if head >= scores.shape[0]:
        head = 0

    attn = scores[head]  # [seq_len]

    fig = go.Figure(
        data=go.Heatmap(
            z=attn.reshape(1, -1),
            x=token_labels,
            y=[f"L{layer} H{head}"],
            colorscale="RdBu",
            zmid=0,
            hovertemplate="Key: %{x}<br>Score: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Attention Scores (pre-softmax) — Layer {layer}, Head {head}",
        xaxis_title="Key tokens",
        height=160,
        margin=dict(l=20, r=20, t=40, b=50),
        xaxis=dict(tickangle=45),
    )
    return fig


# ── Top-K probabilities ──────────────────────────────────────────────────────

def plot_topk(tokens: List[str], probs: List[float]) -> go.Figure:
    """Horizontal bar chart of top-K token probabilities."""
    if not tokens:
        return _empty_fig("Top-K (no data)")

    colors = px.colors.sequential.Blues_r[: len(tokens)][::-1]

    fig = go.Figure(
        data=go.Bar(
            x=probs,
            y=tokens,
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top-K Next Token Predictions",
        xaxis_title="Probability",
        xaxis=dict(range=[0, 1], tickformat="%"),
        yaxis=dict(autorange="reversed"),
        height=220,
        margin=dict(l=20, r=60, t=40, b=20),
    )
    return fig


def plot_top_logits(
    logits: Optional[np.ndarray],
    vocab: Optional[List[str]] = None,
    k: int = 20,
) -> go.Figure:
    """Horizontal bar of top-K logits with token labels."""
    if logits is None or len(logits) == 0:
        return _empty_fig("Logits (no data)")

    top_k = min(k, len(logits))
    indices = np.argsort(logits)[-top_k:][::-1]
    top_vals = logits[indices]

    labels = (
        [vocab[i] if vocab and i < len(vocab) else f"#{i}" for i in indices]
        if vocab
        else [f"#{i}" for i in indices]
    )

    fig = go.Figure(
        data=go.Bar(
            x=top_vals,
            y=labels,
            orientation="h",
            marker_color=px.colors.sequential.Sunset[:top_k][::-1],
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Top-{top_k} Logits",
        xaxis_title="Logit value",
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=20, r=40, t=40, b=20),
    )
    return fig


# ── Q/K/V heatmap grids ──────────────────────────────────────────────────────

def plot_qkv_vector(
    vec: Optional[np.ndarray],
    title: str,
    head_dim: int = 64,
) -> go.Figure:
    """Show a Q/K/V vector as a reshaped head×dim heatmap."""
    if vec is None:
        return _empty_fig(title + " (no data)")

    hidden = vec.shape[0]
    num_heads = hidden // head_dim
    reshaped = vec.reshape(num_heads, head_dim)

    fig = go.Figure(
        data=go.Heatmap(
            z=reshaped,
            colorscale="RdBu",
            zmid=0,
            hovertemplate="Head %{y}<br>Dim %{x}<br>%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Dimension within head",
        yaxis_title="Head",
        height=280,
        margin=dict(l=20, r=20, t=40, b=50),
    )
    return fig


def plot_vector_bar(
    vec: Optional[np.ndarray],
    title: str,
    dim_labels: Optional[List[str]] = None,
) -> go.Figure:
    """Bar chart of a 1-D vector (e.g. embedding)."""
    if vec is None:
        return _empty_fig(title + " (no data)")

    n = min(len(vec), 500)  # cap for legibility
    xs = list(range(n)) if dim_labels is None else dim_labels[:n]

    fig = go.Figure(
        data=go.Bar(
            x=xs,
            y=vec[:n].tolist(),
            marker_color="steelblue",
            hovertemplate="Dim %{x}<br>%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Dimension",
        yaxis_title="Value",
        height=200,
        margin=dict(l=20, r=20, t=40, b=50),
    )
    return fig


# ── Weight matrices ──────────────────────────────────────────────────────────

def plot_weight_matrix(
    mat: Optional[np.ndarray],
    title: str,
    max_rows: int = 256,
    max_cols: int = 256,
) -> go.Figure:
    """2-D heatmap of a weight matrix (subsampled if large)."""
    if mat is None:
        return _empty_fig(title + " (no data)")

    # Subsample if too large
    r, c = mat.shape
    step_r = max(1, r // max_rows)
    step_c = max(1, c // max_cols)
    sub = mat[::step_r, ::step_c]

    fig = go.Figure(
        data=go.Heatmap(
            z=sub,
            colorscale="RdBu",
            zmid=0,
            hovertemplate="Row %{y}<br>Col %{x}<br>%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=f"Input dim (sampled every {step_c})" if step_c > 1 else "Input dim",
        yaxis_title=f"Output dim (sampled every {step_r})" if step_r > 1 else "Output dim",
        height=350,
        width=400,
        margin=dict(l=20, r=20, t=40, b=50),
    )
    return fig


# ── MLP activations ──────────────────────────────────────────────────────────

def plot_mlp_activation(
    vec: Optional[np.ndarray],
    title: str,
    max_dims: int = 300,
) -> go.Figure:
    """Line plot of MLP activation values (first max_dims dims)."""
    if vec is None:
        return _empty_fig(title + " (no data)")

    n = min(len(vec), max_dims)
    fig = go.Figure(
        data=go.Scatter(
            y=vec[:n].tolist(),
            mode="lines",
            line=dict(width=1.2, color="coral"),
            hovertemplate="Dim %{x}<br>%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=f"Dimension (first {n})",
        yaxis_title="Value",
        height=220,
        margin=dict(l=20, r=20, t=40, b=50),
    )
    return fig


def plot_histogram(
    vec: Optional[np.ndarray],
    title: str,
    bins: int = 50,
) -> go.Figure:
    """Histogram of activation values."""
    if vec is None or len(vec) == 0:
        return _empty_fig(title + " (no data)")

    fig = go.Figure(
        data=go.Histogram(
            x=vec.flatten().tolist(),
            nbinsx=bins,
            marker_color="mediumpurple",
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Count",
        height=200,
        margin=dict(l=20, r=20, t=40, b=50),
        bargap=0.05,
    )
    return fig


def plot_silu_scatter(
    gate_raw: Optional[np.ndarray],
    gate_silu: Optional[np.ndarray],
    title: str = "SiLU Activation",
    max_dims: int = 1000,
) -> go.Figure:
    """Scatter: raw gate vs. SiLU-activated gate, coloured by up-projection."""
    if gate_raw is None or gate_silu is None:
        return _empty_fig("SiLU (no data)")

    n = min(len(gate_raw), max_dims)
    fig = go.Figure(
        data=go.Scattergl(
            x=gate_raw[:n],
            y=gate_silu[:n],
            mode="markers",
            marker=dict(size=3, color="coral", opacity=0.5),
            hovertemplate="Raw: %{x:.3f}<br>SiLU: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Gate input (raw)",
        yaxis_title="Gate after SiLU",
        height=280,
        margin=dict(l=20, r=20, t=40, b=50),
    )
    return fig


# ── Logit distribution ───────────────────────────────────────────────────────

def plot_logits_sampled(
    logits: Optional[np.ndarray],
    sample_rate: int = 50,
) -> go.Figure:
    """Down-sampled scatter of the full logit distribution."""
    if logits is None or len(logits) == 0:
        return _empty_fig("Logits (no data)")

    step = max(1, len(logits) // 150)
    indices = np.arange(0, len(logits), step)

    fig = go.Figure(
        data=go.Scattergl(
            x=indices,
            y=logits[indices],
            mode="markers",
            marker=dict(size=2.5, color="seagreen", opacity=0.4),
            hovertemplate="Vocab %{x}<br>%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Logit Distribution (sampled every {step})",
        xaxis_title="Vocabulary index",
        yaxis_title="Logit value",
        height=180,
        margin=dict(l=20, r=20, t=40, b=50),
        showlegend=False,
    )
    return fig


# ── Residual stream & layer contributions ────────────────────────────────────

def plot_residual_delta(
    before: Optional[np.ndarray],
    after: Optional[np.ndarray],
    title: str = "Residual Stream",
    max_dims: int = 300,
) -> go.Figure:
    """Overlay line plot of residual stream before and after an operation."""
    if before is None or after is None:
        return _empty_fig(f"{title} (no data)")

    n = min(len(before), max_dims)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=before[:n].tolist(),
            mode="lines",
            name="Before",
            line=dict(width=1.2, color="steelblue"),
            hovertemplate="Dim %{x}<br>Before: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=after[:n].tolist(),
            mode="lines",
            name="After",
            line=dict(width=1.2, color="coral"),
            hovertemplate="Dim %{x}<br>After: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=f"Dimension (first {n})",
        yaxis_title="Value",
        height=220,
        margin=dict(l=20, r=20, t=40, b=50),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    return fig


def plot_layer_contributions(
    contribs: Optional[List[float]],
    title: str = "Layer Contribution (L2 delta)",
) -> go.Figure:
    """Bar chart showing how much each layer changed the residual."""
    if contribs is None or len(contribs) == 0:
        return _empty_fig(title + " (no data)")

    fig = go.Figure(
        data=go.Bar(
            x=list(range(len(contribs))),
            y=contribs,
            marker_color=px.colors.sequential.Viridis[
                : len(contribs)
            ],
            hovertemplate="Layer %{x}<br>Δ: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="L2 norm of delta",
        height=220,
        margin=dict(l=20, r=20, t=40, b=50),
    )
    return fig


# ── Embedding visualisation ──────────────────────────────────────────────────

def plot_embedding_slice(
    emb_matrix: Optional[np.ndarray],
    token_id: Optional[int] = None,
    vocab: Optional[List[str]] = None,
    top_k: int = 20,
) -> go.Figure:
    """
    Heatmap of a small slice of the embedding table.
    If token_id is given, highlights that row and shows nearby tokens.
    """
    if emb_matrix is None:
        return _empty_fig("Embedding matrix (no data)")

    # Pick a slice around the current token
    if token_id is not None and 0 <= token_id < emb_matrix.shape[0]:
        half = top_k // 2
        start = max(0, token_id - half)
        end = min(emb_matrix.shape[0], token_id + half + 1)
    else:
        start = 0
        end = min(emb_matrix.shape[0], top_k)

    slice_mat = emb_matrix[start:end, :64]  # first 64 dims only for display

    y_labels = (
        [vocab[i] if vocab and i < len(vocab) else f"#{i}" for i in range(start, end)]
        if vocab
        else [f"#{i}" for i in range(start, end)]
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=slice_mat,
            colorscale="RdBu",
            zmid=0,
            y=y_labels if len(y_labels) == slice_mat.shape[0] else list(range(start, end)),
            hovertemplate="Token %{y}<br>Dim %{x}<br>%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Embedding slice (tokens {start}–{end - 1}, dims 0–63)",
        xaxis_title="Embedding dim (first 64)",
        yaxis_title="Token",
        height=350,
        margin=dict(l=20, r=20, t=40, b=50),
    )

    # Highlight the current token row
    if token_id is not None and start <= token_id < end:
        fig.add_hline(
            y=token_id - start,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"← token #{token_id}",
        )

    return fig


# ── RoPE visualisation ───────────────────────────────────────────────────────

def plot_rope_rotation(
    pos: int,
    head_dim: int = 64,
    num_pairs: int = 8,
) -> go.Figure:
    """Polar plot showing how RoPE rotates each pair of dimensions at a position."""
    # Compute rotation angles for this position
    theta = 10000.0
    dims = min(head_dim // 2, num_pairs)
    angles = [pos / (theta ** (2 * i / head_dim)) for i in range(dims)]

    fig = go.Figure()
    for i, angle in enumerate(angles):
        fig.add_trace(
            go.Scatterpolar(
                r=[0, 1],
                theta=[0, angle * 180 / np.pi],
                mode="lines+markers",
                name=f"Pair {i}",
                line=dict(width=1.5),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title=f"RoPE Rotation at Position {pos}",
        showlegend=True,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.2]),
            angularaxis=dict(tickfont_size=8),
        ),
    )
    return fig


# ── RMSNorm comparison ───────────────────────────────────────────────────────

def plot_rmsnorm_comparison(
    inp: Optional[np.ndarray],
    out: Optional[np.ndarray],
    title: str = "RMSNorm",
    max_dims: int = 300,
) -> go.Figure:
    """Overlay bar/line of input vs. output of a RMSNorm layer."""
    if inp is None or out is None:
        return _empty_fig(title + " (no data)")

    n = min(len(inp), max_dims)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=inp[:n].tolist(),
            mode="lines",
            name="Input",
            line=dict(width=1.2, color="steelblue"),
            hovertemplate="Dim %{x}<br>In: %{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=out[:n].tolist(),
            mode="lines",
            name="Output",
            line=dict(width=1.2, color="coral"),
            hovertemplate="Dim %{x}<br>Out: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=f"Dimension (first {n})",
        yaxis_title="Value",
        height=220,
        margin=dict(l=20, r=20, t=40, b=50),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    return fig
