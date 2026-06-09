#!/usr/bin/env python3
"""
Qwen2.5-0.5B Visual Step-Through Explorer
===========================================
Interactive Gradio UI for inspecting transformer internals layer by layer.
Lets you walk through each generated token and inspect attention patterns,
Q/K/V vectors, MLP activations, and token probabilities.

Usage:
    pip install torch transformers gradio plotly
    python app.py
    # Open http://localhost:7860
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import plotly.graph_objects as go
import plotly.express as px
import warnings
import argparse
import time

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
#  Data structures for captured states
# ───────────────────────────────────────────────

@dataclass
class LayerSnapshot:
    """All activations captured in one layer at one generation step."""
    layer_idx: int

    # Attention projections (before reshape into heads)
    q: Optional[np.ndarray] = None       # [hidden_size]
    k: Optional[np.ndarray] = None       # [hidden_size]
    v: Optional[np.ndarray] = None       # [hidden_size]

    # Attention probabilities for the LAST query position
    attn_probs: Optional[np.ndarray] = None  # [num_heads, seq_len]

    # Attention output (after o_proj)
    attn_output: Optional[np.ndarray] = None  # [hidden_size]

    # MLP internals
    mlp_gate: Optional[np.ndarray] = None   # [intermediate_size] (after activation)
    mlp_up: Optional[np.ndarray] = None     # [intermediate_size]
    mlp_down_input: Optional[np.ndarray] = None  # [intermediate_size] (input to down_proj)
    mlp_output: Optional[np.ndarray] = None     # [hidden_size]

    # Residual stream snapshots
    residual_pre_attn: Optional[np.ndarray] = None
    residual_post_attn: Optional[np.ndarray] = None
    residual_post_mlp: Optional[np.ndarray] = None


@dataclass
class StepSnapshot:
    """Full snapshot of one generation step (one new token)."""
    token_id: int
    token_str: str
    position: int
    logits: Optional[np.ndarray] = None
    probs: Optional[np.ndarray] = None
    topk_indices: List[int] = field(default_factory=list)
    topk_tokens: List[str] = field(default_factory=list)
    topk_probs: List[float] = field(default_factory=list)
    layers: Dict[int, LayerSnapshot] = field(default_factory=dict)
    # Past key-values for continuing generation (kept as reference, not serialized)
    _past_key_values: Any = None


# ───────────────────────────────────────────────
#  Engine
# ───────────────────────────────────────────────

class QwenExplorer:
    """Loads the model, manages hooks, runs step-by-step generation, and caches states."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")

        # State
        self.steps: List[StepSnapshot] = []
        self.prompt_ids: Optional[torch.Tensor] = None
        self.full_sequence: Optional[torch.Tensor] = None
        self.prompt_length = 0

        # Hook storage
        self._handles = []
        self._capturing = False
        self._current_mlp: Dict[int, Dict[str, np.ndarray]] = {}
        self._current_qkv: Dict[int, Dict[str, np.ndarray]] = {}
        self._current_residuals: Dict[int, Dict[str, np.ndarray]] = {}

        # Model dimensions (filled by load())
        self.num_layers = 0
        self.num_heads = 0
        self.head_dim = 0
        self.hidden_size = 0
        self.intermediate_size = 0

    # ── Loading ──────────────────────────────────

    def load(self):
        """Download / load the model and tokenizer, register hooks."""
        print(f"[*] Loading {self.model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
            device_map="cpu",
            attn_implementation="eager",
        )
        self.model.eval()

        cfg = self.model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = cfg.intermediate_size

        print(f"    {self.num_layers} layers, {self.num_heads} heads, "
              f"hidden={self.hidden_size}, intermediate={self.intermediate_size}")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._register_all_hooks()
        print("[✓] Model ready.\n")

    def _register_all_hooks(self):
        """Register forward hooks on every layer for Q, K, V, MLP, and residuals."""
        for idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            mlp = layer.mlp

            # ── Q, K, V projections ──
            self._handles.append(
                attn.q_proj.register_forward_hook(self._make_qkv_hook(idx, 'q'))
            )
            self._handles.append(
                attn.k_proj.register_forward_hook(self._make_qkv_hook(idx, 'k'))
            )
            self._handles.append(
                attn.v_proj.register_forward_hook(self._make_qkv_hook(idx, 'v'))
            )

            # ── Attention output projection ──
            self._handles.append(
                attn.o_proj.register_forward_hook(self._make_attn_out_hook(idx))
            )

            # ── MLP projections ──
            self._handles.append(
                mlp.gate_proj.register_forward_hook(self._make_mlp_hook(idx, 'gate'))
            )
            self._handles.append(
                mlp.up_proj.register_forward_hook(self._make_mlp_hook(idx, 'up'))
            )
            # down_proj: capture both input (after activation*gate) and output
            self._handles.append(
                mlp.down_proj.register_forward_hook(self._make_mlp_down_hook(idx))
            )

            # ── Residual stream (layer input) ──
            # input_layernorm receives the pre-attention residual
            self._handles.append(
                layer.input_layernorm.register_forward_hook(
                    self._make_residual_hook(idx, 'pre_attn')
                )
            )
            # post_attention_layernorm receives the post-attention residual
            self._handles.append(
                layer.post_attention_layernorm.register_forward_hook(
                    self._make_residual_hook(idx, 'post_attn')
                )
            )

    def _make_qkv_hook(self, layer_idx: int, name: str):
        """Capture Q/K/V projection outputs (last token only)."""
        def hook(module, inp, out):
            if not self._capturing:
                return
            # out shape: [batch, seq_len, hidden]
            vec = out[0, -1].detach().cpu().numpy()  # last position
            self._current_qkv.setdefault(layer_idx, {})[name] = vec
        return hook

    def _make_attn_out_hook(self, layer_idx: int):
        """Capture attention output (after o_proj)."""
        def hook(module, inp, out):
            if not self._capturing:
                return
            vec = out[0, -1].detach().cpu().numpy()
            self._current_mlp.setdefault(layer_idx, {})['attn_out'] = vec
        return hook

    def _make_mlp_hook(self, layer_idx: int, name: str):
        """Capture MLP gate/up projection outputs."""
        def hook(module, inp, out):
            if not self._capturing:
                return
            vec = out[0, -1].detach().cpu().numpy()
            self._current_mlp.setdefault(layer_idx, {})[name] = vec
        return hook

    def _make_mlp_down_hook(self, layer_idx: int):
        """Capture MLP down_proj input and output."""
        def hook(module, inp, out):
            if not self._capturing:
                return
            # inp[0] is the activated hidden state [batch, seq_len, intermediate]
            down_in = inp[0][0, -1].detach().cpu().numpy()
            down_out = out[0, -1].detach().cpu().numpy()
            self._current_mlp.setdefault(layer_idx, {})['down_in'] = down_in
            self._current_mlp.setdefault(layer_idx, {})['down'] = down_out
        return hook

    def _make_residual_hook(self, layer_idx: int, name: str):
        """Capture residual stream at specific points."""
        def hook(module, inp, out):
            if not self._capturing:
                return
            # The input to layernorm is the residual stream value
            # inp[0] shape: [batch, seq_len, hidden]
            resid = inp[0][0, -1].detach().cpu().numpy()
            self._current_residuals.setdefault(layer_idx, {})[name] = resid
        return hook

    # ── State management ─────────────────────────

    def reset(self):
        """Clear all cached generation state."""
        self.steps = []
        self.prompt_ids = None
        self.full_sequence = None
        self.prompt_length = 0

    # ── Generation ───────────────────────────────

    @torch.no_grad()
    def generate_full(self, prompt: str, max_new_tokens: int = 20,
                      temperature: float = 0.8) -> List[str]:
        """Auto-generate a full sequence, capturing every step."""
        self.reset()

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        self.prompt_ids = inputs.input_ids
        self.full_sequence = self.prompt_ids.clone()
        self.prompt_length = self.prompt_ids.shape[1]

        # ── First forward pass (prompt) ──
        self._capturing = True
        self._current_mlp = {}
        self._current_qkv = {}
        self._current_residuals = {}

        outputs = self.model(
            input_ids=self.full_sequence,
            output_attentions=True,
            use_cache=True,
        )

        # Store the first step (the prompt's last position state)
        # This step represents "before any new token is generated"
        self._store_step(outputs, is_prompt=True)
        past_kv = outputs.past_key_values

        self._capturing = False

        # ── Generate new tokens ──
        tokens_out = []
        for step_i in range(max_new_tokens):
            self._capturing = True
            self._current_mlp = {}
            self._current_qkv = {}
            self._current_residuals = {}

            last_token = self.full_sequence[:, -1:]

            outputs = self.model(
                input_ids=last_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
            )

            self._capturing = False

            # Store step
            step = self._store_step(outputs, is_prompt=False)
            step._past_key_values = outputs.past_key_values  # for continued stepping
            past_kv = outputs.past_key_values

            # Sample next token
            logits = outputs.logits[:, -1, :]
            next_id = self._sample_token(logits, temperature)
            self.full_sequence = torch.cat([self.full_sequence, next_id], dim=-1)

            token_str = self.tokenizer.decode(next_id[0])
            tokens_out.append(token_str)

            # Update the step's token info (it was stored with the *input* token)
            step.token_id = int(next_id[0, 0].item())
            step.token_str = token_str

            if next_id.item() == self.tokenizer.eos_token_id:
                break

        self._capturing = False
        return tokens_out

    @torch.no_grad()
    def step_forward(self, temperature: float = 0.8) -> Tuple[bool, str]:
        """Generate exactly one more token. Returns (is_finished, token_str)."""
        if self.full_sequence is None or self.prompt_ids is None:
            raise RuntimeError("Call generate_full() first, or supply a prompt.")

        past_kv = None
        if len(self.steps) > 0 and hasattr(self.steps[-1], '_past_key_values'):
            past_kv = self.steps[-1]._past_key_values

        self._capturing = True
        self._current_mlp = {}
        self._current_qkv = {}
        self._current_residuals = {}

        last_token = self.full_sequence[:, -1:]

        outputs = self.model(
            input_ids=last_token,
            past_key_values=past_kv,
            use_cache=True,
            output_attentions=True,
        )

        self._capturing = False

        step = self._store_step(outputs, is_prompt=False)
        step._past_key_values = outputs.past_key_values

        # Sample
        logits = outputs.logits[:, -1, :]
        next_id = self._sample_token(logits, temperature)
        self.full_sequence = torch.cat([self.full_sequence, next_id], dim=-1)

        token_str = self.tokenizer.decode(next_id[0])
        step.token_id = int(next_id[0, 0].item())
        step.token_str = token_str

        finished = next_id.item() == self.tokenizer.eos_token_id
        return finished, token_str

    def _store_step(self, outputs, is_prompt: bool = False) -> StepSnapshot:
        """Build a StepSnapshot from model outputs and current hook caches."""
        # Logits and probabilities
        logits = outputs.logits[0, -1, :].detach().cpu().numpy()
        probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()

        # Top-5
        top_indices = np.argsort(probs)[-5:][::-1]
        top_probs = probs[top_indices].tolist()
        top_tokens = [self.tokenizer.decode([int(idx)]) for idx in top_indices]

        step = StepSnapshot(
            token_id=0,   # filled in later for new tokens
            token_str="",
            position=len(self.steps),
            logits=logits,
            probs=probs,
            topk_indices=top_indices.tolist(),
            topk_tokens=top_tokens,
            topk_probs=top_probs,
        )

        # Per-layer data
        for layer_idx in range(self.num_layers):
            layer_snap = LayerSnapshot(layer_idx=layer_idx)

            # QKV from hooks
            if layer_idx in self._current_qkv:
                qkv = self._current_qkv[layer_idx]
                layer_snap.q = qkv.get('q')
                layer_snap.k = qkv.get('k')
                layer_snap.v = qkv.get('v')

            # MLP from hooks
            if layer_idx in self._current_mlp:
                mlp = self._current_mlp[layer_idx]
                layer_snap.mlp_gate = mlp.get('gate')
                layer_snap.mlp_up = mlp.get('up')
                layer_snap.mlp_down_input = mlp.get('down_in')
                layer_snap.mlp_output = mlp.get('down')
                layer_snap.attn_output = mlp.get('attn_out')

            # Residuals from hooks
            if layer_idx in self._current_residuals:
                res = self._current_residuals[layer_idx]
                layer_snap.residual_pre_attn = res.get('pre_attn')
                layer_snap.residual_post_attn = res.get('post_attn')

            # Attention probabilities from output_attentions
            if outputs.attentions is not None and layer_idx < len(outputs.attentions):
                attn = outputs.attentions[layer_idx]  # [batch, heads, q_len, k_len]
                # For the last query position: attn[0, :, -1, :] -> [heads, seq_len]
                layer_snap.attn_probs = attn[0, :, -1, :].detach().cpu().numpy()

            step.layers[layer_idx] = layer_snap

        self.steps.append(step)
        return step

    def _sample_token(self, logits: torch.Tensor,
                      temperature: float = 0.8) -> torch.Tensor:
        """Sample a single token from logits."""
        if temperature < 1e-6:
            return torch.argmax(logits, dim=-1, keepdim=True)
        scaled = logits / temperature
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # ── Accessors ────────────────────────────────

    def num_steps(self) -> int:
        return len(self.steps)

    def get_step(self, idx: int) -> Optional[StepSnapshot]:
        if 0 <= idx < len(self.steps):
            return self.steps[idx]
        return None

    def get_full_text(self) -> str:
        if self.full_sequence is None:
            return ""
        return self.tokenizer.decode(self.full_sequence[0])

    def get_token_labels(self) -> List[str]:
        """Return a list of token strings for the full sequence."""
        if self.full_sequence is None:
            return []
        ids = self.full_sequence[0].tolist()
        # Use the tokenizer's batch_decode to preserve spaces correctly
        return [self.tokenizer.decode([tid]) for tid in ids]

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


# ───────────────────────────────────────────────
#  Visualization helpers (Plotly)
# ───────────────────────────────────────────────

def plot_attention(attn_probs: np.ndarray, token_labels: List[str],
                   layer: int, head: int) -> go.Figure:
    """Attention heatmap: how the last query token attends to all keys."""
    if attn_probs is None or attn_probs.size == 0:
        return _empty_fig("Attention (no data)")

    if head >= attn_probs.shape[0]:
        head = 0

    attn = attn_probs[head]  # [seq_len]

    fig = go.Figure(data=go.Heatmap(
        z=attn.reshape(1, -1),
        x=token_labels,
        y=[f"L{layer} H{head}"],
        colorscale='Viridis',
        zmin=0, zmax=1,
        hovertemplate='Key: %{x}<br>Weight: %{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        title=f"Attention — Layer {layer}, Head {head}",
        xaxis_title="Key tokens",
        height=160,
        margin=dict(l=20, r=20, t=40, b=50),
        xaxis=dict(tickangle=45),
    )
    return fig


def plot_topk(tokens: List[str], probs: List[float]) -> go.Figure:
    """Horizontal bar chart of top-5 token probabilities."""
    if not tokens:
        return _empty_fig("Top-5 (no data)")

    colors = px.colors.sequential.Blues_r[:len(tokens)][::-1]

    fig = go.Figure(data=go.Bar(
        x=probs,
        y=tokens,
        orientation='h',
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition='outside',
        hovertemplate='%{y}: %{x:.2%}<extra></extra>',
    ))
    fig.update_layout(
        title="Top-5 Next Token Predictions",
        xaxis_title="Probability",
        xaxis=dict(range=[0, 1], tickformat='%'),
        yaxis=dict(autorange="reversed"),
        height=220,
        margin=dict(l=20, r=60, t=40, b=20),
    )
    return fig


def plot_qkv_vector(vec: np.ndarray, title: str,
                    num_heads: int = 12) -> go.Figure:
    """Show a Q/K/V vector as a reshaped head×dim heatmap."""
    if vec is None:
        return _empty_fig(title + " (no data)")

    hidden = vec.shape[0]
    head_dim = hidden // num_heads
    reshaped = vec.reshape(num_heads, head_dim)

    fig = go.Figure(data=go.Heatmap(
        z=reshaped,
        colorscale='RdBu',
        zmid=0,
        hovertemplate='Head %{y}<br>Dim %{x}<br>%{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Dimension within head",
        yaxis_title="Head",
        height=280,
        margin=dict(l=20, r=20, t=40, b=50),
    )
    return fig


def plot_mlp_activation(vec: np.ndarray, title: str,
                        max_dims: int = 300) -> go.Figure:
    """Line plot of MLP activation values (shows first max_dims dims)."""
    if vec is None:
        return _empty_fig(title + " (no data)")

    n = min(len(vec), max_dims)
    trace = go.Scatter(
        y=vec[:n].tolist(),
        mode='lines',
        line=dict(width=1.2, color='coral'),
        hovertemplate='Dim %{x}<br>%{y:.3f}<extra></extra>',
    )
    fig = go.Figure(data=trace)
    fig.update_layout(
        title=title,
        xaxis_title=f"Dimension (first {n})",
        yaxis_title="Value",
        height=220,
        margin=dict(l=20, r=20, t=40, b=50),
    )
    return fig


def plot_logits_sampled(logits: np.ndarray, vocab_size: int = None,
                        sample_rate: int = 50) -> go.Figure:
    """Scatter plot of the logit distribution (downsampled)."""
    if logits is None or len(logits) == 0:
        return _empty_fig("Logits (no data)")

    step = max(1, len(logits) // 150)
    indices = np.arange(0, len(logits), step)

    fig = go.Figure(data=go.Scattergl(
        x=indices,
        y=logits[indices],
        mode='markers',
        marker=dict(size=2.5, color='seagreen', opacity=0.4),
        hovertemplate='Vocab %{x}<br>%{y:.2f}<extra></extra>',
    ))
    fig.update_layout(
        title=f"Logit Distribution (sampled every {step})",
        xaxis_title="Vocabulary index",
        yaxis_title="Logit value",
        height=180,
        margin=dict(l=20, r=20, t=40, b=50),
        showlegend=False,
    )
    return fig


def _empty_fig(msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False)
    fig.update_layout(height=150, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ───────────────────────────────────────────────
#  Gradio UI
# ───────────────────────────────────────────────

class UIController:
    """Holds state between Gradio callbacks."""

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


# ── Callbacks ────────────────────────────────────

def on_generate(prompt: str, max_new: int, temperature: float):
    """Full auto-generation."""
    _ensure_loaded()
    explorer = controller.explorer

    try:
        explorer.generate_full(prompt, max_new_tokens=max_new,
                               temperature=temperature)
    except Exception as e:
        return [f"Error: {e}"] + [gr.update()] * 14 + [gr.State(False)]

    return _build_all_outputs(explorer.num_steps() - 1)


def on_step(prompt: str, temperature: float):
    """Generate one more token."""
    _ensure_loaded()
    explorer = controller.explorer

    try:
        if explorer.num_steps() == 0:
            # First step: start generation with just 1 new token
            explorer.generate_full(prompt, max_new_tokens=1,
                                   temperature=temperature)
        else:
            explorer.step_forward(temperature=temperature)
    except Exception as e:
        return [f"Error: {e}"] + [gr.update()] * 14 + [gr.State(False)]

    return _build_all_outputs(explorer.num_steps() - 1)


def on_select_token(token_idx: int, layer_idx: int, head_idx: int):
    """User selected a different token position."""
    _ensure_loaded()
    return _build_all_outputs(token_idx, layer_idx, head_idx)


def on_change_layer(token_idx: int, layer_idx: int, head_idx: int):
    """Layer or head slider changed."""
    return _build_all_outputs(token_idx, layer_idx, head_idx)


def on_change_qkv_layer(token_idx: int, layer_idx: int):
    """QKV tab layer changed."""
    return _build_qkv_outputs(token_idx, layer_idx)


def on_change_mlp_layer(token_idx: int, layer_idx: int):
    """MLP tab layer changed."""
    return _build_mlp_outputs(token_idx, layer_idx)


def on_reset():
    """Reset everything."""
    controller.explorer = None
    controller.ready = False
    return (
        "The meaning of life is",  # prompt
        "",                        # generated text
        "0 tokens",                # counter
        gr.Slider(value=0, maximum=0),  # token selector
        gr.Slider(value=0, maximum=0),  # layer slider
        gr.Slider(value=0, maximum=0),  # head slider
        gr.Slider(value=0, maximum=0),  # qkv layer
        gr.Slider(value=0, maximum=0),  # mlp layer
    ) + tuple([_empty_fig("Press Generate")] * 9) + (gr.State(False),)


def _build_all_outputs(token_idx: int, layer_idx: int = 0, head_idx: int = 0):
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

    # Clamp indices
    layer_idx = min(int(layer_idx), max_layer)
    head_idx = min(int(head_idx), max_head)

    layer = step.layers.get(layer_idx)

    # Attention
    attn_fig = (plot_attention(layer.attn_probs, token_labels, layer_idx, head_idx)
                if layer and layer.attn_probs is not None
                else _empty_fig("Attention (run Generate first)"))

    # QKV (show Q for this layer)
    q_fig = k_fig = v_fig = _empty_fig("Q/K/V (run Generate first)")
    if layer:
        q_fig = plot_qkv_vector(layer.q, f"Q — Layer {layer_idx}", explorer.num_heads) if layer.q is not None else _empty_fig("Q (not captured)")
        k_fig = plot_qkv_vector(layer.k, f"K — Layer {layer_idx}", explorer.num_heads) if layer.k is not None else _empty_fig("K (not captured)")
        v_fig = plot_qkv_vector(layer.v, f"V — Layer {layer_idx}", explorer.num_heads) if layer.v is not None else _empty_fig("V (not captured)")

    # MLP
    gate_fig = (plot_mlp_activation(layer.mlp_gate, f"Gate — Layer {layer_idx}")
                if layer and layer.mlp_gate is not None
                else _empty_fig("MLP gate (run Generate first)"))
    up_fig = (plot_mlp_activation(layer.mlp_up, f"Up — Layer {layer_idx}")
              if layer and layer.mlp_up is not None
              else _empty_fig("MLP up (run Generate first)"))
    out_fig = (plot_mlp_activation(layer.mlp_output, f"MLP Output — Layer {layer_idx}")
               if layer and layer.mlp_output is not None
               else _empty_fig("MLP output (run Generate first)"))

    # Top-K and logits
    topk_fig = plot_topk(step.topk_tokens, step.topk_probs) if step.topk_tokens else _empty_fig("Top-5 (no data)")
    logit_fig = plot_logits_sampled(step.logits) if step.logits is not None else _empty_fig("Logits (no data)")

    return (
        full_text,
        f"{num_steps} tokens (prompt + {num_steps - 1} generated)" if num_steps > 0 else "0 tokens",
        gr.Slider(maximum=max(0, num_steps - 1), value=int(token_idx)),
        gr.Slider(maximum=max_layer, value=layer_idx),
        gr.Slider(maximum=max_head, value=head_idx),
        gr.Slider(maximum=max_layer, value=layer_idx),
        gr.Slider(maximum=max_layer, value=layer_idx),
        attn_fig, q_fig, k_fig, v_fig,
        gate_fig, up_fig, out_fig,
        topk_fig, logit_fig,
        gr.State(True),
    )


def _build_qkv_outputs(token_idx: int, layer_idx: int):
    """Return only Q/K/V plot updates."""
    explorer = controller.explorer
    if explorer is None or not controller.ready:
        return _empty_fig("Q"), _empty_fig("K"), _empty_fig("V")

    step = explorer.get_step(int(token_idx))
    if step is None:
        return _empty_fig("Q"), _empty_fig("K"), _empty_fig("V")

    layer_idx = min(int(layer_idx), explorer.num_layers - 1)
    layer = step.layers.get(layer_idx)

    q_fig = plot_qkv_vector(layer.q, f"Q — Layer {layer_idx}", explorer.num_heads) if layer and layer.q is not None else _empty_fig("Q (not captured)")
    k_fig = plot_qkv_vector(layer.k, f"K — Layer {layer_idx}", explorer.num_heads) if layer and layer.k is not None else _empty_fig("K (not captured)")
    v_fig = plot_qkv_vector(layer.v, f"V — Layer {layer_idx}", explorer.num_heads) if layer and layer.v is not None else _empty_fig("V (not captured)")
    return q_fig, k_fig, v_fig


def _build_mlp_outputs(token_idx: int, layer_idx: int):
    """Return only MLP plot updates."""
    explorer = controller.explorer
    if explorer is None or not controller.ready:
        return _empty_fig("Gate"), _empty_fig("Up"), _empty_fig("Output")

    step = explorer.get_step(int(token_idx))
    if step is None:
        return _empty_fig("Gate"), _empty_fig("Up"), _empty_fig("Output")

    layer_idx = min(int(layer_idx), explorer.num_layers - 1)
    layer = step.layers.get(layer_idx)

    gate_fig = plot_mlp_activation(layer.mlp_gate, f"Gate — Layer {layer_idx}") if layer and layer.mlp_gate is not None else _empty_fig("Gate (not captured)")
    up_fig = plot_mlp_activation(layer.mlp_up, f"Up — Layer {layer_idx}") if layer and layer.mlp_up is not None else _empty_fig("Up (not captured)")
    out_fig = plot_mlp_activation(layer.mlp_output, f"MLP Output — Layer {layer_idx}") if layer and layer.mlp_output is not None else _empty_fig("Output (not captured)")
    return gate_fig, up_fig, out_fig


def _default_outputs():
    """Default outputs when no generation has happened."""
    empty = _empty_fig("Press Generate")
    return (
        "No generation yet",
        "0 tokens",
        gr.Slider(value=0, maximum=0),
        gr.Slider(value=0, maximum=0),
        gr.Slider(value=0, maximum=0),
        gr.Slider(value=0, maximum=0),
        gr.Slider(value=0, maximum=0),
    ) + tuple([empty] * 9) + (gr.State(False),)


# ── Build UI ─────────────────────────────────────

def create_ui():
    """Build the Gradio Blocks interface."""
    with gr.Blocks(
        title="Qwen2.5-0.5B Visual Explorer",
        theme=gr.themes.Soft(),
        css="""
        .gr-slider label { font-size: 0.9em !important; }
        """,
    ) as ui:
        gr.Markdown(
            "# 🧠 Qwen2.5-0.5B Step-Through Explorer\n"
            "Inspect attention patterns, Q/K/V vectors, MLP activations, "
            "and token probabilities — layer by layer, token by token."
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
                max_new_slider = gr.Slider(
                    1, 100, value=20, step=1, label="Max new tokens"
                )
                temp_slider = gr.Slider(
                    0.1, 2.0, value=0.8, step=0.05, label="Temperature"
                )

        # ── Buttons ──
        with gr.Row():
            generate_btn = gr.Button("🚀 Generate", variant="primary", scale=2)
            step_btn = gr.Button("⏭️ One Step", scale=1)
            reset_btn = gr.Button("🔄 Reset", scale=1)

        # ── Token counter ──
        token_counter = gr.Textbox(
            label="Token Counter", value="0 tokens", interactive=False
        )

        # ── Main content ──
        with gr.Row():
            # Left column: generated text + token selector
            with gr.Column(scale=2):
                generated_text = gr.Textbox(
                    label="Generated Text So Far",
                    lines=10,
                    interactive=False,
                )
                token_selector = gr.Slider(
                    0, 0, value=0, step=1,
                    label="◀ Select Token Step ▶",
                )

            # Right column: tabs for internals
            with gr.Column(scale=3):
                with gr.Tabs():
                    # ── Attention tab ──
                    with gr.TabItem("🔍 Attention"):
                        with gr.Row():
                            attn_layer_slider = gr.Slider(
                                0, 0, value=0, step=1, label="Layer",
                                scale=1,
                            )
                            attn_head_slider = gr.Slider(
                                0, 0, value=0, step=1, label="Head",
                                scale=1,
                            )
                        attn_plot = gr.Plot(label="Attention Pattern")

                    # ── QKV tab ──
                    with gr.TabItem("📊 Q / K / V"):
                        qkv_layer_slider = gr.Slider(
                            0, 0, value=0, step=1, label="Layer"
                        )
                        with gr.Row():
                            q_plot = gr.Plot(label="Q")
                            k_plot = gr.Plot(label="K")
                            v_plot = gr.Plot(label="V")

                    # ── MLP tab ──
                    with gr.TabItem("⚙️ MLP"):
                        mlp_layer_slider = gr.Slider(
                            0, 0, value=0, step=1, label="Layer"
                        )
                        with gr.Row():
                            gate_plot = gr.Plot(label="Gate")
                            up_plot = gr.Plot(label="Up")
                        mlp_out_plot = gr.Plot(label="MLP Output")

        # ── Bottom row: probabilities ──
        with gr.Row():
            topk_plot = gr.Plot(label="Top-5 Token Probabilities", scale=1)
            logit_plot = gr.Plot(label="Logit Distribution", scale=1)

        # ── Hidden state ──
        state = gr.State(False)

        # ── Wire events ──
        generate_btn.click(
            fn=on_generate,
            inputs=[prompt_input, max_new_slider, temp_slider],
            outputs=[
                generated_text, token_counter, token_selector,
                attn_layer_slider, attn_head_slider,
                qkv_layer_slider, mlp_layer_slider,
                attn_plot, q_plot, k_plot, v_plot,
                gate_plot, up_plot, mlp_out_plot,
                topk_plot, logit_plot, state,
            ],
        )

        step_btn.click(
            fn=on_step,
            inputs=[prompt_input, temp_slider],
            outputs=[
                generated_text, token_counter, token_selector,
                attn_layer_slider, attn_head_slider,
                qkv_layer_slider, mlp_layer_slider,
                attn_plot, q_plot, k_plot, v_plot,
                gate_plot, up_plot, mlp_out_plot,
                topk_plot, logit_plot, state,
            ],
        )

        token_selector.change(
            fn=on_select_token,
            inputs=[token_selector, attn_layer_slider, attn_head_slider],
            outputs=[
                generated_text, token_counter, token_selector,
                attn_layer_slider, attn_head_slider,
                qkv_layer_slider, mlp_layer_slider,
                attn_plot, q_plot, k_plot, v_plot,
                gate_plot, up_plot, mlp_out_plot,
                topk_plot, logit_plot, state,
            ],
        )

        attn_layer_slider.change(
            fn=on_change_layer,
            inputs=[token_selector, attn_layer_slider, attn_head_slider],
            outputs=[
                generated_text, token_counter, token_selector,
                attn_layer_slider, attn_head_slider,
                qkv_layer_slider, mlp_layer_slider,
                attn_plot, q_plot, k_plot, v_plot,
                gate_plot, up_plot, mlp_out_plot,
                topk_plot, logit_plot, state,
            ],
        )

        attn_head_slider.change(
            fn=on_change_layer,
            inputs=[token_selector, attn_layer_slider, attn_head_slider],
            outputs=[
                generated_text, token_counter, token_selector,
                attn_layer_slider, attn_head_slider,
                qkv_layer_slider, mlp_layer_slider,
                attn_plot, q_plot, k_plot, v_plot,
                gate_plot, up_plot, mlp_out_plot,
                topk_plot, logit_plot, state,
            ],
        )

        qkv_layer_slider.change(
            fn=on_change_qkv_layer,
            inputs=[token_selector, qkv_layer_slider],
            outputs=[q_plot, k_plot, v_plot],
        )

        mlp_layer_slider.change(
            fn=on_change_mlp_layer,
            inputs=[token_selector, mlp_layer_slider],
            outputs=[gate_plot, up_plot, mlp_out_plot],
        )

        reset_btn.click(
            fn=on_reset,
            inputs=[],
            outputs=[
                prompt_input, generated_text, token_counter, token_selector,
                attn_layer_slider, attn_head_slider,
                qkv_layer_slider, mlp_layer_slider,
                attn_plot, q_plot, k_plot, v_plot,
                gate_plot, up_plot, mlp_out_plot,
                topk_plot, logit_plot, state,
            ],
        )

    return ui


# ───────────────────────────────────────────────
#  Entry point
# ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B Visual Explorer")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true",
                        help="Create a public share link (Gradio)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace model name")
    args = parser.parse_args()

    print("=" * 60)
    print("  Qwen2.5-0.5B Visual Step-Through Explorer")
    print("=" * 60)
    print(f"  Model : {args.model}")
    print(f"  Port  : {args.port}")
    print(f"  Share : {args.share}")
    print()
    print("  Open http://localhost:{} in your browser.".format(args.port))
    print("=" * 60)
    print()

    # Pre-load model
    controller.explorer = QwenExplorer(args.model)
    controller.explorer.load()
    controller.ready = True

    ui = create_ui()
    ui.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
