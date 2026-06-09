"""
Data structures for capturing transformer internal states.

LayerSnapshot  — one layer's activations at one generation step.
StepSnapshot   — one complete generation step (all layers + global state).
FlowNode       — placeholder for Phase 2 flowchart topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class LayerSnapshot:
    """All activations captured in one transformer layer at one generation step."""

    layer_idx: int

    # ── Weight matrices (static, captured once at load time) ──
    q_weight: Optional[np.ndarray] = None       # [num_heads*head_dim, hidden_size]
    k_weight: Optional[np.ndarray] = None       # [num_kv_heads*head_dim, hidden_size]
    v_weight: Optional[np.ndarray] = None       # [num_kv_heads*head_dim, hidden_size]
    o_weight: Optional[np.ndarray] = None       # [hidden_size, num_heads*head_dim]
    gate_weight: Optional[np.ndarray] = None    # [intermediate_size, hidden_size]
    up_weight: Optional[np.ndarray] = None      # [intermediate_size, hidden_size]
    down_weight: Optional[np.ndarray] = None    # [hidden_size, intermediate_size]

    # ── Q/K/V projections (output vectors, last token) ──
    q: Optional[np.ndarray] = None              # [hidden_size] = [896]
    k: Optional[np.ndarray] = None              # [num_kv_heads*head_dim] = [128]
    v: Optional[np.ndarray] = None              # [num_kv_heads*head_dim] = [128]

    # ── Q/K pre- / post-RoPE (head×dim shaped) ──
    q_pre_rope: Optional[np.ndarray] = None     # [num_heads, head_dim] = [14, 64]
    k_pre_rope: Optional[np.ndarray] = None     # [num_kv_heads, head_dim] = [2, 64]
    q_post_rope: Optional[np.ndarray] = None    # [num_heads, head_dim] = [14, 64]
    k_post_rope: Optional[np.ndarray] = None    # [num_kv_heads, head_dim] = [2, 64]

    # ── Attention ──
    attn_scores: Optional[np.ndarray] = None    # [num_heads, seq_len] (pre-softmax)
    attn_probs: Optional[np.ndarray] = None     # [num_heads, seq_len] (post-softmax)
    attn_output: Optional[np.ndarray] = None    # [hidden_size] after o_proj

    # ── MLP internals ──
    mlp_gate_raw: Optional[np.ndarray] = None   # [intermediate_size] pre-SiLU
    mlp_gate_silu: Optional[np.ndarray] = None  # [intermediate_size] after SiLU
    mlp_up: Optional[np.ndarray] = None         # [intermediate_size] up projection
    mlp_down_input: Optional[np.ndarray] = None # [intermediate_size] input to down_proj
    mlp_output: Optional[np.ndarray] = None     # [hidden_size] after down_proj

    # ── Layer-normalisation (RMSNorm) outputs ──
    input_layernorm_output: Optional[np.ndarray] = None      # [hidden_size]
    post_attention_layernorm_output: Optional[np.ndarray] = None

    # ── Residual stream ──
    residual_pre_attn: Optional[np.ndarray] = None    # [hidden_size]
    residual_post_attn: Optional[np.ndarray] = None   # [hidden_size]
    residual_post_mlp: Optional[np.ndarray] = None    # [hidden_size]


@dataclass
class StepSnapshot:
    """Full snapshot of one generation step (one new token)."""

    token_id: int
    token_str: str
    position: int

    # ── Token / sequence embeddings ──
    token_embeddings: Optional[np.ndarray] = None  # [seq_len, hidden_size]
    input_embeds: Optional[np.ndarray] = None      # [hidden_size] last token's embedding

    # ── RoPE embeddings ──
    rope_cos: Optional[np.ndarray] = None          # [seq_len, head_dim]
    rope_sin: Optional[np.ndarray] = None          # [seq_len, head_dim]

    # ── Logits & predictions ──
    logits: Optional[np.ndarray] = None
    probs: Optional[np.ndarray] = None
    topk_indices: List[int] = field(default_factory=list)
    topk_tokens: List[str] = field(default_factory=list)
    topk_probs: List[float] = field(default_factory=list)

    # ── Final RMSNorm output (before LM head) ──
    final_norm_output: Optional[np.ndarray] = None  # [hidden_size]

    # ── Per-layer snapshots ──
    layers: Dict[int, LayerSnapshot] = field(default_factory=dict)

    # Internal: past key-values for continued generation (not serialised)
    _past_key_values: Any = None
