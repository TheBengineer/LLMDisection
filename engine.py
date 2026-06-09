"""
QwenExplorer — engine that loads the model, registers forward hooks,
runs step-by-step generation, and caches every internal tensor.

Phase 0 complete: every component listed in the instrumentation table is captured.
"""

from __future__ import annotations

import functools
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_structures import LayerSnapshot, StepSnapshot

warnings.filterwarnings("ignore")


class QwenExplorer:
    """Loads the model, manages hooks, runs step-by-step generation, caches states."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")

        # Generation state
        self.steps: List[StepSnapshot] = []
        self.prompt_ids: Optional[torch.Tensor] = None
        self.full_sequence: Optional[torch.Tensor] = None
        self.prompt_length = 0

        # Hook storage
        self._handles = []
        self._capturing = False

        # Temporary caches filled by hooks during each forward pass
        self._current_qkv: Dict[int, Dict[str, np.ndarray]] = {}
        self._current_mlp: Dict[int, Dict[str, np.ndarray]] = {}
        self._current_residuals: Dict[int, Dict[str, np.ndarray]] = {}
        self._current_norms: Dict[int, Dict[str, np.ndarray]] = {}
        self._current_attn_scores: Dict[int, np.ndarray] = {}
        self._current_embedding: Optional[np.ndarray] = None
        self._current_final_norm: Optional[np.ndarray] = None
        self._current_rope_cos: Optional[np.ndarray] = None
        self._current_rope_sin: Optional[np.ndarray] = None

        # Static weight matrices (captured once at load time)
        self.weight_matrices: Dict = {}

        # Model dimensions (filled by load())
        self.num_layers = 0
        self.num_heads = 0
        self.num_kv_heads = 0
        self.head_dim = 0
        self.hidden_size = 0
        self.intermediate_size = 0

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self):
        """Download / load the model and tokenizer, register hooks, capture weights."""
        print(f"[*] Loading {self.model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            attn_implementation="eager",
        )
        self.model.eval()

        cfg = self.model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.hidden_size = cfg.hidden_size
        self.intermediate_size = cfg.intermediate_size

        print(
            f"    {self.num_layers} layers, {self.num_heads} heads "
            f"({self.num_kv_heads} KV heads), "
            f"hidden={self.hidden_size}, intermediate={self.intermediate_size}"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._capture_static_weights()
        self._register_all_hooks()
        self._patch_eager_attention_forward()
        print("[✓] Model ready.\n")

    # ── Static-weight capture (once at load time) ────────────────────────────

    def _capture_static_weights(self):
        """Store all weight matrices that never change during inference."""
        m = self.model.model
        self.weight_matrices["embedding"] = m.embed_tokens.weight.detach().cpu().numpy()
        self.weight_matrices["lm_head"] = self.model.lm_head.weight.detach().cpu().numpy()

        self.weight_matrices["layers"] = {}
        for idx, layer in enumerate(m.layers):
            attn = layer.self_attn
            mlp = layer.mlp
            self.weight_matrices["layers"][idx] = {
                "q_proj": attn.q_proj.weight.detach().cpu().numpy(),
                "k_proj": attn.k_proj.weight.detach().cpu().numpy(),
                "v_proj": attn.v_proj.weight.detach().cpu().numpy(),
                "o_proj": attn.o_proj.weight.detach().cpu().numpy(),
                "gate_proj": mlp.gate_proj.weight.detach().cpu().numpy(),
                "up_proj": mlp.up_proj.weight.detach().cpu().numpy(),
                "down_proj": mlp.down_proj.weight.detach().cpu().numpy(),
            }

    # ── Hook registration ────────────────────────────────────────────────────

    def _register_all_hooks(self):
        """Register forward hooks on every submodule of interest."""
        for idx, layer in enumerate(self.model.model.layers):
            attn = layer.self_attn
            mlp = layer.mlp

            # ── Q/K/V projection outputs (pre-RoPE, last token) ──
            self._handles.append(
                attn.q_proj.register_forward_hook(self._make_qkv_hook(idx, "q"))
            )
            self._handles.append(
                attn.k_proj.register_forward_hook(self._make_qkv_hook(idx, "k"))
            )
            self._handles.append(
                attn.v_proj.register_forward_hook(self._make_qkv_hook(idx, "v"))
            )

            # ── Attention output projection (o_proj) ──
            self._handles.append(
                attn.o_proj.register_forward_hook(self._make_attn_out_hook(idx))
            )

            # ── MLP projections ──
            self._handles.append(
                mlp.gate_proj.register_forward_hook(self._make_gate_hook(idx))
            )
            self._handles.append(
                mlp.up_proj.register_forward_hook(self._make_mlp_hook(idx, "up"))
            )
            # down_proj: capture both input (gate*up) and output
            self._handles.append(
                mlp.down_proj.register_forward_hook(self._make_mlp_down_hook(idx))
            )

            # ── RMSNorm: capture residual input + norm output ──
            self._handles.append(
                layer.input_layernorm.register_forward_hook(
                    self._make_norm_hook(idx, "pre_attn")
                )
            )
            self._handles.append(
                layer.post_attention_layernorm.register_forward_hook(
                    self._make_norm_hook(idx, "post_attn")
                )
            )

            # ── Layer output = residual_post_mlp ──
            self._handles.append(
                layer.register_forward_hook(self._make_layer_output_hook(idx))
            )

        # ── Token embedding ──
        self._handles.append(
            self.model.model.embed_tokens.register_forward_hook(
                self._make_embed_hook()
            )
        )

        # ── Final RMSNorm ──
        self._handles.append(
            self.model.model.norm.register_forward_hook(self._make_final_norm_hook())
        )

        # ── RoPE ──
        self._handles.append(
            self.model.model.rotary_emb.register_forward_hook(self._make_rope_hook())
        )

    # ── Individual hook factories ────────────────────────────────────────────

    def _make_qkv_hook(self, layer_idx: int, name: str):
        """Capture Q/K/V projection output (last token, pre-RoPE)."""
        def hook(module, inp, out):
            if not self._capturing:
                return
            # out shape: [batch, seq_len, proj_dim]
            self._current_qkv.setdefault(layer_idx, {})[name] = out[0, -1].detach().cpu().numpy()
        return hook

    def _make_attn_out_hook(self, layer_idx: int):
        """Capture attention output (after o_proj)."""
        def hook(module, inp, out):
            if not self._capturing:
                return
            self._current_mlp.setdefault(layer_idx, {})["attn_out"] = (
                out[0, -1].detach().cpu().numpy()
            )
        return hook

    def _make_gate_hook(self, layer_idx: int):
        """
        Capture MLP gate projection output.
        Also store the SiLU-activated gate (since Qwen2 uses SiLU).
        """
        def hook(module, inp, out):
            if not self._capturing:
                return
            raw = out[0, -1].detach()  # keep on device for SiLU
            self._current_mlp.setdefault(layer_idx, {})["gate_raw"] = (
                raw.cpu().numpy()
            )
            self._current_mlp[layer_idx]["gate_silu"] = (
                F.silu(raw).cpu().numpy()
            )
        return hook

    def _make_mlp_hook(self, layer_idx: int, name: str):
        """Capture MLP up-projection output."""
        def hook(module, inp, out):
            if not self._capturing:
                return
            self._current_mlp.setdefault(layer_idx, {})[name] = (
                out[0, -1].detach().cpu().numpy()
            )
        return hook

    def _make_mlp_down_hook(self, layer_idx: int):
        """Capture MLP down_proj input (gated activation) and output."""
        def hook(module, inp, out):
            if not self._capturing:
                return
            # inp[0] is the gated hidden state [batch, seq_len, intermediate]
            self._current_mlp.setdefault(layer_idx, {})["down_in"] = (
                inp[0][0, -1].detach().cpu().numpy()
            )
            self._current_mlp[layer_idx]["down"] = (
                out[0, -1].detach().cpu().numpy()
            )
        return hook

    def _make_norm_hook(self, layer_idx: int, name: str):
        """
        Capture both the residual stream value (input to layernorm)
        and the normalised output.
        """
        def hook(module, inp, out):
            if not self._capturing:
                return
            # Residual = input to layernorm
            self._current_residuals.setdefault(layer_idx, {})[f"residual_{name}"] = (
                inp[0][0, -1].detach().cpu().numpy()
            )
            # Normalised output
            self._current_norms.setdefault(layer_idx, {})[f"{name}_norm"] = (
                out[0, -1].detach().cpu().numpy()
            )
        return hook

    def _make_layer_output_hook(self, layer_idx: int):
        """Capture the decoder layer's final output = residual_post_mlp."""
        def hook(module, inp, out):
            if not self._capturing:
                return
            self._current_residuals.setdefault(layer_idx, {})["residual_post_mlp"] = (
                out[0, -1].detach().cpu().numpy()
            )
        return hook

    def _make_embed_hook(self):
        """Capture full token embedding output (all positions)."""
        def hook(module, inp, out):
            if not self._capturing:
                return
            # out shape: [batch, seq_len, hidden_size]
            self._current_embedding = out[0].detach().cpu().numpy()
        return hook

    def _make_final_norm_hook(self):
        """Capture the final RMSNorm output (before LM head)."""
        def hook(module, inp, out):
            if not self._capturing:
                return
            self._current_final_norm = out[0, -1].detach().cpu().numpy()
        return hook

    def _make_rope_hook(self):
        """Capture RoPE cos/sin for all positions."""
        def hook(module, inp, out):
            if not self._capturing:
                return
            cos, sin = out
            # cos/sin shape: [batch, seq_len, head_dim]
            self._current_rope_cos = cos[0].detach().cpu().numpy()
            self._current_rope_sin = sin[0].detach().cpu().numpy()
        return hook

    # ── Monkey-patch for pre-softmax attention scores ───────────────────────

    def _patch_eager_attention_forward(self):
        """
        Replace eager_attention_forward so we can capture the pre-softmax
        attention scores (Q·K^T / sqrt(d)) before they enter softmax.
        """
        import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

        original_fn = qwen2_mod.eager_attention_forward
        explorer_self = self

        @functools.wraps(original_fn)
        def patched_fn(module, query, key, value, attention_mask, scaling,
                       dropout=0.0, **kwargs):
            key_states = qwen2_mod.repeat_kv(key, module.num_key_value_groups)
            value_states = qwen2_mod.repeat_kv(value, module.num_key_value_groups)

            # Pre-softmax scores
            attn_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            # Store pre-softmax scores (last query position)
            if explorer_self._capturing:
                explorer_self._current_attn_scores[module.layer_idx] = (
                    attn_scores[0, :, -1].detach().cpu().numpy()
                )

            # Continue with softmax (same as original)
            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            attn_weights = F.dropout(attn_weights, p=dropout,
                                     training=module.training)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            return attn_output, attn_weights

        qwen2_mod.eager_attention_forward = patched_fn

    # ── State management ─────────────────────────────────────────────────────

    def reset(self):
        """Clear all cached generation state."""
        self.steps = []
        self.prompt_ids = None
        self.full_sequence = None
        self.prompt_length = 0

    # ── Generation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_full(
        self, prompt: str, max_new_tokens: int = 20, temperature: float = 0.8
    ) -> List[str]:
        """Auto-generate a full sequence, capturing every step."""
        self.reset()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        self.prompt_ids = inputs.input_ids
        self.full_sequence = self.prompt_ids.clone()
        self.prompt_length = self.prompt_ids.shape[1]

        # ── First forward pass (prompt) ──
        self._begin_capture()
        outputs = self.model(
            input_ids=self.full_sequence,
            output_attentions=True,
            use_cache=True,
        )
        self._store_step(outputs, is_prompt=True)
        past_kv = outputs.past_key_values
        self._end_capture()

        # ── Generate new tokens ──
        tokens_out: List[str] = []
        for _ in range(max_new_tokens):
            self._begin_capture()

            last_token = self.full_sequence[:, -1:]
            outputs = self.model(
                input_ids=last_token,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=True,
            )

            step = self._store_step(outputs, is_prompt=False)
            step._past_key_values = outputs.past_key_values
            past_kv = outputs.past_key_values
            self._end_capture()

            # Sample next token
            logits = outputs.logits[:, -1, :]
            next_id = self._sample_token(logits, temperature)
            self.full_sequence = torch.cat([self.full_sequence, next_id], dim=-1)

            token_str = self.tokenizer.decode(next_id[0])
            step.token_id = int(next_id[0, 0].item())
            step.token_str = token_str
            tokens_out.append(token_str)

            if next_id.item() == self.tokenizer.eos_token_id:
                break

        return tokens_out

    @torch.no_grad()
    def step_forward(self, temperature: float = 0.8) -> Tuple[bool, str]:
        """Generate exactly one more token. Returns (is_finished, token_str)."""
        if self.full_sequence is None or self.prompt_ids is None:
            raise RuntimeError("Call generate_full() first, or supply a prompt.")

        past_kv = None
        if len(self.steps) > 0 and hasattr(self.steps[-1], "_past_key_values"):
            past_kv = self.steps[-1]._past_key_values

        self._begin_capture()
        last_token = self.full_sequence[:, -1:]
        outputs = self.model(
            input_ids=last_token,
            past_key_values=past_kv,
            use_cache=True,
            output_attentions=True,
        )
        step = self._store_step(outputs, is_prompt=False)
        step._past_key_values = outputs.past_key_values
        self._end_capture()

        # Sample
        logits = outputs.logits[:, -1, :]
        next_id = self._sample_token(logits, temperature)
        self.full_sequence = torch.cat([self.full_sequence, next_id], dim=-1)

        token_str = self.tokenizer.decode(next_id[0])
        step.token_id = int(next_id[0, 0].item())
        step.token_str = token_str

        finished = next_id.item() == self.tokenizer.eos_token_id
        return finished, token_str

    # ── Capture helpers ──────────────────────────────────────────────────────

    def _begin_capture(self):
        """Reset temporary caches at the start of a forward pass."""
        self._capturing = True
        self._current_qkv = {}
        self._current_mlp = {}
        self._current_residuals = {}
        self._current_norms = {}
        self._current_attn_scores = {}
        self._current_embedding = None
        self._current_final_norm = None
        self._current_rope_cos = None
        self._current_rope_sin = None

    def _end_capture(self):
        """Stop capturing after a forward pass."""
        self._capturing = False

    def _store_step(self, outputs, is_prompt: bool = False) -> StepSnapshot:
        """Build a StepSnapshot from model outputs and hook caches."""
        # ── Logits & probs ──
        logits = outputs.logits[0, -1, :].detach().cpu().numpy()
        probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()

        topk = 5
        top_indices = np.argsort(probs)[-topk:][::-1]
        top_probs = probs[top_indices].tolist()
        top_tokens = [self.tokenizer.decode([int(idx)]) for idx in top_indices]

        step = StepSnapshot(
            token_id=0,
            token_str="",
            position=len(self.steps),
            logits=logits,
            probs=probs,
            topk_indices=top_indices.tolist(),
            topk_tokens=top_tokens,
            topk_probs=top_probs,
        )

        # ── Global tensors ──
        if self._current_embedding is not None:
            step.token_embeddings = self._current_embedding
            step.input_embeds = self._current_embedding[-1]  # last position
        if self._current_rope_cos is not None:
            step.rope_cos = self._current_rope_cos
            step.rope_sin = self._current_rope_sin
        if self._current_final_norm is not None:
            step.final_norm_output = self._current_final_norm

        # ── Per-layer data ──
        for layer_idx in range(self.num_layers):
            w = self.weight_matrices["layers"][layer_idx]
            layer_snap = LayerSnapshot(layer_idx=layer_idx)

            # Static weights
            layer_snap.q_weight = w["q_proj"]
            layer_snap.k_weight = w["k_proj"]
            layer_snap.v_weight = w["v_proj"]
            layer_snap.o_weight = w["o_proj"]
            layer_snap.gate_weight = w["gate_proj"]
            layer_snap.up_weight = w["up_proj"]
            layer_snap.down_weight = w["down_proj"]

            # Q/K/V projection outputs (pre-RoPE)
            if layer_idx in self._current_qkv:
                qkv = self._current_qkv[layer_idx]
                q_val = qkv.get("q")
                k_val = qkv.get("k")
                v_val = qkv.get("v")
                layer_snap.q = q_val
                layer_snap.k = k_val
                layer_snap.v = v_val
                # Reshape to [num_heads, head_dim] for pre-RoPE visualisation
                if q_val is not None:
                    layer_snap.q_pre_rope = q_val.reshape(self.num_heads, self.head_dim)
                if k_val is not None:
                    layer_snap.k_pre_rope = k_val.reshape(self.num_kv_heads, self.head_dim)

            # Q/K post-RoPE (apply rotation using captured cos/sin)
            if (layer_snap.q_pre_rope is not None
                    and step.rope_cos is not None
                    and step.rope_sin is not None):
                cos = step.rope_cos[-1]          # [head_dim] for last position
                sin = step.rope_sin[-1]
                q_pre = layer_snap.q_pre_rope    # [num_heads, head_dim]
                half = self.head_dim // 2
                q_rotated = np.concatenate([-q_pre[:, half:], q_pre[:, :half]], axis=1)
                layer_snap.q_post_rope = (
                    q_pre * cos[np.newaxis, :] + q_rotated * sin[np.newaxis, :]
                )

                k_pre = layer_snap.k_pre_rope    # [num_kv_heads, head_dim]
                k_rotated = np.concatenate([-k_pre[:, half:], k_pre[:, :half]], axis=1)
                layer_snap.k_post_rope = (
                    k_pre * cos[np.newaxis, :] + k_rotated * sin[np.newaxis, :]
                )

            # Attention scores (pre-softmax)
            if layer_idx in self._current_attn_scores:
                layer_snap.attn_scores = self._current_attn_scores[layer_idx]

            # Attention probabilities (from output_attentions)
            if outputs.attentions is not None and layer_idx < len(outputs.attentions):
                attn = outputs.attentions[layer_idx]
                layer_snap.attn_probs = attn[0, :, -1].detach().cpu().numpy()

            # Attention output
            if layer_idx in self._current_mlp:
                mlp = self._current_mlp[layer_idx]
                layer_snap.attn_output = mlp.get("attn_out")
                layer_snap.mlp_gate_raw = mlp.get("gate_raw")
                layer_snap.mlp_gate_silu = mlp.get("gate_silu")
                layer_snap.mlp_up = mlp.get("up")
                layer_snap.mlp_down_input = mlp.get("down_in")
                layer_snap.mlp_output = mlp.get("down")

            # Residuals
            if layer_idx in self._current_residuals:
                res = self._current_residuals[layer_idx]
                layer_snap.residual_pre_attn = res.get("residual_pre_attn")
                layer_snap.residual_post_attn = res.get("residual_post_attn")
                layer_snap.residual_post_mlp = res.get("residual_post_mlp")

            # Normaliser outputs
            if layer_idx in self._current_norms:
                norm = self._current_norms[layer_idx]
                layer_snap.input_layernorm_output = norm.get("pre_attn_norm")
                layer_snap.post_attention_layernorm_output = norm.get(
                    "post_attn_norm"
                )

            step.layers[layer_idx] = layer_snap

        self.steps.append(step)
        return step

    # ── Sampling ─────────────────────────────────────────────────────────────

    @staticmethod
    def _sample_token(
        logits: torch.Tensor, temperature: float = 0.8
    ) -> torch.Tensor:
        """Sample a single token from logits."""
        if temperature < 1e-6:
            return torch.argmax(logits, dim=-1, keepdim=True)
        scaled = logits / temperature
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    # ── Accessors ────────────────────────────────────────────────────────────

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
        return [self.tokenizer.decode([tid]) for tid in ids]

    @property
    def is_loaded(self) -> bool:
        return self.model is not None
