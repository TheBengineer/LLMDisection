# Qwen2.5-0.5B Visual Step-Through Explorer

> **Inspect the internals of a small transformer language model — attention patterns, Q/K/V vectors, MLP activations, and token probabilities — layer by layer, token by token.**

![Model](https://img.shields.io/badge/Model-Qwen2.5--0.5B-orange)
![Framework](https://img.shields.io/badge/Framework-PyTorch-red)
![UI](https://img.shields.io/badge/UI-Gradio-blue)

This project builds an **interactive browser-based UI** around [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), a 500M-parameter causal language model. Using PyTorch forward hooks and Gradio, you can step through each generated token and inspect the model's internal state at every layer.

---

## ✨ Features

| Feature | What it shows |
|---|---|
| **🔍 Attention** | Per-layer, per-head attention heatmaps — how the current token attends to all previous tokens |
| **📊 Q / K / V** | Query, Key, Value vectors reshaped as head×dim heatmaps for any layer |
| **⚙️ MLP** | Line plots of gate activation, up-projection, and MLP output (first 300 dims) |
| **🏆 Top-5 Probabilities** | Horizontal bar chart of the 5 most likely next tokens with percentages |
| **📉 Logit Distribution** | Downsampled scatter plot of all ~15k logits |
| **⏭️ Step-by-Step** | Generate one token at a time and watch every internal value update live |
| **🔄 Token Slider** | Revisit any earlier generation step and re-inspect its full state |
| **🧠 Residual Stream** | Pre-attention, post-attention, and post-MLP residual snapshots (available in code) |

---

## 🚀 Quick Start

### Requirements
- Python 3.10+
- ~2 GB free RAM (CPU inference)
- ~2 GB disk for model cache

### Linux / macOS

```bash
git clone <repo-url> && cd LLMDisection
bash run.sh
```

### Windows

```cmd
run.bat
```

### Manual Setup

```bash
pip install torch transformers gradio plotly
python app.py
# Open http://localhost:7860
```

---

## 🎮 How to Use

1. **Enter a prompt** — try *"The meaning of life is"* (default)
2. **Click 🚀 Generate** — runs the full sequence
3. **Inspect any token** — drag the **Token Step** slider
4. **Explore tabs**: 🔍 Attention, 📊 Q/K/V, ⚙️ MLP
5. **Click ⏭️ One Step** — generate one more token
6. **Click 🔄 Reset** — clear state

---

## 🏗️ Architecture

```
app.py
├── QwenExplorer           # Engine: load model, register hooks, generate
│   ├── load()             # Download model, register hooks on all layers
│   ├── generate_full()    # Auto-generate N tokens
│   ├── step_forward()     # One token at a time
│   └── hooks              # Q/K/V, MLP, residual, attention hooks
├── LayerSnapshot          # One layer's activations at one step
├── StepSnapshot           # One generation step (logits, probs, all layers)
├── Visualization helpers  # plot_attention, plot_qkv, plot_mlp, etc.
└── Gradio UI              # Interactive web interface (create_ui)
```

Every forward hook captures only the **last query position**, minimizing memory. All 24 layers are captured simultaneously in a single forward pass.

---

## 🧪 Why Qwen2.5-0.5B?

| Criterion | Why |
|---|---|
| **Small** | 500M params — runs on CPU, fast inference |
| **Modern** | RoPE, SwiGLU, RMSNorm — same architecture as GPT-4, Llama 3 |
| **Clean code** | Well-structured HuggingFace integration |
| **Full checkpoint** | float32, no quantization needed |

---

## 📁 Project Structure

```
LLMDisection/
├── app.py          # Main application (engine + UI + visualization)
├── run.sh          # Linux/macOS launcher
├── run.bat         # Windows launcher
└── README.md       # This file
```

---

## 🧠 Example Use Cases

- **Education** — Teach how attention, MLPs, and residual streams work inside a real transformer
- **Debugging** — See exactly which tokens a model attends to when it makes mistakes
- **Research** — Probe for induction heads, indirect object identification, etc.
- **Interpretability practice** — Fork and add activation patching, logit lens, or direct logit attribution

---

## 🔧 Configuration

```bash
python app.py --port 7860 --share --model Qwen/Qwen2.5-0.5B
```

| Flag | Default | Description |
|---|---|---|
| `--port` | `7860` | Server port |
| `--share` | `False` | Public Gradio share link |
| `--model` | `Qwen/Qwen2.5-0.5B` | HuggingFace model name |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0 | Model inference |
| `transformers` | ≥ 4.35 | Model & tokenizer loading |
| `gradio` | ≥ 4.0 | Interactive web UI |
| `plotly` | ≥ 5.14 | Interactive charts |
| `numpy` | ≥ 1.24 | Data handling |

---

## 🤝 Ideas for Extensions

- **Activation patching** — patch a layer's output and see how the prediction changes
- **Logit lens** — decode the residual stream at each layer
- **Direct logit attribution** — which heads contribute most to the final prediction
- **Auto-play** — step through one token per second automatically
- **Compare two prompts** — side-by-side view
- **Export snapshots** — save layer data as JSON

---

## 📄 License

Provided as-is for educational purposes. [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) is released under the Qwen License (Apache 2.0 derivative).
