# Phase 1 — Manual Test Checklist

## Prerequisites
- [ ] Python environment with `torch`, `transformers`, `gradio`, `plotly`, `numpy`
- [ ] Run `python app.py` — app starts without errors
- [ ] Browser opens to `http://localhost:7860`

---

## 1. Basic Loading & Generation

| # | Test | Expected | Pass? |
|---|---|---|---|
| 1.1 | App loads without errors | Console shows `Model ready.`, UI appears | ☐ |
| 1.2 | Click **🚀 Generate** with default prompt | Text appears in output box, counter shows >0 tokens | ☐ |
| 1.3 | Click **⏭️ One Step** after generation | One new token appended | ☐ |
| 1.4 | Click **⏭️ One Step** without prior generation | First prompt + one token generated | ☐ |
| 1.5 | Adjust **Max new tokens** slider, click Generate | Respects the max limit | ☐ |
| 1.6 | Adjust **Temperature** slider | Different (or same for t≈0) token distributions | ☐ |
| 1.7 | Click **🔄 Reset** after generation | All plots return to "Press Generate", text cleared | ☐ |

---

## 2. Token Selector

| # | Test | Expected | Pass? |
|---|---|---|---|
| 2.1 | Move **◀ Select Token Step ▶** slider | All plots update to that token position | ☐ |
| 2.2 | Slide to step 0 (the prompt itself) | Plots show prompt-step data | ☐ |
| 2.3 | Slide back and forth | No errors, plots update smoothly | ☐ |

---

## 3. 🔍 Attention Tab

| # | Test | Expected | Pass? |
|---|---|---|---|
| 3.1 | **Attention (post-softmax)** heatmap | Shows attention weights for selected layer/head/token | ☐ |
| 3.2 | **Attention Scores (pre-softmax)** heatmap | Shows raw scores before softmax | ☐ |
| 3.3 | Change **Layer** slider | Both plots update to new layer | ☐ |
| 3.4 | Change **Head** slider | Both plots update to new head | ☐ |
| 3.5 | Hover over heatmap cells | Tooltip shows token + weight/score | ☐ |
| 3.6 | Heatmap colors make sense | Scores use RdBu diverging, weights use Viridis sequential | ☐ |

---

## 4. 📊 Q / K / V Tab

| # | Test | Expected | Pass? |
|---|---|---|---|
| 4.1 | **Q** plot shows head×dim heatmap | 14×64 heatmap with RdBu colors | ☐ |
| 4.2 | **K** plot shows 2×64 heatmap | KV heads = 2 for Qwen2.5-0.5B | ☐ |
| 4.3 | **V** plot shows 2×64 heatmap | Same shape as K | ☐ |
| 4.4 | Change **Layer** slider | All Q/K/V plots update | ☐ |
| 4.5 | **Q Weight** heatmap visible | 896×896 matrix (auto-strided) | ☐ |
| 4.6 | **K Weight**, **V Weight**, **O Weight** visible | All 4 weight matrices shown | ☐ |
| 4.7 | **Gate Weight**, **Up Weight**, **Down Weight** visible | All 3 MLP weight matrices shown | ☐ |
| 4.8 | Large matrices show stride info in axis label | Labels say "sampled every N" | ☐ |

---

## 5. ⚙️ MLP Tab

| # | Test | Expected | Pass? |
|---|---|---|---|
| 5.1 | **Gate (SiLU)** line plot | Shows 4864-dim activation values | ☐ |
| 5.2 | **Gate Pre-Activation Histogram** | Distribution of raw gate values | ☐ |
| 5.3 | **SiLU Scatter** (raw vs activated) | Clear SiLU shape visible (s-shaped curve) | ☐ |
| 5.4 | **MLP Output** line plot | Shows final 896-dim output | ☐ |
| 5.5 | Change **Layer** slider | All 4 plots update | ☐ |

---

## 6. 🧮 RMSNorm Tab

| # | Test | Expected | Pass? |
|---|---|---|---|
| 6.1 | **Pre-Attention Norm** plot | Overlay of input vs output of pre-attn RMSNorm | ☐ |
| 6.2 | **Post-Attention Norm** plot | Overlay of input vs output of post-attn RMSNorm | ☐ |
| 6.3 | Output is scaled, input shows pre-norm distribution | Visual difference visible | ☐ |

---

## 7. 🔄 Residual Tab

| # | Test | Expected | Pass? |
|---|---|---|---|
| 7.1 | **Residual Stream Delta** plot | Shows residual before and after (pre-attn vs post-mlp) | ☐ |
| 7.2 | Two traces visible with legend | "Before" (blue) and "After" (coral) | ☐ |

---

## 8. 🔤 Embedding Tab

| # | Test | Expected | Pass? |
|---|---|---|---|
| 8.1 | **Token Embedding** heatmap | Shows slice of embedding table around current token | ☐ |
| 8.2 | Current token row highlighted | Red dashed line marks the active token | ☐ |

---

## 9. 🔄 RoPE Tab

| # | Test | Expected | Pass? |
|---|---|---|---|
| 9.1 | **RoPE Rotation** polar plot | Spoke diagram showing rotation per pair | ☐ |
| 9.2 | Change **Position** slider | Angles change (more rotation at higher positions) | ☐ |
| 9.3 | Hover shows pair name | Each spoke labelled | ☐ |

---

## 10. 🔄 Q/K Pre vs Post RoPE Tab (NEW)

| # | Test | Expected | Pass? |
|---|---|---|---|
| 10.1 | Tab exists with two sliders (Layer, Head) | UI components visible | ☐ |
| 10.2 | After Generate, 4-panel subplot visible | Q pre, Q post, K pre, K post bar charts | ☐ |
| 10.3 | Change **Layer** slider | Plots update | ☐ |
| 10.4 | Change **Head** slider | Plots update (different head) | ☐ |
| 10.5 | Pre vs Post show visible difference | Rotation changes values | ☐ |

---

## 11. 📊 Contributions Tab

| # | Test | Expected | Pass? |
|---|---|---|---|
| 11.1 | **Layer Contributions** bar chart | One bar per layer showing L2 delta | ☐ |
| 11.2 | Bars use Viridis color scale | Sequential color ramp | ☐ |

---

## 12. 📈 Residual Evolution Tab (NEW)

| # | Test | Expected | Pass? |
|---|---|---|---|
| 12.1 | Tab exists with a single large plot | 2-row plot with lines + bars | ☐ |
| 12.2 | Top panel: gradient-colored lines per layer | Blue (pre-attn), Orange (post-attn), Green (post-mlp) | ☐ |
| 12.3 | Bottom panel: "Attention Δ" and "MLP Δ" bars | Grouped bars per layer | ☐ |
| 12.4 | Rangeslider works on top panel | Zoom in/out works | ☐ |
| 12.5 | Hover over lines shows layer number | Tooltip identifies each trace | ☐ |

---

## 13. Bottom Row: Probabilities

| # | Test | Expected | Pass? |
|---|---|---|---|
| 13.1 | **Top-K Token Probabilities** bar chart | Horizontal bars with top 5 tokens + % | ☐ |
| 13.2 | **Logit Distribution** scatter plot | Sampled scatter of full vocab logits | ☐ |

---

## 14. Regression Checks

| # | Test | Expected | Pass? |
|---|---|---|---|
| 14.1 | No console errors during any operation | Python console clean | ☐ |
| 14.2 | Gradio UI doesn't freeze or lag | Responsive | ☐ |
| 14.3 | All sliders update smoothly | No flickering or stale plots | ☐ |
| 14.4 | Reset clears everything | Back to initial state | ☐ |
| 14.5 | Multiple Generate → Reset → Generate works | No stale state | ☐ |

---

## 15. Performance

| # | Test | Expected | Pass? |
|---|---|---|---|
| 15.1 | First generation completes within reasonable time (<30s on CPU) | ✓ | ☐ |
| 15.2 | Weight matrix plots load quickly even for large (896×896) matrices | Auto-stride keeps display ~512 | ☐ |
| 15.3 | Residual evolution plot with 24 layers renders without timeout | ✓ | ☐ |
