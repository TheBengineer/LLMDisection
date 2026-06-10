# Phase 3 — Two-Panel Flowchart Layout

**Last updated:** 2026-06-09

## ⚠️ PLAN AUDIT: Code is AHEAD of plan document

The original plan enumerated tasks 3.1–3.9 all marked `[ ]` (not started).  
**Reality:** The entire two-panel UI (Phases 3 + 4) is already implemented on disk in `ui.py` (664 lines), along with the SVG rendering engine in `flowchart.py` (767 lines) and all 18 plot functions in `plots.py` (883 lines).

This document has been rewritten to reflect the **actual current state** and list the remaining bugs/issues.

---

## ✅ Already Implemented (verified on disk)

| Subtask | Status | Where |
|---|---|---|
| **3.1 — Two-panel layout** | ✅ Done | `ui.py:551` `create_ui()` — `gr.Row(scale=35:65)` with flowchart left, detail right |
| **3.2 — Node tree state** | ✅ Done | `ui.py` `UIController` class with `node_tree`, `active_node_id`, `collapsed`, `thumbnails` |
| **3.3 — SVG rendering** | ✅ Done | `flowchart.py:571` `render_flowchart_svg()` called from `ui.py` `UIController.current_svg` |
| **3.4 — Plot dispatch** | ✅ Done | `ui.py:156` `_dispatch_plot()` maps all node types to plot functions |
| **3.5 — Node descriptions** | ✅ Done | `ui.py:130` `_get_node_description()` + `_get_node_title()` |
| **3.6 — Callbacks** | ✅ Done | `on_generate()`, `on_step()`, `on_node_select()`, `on_collapse_toggle()`, `on_reset()` |
| **3.7 — JS bridge** | ✅ Done | HTML-injected JS (`_JS_BRIDGE`) listens for `flowchart-node-select` events |
| **3.8 — Polish & CSS** | ✅ Done | `CUSTOM_CSS` with hover effects, detail panel styling |
| **3.9 — app.py integration** | ✅ Done | `app.py` imports `create_ui()` from `ui.py` correctly |

Also implemented (Phases 2.1–2.3, SVG rendering engine):
- `FlowchartNode` dataclass (`flowchart.py`)
- `build_qwen_node_tree()` — full Qwen2.5-0.5B architecture tree
- `render_flowchart_svg()` — generates interactive SVG with click handlers + thumbnails

All 18 plot functions exist in `plots.py` — verified against imports in `ui.py`.

---

## 🔴 Bugs Found

### Bug #1: `_build_response()` calls `plot_layer_contributions(snapshot)` with wrong type
**File:** `ui.py`, line 382  
**Severity:** ~~🔴 Critical — will crash~~ **✅ FIXED**

~~`snapshot` is a `StepSnapshot` object, but `plot_layer_contributions()` expects `Optional[List[float]]` (list of L2 norms).~~  
~~The `_dispatch_plot` root handler (`node_id == "root"`) correctly extracts norms from the snapshot, but the fallback in `_build_response` passes the raw snapshot. This will raise a `TypeError` when the user generates tokens and sees the overview.~~

**Fix applied (2026-06-09):** Replaced `plot_layer_contributions(snapshot)` with `_dispatch_plot("root")`, which already has the correct residual-delta norm extraction logic.

---

### Bug #2 (minor): Unused imports
**File:** `ui.py`, lines 31, 35  
**Severity:** 🟢 Low

`plot_logits_sampled` and `plot_residual_evolution` are imported but never called in `_dispatch_plot()`. Not harmful but dead code.

---

## 🟡 Functional Gaps

### Gap #1: No step-back navigation
**Severity:** 🟡 Medium

The UI has **Step** (advance one token) and **Generate** (generate all), but no way to browse previously generated tokens.  
`UIController.get_snapshot(step_idx)` supports arbitrary step indices, and `explorer.num_steps()` is available — but there's no UI control (slider or dropdown) to select a step.

---

## 📋 Fix Roadmap

**Priority order:**

1. **[P0] Fix Bug #1** — `_build_response()` calling `plot_layer_contributions(snapshot)` → **✅ DONE** (delegated to `_dispatch_plot("root")`)
2. **[P1] Add step-back slider** — Token step slider to browse previous generation steps
3. **[P2] Remove or wire unused imports** — `plot_logits_sampled` and `plot_residual_evolution` are imported but never called → remove dead imports
4. **[P3] Richer node descriptions** — `_get_node_description()` falls back to generic text for many nodes; add tensor shapes
