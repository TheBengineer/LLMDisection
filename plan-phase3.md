# Phase 3 — Two-Panel Flowchart Layout

**Last updated:** 2026-06-10

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
**File:** `ui.py`  
**Severity:** 🟢 Low → **✅ FIXED**

`plot_logits_sampled` and `plot_residual_evolution` were imported but never called — removed.

---

## 📋 Fix Roadmap (All Complete)

| Priority | Description | Status |
|---|---|---|---|
| **[P0]** `_build_response()` calling `plot_layer_contributions(snapshot)` with wrong type | ✅ FIXED — delegated to `_dispatch_plot("root")` |
| **[P1]** Step/Reset buttons unconnected | ✅ FIXED — wired `step_btn.click()` → `on_step()` and `reset_btn.click()` → `on_reset()` |
| **[P2]** Dead imports (`plot_logits_sampled`, `plot_residual_evolution`) | ✅ FIXED — removed from import block |
| **[P3]** Duplicate layer-index parser | ✅ FIXED — `_layer_index_from_node_id()` now delegates to `_parse_layer_idx()` |

---

# Phase 4 — Interactive Flowchart Clicking

**Status:** Complete (2026-06-10)

## Pre-Existing Bugs Fixed

| Bug | Description | Status |
|---|---|---|
| **A** | Collapse toggle broken — `node.collapsed or (id in override)` meant layers could never be expanded. Changed to `node.collapsed and id not in expanded_override` | ✅ FIXED — `flowchart.py:439` |
| **B** | Glow filter defined in `<defs>` but never applied to active node (missing `filter="url(#glow)"`) | ✅ FIXED — `flowchart.py:654` |
| **C** | Toggle direction reversed — `on_collapse_toggle` managed a forced-collapse set, now it manages `expanded_override` | ✅ FIXED — `ui.py:636` |

## Tasks Implemented

| Task | Description | Files |
|---|---|---|
| **4.1** | SVG Node Interactivity — `onclick` and `cursor:pointer` moved from `<rect>` to parent `<g>`, hover styles added | `flowchart.py` |
| **4.2** | Active-Node Highlighting — golden glow on selected node, auto-expand ancestors when selecting a child | `flowchart.py`, `ui.py` |
| **4.3** | Hover Tooltips — all nodes get `<title>`: group nodes show child count, leaf nodes show label + shape | `flowchart.py` |
| **4.4** | Token Step Slider — `gr.Slider` in top bar, scrubbing updates detail view, max updated after generate/step | `ui.py` |
| **4.5** | Expand/Collapse All — 📂 Expand All / 📁 Collapse All buttons in flowchart header | `ui.py` |
| **4.6** | Node Search — search box filters and highlights nodes (cyan), dims non-matching, "No matching nodes" overlay | `flowchart.py`, `ui.py` |
| **4.7** | Keyboard Navigation — Up/Down arrows traverse visible nodes, Enter selects, `data-visible-nodes` attribute on SVG | `flowchart.py`, `ui.py` |
| **4.8** | Auto-Scroll to Active Node — `MutationObserver` scrolls flowchart to keep active node visible | `ui.py` |
| **4.9** | URL Hash Persistence — `#node=...&step=...` saved on selection, restored on page load | `ui.py` |
| **4.10** | Edge Cases — debouncing guard (`controller.generating`), no-match overlay, reset clears generating flag | `ui.py` |
| **4.11** | CSS Fade-In Animation — `@keyframes svg-fade-in` on SVG re-render | `ui.py` |
| **4.12** | Plan document updated — this section | `plan-phase3.md` |
