# Phase 4 Plan: Interactive Flowchart Clicking

**Goal:** Transform the static SVG flowchart into a fully interactive navigation tool — click any node to see its live plot, collapse/expand layer groups, step through tokens, and navigate the tree efficiently.

**Status:** ✅ ALL TASKS COMPLETE (2026-06-10)

**Dependencies:** Phase 3 (ui.py) ✅, Phase 2.1–2.2 (node tree) ✅, Phase 2.3 (SVG rendering) ✅

---

## ⚠️ Pre-Existing Bugs (Must Fix Before or During Phase 4)

### Bug A: Collapse Toggle is Broken

**Files:** `flowchart.py:439` + `ui.py:541–548`
**Root cause:** `_compute_layout` uses `node.collapsed or (node_id in collapsed_override)`. All layer nodes have `collapsed=True` by default (`flowchart.py:81`). Since `True or X = True`, adding/removing a layer from `controller.collapsed` has **no effect** — the layers can never be expanded.

**Fix:** Change the logic to an override-first model. Add an `expanded_override: set[str]` parameter to `render_flowchart_svg()` (and pass through to `_compute_layout`). A node is collapsed iff `node.collapsed and node_id not in expanded_override`. The `collapsed_override` parameter should be removed or replaced with `expanded_override`.

Affected code paths:
- `flowchart.py:439` — `_compute_layout` collapse logic
- `flowchart.py:575` — `render_flowchart_svg` signature
- `ui.py:57` — `controller.current_svg` property
- `ui.py:541–548` — `on_collapse_toggle` callback
- `ui.py:552–560` — `on_reset` (needs to clear expanded set)

### Bug B: Glow Filter Defined but Never Applied

**File:** `flowchart.py:612–615` + `flowchart.py:655–657`
**Issue:** `<filter id="glow">` is added to `<defs>` but the active node's `<rect>` never gets `filter="url(#glow)"`. Only `stroke` and `stroke-width` are applied.

**Fix:** Add `filter="url(#glow)"` to the active node's `<rect>` when `is_active` is true (`flowchart.py:655–657`).

### Bug C: `on_collapse_toggle` Toggles the Wrong Direction

**File:** `ui.py:541–548`
**Issue:** `on_collapse_toggle` toggles membership in `controller.collapsed` (the *forced-collapse* set). But with the current broken `or` logic, this set can only force-collapse expanded nodes, never expand collapsed ones. Even after fixing Bug A, the toggle logic needs review — the callback should toggle an `expanded_override` set instead of a `collapsed` set.

**Fix:** After fixing Bug A, `on_collapse_toggle` should toggle membership in `controller.expanded_override` (add to expand, remove to re-collapse to default).

---

## Task 4.1 — SVG Node Interactivity (fully wire clicks in `flowchart.py`)

**What:** Ensure every rendered SVG `<g>` node fires the appropriate CustomEvent on click.

- Collapsible nodes (`layer_group`, `component_group`) fire `flowchart-collapse-toggle`
- Leaf nodes (all others) fire `flowchart-node-select`
- Add `cursor:pointer` on the outer `<g>` (not just `<rect>`), so clicks on label/icon text also trigger the pointer cursor
- Add `:hover` style (lighten fill, subtle shadow)
- **File:** `flowchart.py` → node rendering in `render_flowchart_svg()`
- **Deliverable:** Click any SVG node → event reaches Gradio callback

**Implementation note:** Currently `onclick` is on `<rect>` (`flowchart.py:668`). Move `onclick` and `cursor:pointer` to the parent `<g class="flowchart-node-group">` (`flowchart.py:660`) so label/icon clicks are captured too.

---

## Task 4.2 — Active-Node Highlighting

**What:** When a node is selected, visually distinguish it in the SVG and ensure it's visible.

- Accept `active_node_id` parameter in `render_flowchart_svg()` ✅ already done
- Render the active node with `stroke: #ffd700; stroke-width: 3; filter: drop-shadow(...)` ✅ stroke done, **filter not applied** (see Bug B)
- **Add `filter="url(#glow)"`** to the active node's `<rect>` when `is_active` is true
- **If active node is a child of a collapsed group, auto-expand the parent** — this requires `_compute_layout` to walk up the `parent_id` chain from the active node and force-expand any collapsed ancestor before computing layout. Add a helper `_ensure_visible(nodes, active_id)` that returns a set of node IDs to expand.
- `<defs>` SVG glow filter already exists ✅
- **Files:** `flowchart.py` + `ui.py` (controller.active_node_id)
- **Deliverable:** Selected node has a golden glow; parent layers auto-expand

---

## Task 4.3 — Hover Tooltips on Every Node

**What:** Show tensor shape and description when hovering over a node.

- Add `<title>` element inside each SVG `<g>` with shape + description
- For leaf nodes: `"{tensor_shape} — {description}"` ✅ partially done (label + shape only)
- **For group nodes:** `"Click to expand — contains N children"` — compute `len(node.children)` from the `nodes` dict, format as tooltip text
- **File:** `flowchart.py` → node rendering in `render_flowchart_svg()`
- **Deliverable:** Hovering over any node shows a native browser tooltip

**Current gap:** Only nodes with `shape` set get `<title>` (`flowchart.py:673`). Group nodes (layers, attention groups, MLP groups) have `shape=""` and get no tooltip. Fix: always emit `<title>`, using child count for group nodes.

---

## Task 4.4 — Token Step Slider

**What:** Let the user scrub through generated tokens.

- Add `gr.Slider(minimum=0, maximum=0, step=1, label="Token step")` to the top bar
- Wire `.change()` → a new callback that passes `token_idx` through to `_build_response`
- **Update `_build_response()` signature** (`ui.py:461`): add `token_idx: int | None = None` parameter, forward it to `controller.get_snapshot(token_idx)`
- Update slider max dynamically after Generate/Step (return slider update as additional output)
- Show "Token 5 of 20" label next to slider (use `gr.Markdown` or slider label)
- **Files:** `ui.py` → `create_ui()` + callbacks
- **Deliverable:** Slider appears after generation, scrubbing updates both flowchart and detail plot

**Edge cases to handle:**
- Slider with 1 step: `maximum=0` (Gradio supports min=max=0)
- `token_idx` out of bounds: clamp to valid range or show empty state

---

## Task 4.5 — Collapse / Expand All Button

**What:** One-click toggle to expand or collapse all 24 layers.

**Depends on:** Bug A fix (collapse logic must use `expanded_override`).

- Add a small button row: `[📂 Expand All] [📁 Collapse All]`
- `on_expand_all()`: add all `layer_0` … `layer_23` to `controller.expanded_override`
- `on_collapse_all()`: clear `controller.expanded_override`
- Updates SVG via `controller.current_svg`
- **File:** `ui.py` → new callbacks + layout
- **Deliverable:** Two buttons in the flowchart header toggle all layers at once

**Implementation note:** The `controller.collapsed` set should be replaced with `controller.expanded_override` (or added alongside it) as part of the Bug A fix. The default collapse state comes from `FlowchartNode.collapsed=True`; expansion is the override.

---

## Task 4.6 — Node Search / Highlight

**What:** A search box that filters and highlights nodes matching the query.

- Add `gr.Textbox(label="🔍 Search nodes", placeholder="e.g. attention, layer 5")`
- `on_node_search(query)` iterates `controller.node_tree`, matches against `node.label` and `node.node_type`
- Returns updated SVG with matching nodes highlighted (cyan border) and non-matching dimmed (opacity: 0.3)
- **File:** `ui.py` + `flowchart.py` (add `highlight_ids: set[str]` parameter to SVG renderer)
- **Deliverable:** Typing "attention" highlights all attention nodes and dims everything else

**Implementation note:** The `highlight_ids` parameter must be passed to `render_flowchart_svg()`. In the render loop, check each entry: if `highlight_ids` is non-empty and the node's ID is not in the set, render the `<rect>` with `opacity="0.3"`. If in the set, render with `stroke="#00BCD4"` (cyan) and `stroke-width="3"`. Clear the highlight when the search box is emptied.

---

## Task 4.7 — Keyboard Navigation

**What:** Navigate the node tree with arrow keys.

- Register a `keydown` event listener on the flowchart container
- Up/Down: traverse sibling nodes in document order
- Left/Right: collapse/expand group nodes
- Enter: select the currently highlighted node
- Forward selection into hidden Gradio textbox
- **File:** `ui.py` → new `_JS_NAV` script in `create_ui()`
- **Deliverable:** Focus the flowchart, use arrow keys to move between nodes

**Implementation note:** Build a flat ordered list of visible node IDs from the layout computation (return this alongside the SVG from `render_flowchart_svg()`). Store it in a `data-visible-nodes` attribute on the SVG or in a global JS variable. On keydown, find the current index in the list, update it, and highlight the new node with a CSS class (not via Gradio re-render — use local DOM manipulation to avoid the re-render flicker). On Enter, dispatch `flowchart-node-select` as usual.

---

## Task 4.8 — Auto-Scroll to Active Node

**What:** When a node is selected (by click, search, or keyboard), the flowchart scrolls to make it visible.

- JavaScript reads the active node's `<g>` bounding box via `document.getElementById('group-{nodeId}')`
- Calls `flowchartContainer.scrollTo({ top: nodeTop - 100, behavior: 'smooth' })`
- Triggered after SVG updates (via Gradio's `.then()` JS callback)
- **File:** `ui.py` → JS in `flowchart_html.change().then(...)`
- **Deliverable:** Selecting a deep layer auto-scrolls it into view

**Implementation note:** The SVG is fully replaced on each Gradio re-render, so DOM references are invalidated. The `.then()` callback must re-query the DOM for the newly-rendered node after the HTML update completes. Use `requestAnimationFrame` or a short `setTimeout` to ensure the DOM has settled.

---

## Task 4.9 — URL Hash State Persistence

**What:** Save the current node and step in the URL hash for bookmarkability.

- On every state change, update `window.location.hash` to `#node=layer_5_attention&step=3`
- On page load, parse the hash and restore state
- Use a lightweight polling or Gradio's `load` event
- **File:** `ui.py` → JS in `create_ui()`
- **Deliverable:** Copy URL → open in new tab → same node and step are shown

**Implementation notes:**
1. **Async timing:** Gradio's `load` event fires before the model is ready. Parse the hash, but only attempt to restore state after the first successful `_build_response()` call (check if `controller.ready` is true).
2. **Gradio router interference:** Gradio's internal SPA routing may overwrite `window.location.hash`. Use `history.replaceState` instead of direct hash assignment to avoid triggering Gradio's navigation.
3. Guard against restoring a `step` index that exceeds the current `explorer.num_steps()` — clamp to max.

---

## Task 4.10 — Edge Cases & Error Handling

**What:** Handle all the edge cases gracefully.

| Scenario | Behavior | Status |
|---|---|---|
| Click node before any generation | Show "Press Generate to start" message | ✅ Done (`_dispatch_plot` returns `_empty_fig`) |
| Click node after reset | No-op, show empty state | ✅ Done (`_empty_response()`) |
| Step slider when only 1 step exists | Slider has 0..0 (single value) | ⏳ Needs Task 4.4 first |
| Search finds no matches | Show "No matching nodes" overlay on SVG | ⏳ Needs Task 4.6 first |
| Click collapsed layer's child | Auto-expand parent before selecting child | 🔴 **Not implemented** — see Task 4.2 |
| Rapid clicking during generation | Debounce callback (ignore clicks while generating) | 🔴 **Not implemented** |
| Model loading in progress | Disable all interactive elements, show spinner | ⚠️ `_ensure_loaded()` is synchronous — Gradio handles this automatically, but buttons should be disabled via `interactive=False` during generation |

- **Files:** `ui.py` + `flowchart.py`
- **Deliverable:** No crashes on any click sequence; sensible messages everywhere

**Implementation notes for debouncing:** Use a `controller.generating: bool` flag. Set it to `True` before generation starts, `False` after. In `on_node_select` and `on_collapse_toggle`, return early if `controller.generating` is True. To make the flag visible to the UI, return `(svg, plot, title, desc)` unchanged (or a "generating..." state).

---

## Task 4.11 — Polish: Smooth Collapse Animations

**What:** Animate expand/collapse instead of instant hide/show.

**SVG constraint:** CSS `max-height` transitions **do not work** on SVG `<g>` elements. The `<g id="children-{nid}">` container is also **empty** — children are rendered outside it, so CSS on the container has no visual effect (`flowchart.py:714-717`).

**Revised approach — use Python re-render with a fade transition:**

1. Remove the empty `<g id="children-{nid}">` container and the JS `toggleCollapse` function's DOM manipulation (the visual toggle doesn't work anyway).
2. Keep the JS → Python event dispatch (arrow click → `flowchart-collapse-toggle` → hidden textbox → Python re-render).
3. Add a CSS `@keyframes` fade-in animation on the SVG element: `animation: fadeIn 0.15s ease-in`.
4. Apply the animation class to the `<svg>` element on re-render by appending a timestamp to the style attribute (force Gradio to re-inject the CSS).

**Alternative (if Python re-render is too slow):** Keep the JS container approach but fix it: render children *inside* the `children-{nid}` container (not inline). Then use `opacity` + `transform` transitions (not `max-height`). The transition would be: collapsed → `opacity: 0; transform: scaleY(0); max-height: 0`, expanded → `opacity: 1; transform: scaleY(1); max-height: 2000px`.

- **File:** `flowchart.py` → SVG output template + `_build_js()`
- **Deliverable:** Layers slide open/closed with a smooth animation

---

## Task 4.12 — Update `plan-phase3.md`

**What:** Refresh the plan document to mark Phase 4 tasks and update Phase 2 status.

- Add full Phase 4 task table
- Mark Phase 2.3 progress (SVG rendering engine)
- Note that Bug A (collapse toggle) exists and was identified during Phase 4 planning
- Update any stale references
- **File:** `plan-phase3.md`
- **Deliverable:** Plan document reflects current reality

---

## Suggested Execution Order

| Batch | Tasks | Why together |
|---|---|---|
| **Batch 0** | Bug A + Bug B + Bug C (pre-existing) | Must fix before any Phase 4 task works correctly |
| **Batch A** | 4.1, 4.2, 4.3 | Core SVG interactivity — makes clicking work |
| **Batch B** | 4.4, 4.5 | UI controls — slider + expand/collapse all |
| **Batch C** | 4.6, 4.7, 4.8 | Navigation — search, keyboard, scroll |
| **Batch D** | 4.9, 4.10, 4.11 | Polish — URL state, edge cases, animations |
| **Batch E** | 4.12 | Update plan document |

Each batch is self-contained and testable independently.
**Batch 0 is a hard prerequisite** — without fixing the collapse toggle logic, Tasks 4.2, 4.5, 4.7, and 4.10 cannot work.
