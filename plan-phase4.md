# Phase 4 Plan: Interactive Flowchart Clicking

**Goal:** Transform the static SVG flowchart into a fully interactive navigation tool — click any node to see its live plot, collapse/expand layer groups, step through tokens, and navigate the tree efficiently.

**Dependencies:** Phase 3 (ui.py) ✅, Phase 2.1–2.2 (node tree) ✅, Phase 2.3 (SVG rendering) ⚠️ partially done.

---

## Task 4.1 — SVG Node Interactivity (fully wire clicks in `flowchart.py`)

**What:** Ensure every rendered SVG `<g>` node fires the appropriate CustomEvent on click.

- Collapsible nodes (`layer_group`, `component_group`) fire `flowchart-collapse-toggle`
- Leaf nodes (all others) fire `flowchart-node-select`
- Add CSS cursor:pointer on all interactive nodes
- Add `:hover` style (lighten fill, subtle shadow)
- **File:** `flowchart.py` → `_render_node()` SVG template
- **Deliverable:** Click any SVG node → event reaches Gradio callback

---

## Task 4.2 — Active-Node Highlighting

**What:** When a node is selected, visually distinguish it in the SVG.

- Accept `active_node_id` parameter in `render_flowchart_svg()`
- Render the active node with `stroke: #ffd700; stroke-width: 3; filter: drop-shadow(...)`
- If active node is a child of a collapsed group, auto-expand the parent
- Add a `<defs>` SVG glow filter
- **Files:** `flowchart.py` + `ui.py` (controller.active_node_id)
- **Deliverable:** Selected node has a golden glow; parent layers auto-expand

---

## Task 4.3 — Hover Tooltips on Every Node

**What:** Show tensor shape and description when hovering over a node.

- Add `<title>` element inside each SVG `<g>` with shape + description
- For leaf nodes: `"{tensor_shape} — {description}"`
- For group nodes: `"Click to expand — contains N children"`
- **File:** `flowchart.py` → `_render_node()`
- **Deliverable:** Hovering over any node shows a native browser tooltip

---

## Task 4.4 — Token Step Slider

**What:** Let the user scrub through generated tokens.

- Add `gr.Slider(minimum=0, maximum=0, step=1, label="Token step")` to the top bar
- Wire `.change()` → `_build_response(token_idx=...)`
- Update slider max dynamically after Generate/Step
- Show "Token 5 of 20" label next to slider
- **File:** `ui.py` → `create_ui()` + callbacks
- **Deliverable:** Slider appears after generation, scrubbing updates both flowchart and detail plot

---

## Task 4.5 — Collapse / Expand All Button

**What:** One-click toggle to expand or collapse all 24 layers.

- Add a small button row: `[📂 Expand All] [📁 Collapse All]`
- `on_expand_all()` removes all `layer_0` … `layer_23` from `controller.collapsed`
- `on_collapse_all()` adds them all
- Updates SVG via `controller.current_svg`
- **File:** `ui.py` → new callbacks + layout
- **Deliverable:** Two buttons in the flowchart header toggle all layers at once

---

## Task 4.6 — Node Search / Highlight

**What:** A search box that filters and highlights nodes matching the query.

- Add `gr.Textbox(label="🔍 Search nodes", placeholder="e.g. attention, layer 5")`
- `on_node_search(query)` iterates `controller.node_tree`, matches against `node.label` and `node.node_type`
- Returns updated SVG with matching nodes highlighted (cyan border) and non-matching dimmed (opacity: 0.3)
- **File:** `ui.py` + `flowchart.py` (add `highlight_ids: set[str]` parameter to SVG renderer)
- **Deliverable:** Typing "attention" highlights all attention nodes and dims everything else

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

---

## Task 4.8 — Auto-Scroll to Active Node

**What:** When a node is selected (by click, search, or keyboard), the flowchart scrolls to make it visible.

- JavaScript reads the active node's `<g>` bounding box
- Calls `flowchartContainer.scrollTo({ top: nodeTop - 100, behavior: 'smooth' })`
- Triggered after SVG updates (via Gradio's `.then()` JS callback)
- **File:** `ui.py` → JS in `flowchart_html.change().then(...)`
- **Deliverable:** Selecting a deep layer auto-scrolls it into view

---

## Task 4.9 — URL Hash State Persistence

**What:** Save the current node and step in the URL hash for bookmarkability.

- On every state change, update `window.location.hash` to `#node=layer_5_attention&step=3`
- On page load, parse the hash and restore state
- Use a lightweight polling or Gradio's `load` event
- **File:** `ui.py` → JS in `create_ui()`
- **Deliverable:** Copy URL → open in new tab → same node and step are shown

---

## Task 4.10 — Edge Cases & Error Handling

**What:** Handle all the edge cases gracefully.

| Scenario | Behavior |
|---|---|
| Click node before any generation | Show "Press Generate to start" message |
| Click node after reset | No-op, show empty state |
| Step slider when only 1 step exists | Slider has 0..0 (single value) |
| Search finds no matches | Show "No matching nodes" overlay on SVG |
| Click collapsed layer's child | Auto-expand parent before selecting child |
| Rapid clicking during generation | Debounce callback (ignore clicks while generating) |
| Model loading in progress | Disable all interactive elements, show spinner |

- **Files:** `ui.py` + `flowchart.py`
- **Deliverable:** No crashes on any click sequence; sensible messages everywhere

---

## Task 4.11 — Polish: Smooth CSS Collapse Animations

**What:** Animate expand/collapse instead of instant hide/show.

- SVG children groups get `class="collapsible"`
- CSS `transition: max-height 0.3s ease, opacity 0.2s ease`
- Use `max-height` trick for smooth expand (set to a large value when expanded)
- **File:** `flowchart.py` → SVG output template
- **Deliverable:** Layers slide open/closed with a smooth animation

---

## Task 4.12 — Update `plan-phase3.md`

**What:** Refresh the plan document to mark Phase 4 tasks and update Phase 2 status.

- Add full Phase 4 task table
- Mark Phase 2.3 progress (SVG rendering engine)
- Update any stale references
- **File:** `plan-phase3.md`
- **Deliverable:** Plan document reflects current reality

---

## Suggested Execution Order

| Batch | Tasks | Why together |
|---|---|---|
| **Batch A** | 4.1, 4.2, 4.3 | Core SVG interactivity — makes clicking work |
| **Batch B** | 4.4, 4.5 | UI controls — slider + expand/collapse all |
| **Batch C** | 4.6, 4.7, 4.8 | Navigation — search, keyboard, scroll |
| **Batch D** | 4.9, 4.10, 4.11 | Polish — URL state, edge cases, animations |
| **Batch E** | 4.12 | Update plan document |

Each batch is self-contained and testable independently.
