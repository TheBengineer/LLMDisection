# Phase 3 — Two-Panel Flowchart Layout

Replace the tabbed layout with an interactive SVG flowchart (left panel) + detail view (right panel).

## Subtasks

- [ ] **3.1 — Two-panel layout structure**: Replace tabs with `gr.Row(scale=35:65)` — flowchart left, detail right. Define the 4-5 output components.
- [ ] **3.2 — Node tree state in UIController**: Build `build_qwen_node_tree()` on init, track `active_node_id`, thumbnail cache, collapsed set.
- [ ] **3.3 — SVG rendering integration**: Call `render_flowchart_svg()` with live state to generate the HTML output.
- [ ] **3.4 — Plot dispatch**: Map `node.plot_fn` → the right `plots.py` function, calling it with the current snapshot.
- [ ] **3.5 — Node descriptions**: Return human-readable descriptions per node type (from flowchart.py metadata).
- [ ] **3.6 — New callbacks**: `on_generate()`, `on_step()`, `on_node_select()`, `on_collapse_toggle()` returning 4-5 outputs.
- [ ] **3.7 — Wire up UI events + JS bridge**: Connect Gradio `.click()` and `.change()`, add JS listener for `flowchart-node-select`.
- [ ] **3.8 — Polish & edge cases**: Handle empty state, loading, CSS styling.
- [ ] **3.9 — Verify app.py integration**: Ensure app.py still works with new ui.py exports.
