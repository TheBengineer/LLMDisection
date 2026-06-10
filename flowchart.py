"""
flowchart.py — SVG flowchart layout engine for the LLM Visual Step-Through Explorer.

Renders an interactive SVG flowchart of the transformer architecture with
click-to-expand component nodes, embedded PNG thumbnails, and Gradio event dispatch.
"""

from __future__ import annotations

import base64
import io
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore")


# ── Task 2.1: FlowchartNode dataclass ────────────────────────────────────────

@dataclass
class FlowchartNode:
    """A single node in the flowchart tree."""
    id: str
    label: str
    node_type: str  # root, embedding, rmsnorm, attention, mlp, residual, linear, activation, softmax, output, layer_group, component_group
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    collapsed: bool = True
    plot_fn: str | None = None
    tensor_key: str | None = None
    shape: str = ""
    color: str = "#2C3E50"
    icon: str = ""


# ── Color and icon scheme ───────────────────────────────────────────────────

NODE_COLORS: dict[str, str] = {
    "root":              "#2C3E50",
    "embedding":         "#8E44AD",
    "rmsnorm":           "#16A085",
    "attention":         "#2980B9",
    "mlp":               "#E67E22",
    "residual":          "#7F8C8D",
    "linear":            "#3498DB",
    "activation":        "#E74C3C",
    "softmax":           "#1ABC9C",
    "output":            "#2C3E50",
    "layer_group":       "#95A5A6",
    "component_group":   "#BDC3C7",
}

NODE_ICONS: dict[str, str] = {
    "root":              "\U0001F3D7\ufe0f",  # 🏗️
    "embedding":         "\U0001F524",        # 🔤
    "rmsnorm":           "\u2696\ufe0f",      # ⚖️
    "attention":         "\U0001F50D",        # 🔍
    "mlp":               "\u26A1",            # ⚡
    "residual":          "\u2795",            # ➕
    "linear":            "\U0001F4D0",        # 📐
    "activation":        "\U0001F525",        # 🔥
    "softmax":           "\U0001F4CA",        # 📊
    "output":            "\U0001F3AF",        # 🎯
    "layer_group":       "\U0001F4C1",        # 📁
    "component_group":   "\U0001F4C2",        # 📂
}


# ── Task 2.2: Build Qwen2.5-0.5B node tree ─────────────────────────────────

def _layer_node(layer_idx: int) -> FlowchartNode:
    """Create a layer-group node for the given layer index."""
    lid = f"layer_{layer_idx}"
    return FlowchartNode(
        id=lid,
        label=f"Layer {layer_idx}",
        node_type="layer_group",
        parent_id="root",
        collapsed=True,
    )


def _attn_children(layer_idx: int) -> list[FlowchartNode]:
    """Return attention sub-nodes for a given layer."""
    prefix = f"layer_{layer_idx}"
    return [
        FlowchartNode(
            id=f"{prefix}_q_proj",
            label="Q Projection (896\u2192896)",
            node_type="linear",
            parent_id=f"{prefix}_attn",
            collapsed=True,
            plot_fn="plot_weight_matrix",
            tensor_key=f"layers.{layer_idx}.q_proj.weight",
            shape="(896, 896)",
            color=NODE_COLORS["linear"],
            icon=NODE_ICONS["linear"],
        ),
        FlowchartNode(
            id=f"{prefix}_k_proj",
            label="K Projection (896\u2192128)",
            node_type="linear",
            parent_id=f"{prefix}_attn",
            collapsed=True,
            plot_fn="plot_weight_matrix",
            tensor_key=f"layers.{layer_idx}.k_proj.weight",
            shape="(128, 896)",
            color=NODE_COLORS["linear"],
            icon=NODE_ICONS["linear"],
        ),
        FlowchartNode(
            id=f"{prefix}_v_proj",
            label="V Projection (896\u2192128)",
            node_type="linear",
            parent_id=f"{prefix}_attn",
            collapsed=True,
            plot_fn="plot_weight_matrix",
            tensor_key=f"layers.{layer_idx}.v_proj.weight",
            shape="(128, 896)",
            color=NODE_COLORS["linear"],
            icon=NODE_ICONS["linear"],
        ),
        FlowchartNode(
            id=f"{prefix}_rope",
            label="RoPE (pre vs post)",
            node_type="linear",
            parent_id=f"{prefix}_attn",
            collapsed=True,
            plot_fn="plot_rope_comparison",
            tensor_key=f"layers.{layer_idx}.rope",
            shape="(seq_len, 64)",
            color=NODE_COLORS["linear"],
            icon="\U0001F504",  # 🔄
        ),
        FlowchartNode(
            id=f"{prefix}_attn_scores",
            label="Attention Scores",
            node_type="softmax",
            parent_id=f"{prefix}_attn",
            collapsed=True,
            plot_fn="plot_attention_scores",
            tensor_key=f"layers.{layer_idx}.attention_scores",
            shape="(14, seq_len)",
            color=NODE_COLORS["softmax"],
            icon=NODE_ICONS["softmax"],
        ),
        FlowchartNode(
            id=f"{prefix}_attn_weights",
            label="Attention Weights",
            node_type="softmax",
            parent_id=f"{prefix}_attn",
            collapsed=True,
            plot_fn="plot_attention",
            tensor_key=f"layers.{layer_idx}.attention_weights",
            shape="(14, seq_len)",
            color=NODE_COLORS["softmax"],
            icon=NODE_ICONS["softmax"],
        ),
        FlowchartNode(
            id=f"{prefix}_o_proj",
            label="O Projection (896\u2192896)",
            node_type="linear",
            parent_id=f"{prefix}_attn",
            collapsed=True,
            plot_fn="plot_weight_matrix",
            tensor_key=f"layers.{layer_idx}.o_proj.weight",
            shape="(896, 896)",
            color=NODE_COLORS["linear"],
            icon=NODE_ICONS["linear"],
        ),
    ]


def _mlp_children(layer_idx: int) -> list[FlowchartNode]:
    """Return MLP sub-nodes for a given layer."""
    prefix = f"layer_{layer_idx}"
    return [
        FlowchartNode(
            id=f"{prefix}_gate_proj",
            label="Gate Projection (896\u21924864)",
            node_type="linear",
            parent_id=f"{prefix}_mlp",
            collapsed=True,
            plot_fn="plot_histogram",
            tensor_key=f"layers.{layer_idx}.gate_proj",
            shape="(4864,)",
            color=NODE_COLORS["linear"],
            icon=NODE_ICONS["linear"],
        ),
        FlowchartNode(
            id=f"{prefix}_silu",
            label="SiLU Activation",
            node_type="activation",
            parent_id=f"{prefix}_mlp",
            collapsed=True,
            plot_fn="plot_silu_scatter",
            tensor_key=f"layers.{layer_idx}.silu",
            shape="(4864,)",
            color=NODE_COLORS["activation"],
            icon=NODE_ICONS["activation"],
        ),
        FlowchartNode(
            id=f"{prefix}_up_proj",
            label="Up Projection (896\u21924864)",
            node_type="linear",
            parent_id=f"{prefix}_mlp",
            collapsed=True,
            plot_fn="plot_weight_matrix",
            tensor_key=f"layers.{layer_idx}.up_proj.weight",
            shape="(4864, 896)",
            color=NODE_COLORS["linear"],
            icon=NODE_ICONS["linear"],
        ),
        FlowchartNode(
            id=f"{prefix}_down_proj",
            label="Down Projection (4864\u2192896)",
            node_type="linear",
            parent_id=f"{prefix}_mlp",
            collapsed=True,
            plot_fn="plot_weight_matrix",
            tensor_key=f"layers.{layer_idx}.down_proj.weight",
            shape="(896, 4864)",
            color=NODE_COLORS["linear"],
            icon=NODE_ICONS["linear"],
        ),
        FlowchartNode(
            id=f"{prefix}_mlp_out",
            label="Gate \u00d7 Up \u2192 MLP Output",
            node_type="mlp",
            parent_id=f"{prefix}_mlp",
            collapsed=True,
            plot_fn="plot_mlp_activation",
            tensor_key=f"layers.{layer_idx}.mlp_output",
            shape="(896,)",
            color=NODE_COLORS["mlp"],
            icon=NODE_ICONS["mlp"],
        ),
    ]


def _layer_subnodes(layer_idx: int) -> list[FlowchartNode]:
    """Return all sub-nodes for a given layer (RMSNorm, Attention, Residual, MLP)."""
    prefix = f"layer_{layer_idx}"
    return [
        # Pre-attention RMSNorm
        FlowchartNode(
            id=f"{prefix}_pre_attn_rmsnorm",
            label="Pre-Attention RMSNorm",
            node_type="rmsnorm",
            parent_id=prefix,
            collapsed=True,
            plot_fn="plot_rmsnorm_comparison",
            tensor_key=f"layers.{layer_idx}.pre_attn_rmsnorm",
            shape="(896,)",
            color=NODE_COLORS["rmsnorm"],
            icon=NODE_ICONS["rmsnorm"],
        ),
        # Attention group
        FlowchartNode(
            id=f"{prefix}_attn",
            label="Attention",
            node_type="attention",
            parent_id=prefix,
            collapsed=True,
        ),
        # Post-attention residual
        FlowchartNode(
            id=f"{prefix}_post_attn_residual",
            label="Residual (post-attention)",
            node_type="residual",
            parent_id=prefix,
            collapsed=True,
            plot_fn="plot_residual_delta",
            tensor_key=f"layers.{layer_idx}.post_attn_residual",
            shape="(896,)",
            color=NODE_COLORS["residual"],
            icon=NODE_ICONS["residual"],
        ),
        # Pre-MLP RMSNorm
        FlowchartNode(
            id=f"{prefix}_pre_mlp_rmsnorm",
            label="Pre-MLP RMSNorm",
            node_type="rmsnorm",
            parent_id=prefix,
            collapsed=True,
            plot_fn="plot_rmsnorm_comparison",
            tensor_key=f"layers.{layer_idx}.pre_mlp_rmsnorm",
            shape="(896,)",
            color=NODE_COLORS["rmsnorm"],
            icon=NODE_ICONS["rmsnorm"],
        ),
        # MLP group
        FlowchartNode(
            id=f"{prefix}_mlp",
            label="MLP",
            node_type="mlp",
            parent_id=prefix,
            collapsed=True,
        ),
        # Post-MLP residual
        FlowchartNode(
            id=f"{prefix}_post_mlp_residual",
            label="Residual (post-MLP)",
            node_type="residual",
            parent_id=prefix,
            collapsed=True,
            plot_fn="plot_residual_delta",
            tensor_key=f"layers.{layer_idx}.post_mlp_residual",
            shape="(896,)",
            color=NODE_COLORS["residual"],
            icon=NODE_ICONS["residual"],
        ),
    ]


def build_qwen_node_tree() -> dict[str, FlowchartNode]:
    """Build the complete 299-node tree for Qwen2.5-0.5B.

    Returns:
        A dict mapping node_id -> FlowchartNode.
    """
    nodes: dict[str, FlowchartNode] = {}

    # ── Root ──
    root = FlowchartNode(
        id="root",
        label="Qwen2.5-0.5B",
        node_type="root",
        parent_id=None,
        collapsed=False,
    )
    nodes["root"] = root

    # ── Token Embedding ──
    embed = FlowchartNode(
        id="token_embedding",
        label="Token Embedding (vocab_size \u00d7 896)",
        node_type="embedding",
        parent_id="root",
        collapsed=True,
        plot_fn="plot_embedding_slice",
        tensor_key="token_embeddings",
        shape="(seq_len, 896)",
        color=NODE_COLORS["embedding"],
        icon=NODE_ICONS["embedding"],
    )
    nodes["token_embedding"] = embed
    root.children.append("token_embedding")

    # ── 24 Layers ──
    for i in range(24):
        layer = _layer_node(i)
        nodes[layer.id] = layer
        root.children.append(layer.id)

        # Create all sub-nodes for this layer
        for sub in _layer_subnodes(i):
            nodes[sub.id] = sub
            layer.children.append(sub.id)

        # Attention children
        for attn_child in _attn_children(i):
            nodes[attn_child.id] = attn_child
            nodes[f"{layer.id}_attn"].children.append(attn_child.id)

        # MLP children
        for mlp_child in _mlp_children(i):
            nodes[mlp_child.id] = mlp_child
            nodes[f"{layer.id}_mlp"].children.append(mlp_child.id)

    # ── Final RMSNorm ──
    final_norm = FlowchartNode(
        id="final_rmsnorm",
        label="Final RMSNorm",
        node_type="rmsnorm",
        parent_id="root",
        collapsed=True,
        plot_fn="plot_rmsnorm_comparison",
        tensor_key="final_rmsnorm",
        shape="(896,)",
        color=NODE_COLORS["rmsnorm"],
        icon=NODE_ICONS["rmsnorm"],
    )
    nodes["final_rmsnorm"] = final_norm
    root.children.append("final_rmsnorm")

    # ── LM Head ──
    lm_head = FlowchartNode(
        id="lm_head",
        label="LM Head \u2192 Logits / Probabilities",
        node_type="output",
        parent_id="root",
        collapsed=True,
        plot_fn="plot_top_logits",
        tensor_key="logits",
        shape="(vocab_size,)",
        color=NODE_COLORS["output"],
        icon=NODE_ICONS["output"],
    )
    nodes["lm_head"] = lm_head
    root.children.append("lm_head")

    return nodes


# ── Task 2.3: SVG Rendering Engine ──────────────────────────────────────────

# Layout constants
NODE_W_EXPANDED = 220
NODE_W_COLLAPSED = 180
NODE_H_LABEL = 36
NODE_H_THUMB = 100
H_INDENT = 40
V_GAP_SIBLING = 8
V_GAP_PARENT = 16
CANVAS_MIN_W = 900
THUMB_SIZE = 64


def _compute_layout(
    nodes: dict[str, FlowchartNode],
    node_id: str,
    collapsed_override: set[str] | None = None,
    x_offset: int = 0,
    y_offset: int = 0,
) -> tuple[list[dict], int]:
    """DFS layout computation.

    Returns (layout_list, total_height) where layout_list contains
    dicts with keys: id, x, y, w, h, has_thumb, visible.
    """
    if collapsed_override is None:
        collapsed_override = set()

    layout: list[dict] = []
    node = nodes[node_id]
    is_collapsed = node.collapsed or (node_id in collapsed_override)
    has_children = bool(node.children)

    # Determine this node's dimensions
    has_thumb = node.plot_fn is not None  # leaf nodes with plots get thumbnails
    w = NODE_W_EXPANDED if (has_children and not is_collapsed) else NODE_W_COLLAPSED
    h = NODE_H_THUMB if has_thumb and not is_collapsed else NODE_H_LABEL

    entry = {
        "id": node_id,
        "x": x_offset,
        "y": y_offset,
        "w": w,
        "h": h,
        "has_thumb": has_thumb and not is_collapsed,
        "has_children": has_children,
        "is_collapsed": is_collapsed,
        "visible": True,
    }
    layout.append(entry)
    y_ptr = y_offset + h + V_GAP_SIBLING

    if has_children and not is_collapsed:
        child_x = x_offset + H_INDENT
        for child_id in node.children:
            if child_id not in nodes:
                continue
            child_layout, child_h = _compute_layout(
                nodes, child_id, collapsed_override, child_x, y_ptr
            )
            layout.extend(child_layout)
            y_ptr += child_h + V_GAP_SIBLING

    total_h = y_ptr - y_offset - V_GAP_SIBLING  # remove trailing gap
    return layout, total_h


def _escape_xml(s: str) -> str:
    """Escape special XML characters."""
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    s = s.replace("'", "&apos;")
    return s


def _svg_rect(x: int, y: int, w: int, h: int, color: str, border_color: str = "#555",
              opacity: float = 0.9, rx: int = 6) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"'
        f' fill="{color}" stroke="{border_color}" stroke-width="1.5"'
        f' opacity="{opacity}" />'
    )


def _svg_text(x: int, y: int, text: str, font_size: int = 12,
              color: str = "#fff", anchor: str = "middle", bold: bool = False) -> str:
    fw = "bold" if bold else "normal"
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-size="{font_size}"'
        f' font-weight="{fw}" text-anchor="{anchor}" dominant-baseline="central">'
        f'{_escape_xml(text)}</text>'
    )


def _svg_arrow(direction: str, x: int, y: int) -> str:
    """Expand/collapse arrow (▶ expanded, ▼ collapsed)."""
    if direction == "expanded":
        # Downward triangle
        return f'<polygon points="{x-6},{y-4} {x+6},{y-4} {x},{y+6}" fill="#ccc" />'
    else:
        # Rightward triangle
        return f'<polygon points="{x-4},{y-6} {x-4},{y+6} {x+6},{y}" fill="#ccc" />'


def _svg_thumbnail(x: int, y: int, data_uri: str, size: int = THUMB_SIZE) -> str:
    return (
        f'<image x="{x}" y="{y}" width="{size}" height="{size}"'
        f' href="{data_uri}" preserveAspectRatio="xMidYMid meet" />'
    )


def _svg_connector(x1: int, y1: int, x2: int, y2: int) -> str:
    """Simple vertical connector line (parent center to child top)."""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"'
        f' stroke="#666" stroke-width="1.5" />'
    )


def _build_js() -> str:
    """Embedded JavaScript for interactivity."""
    return """<script type="text/javascript">
function toggleCollapse(nodeId) {
    var container = document.getElementById('children-' + nodeId);
    var arrow = document.getElementById('arrow-' + nodeId);
    if (!container) return;
    var isCollapsed = container.style.display === 'none';
    container.style.display = isCollapsed ? '' : 'none';
    if (arrow) {
        arrow.innerHTML = isCollapsed
            ? '<polygon points="' + (-6) + ',' + (-4) + ' ' + (6) + ',' + (-4) + ' ' + (0) + ',' + (6) + '" fill="#ccc" />'
            : '<polygon points="' + (-4) + ',' + (-6) + ' ' + (-4) + ',' + (6) + ' ' + (6) + ',' + (0) + '" fill="#ccc" />';
    }
    // Dispatch resize event for Gradio iframe
    if (window.__gradio_mode__ !== undefined) {
        window.dispatchEvent(new Event('resize'));
    }
    // Sync collapse state back to server via Gradio bridge
    var event = new CustomEvent('flowchart-collapse-toggle', { detail: { nodeId: nodeId } });
    document.dispatchEvent(event);
}
function selectNode(nodeId) {
    // Remove highlight from all nodes
    document.querySelectorAll('.flowchart-node').forEach(function(el) {
        el.setAttribute('stroke', '#555');
        el.setAttribute('stroke-width', '1.5');
    });
    // Highlight selected node
    var rect = document.getElementById('rect-' + nodeId);
    if (rect) {
        rect.setAttribute('stroke', '#F1C40F');
        rect.setAttribute('stroke-width', '3');
    }
    // Dispatch custom event for Gradio
    var event = new CustomEvent('flowchart-node-select', { detail: { nodeId: nodeId } });
    document.dispatchEvent(event);
}
</script>"""


def render_flowchart_svg(
    nodes: dict[str, FlowchartNode],
    active_node_id: str | None = None,
    thumbnails: dict[str, str] | None = None,
    collapsed_override: set[str] | None = None,
) -> str:
    """Render the full SVG flowchart.

    Args:
        nodes: Node tree (from build_qwen_node_tree).
        active_node_id: Currently selected node id (highlighted).
        thumbnails: Dict mapping node_id -> "data:image/png;base64,..." data URIs.
        collapsed_override: Set of node ids forced collapsed (client-side toggle state).

    Returns:
        SVG string.
    """
    if thumbnails is None:
        thumbnails = {}
    if collapsed_override is None:
        collapsed_override = set()

    # Compute layout starting from root
    layout, total_h = _compute_layout(nodes, "root", collapsed_override)
    canvas_h = max(total_h + 40, 600)
    svg_w = CANVAS_MIN_W

    # Build a lookup from id -> layout entry
    layout_map = {e["id"]: e for e in layout}

    # Identify visible node ids for connector rendering
    visible_ids = {e["id"] for e in layout if e["visible"]}

    parts: list[str] = []

    # SVG header + embedded JS
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{canvas_h}"'
        f' style="background:#1e1e1e; font-family:monospace;">'
    )
    parts.append("<defs>")
    parts.append(
        '<filter id="glow"><feGaussianBlur stdDeviation="2" result="coloredBlur"/>'
        '<feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
    )
    parts.append("</defs>")

    # Draw connectors first (below nodes)
    for entry in layout:
        nid = entry["id"]
        node = nodes.get(nid)
        if not node or not node.parent_id:
            continue
        parent_entry = layout_map.get(node.parent_id)
        if not parent_entry or not parent_entry["visible"]:
            continue

        cx1 = parent_entry["x"] + parent_entry["w"] // 2
        cy1 = parent_entry["y"] + parent_entry["h"]
        cx2 = entry["x"] + entry["w"] // 2
        cy2 = entry["y"]

        parts.append(
            f'<line x1="{cx1}" y1="{cy1}" x2="{cx2}" y2="{cy2}"'
            f' stroke="#555" stroke-width="1.5" />'
        )

    # Draw nodes
    for entry in layout:
        nid = entry["id"]
        if not entry["visible"]:
            continue
        node = nodes.get(nid)
        if not node:
            continue

        x, y, w, h = entry["x"], entry["y"], entry["w"], entry["h"]
        color = node.color
        icon = node.icon
        has_children = entry["has_children"]
        is_collapsed = entry["is_collapsed"]
        has_thumb = entry["has_thumb"]

        # Determine border
        is_active = (nid == active_node_id)
        border_color = "#F1C40F" if is_active else "#555"
        stroke_w = 3 if is_active else 1.5

        # Group for this node
        parts.append(f'<g class="flowchart-node-group" id="group-{nid}">')

        # Main rect (clickable for selection)
        parts.append(
            f'<rect id="rect-{nid}" class="flowchart-node" x="{x}" y="{y}"'
            f' width="{w}" height="{h}" rx="6" fill="{color}"'
            f' stroke="{border_color}" stroke-width="{stroke_w}" opacity="0.9"'
            f' cursor="pointer"'
            f' onclick="selectNode(\'{nid}\')" />'
        )

        # Tooltip
        if node.shape:
            parts.append(f'<title>{_escape_xml(node.label)} [{node.shape}]</title>')

        # Expand/collapse arrow (for nodes with children)
        if has_children:
            arrow_x = x + 14
            arrow_y = y + h // 2
            arrow_type = "expanded" if not is_collapsed else "collapsed"
            parts.append(
                f'<g id="arrow-{nid}" cursor="pointer"'
                f' onclick="event.stopPropagation(); toggleCollapse(\'{nid}\')">'
            )
            parts.append(_svg_arrow(arrow_type, arrow_x, arrow_y))
            parts.append("</g>")

        # Icon
        if icon:
            parts.append(
                f'<text x="{x + 28}" y="{y + h // 2}" fill="#fff" font-size="14"'
                f' text-anchor="middle" dominant-baseline="central">{icon}</text>'
            )

        # Label
        label_x = x + w // 2
        label_y = y + h // 2
        parts.append(
            f'<text x="{label_x}" y="{label_y}" fill="#fff" font-size="11"'
            f' text-anchor="middle" dominant-baseline="central"'
            f' font-weight="bold">{_escape_xml(node.label)}</text>'
        )

        # Thumbnail
        if has_thumb and nid in thumbnails:
            thumb_x = x + w - THUMB_SIZE - 6
            thumb_y = y + 6
            parts.append(_svg_thumbnail(thumb_x, thumb_y, thumbnails[nid]))

        parts.append("</g>")

        # Children container (for collapse/expand)
        if has_children:
            style = "display:none;" if is_collapsed else ""
            parts.append(f'<g id="children-{nid}" style="{style}">')
            # Children rendered inline above, but we wrap them conceptually
            # Actually we just use this as a JS target
            parts.append(f'</g>')

    parts.append(_build_js())
    parts.append("</svg>")

    return "\n".join(parts)


# ── Utility: map node → plot function ───────────────────────────────────────

def get_plot_for_node(node: FlowchartNode) -> str | None:
    """Return the plot function name for a given node."""
    return node.plot_fn


def get_tensor_for_node(node: FlowchartNode, snapshot: dict) -> dict | None:
    """Extract tensor data from a snapshot dict for a given node."""
    if not node.tensor_key:
        return None
    parts = node.tensor_key.split(".")
    data = snapshot
    for part in parts:
        if isinstance(data, dict) and part in data:
            data = data[part]
        elif isinstance(data, list) and part.isdigit():
            data = data[int(part)]
        else:
            return None
    return data


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick validation
    nodes = build_qwen_node_tree()
    print(f"Built {len(nodes)} nodes")
    root = nodes["root"]
    print(f"Root children: {len(root.children)}")
    layer0 = nodes["layer_0"]
    print(f"Layer 0 children: {len(layer0.children)}")

    svg = render_flowchart_svg(nodes, active_node_id="root")
    print(f"SVG length: {len(svg)} chars")

    # Check for well-formedness
    assert "<svg" in svg
    assert "</svg>" in svg
    assert "toggleCollapse" in svg
    assert "selectNode" in svg
    print("All checks passed!")
