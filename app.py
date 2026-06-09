#!/usr/bin/env python3
"""
Qwen2.5-0.5B Visual Step-Through Explorer
===========================================
Thin entry point — loads the engine, builds the UI, and launches Gradio.

Usage:
    pip install torch transformers gradio plotly numpy
    python app.py [--port 7860] [--share] [--model Qwen/Qwen2.5-0.5B]
"""

from __future__ import annotations

import argparse

from engine import QwenExplorer
from ui import UIController, controller, create_ui


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-0.5B Visual Step-Through Explorer"
    )
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link (Gradio)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Qwen2.5-0.5B Visual Step-Through Explorer")
    print("=" * 60)
    print(f"  Model : {args.model}")
    print(f"  Port  : {args.port}")
    print(f"  Share : {args.share}")
    print()
    print(f"  Open http://localhost:{args.port} in your browser.")
    print("=" * 60)
    print()

    # Pre-load model
    controller.explorer = QwenExplorer(args.model)
    controller.explorer.load()
    controller.ready = True

    ui = create_ui()
    ui.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
