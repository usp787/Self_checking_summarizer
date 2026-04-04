"""
clean_colab_notebook.py
-----------------------
Strip Colab-specific metadata and ipywidget outputs from a .ipynb file
so it renders correctly on GitHub.

Usage:
    python clean_colab_notebook.py notebook.ipynb
    python clean_colab_notebook.py notebook.ipynb --output clean_notebook.ipynb
    python clean_colab_notebook.py *.ipynb          # glob multiple files
"""

import argparse
import json
import sys
from pathlib import Path

# Top-level metadata keys added by Colab that GitHub doesn't need
COLAB_METADATA_KEYS = {"colab", "accelerator", "widgets"}


def clean_notebook(path: Path, output: Path | None = None) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))

    # --- 1. Strip Colab / widget top-level metadata ---
    removed_keys = [k for k in COLAB_METADATA_KEYS if k in nb["metadata"]]
    for key in removed_keys:
        del nb["metadata"][key]

    # --- 2. Fix cell outputs: replace widget-view with text fallback ---
    widget_outputs_removed = 0
    for cell in nb.get("cells", []):
        new_outputs = []
        for cell_output in cell.get("outputs", []):
            data = cell_output.get("data", {})
            if "application/vnd.jupyter.widget-view+json" in data:
                widget_outputs_removed += 1
                text = data.get("text/plain", "")
                if text:
                    new_outputs.append({
                        "output_type": "display_data",
                        "data": {"text/plain": text},
                        "metadata": {},
                    })
                # else drop the output entirely (no useful fallback)
            else:
                new_outputs.append(cell_output)
        cell["outputs"] = new_outputs

    # --- 3. Write result ---
    dest = output or path
    dest.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[{path.name}] cleaned:")
    if removed_keys:
        print(f"  - removed metadata keys: {removed_keys}")
    else:
        print("  - no Colab metadata keys found")
    print(f"  - replaced/dropped {widget_outputs_removed} widget output(s)")
    print(f"  - saved to: {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean Colab notebooks for GitHub.")
    parser.add_argument("notebooks", nargs="+", type=Path, help=".ipynb file(s)")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output path (only valid when a single notebook is given)"
    )
    args = parser.parse_args()

    if args.output and len(args.notebooks) > 1:
        print("Error: --output can only be used with a single input file.", file=sys.stderr)
        sys.exit(1)

    for nb_path in args.notebooks:
        if not nb_path.exists():
            print(f"Warning: {nb_path} not found, skipping.", file=sys.stderr)
            continue
        clean_notebook(nb_path, args.output)


if __name__ == "__main__":
    main()
