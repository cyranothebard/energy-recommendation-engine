#!/usr/bin/env python3
"""
Utility script to create a directory for visualizations and optionally save a demo
visualization as a JPEG image.

Supported engines:
- matplotlib: native JPEG export (no extra deps)
- plotly: JPEG export requires 'kaleido' (pip install kaleido)

Examples:
- Create default 'visualizations' directory only:
  python3 scripts/save_visualizations.py

- Create custom directory and save a matplotlib demo JPEG:
  python3 scripts/save_visualizations.py --dir visualizations --engine matplotlib --demo

- Save a Plotly demo JPEG (requires kaleido):
  python3 scripts/save_visualizations.py --engine plotly --demo

- Choose custom filename and overwrite if exists:
  python3 scripts/save_visualizations.py --name my_chart --demo --overwrite
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional


def ensure_directory(directory_path: Path) -> Path:
    """Ensure the visualization directory exists (create if missing)."""
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


def save_matplotlib_figure_as_jpeg(figure, output_path: Path, dpi: int = 150) -> Path:
    """Save a matplotlib figure to JPEG at the specified path."""
    output_path = output_path.with_suffix(".jpeg") if output_path.suffix == "" else output_path
    # Lazy import to avoid forcing matplotlib dependency if not used by caller
    import matplotlib.pyplot as plt  # noqa: F401  (ensures backend is loaded)

    figure.savefig(output_path, format="jpeg", dpi=dpi, bbox_inches="tight")
    return output_path


def save_plotly_figure_as_jpeg(figure, output_path: Path, scale: float = 2.0) -> Path:
    """Save a Plotly figure to JPEG at the specified path. Requires kaleido."""
    try:
        import plotly.io as pio  # type: ignore
        # Accessing pio.kaleido triggers helpful error if kaleido is missing
        _ = getattr(pio, "kaleido", None)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Saving Plotly figures as JPEG requires 'kaleido'. Install with: pip install kaleido"
        ) from exc

    output_path = output_path.with_suffix(".jpeg") if output_path.suffix == "" else output_path
    figure.write_image(str(output_path), format="jpeg", scale=scale)
    return output_path


def create_demo_matplotlib_figure():
    """Create a small demo matplotlib chart for verification."""
    import matplotlib.pyplot as plt
    import numpy as np

    x_values = np.linspace(0, 2 * np.pi, 200)
    y_values = np.sin(x_values)

    figure, axes = plt.subplots(figsize=(6, 4))
    axes.plot(x_values, y_values, label="sin(x)")
    axes.set_title("Demo Visualization (matplotlib)")
    axes.set_xlabel("x")
    axes.set_ylabel("sin(x)")
    axes.grid(True, alpha=0.3)
    axes.legend()
    return figure


def create_demo_plotly_figure() -> "object":
    """Create a small demo Plotly chart for verification."""
    import plotly.express as px
    import pandas as pd

    frame = pd.DataFrame({
        "x": list(range(10)),
        "y": [value * value for value in range(10)],
    })
    return px.line(frame, x="x", y="y", title="Demo Visualization (Plotly)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a directory for visualizations and optionally save a demo JPEG."
    )
    parser.add_argument(
        "--dir",
        dest="directory",
        default="visualizations",
        help="Directory where visualizations will be saved (default: visualizations)",
    )
    parser.add_argument(
        "--name",
        dest="name",
        default=None,
        help="Base filename without extension (default: auto-generated)",
    )
    parser.add_argument(
        "--engine",
        dest="engine",
        choices=["matplotlib", "plotly"],
        default="matplotlib",
        help="Visualization engine to use for demo save (default: matplotlib)",
    )
    parser.add_argument(
        "--demo",
        dest="demo",
        action="store_true",
        help="Generate and save a demo visualization JPEG using the selected engine",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing file if the same name exists",
    )
    parser.add_argument(
        "--dpi",
        dest="dpi",
        type=int,
        default=150,
        help="JPEG DPI for matplotlib saves (default: 150)",
    )
    parser.add_argument(
        "--scale",
        dest="scale",
        type=float,
        default=2.0,
        help="Scale factor for Plotly JPEG output (default: 2.0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = ensure_directory(Path(args.directory))

    if args.name:
        base_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"viz_{args.engine}_{timestamp}"

    output_path = output_dir / base_name

    # Collision check if not overwriting
    if not args.overwrite and any(
        (output_dir / f"{base_name}{ext}").exists() for ext in (".jpeg", ".jpg")
    ):
        print(
            f"Refusing to overwrite existing file named '{base_name}'. Use --overwrite to replace.",
            file=sys.stderr,
        )
        sys.exit(1)

    # If demo requested, save a sample visualization
    if args.demo:
        try:
            if args.engine == "matplotlib":
                fig = create_demo_matplotlib_figure()
                saved = save_matplotlib_figure_as_jpeg(fig, output_path, dpi=args.dpi)
            else:
                fig = create_demo_plotly_figure()
                saved = save_plotly_figure_as_jpeg(fig, output_path, scale=args.scale)

            print(f"Saved demo visualization to: {saved}")
        except Exception as exc:  # pragma: no cover
            print(f"Failed to save visualization: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Created/validated directory: {output_dir}")
        print("No demo saved. Use --demo to generate a sample JPEG.")


if __name__ == "__main__":
    main()


