#!/usr/bin/env python3
"""
Export key dashboard visualizations as JPEG images into a specified directory.

This script loads the same data used by the dashboard and renders the core
figures using the visualization utilities, saving them as static JPEG files.

Requirements:
- plotly (already in requirements)
- kaleido (for static image export): pip install kaleido

Usage examples:
  python3 scripts/export_dashboard_figures.py
  python3 scripts/export_dashboard_figures.py --out visualizations/exports --scenario heat_wave --overwrite
"""

import argparse
from pathlib import Path
from datetime import datetime
import sys

# Ensure project root is on sys.path so we can import `src.*`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def ensure_out_dir(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def require_kaleido() -> None:
    try:
        import plotly.io as pio  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Plotly is required. Ensure it's installed.") from exc
    # Attempt to access image export to hint if kaleido missing
    try:
        import kaleido  # type: ignore  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Static image export requires 'kaleido'. Install with: pip install kaleido") from exc


def export_figures(output_dir: Path, scenario: str, overwrite: bool) -> None:
    from src.dashboard.data_manager import DataManager
    from src.dashboard.components import EnergyVisualizations
    import plotly.io as pio

    data_manager = DataManager("data")
    integration_results = data_manager.load_lstm_integration_results()
    if not integration_results:
        print("No integration results found. Aborting export.", file=sys.stderr)
        sys.exit(1)

    if scenario not in integration_results:
        # default to first available
        scenario = list(integration_results.keys())[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save(fig, name: str):
        path = output_dir / f"{name}.jpeg"
        if path.exists() and not overwrite:
            print(f"Skip existing: {path}")
            return
        pio.write_image(fig, str(path), format="jpeg", scale=2.0)
        print(f"Saved: {path}")

    # 1) LSTM Forecast (by cohort)
    fig_forecast = EnergyVisualizations.create_lstm_forecast_chart(integration_results, scenario)
    save(fig_forecast, f"lstm_forecast_{scenario}_{timestamp}")

    # 2) Building Cohort Heatmap
    fig_heatmap = EnergyVisualizations.create_building_cohort_heatmap(integration_results, scenario)
    save(fig_heatmap, f"cohort_heatmap_{scenario}_{timestamp}")

    # 3) Weather Scenario Comparison (averages across scenarios)
    fig_compare = EnergyVisualizations.create_weather_scenario_comparison(integration_results)
    save(fig_compare, f"scenario_comparison_{timestamp}")

    # 4) Grid Strain Summary
    fig_strain = EnergyVisualizations.create_strain_prediction_summary(integration_results)
    save(fig_strain, f"strain_summary_{timestamp}")

    # 5) Performance Validation (benchmark overlay)
    perf_metrics = DataManager("data").get_lstm_performance_metrics(integration_results)
    fig_perf = EnergyVisualizations.create_performance_validation_chart(perf_metrics)
    save(fig_perf, f"performance_validation_{timestamp}")

    # 6) Weather Scenario Summary (composite)
    fig_summary = EnergyVisualizations.create_weather_scenario_summary(integration_results)
    save(fig_summary, f"weather_summary_{timestamp}")

    # 7) Energy Consumption Time Series (aggregated across cohorts for selected scenario)
    ts_df = data_manager.get_lstm_time_series_data(integration_results, scenario)
    from src.dashboard.components import EnergyVisualizations as EV
    fig_ts = EV.create_energy_consumption_chart(ts_df, title=f"Energy Consumption - {scenario}")
    save(fig_ts, f"energy_consumption_{scenario}_{timestamp}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dashboard figures as JPEGs.")
    parser.add_argument(
        "--out", dest="out", default="visualizations/exports", help="Output directory for JPEGs"
    )
    parser.add_argument(
        "--scenario", dest="scenario", default="heat_wave", help="Default scenario to render where applicable"
    )
    parser.add_argument(
        "--overwrite", dest="overwrite", action="store_true", help="Overwrite existing files"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(Path(args.out))
    require_kaleido()
    export_figures(out_dir, scenario=args.scenario, overwrite=args.overwrite)


if __name__ == "__main__":
    main()


