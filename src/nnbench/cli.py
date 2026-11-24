"""Command line entry point for nnbench."""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from nnbench.analysis.plots import plot_pareto
from nnbench.analysis.report import export_csv, export_json, summary_table
from nnbench.config import load_config
from nnbench.experiments import run_experiment
from nnbench.logging import configure_logging
from nnbench.optimizers import list_optimizers
from nnbench.registry import list_models
from nnbench.store import ResultStore

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        prog="nnbench",
        description="Reproducible neural network optimization and benchmarking.",
    )
    subparsers = parser.add_subparsers(dest="command")
    show_parser = subparsers.add_parser("config-show", help="Load and print an experiment config")
    show_parser.add_argument("config", help="Path to a YAML config")
    run_parser = subparsers.add_parser("run", help="Run an experiment config")
    run_parser.add_argument("config", help="Config path or experiment=name")
    run_parser.add_argument(
        "--db", default="artifacts/nnbench.sqlite", help="SQLite result database"
    )
    run_parser.add_argument(
        "--synthetic", action="store_true", help="Use offline synthetic samples"
    )
    run_parser.add_argument("--download", action="store_true", help="Download missing datasets")
    report_parser = subparsers.add_parser("report", help="Export saved runs and trade-off plots")
    report_parser.add_argument("--db", default="artifacts/nnbench.sqlite")
    report_parser.add_argument("--out", default="reports")
    subparsers.add_parser("list-models", help="List registered models")
    subparsers.add_parser("list-optimizers", help="List registered optimizer plugins")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "config-show":
        try:
            print(json.dumps(load_config(args.config), indent=2, sort_keys=True))
        except ValueError as exc:
            logger.error("%s", exc)
            return 2
        return 0
    if args.command == "run":
        try:
            config_path = args.config
            if config_path.startswith("experiment="):
                name = config_path.split("=", 1)[1]
                config_path = str(Path("configs/experiment") / f"{name}.yaml")
            results = run_experiment(
                config_path, db_path=args.db, synthetic=args.synthetic, download=args.download
            )
        except (ValueError, OSError, ImportError, RuntimeError) as exc:
            logger.error("Experiment failed: %s", exc)
            return 2
        print(json.dumps(results, indent=2, default=str))
        return 0
    if args.command == "report":
        rows = ResultStore(args.db).query_results()
        out_dir = Path(args.out)
        summary = summary_table(rows)
        export_csv(rows, out_dir / "results.csv")
        export_json(rows, out_dir / "results.json")
        export_csv(summary, out_dir / "summary.csv")
        if rows:
            plot_pareto(rows).write_html(out_dir / "pareto_latency.html")
            plot_pareto(rows, x="model_size_mb").write_html(out_dir / "pareto_size.html")
        print(f"Wrote {len(rows)} run(s) to {out_dir}")
        return 0
    if args.command == "list-models":
        print("\n".join(list_models()))
        return 0
    if args.command == "list-optimizers":
        print("\n".join(list_optimizers()))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
