"""Command line entry point for nnbench."""

import argparse
import json
import logging
from collections.abc import Sequence

from nnbench.config import load_config
from nnbench.logging import configure_logging

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
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
