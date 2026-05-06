"""
Benchmark orchestration CLI.

This module dispatches benchmark, sweep, and analysis commands by
invoking the corresponding Python scripts as subprocesses. When no
arguments are provided, it transfers control to the interactive
product CLI for embedding and inspection workflows.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.text import Text

from stegano_app import product_cli

ROOT = Path(__file__).parent.parent



def _run(script: str, args: list[str]) -> int:
    cmd = [sys.executable, str(ROOT / script), *args]
    return subprocess.run(cmd).returncode

def cmd_benchmark(ns: argparse.Namespace) -> int:
    args = [
        "--dataset-dir", ns.dataset_dir,
        "--payload-size", str(ns.payload_size),
        "--colony-size", str(ns.colony_size),
        "--max-iter", str(ns.max_iter),
        "--min-block", str(ns.min_block),
        "--csv-out", ns.csv_out,
    ]
    if ns.subset_size is not None:
        args.extend(["--subset-size", str(ns.subset_size)])
    return _run("main.py", args)

def cmd_sweep(ns: argparse.Namespace) -> int:
    return _run("payload_sweep.py", ["--n", str(ns.n), "--out", ns.out])

def cmd_analyze(ns: argparse.Namespace) -> int:
    args = ["--csv", ns.csv, "--out", ns.out, "--top-n", str(ns.top_n)]
    if ns.sweep:
        args.extend(["--sweep", ns.sweep])
    if ns.demo:
        args.append("--demo")
    return _run("academic_analysis.py", args)


def cmd_visualize_payload(_: argparse.Namespace) -> int:
    return _run("visualize_payload.py", [])


def cmd_visualize_stego(_: argparse.Namespace) -> int:
    return _run("visualize_stego.py", [])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stegano",
        description="Adaptive Quadtree + D-ABC + LSB-Matching CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version="stegano 0.1.0")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("benchmark", help="Run main benchmark")
    b.add_argument("--dataset-dir", default="./data/BOSSbase-1.01/cover")
    b.add_argument("--subset-size", type=int, default=None)
    b.add_argument("--payload-size", type=int, default=10000)
    b.add_argument("--colony-size", type=int, default=30)
    b.add_argument("--max-iter", type=int, default=50)
    b.add_argument("--min-block", type=int, default=4)
    b.add_argument("--csv-out", default="deney_sonuclari.csv")
    b.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("sweep", help="Run payload sweep")
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--out", default="payload_sweep_results.csv")
    s.set_defaults(func=cmd_sweep)

    a = sub.add_parser("analyze", help="Run academic analysis")
    a.add_argument("--csv", default="deney_sonuclari.csv")
    a.add_argument("--sweep", default=None)
    a.add_argument("--out", default="results")
    a.add_argument("--top-n", type=int, default=5)
    a.add_argument("--demo", action="store_true")
    a.set_defaults(func=cmd_analyze)

    vp = sub.add_parser("visualize-payload", help="Render payload visualization")
    vp.set_defaults(func=cmd_visualize_payload)

    vs = sub.add_parser("visualize-stego", help="Render stego comparison")
    vs.set_defaults(func=cmd_visualize_stego)

    return p


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        console = Console(theme=product_cli.THEME)
        product_cli.run_interactive(console)
        raise SystemExit(0)
    ns = parser.parse_args()
    raise SystemExit(ns.func(ns))


if __name__ == "__main__":
    main()