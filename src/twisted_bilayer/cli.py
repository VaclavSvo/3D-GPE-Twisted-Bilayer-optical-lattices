"""Command line: `twisted-bilayer` (or `python -m twisted_bilayer`).

    twisted-bilayer list                            the scenarios and their kinds
    twisted-bilayer run coupling_scan_quick         run one; files in results/coupling_scan_quick/
    twisted-bilayer run trapped --out runs          ... in runs/trapped/
    twisted-bilayer run my_row --scenarios rows.py  a row of your own scenario file
    twisted-bilayer run bdg --cpu                   stay on the CPU when a GPU is present
    twisted-bilayer run gpu_3d_vs_2d --128          complex128 throughout (publication runs)
    twisted-bilayer gates [--full] [--trapped]      the physics gates (see twisted_bilayer.gates)
    twisted-bilayer panel quick | broad             every check with one verdict (see twisted_bilayer.panel)

Exit codes of `run`: 0 done; 1 a failed physics check (and any error); 2 a usage error (an unknown scenario);
3 a trapped run that finished but whose cloud is not contained in its box (edge density >= 1e-3 of the peak;
its files are written, `checks.contained` is false in summary.json).

The engine picks its device when it is imported, so this module imports it only after the command line
has decided whether the run must stay on the CPU.
"""

from __future__ import annotations

import argparse
import os
import sys

FORCE_CPU = "TWISTED_BILAYER_FORCE_CPU"
CPU_ONLY_KINDS = ("bdg", "tight_binding")  # validated on the CPU only


def _cmd_list(a) -> int:
    from twisted_bilayer.scenarios import load_scenarios

    rows = load_scenarios(a.scenarios)
    width = max(len(n) for n in rows)
    for name, row in rows.items():
        print(f"{name:<{width}s}  {row['kind']}")
    return 0


def _cmd_run(a) -> int:
    from twisted_bilayer.scenarios import load_scenarios

    rows = load_scenarios(a.scenarios)
    if a.name not in rows:
        print(f"unknown scenario {a.name!r}; `twisted-bilayer list` shows the choices", file=sys.stderr)
        return 2
    row = rows[a.name]
    if a.cpu or row["kind"] in CPU_ONLY_KINDS:
        os.environ[FORCE_CPU] = "1"
    from twisted_bilayer.runners import run_scenario

    run_scenario(a.name, row, out_root=a.out, double=a.double)
    return 0


def _cmd_panel(a) -> int:
    if a.cpu:
        os.environ[FORCE_CPU] = "1"
    from twisted_bilayer import panel

    if a.list:
        for s in panel.PANELS[a.name]:
            flags = ", ".join(f for f, on in (("checkout", s.needs_checkout), ("GPU", s.needs_gpu)) if on)
            print(f"{s.name:<14s} {s.what}" + (f"  [needs {flags}]" if flags else ""))
        return 0
    only = [x.strip() for x in a.only.split(",") if x.strip()] if a.only else None
    skip = [x.strip() for x in a.skip.split(",") if x.strip()] if a.skip else None
    try:
        panel.select_steps(a.name, only, skip)
    except ValueError as e:  # a usage error: nothing runs and nothing is written
        print(f"twisted-bilayer panel: {e}", file=sys.stderr)
        return 2
    return panel.run_panel(a.name, out_root=a.out, only=only, skip=skip)


def _cmd_gates(rest: list[str]) -> int:
    wants_cuda = any(arg.startswith("--device") and "cuda" in arg for arg in rest) or (
        "--device" in rest and rest.index("--device") + 1 < len(rest) and rest[rest.index("--device") + 1] == "cuda"
    )
    if not wants_cuda:
        os.environ[FORCE_CPU] = "1"
    from twisted_bilayer.gates.suite import main as gates_main

    return gates_main(rest)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "gates":
        return _cmd_gates(argv[1:])
    ap = argparse.ArgumentParser(
        prog="twisted-bilayer",
        description="Two-component Gross-Pitaevskii simulations of a BEC in twisted-bilayer optical lattices.",
        epilog="`twisted-bilayer gates --help` lists the options of the physics gates.",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list", help="list the scenarios")
    p.add_argument("--scenarios", metavar="FILE", help="a Python file with your own SCENARIOS dict")
    p.set_defaults(fn=_cmd_list)
    p = sub.add_parser("run", help="run one scenario")
    p.add_argument("name", help="scenario name (see `list`)")
    p.add_argument("--out", default="results", help="output folder; files go to OUT/NAME/ (default: results)")
    p.add_argument("--scenarios", metavar="FILE", help="a Python file with your own SCENARIOS dict")
    p.add_argument("--cpu", action="store_true", help="stay on the CPU even when a GPU is present")
    p.add_argument(
        "--128",
        dest="double",
        action="store_true",
        help="run in complex128 throughout (publication figures); without it a row runs in the precision it names",
    )
    p.set_defaults(fn=_cmd_run)
    p = sub.add_parser(
        "panel", help="run a check panel: quick (CPU, before a commit) or broad (GPU, after a big change)"
    )
    p.add_argument("name", choices=["quick", "broad"])
    p.add_argument(
        "--out",
        default=None,
        help="output folder; each run goes to OUT/panel_NAME/<date_time>/ (default: results/ of the source "
        "checkout, or ./results)",
    )
    p.add_argument("--only", default="", help="comma-separated step names (see --list)")
    p.add_argument("--skip", default="", help="comma-separated step names to leave out")
    p.add_argument("--list", action="store_true", help="list the steps and exit")
    p.add_argument("--cpu", action="store_true", help="stay on the CPU even when a GPU is present")
    p.set_defaults(fn=_cmd_panel)
    sub.add_parser("gates", help="run the physics gates")
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
