"""Run every check of the repository; one summary at the end.

    python check.py                 lint + the test suite (~5 min on 2 CPU cores)
    python check.py --all           + the slow test: the quick physics-gate suite (~1 min more)
    python check.py --only tests    one part (see --list)

Run from the repository root after `pip install -e ".[dev]"`. Everything runs on the CPU (GPU runs are
production runs, compared with the CPU by `twisted-bilayer run gpu_check`). Exit status 0 only if every
selected part passed. The checks are pinned to twisted_bilayer/config.py as shipped.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
PY = sys.executable
CPU_ENV = {"PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "TWISTED_BILAYER_FORCE_CPU": "1", "MPLBACKEND": "Agg"}

PARTS = {
    # name: (command, runs by default, what)
    "lint": ([PY, "-m", "ruff", "check", "."], True, "ruff: lint"),
    "format": ([PY, "-m", "ruff", "format", "--check", "."], True, "ruff: formatting"),
    "tests": ([PY, "-m", "pytest", "-q"], True, "pytest: engine, imaging, time of flight, gates (~5 min)"),
    "tests_slow": (
        [PY, "-m", "pytest", "-q", "-m", "slow"],
        False,
        "the quick physics-gate suite as one test (~1 min)",
    ),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="include the slow parts")
    ap.add_argument("--only", default="", help="comma-separated part names")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for k, (_, default, what) in PARTS.items():
            print(f"{k:12s} {'default' if default else 'with --all':11s} {what}")
        return 0
    if a.only:
        names = [n.strip() for n in a.only.split(",") if n.strip()]
        if unknown := [n for n in names if n not in PARTS]:
            ap.error(f"unknown part(s) {unknown}; see --list")
    else:
        names = [k for k, v in PARTS.items() if v[1] or a.all]
    env = dict(os.environ, **CPU_ENV)
    results = []
    for n in names:
        cmd, _, what = PARTS[n]
        print(f"\n== {n}: {what}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=ROOT, env=env)
        results.append((n, "ok" if rc == 0 else f"FAILED (exit {rc})", time.time() - t0))
    print("\n== summary")
    for n, s, dt in results:
        print(f"  {n:12s} {s:18s} {dt:7.0f} s")
    failed = [n for n, s, _ in results if s.startswith("FAILED")]
    print("all selected checks passed" if not failed else f"FAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
