"""Write docs/validation.md from a physics-gate run.

twisted-bilayer gates --full --trapped --out outputs/gates      # ~10 min on 2 CPU cores
python scripts/validation_report.py outputs/gates/gates.json     # -> docs/validation.md
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

GROUPS = {
    "A": (
        "Exact single-particle results",
        "Harmonic-oscillator ground state and breathing, free Gaussian expansion, band bottoms against "
        "plane-wave diagonalisation, Bloch weights, and two-level Rabi oscillation with and without detuning.",
    ),
    "B": (
        "Exact identities on the twisted supercell",
        "Gauge symmetry H(-J) = sz H(J) sz, conservation of N2/N without coupling, C4 isotropy and the "
        "energy = current routes of the superfluid fraction, and time reversibility dt -> -dt.",
    ),
    "C": (
        "Numerical anchors",
        "Dressed band bottom, chemical potential and in-situ contrast at the experiment's coupling, each "
        "reproduced by the author's earlier, separately written solvers (not distributed).",
    ),
    "D": (
        "Pre-flight checks",
        "Kinetic phase per time step of the automatic time step (orders 2 and 4), grid resolution, unit "
        "identities, and the validity range of the quasi-2D (NPSE) reduction.",
    ),
    "E": (
        "Trapped cloud",
        "A cloud of N = 2000 atoms in a 20 Hz trap: contrasts and size against the author's earlier "
        "quasi-2D solver (not distributed), "
        "convergence, conservation, and the cloud staying inside the box.",
    ),
}


def _md(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def main(path: str, out: str = str(ROOT / "docs" / "validation.md")) -> int:
    d = json.loads(pathlib.Path(path).read_text())
    res = d["results"]
    meta = d["meta"]
    by_group = collections.defaultdict(list)
    for r in res:
        by_group[r["name"][0]].append(r)
    kinds = collections.Counter(r["kind"] for r in res)
    n_pass = sum(r["passed"] for r in res)
    lines = [
        "# Validation",
        "",
        f"**{n_pass} of {len(res)} physics gates pass** on {meta.get('device', '?')} "
        f"({'full' if not meta.get('quick', True) else 'quick'} settings, {d.get('elapsed_s', 0):.0f} s).",
        "",
        "A gate is a pass/fail check of the engine against something known independently: an analytic "
        "result, an exact identity of the Hamiltonian, or a number reproduced by a separately written code. "
        "Thresholds and their sources are in `src/twisted_bilayer/gates/references.py`; a threshold is "
        "never loosened to make a gate pass. Regenerate this page with",
        "",
        "```bash",
        "twisted-bilayer gates --full --trapped --out outputs/gates",
        "python scripts/validation_report.py outputs/gates/gates.json",
        "```",
        "",
        "| kind | gates |",
        "| --- | --- |",
        *[f"| {k} | {v} |" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])],
        "",
    ]
    for g in sorted(by_group):
        title, what = GROUPS.get(g, (g, ""))
        rows = by_group[g]
        lines += [
            f"## {g}. {title}",
            "",
            what,
            "",
            "| gate | result | |",
            "| --- | --- | --- |",
        ]
        for r in rows:
            lines.append(f"| {_md(r['name'])} | {_md(r['detail'])} | {'pass' if r['passed'] else '**FAIL**'} |")
        lines.append("")
    pathlib.Path(out).write_text("\n".join(lines))
    print(f"wrote {out}: {n_pass}/{len(res)} gates pass")
    return 0 if n_pass == len(res) else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
