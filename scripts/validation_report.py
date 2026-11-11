"""Write docs/validation.md from a physics-gate run, each gate labelled with what it is checked against.

twisted-bilayer gates --full --trapped --out outputs/gates      # ~10 min on 2 CPU cores
python scripts/validation_report.py outputs/gates/gates.json     # -> docs/validation.md
(on a GPU the suite writes gates_cuda.json)
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
        "The validity range of the quasi-2D (NPSE) reduction at the experiment's density. The automatic time "
        "step's kinetic phase per step (orders 2 and 4), the grid resolution and the unit identity follow from "
        "its constants; they are unit tests (tests/test_gates.py), not gates.",
    ),
    "E": (
        "Trapped cloud",
        "A cloud of N = 2000 atoms in a 20 Hz trap: contrasts and size against the author's earlier "
        "quasi-2D solver (not distributed), "
        "convergence, conservation, and the cloud staying inside the box.",
    ),
}


# What a gate is checked against (the column "reference" of the page):
#   analytic        a closed form or a plane-wave / two-level diagonalisation computed on the spot
#   identity        an exact identity or conservation law of the Hamiltonian and the integrator
#   regression pin  a number pinned in gates/references.py, produced by this code and reproduced by the
#                   author's earlier, separately written solvers (not distributed); it catches changes, it
#                   does not prove the number
#   convergence     the solve under the gate reached its residual target
#   range           a validity or containment condition of the model
REFERENCE_KINDS = ("analytic", "identity", "regression pin", "convergence", "range")


def reference_of(name: str) -> str:
    """The reference kind of a gate, from its name (group letter first)."""
    low = name.lower()
    if "final residual" in low or "converged" in low or "polished" in low:
        return "convergence"
    if low.endswith((": norm", ": energy")) or "high-k weight" in low:
        return "identity"  # conservation by the integrator
    if "kinetic phase" in low:
        return "range"
    group = name[:1]
    if group == "A":
        return "analytic"
    if group == "C" or "anchor" in low:
        return "regression pin"
    if group == "D" or "inside the box" in low:
        return "range"
    return "identity"


def _md(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def render(res: list[dict], header: str) -> str:
    """The page: the header line, the explanation, the counts by kind and by reference, one table per group."""
    by_group = collections.defaultdict(list)
    for r in res:
        by_group[r["name"][0]].append(r)
    kinds = collections.Counter(r["kind"] for r in res)
    refs = collections.Counter(reference_of(r["name"]) for r in res)
    lines = [
        "# Validation",
        "",
        header,
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
        "The column *reference* says what a gate is checked against. **analytic**: a closed form or an exact "
        "diagonalisation computed on the spot. **identity**: an exact identity or conservation law. "
        "**regression pin**: a number produced by this code and reproduced by the author's earlier, separately "
        "written solvers (not distributed); it detects a change, it is not an independent proof. "
        "**convergence**: the solve under the gate reached its residual target. **range**: a validity or "
        "containment condition.",
        "",
        "| reference | gates |",
        "| --- | --- |",
        *[f"| {k} | {refs[k]} |" for k in REFERENCE_KINDS if refs[k]],
        "",
        "| kind (as the suite labels it) | gates |",
        "| --- | --- |",
        *[f"| {k} | {v} |" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])],
        "",
    ]
    for g in sorted(by_group):
        title, what = GROUPS.get(g, (g, ""))
        lines += [
            f"## {g}. {title}",
            "",
            what,
            "",
            "| gate | result | reference | |",
            "| --- | --- | --- | --- |",
        ]
        for r in by_group[g]:
            ok = "pass" if r["passed"] else "**FAIL**"
            lines.append(f"| {_md(r['name'])} | {_md(r['detail'])} | {reference_of(r['name'])} | {ok} |")
        lines.append("")
    return "\n".join(lines)


def main(path: str, out: str = str(ROOT / "docs" / "validation.md")) -> int:
    d = json.loads(pathlib.Path(path).read_text())
    res, meta = d["results"], d["meta"]
    n_pass = sum(r["passed"] for r in res)
    header = (
        f"**{n_pass} of {len(res)} physics gates pass** on {meta.get('device', '?')} "
        f"({'full' if not meta.get('quick', True) else 'quick'} settings, {d.get('elapsed_s', 0):.0f} s)."
        + (f" Machine: {meta['platform']}, Python {meta.get('python', '?')}." if meta.get("platform") else "")
    )
    pathlib.Path(out).write_text(render(res, header) + "\n")
    print(f"wrote {out}: {n_pass}/{len(res)} gates pass")
    return 0 if n_pass == len(res) else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
