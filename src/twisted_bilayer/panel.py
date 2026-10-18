"""Check panels: everything that must pass, in one command with one verdict.

    twisted-bilayer panel quick     lint, format, the test suite, the quick physics gates, and three small
                                    physics checks (3D against quasi-2D, vertical and in-plane resolution)
                                    with pinned references. Runs on a CPU in about 10-15 minutes; faster on
                                    a GPU. Run it before every commit.
    twisted-bilayer panel broad     the quick panel plus the slow tests, the full physics gates (on the GPU
                                    when there is one), GPU against CPU, and the experiment's cloud: 3D
                                    against quasi-2D in complex64 and complex128 against pinned numbers, the
                                    vertical and in-plane resolution, and the 3D size planner. Needs a GPU;
                                    about 30-60 minutes on a 12 GB consumer card. Run it after every big change.
    twisted-bilayer panel broad --only gates_full,gpu_cpu      a subset; --list shows the steps

Each step returns pass or fail with the numbers behind it, and every run keeps all of them on disk
(`records` describes the layout): results/panel_<name>/<date_time>/ holds summary.json (verdict,
provenance, every check), report.txt, log.txt and one folder per step with step.json (its checks), output.txt
(what it printed) and its data: the pytest steps a junit.xml and a tests.csv with every test; the gate steps
gates.json; the physics steps data.json (all observables, convergence rows, comparisons), report.txt (the
tables) and, for the 3D steps, the column densities (.npz) and a comparison figure. latest.txt names the
newest run and history.csv has one line per run. summary.json is rewritten after every step, so a run that
is interrupted keeps what it finished.

The panel exits 0 only if every step ran and passed (2 if a step was skipped, e.g. no GPU). Reference numbers
and their tolerances live in `gates.panel_references`; the criteria of the measurement steps are stated in
each step's docstring. Lint, format and the tests need a source checkout (an editable install); from a plain
install those steps are skipped.
"""

from __future__ import annotations

import csv
import os
import pathlib
import subprocess
import sys
import time
from dataclasses import dataclass, field

from twisted_bilayer import records as REC

ROOT = pathlib.Path(__file__).resolve().parents[2]  # the repository root of a source checkout
CPU_ENV = {"PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "TWISTED_BILAYER_FORCE_CPU": "1", "MPLBACKEND": "Agg"}


@dataclass
class Outcome:
    ok: bool | None  # None: skipped
    lines: list = field(default_factory=list)
    numbers: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)  # [dict(label, ok, detail)]


@dataclass
class Step:
    name: str
    what: str
    fn: object
    needs_checkout: bool = False
    needs_gpu: bool = False


def _has_checkout() -> bool:
    return (ROOT / "tests").is_dir() and (ROOT / "pyproject.toml").is_file()


def _has_gpu() -> bool:
    from twisted_bilayer.engine.backend import HAS_GPU

    return HAS_GPU


def default_out_root() -> str:
    """results/ of the source checkout (git-ignored), wherever the panel is started from; ./results otherwise."""
    return str(ROOT / "results") if _has_checkout() else "results"


def _run(cmd, env_extra=None) -> tuple[int, list]:
    """Run a command from the repository root, echo its output (into the step's output.txt), return
    (exit code, last lines)."""
    env = dict(os.environ, **(env_extra or {}))
    p = subprocess.Popen(
        cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8"
    )
    tail = []
    for line in p.stdout:
        print("    " + line.rstrip(), flush=True)
        tail.append(line.rstrip())
        tail = tail[-15:]
    return p.wait(), tail


def _check(ok_list, label, ok, detail) -> None:
    ok_list.append((label, bool(ok), detail))


def _outcome(checks, numbers=None) -> Outcome:
    lines = [f"{'ok ' if ok else 'FAIL'} {label}: {detail}" for label, ok, detail in checks]
    return Outcome(
        ok=all(ok for _, ok, _ in checks),
        lines=lines,
        numbers=numbers or {},
        checks=[dict(label=label, ok=ok, detail=detail) for label, ok, detail in checks],
    )


def _path(ctx, name) -> str:
    return os.path.join(ctx["dir"], name)


def _write_csv(path, rows) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------- tool steps
def step_lint(ctx) -> Outcome:
    code, tail = _run([sys.executable, "-m", "ruff", "check", "."])
    return Outcome(ok=code == 0, lines=tail[-3:])


def step_format(ctx) -> Outcome:
    code, tail = _run([sys.executable, "-m", "ruff", "format", "--check", "."])
    return Outcome(ok=code == 0, lines=tail[-3:])


def _pytest(ctx, extra) -> Outcome:
    """pytest with a junit.xml and a tests.csv (every test: outcome, seconds, message) in the step folder."""
    xml = _path(ctx, "junit.xml")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *extra,
        f"--junitxml={xml}",
        "--durations=30",
        "-o",
        "junit_family=xunit2",
    ]
    code, tail = _run(cmd, CPU_ENV)
    counts = REC.junit_to_csv(xml, _path(ctx, "tests.csv"))
    return Outcome(
        ok=code == 0, lines=[t for t in tail if " passed" in t or " failed" in t][-1:] or tail[-2:], numbers=counts
    )


def step_tests(ctx) -> Outcome:
    return _pytest(ctx, [])


def step_tests_slow(ctx) -> Outcome:
    return _pytest(ctx, ["-m", "slow"])


def step_gates_quick(ctx) -> Outcome:
    code, tail = _run([sys.executable, "-m", "twisted_bilayer.gates", "--out", ctx["dir"], "--quiet"], CPU_ENV)
    return Outcome(ok=code == 0, lines=tail[-4:])


def step_gates_full(ctx) -> Outcome:
    cmd = [sys.executable, "-m", "twisted_bilayer.gates", "--full", "--trapped", "--out", ctx["dir"], "--quiet"]
    env = None
    if _has_gpu():
        cmd += ["--device", "cuda"]
    else:
        env = CPU_ENV
    code, tail = _run(cmd, env)
    return Outcome(ok=code == 0, lines=tail[-4:])


# ----------------------------------------------------------------------------- physics steps
def _tiny_spec():
    from twisted_bilayer.engine import trapped as TR

    return TR.TrappedSpec(
        V0=4.0, N_atoms=60.0, nu_r_hz=200.0, cell_points=32, m=4, n=1, fillings=(0.5, 1.0, 2.0, 3.0), tol=1e-9
    )


def _experiment_spec():
    from twisted_bilayer.engine import trapped as TR
    from twisted_bilayer.engine import trapped3d as T3

    return T3.scaled(TR.TrappedSpec(tol=1e-9), 3)


def _regression(checks, obs, refs, tols, tag) -> None:
    for model, values in refs.items():
        for k, ref in values.items():
            tol = tols[model][k] if isinstance(tols.get(model), dict) else tols[k]
            d = obs[model][k] - ref
            _check(
                checks,
                f"{tag} {model} {k}",
                abs(d) <= tol,
                f"{obs[model][k]:.10g} vs {ref:.10g} (|d| {abs(d):.1e} <= {tol:g})",
            )


def _save_3d(ctx, tag, res, obs, cmp, title) -> None:
    """The three models' column densities, widths and a comparison figure."""
    from twisted_bilayer.engine import trapped3d as T3

    T3.save_fields(res, _path(ctx, f"columns_{tag}.npz"))
    try:
        T3.plot_comparison(res, _path(ctx, f"comparison_{tag}.png"), title=title)
    except Exception as e:  # a figure must not fail a check
        print(f"  (figure not written: {type(e).__name__}: {e})")


def step_quick_3d(ctx) -> Outcome:
    """The small trapped cloud in 3D (N_z = 16) and both quasi-2D models, complex128: residuals below 1e-8,
    E_3D <= E_fixed (the fixed-g state is a 3D trial state), the NPSE closer to 3D than fixed g in mu, E/N
    and C_raw, and every model against `panel_references.QUICK_3D`."""
    from twisted_bilayer.engine import trapped3d as T3
    from twisted_bilayer.gates import panel_references as R

    spec, vert = _tiny_spec(), T3.Vertical(Nz=16)
    res = T3.solve_models(spec, vert, "complex128")
    obs = T3.model_observables(res)
    cmp = T3.compare_models(res, obs)
    REC.write_json(
        _path(ctx, "data.json"),
        dict(spec=spec, vertical=vert, dtype="complex128", observables=obs, comparison=cmp, references=R.QUICK_3D),
    )
    REC.write_text(_path(ctx, "report.txt"), T3.report_models(obs, cmp))
    _save_3d(ctx, "complex128", res, obs, cmp, "quick_3d: m = 4 cloud, complex128")
    checks = []
    for m in T3.MODELS:
        _check(checks, f"{m} residual", obs[m]["residual"] < 1e-8, f"{obs[m]['residual']:.1e}")
    _check(checks, "variational", obs["3d"]["E_per_N"] <= obs["fixed"]["E_per_N"] + 1e-9, "E_3D <= E_fixed")
    for k in ("mu", "E_per_N", "C_raw"):
        a, b = abs(cmp["3d_vs_npse"]["diff"][k]), abs(cmp["3d_vs_fixed"]["diff"][k])
        _check(checks, f"NPSE closer ({k})", a < b, f"|3d - npse| {a:.2e} < |3d - fixed| {b:.2e}")
    _regression(checks, obs, R.QUICK_3D, R.QUICK_3D_TOL, "ref")
    return _outcome(checks, {m: {k: obs[m][k] for k in ("mu", "E_per_N", "C_raw")} for m in T3.MODELS})


def _save_nz(ctx, conv) -> None:
    """data.json (every N_z row: observables, widths, timings; the NPSE reference), report.txt and the
    column densities of each N_z."""
    import numpy as np

    from twisted_bilayer.engine import trapped3d as T3

    REC.write_json(_path(ctx, "data.json"), conv)
    REC.write_text(_path(ctx, "report.txt"), T3.report_nz(conv))
    np.savez_compressed(
        _path(ctx, "columns.npz"), **{f"n_Nz{r['Nz']}": r["numbers"]["n"].astype(np.float32) for r in conv["rows"]}
    )
    _write_csv(
        _path(ctx, "nz.csv"),
        [
            dict(Nz=r["Nz"], dz_lz=r["dz_lz"], width_max=r["width_max"], width_mean=r["width_mean"])
            | {k: r["observables"][k] for k in (*T3.NZ_KEYS, "residual", "steps", "seconds")}
            for r in conv["rows"]
        ]
        + [
            dict(Nz="npse", dz_lz="", width_max="", width_mean="")
            | {k: conv["npse"].get(k, "") for k in (*T3.NZ_KEYS, "residual", "steps", "seconds")}
        ],
    )


def _nz_checks(conv, limit, tag) -> list:
    from twisted_bilayer.engine import trapped3d as T3

    rows, checks = conv["rows"], []
    fin, sec = rows[-1]["observables"], rows[-2]["observables"]
    for k in T3.NZ_KEYS:
        gap = abs(fin[k] - conv["npse"][k])
        if gap < 1e-6:
            continue
        rel = abs(sec[k] - fin[k]) / gap
        _check(
            checks,
            f"{tag} N_z {rows[-2]['Nz']} vs {rows[-1]['Nz']}: {k}",
            rel <= limit,
            f"{rel:.1%} of the 3D - NPSE gap (<= {limit:.0%})",
        )
    return checks


def step_quick_nz(ctx) -> Outcome:
    """Vertical resolution on the small cloud, complex128: N_z = 16 within 2 % of the 3D - NPSE gap of
    N_z = 24 in every observable (the convergence is spectral; N_z = 8 is not converged)."""
    from twisted_bilayer.engine import trapped3d as T3

    conv = T3.nz_convergence(_tiny_spec(), (12, 16, 24), dtype="complex128", dtype_2d="complex128")
    _save_nz(ctx, conv)
    return _outcome(_nz_checks(conv, 0.02, "small cloud"))


def _save_xy(ctx, conv_sc, conv_tr) -> None:
    """data.json (every resolution row, the differences from the finest and the resolution each observable
    needs), report.txt, supercell.csv and trapped.csv (one line per resolution and coupling)."""
    from twisted_bilayer.engine import convergence as CV

    REC.write_json(_path(ctx, "data.json"), CV.to_record(conv_sc, conv_tr))
    REC.write_text(_path(ctx, "report.txt"), CV.report(conv_sc, conv_tr))
    sc_rows, tr_rows = CV.table_rows(conv_sc, conv_tr)
    _write_csv(_path(ctx, "supercell.csv"), sc_rows)
    _write_csv(_path(ctx, "trapped.csv"), tr_rows)


def _xy_checks(conv_sc, conv_tr, sc_limit, tr_limit, tr_keys) -> list:
    from twisted_bilayer.engine import convergence as CV

    checks = []
    if conv_sc:
        for V0, rows in conv_sc["rows"].items():
            nd = CV.needed(rows, CV.SUPERCELL_KEYS, "ppa")
            diffs = CV.differences(rows, CV.SUPERCELL_KEYS)
            for k, v in nd.items():
                _check(
                    checks,
                    f"supercell V0 = {V0:g}: {k}",
                    v is not None and v <= sc_limit,
                    f"needs {v} points per a (<= {sc_limit}); |value - finest| by resolution: "
                    + ", ".join(f"{r['ppa']:g}: {d[k]:.1e}" for r, d in zip(rows[:-1], diffs[:-1]))
                    + f" (tol {CV.TOLERANCES[k]:g})",
                )
    if conv_tr:
        rows = conv_tr["rows"]
        nd = CV.needed(rows, CV.TRAPPED_KEYS, "cell_points")
        diffs = CV.differences(rows, CV.TRAPPED_KEYS)
        for k in CV.TRAPPED_KEYS:
            if k in tr_keys:
                _check(
                    checks,
                    f"trapped: {k}",
                    nd[k] is not None and nd[k] <= tr_limit,
                    f"needs {nd[k]} per supercell (<= {tr_limit}); |value - finest|: "
                    + ", ".join(f"{r['cell_points']}: {d[k]:.1e}" for r, d in zip(rows[:-1], diffs[:-1]))
                    + f" (tol {CV.TOLERANCES[k]:g})",
                )
    return checks


def step_quick_xy(ctx) -> Outcome:
    """In-plane resolution on the small cell, complex128: the supercell at V0 = 4 and 8 E_R needs at most
    8 points per a in every observable (tolerances of `engine.convergence`), the trapped cloud at most 32
    points per supercell (7.8 per a) in mu, E/N, C_raw, C_coh, R_rms. n_AA (a point-sampled density) and
    n_peak are reported, not gated."""
    from twisted_bilayer.engine import convergence as CV

    sc = CV.supercell_convergence((4.0, 8.0), (0.0, 0.24, 1.0), (6, 8, 12), m=4, tol=1e-10)
    tr = CV.trapped_convergence(_tiny_spec(), (24, 32, 48))
    _save_xy(ctx, sc, tr)
    checks = _xy_checks(sc, tr, 8, 32, ("mu", "E_per_N", "C_raw", "C_coh", "R_rms"))
    out = _outcome(checks)
    out.lines.append(CV.report(sc, tr).splitlines()[-2])
    return out


def step_gpu_cpu(ctx) -> Outcome:
    """The quick coupling scan on the GPU and in a CPU subprocess, every number compared (the tolerances
    of `gpu_check`: 1e-6 on energies, 1e-4 on superfluid fractions). Both scans are saved (gpu/, cpu/)."""
    import numpy as np

    from twisted_bilayer.runners import GPU_KEYS, GPU_TOL

    def run(outdir, force_cpu):
        env = dict(os.environ)
        if force_cpu:
            env["TWISTED_BILAYER_FORCE_CPU"] = "1"
        code = (
            "from twisted_bilayer.engine.sweep import SweepSpec, coupling_scan; "
            "spec = SweepSpec(V0=8.0, Omega_R=(0.0, 0.24, 1.0), m=4, points_per_a=12, tol=1e-9, "
            "dtype_relax='complex128', dtype_polish='complex128'); "
            f"res = coupling_scan(spec, verbose=False); res.save({outdir!r})"
        )
        p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-2000:])
        return dict(np.load(os.path.join(outdir, "coupling_scan.npz")))

    gpu = run(_path(ctx, "gpu"), False)
    cpu = run(_path(ctx, "cpu"), True)
    checks, diffs = [], {}
    for k in GPU_KEYS:
        d = float(np.max(np.abs(gpu[k] - cpu[k])))
        diffs[k] = dict(max_abs_diff=d, tol=GPU_TOL.get(k, 1e-6), gpu=gpu[k], cpu=cpu[k])
        tol = GPU_TOL.get(k, 1e-6)
        _check(checks, k, d < tol, f"max |GPU - CPU| {d:.1e} (< {tol:g})")
    REC.write_json(_path(ctx, "data.json"), diffs)
    return _outcome(checks)


def step_experiment_3d(ctx) -> Outcome:
    """The experiment's cloud (3 x 3 supercells, N = 2000, 20 Hz) in 3D (N_z = 32) and both quasi-2D
    models, in complex128 and complex64: complex64 resolves the 3D - NPSE difference in every
    observable/model pair (error < 10 % of it), and the complex128 numbers reproduce
    `panel_references.EXPERIMENT_3D`."""
    from twisted_bilayer.engine import trapped as TR
    from twisted_bilayer.engine import trapped3d as T3
    from twisted_bilayer.gates import panel_references as R

    spec, vert = _experiment_spec(), T3.Vertical(Nz=32)
    scan = TR.supercell_scan(spec)
    r128 = T3.solve_models(spec, vert, "complex128", scan=scan)
    o128 = T3.model_observables(r128)
    c128 = T3.compare_models(r128, o128)
    _save_3d(ctx, "complex128", r128, o128, c128, "experiment_3d: the experiment's cloud, complex128")
    del r128
    r64 = T3.solve_models(spec, vert, "complex64", scan=scan)
    o64 = T3.model_observables(r64)
    c64 = T3.compare_models(r64, o64)
    _save_3d(ctx, "complex64", r64, o64, c64, "experiment_3d: the experiment's cloud, complex64")
    del r64
    table, numbers = T3.precision_table(o64, o128, c128)
    REC.write_json(
        _path(ctx, "data.json"),
        dict(
            spec=spec,
            vertical=vert,
            complex128=dict(observables=o128, comparison=c128),
            complex64=dict(observables=o64, comparison=c64),
            precision=numbers,
            references=R.EXPERIMENT_3D,
            tolerances=R.EXPERIMENT_3D_TOL,
        ),
    )
    REC.write_text(
        _path(ctx, "report.txt"),
        "complex128\n" + T3.report_models(o128, c128) + "\n\ncomplex64\n" + T3.report_models(o64, c64) + "\n\n" + table,
    )
    checks = []
    _check(
        checks,
        "complex64 resolves 3D - NPSE",
        numbers["resolved"] == numbers["pairs"],
        f"{numbers['resolved']} of {numbers['pairs']} pairs",
    )
    _regression(checks, o128, R.EXPERIMENT_3D, R.EXPERIMENT_3D_TOL, "ref complex128")
    for m in T3.MODELS:
        _check(checks, f"{m} residual (complex128)", o128[m]["residual"] < 1e-8, f"{o128[m]['residual']:.1e}")
    secs = {d: sum(o[m]["seconds"] for m in T3.MODELS) for d, o in (("complex128", o128), ("complex64", o64))}
    return _outcome(checks, dict(seconds=secs, mu={m: o128[m]["mu"] for m in T3.MODELS}))


def step_experiment_nz(ctx) -> Outcome:
    """Vertical resolution on the experiment's cloud (3D complex64, 2D complex128): N_z = 32 within 5 % of
    the 3D - NPSE gap of N_z = 48 in every observable (5 %: the complex64 floor)."""
    from twisted_bilayer.engine import trapped3d as T3

    conv = T3.nz_convergence(_experiment_spec(), (16, 32, 48), dtype="complex64", dtype_2d="complex128")
    _save_nz(ctx, conv)
    return _outcome(_nz_checks(conv, 0.05, "experiment"))


def step_experiment_xy(ctx) -> Outcome:
    """In-plane resolution on the experiment's geometry, complex128: the m = 22 supercell at V0 = 4 and 8 E_R
    needs at most 8 points per a in every observable, the trapped cloud at most 176 points per supercell
    (8 per a) in mu, E/N, C_raw, C_coh, R_rms."""
    from twisted_bilayer.engine import convergence as CV
    from twisted_bilayer.engine import trapped as TR

    sc = CV.supercell_convergence((4.0, 8.0), (0.0, 0.24, 1.0), (6, 8, 12, 16), m=22, tol=1e-9)
    tr = CV.trapped_convergence(TR.TrappedSpec(tol=1e-9), (132, 176, 264))
    _save_xy(ctx, sc, tr)
    return _outcome(_xy_checks(sc, tr, 8, 176, ("mu", "E_per_N", "C_raw", "C_coh", "R_rms")))


def step_planner(ctx) -> Outcome:
    """The 3D size planner on this GPU: the probe runs and the plan holds at least the 3 x 3 cloud at
    N_z >= 16 in complex64 (true on any GPU with >= 6 GB). data.json also has the largest box at
    N_z = 16, 32, 48, 64 and the predicted solve time of each."""
    from twisted_bilayer.engine import trapped as TR
    from twisted_bilayer.engine import trapped3d as T3

    spec0, vert = TR.TrappedSpec(), T3.Vertical()
    pr = T3.probe(spec0, vert, "complex64")
    budget = T3.memory_budget(0.85)
    mem = lambda b, z: pr["memory"](T3.points_3d(b, z, 176))  # noqa: E731
    plan = T3.plan(mem, budget, 176, 16, 128)
    by_nz = []
    for Nz in (16, 32, 48, 64):
        p = T3.plan(mem, budget, 176, Nz_fixed=Nz)
        if p is not None:
            box = p[0]
            by_nz.append(
                dict(
                    Nz=Nz,
                    box_cells=box,
                    points=T3.points_3d(box, Nz, 176),
                    gib=mem(box, Nz) / 2**30,
                    predicted_seconds=T3.predicted_seconds(pr["step_seconds"], box, Nz, 176),
                )
            )
    REC.write_json(_path(ctx, "data.json"), dict(probe=pr, budget_bytes=budget, plan=plan, largest_by_Nz=by_nz))
    _write_csv(_path(ctx, "largest_by_Nz.csv"), by_nz)
    checks = []
    _check(checks, "plan", plan is not None and plan[0] >= 3, f"box x N_z = {plan} within {budget / 2**30:.2f} GiB")
    return _outcome(checks, dict(plan=plan, bytes_per_point=pr["memory"].b))


QUICK = [
    Step("lint", "ruff check", step_lint, needs_checkout=True),
    Step("format", "ruff format --check", step_format, needs_checkout=True),
    Step("tests", "pytest (CPU)", step_tests, needs_checkout=True),
    Step("gates_quick", "the quick physics gates (CPU)", step_gates_quick),
    Step("quick_3d", "small cloud: 3D against quasi-2D, pinned numbers", step_quick_3d),
    Step("quick_nz", "small cloud: vertical resolution", step_quick_nz),
    Step("quick_xy", "small cell: in-plane resolution", step_quick_xy),
]
BROAD = QUICK + [
    Step("tests_slow", "pytest -m slow (CPU)", step_tests_slow, needs_checkout=True),
    Step("gates_full", "the full physics gates with the trapped cloud (GPU if present)", step_gates_full),
    Step("gpu_cpu", "GPU against CPU on the quick scan", step_gpu_cpu, needs_gpu=True),
    Step(
        "experiment_3d",
        "experiment's cloud: 3D against quasi-2D, complex64 and complex128",
        step_experiment_3d,
        needs_gpu=True,
    ),
    Step("experiment_nz", "experiment's cloud: vertical resolution", step_experiment_nz, needs_gpu=True),
    Step("experiment_xy", "experiment's geometry: in-plane resolution", step_experiment_xy, needs_gpu=True),
    Step("planner", "the 3D size planner on this GPU", step_planner, needs_gpu=True),
]
PANELS = {"quick": QUICK, "broad": BROAD}


def _provenance() -> dict:
    try:
        from twisted_bilayer.runners import _provenance as prov

        return prov()
    except Exception as e:  # provenance must not stop a check run
        return dict(error=f"{type(e).__name__}: {e}")


def _verdict(results) -> tuple[str, int, int]:
    n_fail = sum(r["ok"] is False for r in results)
    n_skip = sum(r["ok"] is None for r in results)
    return ("PASS" if n_fail == 0 and n_skip == 0 else ("FAIL" if n_fail else "INCOMPLETE")), n_fail, n_skip


def run_panel(name: str, out_root: str | None = None, only=None, skip=None) -> int:
    """Run the panel's steps; every run goes to its own folder <out_root>/panel_<name>/<date_time>/ (layout in
    `records` and the module docstring). Returns 0 if all steps ran and passed, 1 on a failure, 2 if a step
    was skipped."""
    from twisted_bilayer.engine.backend import describe_device

    out_root = out_root or default_out_root()
    kind = f"panel_{name}"
    steps = [s for s in PANELS[name] if (not only or s.name in only) and s.name not in (skip or ())]
    out, stamp = REC.run_dir(out_root, kind)
    device = describe_device()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    prov = _provenance()
    results = []
    t_all = time.time()

    def write_summary(state):
        verdict, _, _ = _verdict(results)
        REC.write_json(
            os.path.join(out, "summary.json"),
            dict(
                panel=name,
                state=state,
                verdict=verdict if state == "finished" else "RUNNING",
                started=started,
                device=device,
                provenance=prov,
                selected=[s.name for s in steps],
                steps=results,
            ),
        )

    with REC.Tee(os.path.join(out, "log.txt")):
        print(f"panel {name} on {device}: {len(steps)} steps; results in {out}", flush=True)
        write_summary("running")
        for s in steps:
            print(f"\n== {s.name}: {s.what}", flush=True)
            sdir = os.path.join(out, s.name)
            os.makedirs(sdir, exist_ok=True)
            t0 = time.time()
            with REC.Tee(os.path.join(sdir, "output.txt")):
                if s.needs_checkout and not _has_checkout():
                    o = Outcome(ok=None, lines=["skipped: needs a source checkout (pip install -e .)"])
                elif s.needs_gpu and not _has_gpu():
                    o = Outcome(ok=None, lines=["skipped: needs a GPU"])
                else:
                    try:
                        o = s.fn(dict(out=out, dir=sdir))
                    except Exception as e:  # a crashing step is a failing step
                        import traceback

                        traceback.print_exc()
                        o = Outcome(ok=False, lines=[f"error: {type(e).__name__}: {e}"])
                for line in o.lines:
                    print("  " + line, flush=True)
            r = dict(
                step=s.name,
                what=s.what,
                ok=o.ok,
                seconds=round(time.time() - t0, 1),
                lines=o.lines,
                checks=o.checks,
                numbers=o.numbers,
                files=sorted(os.listdir(sdir)),
            )
            REC.write_json(os.path.join(sdir, "step.json"), r)
            results.append(r)
            write_summary("running")
        verdict, n_fail, n_skip = _verdict(results)
        minutes = (time.time() - t_all) / 60
        summary = [
            f"panel {name}: {verdict} ({len(results) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} skipped) "
            f"in {minutes:.1f} min"
        ]
        for r in results:
            status = {True: "ok  ", False: "FAIL", None: "skip"}[r["ok"]]
            summary.append(f"  {status} {r['step']:<14s} {r['seconds']:>7.0f} s")
        text = "\n".join(summary)
        print("\n" + text)
        write_summary("finished")
        header = (
            f"{text}\n\nstarted {started} on {device}\ngit commit {prov.get('git_commit')} "
            f"(uncommitted changes: {prov.get('git_dirty')}), twisted_bilayer {prov.get('twisted_bilayer')}, "
            f"torch {prov.get('torch')}\nfolder {out}\n\n"
        )
        REC.write_text(
            os.path.join(out, "report.txt"),
            header + "\n".join(f"[{r['step']}]\n  " + "\n  ".join(r["lines"]) for r in results),
        )
        REC.mark_latest(out_root, kind, stamp)
        REC.append_history(
            out_root,
            kind,
            dict(
                run=stamp,
                verdict=verdict,
                passed=len(results) - n_fail - n_skip,
                failed=n_fail,
                skipped=n_skip,
                minutes=round(minutes, 1),
                failed_steps=" ".join(r["step"] for r in results if r["ok"] is False),
                device=device,
                git_commit=prov.get("git_commit"),
                git_dirty=prov.get("git_dirty"),
            ),
        )
        print(
            f"wrote {out} (summary.json, report.txt, log.txt, one folder per step); history in {os.path.dirname(out)}"
        )
    return 0 if verdict == "PASS" else (1 if n_fail else 2)
