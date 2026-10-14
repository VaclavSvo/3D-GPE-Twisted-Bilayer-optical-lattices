"""Physics gates: pass/fail checks of the engine against something known exactly or pinned
independently -- an analytic single-particle result, an exact identity of the bilayer Hamiltonian,
or a numerical anchor reproduced by more than one independent code.

core.py holds the gate primitives and the gates on a recorded history (conservation, imaginary-time
convergence, pre-flight resolution and time step); references.py the analytic references and the
pinned anchors (the one place for numbers); suite.py runs the scenarios on the engine:

    twisted-bilayer gates [--full] [--trapped] [--out DIR] [--device cuda]
    python -m twisted_bilayer.gates [--full] [--trapped] [--out DIR] [--device cuda]
"""
