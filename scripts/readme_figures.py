"""Regenerate the README figures from the examples (about 30 s on 2 CPU cores).

python scripts/readme_figures.py
"""

import pathlib
import runpy
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGURES = {
    "01_ground_state": "ground_state.png",
    "03_time_step_convergence": "time_step_convergence.png",
}

for example, target in FIGURES.items():
    runpy.run_path(str(ROOT / "examples" / f"{example}.py"), run_name="__main__")
    shutil.copyfile(ROOT / "examples" / "output" / f"{example}.png", ROOT / "docs" / "figures" / target)
    print(f"-> docs/figures/{target}")
