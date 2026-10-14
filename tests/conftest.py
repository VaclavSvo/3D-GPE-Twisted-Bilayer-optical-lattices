"""Test session setup: the tests run on the CPU with two threads (reproducible and fast on small grids).

TWISTED_BILAYER_FORCE_CPU must be set before `engine.backend` is imported, since it picks the device at import.
GPU runs are production runs (`twisted-bilayer run gpu_check` compares them with the CPU), not tests.
"""

import os

os.environ.setdefault("TWISTED_BILAYER_FORCE_CPU", "1")

import torch  # noqa: E402

torch.set_num_threads(2)
