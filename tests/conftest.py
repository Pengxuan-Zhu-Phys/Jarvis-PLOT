from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvisplot.inner_func import clear_external_funcs


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _clear_external_funcs():
    clear_external_funcs()
    yield
    clear_external_funcs()
