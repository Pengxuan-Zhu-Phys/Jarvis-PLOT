from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

from jarvisplot.utils import cmaps


COLORMAPS_JSON = (
    Path(__file__).resolve().parents[1]
    / "jarvisplot"
    / "cards"
    / "colors"
    / "colormaps.json"
)


def test_official_tab5_is_registered_as_five_discrete_tab10_colors():
    summary = cmaps.register_from_json(COLORMAPS_JSON, force=True)

    assert "tab5" in summary["registered"]
    assert "tab5_r" in summary["registered"]

    tab5 = mpl.colormaps["tab5"]
    assert tab5.N == 5
    assert [mpl.colors.to_hex(tab5(i), keep_alpha=False) for i in range(tab5.N)] == [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ]

    spec = next(
        item
        for item in json.loads(COLORMAPS_JSON.read_text(encoding="utf-8"))["colormaps"]
        if item["name"] == "tab5"
    )
    assert spec["type"] == "listed"
    assert len(spec["colors"]) == 5


def test_jpurples_matches_purples_style_with_a_pure_white_zero_endpoint():
    summary = cmaps.register_from_json(COLORMAPS_JSON, force=True)

    assert "Jpurples" in summary["registered"]
    assert "Jpurples_r" in summary["registered"]

    jpurples = mpl.colormaps["Jpurples"]
    purples = mpl.colormaps["Purples"]

    assert mpl.colors.to_hex(jpurples(0.0), keep_alpha=False) == "#ffffff"
    for value in (0.125, 0.25, 0.5, 0.75, 1.0):
        assert np.allclose(jpurples(value), purples(value), atol=1 / 255)
