from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

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
