"""Packaged forest examples stay runnable through the formal CLI in isolation."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


EXAMPLES = Path(__file__).resolve().parents[1] / "Example" / "forest"


@pytest.mark.parametrize("example,point_count", [("multi-interval.yaml", 20), ("intervals-only.yaml", 0)])
def test_multi_interval_examples_formal_cli_and_supplied_data(example, point_count, tmp_path):
    config = yaml.safe_load((EXAMPLES / example).read_text())
    for entry in config["DataSet"]:
        entry["path"] = str((EXAMPLES / entry["path"]).resolve())
    config["project"] = {"workdir": str(tmp_path)}
    config["output"]["dir"] = str(tmp_path / "plots")
    path = tmp_path / example
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    env = dict(os.environ, JARVIS_HOME=str(tmp_path / "registry"), MPLCONFIGDIR=str(tmp_path / "matplotlib"))
    env.pop("PYTHONPATH", None)

    def command(*args):
        # Invoke the installed package's exact public entry point from outside
        # the checkout, with no preview implementation on PYTHONPATH.
        result = subprocess.run(
            [sys.executable, "-c", "import sys; from jarvisplot.client import main; raise SystemExit(main(sys.argv[1:]))", *args],
            cwd=tmp_path, env=env, text=True, capture_output=True, check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    assert json.loads(command("doctor", str(path), "--json").stdout)["ok"] is True
    command(str(path), "--report")

    name = config["Figures"][0]["name"]
    output = tmp_path / "plots"
    assert (output / f"{name}.png").read_bytes().startswith(b"\x89PNG")
    assert (output / f"{name}.pdf").read_bytes().startswith(b"%PDF")
    layout = json.loads((output / f"{name}.layout.json").read_text())
    assert layout["canvas_axes"] == 2
    assert layout["point_count"] == point_count
    assert layout["interval_count"] == 6
    assert layout["shared_ticks"] == config["Figures"][0]["x"]["ticks"]
    assert layout["verification"]["axis_tick_count"] == 7
    assert layout["verification"]["annotation_collisions"] == "passed"
    assert layout["verification"]["numeric_x_preserved"] is True
    with (EXAMPLES / "data" / "summary-intervals.csv").open(newline="") as handle:
        expected = {(r["row"], r["id"]): (float(r["estimate"]), float(r["lower"]), float(r["upper"]), r["stats"])
                    for r in csv.DictReader(handle)}
    actual = {(r["row"], i["id"]): (i["estimate"], i["lower"], i["upper"], i["annotation"])
              for r in layout["rows"] for i in r["intervals"]}
    assert actual == expected
    if point_count:
        with (EXAMPLES / "data" / "summary-points.csv").open(newline="") as handle:
            points = {(r["row"], r["id"]): float(r["x"]) for r in csv.DictReader(handle)}
        assert {(r["row"], p["id"]): p["x"] for r in layout["rows"] for p in r["points"]} == points
        with (EXAMPLES / "data" / "summary-rows.csv").open(newline="") as handle:
            rows = {r["row"]: r for r in csv.DictReader(handle)}
        for row in layout["rows"]:
            source = rows[row["row"]]
            assert row["box_annotation"] == source["stats"]
            assert row["coordinates"] == {key: float(source[key]) for key in ("lower", "q1", "median", "q3", "upper")}
    command(str(path), "--print")
    assert json.loads((output / f"{name}.layout.json").read_text())["canvas_axes"] == 1
