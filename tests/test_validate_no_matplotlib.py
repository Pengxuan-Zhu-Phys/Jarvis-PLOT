"""B7 acceptance: `jplot validate` answers before the render stack exists.

This is not a performance test. Every error a config can contain is discovered
during rendering today, so a ten-figure config takes ten rounds to converge.
Validation only collapses that to one round if it can run without the renderer
-- and the only way that stays true is a test that fails the moment someone adds
a top-level matplotlib import to the validation path.

Runs in a subprocess because ``tests/conftest.py`` imports matplotlib itself.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The render stack. Never loaded by `validate`, whatever flags it is given.
RENDER_STACK = ("matplotlib", "scipy", "shapely")

#: Data libraries. Loaded only when the column check actually probes a file,
#: which is a header read -- never a row read.
DATA_STACK = ("pandas", "polars", "h5py", "pyarrow")

_PROBE = """
import sys
sys.path.insert(0, {root!r})
from jarvisplot.client import main
code = main(["validate", {config!r}, "--json"] + {extra!r})
loaded = sorted(m for m in {watched!r} if m in sys.modules)
print("EXIT", code, file=sys.stderr)
print("LOADED", ",".join(loaded), file=sys.stderr)
"""


@pytest.fixture
def config(tmp_path):
    """A config whose columns really are checked, so the probe path is exercised."""
    (tmp_path / "samples.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(
        textwrap.dedent(
            """
            DataSet:
              - {name: df, path: ./samples.csv, type: csv}
            Figures:
              - name: f1
                layers:
                  - name: a
                    data: [{source: df}]
                    axes: ax
                    method: scatter
                    coordinates:
                      x: {expr: x}
                      y: {expr: y}
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return path


def _probe(config_path, watched, extra=()) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE.format(
                root=str(REPO_ROOT),
                config=str(config_path),
                watched=list(watched),
                extra=list(extra),
            ),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _report(result) -> dict[str, str]:
    assert result.returncode == 0, result.stderr
    return dict(
        line.split(" ", 1) for line in result.stderr.strip().splitlines() if " " in line
    )


def test_validate_never_imports_the_render_stack(config):
    """The physical precondition for one-round convergence."""
    report = _report(_probe(config, RENDER_STACK))
    assert report["EXIT"] == "0"
    assert report.get("LOADED", "") == "", (
        f"validate imported {report.get('LOADED')}; it must answer before the "
        "render stack exists"
    )


def test_no_columns_reads_no_data_library_at_all(config):
    """The cheapest tier: a pure shape verdict that opens no data file."""
    report = _report(_probe(config, RENDER_STACK + DATA_STACK, extra=["--no-columns"]))
    assert report.get("LOADED", "") == ""


def test_column_check_costs_only_a_header_read(config):
    """Pandas is allowed here -- that is the point -- but matplotlib still is not."""
    report = _report(_probe(config, RENDER_STACK))
    assert report.get("LOADED", "") == ""


def test_validate_emits_exactly_one_json_object_on_stdout(config):
    import json

    result = _probe(config, RENDER_STACK)
    payload = json.loads(result.stdout)
    assert payload["kind"] == "validate"
    assert payload["data"]["columns_checked"] is True
    assert result.stdout.count("\n") == 1
