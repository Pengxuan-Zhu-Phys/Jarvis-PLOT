"""B6 acceptance: case D -- a column that does not exist.

Observed today: `Figure f3 failed: name 'aa' is not defined`, raised from deep
inside the renderer with no YAML path and no list of what *does* exist.

The bar for this check is precision. A spurious "no such column" teaches an agent
to ignore the diagnostic, which is worse than not having it, so the parametrised
no-false-positive cases below are the load-bearing half of this file.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from jarvisplot.column_demand import plan_source_demand
from jarvisplot.column_probe import ColumnProbe, probe_dataset_columns
from jarvisplot.validation import validate_config


@pytest.fixture
def csv_data(tmp_path):
    (tmp_path / "samples.csv").write_text(
        "m_A,tanb,LogL\n100,10,-5\n200,20,-3\n", encoding="utf-8"
    )
    return tmp_path


def _validate(tmp_path, text):
    config = yaml.safe_load(textwrap.dedent(text).lstrip())
    return validate_config(config, base_dir=str(tmp_path))


def _column_errors(bag):
    return [d for d in bag if d.code == "JP-COL-001"]


# --------------------------------------------------------------------------- #
# The case
# --------------------------------------------------------------------------- #


def test_missing_column_is_reported_at_the_expression_that_named_it(csv_data):
    bag = _validate(
        csv_data,
        """
        DataSet:
          - {name: df, path: ./samples.csv, type: csv}
        Figures:
          - name: f1
            layers:
              - name: s
                data: [{source: df}]
                axes: ax
                method: scatter
                coordinates:
                  x: {expr: aa}
                  y: {expr: tanb}
        """,
    )
    errors = _column_errors(bag)
    assert len(errors) == 1
    assert errors[0].path == "$.Figures[0].layers[0].coordinates.x.expr", (
        "pointing at the DataSet entry is much less useful than pointing at the "
        "expression that named the column"
    )
    assert "aa" in errors[0].message
    assert errors[0].context["available_columns"] == ["LogL", "m_A", "tanb"]


def test_near_miss_gets_a_did_you_mean_and_a_heuristic_fix(csv_data):
    bag = _validate(
        csv_data,
        """
        DataSet:
          - {name: df, path: ./samples.csv, type: csv}
        Figures:
          - name: f1
            layers:
              - name: s
                data: [{source: df}]
                axes: ax
                method: scatter
                coordinates:
                  c: {expr: "exp(LogLL)"}
        """,
    )
    error = _column_errors(bag)[0]
    assert error.context["did_you_mean"] == ["LogL"]
    assert error.fix.to == "LogL"
    assert error.fix.confidence == "heuristic", "a column rename is never certain"


def test_columns_are_attributed_to_the_source_the_layer_names(tmp_path):
    """The pruning planner unions demand across datasets; this must not."""
    (tmp_path / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("y\n2\n", encoding="utf-8")
    bag = _validate(
        tmp_path,
        """
        DataSet:
          - {name: A, path: ./a.csv, type: csv}
          - {name: B, path: ./b.csv, type: csv}
        Figures:
          - name: f1
            layers:
              - {name: la, data: [{source: A}], axes: ax, method: plot, coordinates: {x: {expr: x}}}
              - {name: lb, data: [{source: B}], axes: ax, method: plot, coordinates: {x: {expr: y}}}
        """,
    )
    assert _column_errors(bag) == [], "A needs only x; B needs only y"


def test_no_columns_flag_skips_the_check(csv_data):
    config = yaml.safe_load(
        textwrap.dedent(
            """
            DataSet:
              - {name: df, path: ./samples.csv, type: csv}
            Figures:
              - name: f1
                layers:
                  - {name: s, data: [{source: df}], axes: ax, method: plot, coordinates: {x: {expr: aa}}}
            """
        ).lstrip()
    )
    bag = validate_config(config, base_dir=str(csv_data), check_columns=False)
    assert _column_errors(bag) == []


# --------------------------------------------------------------------------- #
# No false positives -- each of these was hit on the repo's own configs
# --------------------------------------------------------------------------- #


def test_transform_output_columns_are_not_reported_missing(csv_data):
    bag = _validate(
        csv_data,
        """
        DataSet:
          - {name: df, path: ./samples.csv, type: csv}
        Figures:
          - name: f1
            layers:
              - name: s
                data:
                  - source: df
                    transform:
                      - add_column: {name: ratio, expr: "m_A / tanb"}
                axes: ax
                method: plot
                coordinates:
                  x: {expr: ratio}
        """,
    )
    assert _column_errors(bag) == []


def test_share_data_consumers_are_not_checked_against_files(csv_data):
    bag = _validate(
        csv_data,
        """
        DataSet:
          - {name: df, path: ./samples.csv, type: csv}
        Figures:
          - name: f1
            layers:
              - name: producer
                data:
                  - source: df
                    transform:
                      - add_column: {name: derived, expr: "m_A * 2"}
                share_data: prepped
                axes: ax
                method: plot
                coordinates: {x: {expr: derived}}
              - name: consumer
                data: [{source: prepped}]
                axes: ax
                method: plot
                coordinates: {x: {expr: derived}}
        """,
    )
    assert _column_errors(bag) == []


def test_dotted_column_names_survive_expression_tokenisation(tmp_path):
    """`pVa.E` tokenises to `pVa` and `E`; neither is a missing column."""
    (tmp_path / "d.csv").write_text("pVa.E,pVb.E\n1,2\n", encoding="utf-8")
    bag = _validate(
        tmp_path,
        """
        DataSet:
          - {name: d, path: ./d.csv, type: csv}
        Figures:
          - name: f1
            layers:
              - {name: s, data: [{source: d}], axes: ax, method: plot, coordinates: {x: {expr: "pVa.E"}}}
        """,
    )
    assert _column_errors(bag) == []


def test_typed_figures_are_skipped_until_expansion(csv_data):
    """A `type:` figure's layers do not exist yet, so nothing can be attributed."""
    bag = _validate(
        csv_data,
        """
        DataSet:
          - {name: df, path: ./samples.csv, type: csv}
        Figures:
          - name: f1
            type: posterior_2d
        """,
    )
    assert _column_errors(bag) == []


def test_missing_data_file_suppresses_the_column_check(tmp_path):
    """One problem, one diagnostic: report the missing file, not phantom columns."""
    bag = _validate(
        tmp_path,
        """
        DataSet:
          - {name: df, path: ./nowhere.csv, type: csv}
        Figures:
          - name: f1
            layers:
              - name: s
                data: [{source: df}]
                axes: ax
                method: plot
                coordinates: {x: {expr: aa}, y: {expr: bb}}
        """,
    )
    assert {d.code for d in bag} == {"JP-DAT-004"}


# --------------------------------------------------------------------------- #
# Probe unit behaviour
# --------------------------------------------------------------------------- #


def test_probe_reads_csv_header_only(csv_data):
    probe = probe_dataset_columns({"type": "csv"}, str(csv_data / "samples.csv"))
    assert probe.supported and probe.error is None
    assert probe.names == {"m_A", "tanb", "LogL"}


def test_probe_reports_unknown_formats_instead_of_guessing():
    probe = probe_dataset_columns({"type": "sqlite"}, "/nonexistent")
    assert probe.supported is False
    assert probe.reason


def test_probe_resolves_accepts_fragments_of_dotted_names():
    probe = ColumnProbe(names={"pVa.E", "LogL"})
    assert probe.resolves("pVa.E")
    assert probe.resolves("pVa")
    assert probe.resolves("E")
    assert not probe.resolves("aa")


# --------------------------------------------------------------------------- #
# Demand provenance
# --------------------------------------------------------------------------- #


def test_demand_records_where_each_column_was_asked_for():
    config = yaml.safe_load(
        textwrap.dedent(
            """
            DataSet:
              - {name: df, path: ./x.csv, type: csv}
            Figures:
              - name: f1
                layers:
                  - {name: s, data: [{source: df}], method: plot, coordinates: {y: {expr: tanb}}}
            """
        ).lstrip()
    )
    demand = plan_source_demand(config)
    assert demand["df"].where("tanb") == ["$.Figures[0].layers[0].coordinates.y.expr"]


def test_disabled_figures_contribute_no_demand():
    config = yaml.safe_load(
        textwrap.dedent(
            """
            DataSet:
              - {name: df, path: ./x.csv, type: csv}
            Figures:
              - name: f1
                enable: false
                layers:
                  - {name: s, data: [{source: df}], method: plot, coordinates: {x: {expr: gone}}}
            """
        ).lstrip()
    )
    assert "gone" not in plan_source_demand(config)["df"].required
