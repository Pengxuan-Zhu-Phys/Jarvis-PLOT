"""B1/B2/B3 acceptance: the schema catalog, its self-check, and closed vocabulary.

The four cases at the top of `docs/roadmap/V2_YAML_AGENT_ERGONOMICS.md` §1 are
the reason this track exists; three of them are regression-tested here.
"""

from __future__ import annotations

import json
import textwrap

import pytest
from jsonschema import Draft202012Validator

from jarvisplot.Figure.method_registry import METHOD_DISPATCH
from jarvisplot.diagnostics import DiagnosticBag
from jarvisplot.schema_catalog import (
    ZONES,
    catalog_lint_errors,
    config_validator,
    ignored_properties,
    iter_schema_files,
    load_manifest,
    subschema,
)
from jarvisplot.validation import FIGURE_SCHEMA, LAYER_SCHEMA, validate_config


# --------------------------------------------------------------------------- #
# B1: the catalog loads, and only from disk
# --------------------------------------------------------------------------- #


def test_every_manifest_file_exists_and_is_a_valid_draft_2020_12_schema():
    files = list(iter_schema_files())
    assert files, "manifest lists no schema files"
    for relative_name, path in files:
        assert path.exists(), f"{relative_name} is in the manifest but missing on disk"
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        assert "$id" in schema, f"{relative_name} has no $id"


def test_manifest_is_data_only():
    """No logic, and no dispatch entry pointing at a file we do not bundle."""
    manifest = load_manifest()
    bundled_ids = {
        json.loads(path.read_text(encoding="utf-8"))["$id"]
        for _, path in iter_schema_files()
    }
    assert manifest["root"] in bundled_ids
    for axis in ("figure_types", "methods", "transforms", "dataset_types"):
        for key, uri in manifest.get(axis, {}).items():
            assert uri in bundled_ids, f"{axis}.{key} points at an unbundled schema"


def test_no_schema_ref_reaches_the_network():
    """Every $ref must resolve inside the bundled catalog, never over HTTP."""
    bundled_prefixes = {
        json.loads(path.read_text(encoding="utf-8"))["$id"]
        for _, path in iter_schema_files()
    }

    def walk(node, location):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith(("http://", "https://")):
                base = ref.split("#", 1)[0]
                assert base in bundled_prefixes, f"{location}: {ref} is not bundled"
            for key, child in node.items():
                walk(child, f"{location}/{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{location}/{index}")

    for relative_name, path in iter_schema_files():
        walk(json.loads(path.read_text(encoding="utf-8")), relative_name)


# --------------------------------------------------------------------------- #
# B2: the catalog self-check
# --------------------------------------------------------------------------- #


def test_catalog_lints_clean():
    assert catalog_lint_errors() == []


def test_no_open_zone():
    """Jarvis-HEP v2 has an `open` zone for legacy surfaces; PLOT V2 does not."""
    assert ZONES == {"closed", "delegated"}


def test_lint_rejects_an_object_without_a_zone(monkeypatch, tmp_path):
    """The check that makes a missing zone a CI failure rather than a silent hole."""
    import jarvisplot.schema_catalog as catalog

    rogue = tmp_path / "rogue.json"
    rogue.write_text(
        json.dumps({"$id": "https://x/rogue.json", "type": "object", "properties": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(catalog, "iter_schema_files", lambda: iter([("rogue.json", rogue)]))
    errors = catalog.catalog_lint_errors()
    assert any("x-jarvis-zone" in e for e in errors)


# --------------------------------------------------------------------------- #
# DR-05: hand-written schema, kept honest by CI
# --------------------------------------------------------------------------- #


def test_method_enum_matches_the_registry():
    """The authoritative list is METHOD_DISPATCH; the schema must not drift."""
    schema_methods = set(subschema(LAYER_SCHEMA, "$defs", "methodName")["enum"])
    assert schema_methods == set(METHOD_DISPATCH), (
        "jarvisplot/schema/core/layer.json methodName enum has drifted from "
        "Figure/method_registry.py::METHOD_DISPATCH"
    )


# --------------------------------------------------------------------------- #
# B3: closed vocabulary -- the four observed failure cases
# --------------------------------------------------------------------------- #


def _validate(text: str) -> DiagnosticBag:
    import yaml

    return validate_config(yaml.safe_load(textwrap.dedent(text).lstrip()))


def _codes(bag: DiagnosticBag) -> set[str]:
    return {d.code for d in bag}


def _find(bag: DiagnosticBag, code: str, path: str):
    return next(d for d in bag if d.code == code and d.path == path)


def test_case_a_capitalized_layers_and_typoed_output():
    """Observed today: completely silent, exit 0, an empty PNG on disk."""
    bag = _validate(
        """
        DataSet: []
        Figures:
          - name: f1
            Layers: []
        outputs:
          dir: ./plots
        """
    )
    unknown = _find(bag, "JP-SCH-001", "$.Figures[0].Layers")
    assert unknown.context["did_you_mean"] == ["layers"]
    assert unknown.fix.op == "rename_key"
    assert unknown.fix.confidence == "certain", "case differs only; this is not a guess"

    typo = _find(bag, "JP-SCH-001", "$.outputs")
    assert typo.context["did_you_mean"] == ["output"]


def test_case_b_styles_typo_points_at_style():
    """Observed today: an unrelated KeyError, `Failed to configure figure 'f1': 'axes'`."""
    bag = _validate(
        """
        DataSet: []
        Figures:
          - name: f1
            styles: [a4paper_2x1, rect]
            layers: []
        """
    )
    unknown = _find(bag, "JP-SCH-001", "$.Figures[0].styles")
    assert unknown.context["did_you_mean"][0] == "style"
    assert unknown.suggestion == "Rename it to 'style'."


def test_case_c_unknown_method_is_caught_before_any_data_is_read():
    """Observed today: a KeyError after every dataset has been loaded, no suggestion."""
    bag = _validate(
        """
        DataSet: []
        Figures:
          - name: f1
            layers:
              - {name: s, axes: ax, method: scattr, coordinates: {x: {expr: a}}}
        """
    )
    bad = _find(bag, "JP-SCH-003", "$.Figures[0].layers[0].method")
    assert bad.context["did_you_mean"] == ["scatter"]
    assert "scatter" in bad.context["allowed_values"]


def test_unknown_key_reports_the_allowed_vocabulary():
    bag = _validate(
        """
        DataSet: []
        Figures: []
        wibble: 1
        """
    )
    unknown = _find(bag, "JP-SCH-001", "$.wibble")
    assert set(unknown.context["allowed_keys"]) == {
        "version",
        "project",
        "DataSet",
        "Figures",
        "Functions",
        "output",
    }
    assert unknown.fix is None, "no near match means no fix to offer"


def test_valid_config_produces_nothing():
    bag = _validate(
        """
        version: "0.3"
        project: {name: demo}
        DataSet: []
        Figures:
          - name: f1
            style: [a4paper_2x1, rect]
            frame: {ax: {xlim: [0, 1]}}
            layers: []
        output: {dir: ./plots, dpi: 150, formats: [png, pdf]}
        """
    )
    assert list(bag) == []


# --------------------------------------------------------------------------- #
# Shapes the real corpus depends on -- regressions against over-tightening
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "coordinates",
    [
        "{x: {expr: a}, y: {expr: b}}",
        "{x: [1, 2], y: [[0.1, 0.6], [0.01, 6.0]]}",
        "{left: {expr: a}, right: {expr: b}, bottom: {expr: c}}",
    ],
    ids=["expressions", "literal-arrays", "ternary-axes"],
)
def test_layer_coordinates_accept_every_form_the_runtime_supports(coordinates):
    bag = _validate(
        f"""
        DataSet: []
        Figures:
          - name: f1
            layers:
              - {{name: s, axes: ax, method: plot, coordinates: {coordinates}}}
        """
    )
    assert list(bag) == []


def test_combine_accepts_the_runtime_spelling():
    """`seperate` is spelled that way in Figure/layer_runtime.py; the schema follows."""
    assert set(subschema(LAYER_SCHEMA, "properties", "combine")["enum"]) == {
        "concat",
        "seperate",
    }


# --------------------------------------------------------------------------- #
# JP-OWN-001: keys the runtime accepts and never reads
# --------------------------------------------------------------------------- #


def test_figure_level_legend_is_reported_as_having_no_effect():
    """13 figures in this repo's own configs set it; none of them render a legend."""
    bag = _validate(
        """
        DataSet: []
        Figures:
          - name: f1
            legend: {loc: 1}
            layers: []
        """
    )
    warning = _find(bag, "JP-OWN-001", "$.Figures[0].legend")
    assert warning.level == "warning"
    assert warning.context["belongs_to"] == "frame.<axes>.legend"
    assert bag.ok, "an ignored key must not fail a config that renders today"


def test_layer_coordinate_lim_and_scale_are_reported():
    bag = _validate(
        """
        DataSet: []
        Figures:
          - name: f1
            layers:
              - name: s
                axes: ax
                method: scatter
                coordinates:
                  x: {expr: a, lim: [0, 5], scale: log, name: xx, label: "$x$"}
        """
    )
    reported = {d.path.rsplit(".", 1)[-1] for d in bag if d.code == "JP-OWN-001"}
    assert reported == {"lim", "scale", "name", "label"}


def test_ignored_key_annotations_come_from_the_schema():
    assert ignored_properties(FIGURE_SCHEMA) == {"legend": "frame.<axes>.legend"}
    coord = ignored_properties(LAYER_SCHEMA, "$defs", "coordinateSpec")
    assert set(coord) == {"name", "lim", "scale", "label"}
    assert all(target for target in coord.values())


# --------------------------------------------------------------------------- #
# Every diagnostic must be actionable
# --------------------------------------------------------------------------- #


def test_every_schema_diagnostic_carries_a_suggestion():
    bag = _validate(
        """
        DataSet:
          - {name: df, pth: ./x.csv, type: csvv}
        Figures:
          - Name: f1
            layers: {}
        junk: 1
        """
    )
    assert len(bag) >= 4
    for diagnostic in bag:
        assert diagnostic.suggestion, f"{diagnostic.code} at {diagnostic.path} has no suggestion"
        assert diagnostic.path.startswith("$")


def test_validator_is_cached():
    assert config_validator() is config_validator()
