"""A1 acceptance: the diagnostic model and the JSON envelope.

The envelope key set is a wire contract; these tests are the thing that makes
``api_version`` mean something.
"""

from __future__ import annotations

import io
import json

import pytest

from jarvisplot.agent_io import (
    API_VERSION,
    EXIT_FAILED,
    EXIT_OK,
    EXIT_USAGE,
    emit,
    envelope,
    error_payload,
    exit_code_for,
)
from jarvisplot.diagnostics import (
    Diagnostic,
    DiagnosticBag,
    Fix,
    did_you_mean,
    join_path,
)

ENVELOPE_KEYS = {"api_version", "kind", "ok", "data", "diagnostics", "error"}


# --------------------------------------------------------------------------- #
# envelope
# --------------------------------------------------------------------------- #


def test_envelope_key_set_is_frozen():
    env = envelope("validate", True)
    assert set(env) == ENVELOPE_KEYS
    assert env["api_version"] == API_VERSION
    assert env["kind"] == "validate"
    assert env["data"] == {}
    assert env["diagnostics"] == []
    assert env["error"] is None


def test_envelope_is_json_serializable_with_exotic_values():
    env = envelope("describe", True, data={"path": object()})
    text = json.dumps(env, default=str)
    assert json.loads(text)["kind"] == "describe"


def test_envelope_accepts_exception_as_error():
    env = envelope("run", False, error=FileNotFoundError("no such file"))
    assert env["error"] == {"type": "FileNotFoundError", "message": "no such file"}


def test_envelope_ok_is_independent_of_diagnostics():
    """describe may succeed while warning; validate may fail with no exception."""
    bag = DiagnosticBag()
    bag.warning("JP-COL-900", "$.DataSet[0]", "column is all NaN")
    env = envelope("describe", True, diagnostics=bag)
    assert env["ok"] is True
    assert len(env["diagnostics"]) == 1


def test_emit_writes_exactly_one_json_line():
    buffer = io.StringIO()
    code = emit(envelope("cap", True, data={"methods": ["plot"]}), buffer)
    text = buffer.getvalue()
    assert code == EXIT_OK
    assert text.count("\n") == 1
    assert json.loads(text)["data"]["methods"] == ["plot"]


@pytest.mark.parametrize(
    "env, expected",
    [
        (envelope("validate", True), EXIT_OK),
        (envelope("validate", False), EXIT_FAILED),
        (envelope("validate", False, error=error_payload("UsageError", "bad flag")), EXIT_USAGE),
    ],
)
def test_exit_codes(env, expected):
    assert exit_code_for(env) == expected


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #


def test_diagnostic_rejects_unknown_level():
    with pytest.raises(ValueError):
        Diagnostic(code="JP-SCH-001", level="fatal", path="$", message="x")


def test_fix_rejects_unknown_confidence():
    with pytest.raises(ValueError):
        Fix(op="rename_key", path="$.Layers", to="layers", confidence="maybe")


def test_diagnostic_to_dict_omits_empty_optionals():
    payload = Diagnostic(
        code="JP-SCH-001", level="error", path="$.Layers", message="unknown key"
    ).to_dict()
    assert set(payload) == {"code", "level", "path", "message", "suggestion"}


def test_diagnostic_to_dict_carries_fix_with_from_key():
    fix = Fix(op="rename_key", path="$.Figures[0].Layers", old="Layers", to="layers")
    payload = Diagnostic(
        code="JP-SCH-001",
        level="error",
        path="$.Figures[0].Layers",
        message="unknown key 'Layers'",
        suggestion="rename to 'layers'",
        fix=fix,
    ).to_dict()
    assert payload["fix"]["from"] == "Layers"
    assert payload["fix"]["to"] == "layers"
    assert payload["fix"]["confidence"] == "certain"


def test_bag_ok_ignores_warnings():
    bag = DiagnosticBag()
    bag.warning("JP-VIZ-002", "$.Figures[0]", "60% of points clipped")
    assert bag.ok is True
    bag.error("JP-COL-001", "$.Figures[0]", "column 'aa' not found")
    assert bag.ok is False


def test_bag_sorts_errors_before_warnings_then_by_path():
    bag = DiagnosticBag()
    bag.warning("JP-VIZ-002", "$.b", "w")
    bag.error("JP-COL-001", "$.z", "e2")
    bag.error("JP-COL-001", "$.a", "e1")
    assert [d.path for d in bag.sorted()] == ["$.a", "$.z", "$.b"]


def test_bag_fixable_excludes_heuristic_by_default():
    bag = DiagnosticBag()
    bag.error("JP-SCH-001", "$.a", "m", fix=Fix(op="rename_key", path="$.a", to="b"))
    bag.error(
        "JP-SCH-002",
        "$.c",
        "m",
        fix=Fix(op="set_value", path="$.c", to=1, confidence="heuristic"),
    )
    assert len(bag.fixable()) == 1
    assert len(bag.fixable(include_heuristic=True)) == 2


def test_diagnostic_render_human_includes_suggestion_and_example():
    text = Diagnostic(
        code="JP-MTH-001",
        level="error",
        path="$.Figures[0].layers[1]",
        message="method 'pcolormesh' requires coordinates.z",
        suggestion="add a z coordinate",
        example="coordinates:\n  z: {expr: mass}",
    ).render_human()
    assert "JP-MTH-001" in text
    assert "-> add a z coordinate" in text
    assert "z: {expr: mass}" in text


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "parts, expected",
    [
        (("Figures", 0, "layers", 1, "coordinates"), "$.Figures[0].layers[1].coordinates"),
        (("DataSet",), "$.DataSet"),
        ((), "$"),
        (("$.Figures[0]", "style"), "$.Figures[0].style"),
    ],
)
def test_join_path(parts, expected):
    assert join_path(*parts) == expected


def test_did_you_mean_catches_case_only_typos_first():
    """The observed case-A failure: 'Layers:' silently ignored next to 'layers'."""
    assert did_you_mean("Layers", ["layers", "label", "lim"]) == ["layers"]


def test_did_you_mean_catches_near_spellings():
    assert "output" in did_you_mean("outputs", ["output", "Figures", "DataSet"])


def test_did_you_mean_returns_empty_for_nonsense():
    assert did_you_mean("zzzzzz", ["layers", "output"]) == []
