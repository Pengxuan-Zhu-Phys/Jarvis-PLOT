"""B4: every JP-* code has non-empty guidance; bag fills empty suggestions."""

from __future__ import annotations

from jarvisplot.diagnostic_guidance import KNOWN_CODES, guidance_for
from jarvisplot.diagnostics import Diagnostic, DiagnosticBag


def test_every_known_code_has_nonempty_suggestion():
    for code in sorted(KNOWN_CODES):
        suggestion, _example = guidance_for(code, "$.Figures[0]", "placeholder message")
        assert suggestion.strip(), f"{code} has empty suggestion"


def test_parameter_code_wins_over_path_prefix():
    """HEP D21.14 lesson: code table beats path prefix rules."""
    suggestion, example = guidance_for(
        "JP-COL-001",
        "$.Figures[0].layers[0].coordinates.x.expr",
        "dataset 'df' has no column 'aa'",
    )
    assert "column" in suggestion.lower() or "describe" in suggestion.lower()
    assert example is not None or "describe" in suggestion.lower()


def test_bag_fills_empty_suggestion_from_guidance():
    bag = DiagnosticBag()
    bag.add(
        Diagnostic(
            code="JP-YML-002",
            level="error",
            path="$",
            message="config is empty",
            # deliberately no suggestion
        )
    )
    d = list(bag)[0]
    assert d.suggestion
    assert "DataSet" in d.suggestion or "Figures" in d.suggestion


def test_bag_preserves_explicit_suggestion():
    bag = DiagnosticBag()
    bag.error(
        "JP-SCH-001",
        "$.Layers",
        "unknown key 'Layers'",
        suggestion="Rename it to 'layers'.",
    )
    assert list(bag)[0].suggestion == "Rename it to 'layers'."


def test_family_fallback_for_unknown_code():
    suggestion, _ = guidance_for("JP-SCH-999", "$.x", "something required is missing")
    assert suggestion.strip()
