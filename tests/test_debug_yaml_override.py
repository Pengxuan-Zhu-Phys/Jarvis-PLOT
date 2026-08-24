"""`Figures[].debug` accepts a boolean or a per-figure override mapping.

Precedence is DEFAULT_DEBUG < style card `Debug` < YAML `debug:` mapping.
The ordering trap this guards: apply_figure_config assigns `debug` before
`style`, and the style setter replaces `_debug_config` wholesale -- so the YAML
override has to live in its own slot or it is silently lost.
"""

from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")

import pytest
import yaml

from jarvisplot.Figure.debug_config import DEFAULT_DEBUG, resolve_debug_config
from jarvisplot.Figure.figure import Figure
from jarvisplot.validation import validate_config


# --------------------------------------------------------------------------- #
# Setter
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, expected_on, expected_overrides",
    [
        (True, True, {}),
        (False, False, {}),
        (1, True, {}),
        (None, False, {}),
        ({}, True, {}),
        ({"panel": {"show": False}}, True, {"panel": {"show": False}}),
        ({"show": False}, False, {}),
        ({"show": False, "panel": {"entry_gap": 4.0}}, False, {"panel": {"entry_gap": 4.0}}),
    ],
)
def test_debug_setter_forms(value, expected_on, expected_overrides):
    fig = Figure()
    fig.debug = value
    assert fig.debug is expected_on
    assert fig._debug_overrides == expected_overrides


def test_debug_stays_a_bool_so_the_master_switch_is_one_thing():
    """`self.debug` is the single internal on/off; it never holds the mapping."""
    fig = Figure()
    fig.debug = {"panel": {"show": False}}
    assert isinstance(fig.debug, bool)


# --------------------------------------------------------------------------- #
# Precedence
# --------------------------------------------------------------------------- #


def test_yaml_override_wins_over_the_style_card():
    fig = Figure()
    fig._debug_config = {"panel": {"entry_gap": 3.0, "char_width_factor": 0.7}}
    fig.debug = {"panel": {"entry_gap": 4.0}}
    merged = fig.debug_config
    assert merged["panel"]["entry_gap"] == 4.0
    assert merged["panel"]["char_width_factor"] == 0.7, "sibling card keys survive"


def test_card_wins_over_the_packaged_defaults():
    fig = Figure()
    fig._debug_config = {"panel": {"char_width_factor": 0.33}}
    fig.debug = True
    resolved = resolve_debug_config(fig)
    assert resolved["panel"]["char_width_factor"] == 0.33
    assert resolved["panel"]["entry_gap"] == DEFAULT_DEBUG["panel"]["entry_gap"]


def test_full_precedence_chain():
    fig = Figure()
    fig._debug_config = {"panel": {"char_width_factor": 0.33, "entry_gap": 9.0}}
    fig.debug = {"panel": {"char_width_factor": 0.11}}
    resolved = resolve_debug_config(fig)
    assert resolved["panel"]["char_width_factor"] == 0.11, "YAML wins"
    assert resolved["panel"]["entry_gap"] == 9.0, "card wins where YAML is silent"
    assert resolved["panel"]["header"] == DEFAULT_DEBUG["panel"]["header"], "defaults fill the rest"


def test_override_does_not_leak_between_figures():
    a, b = Figure(), Figure()
    a.debug = {"panel": {"entry_gap": 1.0}}
    b.debug = True
    assert b.debug_config == {}


def test_a_later_style_assignment_does_not_clobber_the_yaml_override():
    """The ordering trap: config_runtime sets debug first, style second."""
    fig = Figure()
    fig.debug = {"panel": {"entry_gap": 4.2}}
    fig._debug_config = {"panel": {"char_width_factor": 0.7}}  # the style setter
    assert fig.debug_config["panel"]["entry_gap"] == 4.2


def test_list_valued_keys_are_replaced_not_appended():
    fig = Figure()
    fig._debug_config = {"primary_order": ["ax", "axtri"]}
    fig.debug = {"primary_order": ["axtri"]}
    assert fig.debug_config["primary_order"] == ["axtri"]


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def _validate(debug_literal):
    text = textwrap.dedent(
        f"""
        DataSet: []
        Figures:
          - name: f1
            debug: {debug_literal}
            layers: []
        """
    ).lstrip()
    return validate_config(yaml.safe_load(text), check_columns=False)


@pytest.mark.parametrize(
    "literal",
    ["true", "false", "{panel: {show: false}}", "{show: false}",
     '{dimension: {cap: {plot: {color: "#888888"}}}}', "{primary_order: [axtri]}"],
)
def test_schema_accepts_both_forms(literal):
    assert [d.code for d in _validate(literal)] == []


def test_schema_rejects_an_unknown_group_with_a_suggestion():
    bag = _validate("{pannel: {show: false}}")
    unknown = next(d for d in bag if d.code == "JP-SCH-001")
    assert unknown.path == "$.Figures[0].debug.pannel"
    assert unknown.context["did_you_mean"][0] == "panel"


def test_schema_still_rejects_a_bogus_scalar():
    assert [d.code for d in _validate("[1, 2]")] != []
