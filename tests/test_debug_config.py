"""The Debug-block merge: precedence, and what it refuses to do in silence.

Before this suite, `merge_debug_config` had never been exercised with a
non-None override -- the only card carrying a `Debug` block is a byte-for-byte
copy of the defaults, so every existing test ran the defaults path. The two
silent-failure modes below are what that gap was hiding.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvisplot.Figure.debug_config import (
    DEFAULT_DEBUG,
    merge_debug_config,
    resolve_debug_config,
)


def _merge(override):
    return merge_debug_config(DEFAULT_DEBUG, override)


# --------------------------------------------------------------------------- #
# Precedence
# --------------------------------------------------------------------------- #


def test_no_override_returns_the_defaults():
    merged, problems = _merge(None)
    assert problems == []
    assert merged == DEFAULT_DEBUG


def test_override_wins_on_a_leaf_and_leaves_siblings_alone():
    merged, problems = _merge({"panel": {"right": 0.55}})
    assert problems == []
    assert merged["panel"]["right"] == 0.55
    assert merged["panel"]["entry_gap"] == DEFAULT_DEBUG["panel"]["entry_gap"]
    assert merged["dimension"] == DEFAULT_DEBUG["dimension"]


def test_merge_does_not_mutate_the_defaults():
    merged, _ = _merge({"figure": {"caption": {"x": 0.1}}})
    assert merged["figure"]["caption"]["x"] == 0.1
    assert DEFAULT_DEBUG["figure"]["caption"]["x"] == 0.5


# --------------------------------------------------------------------------- #
# The two silent failures
# --------------------------------------------------------------------------- #


def test_unknown_key_is_reported_with_a_suggestion_and_dropped():
    """It used to be carried through as inert junk -- a typo looked like a win."""
    merged, problems = _merge({"pannel": {"right": 0.5}})
    assert "pannel" not in merged
    assert len(problems) == 1
    assert "unknown Debug key" in problems[0]
    assert "'panel'" in problems[0]


def test_unknown_nested_key_reports_its_full_path():
    merged, problems = _merge({"panel": {"rigth": 0.5}})
    assert "rigth" not in merged["panel"]
    assert problems and problems[0].startswith("panel.rigth:")
    assert "'right'" in problems[0]


def test_scalar_over_mapping_is_reported_instead_of_discarded():
    """It used to `continue` past this, keeping the default and saying nothing."""
    merged, problems = _merge({"panel": "#ffffff"})
    assert merged["panel"] == DEFAULT_DEBUG["panel"]
    assert len(problems) == 1
    assert "expected a mapping" in problems[0]


def test_unknown_key_with_no_close_match_still_reports():
    _, problems = _merge({"zzzzzz": 1})
    assert len(problems) == 1
    assert "Did you mean" not in problems[0]


# --------------------------------------------------------------------------- #
# Delegated leaves
# --------------------------------------------------------------------------- #


def test_call_blocks_accept_arbitrary_matplotlib_kwargs():
    """A call block is matplotlib's vocabulary, not Jarvis-PLOT's."""
    base = {"outline": {"Rectangle": {"linewidth": 0.45}}}
    merged, problems = merge_debug_config(base, {"outline": {"Rectangle": {"hatch": "//"}}})
    assert problems == []
    assert merged["outline"]["Rectangle"] == {"linewidth": 0.45, "hatch": "//"}


def test_call_block_merges_rather_than_replaces():
    base = {"caption": {"text": {"fontsize": 6.0, "va": "top"}}}
    merged, _ = merge_debug_config(base, {"caption": {"text": {"fontsize": 9.0}}})
    assert merged["caption"]["text"] == {"fontsize": 9.0, "va": "top"}


def test_a_nested_call_block_leaf_keeps_the_keys_the_override_omits():
    """`bbox` and `arrowprops` sit inside a call block; a partial override
    must not wipe the rest of them."""
    base = {"hlabel": {"text": {"bbox": {"fc": "white", "alpha": 0.5}}}}
    merged, _ = merge_debug_config(base, {"hlabel": {"text": {"bbox": {"alpha": 0.9}}}})
    assert merged["hlabel"]["text"]["bbox"] == {"fc": "white", "alpha": 0.9}


# --------------------------------------------------------------------------- #
# resolve_debug_config
# --------------------------------------------------------------------------- #


def test_resolve_reads_the_public_attribute_first():
    fig = SimpleNamespace(
        debug_config={"panel": {"right": 0.4}},
        _debug_config={"panel": {"right": 0.9}},
        logger=None,
    )
    assert resolve_debug_config(fig)["panel"]["right"] == 0.4


def test_resolve_falls_back_to_the_private_attribute():
    fig = SimpleNamespace(_debug_config={"panel": {"right": 0.9}}, logger=None)
    assert resolve_debug_config(fig)["panel"]["right"] == 0.9


def test_resolve_warns_once_about_everything_it_dropped():
    warnings = []
    fig = SimpleNamespace(
        _debug_config={"pannel": {}, "panel": 1},
        logger=SimpleNamespace(warning=warnings.append),
    )
    resolve_debug_config(fig)
    assert len(warnings) == 1
    assert "pannel" in warnings[0] and "expected a mapping" in warnings[0]


@pytest.mark.parametrize("logger", [None, SimpleNamespace(), object()])
def test_resolve_never_raises_on_a_useless_logger(logger):
    """A broken overlay must not be able to break a plot."""
    fig = SimpleNamespace(_debug_config={"nope": 1}, logger=logger)
    assert resolve_debug_config(fig)["panel"] == DEFAULT_DEBUG["panel"]
