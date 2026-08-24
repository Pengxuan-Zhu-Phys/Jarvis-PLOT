"""Every card's Debug block must use keys the overlay actually reads.

Card JSON is not schema-validated, and design_runtime swallows exceptions, so
a typo in a Debug key produces no error, no warning at default log level, and
no visible effect -- it just silently does nothing. This walks the shipped
cards at CI time instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvisplot.Figure.debug_config import DEFAULT_DEBUG, merge_debug_config

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "jarvisplot" / "cards"


def _registered_cards():
    pref = json.loads((CARDS / "style_preference.json").read_text(encoding="utf-8"))
    seen = {}
    for family, variants in pref.items():
        for token, path in variants.items():
            seen.setdefault(path.replace("&JP/", ""), []).append(f"{family}/{token}")
    return sorted(seen.items())


CARD_FILES = _registered_cards()


def _load(relpath):
    return json.loads((ROOT / relpath).read_text(encoding="utf-8"))


@pytest.mark.parametrize("relpath, used_by", CARD_FILES, ids=[c[0] for c in CARD_FILES])
def test_every_registered_card_declares_a_debug_block(relpath, used_by):
    assert "Debug" in _load(relpath), f"{relpath} is used by {used_by}"


@pytest.mark.parametrize("relpath, _used_by", CARD_FILES, ids=[c[0] for c in CARD_FILES])
def test_debug_block_uses_only_keys_the_overlay_reads(relpath, _used_by):
    _, problems = merge_debug_config(DEFAULT_DEBUG, _load(relpath)["Debug"])
    assert problems == [], f"{relpath}:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("relpath, _used_by", CARD_FILES, ids=[c[0] for c in CARD_FILES])
def test_every_show_is_a_boolean(relpath, _used_by):
    bad = []

    def walk(node, path):
        if not isinstance(node, dict):
            return
        if "show" in node and not isinstance(node["show"], bool):
            bad.append(f"{path}show = {node['show']!r}")
        for key, value in node.items():
            walk(value, f"{path}{key}.")

    walk(_load(relpath)["Debug"], "")
    assert bad == [], f"{relpath}: " + ", ".join(bad)


@pytest.mark.parametrize("relpath, _used_by", CARD_FILES, ids=[c[0] for c in CARD_FILES])
def test_ternary_knobs_only_ship_on_cards_that_have_a_ternary_axes(relpath, _used_by):
    """Advertising settings that cannot take effect is worse than omitting them."""
    card = _load(relpath)
    axes = list((card.get("Frame", {}) or {}).get("axes", {}))
    has_axtri = any(str(name).startswith("axtri") for name in axes)
    assert ("ternary" in card["Debug"]) == has_axtri


def test_all_registered_cards_are_covered():
    assert len(CARD_FILES) >= 13
