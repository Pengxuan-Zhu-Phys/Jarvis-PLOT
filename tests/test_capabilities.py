"""D1–D5: capability collectors stay locked to the runtime registries."""

from __future__ import annotations

import json

import pytest

from jarvisplot.capabilities import CAPABILITY_SECTIONS, capabilities, digest, section
from jarvisplot.client import main
from jarvisplot.Figure.method_registry import METHOD_DISPATCH


def test_methods_match_registry():
    names = {entry["name"] for entry in section("methods")}
    assert names == set(METHOD_DISPATCH)


def test_transforms_include_to_parquet_and_to_csv():
    names = {entry["name"] for entry in section("transforms")}
    assert "to_csv" in names
    assert "to_parquet" in names
    assert "filter" in names


def test_style_cards_expose_axes_and_usable():
    styles = section("styles")
    assert styles, "style_preference.json should advertise at least one card"

    by_key = {(s["bundle"], s["token"]): s for s in styles}
    # rectcmap cards expose axc; plain rect does not
    cmap_hits = [s for s in styles if "cmap" in s["token"].lower() or "Cmap" in s["token"]]
    assert cmap_hits, "expected at least one *cmap style token"
    assert any("axc" in (s.get("axes") or []) for s in cmap_hits)

    # Broken 1x1 Ternary cards (Figure top-level, no Frame) must be unusable.
    broken = [
        s
        for s in styles
        if s["token"] == "Ternary" and "1x1" in s["bundle"] and not s.get("usable", True)
    ]
    assert broken, "a4paper_1x1/gambit_1x1 Ternary should be flagged unusable"
    for entry in broken:
        assert entry.get("axes") == []
        assert "Frame" in (entry.get("error") or "")


def test_cmaps_include_jarvis_and_reversed():
    cmaps = section("cmaps")
    jarvis = cmaps["jarvis"]
    assert jarvis
    assert cmaps["jarvis_reversed"] == [f"{n}_r" for n in jarvis]


def test_funcs_are_nonempty():
    funcs = section("funcs")
    assert "exp" in funcs["names"] or "ln" in funcs["names"]
    assert "np" in funcs["namespaces"] or "np" in funcs["names"]


def test_cap_cli_matches_spec():
    from pathlib import Path
    import jarvisplot

    spec_path = Path(jarvisplot.__file__).with_name("cards") / "args.json"
    on_disk = json.loads(spec_path.read_text(encoding="utf-8"))
    assert section("cli") == on_disk


def test_digest_stability():
    a = digest()
    b = digest()
    assert a == b
    assert len(a) == 16
    full = capabilities()
    assert full["digest"] == a
    for name in CAPABILITY_SECTIONS:
        assert name in full


def test_jplot_cap_all_json(capsys):
    assert main(["cap", "all", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "cap.all"
    assert env["ok"] is True
    assert "methods" in env["data"]
    assert "styles" in env["data"]
    assert "digest" in env["data"]
    assert len(env["data"]["methods"]) == len(METHOD_DISPATCH)


def test_jplot_cap_styles_json_marks_broken_cards(capsys):
    assert main(["cap", "styles", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "cap.styles"
    styles = env["data"]["styles"]
    broken = [s for s in styles if not s.get("usable", True)]
    assert broken
    axc_cards = [s for s in styles if "axc" in (s.get("axes") or [])]
    assert axc_cards


def test_jplot_cap_unknown_section_is_usage(capsys):
    assert main(["cap", "nope", "--json"]) == 2
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is False
    assert env["error"]["type"] == "UsageError"
