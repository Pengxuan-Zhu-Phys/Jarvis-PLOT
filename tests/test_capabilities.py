"""D1–D5: capability collectors stay locked to the runtime registries."""

from __future__ import annotations

import json
import sys

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
    by = {e["name"]: e for e in section("transforms")}
    # contracts, not delegated stubs
    assert by["make_interp_2d"].get("required")
    assert "delegated" not in str(by["profile"].get("keys", ""))
    assert by["profile"].get("defaults", {}).get("method") == "bridson"


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


def test_correlation_capabilities_expose_type_method_and_cards():
    types = {entry["name"]: entry for entry in section("types")}
    corr_type = types["correlation_matrix"]
    assert corr_type["method"] == "corrplot"
    assert {tuple(style) for style in corr_type["styles"]} == {
        ("corrplot", "matrix"),
        ("corrplot", "diamond"),
    }
    assert {"axcorr", "axccorr"} <= set(corr_type["axes"])

    methods = {entry["name"]: entry for entry in section("methods")}
    corr_method = methods["corrplot"]
    assert corr_method["figure_type"] == "correlation_matrix"
    assert {tuple(style) for style in corr_method["compatible_styles"]} == {
        ("corrplot", "matrix"),
        ("corrplot", "diamond"),
    }
    assert "side" in corr_method["style_keys"]["corrplot.diamond"]["corrplot"]

    styles = {
        (entry["bundle"], entry["token"]): entry for entry in section("styles")
    }
    assert styles[("corrplot", "diamond")]["contract"]["layout"] == "diamond"
    assert styles[("corrplot", "matrix")]["layout"] == "matrix"
    assert "stripe" in styles[("corrplot", "diamond")]["style_keys"]["corrplot"]


def test_cmaps_include_jarvis_and_reversed():
    cmaps = section("cmaps")
    jarvis = cmaps["jarvis"]
    assert jarvis
    assert cmaps["jarvis_reversed"] == [f"{n}_r" for n in jarvis]
    matplotlib = cmaps["matplotlib"]
    assert matplotlib
    by_name = {entry["name"]: entry for entry in matplotlib}
    assert by_name["viridis"]["type"] == "ListedColormap"
    assert by_name["viridis"]["N"] == 256
    assert by_name["viridis"]["reverse"] == "viridis_r"


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


def test_jplot_cap_bare_is_index_not_full_dump(capsys):
    """Bare `jplot cap` must not emit cap.all (P2.1); agents pass section or --json index."""
    assert main(["cap", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is True
    assert env["kind"] == "cap.index"
    sections = env["data"].get("sections") or []
    assert "all" in sections
    assert "methods" in sections
    # Must not dump the full catalogue (no digest / method list payload)
    assert "digest" not in env["data"]
    assert not isinstance(env["data"].get("methods"), list)


def test_jplot_cap_bare_human_is_a_rich_section_card(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(["cap"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "jplot cap" in captured.err
    assert "Sections · jplot cap" in captured.err
    assert "Open next" in captured.err
    assert "methods" in captured.err
    assert "cap.all" not in captured.err


def test_jplot_cap_methods_human_uses_table_and_man_links(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(["cap", "methods"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Methods · jplot cap methods" in captured.err
    assert "Required" in captured.err
    assert "scatter" in captured.err
    assert "jplot man scatter" in captured.err
    assert not captured.err.lstrip().startswith("{")


def test_jplot_cap_all_human_is_summary_card(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(["cap", "all"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Capabilities · jplot cap all" in captured.err
    assert "Digest:" in captured.err
    assert "methods" in captured.err
    assert "jplot cap all --json" in captured.err


def test_jplot_cap_cmaps_human_lists_each_name(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(["cap", "cmaps"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Colormaps · jplot cap cmaps" in captured.err
    assert "Jarvis reversed" not in captured.err
    for name in section("cmaps")["jarvis"]:
        assert name in captured.err
    assert "Diverging red" in captured.err
    assert "automatic _r reverse" in captured.err


def test_jplot_cap_cmaps_human_separates_matplotlib_registry(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(["cap", "cmaps"]) == 0
    captured = capsys.readouterr()
    out = captured.err
    assert "viridis" in out
    assert "viridis_r" in out
    assert "Matplotlib ListedColormap" in out
    assert "Matplotlib-defined" in out
    assert out.index("RdBuB") < out.index("viridis")
    # The SIMPLE_HEAVY table has a horizontal section rule between registries.
    assert out.count("━━━━━━━━") >= 2


def test_jplot_cap_styles_json_marks_broken_cards(capsys):
    assert main(["cap", "styles", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["kind"] == "cap.styles"
    styles = env["data"]["styles"]
    broken = [s for s in styles if not s.get("usable", True)]
    assert broken
    axc_cards = [s for s in styles if "axc" in (s.get("axes") or [])]
    assert axc_cards


def test_jplot_cap_cmaps_json_includes_matplotlib_registry(capsys):
    assert main(["cap", "cmaps", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    cmaps = env["data"]["cmaps"]
    entries = cmaps["matplotlib"]
    assert len(entries) >= 100
    by_name = {entry["name"]: entry for entry in entries}
    assert by_name["viridis"] == {
        "name": "viridis",
        "type": "ListedColormap",
        "N": 256,
        "reverse": "viridis_r",
    }
    assert "matplotlib_note" in cmaps


def test_jplot_cap_unknown_section_is_usage(capsys):
    assert main(["cap", "nope", "--json"]) == 2
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is False
    assert env["error"]["type"] == "UsageError"
