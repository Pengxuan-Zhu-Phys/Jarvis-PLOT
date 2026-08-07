"""Transform contracts stay locked to schema vocabulary + heavy runtime keys.

Mirrors ``tests/test_method_contracts.py`` for the methods axis: without this
guard, ``jplot cap transforms`` / ``man transform.*`` silently drift from
``Figure/*_runtime.py`` (see AGENT_CLI_FULL_REVIEW P1.2).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvisplot.capabilities import section
from jarvisplot.client import main
from jarvisplot.schema_catalog import subschema
from jarvisplot.transform_contracts import (
    RUNTIME_TOP_LEVEL_KEYS,
    TRANSFORM_NAMES,
    contract_for,
    contract_top_level_keys,
    list_contracts,
)

SCHEMA_TRANSFORM = "https://jarvis-plot.org/schema/v2/core/transform.json"


def test_schema_vocabulary_matches_contracts():
    schema = subschema(SCHEMA_TRANSFORM)
    schema_names = {n for n in (schema.get("properties") or {}) if n != "type"}
    assert set(TRANSFORM_NAMES) == schema_names


def test_list_contracts_covers_every_name():
    by_name = {c["name"] for c in list_contracts()}
    assert by_name == set(TRANSFORM_NAMES)


@pytest.mark.parametrize("name", sorted(RUNTIME_TOP_LEVEL_KEYS))
def test_heavy_runtime_keys_match_contract(name: str):
    """Contract required∪optional must equal the locked runtime key set."""
    expected = RUNTIME_TOP_LEVEL_KEYS[name]
    actual = contract_top_level_keys(name)
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"{name}: contract missing runtime keys {sorted(missing)}"
    assert not extra, f"{name}: contract has keys runtime lock does not list {sorted(extra)}"


def test_profile_exposes_pregrid_not_ghost_bins_seed():
    keys = contract_top_level_keys("profile")
    assert "pregrid" in keys
    assert "pregrid_bin" in keys
    assert "bins" not in keys  # ghost — belongs to density / posterior
    assert "seed" not in keys  # ghost — profile_runtime never reads seed
    c = contract_for("profile")
    assert c is not None
    pre = (c.get("optional") or {}).get("pregrid") or {}
    assert "bin" in (pre.get("properties") or {})
    assert "enable" in (pre.get("properties") or {})


def test_cap_transforms_advertise_profile_pregrid():
    by = {e["name"]: e for e in section("transforms")}
    opt = by["profile"].get("optional") or {}
    assert "pregrid" in opt
    assert "pregrid_bin" in opt
    assert "bins" not in opt
    assert "seed" not in opt


def test_man_transform_profile_json_includes_pregrid(capsys):
    assert main(["man", "transform.profile", "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["ok"] is True
    # live card surfaces optional keys under data / sections depending on renderer
    blob = json.dumps(env.get("data") or env)
    assert "pregrid" in blob
    assert "pregrid_bin" in blob
