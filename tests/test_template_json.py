from __future__ import annotations

import json
from pathlib import Path

import pytest


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "docs" / "templates"
TEMPLATE_JSON_FILES = sorted(TEMPLATE_DIR.glob("*.json"))
TEMPLATE_JSON_NAMES = {path.name for path in TEMPLATE_JSON_FILES}
REQUIRED_TEMPLATE_JSON_NAMES = {
    "scene.example.json",
    "style.example.json",
    "profile.example.json",
}

assert TEMPLATE_JSON_FILES, f"No template JSON files found in {TEMPLATE_DIR}"
assert REQUIRED_TEMPLATE_JSON_NAMES.issubset(TEMPLATE_JSON_NAMES), (
    f"Missing template JSON files: {sorted(REQUIRED_TEMPLATE_JSON_NAMES - TEMPLATE_JSON_NAMES)}"
)


@pytest.mark.parametrize("path", TEMPLATE_JSON_FILES)
def test_template_json_files_parse(path):
    json.loads(path.read_text(encoding="utf-8"))
