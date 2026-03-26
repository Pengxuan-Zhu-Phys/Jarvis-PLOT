from __future__ import annotations

import json
from pathlib import Path


def test_template_json_files_parse():
    template_dir = Path(__file__).resolve().parents[1] / "docs" / "templates"
    for path in sorted(template_dir.glob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))
