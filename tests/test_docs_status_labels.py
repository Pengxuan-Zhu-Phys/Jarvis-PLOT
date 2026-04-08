from __future__ import annotations

from pathlib import Path
import re


ALLOWED_STATUSES = {
    "active backlog",
    "active",
    "implemented",
    "implemented but mixed",
    "partial",
    "spec only",
    "historical",
}


def _status_line(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines()[:12]:
        match = re.match(r"^Status:\s*(.+?)\s*$", line)
        if match:
            return match.group(1)
    return None


def test_docs_status_labels_are_consistent() -> None:
    docs_root = Path(__file__).resolve().parents[1] / "docs"
    assert docs_root.exists()

    seen_status = False
    for path in docs_root.rglob("*.md"):
        if path.name == "README.md":
            continue
        status = _status_line(path)
        if status is None:
            continue

        seen_status = True
        assert status in ALLOWED_STATUSES, f"Unexpected status '{status}' in {path}"

        if status == "historical":
            assert "/docs/archive/" in f"/{path.as_posix()}/", (
                f"Historical doc should live in docs/archive: {path}"
            )
        if status == "active backlog":
            assert "/docs/roadmap/" in f"/{path.as_posix()}/", (
                f"Active backlog doc should live in docs/roadmap: {path}"
            )

    assert seen_status, "No docs status labels found under docs/"
