"""Every shipped example must pass ``jplot validate`` with nothing to report.

The examples are what users copy from, so a validator rule that rejects one of
them is a bug in the validator (or in the example) either way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvisplot.validation import validate_file

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "Example"
EXAMPLES = sorted(EXAMPLE_DIR.glob("*.yaml"))


def test_examples_are_present():
    assert EXAMPLES, f"no example configs found under {EXAMPLE_DIR}"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_validates_cleanly(path):
    config, bag = validate_file(str(path))

    assert config is not None
    assert bag.ok, [(d.code, d.path, d.message) for d in bag.errors]
    assert not bag.warnings, [(d.code, d.path, d.message) for d in bag.warnings]
