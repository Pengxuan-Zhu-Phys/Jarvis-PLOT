from __future__ import annotations

from jarvisplot.utils.interpolator import InterpolatorManager


def test_interpolator_manager_from_yaml_accepts_empty_config(tmp_path):
    mgr = InterpolatorManager.from_yaml({}, tmp_path)

    assert mgr.summary() == {"registered": [], "built": []}
