from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import yaml

from jarvisplot.core import JarvisPLOT
from jarvisplot.data_loader_hdf5 import scan_hdf5_leaf_metadata


def _write_yaml(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _make_parse_args(file_path: Path, out_path: Path):
    return SimpleNamespace(
        file=str(file_path),
        out=str(out_path),
        inplace=False,
        parse_data=True,
        debug=False,
        no_logo=True,
        rebuild_cache=False,
    )


def test_scan_hdf5_leaf_metadata_returns_leaf_paths(tmp_path):
    h5_path = tmp_path / "fixture.hdf5"
    with h5py.File(h5_path, "w") as h5f:
        grp = h5f.create_group("data")
        grp.create_dataset("signal", data=np.array([1, 2, 3], dtype=np.int64))
        grp.create_dataset("signal_isvalid", data=np.array([1, 0, 1], dtype=np.int8))
        nested = grp.create_group("nested")
        nested.create_dataset("value", data=np.array([4, 5, 6], dtype=np.int64))

    metadata = scan_hdf5_leaf_metadata(str(h5_path), group="data")
    paths = [item["path"] for item in metadata]

    assert sorted(paths) == sorted([
        "data/signal",
        "data/signal_isvalid",
        "data/nested/value",
    ])
    assert any(item["path"] == "data/signal" and item["shape"] == (3,) and "int64" in item["dtype"] for item in metadata)


def test_parse_data_writes_yaml_from_metadata_only(tmp_path, monkeypatch):
    workdir = tmp_path / "workdir"
    data_dir = workdir / "datas"
    data_dir.mkdir(parents=True)
    h5_path = data_dir / "fixture.hdf5"
    with h5py.File(h5_path, "w") as h5f:
        grp = h5f.create_group("data")
        grp.create_dataset("signal", data=np.array([1, 2, 3], dtype=np.int64))
        grp.create_dataset("signal_isvalid", data=np.array([1, 0, 1], dtype=np.int8))
        nested = grp.create_group("nested")
        nested.create_dataset("value", data=np.array([4, 5, 6], dtype=np.int64))

    yaml_path = tmp_path / "input.yaml"
    out_path = tmp_path / "output.yaml"
    _write_yaml(
        yaml_path,
        {
            "DataSet": [
                    {
                        "dataset": "data",
                        "name": "h5",
                        "path": "./datas/fixture.hdf5",
                        "type": "hdf5",
                        "columns": {
                            "load_whitelist": "only_in_list",
                            "note": "keep-me",
                        },
                    }
                ],
                "output": {"dir": str(tmp_path / "plots"), "dpi": 400, "formats": ["png"]},
                "project": {"name": "test", "workdir": str(workdir)},
                "version": 1.1,
            },
        )

    def _fail(*args, **kwargs):
        raise AssertionError("runtime load path should not be called for parse-data")

    monkeypatch.setattr(JarvisPLOT, "load_dataset", _fail)
    monkeypatch.setattr(JarvisPLOT, "prepare_project_layout", _fail)

    app = JarvisPLOT()
    monkeypatch.setattr(app.cli.args, "parse_args", lambda: _make_parse_args(yaml_path, out_path))

    app.init()

    out_cfg = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    ds = out_cfg["DataSet"][0]
    cols = ds["columns"]
    rename = cols["rename"]
    rename_map = {item["source"]: item["target"] for item in rename}

    assert sorted(item["source"] for item in rename) == sorted([
        "data/signal",
        "data/nested/value",
    ])
    assert set(rename_map.keys()) == {
        "data/signal",
        "data/nested/value",
    }
    assert set(rename_map.values()) == {
        "Var0@h5",
        "Var1@h5",
    }
    assert cols["load_whitelist"] == "only_in_list"
    assert cols["note"] == "keep-me"


def test_load_yaml_missing_file_logs_error_and_exits(tmp_path):
    app = JarvisPLOT()
    app.args = SimpleNamespace(file=str(tmp_path / "missing.yaml"))

    errors = []
    app.logger = SimpleNamespace(
        error=lambda msg: errors.append(str(msg)),
        warning=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )

    with pytest.raises(SystemExit) as excinfo:
        app.load_yaml()

    assert excinfo.value.code == 2
    assert errors
    assert "YAML file not found" in errors[0]


@pytest.mark.parametrize(
    "group_name, expected",
    [
        ("missing", "not found"),
        ("empty", "contains no leaf datasets"),
    ],
)
def test_scan_hdf5_leaf_metadata_reports_group_errors(tmp_path, group_name, expected):
    h5_path = tmp_path / "fixture.hdf5"
    with h5py.File(h5_path, "w") as h5f:
        h5f.create_group("empty")
        data = h5f.create_group("data")
        data.create_dataset("signal", data=np.array([1, 2, 3], dtype=np.int64))

    with pytest.raises(RuntimeError, match=expected):
        scan_hdf5_leaf_metadata(str(h5_path), group=group_name)
