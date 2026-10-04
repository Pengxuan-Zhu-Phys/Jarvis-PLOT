from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jarvisplot.cache_store import ProjectCache


def test_cleanup_removes_used_and_new_payloads_and_dangling_references(tmp_path):
    df = pd.DataFrame({"x": [1.0]})
    seed = ProjectCache(str(tmp_path))
    used_fp = {"path": "used.csv"}
    other_fp = {"path": "other.csv"}
    for key, fp in (("used", used_fp), ("other", other_fp)):
        seed.put_dataframe(key, df)
        seed.put_summary(fp, key)
        seed.put_named(key, "sig", df)
        seed.put_named_reference(f"{key}-ref", "sig", key)
        seed.put_materialized_manifest(key, {"parts": ["part.parquet"]})
        (seed.materialized_slot(key) / "part.parquet").write_bytes(b"parquet payload")
    # This alias will not be read, but its payload is about to be deleted.
    seed.put_named_reference("unread-alias", "sig", "used")

    cache = ProjectCache(str(tmp_path))
    assert cache.get_named("used-ref", "sig") is not None
    assert cache.get_named("used", "sig") is not None
    assert cache.get_summary(used_fp) == "used"
    assert cache.get_materialized_manifest("used") is not None
    cache.put_dataframe("new", df)
    stats = cache.clear_used()

    assert stats["failed"] == 0
    assert not (cache.data_dir / "used.pkl").exists()
    assert not (cache.data_dir / "used.json").exists()
    assert not (cache.data_dir / "new.pkl").exists()
    assert not (cache.data_dir / "new.json").exists()
    assert not (cache.materialized_dir / "used").exists()
    assert len(list(cache.summary_dir.iterdir())) == 1
    assert len(list(cache.named_dir.iterdir())) == 1
    fresh = ProjectCache(str(tmp_path))
    assert fresh.get_dataframe("other") is not None
    assert fresh.get_summary(other_fp) == "other"
    assert fresh.get_materialized_manifest("other") is not None
    assert set(fresh.manifest["named"]) == {"other", "other-ref"}
    assert fresh.get_named("other", "sig") is not None
    assert fresh.get_named("other-ref", "sig") is not None


def test_cleanup_reports_delete_failure_and_keeps_valid_reference(tmp_path, monkeypatch):
    cache = ProjectCache(str(tmp_path))
    cache.put_dataframe("locked", pd.DataFrame({"x": [1]}))
    cache.put_named_reference("ref", "sig", "locked")
    original_unlink = Path.unlink

    def locked_unlink(path, *args, **kwargs):
        if path.name == "locked.pkl":
            raise PermissionError("locked")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked_unlink)
    assert cache.clear_used() == {"removed": 1, "failed": 1}
    assert (cache.data_dir / "locked.pkl").is_file()
    assert "ref" in json.loads(cache.manifest_path.read_text())["named"]


def test_cleanup_never_deletes_external_named_payload(tmp_path):
    external = tmp_path / "external.pkl"
    pd.DataFrame({"x": [1]}).to_pickle(external)
    cache = ProjectCache(str(tmp_path))
    cache.manifest["named"]["external"] = {"signature": "sig", "path": str(external)}
    assert cache.get_named("external", "sig") is not None
    cache.clear_used()
    assert external.is_file()
