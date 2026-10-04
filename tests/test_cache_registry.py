from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from rich.cells import cell_len
from rich.text import Text

from jarvisplot.cache_registry import CACHE_SCHEMA, MARKER, PIPELINE_COMPONENTS, CacheRegistry
from jarvisplot.cache_store import ProjectCache
from jarvisplot.client import main


def _cache(workdir: Path, *, config_path=None):
    with ProjectCache(str(workdir), config_path=config_path) as cache:
        cache.put_dataframe("payload", pd.DataFrame({"x": [1.0]}))
    return workdir / ".cache"


def _command(capsys, *args, expected=0):
    assert main(["cache", *args, "--json"]) == expected
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    result = json.loads(captured.out)
    assert set(result) == {"api_version", "kind", "ok", "data", "diagnostics", "error"}
    return result


def test_register_merges_configs_and_lists_current_size(tmp_path, capsys):
    root = _cache(tmp_path / "project", config_path=str(tmp_path / "one.yaml"))
    _cache(root.parent, config_path=str(tmp_path / "two.yaml"))
    registry = CacheRegistry()
    before = _command(capsys, "list")["data"]["caches"]
    assert len(before) == 1
    assert before[0]["status"] == "ready"
    assert before[0]["configs"] == [str(tmp_path / "one.yaml"), str(tmp_path / "two.yaml")]
    assert Path(json.loads(registry.path.read_text())["caches"][str(root)]["workdir"]) == root.parent
    (root / "data" / "extra.pkl").write_bytes(b"x" * 128)
    after = _command(capsys, "list")["data"]["caches"]
    assert after[0]["bytes"] == before[0]["bytes"] + 128


def test_clean_preview_then_one_workdir_preserves_other_files(tmp_path, capsys):
    root = _cache(tmp_path / "one")
    other = _cache(tmp_path / "two")
    (root / "foreign").mkdir()
    foreign = root / "foreign" / "keep.txt"
    foreign.write_text("another tool")
    source = root.parent / "samples.csv"
    source.write_text("x\n1\n")
    plot = root.parent / "plots" / "figure.png"
    plot.parent.mkdir()
    plot.write_bytes(b"final image")
    before = CacheRegistry().path.read_bytes()
    preview = _command(capsys, "clean", "--workdir", str(root.parent), "--dry-run")["data"]
    assert preview["removed"] == [str(root)]
    assert (root / "data" / "payload.pkl").exists()
    assert CacheRegistry().path.read_bytes() == before
    result = _command(capsys, "clean", "--workdir", str(root.parent))["data"]
    assert result["removed"] == [str(root)]
    assert not (root / "data").exists()
    assert not (root / MARKER).exists()
    assert foreign.read_text() == "another tool"
    assert source.exists() and plot.exists()
    assert (other / "data" / "payload.pkl").exists()
    assert set(json.loads(CacheRegistry().path.read_text())["caches"]) == {str(other)}


@pytest.mark.parametrize("include_parent_cache", [False, True])
def test_clean_workdir_selects_entire_directory_tree(tmp_path, capsys, include_parent_cache):
    parent = tmp_path / "projects"
    roots = [_cache(parent / "one"), _cache(parent / "group" / "two")]
    if include_parent_cache:
        roots.append(_cache(parent))
    outside = _cache(tmp_path / "projects-backup" / "one")
    source = parent / "group" / "two" / "source.csv"
    source.write_text("x\n1\n")
    foreign = parent / "foreign" / ".cache" / "keep"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("another tool")
    before = CacheRegistry().path.read_bytes()

    preview = _command(capsys, "clean", "--workdir", str(parent), "--dry-run")["data"]
    assert set(preview["removed"]) == {str(root) for root in roots}
    assert CacheRegistry().path.read_bytes() == before
    assert all((root / "data" / "payload.pkl").exists() for root in roots)

    # Repeated and overlapping selectors must clean each cache only once.
    result = _command(capsys, "clean", "--workdir", str(parent),
                      "--workdir", str(parent / "group"), "--workdir", str(parent))["data"]
    assert set(result["removed"]) == set(preview["removed"])
    assert len(result["removed"]) == len(roots)
    assert result["bytes"] == preview["bytes"]
    assert all(not root.exists() for root in roots)
    assert (outside / "data" / "payload.pkl").exists()
    assert source.exists() and foreign.read_text() == "another tool"
    assert set(json.loads(CacheRegistry().path.read_text())["caches"]) == {str(outside)}


def test_clean_nested_workdirs_retains_age_and_active_guards(tmp_path, capsys):
    parent = tmp_path / "projects"
    old = _cache(parent / "nested" / "old")
    recent = _cache(parent / "recent")
    registry = CacheRegistry()
    stamp = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    with registry.lease(old.parent):
        registry.register(old.parent, last_used=stamp)
    with ProjectCache(str(parent / "nested" / "active")) as active:
        result = _command(capsys, "clean", "--workdir", str(parent), "--older-than", "30")
        assert result["ok"] is None
        assert result["data"]["removed"] == [str(old)]
        assert result["data"]["skipped"] == [{"path": str(active.root), "reason": "active"}]
        assert active.root.exists() and recent.exists()


def test_age_filter_and_missing_records(tmp_path, capsys):
    old = _cache(tmp_path / "old")
    recent = _cache(tmp_path / "recent")
    missing = _cache(tmp_path / "missing")
    registry = CacheRegistry()
    stamp = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat()
    with registry.lease(old.parent):
        registry.register(old.parent, last_used=stamp)
    shutil.rmtree(missing)
    preview = _command(capsys, "clean", "--all", "--older-than", "30", "--dry-run")["data"]
    assert preview["removed"] == [str(old)]
    assert preview["missing"] == [str(missing)]
    assert str(missing) in json.loads(registry.path.read_text())["caches"]
    result = _command(capsys, "clean", "--all", "--older-than", "30")["data"]
    assert result["removed"] == [str(old)]
    assert not old.exists()
    assert recent.exists()
    assert str(missing) not in json.loads(registry.path.read_text())["caches"]


def test_active_cache_is_skipped_until_closed(tmp_path, capsys):
    cache = ProjectCache(str(tmp_path / "active"))
    cache.put_dataframe("payload", pd.DataFrame({"x": [1]}))
    assert _command(capsys, "list")["data"]["caches"][0]["status"] == "active"
    result = _command(capsys, "clean", "--all")
    assert result["ok"] is None
    assert result["data"]["skipped"] == [{"path": str(cache.root), "reason": "active"}]
    assert (cache.root / "data" / "payload.pkl").exists()
    cache.close()
    assert _command(capsys, "clean", "--all")["data"]["removed"] == [str(cache.root)]
    assert not cache.root.exists()


def test_crashed_process_releases_active_lock(tmp_path, capsys):
    workdir = tmp_path / "crashed"
    worker = subprocess.Popen(
        [sys.executable, "-c",
         "import sys; from jarvisplot.cache_store import ProjectCache; "
         "cache = ProjectCache(sys.argv[1]); print('ready', flush=True); sys.stdin.read()",
         str(workdir)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert worker.stdout.readline().strip() == "ready"
        assert _command(capsys, "clean", "--all")["data"]["skipped"]
        worker.kill()
        worker.communicate(timeout=15)
        assert _command(capsys, "clean", "--all")["data"]["removed"] == [str(workdir / ".cache")]
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.communicate(timeout=15)


def test_scan_adopts_legacy_layout_and_retains_age(tmp_path, capsys):
    root = tmp_path / "legacy" / ".cache"
    for component in PIPELINE_COMPONENTS:
        (root / component).mkdir(parents=True)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema": 1, "files": {}, "named": {}}))
    payload = root / "data" / "old.pkl"
    payload.write_bytes(b"legacy data")
    old = (datetime.now(timezone.utc) - timedelta(days=45)).timestamp()
    for path in (manifest, payload):
        os.utime(path, (old, old))
    foreign = tmp_path / "foreign" / ".cache"
    foreign.mkdir(parents=True)
    (foreign / "keep").write_text("other cache")
    result = _command(capsys, "scan", str(tmp_path))["data"]
    assert [entry["cache_dir"] for entry in result["registered"]] == [str(root)]
    assert datetime.fromisoformat(result["registered"][0]["last_used"]).timestamp() == pytest.approx(old)
    assert not (foreign / MARKER).exists()
    assert _command(capsys, "clean", "--all", "--older-than", "30")["data"]["removed"] == [str(root)]
    assert (foreign / "keep").exists()


def test_invalid_index_path_and_symlink_are_not_deleted(tmp_path, capsys):
    root = _cache(tmp_path / "plot")
    registry = CacheRegistry()
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep").write_text("keep")
    shutil.rmtree(root)
    root.symlink_to(external, target_is_directory=True)
    result = _command(capsys, "clean", "--all", expected=1)
    assert result["data"]["errors"]
    assert (external / "keep").exists()
    assert root.is_symlink()
    data = json.loads(registry.path.read_text())
    data["caches"][str(external)] = {"workdir": str(external.parent), "cache_dir": str(external)}
    registry.path.write_text(json.dumps(data))
    assert len(_command(capsys, "clean", "--all", expected=1)["data"]["errors"]) == 2
    assert (external / "keep").exists()


def test_symlink_component_only_removes_link(tmp_path, capsys):
    root = _cache(tmp_path / "plot")
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep").write_text("keep")
    shutil.rmtree(root / "materialized")
    (root / "materialized").symlink_to(external, target_is_directory=True)
    _command(capsys, "clean", "--all")
    assert (external / "keep").exists()


def test_failed_delete_retains_record_and_other_targets_continue(tmp_path, monkeypatch, capsys):
    locked = _cache(tmp_path / "locked")
    good = _cache(tmp_path / "good")
    original = shutil.rmtree

    def fail(path, *args, **kwargs):
        if Path(path) == locked / "data":
            raise PermissionError("permission denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", fail)
    result = _command(capsys, "clean", "--all", expected=1)["data"]
    assert result["removed"] == [str(good)]
    assert result["errors"]
    assert set(json.loads(CacheRegistry().path.read_text())["caches"]) == {str(locked)}
    assert (locked / MARKER).exists()


def test_concurrent_registrations_do_not_overwrite_entries(tmp_path):
    def register(number):
        workdir = tmp_path / f"project-{number}"
        registry = CacheRegistry()
        with registry.lease(workdir):
            registry.register(workdir)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(register, range(16)))
    rows = CacheRegistry().list()
    assert len(rows) == 16
    assert all(row["status"] == "ready" for row in rows)


def test_corrupt_registry_reports_failure_without_overwriting(tmp_path, capsys):
    root = _cache(tmp_path / "plot")
    registry = CacheRegistry()
    registry.path.write_text("invalid json")
    result = _command(capsys, "clean", "--all", expected=1)
    assert result["error"]
    assert registry.path.read_text() == "invalid json"
    assert (root / "data" / "payload.pkl").exists()


@pytest.mark.parametrize("arguments", [[], ["clean"], ["clean", "--all", "--workdir", "."],
                                      ["clean", "--all", "--older-than", "-1"],
                                      ["clean", "--all", "--older-than", "nan"]])
def test_usage_errors_use_rich_cards(arguments, capsys):
    assert main(["cache", *arguments]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "╭─" in output.err
    assert "Usage:" in output.err


def test_unregistered_target_is_usage_error(tmp_path, capsys):
    root = _cache(tmp_path / "registered")
    before = CacheRegistry().path.read_bytes()
    result = _command(capsys, "clean", "--workdir", str(root.parent),
                      "--workdir", str(tmp_path / "unknown"), expected=2)
    assert result["error"]["type"] == "UsageError"
    assert "No registered plot caches under:" in result["error"]["message"]
    assert (root / "data" / "payload.pkl").exists()
    assert CacheRegistry().path.read_bytes() == before


@pytest.mark.parametrize("arguments", [[], ["list"], ["scan"], ["clean"]])
def test_help_uses_existing_geometry_and_forwarded_prog(arguments, capsys):
    assert main(["cache", *arguments, "-h"], prog="Jarvis2 plot") == 0
    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.startswith(("╭", "│", "╰"))]
    assert {cell_len(line) for line in lines} == {80}
    assert "Jarvis2 plot cache" in output


def test_human_output_stays_on_stderr(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(["cache", "list"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "No registered caches" in captured.err


@pytest.mark.parametrize("width", [80, 96])
@pytest.mark.parametrize("arguments,section", [(["list"], "Caches"), (["scan"], "Registered caches"),
                                              (["clean", "--all", "--dry-run"], "Selected caches"),
                                              (["clean", "--all"], "Cleaned caches")])
def test_human_results_use_existing_panels_and_grid(tmp_path, monkeypatch, capsys, width, arguments, section):
    import jarvisplot.cli_help as cli_help

    root = _cache(tmp_path / "project[one]")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(cli_help.shutil, "get_terminal_size", lambda: SimpleNamespace(columns=width))
    tokens = [*arguments, str(tmp_path)] if arguments[0] == "scan" else arguments
    assert main(["cache", *tokens]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    # Strip colors for geometry checks; retain the original for TTY styling.
    output = Text.from_ansi(captured.err).plain
    lines = output.splitlines()
    assert lines[0].startswith(f"╭─ cache {arguments[0]}")
    assert f"╭─ {section}" in output
    assert all(line.startswith(("╭", "│", "╰")) for line in lines)
    assert {cell_len(line) for line in lines} == {width}
    assert "\x1b[" in captured.err
    # Long paths fold across description lines, preserving literal brackets.
    description_start = 2 if arguments[0] == "list" else 36
    descriptions = "".join(line[description_start:-2].strip() for line in lines if line.startswith("│"))
    assert str(root) in descriptions
    if arguments[0] == "list":
        assert "1. " + str(root) in descriptions
        assert "ready · " in output
        assert "Last used: " in output and " UTC" in output
    elif "--dry-run" in arguments:
        assert (root / "data" / "payload.pkl").exists()


def test_human_skip_missing_and_errors_stay_inside_panels(tmp_path, monkeypatch, capsys):
    missing = _cache(tmp_path / "missing")
    shutil.rmtree(missing)
    invalid = _cache(tmp_path / "invalid")
    (invalid / MARKER).unlink()
    cache = ProjectCache(str(tmp_path / "active"))
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    try:
        assert main(["cache", "clean", "--all"]) == 1
        captured = capsys.readouterr()
        output = Text.from_ansi(captured.err).plain
        assert "╭─ Missing records" in output
        assert "╭─ Skipped caches" in output
        assert "╭─ Errors" in output
        assert all(line.startswith(("╭", "│", "╰")) for line in output.splitlines())
    finally:
        cache.close()


def test_human_semantic_error_uses_panels(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(["cache", "clean", "--workdir", str(tmp_path)]) == 2
    captured = capsys.readouterr()
    output = Text.from_ansi(captured.err).plain
    assert captured.out == ""
    assert "╭─ cache clean" in output and "╭─ Error" in output
    assert "UsageError" in output


def test_empty_human_list_uses_forwarded_command_in_next_panel(monkeypatch, capsys):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert main(["cache", "list"], prog="Jarvis2 plot") == 0
    output = Text.from_ansi(capsys.readouterr().err).plain
    assert "╭─ Next" in output
    assert "Use Jarvis2 plot cache scan" in output


def test_cache_cli_never_imports_plot_or_data_stack():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; from jarvisplot.client import main; assert main(['cache', 'list', '--json']) == 0; "
         "assert not any(name in sys.modules for name in ('matplotlib', 'pandas', 'polars', 'h5py', 'scipy'))"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["kind"] == "cache.list"


def test_data_describe_registers_cache_and_releases_lease(tmp_path, capsys):
    source = tmp_path / "samples.csv"
    source.write_text("x,y\n1,2\n")
    assert main(["data", "describe", str(source), "--json"]) == 0
    capsys.readouterr()
    row = _command(capsys, "list")["data"]["caches"][0]
    assert row["cache_dir"] == str(tmp_path / ".cache")
    assert row["status"] == "ready"
    assert _command(capsys, "clean", "--all")["data"]["removed"] == [row["cache_dir"]]
    assert source.exists()


def test_default_twins_register_and_custom_exports_stay_unmanaged(tmp_path, capsys):
    from jarvisplot.dryrun_runtime import dryrun_file
    import yaml

    source = tmp_path / "samples.csv"
    source.write_text("x,y\n1,2\n3,4\n")
    config = tmp_path / "plot.yaml"
    config.write_text(yaml.safe_dump({
        "DataSet": [{"name": "samples", "type": "csv", "path": str(source)}],
        "Figures": [{"name": "points", "layers": [{"name": "data", "method": "scatter", "axes": "ax",
            "data": [{"source": "samples"}], "coordinates": {"x": {"expr": "x"}, "y": {"expr": "y"}}}]}],
    }))
    report, bag = dryrun_file(str(config), with_data=True)
    assert bag.ok
    assert report["twins"]
    row = _command(capsys, "list")["data"]["caches"][0]
    assert row["status"] == "ready"
    assert row["configs"] == [str(config)]
    marker = json.loads((tmp_path / ".cache" / MARKER).read_text())
    assert marker["components"] == ["agent_twins"]
    _command(capsys, "clean", "--all")
    assert not (tmp_path / ".cache").exists()
    report, bag = dryrun_file(str(config), with_data=True, out_dir=str(tmp_path / "export"))
    assert bag.ok and report["twins"]
    assert _command(capsys, "list")["data"]["caches"] == []
    assert list((tmp_path / "export").iterdir())


def test_registry_unavailable_does_not_prevent_cache_use(tmp_path, monkeypatch):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    monkeypatch.setenv("JARVIS_HOME", str(blocked))
    with ProjectCache(str(tmp_path / "plot")) as cache:
        cache.put_dataframe("payload", pd.DataFrame({"x": [1]}))
        assert cache.get_dataframe("payload") is not None


@pytest.mark.parametrize("field,value", [("last_used", None), ("configs", [1])])
def test_malformed_marker_does_not_block_use_but_prevents_deletion(tmp_path, capsys, field, value):
    root = _cache(tmp_path / "plot")
    marker = json.loads((root / MARKER).read_text())
    marker[field] = value
    (root / MARKER).write_text(json.dumps(marker))
    with ProjectCache(str(root.parent)) as cache:
        assert cache.get_dataframe("payload") is not None
    result = _command(capsys, "clean", "--all", expected=1)
    assert result["data"]["errors"]
    assert (root / "data" / "payload.pkl").exists()
