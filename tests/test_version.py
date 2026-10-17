"""CLI and generated images report one distribution provenance version."""
from importlib.metadata import PackageNotFoundError

import json
import sys

import pytest

from jarvisplot import version


@pytest.mark.parametrize("tokens", [["-v"], ["--version"], ["-v", "--json"], ["--json", "--version"]])
def test_cli_version_without_render_runtime(tokens, monkeypatch, capsys):
    from jarvisplot.client import main

    monkeypatch.setitem(sys.modules, "jarvisplot.core", None)
    assert main(tokens) == 0
    output = capsys.readouterr().out
    if "--json" in tokens:
        assert json.loads(output) == {
            "ok": True, "command": "version", "name": "Jarvis-PLOT",
            "version": version.get_version(),
        }
    else:
        assert output.strip() == f"Jarvis-PLOT {version.get_version()}"


def test_official_installed_distribution_wins_over_source_and_legacy(monkeypatch):
    seen = []
    def metadata(name):
        seen.append(name)
        return '2.1.9' if name == 'JarvisPLOT' else '1.0.0'
    monkeypatch.setattr(version, 'distribution_version', metadata)
    from jarvisplot.cli import _resolve_version
    from jarvisplot.Figure.figure import _resolve_jplot_version, _savefig_metadata
    assert version.get_version() == _resolve_version() == _resolve_jplot_version() == '2.1.9'
    assert _savefig_metadata('png')['Jarvis-PLOT version'] == '2.1.9'
    assert set(seen) == {'JarvisPLOT'}


def test_legacy_distribution_is_a_bounded_fallback(monkeypatch):
    def metadata(name):
        if name == 'JarvisPLOT':
            raise PackageNotFoundError(name)
        return '1.2.6'
    monkeypatch.setattr(version, 'distribution_version', metadata)
    assert version.get_version() == '1.2.6'


def test_uninstalled_checkout_uses_version_file(monkeypatch, tmp_path):
    def absent(name):
        raise PackageNotFoundError(name)
    monkeypatch.setattr(version, 'distribution_version', absent)
    monkeypatch.setattr(version, '__file__', str(tmp_path/'jarvisplot'/'version.py'))
    (tmp_path/'pyproject.toml').touch()
    (tmp_path/'VERSION').write_text('2.1.9\n')
    assert version.get_version() == '2.1.9'
    (tmp_path/'VERSION').unlink()
    assert version.get_version() == '0.0.0'


def test_unrelated_parent_version_file_is_not_used(monkeypatch, tmp_path):
    def absent(name):
        raise PackageNotFoundError(name)
    monkeypatch.setattr(version, 'distribution_version', absent)
    monkeypatch.setattr(version, '__file__', str(tmp_path/'jarvisplot'/'version.py'))
    (tmp_path/'VERSION').write_text('99.0.0\n')
    assert version.get_version() == '0.0.0'
