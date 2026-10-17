"""Version/resource drift is a release failure, not a successful local smoke."""
from pathlib import Path
import zipfile

import pytest

from tools.check_release import LIVE_DOCS, check_source, check_wheel, main


def source(tmp_path):
    (tmp_path/'pyproject.toml').write_text('[project]\nname = "JarvisPLOT"\nversion = "2.1.9"\n')
    (tmp_path/'VERSION').write_text('2.1.9\n')
    for relative, marker in LIVE_DOCS.items():
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker+' **2.1.9**\n')
    (tmp_path/'CHANGELOG.md').write_text('# Changelog\n\n## Unreleased\n\n## 2.1.9 \u2014 2026-10-13\n\n## 2.1.8\n')
    module = tmp_path/'jarvisplot'/'version.py'
    module.parent.mkdir()
    module.write_text('VERSION = "2.1.9"\n')
    resource = tmp_path/'jarvisplot'/'manual_cards'/'agent-readme.yaml'
    resource.parent.mkdir()
    resource.write_text('id: agent-readme\n')
    return tmp_path


def wheel(root, *, omit=None, stale=None):
    target = root/'jarvisplot-2.1.9-py3-none-any.whl'
    with zipfile.ZipFile(target, 'w') as archive:
        archive.writestr('jarvisplot-2.1.9.dist-info/METADATA', 'Name: JarvisPLOT\nVersion: 2.1.9\n')
        archive.writestr('jarvisplot-2.1.9.dist-info/entry_points.txt', '[console_scripts]\njplot = jarvisplot.client:main\n')
        for path in (root/'jarvisplot').rglob('*'):
            if path.is_file():
                name = path.relative_to(root).as_posix()
                if name != omit:
                    archive.writestr(name, b'stale' if name == stale else path.read_bytes())
    return target


def test_live_source_and_changelog_agree():
    assert check_source(Path(__file__).resolve().parents[1])


def test_source_gate_rejects_version_and_live_doc_drift(tmp_path):
    root = source(tmp_path)
    assert check_source(root) == '2.1.9'
    (root/'VERSION').write_text('2.1.8\n')
    with pytest.raises(ValueError, match='disagree'):
        check_source(root)
    (root/'VERSION').write_text('2.1.9\n')
    # Historical text containing 2.1.9 must not mask a stale live-version line.
    (root/'README.md').write_text('Current source candidate: **2.1.8**\nOld note: 2.1.9\n')
    with pytest.raises(ValueError, match='live-version'):
        check_source(root)


def test_source_gate_requires_current_changelog_entry(tmp_path):
    root = source(tmp_path)
    # An older entry below the latest heading must not satisfy the gate.
    (root/'CHANGELOG.md').write_text('## Unreleased\n\n## 2.1.8 \u2014 2026-10-01\n\n## 2.1.9\n')
    with pytest.raises(ValueError, match='CHANGELOG'):
        check_source(root)
    (root/'CHANGELOG.md').unlink()
    with pytest.raises(OSError):
        check_source(root)


@pytest.mark.parametrize('damage', ['omit', 'stale'])
def test_wheel_gate_rejects_missing_or_stale_packaged_manual(tmp_path, damage):
    root = source(tmp_path)
    name = 'jarvisplot/manual_cards/agent-readme.yaml'
    with pytest.raises(ValueError, match='missing|stale'):
        check_wheel(root, wheel(root, **{damage: name}), '2.1.9')


def test_wheel_and_tag_gates_are_read_only(tmp_path, capsys):
    root = source(tmp_path)
    artifact = wheel(root)
    assert check_wheel(root, artifact, '2.1.9')['modules_and_resources'] == 2
    assert main(['--root', str(root), '--wheel', str(artifact), '--tag', 'v2.1.9']) == 0
    assert '"publishes": false' in capsys.readouterr().out
    with pytest.raises(SystemExit) as failure:
        main(['--root', str(root), '--tag', 'v2.1.8'])
    assert failure.value.code == 1
