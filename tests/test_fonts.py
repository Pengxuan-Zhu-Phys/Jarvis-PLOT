"""Durable font-card and strict discovery contracts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from jarvisplot.fonts import (
    FontCardError, check_fonts, font_card_path, font_card_payload, font_inventory,
    load_font_card, reset_font_card, resolve_font, resolve_role, update_font_card,
)


def test_recommendations_are_read_only_and_do_not_import_render_stack(tmp_path):
    home = tmp_path / 'fresh'
    result = subprocess.run([sys.executable, '-c',
        'import json,sys; from jarvisplot.fonts import font_card_payload; '
        'print(json.dumps(font_card_payload())); '
        'assert "matplotlib" not in sys.modules; assert "jarvisplot.core" not in sys.modules'],
        env={**os.environ, 'JARVIS_HOME': str(home)}, capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout)
    assert payload['roles']['title']['family'] == 'Helvetica'
    assert payload['roles']['axis_label']['family'] == 'STIXGeneral'
    assert payload['roles']['legend']['math_fontfamily'] == 'stix'
    assert payload['roles']['title']['size_pt'] is None
    assert not home.exists()


def test_overrides_persist_across_processes_without_changing_recommendations():
    update_font_card('families.sans', 'DejaVu Sans')
    update_font_card('roles.title.size_pt', 11)
    update_font_card('roles.legend.family', 'serif')
    assert load_font_card(recommended=True)['families']['sans'] == 'Helvetica'
    assert resolve_role('title')['family'] == 'DejaVu Sans'
    assert resolve_role('legend')['family'] == 'STIXGeneral'
    result = subprocess.run([sys.executable, '-c',
        'import json; from jarvisplot.fonts import font_card_payload; print(json.dumps(font_card_payload()))'],
        capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['roles']['title']['size_pt'] == 11
    saved = json.loads(font_card_path().read_text())
    assert saved['overrides'] == {'families': {'sans': 'DejaVu Sans'},
                                  'roles': {'title': {'size_pt': 11}, 'legend': {'family': 'serif'}}}
    assert font_card_payload()['has_overrides']


@pytest.mark.parametrize(('path', 'value'), [
    ('roles.typo.family', 'Helvetica'), ('roles.title.recommended_size_pt', 7),
    ('schema_version', 2), ('roles.title.size_pt', 0), ('roles.title.size_pt', True),
    ('roles.title.size_pt', float('nan')), ('roles.title.size_pt', float('inf')),
    ('roles.legend.weight', 'bogus'), ('roles.text.weight', 1001), ('roles.text.style', 'tilted'),
    ('families.sans', []), ('families.serif', ''), ('families.serif', ' STIX'),
    ('math.fontset', 'external-tex'),
])
def test_invalid_updates_never_change_existing_defaults(path, value):
    update_font_card('roles.title.size_pt', 10)
    before = font_card_path().read_bytes()
    with pytest.raises(FontCardError):
        update_font_card(path, value)
    assert font_card_path().read_bytes() == before


def test_reset_one_field_then_everything():
    update_font_card('families.sans', 'DejaVu Sans')
    update_font_card('roles.title.size_pt', 12)
    reset_font_card('roles.title.size_pt')
    assert resolve_role('title')['size_pt'] is None
    assert resolve_role('title')['family'] == 'DejaVu Sans'
    reset_font_card()
    assert not font_card_path().exists()
    assert resolve_role('title')['family'] == 'Helvetica'


def test_full_reset_recovers_corrupt_file():
    path = font_card_path()
    path.parent.mkdir(parents=True)
    path.write_text('{ broken')
    with pytest.raises(FontCardError):
        load_font_card()
    assert reset_font_card()['roles']['title']['family'] == 'Helvetica'


def test_unknown_persisted_fields_fail_explicitly():
    path = font_card_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'schema_version': 1, 'overrides': {'roles': {'typo': {}}}}))
    with pytest.raises(FontCardError, match='Unknown'):
        load_font_card()


def test_inventory_and_resolution_report_real_files_and_no_fallback():
    payload = font_inventory(query='STIXGeneral')
    assert payload['count'] > 0
    assert all(row['family'] == 'STIXGeneral' and Path(row['path']).is_file() for row in payload['fonts'])
    resolved = resolve_font('STIX', weight='bold', style='italic')
    assert resolved['family'] == 'STIXGeneral'
    assert 'Bold' in resolved['face'] and 'Italic' in resolved['face']
    assert Path(resolved['path']).is_file()
    with pytest.raises(ValueError, match='unavailable'):
        resolve_font('Jarvis intentionally missing test font')


def test_check_reports_missing_roles_and_passes_portable_fonts():
    update_font_card('families.sans', 'Jarvis intentionally missing test font')
    assert check_fonts()['ok'] is False
    assert any('title' in row['roles'] and not row['available'] for row in check_fonts()['results'])
    update_font_card('families.sans', 'DejaVu Sans')
    assert check_fonts()['ok'] is True


def test_missing_helvetica_uses_first_installed_similar_font(monkeypatch):
    from matplotlib import font_manager
    real_findfont = font_manager.findfont
    attempted = []

    def missing_helvetica(prop, **kwargs):
        family = prop.get_family()[0]
        attempted.append(family)
        if family in ('Helvetica', 'Arial', 'Liberation Sans', 'Nimbus Sans'):
            raise ValueError('simulated missing system font')
        return real_findfont(prop, **kwargs)
    monkeypatch.setattr(font_manager, 'findfont', missing_helvetica)
    result = resolve_font('Helvetica')
    assert result['requested_family'] == 'Helvetica'
    assert result['family'] == 'DejaVu Sans'
    assert result['fallback'] is True
    assert attempted == ['Helvetica', 'Arial', 'Liberation Sans', 'Nimbus Sans', 'DejaVu Sans']
    assert check_fonts()['ok'] is True


def test_concurrent_writers_keep_both_disjoint_edits():
    scripts = [('roles.title.size_pt', '11'), ('roles.legend.size_pt', '7')]
    processes = [subprocess.Popen([sys.executable, '-c',
        f'from jarvisplot.fonts import update_font_card; update_font_card({path!r}, {value})'])
        for path, value in scripts]
    assert all(process.wait(timeout=30) == 0 for process in processes)
    assert resolve_role('title')['size_pt'] == 11
    assert resolve_role('legend')['size_pt'] == 7
