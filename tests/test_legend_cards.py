from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import pytest

from jarvisplot.capabilities import section
from jarvisplot.client import main
from jarvisplot.core_assets import load_styles
from jarvisplot.Figure.figure import Figure
from jarvisplot.Figure.legend_runtime import apply_legend
from jarvisplot.legend_cards import (
    builtin_card_names, legend_card_catalog, load_legend_card, merge_axes_legend,
    resolve_legend_config, assemble_legend_entries,
)
from jarvisplot.schema_catalog import config_validator
from jarvisplot.validation import validate_config


def test_bundled_cards_are_resolved_and_independent():
    assert builtin_card_names() == ['compact', 'default', 'paper', 'presentation']
    for card in legend_card_catalog():
        assert set(card['items']) == {
            'line', 'marker', 'band', 'errorbar', 'band_line', 'line_marker',
            'line_errorbar', 'error_bounds',
        }
        assert 'extends' not in card
    compact = load_legend_card('compact')
    assert load_legend_card('paper')['legend']['handleheight'] > compact['legend']['handleheight']
    compact['items']['line_errorbar']['roles'].clear()
    assert load_legend_card('compact')['items']['line_errorbar']['roles'] == ['errorbar', 'line']


def test_custom_json_inheritance_and_yaml_precedence(tmp_path):
    card = tmp_path / 'custom.json'
    card.write_text(json.dumps({
        'schema_version': 1, 'extends': 'paper',
        'legend': {'handlelength': 2.5, 'prop': {'family': 'STIXGeneral', 'size': 6}},
        'items': {'error_bounds': {'roles': ['lower', 'line', 'upper']}},
    }))
    config = {
        'card': './custom.json', 'handlelength': 3, 'prop': {'size': 8},
        'item_formats': {'error_bounds': {'key': 'overlay'}},
    }
    original = deepcopy(config)
    kwargs, items = resolve_legend_config(config, base_dir=tmp_path)
    assert kwargs['handlelength'] == 3
    assert kwargs['prop'] == {'family': 'STIXGeneral', 'size': 8}
    assert items['error_bounds']['roles'] == ['lower', 'line', 'upper']
    assert items['error_bounds']['key'] == 'overlay'
    assert config == original
    entries = assemble_legend_entries([
        {'legend': {'object': 'test', 'label': 'Test', 'role': role}}
        for role in ('line', 'upper', 'lower')
    ], items)
    assert entries[0]['layers'] == ['@layer:2', '@layer:0', '@layer:1']
    assert entries[0]['key'] == 'overlay'


def test_relative_parent_card_and_invalid_cycle(tmp_path):
    (tmp_path / 'parent.json').write_text(json.dumps({'schema_version': 1, 'extends': 'compact'}))
    child = tmp_path / 'child.json'
    child.write_text(json.dumps({'schema_version': 1, 'extends': './parent.json'}))
    assert load_legend_card(str(child))['legend']['labelspacing'] == .25
    (tmp_path / 'parent.json').write_text(json.dumps({'schema_version': 1, 'extends': './child.json'}))
    with pytest.raises(ValueError, match='cycle'):
        load_legend_card(str(child))


@pytest.mark.parametrize('payload', [
    {'schema_version': 1, 'legend': {'entries': []}},
    {'schema_version': 1, 'items': {'broken': {'key': 'overlay'}}},
    {'schema_version': 1, 'items': {'broken': {'roles': ['line'], 'key': 'wrong'}}},
    {'schema_version': 1, 'typo': {}},
])
def test_invalid_card_is_rejected_before_rendering(tmp_path, payload):
    card = tmp_path / 'bad.json'
    card.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='Invalid legend'):
        load_legend_card(str(card))


def test_line_errorbar_template_inherits_styles_and_keeps_native_error_caps():
    fig, ax = plt.subplots()
    error = ax.errorbar([0, 1], [.5, .5], yerr=.1, fmt='none',
                        ecolor='orange', alpha=.4, capsize=3)
    mean = ax.plot([0, 1], [.5, .5], color='blue', linestyle='--')
    counts = (len(ax.lines), len(ax.collections), len(ax.patches))
    limits = (ax.get_xlim(), ax.get_ylim())
    legend = apply_legend(ax, {'card': 'paper'},
                          {'@layer:0': mean, '@layer:1': error}, layer_specs=[
        {'legend': {'object': 'estimate', 'label': 'Estimate', 'role': role}}
        for role in ('line', 'errorbar')
    ])
    fig.canvas.draw()
    assert legend.get_texts()[0].get_text() == 'Estimate'
    assert legend.handleheight == 1
    assert any(line.get_color() == 'blue' and line.get_linestyle() == '--'
               for line in legend.findobj(Line2D))
    assert any(collection.get_colors()[0][3] == pytest.approx(.4)
               for collection in legend.findobj(LineCollection))
    assert len([line for line in legend.findobj(Line2D) if line.get_marker() == '_']) == 2
    assert (len(ax.lines), len(ax.collections), len(ax.patches)) == counts
    assert (ax.get_xlim(), ax.get_ylim()) == limits


def test_axes_declaration_and_yaml_order_replace_card_defaults():
    fig = Figure()
    fig.logger = SimpleNamespace(**{key: lambda *a, **k: None for key in ('debug', 'info', 'warning', 'error')})
    fig.jpstyles = load_styles(fig.load_path)
    fig.style = ['a4paper_2x1', 'rectRatio']
    assert fig.frame['ax']['legend'] == {'enabled': False, 'card': 'compact'}
    fig.frame['ax']['legend']['order'] = ['old']
    override = {'ax': {'legend': {'order': ['new'], 'handleheight': 1.2}}}
    original = deepcopy(override)
    fig.frame = override
    assert fig.frame['ax']['legend']['order'] == ['new']
    assert fig.frame['ax']['legend']['card'] == 'compact'
    assert fig.frame['ax']['legend']['enabled'] is True
    assert fig.frame['axr']['legend']['enabled'] is False
    assert override == original
    assert merge_axes_legend({'card': 'paper', 'enabled': False}, {'card': 'compact'}) == {
        'enabled': True, 'card': 'compact',
    }


def _config(legend):
    return {'DataSet': [], 'Figures': [{
        'name': 'f', 'frame': {'ax': {'legend': legend}},
        'layers': [{'name': 'mean', 'axes': 'ax', 'method': 'plot',
                    'coordinates': {'x': [0, 1], 'y': [0, 1]},
                    'legend': {'object': 'mean', 'label': 'Mean'}}],
    }]}


@pytest.mark.parametrize('legend, code', [
    ({'card': 'missing'}, 'JP-LEG-003'),
    ({'item_formats': {'new': {'key': 'stack'}}}, 'JP-LEG-003'),
])
def test_card_and_role_diagnostics(legend, code):
    config = _config(legend)
    assert not list(config_validator().iter_errors(config))
    diagnostics = list(validate_config(config, check_columns=False))
    assert any(d.code == code for d in diagnostics)


def test_man_exposes_live_card_catalog_and_axes_style_selection(capsys):
    assert main(['man', 'legend', '--json']) == 0
    payload = json.loads(capsys.readouterr().out)['data']
    assert {card['name'] for card in payload['legend_cards']} == set(builtin_card_names())
    assert any('line_errorbar' in card['items'] for card in payload['legend_cards'])
    style = next(item for item in section('styles') if item['style'] == ['a4paper_2x1', 'rectRatio'])
    assert style['legend_cards'] == {'ax': 'compact', 'axr': 'compact'}


def test_card_validation_and_man_do_not_import_render_or_data_stack(tmp_path):
    card = tmp_path / 'custom.json'
    card.write_text(json.dumps({'schema_version': 1, 'extends': 'paper'}))
    config = _config({'card': './custom.json'})
    probe = """
import json, sys
from jarvisplot.validation import validate_config
from jarvisplot.man_render_agent import agent_topic_data
assert not list(validate_config(json.loads(sys.argv[1]), base_dir=sys.argv[2], check_columns=False))
assert agent_topic_data('legend')['legend_cards']
assert not set(('matplotlib','pandas','polars','scipy','shapely')) & set(sys.modules)
"""
    result = subprocess.run([sys.executable, '-c', probe, json.dumps(config), str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('style', [[{}], [['a']], [None], ['a4paper_2x1', {}]])
def test_invalid_style_tokens_report_schema_errors_without_crashing(style):
    config = _config({'card': 'compact'})
    config['Figures'][0]['style'] = style
    assert list(config_validator().iter_errors(config))
    diagnostics = list(validate_config(config, check_columns=False))
    assert any(d.level == 'error' and '.style' in d.path for d in diagnostics)
