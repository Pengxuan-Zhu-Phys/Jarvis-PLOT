from copy import deepcopy
import json
import subprocess
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pytest

from jarvisplot.core_assets import load_styles
from jarvisplot.Figure.figure import Figure
from jarvisplot.Figure.legend_runtime import apply_legend
from jarvisplot.legend_cards import (
    assemble_legend_entries, load_legend_card, resolve_axes_legend,
)
from jarvisplot.schema_catalog import config_validator
from jarvisplot.validation import validate_config


def _layer(obj='measurement', label='Measurement', role=None, axes='ax', **kw):
    declaration = {'object': obj, 'label': label}
    if role is not None:
        declaration['role'] = role
    return {'axes': axes, 'method': 'plot', 'legend': declaration,
            'coordinates': {'x': [0, 1], 'y': [1, 2]}, **kw}


def _config(layers, legend=None):
    figure = {'name': 'forward', 'style': ['a4paper_2x1', 'rectRatio'], 'layers': layers}
    if legend is not None:
        figure['frame'] = {'ax': {'legend': legend}}
    return {'DataSet': [], 'Figures': [figure]}


def _figure(figure):
    fig = Figure()
    fig.logger = SimpleNamespace(**{key: lambda *a, **k: None for key in ('debug', 'info', 'warning', 'error')})
    fig.jpstyles = load_styles(fig.load_path)
    assert fig.from_dict(figure)
    return fig


def test_grouping_is_independent_of_layer_order_and_respects_card_paint_order():
    layers = [_layer(role='line'), _layer('solo', 'Solo'), _layer(role='errorbar')]
    original = deepcopy(layers)
    config = {'card': 'paper', 'order': ['solo'], 'ncols': 2}
    kwargs, entries = resolve_axes_legend(config, layers)
    assert kwargs['ncols'] == 2 and 'order' not in kwargs
    assert [entry['object'] for entry in entries] == ['solo', 'measurement']
    assert (entries[0]['layers'], entries[0]['key']) == (['@layer:1'], 'overlay')
    assert (entries[1]['layers'], entries[1]['key']) == (['@layer:2', '@layer:0'], 'overlay')
    assert layers == original
    assert config == {'card': 'paper', 'order': ['solo'], 'ncols': 2}


def test_full_render_enables_card_and_isolates_axes_with_unnamed_layers():
    layers = [
        _layer('band', 'Band', 'line', style={'color': 'red'}),
        _layer('single', 'Single', style={'color': 'green', 'label': 'old label'}),
        _layer('band', 'Band', 'band', method='fill_between',
               coordinates={'x': [0, 1], 'y1': [.8, 1.8], 'y2': [1.2, 2.2]},
               style={'color': 'red', 'alpha': .25}),
        _layer('band', 'Ratio', axes='axr', coordinates={'x': [0, 1], 'y': [1, 1]}),
        {'name': 'unlisted', 'axes': 'ax', 'method': 'plot',
         'coordinates': {'x': [0, 1], 'y': [.5, .5]}, 'style': {'label': 'Native only'}},
    ]
    config = _config(layers, {'order': ['single'], 'ncols': 2})
    assert not list(validate_config(config, check_columns=False))
    original = deepcopy(config)
    fig = _figure(config['Figures'][0])
    fig.render()
    fig.fig.canvas.draw()
    legend = fig.axes['ax'].ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ['Single', 'Band']
    assert [t.get_text() for t in fig.axes['axr'].ax.get_legend().get_texts()] == ['Ratio']
    assert legend.labelspacing == .25
    assert config == original


@pytest.mark.parametrize('disabled', [False, {'enabled': False}])
def test_explicit_disable_wins_over_layer_declarations(disabled):
    config = _config([_layer()], disabled)
    assert not list(validate_config(config, check_columns=False))
    fig = _figure(config['Figures'][0])
    fig.render()
    assert fig.axes['ax'].ax.get_legend() is None


def test_single_native_errorbar_preserves_caps_and_style():
    fig, ax = plt.subplots()
    error = ax.errorbar([0, 1], [1, 2], yerr=.1, fmt='o', color='purple', capsize=3)
    limits = (ax.get_xlim(), ax.get_ylim())
    legend = apply_legend(ax, {'card': 'paper'}, {'@layer:0': error},
                          layer_specs=[_layer(method='errorbar')])
    fig.canvas.draw()
    assert [t.get_text() for t in legend.get_texts()] == ['Measurement']
    assert len(legend.findobj(LineCollection)) == 1
    assert any(line.get_marker() == 'o' and line.get_color() == 'purple'
               for line in legend.findobj(Line2D))
    assert len([line for line in legend.findobj(Line2D) if line.get_marker() == '_']) == 2
    assert (ax.get_xlim(), ax.get_ylim()) == limits


def test_forward_stack_uses_template_order_and_separation():
    fig, ax = plt.subplots()
    specs = [_layer(role='lower'), _layer(role='line'), _layer(role='upper')]
    handles = {f'@layer:{i}': ax.plot([0, 1], [i, i], alpha=.3 if i != 1 else 1)
               for i in range(3)}
    legend = apply_legend(ax, {'card': 'paper'}, handles, layer_specs=specs)
    fig.canvas.draw()
    lines = legend.findobj(Line2D)
    assert [line.get_alpha() for line in lines] == [.3, 1, .3]
    centers = [line.get_transform().transform((0, np.mean(line.get_ydata())))[1] for line in lines]
    # Matplotlib subtracts its handle-box descent when handleheight differs from .7.
    height = legend.handleheight - .35 * (legend.handleheight - .7)
    gap = legend.get_texts()[0].get_fontsize() * height / 3 * fig.dpi / 72
    assert np.diff(centers) == pytest.approx([-gap, -gap])


@pytest.mark.parametrize('layers, legend, message, code', [
    ([_layer(role='line'), _layer(role='line')], {}, 'repeats role', 'JP-LEG-005'),
    ([_layer(role='line'), _layer(label='Other', role='band')], {}, 'conflicting labels', 'JP-LEG-005'),
    ([_layer(), _layer(role='band')], {}, 'every layer needs role', 'JP-LEG-005'),
    ([_layer(role='upper'), _layer(role='line')], {}, 'no card item matches', 'JP-LEG-005'),
    ([_layer(role='bogus')], {}, 'no card item matches', 'JP-LEG-005'),
    ([_layer()], {'order': ['missing']}, 'unknown objects', 'JP-LEG-005'),
    ([_layer()], {'entries': [{'label': 'legacy', 'layers': ['mean']}]}, 'Unsupported axes legend options', 'JP-LEG-001'),
    ([_layer()], {'labels': ['native']}, 'Unsupported axes legend options', 'JP-LEG-001'),
    ([_layer()], {'handles': []}, 'Unsupported axes legend options', 'JP-LEG-001'),
    ([], {'order': ['missing']}, 'unknown objects', 'JP-LEG-005'),
])
def test_forward_errors_are_detected_by_validation_and_render_resolver(layers, legend, message, code):
    config = _config(layers, legend)
    diagnostics = list(validate_config(config, check_columns=False))
    assert any(d.code == code and message in d.message for d in diagnostics)
    with pytest.raises(ValueError, match=message):
        resolve_axes_legend(legend, layers)


def test_layer_error_points_to_original_layer_even_with_other_axes_interleaved():
    config = _config([_layer(axes='axr'), _layer(role='line'), _layer(role='line')])
    diagnostics = list(validate_config(config, check_columns=False))
    assert any(d.code == 'JP-LEG-005' and d.path == '$.Figures[0].layers[2].legend' for d in diagnostics)


def test_custom_templates_match_roles_and_conflicting_layouts_fail():
    layers = [_layer(role='ribbon'), _layer(role='central')]
    config = {'item_formats': {'ribbon_line': {'roles': ['ribbon', 'central'], 'key': 'overlay'}}}
    _, entries = resolve_axes_legend(config, layers)
    assert (entries[0]['layers'], entries[0]['key']) == (['@layer:0', '@layer:1'], 'overlay')
    config = {'item_formats': {'same': {'roles': ['band', 'line'], 'key': 'overlay'}}}
    assert resolve_axes_legend(config, [_layer(role='line'), _layer(role='band')])[1]
    config['item_formats']['same']['key'] = 'stack'
    with pytest.raises(ValueError, match='conflicting item formats'):
        resolve_axes_legend(config, [_layer(role='line'), _layer(role='band')])


@pytest.mark.parametrize('declaration', [
    {}, {'object': 'x'}, {'label': 'x'}, {'object': '', 'label': 'x'},
    {'object': 'x', 'label': 'x', 'role': ''},
    {'object': 'x', 'label': 'x', 'role': None},
    {'object': 'x', 'label': 'x', 'format': 'line'}, True,
])
def test_layer_legend_schema_and_runtime_reject_invalid_declarations(declaration):
    layer = _layer(legend=declaration)
    config = _config([layer])
    assert list(config_validator().iter_errors(config))
    with pytest.raises(ValueError):
        assemble_legend_entries([layer], load_legend_card()['items'])


def test_forward_validation_and_manual_do_not_load_matplotlib():
    config = _config([_layer(role='line'), _layer(role='errorbar')], {'card': 'paper'})
    probe = '''
import json, sys
from jarvisplot.validation import validate_config
from jarvisplot.man_render_agent import agent_topic_data
assert not list(validate_config(json.loads(sys.argv[1]), check_columns=False))
assert agent_topic_data('legend')['legend_cards']
assert not set(('matplotlib','pandas','polars','scipy','shapely')) & set(sys.modules)
'''
    result = subprocess.run([sys.executable, '-c', probe, json.dumps(config)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
