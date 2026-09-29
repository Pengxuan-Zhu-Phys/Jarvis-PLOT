from copy import deepcopy
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.container import BarContainer, ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pytest

from jarvisplot.Figure.figure import Figure
from jarvisplot.Figure.legend_runtime import apply_legend, representative_handle
from jarvisplot.core_assets import load_styles
from jarvisplot.schema_catalog import config_validator
from jarvisplot.validation import validate_config


def _parts(ax):
    return [
        ax.fill_between([0, 1], [0, 0], [1, 1], color='skyblue', alpha=.3),
        ax.plot([0, 1], [.5, .5], color='blue', linewidth=2),
        ax.plot([0, 1], [.8, .8], color='blue', alpha=.3),
        ax.plot([0, 1], [.5, .5], color='blue', linewidth=2),
        ax.plot([0, 1], [.2, .2], color='blue', alpha=.3),
    ]


def _specs():
    return [{'legend': {'object': obj, 'label': obj.title(), 'role': role}}
            for obj, role in [('train', 'band'), ('train', 'line'),
                              ('test', 'upper'), ('test', 'line'), ('test', 'lower')]]


def test_compact_composite_keys_keep_real_styles_and_data_limits():
    fig, ax = plt.subplots()
    handles = {f'@layer:{i}': handle for i, handle in enumerate(_parts(ax))}
    limits = (ax.get_xlim(), ax.get_ylim())
    counts = (len(ax.lines), len(ax.collections), len(ax.patches))
    config = {'card': 'compact', 'ncols': 2, 'handlelength': 2.1}
    original = deepcopy(config)
    legend = apply_legend(ax, config, handles, layer_specs=_specs())
    fig.canvas.draw()
    assert [t.get_text() for t in legend.get_texts()] == ['Train', 'Test']
    assert legend.handlelength == 2.1 and legend.labelspacing == .25
    assert not legend.get_frame_on()
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    assert (len(ax.lines), len(ax.collections), len(ax.patches)) == counts
    assert config == original
    lines = legend.findobj(Line2D)
    assert len(lines) == 4
    assert legend.findobj(Rectangle)[0].get_facecolor()[3] == pytest.approx(.3)
    assert [line.get_alpha() for line in lines[1:]] == [.3, None, .3]
    centers = [line.get_transform().transform((0, np.mean(line.get_ydata())))[1]
               for line in lines[1:]]
    expected_gap = legend.get_texts()[0].get_fontsize() * .7 / 3 * fig.dpi / 72
    assert np.diff(centers) == pytest.approx([-expected_gap, -expected_gap])
    plt.close(fig)


@pytest.mark.parametrize('config', [True, {}, {'card': 'compact'}])
def test_native_artist_labels_do_not_implicitly_create_legend_items(config):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label='Native')
    with pytest.raises(ValueError, match='no layer declares'):
        apply_legend(ax, config)
    assert ax.get_legend() is None
    assert apply_legend(ax, False) is None
    plt.close(fig)


def test_handle_selection_preserves_containers_and_rejects_ambiguous_series():
    fig, ax = plt.subplots()
    error = ax.errorbar([0, 1], [0, 1], yerr=.1)
    assert isinstance(error, ErrorbarContainer)
    assert representative_handle(error) is error
    assert isinstance(representative_handle(ax.hist([0, .1, .8])), BarContainer)
    assert representative_handle([]) is None
    with pytest.raises(ValueError, match='multiple legend handles'):
        representative_handle(ax.plot([0, 1], [[0, 2], [1, 3]]))
    with pytest.raises(ValueError, match='multiple legend handles'):
        representative_handle(ax.hist([[0, .1, .8], [.2, .5, .7]]))
    plt.close(fig)


@pytest.mark.parametrize('option,value', [
    ('entries', [{'label': 'Old', 'layers': ['line']}]),
    ('preset', 'compact'), ('handles', []), ('labels', ['Old']), ('axes', 'ax'),
])
@pytest.mark.parametrize('enabled', [True, False])
def test_removed_declaration_forms_fail_schema_validation_and_runtime(option, value, enabled):
    legend = {option: value, 'enabled': enabled}
    config = {'Figures': [{'name': 'f', 'layers': [], 'frame': {'ax': {'legend': legend}}}]}
    assert list(config_validator().iter_errors(config))
    assert any(d.code == 'JP-LEG-001' for d in validate_config(config, check_columns=False))
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match='Unsupported axes legend options'):
        apply_legend(ax, legend)
    plt.close(fig)


@pytest.mark.parametrize('height', [.7, 1., 1.2, 2.])
def test_stacked_center_aligns_with_native_line_for_taller_keys(height):
    fig, ax = plt.subplots()
    specs = _specs()[1:]
    specs[0]['legend'].pop('role')
    handles = {f'@layer:{i}': h for i, h in enumerate(_parts(ax)[1:])}
    legend = apply_legend(ax, {'card': 'paper', 'handleheight': height, 'ncols': 2},
                          handles, layer_specs=specs)
    fig.canvas.draw()
    centers = [line.get_transform().transform((0, np.mean(line.get_ydata())))[1]
               for line in legend.findobj(Line2D)]
    assert centers[0] == pytest.approx(centers[2])
    assert centers[1] > centers[2] > centers[3]
    plt.close(fig)


def test_figure_names_do_not_affect_forward_binding_and_render_errors_are_not_swallowed():
    fig = Figure()
    fig.logger = SimpleNamespace(**{key: lambda *a, **k: None for key in ('debug', 'info', 'warning', 'error')})
    fig.jpstyles = load_styles(fig.load_path)
    config = {
        'name': 'references', 'style': ['a4paper_2x1', 'rect'],
        'layers': [
            {'name': name, 'axes': 'ax', 'method': 'plot',
             'coordinates': {'x': [0, 1], 'y': [0, 1]}, 'style': {'color': color},
             'legend': {'object': color, 'label': color}}
            for name, color in [('red', 'red'), ('@layer:0', 'blue')]
        ],
    }
    assert fig.from_dict(config)
    fig.render()
    assert [h.get_color() for h in fig.axes['ax'].ax.get_legend().findobj(Line2D)] == ['red', 'blue']
    plt.close(fig.fig)
    config['frame'] = {'ax': {'legend': {'loc': 'not-a-location'}}}
    assert fig.from_dict(config)
    with pytest.raises(ValueError, match='Legend draw failed'):
        fig.render()
    plt.close(fig.fig)
