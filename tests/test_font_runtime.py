"""Font defaults must match actual artists, measurements and local overrides."""
from copy import deepcopy
from types import SimpleNamespace

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from jarvisplot.Figure.figure import Figure
from jarvisplot.Figure.legend_runtime import apply_legend
from jarvisplot.fonts import load_font_card, update_font_card
from jarvisplot.forest_text import text_metrics_mm


@pytest.fixture(autouse=True)
def portable_card(tmp_path, monkeypatch):
    monkeypatch.setenv('JARVIS_HOME', str(tmp_path / 'jarvis'))
    update_font_card('families.sans', 'DejaVu Sans')


def standard_figure(info):
    figure = Figure()
    figure.logger = SimpleNamespace(**{key: lambda *a, **k: None for key in ('debug', 'info', 'warning', 'error')})
    figure.jpstyles = {'test': {'rect': {
        'Frame': {'figure': {'figsize': [4, 3]}, 'axes': {'ax': {'rect': [.15, .2, .8, .7]}},
                  'ax': {'title_params': {'fontsize': 9, 'family': 'unused'},
                         'labels': {'x': '$x$', 'y': 'Response', 'xlabel': {'fontsize': 6}, 'ylabel': {'fontsize': 6}, 'ylabel_coords': {'x': -.1, 'y': .5}},
                         'ticks': {'both': {'labelsize': 5}},
                         'legend': {'enabled': False, 'fontsize': 5}}},
        'Style': {},
    }}}
    assert figure.from_dict({'name': 'fonts', 'style': ['test'], **info}), figure._setup_error
    return figure


def test_standard_actual_roles_local_aliases_and_scoped_math_default():
    update_font_card('roles.text.size_pt', 11)
    update_font_card('roles.title.size_pt', 12)
    update_font_card('roles.legend.size_pt', 8)
    update_font_card('roles.tick.style', 'italic')
    update_font_card('math.fontset', 'dejavusans')
    with mpl.rc_context({'mathtext.fontset': 'cm', 'font.family': ['STIXGeneral'], 'font.size': 17}):
        figure = standard_figure({'frame': {'ax': {
            'title': r'$\rho$ title', 'title_params': {'family': 'STIXGeneral', 'size': 10, 'weight': 'normal'},
            'legend': {'fontsize': 6, 'prop': {'math_fontfamily': 'stix'}},
            'text': [{'x': .5, 'y': .5, 's': 'Default body'},
                     {'x': .5, 'y': .25, 's': 'Explicit body', 'family': 'STIXGeneral', 'size': 13}],
        }}})
        ax = figure.axes['ax']
        line, = ax.plot(x=[0, 1], y=[0, 1])
        legend = apply_legend(ax, {**figure.frame['ax']['legend'], 'enabled': True}, {'@layer:0': line},
                              layer_specs=[{'legend': {'object': 'line', 'label': '$x$ line'}}])
        # Render's layer legend pass should leave this explicit test legend.
        figure.frame['ax']['legend']['enabled'] = False
        figure.render()
        figure.fig.canvas.draw()
        title = next(text for text in ax.texts if text.get_text().endswith(' title'))
        body = next(text for text in ax.texts if text.get_text() == 'Default body')
        explicit = next(text for text in ax.texts if text.get_text() == 'Explicit body')
        assert title.get_fontfamily() == ['STIXGeneral'] and title.get_fontsize() == 10
        assert title.get_fontweight() == 'normal'
        assert title.get_math_fontfamily() == 'dejavusans'
        assert body.get_fontfamily() == ['DejaVu Sans'] and body.get_fontsize() == 11
        assert explicit.get_fontfamily() == ['STIXGeneral'] and explicit.get_fontsize() == 13
        assert ax.xaxis.label.get_fontfamily() == ['STIXGeneral'] and ax.xaxis.label.get_fontsize() == 6
        assert ax.get_xticklabels()[0].get_fontstyle() == 'italic'
        assert legend.get_texts()[0].get_fontsize() == 6
        assert legend.get_texts()[0].get_fontfamily() == ['DejaVu Sans']
        assert legend.get_texts()[0].get_math_fontfamily() == 'stix'
        assert all(artist.get_fontproperties().get_file() for artist in (title, body, explicit, ax.xaxis.label))
        assert mpl.rcParams['mathtext.fontset'] == 'cm'
        assert mpl.rcParams['font.family'] == ['STIXGeneral']
        assert mpl.rcParams['font.size'] == 17
        plt.close(figure.fig)


def test_new_yaml_numbered_axes_receive_role_defaults():
    update_font_card('roles.axis_label.size_pt', 7)
    update_font_card('roles.title.size_pt', 10)
    figure = standard_figure({'frame': {
        'axes': {'ax1': {'rect': [.2, .25, .2, .2]}},
        'ax1': {'labels': {'x': 'New axis', 'y': 'New value', 'ylabel': {'fontsize': 8}}},
    }})
    figure.render()
    assert figure.axes['ax1'].xaxis.label.get_fontfamily() == ['STIXGeneral']
    assert figure.axes['ax1'].xaxis.label.get_fontsize() == 7
    assert figure.axes['ax1'].yaxis.label.get_fontsize() == 8
    plt.close(figure.fig)


def test_correlation_solve_measures_explicit_fonts_and_renderer_uses_card_roles():
    from jarvisplot.core_assets import load_styles
    from jarvisplot.core_runtime import _corr_card, _corr_tick_sizes, _solve_corr_square
    from jarvisplot.Figure.corr_layout import measure_text_mm
    from jarvisplot.Figure.correlation_runtime import correlation
    from jarvisplot.Figure.corrplot_runtime import draw_corrplot
    figure = Figure()
    styles = load_styles(figure.load_path)
    styles['corrplot']['matrix']['Geometry']['margin']['fit'] = True
    info = {'style': ['corrplot', 'matrix'], 'frame': {
        'axcorr': {'ticks': {'both': {'labelfontfamily': 'STIXGeneral', 'labelsize': 9}}},
        'axccorr': {'label': {'ylabel': '$\\rho$', 'family': 'DejaVu Sans', 'size': 8}},
    }}
    original = deepcopy(info)
    card = _corr_card(SimpleNamespace(style=styles), info)
    assert not {'labels', 'legend'} & card['Frame']['axcorr'].keys()
    label_size, _, family, _ = _corr_tick_sizes(card)
    assert label_size == 9 and family.get_family() == ['STIXGeneral']
    names = ['Longer variable name one', 'Another variable name']
    geometry, solved = _solve_corr_square(card, info, card['Style']['corrplot'], names, 'axcorr')
    widest = max(measure_text_mm(name, size_pt=9, family=family)[0] for name in names)
    assert geometry.width_mm > widest + geometry.panel_mm
    assert info == original
    figure.logger = SimpleNamespace(**{key: lambda *a, **k: None for key in ('debug', 'info', 'warning', 'error')})
    figure.jpstyles = styles
    from jarvisplot.legend_cards import merge_legend_settings
    assert figure.from_dict({'name': 'corr-fonts', 'style': info['style'],
                            'frame': merge_legend_settings(info['frame'], solved), 'layers': []})
    figure.render()
    tick = figure.axes['axcorr'].get_yticklabels()[0]
    assert tick.get_fontfamily() == ['STIXGeneral'] and tick.get_fontsize() == 9
    actual = text_metrics_mm(names[0], 9, 'unused', 'normal', prop=tick.get_fontproperties())[0]
    assert measure_text_mm(names[0], size_pt=9, family=family)[0] == pytest.approx(actual)
    ax = figure.axes['axcorr'].ax
    table = correlation(pd.DataFrame({'a': [1, 2, 4], 'b': [3, 2, 1]}), {'columns': ['a', 'b']})
    draw_corrplot(ax, __df__=table, method='number', __corr_fonts__=card['Style']['corrplot']['__corr_fonts__'])
    assert all(text.get_fontfamily() == ['STIXGeneral'] for text in ax.texts)
    plt.close(figure.fig)


def test_explicit_offset_text_survives_tick_defaults():
    from jarvisplot.font_runtime import pin_figure_fonts
    update_font_card('roles.tick.weight', 'bold')
    figure, ax = plt.subplots()
    ax.yaxis.offsetText.set_text('$10^3$')
    ax.yaxis.offsetText.set(fontweight='normal', fontstyle='italic', math_fontfamily='stix')
    figure._jarvis_axes = {'axc': ax}
    pin_figure_fonts(figure, {'axc': {'offset_text': {'fontweight': 'normal', 'fontstyle': 'italic', 'math_fontfamily': 'stix'}}}, load_font_card())
    assert ax.yaxis.offsetText.get_fontweight() == 'normal'
    assert ax.yaxis.offsetText.get_fontstyle() == 'italic'
    assert ax.yaxis.offsetText.get_math_fontfamily() == 'stix'
    plt.close(figure)


def test_missing_helvetica_uses_actual_similar_font_for_standard_artists(monkeypatch):
    from matplotlib import font_manager
    from jarvisplot.forest_text import font_path
    update_font_card('families.sans', 'Helvetica')
    original = font_manager.findfont
    unavailable = {'Helvetica', 'Arial', 'Liberation Sans', 'Nimbus Sans'}
    def missing_fonts(properties, *args, **kwargs):
        if unavailable.intersection(font_manager.FontProperties._from_any(properties).get_family()):
            raise ValueError('Simulated missing font')
        return original(properties, *args, **kwargs)
    monkeypatch.setattr(font_manager, 'findfont', missing_fonts)
    font_path.cache_clear()
    figure = standard_figure({'frame': {'ax': {'title': 'Fallback title',
        'text': [{'x': .5, 'y': .5, 's': 'Fallback body'}]}}})
    try:
        figure.render()
        figure.fig.canvas.draw()
        for artist in figure.axes['ax'].texts:
            assert artist.get_fontfamily() == ['DejaVu Sans']
            assert 'DejaVuSans' in artist.get_fontproperties().get_file()
    finally:
        plt.close(figure.fig)
        font_path.cache_clear()


def test_ttc_small_caption_direct_canvas_draw_keeps_native_outline():
    from pathlib import Path
    from matplotlib.ft2font import FT2Font
    from jarvisplot.fonts import match_font
    from jarvisplot.forest_text import font_properties
    match = match_font('Helvetica')
    if Path(match['path']).suffix.lower() != '.ttc':
        pytest.skip('Installed TTC face required for macOS hint-program regression')
    figure, ax = plt.subplots(dpi=100)
    prop = font_properties(4, 'Helvetica')
    assert FT2Font(prop.get_file()).family_name == match['family']
    ax.text(.1, .2, '       Powered by', fontproperties=prop)
    try:
        with mpl.rc_context({'text.hinting': 'default'}):
            figure.canvas.draw()
    finally:
        plt.close(figure)


def test_forest_role_measurements_render_and_explicit_native_family(tmp_path):
    from jarvisplot.Figure.forest_runtime import ForestFigure
    from jarvisplot.Figure.data_pipelines import DataContext, SharedContent
    from jarvisplot.forest_contract import appearance
    update_font_card('roles.text.weight', 'bold')
    info = {'name': 'forest-fonts', 'type': 'forest', 'data': 'rows',
            'title': {'text': 'Role title', 'fontsize': 8},
            'description': {'text': '$p_T$ caption', 'math': True},
            'frame': {'axlogo': False}, 'forest': {'geometry': {'footer_mm': 30}},
            'columns': [{'field': 'label', 'label': 'Group', 'role': 'row_name'},
                        {'field': 'n', 'label': 'N', 'role': 'right', 'fontweight': 'bold'}]}
    figure = ForestFigure()
    figure.jpstyles = {'forest': {'swiss_column': appearance(info)[3]}}
    figure.config = {'output': {'dir': str(tmp_path), 'formats': [], 'dpi': 160}}
    figure.context = DataContext(SharedContent())
    rows = pd.DataFrame([{'row': '1', 'label': 'Alpha', 'n': '1234', 'estimate': 1, 'lower': 0, 'upper': 2}])
    figure.context.register('rows', lambda shared: rows)
    with mpl.rc_context({'mathtext.fontset': 'cm'}):
        assert figure.from_dict(info)
        figure.render()
        ax = figure.axes['ax']
        artist = next(text for text in ax.texts if text.get_text() == 'Alpha')
        right = next(text for text in ax.texts if text.get_text() == '1234')
        header = next(text for text in ax.texts if text.get_text() == 'Group')
        assert artist.get_fontfamily() == ['DejaVu Sans'] and artist.get_fontweight() == 'bold'
        assert right.get_fontfamily() == ['STIXGeneral'] and right.get_fontweight() == 'bold'
        assert header.get_fontfamily() == ['STIXGeneral'] and header.get_fontweight() == 'normal'
        measured = text_metrics_mm('1234', right.get_fontsize(), 'unused', 'normal', prop=right.get_fontproperties())[0]
        assert figure.forest_layout['columns'][1]['content_width_mm'] == pytest.approx(measured)
        assert next(text for text in ax.texts if text.get_gid() == 'forest-description-00').get_math_fontfamily() == 'stix'
        assert mpl.rcParams['mathtext.fontset'] == 'cm'
    plt.close(figure.fig)
    override = deepcopy(info)
    override['forest']['typography'] = {'family': 'DejaVu Sans', 'header_pt': 6.1}
    fonts = appearance(override)[1]
    assert all(role['family'] == 'DejaVu Sans' for role in fonts['_font_roles'].values())
    assert fonts['header_pt'] == 6.1
