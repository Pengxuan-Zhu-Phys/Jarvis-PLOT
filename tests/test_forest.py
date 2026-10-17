"""Native forest invariants: generic data, common x mapping, physical bounds."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from jarvisplot.forest_contract import (
    appearance, load_forest_tables, solve_forest, x_to_mm,
)
from jarvisplot.validation import validate_config, validate_file

from matplotlib.font_manager import FontProperties, findfont
try:
    findfont(FontProperties(family='Helvetica'), fallback_to_default=False)
except ValueError:
    pytestmark = pytest.mark.skip(reason='Swiss regression fixtures require installed Helvetica; portable forest CLI coverage runs separately.')

EXAMPLE = Path(__file__).resolve().parents[1] / 'Example' / 'forest'


def fixture(glyph='interval'):
    info = {'name': 'comparison', 'type': 'forest', 'data': 'rows', 'points': 'points', 'glyph': glyph}
    row = {'row': '001', 'label': 'Group A', 'estimate': 1, 'lower': 0, 'upper': 2}
    if glyph == 'box':
        row.update(q1=.5, median=1, q3=1.5)
    frames = {'rows': pd.DataFrame([row]), 'points': pd.DataFrame([{'row': '001', 'id': '01', 'x': 1, 'class': 'normal'}])}
    return info, frames


def draw(info, frames, tmp_path):
    from jarvisplot.Figure.forest_runtime import ForestFigure
    from jarvisplot.Figure.data_pipelines import SharedContent, DataContext
    import matplotlib.pyplot as plt
    fig = ForestFigure()
    card = appearance(info)[3]
    fig.jpstyles = {'forest': {'swiss_column': card}}
    fig.config = {'output': {'dir': str(tmp_path), 'formats': [], 'dpi': 220}}
    fig.context = DataContext(SharedContent())
    for name, table in frames.items():
        fig.context.register(name, lambda _shared, table=table: table)
    try:
        assert fig.from_dict(info)
        fig.render()
        return fig
    finally:
        if hasattr(fig, 'fig'):
            plt.close(fig.fig)


def test_csv_literal_ids_optional_annotations_and_narrow_tables(tmp_path):
    path = tmp_path / 'rows.csv'
    path.write_text('row,label,estimate,lower,upper,irrelevant\n001,NA,1,0,2,metadata\n')
    cfg = {'DataSet': [{'name': 'rows', 'path': str(path), 'type': 'csv'}],
           'Figures': [{'name': 'f', 'type': 'forest', 'data': 'rows'}]}
    frames = load_forest_tables(cfg, base_dir=str(tmp_path))
    assert frames['rows']['row'].tolist() == ['001']
    assert frames['rows']['label'].tolist() == ['NA']
    assert 'irrelevant' not in frames['rows']
    assert solve_forest(cfg['Figures'][0], frames)['rows'][0]['label'] == 'NA'
    path.write_text('row,estimate,lower,upper\n001,NaN,0,2\n')
    with pytest.raises(ValueError, match='nonfinite'):
        solve_forest(cfg['Figures'][0], load_forest_tables(cfg, base_dir=str(tmp_path)))


@pytest.mark.parametrize('glyph', ['interval', 'box'])
def test_generic_render_and_physical_sizes(glyph, tmp_path):
    info, frames = fixture(glyph)
    fig = draw(info, frames, tmp_path)
    layout = fig.forest_layout
    assert tuple(fig.fig.get_size_inches()*25.4) == pytest.approx((82.5, 45.5))
    assert len(fig.fig.axes) == 2
    assert layout['plot_width_mm'] == 78.5
    assert layout['geometry']['cell_y_mm'] < layout['geometry']['plot_y_mm']-layout['geometry']['point_band_mm']/2
    assert layout['row_pitch_mm'] == 5.5
    assert layout['verification']['tick_label_count'] == 6
    assert layout['verification']['text_and_legend_bounds'] == 'passed'
    p = layout['rows'][0]['points'][0]
    assert p['style']['diameter_mm'] == .8
    assert layout['typography']['label_pt'] == 6
    assert p['x'] == 1
    assert p['x_mm'] == x_to_mm(1, layout['shared_domain'], layout['geometry'])


def test_real_demo_all_rows_points_shared_mapping_and_determinism():
    config, bag = validate_file(str(EXAMPLE / 'real-demo.yaml'))
    assert bag.ok, bag.render_human()
    frames = load_forest_tables(config, base_dir=str(EXAMPLE))
    info = config['Figures'][0]
    layout = solve_forest(info, frames)
    assert layout == solve_forest(info, frames)
    assert (layout['row_count'], layout['point_count']) == (97, 241)
    assert (layout['width_mm'], layout['height_mm']) == (82.5, 573.5)
    assert layout['shared_domain'] == [-40, 40]
    assert layout['shared_ticks'] == [-40, 0, 40]
    actual = {(r['row'], p['id']): p['x'] for r in layout['rows'] for p in r['points']}
    expected = {(r.row, r.id): r.x for r in frames['observations'].itertuples()}
    assert actual == expected
    g = layout['geometry']
    for row in layout['rows']:
        for p in row['points']:
            assert p['x_mm'] == x_to_mm(p['x'], layout['shared_domain'], g)
            assert abs(p['offset_mm'])+p['radius_mm'] <= g['point_band_mm']/2+1e-8
        for i, p in enumerate(row['points']):
            for q in row['points'][i+1:]:
                assert math.hypot(p['x_mm']-q['x_mm'], p['offset_mm']-q['offset_mm']) >= p['radius_mm']+q['radius_mm']+max(p['style']['gap_mm'], q['style']['gap_mm'])-1e-8


def test_range_includes_outlier_reference_and_singleton():
    info, frames = fixture()
    frames['points'].loc[0, ['x', 'class']] = [100, 'outlier']
    info['x'] = {'reference': -20}
    layout = solve_forest(info, frames)
    assert layout['shared_domain'][0] < -20 and layout['shared_domain'][1] > 100
    info['x']['lim'] = [-10, 10]
    with pytest.raises(ValueError, match='excludes'):
        solve_forest(info, frames)


def test_degenerate_global_range_and_blank_box_with_points():
    info, frames = fixture('box')
    for k in ('lower', 'q1', 'median', 'q3', 'upper'):
        frames['rows'][k] = None
    info['x'] = {'minimum_span': 2}
    layout = solve_forest(info, frames)
    assert layout['shared_domain'][1]-layout['shared_domain'][0] >= 2
    assert all(v is None for v in layout['rows'][0]['coordinates'].values())


@pytest.mark.parametrize('target, column, value, error', [
    ('points', 'row', 'unknown', 'foreign key'),
    ('points', 'id', '', 'nonempty string'),
    ('points', 'x', float('inf'), 'nonfinite'),
    ('points', 'class', 'inferred', 'unknown class'),
    ('rows', 'estimate', float('nan'), 'nonfinite'),
    ('rows', 'lower', 3, 'unordered'),
    ('rows', 'lower', '', 'partial'),
])
def test_invalid_data_names_field_and_row(target, column, value, error):
    info, frames = fixture()
    frames[target][column] = frames[target][column].astype(object)
    frames[target].loc[0, column] = value
    with pytest.raises(ValueError, match=error):
        solve_forest(info, frames)


@pytest.mark.parametrize('table', ['rows', 'points'])
def test_duplicate_keys(table):
    info, frames = fixture()
    frames[table] = pd.concat([frames[table], frames[table]], ignore_index=True)
    with pytest.raises(ValueError, match='duplicate'):
        solve_forest(info, frames)


def test_no_drawable_row_and_partial_box():
    info, frames = fixture('box')
    frames['rows']['q1'] = None
    with pytest.raises(ValueError, match='partial box'):
        solve_forest(info, frames)
    for key in ('lower', 'q1', 'median', 'q3', 'upper'):
        frames['rows'][key] = None
    frames['points'] = frames['points'].iloc[:0]
    with pytest.raises(ValueError, match='no drawable'):
        solve_forest(info, frames)


def test_mapping_literal_slots_sparse_styles_and_canonical_legend():
    info, frames = fixture()
    frames['rows'] = frames['rows'].rename(columns={'row': 'study', 'label': 'title'})
    info['fields'] = {'row': 'study'}
    info['annotations'] = {'label': 'title', 'detail': False, 'stats': False}
    info['forest'] = {'scatter': {'classes': {'normal': {'color': '#112233'}}}}
    info['frame'] = {'ax': {'legend': {'order': ['normal'], 'ncols': 1}}}
    layout = solve_forest(info, frames)
    assert layout['rows'][0]['label'] == 'Group A'
    assert layout['style']['scatter']['classes']['normal']['color'] == '#112233'
    assert layout['style']['scatter']['classes']['normal']['marker'] == 'o'
    assert layout['legend_layers'] == [{'legend': {'object': 'glyph', 'label': 'Estimate + bounds'}}, {'legend': {'object': 'normal', 'label': 'Ordinary'}}]
    assert layout['legend']['ncols'] == 1
    info['frame']['ax']['legend']['order'] = ['unknown']
    with pytest.raises(ValueError, match='unknown objects'):
        solve_forest(info, frames)


def test_point_overflow_is_actionable_and_larger_band_preserves_all_points():
    info, frames = fixture()
    frames['points'] = pd.DataFrame([{'row': '001', 'id': str(i), 'x': 1} for i in range(9)])
    with pytest.raises(ValueError, match='001.*packing.*increase'):
        solve_forest(info, frames)
    info['forest'] = {'geometry': {'point_band_mm': 9, 'row_band_mm': 13, 'row_pitch_mm': 13.5, 'plot_y_mm': 7.5}}
    assert solve_forest(info, frames)['point_count'] == 9


def test_text_and_glyph_overflow_fail(tmp_path):
    info, frames = fixture()
    frames['rows']['label'] = 'Long literal annotation '*20
    with pytest.raises(ValueError, match='measured columns.*text line'):
        draw(info, frames, tmp_path)
    info, frames = fixture('box')
    info['forest'] = {'box': {'height_mm': 5}}
    with pytest.raises(ValueError, match='glyph exceeds'):
        draw(info, frames, tmp_path)


@pytest.mark.parametrize('bad', [{'layers': []}, {'method': 'scatter'}, {'x': {'scale': 'log'}}, {'forest': {'scatter': {'jitter': 1}}}, {'frame': {'ax': {'rect': [0,0,1,1]}}}, {'forest': {'geometry': {'width_mm': 100}}}])
def test_closed_schema_rejects_contradictory_or_unknown_slots(bad):
    figure = {'name': 'f', 'type': 'forest', 'data': 'rows', **bad}
    bag = validate_config({'DataSet': [{'name': 'rows', 'path': '/tmp/absent.csv', 'type': 'csv'}], 'Figures': [figure]}, check_columns=False)
    assert any(d.path.startswith('$.Figures') for d in bag.errors)


def test_cli_discovery_doctor_and_native_config_inspection(capsys):
    from jarvisplot.client import main
    from jarvisplot.capabilities import section
    from jarvisplot.templates_catalog import render_template_yaml
    import yaml
    assert main(['man', 'forest', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['data']['topic'] == 'type-forest'
    native = next(t for t in section('types') if t['name'] == 'forest')
    assert native['expands_to'] == 'native'
    assert native['styles'] == [['forest', 'swiss_column']]
    card = next(s for s in section('styles') if s['bundle'] == 'forest')
    assert card['axes'] == ['ax', 'axlogo']
    assert yaml.safe_load(render_template_yaml('forest'))['Figures'][0]['type'] == 'forest'
    assert main(['doctor', str(EXAMPLE/'interval.yaml'), '--json']) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc['ok'] is True
    assert doc['data']['dryrun']['forests'][0]['rows'][0]['row'] == '001'
    assert main(['config', 'expand', str(EXAMPLE/'box.yaml'), '--figure', 'forest-box', '--json']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['ok'] is True
    assert report['data']['status'] == 'resolved'
    assert report['data']['forests'][0]['point_count'] == 4
    assert report['data']['wrote'] is False


def test_columns_have_one_header_and_only_formatted_values(tmp_path):
    info, frames = fixture()
    frames['rows']['n'] = 11
    frames['rows']['cv'] = 9.673
    info['columns'] = [{'field': 'label', 'label': 'Group', 'role': 'row_name'},
                       {'field': 'n', 'label': 'n', 'role': 'right', 'format': '.0f'},
                       {'field': 'cv', 'label': 'CV [%]', 'role': 'right', 'format': '.1f'}]
    info['title'] = {'text': 'Forest comparison'}
    info['description'] = {'text': 'Supplied estimates and bounds', 'fontsize': 5}
    fig = draw(info, frames, tmp_path)
    layout = fig.forest_layout
    assert layout['rows'][0]['cells'] == ['Group A', '11', '9.7']
    texts = [t.get_text() for t in fig.axes['ax'].texts]
    assert texts.count('n') == texts.count('CV [%]') == 1
    assert '11' in texts and '9.7' in texts
    assert 'n 11' not in texts
    legend = fig.axes['ax'].get_legend()
    assert 'Estimate + bounds' in [t.get_text().replace('\n', ' ') for t in legend.get_texts()]
    assert legend.get_window_extent().y1 < fig.axes['ax'].transData.transform((0, 28.5))[1]
    assert 'axlogo' in fig.axes
    assert any(t.get_text() == 'Jarvis-HEP' for t in fig.axlogo.texts)
    assert layout['verification']['text_and_legend_bounds'] == 'passed'


def test_column_format_missing_nan_and_width_errors():
    info, frames = fixture()
    frames['rows']['cv'] = ''
    info['columns'] = [{'field': 'cv', 'label': 'CV [%]', 'role': 'right', 'format': '.1f'}]
    assert solve_forest(info, frames)['rows'][0]['cells'] == ['—']
    frames['rows']['cv'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        solve_forest(info, frames)
    frames['rows']['cv'] = 1
    info['columns'][0]['label'] = 'Very long header '*20
    with pytest.raises(ValueError, match='measured columns.*text line'):
        solve_forest(info, frames)
    info['columns'][0]['field'] = 'absent'
    with pytest.raises(ValueError, match='missing fields'):
        solve_forest(info, frames)


def test_measured_columns_use_all_rows_headers_and_roles(tmp_path):
    info, frames = fixture()
    frames['rows'] = pd.concat([frames['rows'], frames['rows'].assign(row='002', label='Longer group')], ignore_index=True)
    frames['rows']['W'] = ['1', '888']
    frames['rows']['n'] = [2, 11]
    # Row-name placement and font do not depend on list position. Within each
    # side, declaration order is retained, even when the sides are interleaved.
    info['columns'] = [{'field': 'n', 'label': 'N', 'role': 'right', 'format': '.0f'},
                       {'field': 'W', 'label': 'Weight'},
                       {'field': 'label', 'label': 'Group', 'role': 'row_name'}]
    fig = draw(info, frames, tmp_path)
    layout = fig.forest_layout
    n, w, label = layout['columns']
    from jarvisplot.forest_text import text_metrics_mm
    from jarvisplot.font_runtime import forest_font_properties
    fonts = layout['typography']
    assert label['content_width_mm'] == pytest.approx(text_metrics_mm('Longer group', 6, fonts['family'], 'normal',
        prop=forest_font_properties(fonts, 'text', 6))[0])
    assert w['content_width_mm'] == pytest.approx(text_metrics_mm('Weight', fonts['header_pt'], fonts['family'], 'normal',
        prop=forest_font_properties(fonts, 'axis_label', fonts['header_pt']))[0])
    assert label['left_mm'] < w['left_mm'] < n['left_mm']
    assert layout['geometry']['plot_left_mm'] < w['left_mm']
    assert layout['geometry']['plot_right_mm'] > n['left_mm']
    assert n['align'] == 'right' and w['align'] == label['align'] == 'left'
    assert label['fontweight'] == w['fontweight'] == 'normal'
    assert w['left_mm'] == pytest.approx(label['left_mm']+label['width_mm']+.8)
    assert n['left_mm']+n['width_mm'] == pytest.approx(80.5)
    # A shorter body value never changes a row's common numerical mapping.
    assert layout['rows'][0]['points'][0]['x_mm'] == x_to_mm(1, layout['shared_domain'], layout['geometry'])


def test_column_width_changes_with_actual_text_and_explicit_typography():
    info, frames = fixture()
    info['columns'] = [{'field': 'label', 'label': 'G', 'role': 'row_name'}]
    short = solve_forest(info, frames)
    frames['rows']['label'] = 'Group A extended'
    long = solve_forest(info, frames)
    assert long['columns'][0]['width_mm'] > short['columns'][0]['width_mm']
    assert long['plot_width_mm'] == short['plot_width_mm'] == 78.5
    assert long['geometry']['plot_left_mm'] == short['geometry']['plot_left_mm']
    info['columns'][0]['fontsize'] = 8
    assert solve_forest(info, frames)['columns'][0]['width_mm'] > long['columns'][0]['width_mm']
    info['columns'].append({'field': 'row', 'label': 'ID', 'role': 'row_name'})
    with pytest.raises(ValueError, match='at most one row_name'):
        solve_forest(info, frames)


@pytest.mark.parametrize('legacy', [{'width_mm': 8}, {'align': 'center'}])
def test_columns_reject_manual_width_and_legacy_alignment(legacy):
    from jarvisplot.schema_catalog import config_validator
    info, _ = fixture()
    info['columns'] = [{'field': 'label', 'label': 'G', **legacy}]
    assert list(config_validator().iter_errors({'DataSet': [], 'Figures': [info]}))


def test_large_mark_cannot_cross_canvas_boundary():
    info, frames = fixture()
    frames['points']['x'] = 0
    info['x'] = {'lim': [0, 2]}
    info['forest'] = {'scatter': {'diameter_mm': 5},
                      'geometry': {'point_band_mm': 6, 'row_band_mm': 9,
                                   'row_pitch_mm': 9.5, 'plot_y_mm': 6}}
    with pytest.raises(ValueError, match='001.*01.*figure width'):
        solve_forest(info, frames)


def test_marks_extend_beneath_both_text_groups_on_lower_band(tmp_path):
    info, frames = fixture('box')
    frames['rows']['n'] = 11
    info['columns'] = [{'field': 'label', 'label': 'Group', 'role': 'row_name'},
                       {'field': 'n', 'label': 'N', 'role': 'right'}]
    info['x'] = {'lim': [0, 2]}
    frames['points'] = pd.DataFrame([{'row': '001', 'id': 'left', 'x': 0},
                                    {'row': '001', 'id': 'right', 'x': 2}])
    fig = draw(info, frames, tmp_path)
    s = fig.forest_layout
    g = s['geometry']
    assert g['plot_left_mm'] == s['columns'][0]['left_mm']
    assert g['plot_right_mm'] == s['columns'][1]['left_mm']+s['columns'][1]['width_mm']
    assert s['verification']['text_and_legend_bounds'] == 'passed'
    top = s['rows'][0]['top_mm']
    body = [t for t in fig.axes['ax'].texts if t.get_text() in ('Group A', '11')]
    assert all(t.get_position()[1] == top+g['cell_y_mm'] for t in body)
    assert all(g['plot_y_mm']+p['offset_mm']-p['radius_mm'] >= g['plot_y_mm']-g['point_band_mm']/2
               for p in s['rows'][0]['points'])


def test_text_cannot_enter_lower_mark_band(tmp_path):
    info, frames = fixture()
    info['columns'] = [{'field': 'label', 'label': 'G', 'role': 'row_name', 'fontsize': 12}]
    with pytest.raises(ValueError, match='text/layout overflow'):
        draw(info, frames, tmp_path)
    info, frames = fixture()
    info['forest'] = {'geometry': {'cell_y_mm': 3}}
    with pytest.raises(ValueError, match='text must be above'):
        solve_forest(info, frames)


def test_swiss_reference_grid_palette_typography_and_debug(tmp_path):
    info, frames = fixture('box')
    info['title'] = {'text': 'Swiss comparison'}
    info['meta'] = {'number': '01', 'year': '2026', 'place': 'ADELAIDE'}
    info['debug'] = True
    fig = draw(info, frames, tmp_path)
    s = fig.forest_layout
    g = s['geometry']
    assert g['grid_columns'] == 12 and len(g['grid_lines_mm']) == 13
    for column in s['columns']:
        assert column['measured_width_mm'] == column['width_mm']
        assert 'grid_span' not in column
    assert s['typography']['family'] == 'Helvetica'
    assert s['typography']['title_pt'] == 9
    assert s['style']['rows']['fills'] == ['none']
    assert s['style']['scatter']['classes']['outlier']['color'] == '#DA291C'
    assert s['style']['box']['alpha'] == 1
    assert len(fig.axlogo.images) == 1  # shared Jarvis-HEP icon
    assert all(t.get_ha() == 'left' or (t.get_ha() == 'center' and t.get_text().endswith('%')) for ax in fig.fig.axes for t in ax.texts)
    fig.fig.canvas.draw()
    legend_texts = fig.axes['ax'].get_legend().get_texts()
    for artist in legend_texts:
        x = artist.get_window_extent().x0/fig.fig.dpi*25.4
        assert min(abs(x-line) for line in g['grid_lines_mm']) < .03 or x == pytest.approx(g['grid_lines_mm'][4]+g['legend_point_key_mm']+.6*5*25.4/72, abs=.03)
    original = deepcopy(s)
    from jarvisplot.Figure.design_runtime import draw_design_reference
    draw_design_reference(fig)
    overlay = fig.fig.axes[-1]
    assert overlay.get_label() == 'forest-design-reference'
    assert sum((line.get_gid() or '').startswith('forest-grid-') for line in overlay.lines) == 13
    assert len(overlay.patches) == 2
    assert s == original  # references never alter data, packing, or layout
    draw_design_reference(fig)
    assert len(fig.fig.axes) == 3  # overlay is idempotent


def test_swiss_debug_switches_and_bold_font_face(tmp_path):
    info, frames = fixture()
    info['debug'] = {'forest_grid': {'columns': {'show': False}, 'bands': {'show': False}}}
    fig = draw(info, frames, tmp_path)
    from jarvisplot.Figure.design_runtime import draw_design_reference
    draw_design_reference(fig)
    overlay = fig.fig.axes[-1]
    assert not overlay.patches
    assert not any((line.get_gid() or '').startswith('forest-grid-') for line in overlay.lines)
    from jarvisplot.forest_text import font_path
    from matplotlib.ft2font import FT2Font
    assert FT2Font(font_path('Helvetica','normal')).style_name == 'Regular'
    assert FT2Font(font_path('Helvetica','bold')).style_name == 'Bold'


def test_swiss_legend_cannot_overflow_modular_span(tmp_path):
    info, frames = fixture()
    info['frame'] = {'ax': {'legend': {'ncols': 6}}}
    with pytest.raises(ValueError, match='legend item exceeds.*modular span'):
        draw(info, frames, tmp_path)


def test_centered_percentage_end_ticks_keep_coordinates_and_fit_canvas(tmp_path):
    info, frames = fixture()
    frames['rows'] = frames['rows'].astype({'lower': float, 'estimate': float, 'upper': float})
    frames['rows'].loc[0, ['lower','estimate','upper']] = [.6, .8, 1]
    frames['points']['x'] = .8
    info['x'] = {'lim': [-100000, 100000]}
    fig = draw(info, frames, tmp_path)
    g = fig.forest_layout['geometry']
    assert g['margin_right_mm'] > g['requested_margin_right_mm'] == 2
    labels = [t for t in fig.axes['ax'].texts if t.get_text() == '100000%']
    assert len(labels) == 2
    assert g['margin_left_mm'] > g['requested_margin_left_mm'] == 2
    assert all(t.get_ha() == 'center' and t.get_position()[0] == g['plot_right_mm'] for t in labels)
    all_ticks = [t for t in fig.axes['ax'].texts if t.get_text() in ['-100000%', '0%', '100000%']]
    assert len(all_ticks) == 6 and all(t.get_ha() == 'center' for t in all_ticks)
    fig.fig.canvas.draw()
    assert all(t.get_window_extent().x1 < fig.fig.bbox.x1 for t in labels)


def test_badge_follows_grid_when_margins_change(tmp_path):
    info, frames = fixture()
    info['forest'] = {'geometry': {'margin_left_mm': 4, 'margin_right_mm': 5}}
    fig = draw(info, frames, tmp_path)
    g = fig.forest_layout['geometry']
    bbox = fig.axlogo.get_position()
    assert bbox.x0*82.5 == pytest.approx(g['grid_lines_mm'][0])
    assert bbox.x1*82.5 == pytest.approx(g['grid_lines_mm'][1])


def test_compact_columns_right_alignment_and_shared_vertical_grid(tmp_path):
    config, bag = validate_file(str(EXAMPLE/'real-demo.yaml'))
    assert bag.ok
    info = config['Figures'][0]
    frames = load_forest_tables(config, base_dir=str(EXAMPLE))
    fig = draw(info, frames, tmp_path)
    s = fig.forest_layout
    g = s['geometry']
    _, w, p, e, n, cv = s['columns']
    assert p['left_mm']-w['left_mm']-w['width_mm'] == pytest.approx(.8)
    assert e['left_mm']-p['left_mm']-p['width_mm'] == pytest.approx(.8)
    assert cv['left_mm']+cv['width_mm'] == pytest.approx(g['grid_lines_mm'][-1])
    assert n['left_mm']+n['width_mm']+.8 == pytest.approx(cv['left_mm'])
    assert n['align'] == cv['align'] == 'right'
    headings = [a for a in fig.axes['ax'].texts if a.get_text() in ('n','CV [%]')]
    assert all(a.get_ha() == 'right' for a in headings)
    lines = fig.axes['ax'].lines
    assert not any((a.get_gid() or '').startswith('forest-row-separator-') for a in lines)
    top_axes = [a for a in lines if a.get_gid() == 'forest-top-axis']
    assert len(top_axes) == 1
    assert list(top_axes[0].get_ydata()) == [g['top_axis_y_mm']]*2
    grid = [a for a in lines if (a.get_gid() or '').startswith('forest-numerical-grid-')]
    assert len(grid) == 5
    base = g['header_mm']+s['row_count']*g['row_pitch_mm']
    for line, value in zip(grid, [-40,-20,0,20,40]):
        assert list(line.get_xdata()) == [x_to_mm(value,s['shared_domain'],g)]*2
        assert list(line.get_ydata()) == [g['top_axis_y_mm'],base]
    assert all(a is top_axes[0] or not (list(a.get_xdata()) == [g['plot_left_mm'],g['plot_right_mm']] and len(set(a.get_ydata())) == 1) for a in lines)


@pytest.mark.parametrize('grid,expected_grid_count', [
    ({'show': True, 'divisions': 4}, 5),
    ({'show': True, 'divisions': 3}, 5),
    ({'show': False, 'divisions': 4}, 0),
])
def test_axis_tick_marks_do_not_follow_grid_settings(grid, expected_grid_count, tmp_path):
    info, frames = fixture('box')
    info['x'] = {'lim': [0, 2]}
    info['forest'] = {'grid': grid}
    fig = draw(info, frames, tmp_path)
    layout = fig.forest_layout
    geometry = layout['geometry']
    expected_x = [x_to_mm(value, layout['shared_domain'], geometry) for value in [0, 1, 2]]
    tick_lines = [line for line in fig.axes['ax'].lines
                  if (line.get_gid() or '').startswith('forest-axis-tick-')]
    assert len(tick_lines) == 3
    assert [line.get_xdata()[0] for line in tick_lines] == pytest.approx(expected_x)
    assert all(list(line.get_ydata()) == [geometry['top_axis_y_mm']-.5, geometry['top_axis_y_mm']]
               for line in tick_lines)
    grid_lines = [line for line in fig.axes['ax'].lines
                  if (line.get_gid() or '').startswith('forest-numerical-grid-')]
    assert len(grid_lines) == expected_grid_count
    assert layout['verification']['axis_tick_count'] == 3
    assert layout['verification']['grid_line_count'] == expected_grid_count


def test_six_shared_ticks_align_marks_and_labels_in_png_and_pdf(tmp_path):
    shared_ticks = [.61, .62, .63, .64, .65, .66]
    info, frames = fixture('box')
    for field, value in zip(['lower', 'q1', 'median', 'estimate', 'q3', 'upper'],
                            [.615, .625, .635, .635, .645, .655]):
        frames['rows'][field] = [value]
    frames['points']['x'] = .635
    info['x'] = {'lim': [.61, .66], 'ticks': shared_ticks}
    info['forest'] = {'axis': {'tick_suffix': ''}, 'grid': {'show': True, 'divisions': 4}}
    fig = draw(info, frames, tmp_path)
    layout = fig.forest_layout
    geometry = layout['geometry']
    expected_x = [x_to_mm(value, layout['shared_domain'], geometry) for value in shared_ticks]
    tick_lines = [line for line in fig.axes['ax'].lines
                  if (line.get_gid() or '').startswith('forest-axis-tick-')]
    assert len(tick_lines) == 6
    assert [line.get_xdata()[0] for line in tick_lines] == pytest.approx(expected_x)
    top_labels = [label for label in fig.axes['ax'].texts
                  if label.get_position()[1] == geometry['top_tick_y_mm']]
    assert [label.get_text() for label in top_labels] == [f'{value:g}' for value in shared_ticks]
    assert [label.get_position()[0] for label in top_labels] == pytest.approx(expected_x)
    assert layout['verification']['axis_tick_count'] == 6
    assert layout['verification']['tick_label_count'] == 12
    # Six ticks plus five grid divisions share the two endpoints: nine lines.
    grid_lines = [line for line in fig.axes['ax'].lines
                  if (line.get_gid() or '').startswith('forest-numerical-grid-')]
    assert len(grid_lines) == 9
    assert len({round(line.get_xdata()[0], 12) for line in grid_lines}) == 9
    fig.fmts = ['png', 'pdf']
    fig.dpi = 300
    fig.savefig()
    png = tmp_path / f'{fig.name}.png'
    pdf = tmp_path / f'{fig.name}.pdf'
    assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert pdf.read_bytes().startswith(b'%PDF-')
    # Inspect the saved raster where only the short ticks can contribute ink.
    # This catches exported extra strokes even if labels or receipts look right.
    from PIL import Image
    with Image.open(png) as output:
        raster_scale = output.width/layout['width_mm']
        gray = output.convert('L')
        y0 = math.ceil((geometry['top_axis_y_mm']-.4)*raster_scale)
        y1 = math.floor((geometry['top_axis_y_mm']-.15)*raster_scale)
        ink_columns = [x for x in range(output.width)
                       if min(gray.getpixel((x, y)) for y in range(y0, y1+1)) < 180]
        clusters = []
        for x in ink_columns:
            if not clusters or x > clusters[-1][-1]+1:
                clusters.append([x])
            else:
                clusters[-1].append(x)
    assert len(clusters) == 6
    actual_x = [(cluster[0]+cluster[-1])/2/raster_scale for cluster in clusters]
    assert actual_x == pytest.approx(expected_x, abs=2/raster_scale)


def test_exported_pdf_six_tick_vector_paths_and_top_labels(tmp_path):
    fitz = pytest.importorskip('fitz', reason='PDF vector audit requires optional PyMuPDF')
    info, frames = fixture('box')
    frames['rows'] = frames['rows'].astype({key: float for key in ('lower', 'q1', 'median', 'q3', 'upper')})
    frames['rows'].loc[0, ['lower', 'q1', 'median', 'q3', 'upper']] = [.615, .625, .635, .645, .655]
    frames['points']['x'] = .635
    info['x'] = {'lim': [.61, .66], 'ticks': [.61, .62, .63, .64, .65, .66]}
    info['forest'] = {'axis': {'tick_suffix': ''}, 'grid': {'divisions': 4}}
    fig = draw(info, frames, tmp_path)
    fig.fmts, fig.dpi = ['pdf'], 300
    fig.savefig()
    layout = fig.forest_layout
    g = layout['geometry']
    expected = [x_to_mm(value, layout['shared_domain'], g) for value in layout['shared_ticks']]
    with fitz.open(tmp_path/f'{fig.name}.pdf') as document:
        page = document[0]
        mm = 25.4/72
        strokes = []
        for path in page.get_drawings():
            for item in path['items']:
                if item[0] != 'l':
                    continue
                p, q = item[1:]
                if (abs(p.x-q.x)*mm < .001 and abs(abs(p.y-q.y)*mm-.5) < .002 and
                        abs(max(p.y, q.y)*mm-g['top_axis_y_mm']) < .002):
                    strokes.append(p.x*mm)
        assert len(strokes) == 6
        assert sorted(strokes) == pytest.approx(expected, abs=.002)
        words = page.get_text('words')
        for value, target in zip(layout['shared_ticks'], expected):
            labels = [word for word in words if word[4] == f'{value:g}' and
                      abs((word[1]+word[3])/2*mm-g['top_tick_y_mm']) < 1.1]
            assert len(labels) == 1
            assert (labels[0][0]+labels[0][2])/2*mm == pytest.approx(target, abs=.12)


def test_grid_coordinate_merge_ignores_floating_point_overlap():
    from jarvisplot.Figure.forest_runtime import _merged_grid_values
    values = _merged_grid_values([.1, .3], [.1, .2+1e-15, .3], 2)
    assert len(values) == 3
    assert values == pytest.approx([.1, .2, .3])


def test_logo_switch_and_title_block_schema(tmp_path):
    info, frames = fixture()
    info['frame'] = {'axlogo': False}
    assert len(draw(info, frames, tmp_path).axes) == 1
    info['title'] = 'old unstructured title'
    from jarvisplot.schema_catalog import config_validator
    assert list(config_validator().iter_errors({'DataSet': [], 'Figures': [info]}))


def test_parquet_preserves_null_box_and_rejects_nan(tmp_path):
    import polars as pl
    box = tmp_path/'rows.parquet'
    points = tmp_path/'points.csv'
    points.write_text('row,id,x\n001,01,1\n')
    fields = {'row': ['001'], **{k: [None] for k in ('lower', 'q1', 'median', 'q3', 'upper')}}
    pl.DataFrame(fields).write_parquet(box)
    cfg = {'DataSet': [{'name':'rows','path':str(box),'type':'parquet'}, {'name':'points','path':str(points),'type':'csv'}],
           'Figures': [{'name':'f','type':'forest','glyph':'box','data':'rows','points':'points'}]}
    assert solve_forest(cfg['Figures'][0], load_forest_tables(cfg, base_dir=str(tmp_path)))['point_count'] == 1
    fields['lower'] = [float('nan')]
    pl.DataFrame(fields).write_parquet(box)
    with pytest.raises(ValueError, match='nonfinite'):
        solve_forest(cfg['Figures'][0], load_forest_tables(cfg, base_dir=str(tmp_path)))


def test_grouped_footer_legend_shares_height_with_standard_icon(tmp_path):
    info, frames = fixture('box')
    classes = ['normal', 'outlier', 'unrated', 'singleton']
    frames['points'] = pd.DataFrame([
        {'row': '001', 'id': str(i), 'x': .7+i*.2, 'class': cls}
        for i, cls in enumerate(classes)
    ])
    info['frame'] = {'ax': {'legend': {'order': ['singleton', 'outlier']}}}
    fig = draw(info, frames, tmp_path)
    legend = fig.axes['ax'].get_legend()
    columns = legend._legend_handle_box.get_children()
    labels = [[item.get_children()[-1]._text.get_text() for item in column.get_children()] for column in columns]
    assert labels == [['Box + whiskers'], ['Singleton', 'Potential outlier', 'Ordinary', 'Unrated']]
    assert fig.forest_layout['geometry']['footer_mm'] == 20
    assert len(fig.axlogo.images) == 1
    from PIL import Image
    import numpy as np
    icon = Path(__file__).resolve().parents[1]/'jarvisplot/cards/icons/JarvisHEP.png'
    with Image.open(icon) as source:
        assert np.array_equal(fig.axlogo.images[0].get_array(), np.asarray(source.convert('RGBA')))
    renderer = fig.fig._get_renderer()
    left, right = [column.get_window_extent(renderer) for column in columns]
    logo = fig.axlogo.get_window_extent(renderer)
    assert left.y1 == pytest.approx(right.y1)
    point_items = columns[1].get_children()
    label_x = [item.get_children()[-1]._text.get_window_extent(renderer).x0 for item in point_items]
    assert label_x == pytest.approx([label_x[0]]*4)
    for item in point_items:
        marker = item.get_children()[0].get_children()[0]
        centre = marker.get_offset_transform().transform(marker.get_offsets()[0])[0]/fig.fig.dpi*25.4
        assert centre == pytest.approx(fig.forest_layout['geometry']['grid_lines_mm'][4]+.6, abs=.03)
    assert fig.fig.get_facecolor() == (1, 1, 1, 1)
    assert not any(p.get_zorder() == 0 for p in fig.axes['ax'].patches)
    assert not any(t.get_text().startswith('02') for t in fig.axes['ax'].texts)
    grid = fig.forest_layout['geometry']['grid_lines_mm']
    assert left.x0/fig.fig.dpi*25.4 == pytest.approx(grid[1], abs=.03)
    assert right.x0/fig.fig.dpi*25.4 == pytest.approx(grid[4], abs=.03)
    assert logo.y1 > right.y0 and logo.y0 < right.y1
    assert not logo.overlaps(left) and not logo.overlaps(right)
    assert all(not t.get_window_extent(renderer).overlaps(right) for t in fig.axlogo.texts)


@pytest.mark.parametrize('group', ['glyph', 'points'])
def test_grouped_legend_retains_empty_semantic_column(group, tmp_path):
    info, frames = fixture()
    if group == 'glyph':
        info.pop('points')
    else:
        for key in ['estimate', 'lower', 'upper']:
            frames['rows'][key] = None
    fig = draw(info, frames, tmp_path)
    columns = fig.axes['ax'].get_legend()._legend_handle_box.get_children()
    assert len(columns) == 2
    occupied = columns[0 if group == 'glyph' else 1]
    assert occupied.get_children()[0].get_children()[-1]._text.get_text().replace('\n', ' ') == ('Estimate + bounds' if group == 'glyph' else 'Ordinary')


def test_yaml_description_moves_to_third_footer_column_without_losing_text(tmp_path):
    config, bag = validate_file(str(EXAMPLE/'real-demo.yaml'))
    assert bag.ok
    info = config['Figures'][0]
    frames = load_forest_tables(config, base_dir=str(EXAMPLE))
    fig = draw(info, frames, tmp_path)
    g = fig.forest_layout['geometry']
    base = g['header_mm']+97*g['row_pitch_mm']
    lines = [t for t in fig.axes['ax'].texts
             if t.get_position()[0] == g['grid_lines_mm'][7] and t.get_position()[1] >= base]
    assert lines
    assert ' '.join(' '.join(t.get_text() for t in lines).split()) == ' '.join(info['description']['text'].split())
    assert lines[0].get_position()[1] == base+g['description_y_mm']
    assert all(t.get_position()[1] >= base for t in lines)
    assert fig.forest_layout['description'] == info['description']
    assert g['header_mm'] == g['footer_mm'] == 20
    assert fig.forest_layout['height_mm'] == 573.5


@pytest.mark.parametrize('dpi,formats', [(96, ['svg', 'png']), (220, ['png', 'svg']), (600, ['svg', 'png'])])
def test_exported_svg_and_png_keep_physical_legend_columns(dpi, formats, tmp_path):
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from xml.etree import ElementTree as ET
    info, frames = fixture('box')
    info['description'] = {'text': 'W = sidewt; P = P8; E = EMA. Outliers: 1.5 x IQR, n >= 6.\nBox = IQR; whiskers = non-outlier extent. CV includes all points.'}
    info['debug'] = True
    fig = draw(info, frames, tmp_path)
    g = fig.forest_layout['geometry']
    fig.fig.set_dpi(dpi)
    canvas = FigureCanvasAgg(fig.fig)
    fig.fmts = formats
    fig.dpi = dpi
    fig.savefig()
    # Check real exported SVG coordinates, not only the initial Agg canvas.
    texts = {''.join(node.itertext()): node for node in ET.parse(tmp_path/f'{fig.name}.svg').iter('{http://www.w3.org/2000/svg}text')}
    assert float(texts['Box + whiskers'].attrib['x'])*25.4/72 == pytest.approx(g['grid_lines_mm'][2], abs=.03)
    point_text_x = g['grid_lines_mm'][4]+g['legend_point_key_mm']+.6*5*25.4/72
    assert float(texts['Ordinary'].attrib['x'])*25.4/72 == pytest.approx(point_text_x, abs=.03)
    description = next(node for value, node in texts.items() if value.startswith('W = sidewt;'))
    assert float(description.attrib['x'])*25.4/72 == pytest.approx(g['grid_lines_mm'][7], abs=.03)
    assert float(texts['Ordinary'].attrib['x']) < float(description.attrib['x'])
    for value in fig.forest_layout['shared_ticks']:
        tick = texts[f'{value:g}%']
        assert 'text-anchor: middle' in tick.attrib['style']
        assert float(tick.attrib['x'])*25.4/72 == pytest.approx(x_to_mm(value, fig.forest_layout['shared_domain'], g), abs=.03)
    # A vector save must not leave raster column widths stale for later draws.
    with mpl.rc_context({'text.hinting': 'none'}):
        canvas.draw()
    columns = fig.axes['ax'].get_legend()._legend_handle_box.get_children()
    starts = [column.get_window_extent(canvas.get_renderer()).x0/dpi*25.4 for column in columns]
    assert starts == pytest.approx([g['grid_lines_mm'][1], g['grid_lines_mm'][4]], abs=.03)


def layered_fixture():
    """Supplied distributions and summaries: nothing is inferred by forest."""
    info, frames = fixture('box')
    info['intervals'] = 'summaries'
    info['columns'] = [{'field': 'label', 'label': 'Input set', 'role': 'row_name'}]
    info['x'] = {'lim': [0, 3], 'ticks': [0, .5, 1, 1.5, 2, 2.5, 3]}
    info['glyph_label'] = 'Median [Q1, Q3]'
    frames['rows'] = frames['rows'].astype({k: float for k in ('lower', 'q1', 'median', 'q3', 'upper')})
    frames['rows'].loc[0, ['lower', 'q1', 'median', 'q3', 'upper']] = [.6, .7, .8, .9, 1.05]
    frames['rows']['stats'] = '0.8000 [0.7000, 0.9000]'
    frames['points'] = pd.DataFrame([
        {'row': '001', 'id': f'{i:02}', 'x': .64+i*.04, 'class': 'sample'}
        for i in range(10)
    ])
    frames['summaries'] = pd.DataFrame([
        {'row': '001', 'id': '01', 'estimate': .78, 'lower': .71, 'upper': .85,
         'class': 'first', 'stats': '0.7800 ± 0.0700'},
        {'row': '001', 'id': '02', 'estimate': .8, 'lower': .73, 'upper': .87,
         'class': 'second', 'stats': '0.8000 ± 0.0700'},
        {'row': '001', 'id': '03', 'estimate': .76, 'lower': .65, 'upper': .87,
         'class': 'third', 'stats': '0.7600 ± 0.1100'},
    ])
    info['forest'] = {
        'geometry': {'point_band_mm': 10, 'row_band_mm': 13, 'row_pitch_mm': 13.3,
                     'plot_y_mm': 8, 'cell_y_mm': 1.35, 'glyph_offset_mm': -2.5,
                     'value_column_mm': 0, 'footer_mm': 40},
        'box': {'height_mm': 1.3, 'annotation': {'field': 'stats', 'position': 'row_text',
                                               'fontsize': 4.3, 'color': '#0072B2'}},
        'scatter': {'gap_mm': .08, 'classes': {'sample': {'label': 'Fits',
                                                       'color': '#0072B2', 'diameter_mm': .65,
                                                       'marker': 'o', 'fill': True}}},
        'overlays': {
            'annotation': {'field': 'stats', 'fontsize': 4.3, 'gap_mm': .5},
            'legend': {'label': 'Mean ± SD', 'color': '#000000'},
            'classes': {
                'first': {'label': 'First group', 'color': '#0072B2', 'alpha': .9,
                          'diameter_mm': .5, 'cap_height_mm': .5, 'offset_mm': 0},
                'second': {'label': 'Second group', 'color': '#D55E00', 'alpha': .4,
                           'diameter_mm': .5, 'cap_height_mm': .5, 'offset_mm': 1.6,
                           'linestyle': '--'},
                'third': {'label': 'Third group', 'color': '#009E73', 'alpha': .4,
                          'diameter_mm': .5, 'cap_height_mm': .5, 'offset_mm': 3.2},
            },
        },
    }
    return info, frames


def _artist(fig, gid):
    matches = [a for a in fig.axes['ax'].get_children() if a.get_gid() == gid]
    assert len(matches) == 1, (gid, len(matches))
    return matches[0]


def test_layered_forest_preserves_every_observation_box_bound_and_interval(tmp_path):
    info, frames = layered_fixture()
    fig = draw(info, frames, tmp_path)
    layout = fig.forest_layout
    row = layout['rows'][0]
    g = layout['geometry']
    to_x = lambda value: x_to_mm(value, layout['shared_domain'], g)
    assert (layout['row_count'], layout['point_count'], layout['interval_count']) == (1, 10, 3)
    assert {p['id']: p['x'] for p in row['points']} == dict(zip(frames['points']['id'], frames['points']['x']))
    box = _artist(fig, 'forest-primary-box-001')
    assert [box.get_x(), box.get_x()+box.get_width()] == pytest.approx([to_x(.7), to_x(.9)])
    primary_y = row['top_mm']+g['plot_y_mm']-2.5
    assert box.get_y()+box.get_height()/2 == pytest.approx(primary_y)
    assert list(_artist(fig, 'forest-primary-whisker-001').get_xdata()) == pytest.approx([to_x(.6), to_x(1.05)])
    assert list(_artist(fig, 'forest-primary-median-001').get_xdata()) == pytest.approx([to_x(.8)]*2)
    for p in row['points']:
        actual = _artist(fig, 'forest-point-001-'+p['id']).get_offsets()[0]
        assert tuple(actual) == pytest.approx((to_x(p['x']), row['top_mm']+g['plot_y_mm']+p['offset_mm']))
        assert abs(p['offset_mm']+2.5) <= g['point_band_mm']/2
    for expected, actual in zip(frames['summaries'].to_dict('records'), row['intervals']):
        assert {key: actual[key] for key in ('id', 'estimate', 'lower', 'upper', 'class', 'annotation')} == {
            **{key: expected[key] for key in ('id', 'estimate', 'lower', 'upper', 'class')},
            'annotation': expected['stats'],
        }
        y = row['top_mm']+g['plot_y_mm']+actual['offset_mm']
        whisker = _artist(fig, 'forest-overlay-001-'+actual['id'])
        assert list(whisker.get_xdata()) == pytest.approx([to_x(expected['lower']), to_x(expected['upper'])])
        assert list(whisker.get_ydata()) == pytest.approx([y, y])
        for end in ('lower', 'upper'):
            cap = _artist(fig, f"forest-overlay-cap-001-{actual['id']}-{end}")
            assert list(cap.get_xdata()) == pytest.approx([to_x(expected[end])]*2)
        assert whisker.get_color() == actual['style']['color']
        assert whisker.get_alpha() == actual['style']['alpha']
        label = _artist(fig, 'forest-interval-annotation-001-'+actual['id'])
        assert label.get_text() == expected['stats']
        assert label.get_color() == actual['style']['color']
        assert label.get_position()[1] == pytest.approx(y)
    assert _artist(fig, 'forest-box-annotation-001').get_text() == frames['rows']['stats'][0]
    assert layout['verification']['all_intervals_once']
    assert layout['verification']['numeric_x_preserved']


def test_interval_mappings_and_literal_csv_ids_annotations_are_demanded(tmp_path):
    from jarvisplot.forest_contract import source_columns
    info, frames = layered_fixture()
    remap = {'row': 'study', 'id': 'summary_id', 'estimate': 'given_mean',
             'lower': 'given_min', 'upper': 'given_max', 'class': 'population'}
    info['interval_fields'] = remap
    frames['summaries'] = frames['summaries'].rename(columns=remap).assign(metadata='not requested')
    demanded = next(entry for entry in source_columns(info, required=True) if entry[0] == 'summaries')
    assert set(remap.values()) | {'stats'} <= demanded[1]
    assert {'study', 'summary_id', 'population', 'stats'} <= demanded[2]
    source_files = []
    for name, frame in frames.items():
        path = tmp_path / f'{name}.csv'
        frame.to_csv(path, index=False)
        source_files.append({'name': name, 'path': str(path), 'type': 'csv'})
    config = {'DataSet': source_files, 'Figures': [info]}
    bag = validate_config(config)
    assert bag.ok, bag.render_human()
    loaded = load_forest_tables(config, base_dir=str(tmp_path))
    assert loaded['summaries']['study'].tolist() == ['001']*3
    assert loaded['summaries']['summary_id'].tolist() == ['01', '02', '03']
    assert 'metadata' not in loaded['summaries']
    assert loaded['summaries']['stats'].tolist() == frames['summaries']['stats'].tolist()
    layout = solve_forest(info, loaded)
    assert [o['id'] for o in layout['rows'][0]['intervals']] == ['01', '02', '03']


@pytest.mark.parametrize('blank_primary', [True, False])
def test_intervals_only_rows_render_without_primary_legend_or_scatter(blank_primary, tmp_path):
    info, frames = layered_fixture()
    info.pop('points')
    info['glyph'] = 'interval'
    info['forest']['geometry']['glyph_offset_mm'] = 0
    info['forest']['box'].pop('annotation')
    frames['rows'] = frames['rows'][['row', 'label']]
    if blank_primary:
        frames['rows'] = frames['rows'].assign(estimate=None, lower=None, upper=None)
    fig = draw(info, frames, tmp_path)
    layout = fig.forest_layout
    assert (layout['point_count'], layout['interval_count']) == (0, 3)
    assert all(value is None for value in layout['rows'][0]['coordinates'].values())
    assert not any((a.get_gid() or '').startswith(('forest-primary-', 'forest-point-'))
                   for a in fig.axes['ax'].get_children())
    labels = [text.get_text().replace('\n', ' ') for text in fig.axes['ax'].get_legend().get_texts()]
    assert labels == ['Mean ± SD', 'First group', 'Second group', 'Third group']


@pytest.mark.parametrize('field,value,error', [
    ('row', 'unknown', 'foreign key'), ('row', '', 'nonempty'),
    ('id', '', 'nonempty'), ('class', 'unconfigured', 'unknown.*class|unknown overlay'),
    ('estimate', float('inf'), 'nonfinite'), ('estimate', True, 'invalid'),
    ('lower', float('nan'), 'nonfinite'), ('upper', '', 'blank|finite'),
    ('estimate', .9, 'unordered'), ('lower', .79, 'unordered'), ('upper', .77, 'unordered'),
])
def test_interval_bad_data_is_rejected_with_context(field, value, error):
    info, frames = layered_fixture()
    frames['summaries'][field] = frames['summaries'][field].astype(object)
    frames['summaries'].loc[0, field] = value
    with pytest.raises(ValueError, match=error):
        solve_forest(info, frames)


def test_interval_duplicate_source_foreign_keys_domain_and_offset_errors():
    info, frames = layered_fixture()
    frames['summaries'] = pd.concat([frames['summaries'], frames['summaries'].iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match='duplicate.*interval'):
        solve_forest(info, frames)
    info, frames = layered_fixture()
    info['intervals'] = 'absent'
    with pytest.raises(ValueError, match='unknown|unavailable'):
        solve_forest(info, frames)
    info, frames = layered_fixture()
    frames['summaries'].loc[0, 'upper'] = 3.1
    with pytest.raises(ValueError, match='excludes'):
        solve_forest(info, frames)
    info, frames = layered_fixture()
    info['forest']['overlays']['classes']['third']['offset_mm'] = 6
    with pytest.raises(ValueError, match='plot band|point band'):
        solve_forest(info, frames)


def test_interval_annotation_and_partial_primary_columns_are_validated():
    info, frames = layered_fixture()
    frames['summaries'] = frames['summaries'].drop(columns='stats')
    with pytest.raises(ValueError, match='missing fields.*stats'):
        solve_forest(info, frames)
    info, frames = layered_fixture()
    frames['rows'] = frames['rows'].drop(columns=['lower', 'upper'])
    with pytest.raises(ValueError, match='partial|missing fields'):
        solve_forest(info, frames)
    info, frames = layered_fixture()
    info['glyph'] = 'interval'
    with pytest.raises(ValueError, match='requires glyph.*box'):
        solve_forest(info, frames)


def test_automatic_domain_includes_supplied_overlay_endpoints_without_annotation():
    info, frames = layered_fixture()
    info.pop('x')
    info['forest']['overlays'].pop('annotation')
    frames['summaries'] = frames['summaries'].drop(columns='stats')
    frames['summaries'].loc[0, ['lower', 'upper']] = [-.2, 3.2]
    layout = solve_forest(info, frames)
    assert layout['shared_domain'][0] <= -.2
    assert layout['shared_domain'][1] >= 3.2
    assert all(interval['annotation'] == '' for interval in layout['rows'][0]['intervals'])


@pytest.mark.parametrize('ticks,error', [
    ([0], 'at least|two|2|too short'), ([0, 0, 1], 'ascending|increasing'),
    ([0, 2, 1], 'ascending|increasing'), ([-.1, 1], 'domain|inside|within'),
    ([0, 3.1], 'domain|inside|within'), ([0, float('nan')], 'nonfinite|finite'),
    ([0, float('inf')], 'nonfinite|finite'), ([False, 1], 'invalid'),
])
def test_shared_explicit_ticks_reject_invalid_coordinates(ticks, error):
    info, frames = fixture()
    info['x'] = {'lim': [0, 2], 'ticks': ticks}
    with pytest.raises(ValueError, match=error):
        solve_forest(info, frames)


@pytest.mark.parametrize('divisions,show', [(4, True), (5, True), (7, True), (4, False)])
def test_explicit_ticks_remain_identical_across_grid_options(divisions, show, tmp_path):
    info, frames = fixture()
    ticks = [0, .4, .8, 1.2, 1.6, 2]
    info['x'] = {'lim': [0, 2], 'ticks': ticks}
    info['forest'] = {'grid': {'divisions': divisions, 'show': show}, 'axis': {'tick_suffix': ''}}
    fig = draw(info, frames, tmp_path)
    g = fig.forest_layout['geometry']
    marks = [a for a in fig.axes['ax'].lines if (a.get_gid() or '').startswith('forest-axis-tick-')]
    assert len(marks) == len(ticks)
    assert [a.get_xdata()[0] for a in marks] == pytest.approx([x_to_mm(v, [0, 2], g) for v in ticks])
    assert all(list(a.get_ydata()) == [g['top_axis_y_mm']-.5, g['top_axis_y_mm']] for a in marks)
    labels = [a for a in fig.axes['ax'].texts if a.get_position()[1] == g['top_tick_y_mm']]
    assert [a.get_text() for a in labels] == [f'{v:g}' for v in ticks]
    assert [a.get_position()[0] for a in labels] == pytest.approx([a.get_xdata()[0] for a in marks])
    grids = [a for a in fig.axes['ax'].lines if (a.get_gid() or '').startswith('forest-numerical-grid-')]
    grid_positions = sorted(a.get_xdata()[0] for a in grids)
    assert all(b-a > 1e-9 for a, b in zip(grid_positions, grid_positions[1:]))
    if not show:
        assert not grids
    elif divisions == 5:
        assert len(grids) == 6


def test_math_wrapping_preserves_formulas_paragraph_breaks_and_literal_default():
    from jarvisplot.forest_text import wrap_text, text_metrics_mm
    formula = r'$m_{t\bar t}^{\mathrm{gen}} < 355\,\mathrm{GeV}$'
    math_width = text_metrics_mm(formula, 4.5, 'Helvetica', 'normal', math=True)[0]
    assert math_width < text_metrics_mm(formula, 4.5, 'Helvetica', 'normal')[0]
    lines = wrap_text('First cut: '+formula+'\n\nSecond cut:\n'+formula,
                      4.5, 'Helvetica', 'normal', math_width+.1, math=True)
    assert lines.count(formula) == 2
    assert '' in lines
    assert all(line.count('$') % 2 == 0 for line in lines)
    assert 'Second cut:' in lines


def test_math_rows_cut_sections_legend_and_header_note_use_actual_layout(tmp_path):
    from jarvisplot.forest_text import text_metrics_mm
    info, frames = layered_fixture()
    formula = r'$\left|\rho_{ij}\right|_{\max} < 0.1$'
    frames['rows']['label'] = formula
    info['columns'][0]['math'] = True
    info['glyph_label'] = r'$Q_{0.50}[Q_{0.25},Q_{0.75}]$'
    info['forest']['legend_math'] = True
    info['forest']['geometry'].update(footer_mm=55, legend_start_columns=[2, 6],
                                    description_start_column=9, description_columns=4)
    info['forest']['overlays']['legend']['label'] = r'$\mathrm{mean}\pm\mathrm{SD}$'
    info['forest']['overlays']['classes']['third']['label'] = r'$2\times5$ CV'
    info['title'] = {'text': 'Provided summaries', 'fontsize': 8}
    info['header_note'] = {'text': 'MC entries: signal = 123; background = 456', 'fontsize': 4.5}
    cut = r'$m_{t\bar t}^{\mathrm{gen}} < 355\,\mathrm{GeV}$'
    info['description'] = {'text': 'Parton cuts:\n'+cut+'\n\nReco cuts:\n'+r'$p_T > 25\,\mathrm{GeV}$',
                           'fontsize': 4.1, 'math': True}
    fig = draw(info, frames, tmp_path)
    label = next(a for a in fig.axes['ax'].texts if a.get_text() == formula)
    assert label.get_parse_math()
    assert fig.forest_layout['columns'][0]['content_width_mm'] == pytest.approx(
        text_metrics_mm(formula, 6, 'Helvetica', 'normal', math=True, prop=label.get_fontproperties())[0])
    assert next(a for a in fig.axes['ax'].texts if a.get_text() == cut).get_parse_math()
    assert all(a.get_parse_math() for a in fig.axes['ax'].get_legend().get_texts())
    g = fig.forest_layout['geometry']
    headers = [a for a in fig.axes['ax'].texts
               if a.get_position()[1] < g['top_tick_y_mm'] and a.get_text()]
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig.fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    for i, first in enumerate(headers):
        for second in headers[i+1:]:
            assert not first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)), (
                first.get_text(), second.get_text())
    assert fig.forest_layout['verification']['text_and_legend_bounds'] == 'passed'


def test_shared_interval_format_legend_is_black_and_never_contains_row_values(tmp_path):
    info, frames = layered_fixture()
    fig = draw(info, frames, tmp_path)
    legend = fig.axes['ax'].get_legend()
    labels = [t.get_text().replace('\n', ' ') for t in legend.get_texts()]
    assert sorted(labels) == sorted(['Median [Q1, Q3]', 'Fits', 'Mean ± SD',
                                    'First group', 'Second group', 'Third group'])
    assert len(labels) == len(set(labels))
    assert not set(labels) & set(frames['summaries']['stats'])
    assert frames['rows']['stats'][0] not in labels
    format_index = labels.index('Mean ± SD')
    assert legend.legend_handles[format_index].get_color() == '#000000'
    for cls in info['forest']['overlays']['classes'].values():
        i = labels.index(cls['label'])
        assert legend.get_texts()[i].get_color() == cls['color']


@pytest.mark.parametrize('row_pitch', [13.3, 15, 18])
def test_distinct_row_pitches_preserve_physical_lanes_and_full_width_axis(row_pitch, tmp_path):
    info, frames = layered_fixture()
    info['forest']['geometry']['row_pitch_mm'] = row_pitch
    frames['rows'] = pd.concat([frames['rows'], frames['rows'].assign(row='002', label='Group B')], ignore_index=True)
    frames['points'] = pd.concat([frames['points'], frames['points'].assign(row='002')], ignore_index=True)
    frames['summaries'] = pd.concat([frames['summaries'], frames['summaries'].assign(row='002')], ignore_index=True)
    fig = draw(info, frames, tmp_path)
    layout = fig.forest_layout
    g = layout['geometry']
    assert layout['rows'][1]['top_mm']-layout['rows'][0]['top_mm'] == pytest.approx(row_pitch)
    assert g['plot_right_mm'] == g['value_right_mm']
    assert layout['plot_width_mm'] == pytest.approx(g['width_mm']-g['margin_left_mm']-g['margin_right_mm'])
    for rid in ('001', '002'):
        positions = [_artist(fig, f'forest-primary-box-{rid}').get_y()+.65,
                     *[_artist(fig, f'forest-overlay-{rid}-{oid}').get_ydata()[0] for oid in ('01', '02', '03')]]
        assert positions == sorted(positions)


def test_reserved_value_column_keeps_long_supplied_values_outside_axis(tmp_path):
    info, frames = layered_fixture()
    info['forest']['geometry']['value_column_mm'] = 25
    info['forest']['box']['annotation']['position'] = 'box'
    frames['rows']['stats'] = '0.8000 [0.7000, 0.9000]'
    fig = draw(info, frames, tmp_path)
    g = fig.forest_layout['geometry']
    assert g['value_right_mm']-g['plot_right_mm'] == 25
    assert _artist(fig, 'forest-box-annotation-001').get_position()[0] == g['value_right_mm']
    for oid in ('01', '02', '03'):
        assert _artist(fig, 'forest-interval-annotation-001-'+oid).get_position()[0] == g['value_right_mm']
    assert fig.forest_layout['verification']['text_and_legend_bounds'] == 'passed'


def test_near_axis_boundary_error_line_annotation_and_lane_collisions_are_detected(tmp_path):
    info, frames = layered_fixture()
    # There is less room than the measured value label plus its declared gap.
    # The line must never be silently shortened or covered to make it fit.
    frames['summaries'].loc[0, ['estimate', 'lower', 'upper']] = [2.85, 2.8, 2.99]
    frames['summaries'].loc[0, 'stats'] = 'A very long supplied mean and standard deviation annotation'
    with pytest.raises(ValueError, match='overflow|collid|overlap|annotation'):
        draw(info, frames, tmp_path)
    info, frames = layered_fixture()
    info['forest']['overlays']['classes']['second']['offset_mm'] = .1
    with pytest.raises(ValueError, match='collid|overlap|lane|offset|band'):
        draw(info, frames, tmp_path)
    info, frames = layered_fixture()
    info['forest']['overlays']['classes']['first']['offset_mm'] = -2.5
    with pytest.raises(ValueError, match='collid|overlap|lane|offset|band'):
        draw(info, frames, tmp_path)


def test_math_ink_overflow_and_header_overlap_fail_before_export(tmp_path):
    info, frames = layered_fixture()
    frames['rows']['label'] = r'$\frac{\sum_{i=1}^{100} x_i}{\sqrt{n}}$'
    info['columns'][0].update(math=True, fontsize=11)
    with pytest.raises(ValueError, match='overflow|text|band'):
        draw(info, frames, tmp_path)
    info, frames = layered_fixture()
    info['header_note'] = {'text': 'MC entry annotation '*20, 'fontsize': 8}
    info['title'] = {'text': 'Main title'}
    with pytest.raises(ValueError, match='overflow|overlap|header|note'):
        draw(info, frames, tmp_path)


def test_overlay_svg_contains_every_short_tick_and_supplied_line_once(tmp_path):
    from xml.etree import ElementTree as ET
    info, frames = layered_fixture()
    fig = draw(info, frames, tmp_path)
    fig.fmts = ['svg', 'png', 'pdf']
    fig.dpi = 300
    fig.savefig()
    root = ET.parse(tmp_path/f'{fig.name}.svg').getroot()
    gids = [node.attrib['id'] for node in root.iter() if 'id' in node.attrib]
    assert len([gid for gid in gids if gid.startswith('forest-axis-tick-')]) == 7
    assert len([gid for gid in gids if gid.startswith('forest-overlay-001-')]) == 3
    assert len([gid for gid in gids if gid.startswith('forest-point-001-')]) == 10
    assert len([gid for gid in gids if gid.startswith('forest-interval-annotation-001-')]) == 3
    for suffix, signature in [('png', b'\x89PNG\r\n\x1a\n'), ('pdf', b'%PDF-')]:
        assert (tmp_path/f'{fig.name}.{suffix}').read_bytes().startswith(signature)


@pytest.mark.parametrize('point_class', ['@interval:first', '@interval-summary'])
@pytest.mark.parametrize('shared_format', [False, True])
def test_literal_point_class_names_cannot_alias_overlay_legend_objects(point_class, shared_format, tmp_path):
    info, frames = layered_fixture()
    frames['points']['class'] = point_class
    info['forest']['scatter']['classes'] = {point_class: info['forest']['scatter']['classes']['sample']}
    if not shared_format:
        info['forest']['overlays'].pop('legend')
    fig = draw(info, frames, tmp_path)
    layout = fig.forest_layout
    legend_objects = [item['legend']['object'] for item in layout['legend_layers']]
    assert len(set(legend_objects)) == len(legend_objects)
    assert point_class in legend_objects
    assert all(interval['legend_object'] != point_class for interval in layout['rows'][0]['intervals'])
    if shared_format:
        assert layout['interval_summary_object'] != point_class
    expected = ['Median [Q1, Q3]', 'Fits', 'First group', 'Second group', 'Third group']
    if shared_format:
        expected.append('Mean ± SD')
    actual = [text.get_text().replace('\n', ' ') for text in fig.axes['ax'].get_legend().get_texts()]
    assert sorted(actual) == sorted(expected)
    assert len([a for a in fig.axes['ax'].get_children()
                if (a.get_gid() or '').startswith('forest-point-001-')]) == 10
    assert len([a for a in fig.axes['ax'].get_children()
                if (a.get_gid() or '').startswith('forest-overlay-001-')]) == 3


def test_tall_math_footer_lines_cannot_overlap_even_when_each_fits_its_slot(tmp_path):
    info, frames = fixture()
    formula = r'$\frac{x}{y}$'
    info['description'] = {'text': formula+'\n'+formula, 'fontsize': 8, 'math': True}
    info['forest'] = {'geometry': {'footer_mm': 25, 'description_y_mm': 12,
                                   'text_line_pitch_mm': 2.4}}
    with pytest.raises(ValueError, match='description.*overlap|footer.*overlap|description.*collid'):
        draw(info, frames, tmp_path)
    info['forest']['geometry']['text_line_pitch_mm'] = 6
    fig = draw(info, frames, tmp_path)
    rendered = [text for text in fig.axes['ax'].texts if text.get_text() == formula]
    assert len(rendered) == 2
    assert rendered[1].get_position()[1]-rendered[0].get_position()[1] == pytest.approx(6)
    assert all(text.get_parse_math() for text in rendered)
    assert fig.forest_layout['verification']['text_and_legend_bounds'] == 'passed'


def test_footer_description_still_checks_badge_overlap_without_legend(tmp_path):
    info, frames = fixture()
    info['frame'] = {'ax': {'legend': False}}
    info['description'] = {'text': 'XX\nXX\nXX', 'fontsize': 5}
    info['forest'] = {'geometry': {'description_start_column': 3,
                                   'legend_start_columns': [1, 2]}}
    with pytest.raises(ValueError, match='description.*axlogo|description.*badge|footer.*overlap'):
        draw(info, frames, tmp_path)


def test_explicit_blank_missing_column_has_no_body_text_and_still_renders(tmp_path):
    from jarvisplot.forest_text import text_metrics_mm
    info, frames = fixture()
    frames['rows']['empty'] = ''
    info['columns'] = [{'field': 'empty', 'label': 'Input', 'missing': ''}]
    assert text_metrics_mm('', 5.5, 'Helvetica', 'normal') == (0, 0, 0)
    layout = solve_forest(info, frames)
    assert layout['rows'][0]['cells'] == ['']
    fig = draw(info, frames, tmp_path)
    texts = fig.axes['ax'].texts
    assert sum(artist.get_text() == 'Input' for artist in texts) == 1
    assert not any(artist.get_text() == '' for artist in texts)
    row_y = layout['rows'][0]['top_mm']+layout['geometry']['cell_y_mm']
    assert not any(artist.get_position()[1] == row_y for artist in texts)
    assert fig.forest_layout['verification']['text_and_legend_bounds'] == 'passed'


def test_yaml_nan_overlay_offset_fails_before_graphics_without_mutating_data():
    import yaml
    info, frames = layered_fixture()
    info['forest']['overlays']['classes']['first']['offset_mm'] = yaml.safe_load('.nan')
    snapshots = {name: frame.copy(deep=True) for name, frame in frames.items()}
    with pytest.raises(ValueError, match=r'forest\.overlays\.classes\.first\.offset_mm.*finite'):
        solve_forest(info, frames)
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(frame, snapshots[name])


@pytest.mark.parametrize('field,value', [('glyph_offset_mm', float('inf')),
                                        ('value_column_mm', float('nan'))])
def test_nonfinite_new_geometry_is_rejected_without_mutating_data(field, value):
    info, frames = layered_fixture()
    info['forest']['geometry'][field] = value
    snapshots = {name: frame.copy(deep=True) for name, frame in frames.items()}
    with pytest.raises(ValueError, match=r'forest\.geometry\.'+field+'.*finite'):
        solve_forest(info, frames)
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(frame, snapshots[name])


def test_finite_axis_endpoints_with_unrepresentable_span_fail_without_mutating_data():
    info, frames = layered_fixture()
    info['x'] = {'lim': [-1e308, 1e308]}
    snapshots = {name: frame.copy(deep=True) for name, frame in frames.items()}
    with pytest.raises(ValueError, match='domain.*representable finite span'):
        solve_forest(info, frames)
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(frame, snapshots[name])


@pytest.mark.parametrize('fontset', ['stix', 'cm', 'dejavuserif'])
def test_mathtext_fontset_changes_refresh_metrics_and_keep_native_ink_inside_slots(fontset, tmp_path):
    import matplotlib as mpl
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.textpath import TextToPath
    from jarvisplot.forest_text import font_properties, text_metrics_mm

    formula = r'$p_T > 25\,\mathrm{GeV}$'
    # The same public call has already been measured under another MathText
    # font set. Its answer must still match a fresh engine in the new context.
    warm_fontset = 'cm' if fontset == 'stix' else 'stix'
    with mpl.rc_context({'mathtext.fontset': warm_fontset}):
        warmed = text_metrics_mm(formula, 6.23, 'Helvetica', 'normal', math=True)
    with mpl.rc_context({'mathtext.fontset': fontset}):
        direct_advance = TextToPath().get_text_width_height_descent(
            formula, font_properties(6.23, 'Helvetica'), ismath=True)[0]*25.4/72
        actual = text_metrics_mm(formula, 6.23, 'Helvetica', 'normal', math=True)
        assert actual[0] == pytest.approx(direct_advance)
        assert actual[0] != pytest.approx(warmed[0], abs=1e-5)

        info, frames = fixture()
        frames['rows']['label'] = formula
        info['columns'] = [{'field': 'label', 'label': r'$p_T$', 'role': 'row_name', 'math': True}]
        info['title'] = {'text': r'$p_T$ comparison', 'fontsize': 8, 'math': True}
        info['header_note'] = {'text': r'MC entries: $N_S=123$, $N_B=456$', 'fontsize': 4, 'math': True}
        info['description'] = {'text': 'Cuts:\n'+formula, 'fontsize': 4.5, 'math': True,
                               'color': '#8E24AA'}
        info['glyph_label'] = r'$\mathrm{mean}\pm\mathrm{SD}$'
        info['forest'] = {'legend_math': True, 'rows': {'text_color': '#8E24AA'}}
        snapshots = {name: frame.copy(deep=True) for name, frame in frames.items()}
        fig = draw(info, frames, tmp_path)
        layout = fig.forest_layout
        assert layout['verification']['text_and_legend_bounds'] == 'passed'
        for name, frame in frames.items():
            pd.testing.assert_frame_equal(frame, snapshots[name])

        # Independently locate rendered row-formula pixels. The purple color
        # distinguishes them from the black grid and blue data marks.
        canvas = FigureCanvasAgg(fig.fig)
        canvas.draw()
        raster = np.asarray(canvas.buffer_rgba()).astype(int)
        g = layout['geometry']
        scale = raster.shape[1]/layout['width_mm']
        y0 = math.ceil(layout['rows'][0]['top_mm']*scale)
        y1 = math.floor((layout['rows'][0]['top_mm']+g['plot_y_mm']-g['point_band_mm']/2)*scale)
        band = raster[y0:y1+1, :, :3]
        purple = (band[:, :, 0] > band[:, :, 1]+30) & (band[:, :, 2] > band[:, :, 0]+10)
        positions = np.nonzero(purple)[1]
        assert positions.size > 0
        column = layout['columns'][0]
        assert positions.min()/scale >= column['left_mm']-1/scale
        assert positions.max()/scale <= column['left_mm']+column['width_mm']+1/scale
        # The editorial formula has its own left edge, including font-specific
        # overhang. Inspect exported ink there as well as the table-column ink.
        footer = _artist(fig, 'forest-description-01')
        footer_y = footer.get_position()[1]
        footer_band = raster[math.ceil((footer_y-1)*scale):math.floor((footer_y+1)*scale)+1, :, :3]
        purple = ((footer_band[:, :, 0] > footer_band[:, :, 1]+30) &
                  (footer_band[:, :, 2] > footer_band[:, :, 0]+10))
        positions = np.nonzero(purple)[1]
        assert positions.size > 0
        desc_left = g['grid_lines_mm'][g['description_start_column']-1]
        desc_right = desc_left+g['description_columns']*g['grid_unit_mm']
        assert positions.min()/scale >= desc_left-1/scale
        assert positions.max()/scale <= desc_right+1/scale


@pytest.mark.parametrize('math_fontfamily', ['stix', 'dejavusans'])
def test_forest_legend_math_family_uses_the_same_fonts_for_wrap_bounds_and_exports(math_fontfamily, tmp_path):
    import matplotlib as mpl
    from matplotlib.textpath import TextToPath
    from jarvisplot.forest_text import text_metrics_mm
    info, frames = layered_fixture()
    info['glyph_label'] = r'$Q_{0.50}[Q_{0.25},Q_{0.75}]$'
    info['forest']['legend_math'] = True
    info['forest']['overlays']['legend']['label'] = r'$\mathrm{mean}\pm\mathrm{SD}$'
    info['forest']['geometry'].update(legend_start_columns=[2, 6], description_start_column=9,
                                    description_columns=4)
    info['frame'] = {'ax': {'legend': {'prop': {'math_fontfamily': math_fontfamily}}}}
    snapshots = {name: frame.copy(deep=True) for name, frame in frames.items()}
    with mpl.rc_context({'mathtext.fontset': 'cm', 'legend.fontsize': 17}):
        fig = draw(info, frames, tmp_path)
        labels = fig.axes['ax'].get_legend().get_texts()
        assert all(label.get_math_fontfamily() == math_fontfamily for label in labels)
        assert all(label.get_fontfamily() == ['Helvetica'] for label in labels)
        assert all(label.get_fontsize() == 5 for label in labels)
        label = next(t for t in labels if r'\mathrm{mean}' in t.get_text())
        prop = label.get_fontproperties()
        direct = TextToPath().get_text_width_height_descent(label.get_text(), prop, ismath=True)[0]*25.4/72
        measured = text_metrics_mm(label.get_text(), 5, 'Helvetica', 'normal', math=True, prop=prop)[0]
        assert measured == pytest.approx(direct)
        assert fig.forest_layout['legend_fonts']['math_fontfamily'] == math_fontfamily
        assert fig.forest_layout['legend_fonts']['size_pt'] == 5
        fig.fmts = ['png', 'pdf', 'svg']
        fig.savefig()
        assert (tmp_path/f'{fig.name}.png').read_bytes().startswith(b'\x89PNG')
        assert (tmp_path/f'{fig.name}.pdf').read_bytes().startswith(b'%PDF-')
        assert mpl.rcParams['mathtext.fontset'] == 'cm'
    for name, snapshot in snapshots.items():
        pd.testing.assert_frame_equal(frames[name], snapshot)


def test_forest_legend_honours_explicit_ordinary_family_weight_and_size(tmp_path):
    info, frames = fixture('box')
    info['glyph_label'] = r'$\mathrm{mean}\pm\mathrm{SD}$'
    info['forest'] = {'legend_math': True, 'geometry': {'footer_mm': 32}}
    info['frame'] = {'ax': {'legend': {'prop': {'family': 'DejaVu Sans', 'size': 6,
                                             'weight': 'bold', 'math_fontfamily': 'dejavusans'}}}}
    fig = draw(info, frames, tmp_path)
    labels = fig.axes['ax'].get_legend().get_texts()
    assert all(label.get_fontfamily() == ['DejaVu Sans'] for label in labels)
    assert all(label.get_fontsize() == 6 for label in labels)
    assert all(label.get_fontweight() == 'bold' for label in labels)
    assert all(label.get_math_fontfamily() == 'dejavusans' for label in labels)


@pytest.mark.parametrize('math_fontfamily', ['stix', 'dejavusans'])
def test_cut_formulas_select_their_own_font_for_metrics_and_exports(math_fontfamily, tmp_path):
    import matplotlib as mpl
    from matplotlib.textpath import TextToPath
    from jarvisplot.forest_text import text_metrics_mm
    info, frames = fixture()
    formula = r'$m_{t\bar t}^{\mathrm{gen}} < 355\,\mathrm{GeV}$'
    info['description'] = {'text': 'Parton cuts:\n'+formula+'\n\nSupplied cut.',
                           'fontsize': 4.5, 'math': True, 'math_fontfamily': math_fontfamily}
    info['forest'] = {'geometry': {'footer_mm': 32}}
    # Prove the description selector is independent of both legend and rc.
    info['frame'] = {'ax': {'legend': {'prop': {'math_fontfamily': 'dejavusans'}}}}
    original = deepcopy(info)
    snapshots = {name: frame.copy(deep=True) for name, frame in frames.items()}
    with mpl.rc_context({'mathtext.fontset': 'cm'}):
        fig = draw(info, frames, tmp_path)
        artist = _artist(fig, 'forest-description-01')
        assert artist.get_text() == formula
        assert artist.get_parse_math()
        assert not artist.get_usetex()
        assert artist.get_math_fontfamily() == math_fontfamily
        assert artist.get_fontfamily() == ['Helvetica']
        prop = artist.get_fontproperties()
        direct = TextToPath().get_text_width_height_descent(formula, prop, ismath=True)[0]*25.4/72
        assert text_metrics_mm(formula, 4.5, 'Helvetica', 'normal', math=True, prop=prop)[0] == pytest.approx(direct)
        assert fig.forest_layout['description_fonts']['math_fontfamily'] == math_fontfamily
        assert fig.forest_layout['verification']['text_and_legend_bounds'] == 'passed'
        assert all(t.get_math_fontfamily() == 'dejavusans' for t in fig.axes['ax'].get_legend().get_texts())
        fig.fmts = ['png', 'pdf', 'svg']
        fig.savefig()
        assert all((tmp_path/f'comparison.{ext}').stat().st_size > 100 for ext in fig.fmts)
        assert mpl.rcParams['mathtext.fontset'] == 'cm'
    assert info == original
    for name, table in frames.items():
        pd.testing.assert_frame_equal(table, snapshots[name])


def test_cut_formula_font_does_not_enable_parsing_and_schema_rejects_unknown_fonts(tmp_path):
    import matplotlib as mpl
    info, frames = fixture()
    info['description'] = {'text': '$p_T$', 'math': False, 'math_fontfamily': 'stix'}
    info['forest'] = {'geometry': {'footer_mm': 32}}
    fig = draw(info, frames, tmp_path)
    assert not _artist(fig, 'forest-description-00').get_parse_math()
    for name, frame in frames.items():
        frame.to_csv(tmp_path/f'{name}.csv', index=False)
    config = {'DataSet': [{'name': name, 'type': 'csv', 'path': str(tmp_path/f'{name}.csv')}
                          for name in frames],
              'Figures': [info]}
    assert validate_config(config, check_columns=False).ok
    info['description']['math_fontfamily'] = 'invented'
    assert not validate_config(config, check_columns=False).ok
    # Omitted local selectors use the effective font card, without changing rc.
    info['description'].pop('math_fontfamily')
    with mpl.rc_context({'mathtext.fontset': 'cm'}):
        fig = draw(info, frames, tmp_path)
        assert _artist(fig, 'forest-description-00').get_math_fontfamily() == 'stix'
        assert mpl.rcParams['mathtext.fontset'] == 'cm'


@pytest.mark.parametrize('explicit_family', [False, True])
def test_legend_ttc_face_honours_numeric_weight_and_italic_style(explicit_family, tmp_path):
    from matplotlib.ft2font import FT2Font
    info, frames = fixture()
    prop = {'weight': 600, 'style': 'italic'}
    if explicit_family:
        prop['family'] = 'Helvetica'
    info['frame'] = {'ax': {'legend': {'prop': prop}}}
    fig = draw(info, frames, tmp_path)
    for label in fig.axes['ax'].get_legend().get_texts():
        resolved = label.get_fontproperties()
        assert resolved.get_weight() == 600
        assert resolved.get_style() == 'italic'
        # Both desired features must survive TTC extraction, not merely sit
        # on FontProperties while the first regular face is rendered.
        style_name = FT2Font(resolved.get_file()).style_name
        assert 'Bold' in style_name and ('Oblique' in style_name or 'Italic' in style_name)


def test_equal_halves_put_primary_distribution_above_three_uniform_summary_lanes(tmp_path):
    info, frames = layered_fixture()
    info['forest']['geometry']['plot_y_mm'] = 7.5
    for settings, offset in zip(info['forest']['overlays']['classes'].values(), [.85, 2.5, 4.15]):
        settings['offset_mm'] = offset
    fig = draw(info, frames, tmp_path)
    row = fig.forest_layout['rows'][0]
    g = fig.forest_layout['geometry']
    half = g['point_band_mm']/2
    assert half == 5
    centre = row['top_mm']+g['plot_y_mm']
    primary = _artist(fig, 'forest-primary-box-001')
    assert primary.get_y()+primary.get_height()/2 == pytest.approx(centre-half/2)
    for point in row['points']:
        assert -half <= point['offset_mm']-point['radius_mm']
        assert point['offset_mm']+point['radius_mm'] <= 0
    lane_y = []
    for interval in row['intervals']:
        assert interval['offset_mm']-interval['extent_mm'] >= 0
        assert interval['offset_mm']+interval['extent_mm'] <= half
        lane_y.append(_artist(fig, 'forest-overlay-001-'+interval['id']).get_ydata()[0])
    assert lane_y[1] == pytest.approx(centre+half/2)
    assert lane_y[1]-lane_y[0] == pytest.approx(lane_y[2]-lane_y[1])
