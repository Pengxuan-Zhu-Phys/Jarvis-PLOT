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
    fonts = layout['typography']
    assert label['content_width_mm'] == pytest.approx(text_metrics_mm('Longer group', 6, fonts['family'], 'bold')[0])
    assert w['content_width_mm'] == pytest.approx(text_metrics_mm('Weight', fonts['header_pt'], fonts['family'], 'bold')[0])
    assert label['left_mm'] < w['left_mm'] < n['left_mm']
    assert layout['geometry']['plot_left_mm'] < w['left_mm']
    assert layout['geometry']['plot_right_mm'] > n['left_mm']
    assert n['align'] == 'right' and w['align'] == label['align'] == 'left'
    assert label['fontweight'] == 'bold' and w['fontweight'] == 'normal'
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
