"""Forest contract and physical layout. No statistical inference or renderer.

The same solve drives render, doctor and config inspection. Input fields are
literal column names, not expressions. Only declared columns reach the scene.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

MM_PER_PT = 25.4 / 72
ROW_FIELDS = ('row', 'label', 'detail', 'stats', 'estimate', 'lower', 'q1', 'median', 'q3', 'upper')
POINT_FIELDS = ('row', 'id', 'x', 'class')
STYLE = ['forest', 'swiss_column']


def merge(base, override):
    out = deepcopy(base)
    for key, value in override.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else deepcopy(value)
    return out


def appearance(info):
    from .schema_catalog import schema_catalog
    from jsonschema import Draft202012Validator
    tokens = info.get('style', STYLE)
    if tokens != STYLE:
        raise ValueError('forest style must be [forest, swiss_column] (or omitted)')
    path = Path(__file__).with_name('cards') / 'forest' / 'swiss_column.json'
    card = json.loads(path.read_text())
    overrides = deepcopy(info.get('forest', {}))
    geometry = merge(card['Geometry'], overrides.pop('geometry', {}))
    typography = merge(card['Typography'], overrides.pop('typography', {}))
    style = merge(card['Style']['forest'], overrides)
    _, registry = schema_catalog()
    schema = registry.contents('https://jarvis-plot.org/schema/v2/types/forest.json')
    validator = Draft202012Validator(schema['$defs']['appearance'], registry=registry)
    errors = sorted(validator.iter_errors({**style, 'geometry': geometry, 'typography': typography}), key=lambda e: str(list(e.path)))
    if errors:
        error = errors[0]
        raise ValueError(f"forest.{'.'.join(map(str, error.path))}: {error.message}")
    for name, settings in style['scatter']['classes'].items():
        missing = {'color', 'marker', 'fill'} - set(settings)
        if missing:
            raise ValueError(f'forest.scatter.classes.{name}: missing style fields {sorted(missing)}')
    if sum(c.get('role', 'left') == 'row_name' for c in info.get('columns', [])) > 1:
        raise ValueError('forest.columns: permits at most one row_name')
    g = geometry
    if g['margin_left_mm'] + g['margin_right_mm'] >= g['width_mm']:
        raise ValueError('forest.geometry: margins must leave space for the numerical region')
    if not (g['row_band_mm'] <= g['row_pitch_mm'] and
            g['point_band_mm']/2 <= g['plot_y_mm'] <= g['row_band_mm']-g['point_band_mm']/2 and
            0 < g['cell_y_mm'] < g['plot_y_mm']-g['point_band_mm']/2):
        raise ValueError('forest.geometry: text must be above the point band, which must fit inside row band/pitch; adjust baselines/band/pitch')
    if not (0 < g['top_tick_y_mm'] < g['top_axis_y_mm'] < g['header_mm']):
        raise ValueError('forest.geometry: top tick labels and axis must be ordered inside header')
    if not (1 <= g['legend_start_columns'][0] < g['legend_start_columns'][1] <= g['grid_columns']):
        raise ValueError('forest.geometry.legend_start_columns: requires two increasing reference column numbers inside the grid')
    if not (g['legend_start_columns'][1] < g['description_start_column'] and
            g['description_start_column']+g['description_columns']-1 <= g['grid_columns']):
        raise ValueError('forest.geometry: description must follow the point legend column and fit inside the reference grid')
    return geometry, typography, style, card


def source_columns(info, *, required=False):
    """(source, columns, literal columns) for both narrow input tables."""
    glyph = info.get('glyph', 'interval')
    mapping = {k: info.get('fields', {}).get(k, k) for k in ROW_FIELDS}
    keys = ['row', *(['estimate', 'lower', 'upper'] if glyph == 'interval' else ['lower', 'q1', 'median', 'q3', 'upper'])]
    texts = {mapping[k] for k in ('row', 'label', 'detail', 'stats')}
    cols = {mapping[k] for k in keys}
    if not required:
        cols.update(texts)
    cols.update(v for v in info.get('fields', {}).values())
    for column in info.get('columns', []):
        cols.add(column['field'])
        texts.add(column['field'])
    for key, value in info.get('annotations', {}).items():
        if value is not False:
            cols.add(value)
            texts.add(value)
    result = [(info['data'], cols, texts)]
    if info.get('points'):
        pm = {k: info.get('point_fields', {}).get(k, k) for k in POINT_FIELDS}
        pc = {pm[k] for k in ('row', 'id', 'x')}
        if not required:
            pc.add(pm['class'])
        pc.update(info.get('point_fields', {}).values())
        result.append((info['points'], pc, {pm[k] for k in ('row', 'id', 'class')}))
    return result


def plan_forest_sources(config):
    plan = {}
    for info in config.get('Figures', []):
        if isinstance(info, dict) and info.get('type') == 'forest' and info.get('enable', True):
            for name, cols, texts in source_columns(info):
                bucket = plan.setdefault(name, {'columns': set(), 'literal_columns': set()})
                bucket['columns'].update(cols)
                bucket['literal_columns'].update(texts)
    return plan


def load_forest_tables(config, *, base_dir, light_only=False, skipped=None):
    """Use the production DataSet loader, including transforms and projections."""
    from .data_loader import DataSet
    from .data_access import _NullLogger
    frames = {}
    plan = plan_forest_sources(config)
    workdir = config.get('project', {}).get('workdir', base_dir)
    from .utils.pathing import resolve_project_path
    workdir = resolve_project_path(workdir, base_dir)
    for entry in config.get('DataSet', []):
        name = entry.get('name')
        if name not in plan:
            continue
        ds = DataSet()
        ds.logger = _NullLogger()
        ds.setinfo(entry, rootpath=workdir, eager=False)
        transform = ds.transform
        ds.literal_columns = plan[name]['literal_columns']
        ds.preserve_empty_strings = True
        from .column_demand import transform_columns, transform_output_columns, transform_needs_all_columns
        cols = plan[name]['columns']
        if not transform_needs_all_columns(ds.transform):
            ds.set_required_columns(cols | transform_columns(ds.transform), retained=cols | transform_output_columns(ds.transform))
        if light_only:
            ds.transform = None
        ds.load(force=True)
        frame = ds.get_data()
        if light_only and transform:
            from .Figure.preprocessor_runtime import apply_light_transforms
            frame, _, heavy = apply_light_transforms(frame, transform, logger=None)
            if skipped is not None:
                skipped[name] = heavy
        frames[name] = frame
    return frames


def _blank(value):
    # None is a null coordinate in typed tables; NaN is always invalid data.
    return value is None or (isinstance(value, str) and value == '')


def _text(value):
    if value is None:
        return ''
    try:
        if math.isnan(value):
            return ''
    except (TypeError, ValueError):
        pass
    return str(value)


def _identifier(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{context}: must be a nonempty string (preserve identifiers when exporting typed tables)')
    return value


def _number(value, context, *, optional=False):
    if _blank(value):
        if optional:
            return None
        raise ValueError(f'{context}: required finite coordinate is blank')
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{context}: expected a finite number, got {value!r}') from None
    if isinstance(value, bool) or not math.isfinite(result):
        raise ValueError(f'{context}: nonfinite or invalid coordinate {value!r}')
    return result


def shared_domain(values, x, geometry):
    values = list(values)
    if 'reference' in x:
        values.append(_number(x['reference'], 'x.reference'))
    low, high = min(values), max(values)
    minimum = _number(x.get('minimum_span', 1), 'x.minimum_span')
    if minimum <= 0:
        raise ValueError('x.minimum_span must be positive')
    if 'lim' in x:
        domain = [_number(v, 'x.lim') for v in x['lim']]
        if len(domain) != 2 or domain[0] >= domain[1]:
            raise ValueError('x.lim must be two finite ascending values')
        if low < domain[0] or high > domain[1]:
            raise ValueError(f'x.lim excludes drawable values/reference [{low:g}, {high:g}]')
    else:
        centre = low / 2 + high / 2
        span = max(high-low, minimum) * (1 + 2*geometry['padding_fraction'])
        if not math.isfinite(span) or span <= 0:
            raise ValueError('x domain cannot be represented; use finite coordinates with a representable span')
        rough = span/4
        power = 10 ** math.floor(math.log10(rough))
        step = min(s for s in (1, 2, 2.5, 5, 10) if s*power >= rough) * power
        domain = [math.floor((centre-span/2)/step)*step, math.ceil((centre+span/2)/step)*step]
    if not all(math.isfinite(v) for v in domain) or domain[1] <= domain[0]:
        raise ValueError('x domain has no representable finite span')
    ticks = [domain[0], 0 if domain[0] < 0 < domain[1] else domain[0]/2+domain[1]/2, domain[1]]
    return domain, ticks


def x_to_mm(value, domain, geometry):
    return geometry['plot_left_mm'] + (value-domain[0])/(domain[1]-domain[0])*(geometry['plot_right_mm']-geometry['plot_left_mm'])


def layout_columns(columns, rows, geometry, typography, name):
    """Measure literal text with the same fonts as render; solve one global grid.

    Font metrics are loaded lazily, without a figure or a rendering canvas.
    Validation and discovery do not call this data-dependent layout stage.
    """
    from .forest_text import text_metrics_mm
    g = geometry
    if sum(c['role'] == 'row_name' for c in columns) > 1:
        raise ValueError(f'figure {name!r}: columns permits at most one row_name')
    for index, column in enumerate(columns):
        samples = [(column['label'], typography['header_pt'], 'bold')]
        samples.extend((r['cells'][index], column['fontsize'], column['fontweight']) for r in rows)
        try:
            metrics = [text_metrics_mm(value, size, typography['family'], weight) for value, size, weight in samples]
        except ValueError as exc:
            raise ValueError(f'figure {name!r}, column {column["field"]!r}: {exc}') from exc
        column['content_width_mm'] = max(m[0] for m in metrics)
        column['text_left_padding_mm'] = g['cell_padding_mm'] + max(m[1] for m in metrics)
        column['text_right_padding_mm'] = g['cell_padding_mm'] + max(m[2] for m in metrics)
        column['width_mm'] = column['content_width_mm'] + column['text_left_padding_mm'] + column['text_right_padding_mm']
    left = [c for c in columns if c['role'] == 'row_name'] + [c for c in columns if c['role'] == 'left']
    right = [c for c in columns if c['role'] == 'right']
    unit = (g['width_mm']-g['margin_left_mm']-g['margin_right_mm'])/g['grid_columns']
    g['grid_unit_mm'] = unit
    g['grid_lines_mm'] = [g['margin_left_mm']+i*unit for i in range(g['grid_columns']+1)]
    for column in columns:
        column['measured_width_mm'] = column['width_mm']
        column['align'] = 'right' if column['role'] == 'right' else 'left'
    def group_width(group):
        return sum(c['width_mm'] for c in group) + max(0, len(group)-1)*g['column_gap_mm']
    left_end = g['margin_left_mm'] + group_width(left)
    right_start = g['width_mm'] - g['margin_right_mm'] - group_width(right)
    available = right_start-left_end
    required_gap = g['group_gap_mm'] if left and right else 0
    if available < required_gap:
        raise ValueError(f'figure {name!r}: measured columns leave {available:.3f} mm on the text line; '
                         f'need at least {required_gap:g} mm between text groups. Shorten column values/headers, '
                         'remove columns or explicitly adjust typography/gaps.')
    # Text and marks have separate vertical bands. The numerical span can extend
    # beneath both text groups without depending on their measured widths.
    g['plot_left_mm'] = g['margin_left_mm']
    g['plot_right_mm'] = g['width_mm']-g['margin_right_mm']
    g['marks_left_mm'] = 0
    g['marks_right_mm'] = g['width_mm']
    for group, start in ((left, g['margin_left_mm']), (right, right_start)):
        for column in group:
            column['left_mm'] = start
            start += column['width_mm']+g['column_gap_mm']


def pack_points(points, domain, geometry, scatter, row):
    """Deterministic tangent candidates, in mm. Preserve x; never discard marks."""
    placed = []
    for p in sorted(points, key=lambda p: (p['x'], p['id'])):
        style = {k: v for k, v in scatter.items() if k != 'classes'} | scatter['classes'][p['class']]
        px = x_to_mm(p['x'], domain, geometry)
        # o and x have equal nominal extents. Other marker shapes are explicitly
        # outside v1, avoiding unsupported bounding-box assumptions.
        radius = style['diameter_mm']/2*(math.sqrt(2) if style['marker'] == 'x' else 1) + style['linewidth_pt']*MM_PER_PT/2
        candidates = [0.0]
        for q in placed:
            distance = radius+q['radius_mm']+max(style['gap_mm'], q['style']['gap_mm'])
            dx = abs(px-q['x_mm'])
            if dx < distance:
                dy = math.sqrt(distance*distance-dx*dx)+1e-8
                candidates.extend((q['offset_mm']+dy, q['offset_mm']-dy))
        for py in sorted(candidates, key=lambda v: (abs(v), v)):
            if all(math.hypot(px-q['x_mm'], py-q['offset_mm']) >= radius+q['radius_mm']+max(style['gap_mm'], q['style']['gap_mm'])-1e-9 for q in placed):
                placed.append({**p, 'x_mm': px, 'offset_mm': py, 'radius_mm': radius, 'style': style})
                break
        else:
            raise ValueError(f'row {row!r}: could not pack point {p["id"]!r}; increase forest.geometry.point_band_mm and row band/pitch')
    if placed:
        low = min(p['offset_mm']-p['radius_mm'] for p in placed)
        high = max(p['offset_mm']+p['radius_mm'] for p in placed)
        if high-low > geometry['point_band_mm']+1e-9:
            raise ValueError(f'row {row!r}: point packing needs {high-low:.3f} mm, band is {geometry["point_band_mm"]:g} mm; increase forest.geometry.point_band_mm and row band/pitch')
        for p in placed:
            p['offset_mm'] -= (high+low)/2
            if p['x_mm']-p['radius_mm'] < geometry['marks_left_mm'] or p['x_mm']+p['radius_mm'] > geometry['marks_right_mm']:
                raise ValueError(f'row {row!r}, point {p["id"]!r}: marker exceeds figure width; enlarge x.lim/padding or margins')
    return placed


def solve_forest(info, frames):
    """Validate generic tables and return JSON-serializable solved scene metadata."""
    g, typography, style, card = appearance(info)
    name = info.get('name', 'forest')
    def table(source):
        if source not in frames:
            raise ValueError(f'figure {name!r}: unknown/unavailable DataSet {source!r}')
        data = frames[source]
        from .utils.dataframes import polars_to_pandas
        if not hasattr(data, 'to_dict') or not hasattr(data, 'iloc'):
            data = polars_to_pandas(data, logger=None, stage='forest.selection')
        return data
    raw = table(info['data'])
    mapping = {k: info.get('fields', {}).get(k, k) for k in ROW_FIELDS}
    slots = {k: info.get('annotations', {}).get(k, mapping[k]) for k in ('label', 'detail', 'stats')}
    required = {'row', *(['estimate', 'lower', 'upper'] if info.get('glyph', 'interval') == 'interval' else ['lower', 'q1', 'median', 'q3', 'upper'])}
    requested = {mapping[k] for k in required} | set(info.get('fields', {}).values()) | {v for v in info.get('annotations', {}).values() if v is not False}
    requested.update(c['field'] for c in info.get('columns', []))
    missing = requested-set(raw.columns)
    if missing:
        raise ValueError(f'figure {name!r}, rows: missing fields {sorted(missing)}')
    rows, by_id = [], {}
    records = raw.to_dict('records')
    for record in records:
        rid = _identifier(record[mapping['row']], f'figure {name!r}, row field {mapping["row"]!r}')
        if rid in by_id:
            raise ValueError(f'figure {name!r}: duplicate row {rid!r}')
        row = {'row': rid, 'points': []}
        for key, column in slots.items():
            row[key] = '' if column is False else _text(record.get(column))
        if slots['label'] is not False and not row['label']:
            row['label'] = rid
        keys = ['estimate', 'lower', 'upper'] if info.get('glyph', 'interval') == 'interval' else ['lower', 'q1', 'median', 'q3', 'upper']
        coordinates = {k: _number(record[mapping[k]], f'figure {name!r}, row {rid!r}, {mapping[k]}', optional=True) for k in keys}
        supplied = [v for v in coordinates.values() if v is not None]
        if supplied and len(supplied) != len(keys):
            raise ValueError(f'figure {name!r}, row {rid!r}: partial {info.get("glyph", "interval")} glyph; supply {keys} or leave all blank with observations')
        ordered = ['lower', 'estimate', 'upper'] if info.get('glyph', 'interval') == 'interval' else keys
        vals = [coordinates[k] for k in ordered]
        if supplied and vals != sorted(vals):
            raise ValueError(f'figure {name!r}, row {rid!r}: unordered coordinates {ordered}')
        row['coordinates'] = coordinates
        by_id[rid] = row
        rows.append(row)
    if not rows:
        raise ValueError(f'figure {name!r}: row table is empty')
    columns = deepcopy(info.get('columns'))
    if columns is None:
        active = [key for key in ('label', 'detail', 'stats') if any(row[key] for row in rows)]
        columns = []
        for key in active:
            columns.append({'field': slots[key], 'label': key.capitalize(),
                            'role': {'label': 'row_name', 'detail': 'left', 'stats': 'right'}[key], '_slot': key})
    for column in columns:
        role = column.setdefault('role', 'left')
        if role not in ('row_name', 'left', 'right'):
            raise ValueError(f'column {column["field"]!r}: unknown role {role!r}')
        column['align'] = 'right' if role == 'right' else 'left'
        column.setdefault('fontsize', typography['label_pt'] if role == 'row_name' else typography['stats_pt'] if role == 'right' else typography['detail_pt'])
        column.setdefault('fontweight', 'bold' if role == 'row_name' else 'normal')
        column.setdefault('missing', '—')
    for row, record in zip(rows, records):
        row['cells'] = []
        for column in columns:
            value = row[column['_slot']] if '_slot' in column else record[column['field']]
            text = _text(value)
            if 'format' in column and not _blank(value):
                numeric = _number(value, f'figure {name!r}, row {row["row"]!r}, column {column["field"]}')
                try:
                    text = format(numeric, column['format'])
                except ValueError as exc:
                    raise ValueError(f'column {column["field"]!r}: invalid numeric format {column["format"]!r}: {exc}') from exc
            row['cells'].append(text or column['missing'])
    if info.get('points'):
        points = table(info['points'])
        pm = {k: info.get('point_fields', {}).get(k, k) for k in POINT_FIELDS}
        missing = {pm[k] for k in ('row', 'id', 'x')} | set(info.get('point_fields', {}).values())
        missing -= set(points.columns)
        if missing:
            raise ValueError(f'figure {name!r}, points: missing fields {sorted(missing)}')
        seen = set()
        for record in points.to_dict('records'):
            rid = _identifier(record[pm['row']], f'figure {name!r}, points.row')
            pid = _identifier(record[pm['id']], f'figure {name!r}, row {rid!r}, points.id')
            if rid not in by_id:
                raise ValueError(f'figure {name!r}, point {pid!r}: unknown row foreign key {rid!r}')
            if (rid, pid) in seen:
                raise ValueError(f'figure {name!r}, row {rid!r}: duplicate point id {pid!r}')
            seen.add((rid, pid))
            cls = _text(record.get(pm['class'])) or 'normal'
            if cls not in style['scatter']['classes']:
                raise ValueError(f'figure {name!r}, row {rid!r}, point {pid!r}: unknown class {cls!r}; define forest.scatter.classes.{cls}')
            value = _number(record[pm['x']], f'figure {name!r}, row {rid!r}, point {pid!r}, {pm["x"]}')
            by_id[rid]['points'].append({'id': pid, 'x': value, 'class': cls})
    values = []
    for row in rows:
        drawable = [v for v in row['coordinates'].values() if v is not None] + [p['x'] for p in row['points']]
        if not drawable:
            raise ValueError(f'figure {name!r}, row {row["row"]!r}: no drawable glyph or observations')
        values.extend(drawable)
    domain, ticks = shared_domain(values, info.get('x', {}), g)
    from .forest_text import text_metrics_mm
    # Keep labels on their exact ticks; reserve only their outward ink on
    # both ends, according to the alignment and unit suffix in the card.
    axis = style['axis']
    for side, value in [('left', ticks[0]), ('right', ticks[-1])]:
        advance, left, right = text_metrics_mm(f"{value:g}{axis['tick_suffix']}", typography['tick_pt'], typography['family'], 'normal')
        shift = {'left': 0, 'center': advance/2, 'right': advance}[axis['tick_align']]
        extent = shift+left if side == 'left' else advance-shift+right
        key = f'margin_{side}_mm'
        g[f'requested_{key}'] = g[key]
        g[key] = max(g[key], extent+.02)
    layout_columns(columns, rows, g, typography, name)
    for i, row in enumerate(rows):
        row['top_mm'] = g['header_mm']+i*g['row_pitch_mm']
        row['points'] = pack_points(row['points'], domain, g, style['scatter'], row['row'])
    used = {p['class'] for r in rows for p in r['points']}
    # Canonical single-component declarations: labels belong to style categories,
    # ordering/layout belong to frame.ax.legend, exactly as for drawing layers.
    declarations = [{'legend': {'object': cls, 'label': s.get('label', cls)}} for cls, s in style['scatter']['classes'].items() if cls in used]
    glyph_object = 'glyph'
    while glyph_object in used:
        glyph_object = '_' + glyph_object
    if any(row['coordinates']['lower'] is not None for row in rows):
        label = info.get('glyph_label', 'Box + whiskers' if info.get('glyph') == 'box' else 'Estimate + bounds')
        declarations.insert(0, {'legend': {'object': glyph_object, 'label': label}})
    from .legend_cards import merge_axes_legend, resolve_axes_legend
    legend = merge_axes_legend(card['Frame']['ax']['legend'], info.get('frame', {}).get('ax', {}).get('legend', {}))
    if not declarations:
        legend = False
    resolve_axes_legend(legend, declarations)
    return {'figure': name, 'type': 'forest', 'geometry': g, 'typography': typography, 'style': style,
            'width_mm': g['width_mm'], 'height_mm': g['header_mm']+len(rows)*g['row_pitch_mm']+g['footer_mm'],
            'plot_width_mm': g['plot_right_mm']-g['plot_left_mm'], 'row_pitch_mm': g['row_pitch_mm'],
            'row_count': len(rows), 'point_count': sum(len(r['points']) for r in rows),
            'shared_domain': domain, 'shared_ticks': ticks, 'x_scale': 'shared', 'canvas_axes': 1,
            'rows': rows, 'columns': columns, 'legend': legend, 'legend_layers': declarations, 'glyph_object': glyph_object,
            'title': deepcopy(info.get('title', {})), 'description': deepcopy(info.get('description', {})),
            'meta': deepcopy(info.get('meta', {})),
            'verification': {'data_contract': 'passed', 'packing': 'deterministic_physical_mm', 'text_and_legend_bounds': 'requires_native_render'}}
