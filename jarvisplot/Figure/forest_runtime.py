"""Native physical forest renderer on one continuous canvas."""
from __future__ import annotations

import json
from pathlib import Path

from .figure import Figure
from ..forest_contract import solve_forest, x_to_mm, MM_PER_PT


def _merged_grid_values(domain, ticks, divisions):
    """Return tick and division grid coordinates without near-duplicates.

    Values are compared in normalized domain coordinates so round-off in a
    division expression cannot create a second line at an existing tick.
    When the coordinates coincide, retain the declared tick value.
    """
    low, high = domain
    span = high-low
    candidates = [
        (low+i*span/divisions, False)
        for i in range(divisions+1)
    ] + [(value, True) for value in ticks]
    candidates.sort(key=lambda item: item[0])
    merged = []
    for value, is_tick in candidates:
        if merged and abs((value-merged[-1][0])/span) <= 1e-12:
            if is_tick and not merged[-1][1]:
                merged[-1] = (value, True)
        else:
            merged.append((value, is_tick))
    return [value for value, _ in merged]


class ForestFigure(Figure):
    def from_dict(self, info):
        self.name = info['name']
        if info.get('enable', True) is False:
            self._setup_status = 'disabled'
            return False
        from ..schema_catalog import config_validator
        errors = list(config_validator().iter_errors({'DataSet': [], 'Figures': [info]}))
        if errors:
            raise ValueError(f'figure {self.name!r}: {errors[0].message}')
        self.info = info
        self.style = info.get('style', ['forest', 'swiss_column'])
        self.debug = info.get('debug', False)
        sources = list(dict.fromkeys([info['data'], *[info[key] for key in ('points', 'intervals') if info.get(key)]]))
        frames = {name: self.context.get(name) for name in sources}
        self.forest_layout = solve_forest(info, frames)
        for source in sources:
            self.context.consume(source)
        g = self.forest_layout
        from ..forest_contract import merge
        logo = self._frame.get('axlogo', {})
        logo_override = info.get('frame', {}).get('axlogo', {})
        if isinstance(logo_override, dict):
            logo = merge(logo, logo_override)
        self._frame = {'figure': {'figsize': [g['width_mm']/25.4, g['height_mm']/25.4]},
                       'axes': {'ax': {'rect': [0, 0, 1, 1]}}}
        if not self.print:
            self._frame['axlogo'] = logo
            size = g['geometry']['logo_size_mm']
            self._frame['axes']['axlogo'] = {'rect': [g['geometry']['margin_left_mm']/g['width_mm'], g['geometry']['logo_bottom_mm']/g['height_mm'], g['geometry']['grid_unit_mm']/g['width_mm'], size/g['height_mm']], 'frameon': False, 'xticks': [], 'yticks': []}
        import matplotlib.pyplot as plt
        self.fig = plt.figure(figsize=self._frame['figure']['figsize'], dpi=self.dpi, facecolor=g['style']['rows']['background'])
        self.fig.set_size_inches(*self._frame['figure']['figsize'], forward=False)
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, g['width_mm']), ylim=(g['height_mm'], 0))
        ax.axis('off')
        self.axes = {'ax': ax}
        if 'axlogo' in self._frame['axes']:
            self.axlogo = self._frame['axes']['axlogo']
            from ..forest_text import font_properties
            for artist in self.axlogo.texts:
                artist.set_fontproperties(font_properties(artist.get_fontsize(), artist.get_fontfamily()[0], artist.get_fontweight(), style=artist.get_fontstyle()))
                artist.set_parse_math(False)
                artist.set_usetex(False)
        self.forest_layout['canvas_axes'] = len(self.axes)
        self._setup_status = 'ok'
        return True

    def render(self):
        import matplotlib as mpl
        from ..font_runtime import rc_font_settings
        # Unhinted glyphs agree with physical font measurements and avoid
        # native-hinting failures for small Helvetica ticks at low DPI.
        with mpl.rc_context({**rc_font_settings(self._font_card), 'text.hinting': 'none', 'text.usetex': False}):
            self._render()

    def _render(self):
        from matplotlib.patches import Rectangle
        from .legend_runtime import apply_legend
        from .forest_legend import (
            ForestGlyph, HandlerForestGlyph, HandlerForestPoint,
            PhysicalLegendColumn, boxes_overlap, group_glyph_and_points,
        )
        from matplotlib.collections import PathCollection
        info, layout, ax = self.info, self.forest_layout, self.axes['ax']
        g, fonts, style = layout['geometry'], layout['typography'], layout['style']
        width, height = layout['width_mm'], layout['height_mm']
        domain, ticks = layout['shared_domain'], layout['shared_ticks']
        to_x = lambda value: x_to_mm(value, domain, g)
        checks = []
        annotations = []
        row_graphics = {}
        header_artists = []
        from ..forest_text import font_properties, wrap_text, text_metrics_mm
        from ..font_runtime import forest_font_properties

        def text(x, y, value, size, *, slot=None, row=None, **kwargs):
            if not value:
                return None
            weight = kwargs.pop('fontweight', None)
            font_role = kwargs.pop('font_role', 'text')
            prop = kwargs.pop('fontproperties', None)
            kwargs.setdefault('ha', 'left')
            parse_math = kwargs.pop('parse_math', False)
            artist = ax.text(x, y, value, fontproperties=prop if prop is not None else forest_font_properties(fonts, font_role, size, weight), va='center', parse_math=parse_math, usetex=False, **kwargs)
            checks.append((artist, slot, row))
            if row is None and y < g['top_axis_y_mm']:
                header_artists.append(artist)
            return artist

        title = info.get('title', {})
        origin = g['margin_left_mm']
        title_size = title.get('fontsize', fonts['title_pt'])
        title_weight = title.get('fontweight', fonts['_font_roles']['title']['weight'])
        title_prop = forest_font_properties(fonts, 'title', title_size, title_weight)
        title_right = origin+g['title_columns']*g['grid_unit_mm']
        for i, line in enumerate(wrap_text(title.get('text', ''), title_size, fonts['family'], title_weight, title_right-origin, math=title.get('math', False), prop=title_prop)):
            left_overhang = text_metrics_mm(line, title_size, fonts['family'], title_weight, math=title.get('math', False), prop=title_prop)[1]
            artist = text(origin+left_overhang, g['title_y_mm']+i*g['title_line_pitch_mm'], line, title_size, parse_math=title.get('math', False), fontproperties=title_prop, color=title.get('color', style['rows']['text_color']), slot=(origin, title_right, g['meta_y_mm']+1, g['x_label_y_mm']-1.5))
            if artist is not None:
                artist.set_gid(f'forest-title-{i:02d}')
        note = info.get('header_note', {})
        note_size, note_weight = note.get('fontsize', fonts['meta_pt']), note.get('fontweight', fonts['_font_roles']['caption']['weight'])
        note_prop = forest_font_properties(fonts, 'caption', note_size, note_weight)
        for i, line in enumerate(wrap_text(note.get('text', ''), note_size, fonts['family'], note_weight, width-g['margin_right_mm']-origin, math=note.get('math', False), prop=note_prop)):
            left_overhang = text_metrics_mm(line, note_size, fonts['family'], note_weight, math=note.get('math', False), prop=note_prop)[1]
            artist = text(origin+left_overhang, g.get('header_note_y_mm', g['meta_y_mm'])+i*g['text_line_pitch_mm'], line, note_size,
                          color=note.get('color', style['rows']['text_color']), fontproperties=note_prop, parse_math=note.get('math', False),
                          slot=(origin, width-g['margin_right_mm'], 0, g['title_y_mm']-1.5))
            if artist is not None:
                artist.set_gid(f'forest-header-note-{i:02d}')
        meta = info.get('meta', {})
        for key, column in (('number', 0), ('year', 9), ('place', 10)):
            text(g['grid_lines_mm'][column], g['meta_y_mm'], meta.get(key, ''), fonts['meta_pt'], font_role='caption', color=style['rows']['accent'] if key == 'number' else style['rows']['text_color'], slot=(g['grid_lines_mm'][column], g['grid_lines_mm'][10 if key=='year' else 12], 0, g['title_y_mm']-1.5))
        xlabel = info.get('x', {}).get('label', '')
        text(origin, g['x_label_y_mm'], xlabel, fonts['header_pt'], font_role='axis_label', slot=(origin, g['plot_right_mm'], g['x_label_y_mm']-1.5, g['column_header_y_mm']-1.5))
        handles = {}
        class_index = {layer['legend']['object']: i for i, layer in enumerate(layout['legend_layers'])}
        def cell_x(column):
            return column['left_mm'] + (column['width_mm']-column['text_right_padding_mm'] if column['align']=='right' else column['text_left_padding_mm'])
        for column in layout['columns']:
            text(cell_x(column), g['column_header_y_mm'], column['label'], fonts['header_pt'], font_role='axis_label', parse_math=column.get('math', False), ha=column['align'], slot=(column['left_mm'], column['left_mm']+column['width_mm'], g['column_header_y_mm']-1.5, g['top_tick_y_mm']-1))
        overlay_annotation = style.get('overlays', {}).get('annotation', {})
        if overlay_annotation.get('label'):
            text(g.get('value_right_mm', g['plot_right_mm']), g['column_header_y_mm'], overlay_annotation['label'], fonts['header_pt'], font_role='axis_label', parse_math=overlay_annotation.get('math', False), ha='right',
                 slot=(g['plot_left_mm'], g.get('value_right_mm', g['plot_right_mm']), g['column_header_y_mm']-1.5, g['top_tick_y_mm']-1))
        tick_artists = []
        def tick_labels(y):
            for value in ticks:
                # Numerical annotations follow the exact tick coordinate, just
                # like the data marks; the modular grid governs layout edges.
                artist = text(to_x(value), y, f"{value:g}{style['axis']['tick_suffix']}", fonts['tick_pt'], font_role='tick', ha=style['axis']['tick_align'], color=style['rows']['secondary_color'], slot=(0, width, y-1, y+1))
                tick_artists.append(artist)
        tick_labels(g['top_tick_y_mm'])
        base = g['header_mm']+layout['row_count']*g['row_pitch_mm']
        axis = style['axis']
        line, = ax.plot([g['plot_left_mm'],g['plot_right_mm']], [g['top_axis_y_mm']]*2, color=axis['color'], linewidth=axis['linewidth_pt'])
        line.set_gid('forest-top-axis')
        for index, value in enumerate(ticks):
            line, = ax.plot([to_x(value)]*2, [g['top_axis_y_mm']-.5,g['top_axis_y_mm']], color=axis['color'], linewidth=axis['linewidth_pt'])
            line.set_gid(f'forest-axis-tick-{index:02d}')
        grid = style['grid']
        grid_values = []
        if grid['show']:
            grid_values = _merged_grid_values(domain, ticks, grid['divisions'])
            for index, value in enumerate(grid_values):
                line, = ax.plot([to_x(value)]*2, [g['top_axis_y_mm'],base], color=grid['color'], linewidth=grid['linewidth_pt'], alpha=grid['alpha'], linestyle=grid['linestyle'], zorder=1)
                line.set_gid(f'forest-numerical-grid-{index:02d}')
        if 'reference' in info.get('x', {}):
            ref = style['reference']
            line, = ax.plot([to_x(info['x']['reference'])]*2, [g['top_axis_y_mm'],base], color=ref['color'], linewidth=ref['linewidth_pt'], linestyle=ref['linestyle'], zorder=2)
            line.set_gid('forest-reference')
        for index, row in enumerate(layout['rows']):
            top = row['top_mm']
            rows = style['rows']
            row_fill = rows['fills'][index % len(rows['fills'])]
            from matplotlib.colors import to_rgba
            if to_rgba(row_fill)[3]:
                ax.add_patch(Rectangle((0, top), width, g['row_band_mm'], facecolor=row_fill, edgecolor='none', zorder=0))
            if rows['separators']:
                line, = ax.plot([origin, g['plot_right_mm']], [top+g['row_band_mm']]*2, color=rows['separator_color'], linewidth=rows['separator_linewidth_pt'])
                line.set_gid(f'forest-row-separator-{index:03d}')
            trio = [text(cell_x(column), top+g['cell_y_mm'], value, column['fontsize'],
                         slot=(column['left_mm'], column['left_mm']+column['width_mm'], top, top+g['plot_y_mm']-g['point_band_mm']/2), row=row['row'],
                         font_role=column['_font_role'], parse_math=column.get('math', False), fontweight=column['fontweight'], ha=column['align'], color=rows['text_color'] if column['role'] != 'left' else rows['secondary_color'],
                         bbox={'facecolor':row_fill if to_rgba(row_fill)[3] else rows['background'], 'edgecolor':'none', 'boxstyle':'square,pad=0.04'})
                    for column, value in zip(layout['columns'], row['cells'])]
            annotations.append((row['row'], [a for a in trio if a is not None]))
            yc = top+g['plot_y_mm']
            point_yc = yc
            graphic_groups = {'primary': []}
            row_graphics[row['row']] = graphic_groups
            band = (0, width, yc-g['point_band_mm']/2, yc+g['point_band_mm']/2)
            for overlay in row.get('intervals', []):
                s = overlay['style']
                oy = yc+s['offset_mm']
                line, = ax.plot([to_x(overlay['lower']), to_x(overlay['upper'])], [oy]*2,
                                color=s['color'], alpha=s['alpha'], linewidth=s['linewidth_pt'], linestyle=s['linestyle'], zorder=2)
                line.set_gid('forest-overlay-'+row['row']+'-'+overlay['id'])
                marks = [line]
                for bound in ('lower', 'upper'):
                    cap, = ax.plot([to_x(overlay[bound])]*2, [oy-s['cap_height_mm']/2, oy+s['cap_height_mm']/2],
                                   color=s['color'], alpha=s['alpha'], linewidth=s['linewidth_pt'], zorder=2)
                    cap.set_gid('forest-overlay-cap-'+row['row']+'-'+overlay['id']+'-'+bound)
                    marks.append(cap)
                point, = ax.plot([to_x(overlay['estimate'])], [oy], linestyle='', marker=s['marker'],
                                color=s['color'], alpha=s['alpha'], markersize=s['diameter_mm']/MM_PER_PT,
                                markeredgewidth=s['linewidth_pt'], zorder=2)
                point.set_gid('forest-overlay-estimate-'+row['row']+'-'+overlay['id'])
                marks.append(point)
                graphic_groups['interval:'+overlay['id']] = marks
                checks.extend((mark, band, row['row']) for mark in marks)
                shared_key = style['overlays'].get('legend', {})
                handles.setdefault(f"@layer:{class_index[overlay['legend_object']]}", point if shared_key else ForestGlyph((line, point), glyph='interval'))
                if shared_key:
                    handles.setdefault(f"@layer:{class_index[layout.get('interval_summary_object', '@interval-summary')]}", ForestGlyph((line, point), glyph='interval', legend_color=shared_key.get('color', '#000000')))
                if overlay.get('annotation'):
                    annotation = style['overlays']['annotation']
                    gap = annotation.get('gap_mm', .5)
                    left = to_x(overlay['upper'])+gap
                    if g.get('value_column_mm', 0):
                        left = max(left, g['plot_right_mm']+gap)
                    label = text(g.get('value_right_mm', g['plot_right_mm']), oy, overlay['annotation'], annotation.get('fontsize', fonts['stats_pt']), font_role='annotation',
                                 parse_math=annotation.get('math', False), ha='right', color=s['color'], row=row['row'], slot=(left, g.get('value_right_mm', g['plot_right_mm']), top, top+g['row_band_mm']),
                                 bbox={'facecolor':rows['background'], 'edgecolor':'none', 'boxstyle':'square,pad=0.04'})
                    label.set_gid('forest-interval-annotation-'+row['row']+'-'+overlay['id'])
                    annotations[-1][1].append(label)
            yc += g.get('glyph_offset_mm', 0)
            c = row['coordinates']
            if c['lower'] is not None:
                s = style['interval']
                whisker, = ax.plot([to_x(c['lower']), to_x(c['upper'])], [yc]*2, color=s['color'], linewidth=s['linewidth_pt'], zorder=3)
                whisker.set_gid('forest-primary-whisker-'+row['row'])
                graphic_groups['primary'].append(whisker)
                for bound in ('lower', 'upper'):
                    cap, = ax.plot([to_x(c[bound])]*2, [yc-s['cap_height_mm']/2, yc+s['cap_height_mm']/2], color=s['color'], linewidth=s['linewidth_pt'], zorder=3)
                    cap.set_gid('forest-primary-cap-'+row['row']+'-'+bound)
                    graphic_groups['primary'].append(cap)
                if info.get('glyph', 'interval') == 'box':
                    b = style['box']
                    box = Rectangle((to_x(c['q1']), yc-b['height_mm']/2), to_x(c['q3'])-to_x(c['q1']), b['height_mm'], facecolor=b['facecolor'], edgecolor=b['edgecolor'], alpha=b['alpha'], linewidth=b['linewidth_pt'], zorder=3)
                    ax.add_patch(box)
                    box.set_gid('forest-primary-box-'+row['row'])
                    median, = ax.plot([to_x(c['median'])]*2, [yc-b['height_mm']/2, yc+b['height_mm']/2], color=b['median_color'], linewidth=b['linewidth_pt'], zorder=4)
                    median.set_gid('forest-primary-median-'+row['row'])
                    graphic_groups['primary'].extend([box, median])
                    glyph = ForestGlyph((whisker, box, median), glyph='box')
                    extent = b['height_mm']/2+b['linewidth_pt']*MM_PER_PT/2
                else:
                    point, = ax.plot([to_x(c['estimate'])], [yc], linestyle='', marker=s['marker'], color=s['color'], markersize=s['diameter_mm']/MM_PER_PT, markeredgewidth=s['linewidth_pt'], zorder=4)
                    point.set_gid('forest-primary-estimate-'+row['row'])
                    graphic_groups['primary'].append(point)
                    glyph = ForestGlyph((whisker, point), glyph='interval')
                    extent = (s['diameter_mm']+s['linewidth_pt']*MM_PER_PT)/2
                extent = max(extent, (s['cap_height_mm']+s['linewidth_pt']*MM_PER_PT)/2)
                if abs(g.get('glyph_offset_mm', 0))+extent > g['point_band_mm']/2:
                    raise ValueError(f"row {row['row']!r}: glyph exceeds reserved plot band; increase forest.geometry.point_band_mm and row band/pitch")
                handles.setdefault(f"@layer:{class_index[layout['glyph_object']]}", glyph)
                if row.get('box_annotation'):
                    annotation = style['box']['annotation']
                    in_text_band = annotation.get('position', 'box') == 'row_text'
                    ay = top+g['cell_y_mm'] if in_text_band else yc
                    gap = annotation.get('gap_mm', .5)
                    left = to_x(c['upper'])+gap
                    if g.get('value_column_mm', 0):
                        left = max(left, g['plot_right_mm']+gap)
                    slot = ((g['plot_left_mm'], g.get('value_right_mm', g['plot_right_mm']), top, top+g['plot_y_mm']-g['point_band_mm']/2) if in_text_band else
                            (left, g.get('value_right_mm', g['plot_right_mm']), top, top+g['row_band_mm']))
                    label = text(g.get('value_right_mm', g['plot_right_mm']), ay, row['box_annotation'], annotation.get('fontsize', fonts['stats_pt']), font_role='annotation',
                                 parse_math=annotation.get('math', False), ha='right', color=annotation.get('color', style['interval']['color']), row=row['row'], slot=slot,
                                 bbox={'facecolor':rows['background'], 'edgecolor':'none', 'boxstyle':'square,pad=0.04'})
                    label.set_gid('forest-box-annotation-'+row['row'])
                    annotations[-1][1].append(label)
            for p in row['points']:
                s = p['style']
                colors = {'color': s['color']} if s['marker'] == 'x' else {'facecolors': s['color'] if s['fill'] else 'none', 'edgecolors': s.get('edgecolor', s['color'])}
                mark = ax.scatter([p['x_mm']], [point_yc+p['offset_mm']], marker=s['marker'], s=(s['diameter_mm']/MM_PER_PT)**2, **colors, linewidths=s['linewidth_pt'], zorder=5)
                mark.set_gid('forest-point-'+row['row']+'-'+p['id'])
                graphic_groups['primary'].append(mark)
                handles.setdefault(f"@layer:{class_index[p['class']]}", mark)
            checks.extend((mark, band, row['row']) for mark in graphic_groups['primary'])
        legend_config = dict(layout['legend']) if isinstance(layout['legend'], dict) else layout['legend']
        base = g['header_mm']+layout['row_count']*g['row_pitch_mm']
        description = info.get('description', {})
        desc_origin = g['grid_lines_mm'][g['description_start_column']-1]
        desc_right = desc_origin+g['description_columns']*g['grid_unit_mm']
        desc_size, desc_weight = description.get('fontsize', fonts['footer_pt']), description.get('fontweight', fonts['_font_roles']['caption']['weight'])
        desc_prop = forest_font_properties(fonts, 'caption', desc_size, desc_weight)
        if 'math_fontfamily' in description:
            desc_prop.set_math_fontfamily(description['math_fontfamily'])
        layout['description_fonts'] = {'family': desc_prop.get_family(), 'size_pt': desc_size,
                                       'math_fontfamily': desc_prop.get_math_fontfamily(),
                                       'parse_math': description.get('math', False)}
        description_artists = []
        for i, line in enumerate(wrap_text(description.get('text', '').strip(), desc_size, fonts['family'], desc_weight, desc_right-desc_origin, math=description.get('math', False), prop=desc_prop)):
            left_overhang = text_metrics_mm(line, desc_size, fonts['family'], desc_weight, math=description.get('math', False), prop=desc_prop)[1]
            artist = text(desc_origin+left_overhang, base+g['description_y_mm']+i*g['text_line_pitch_mm'], line, desc_size,
                parse_math=description.get('math', False), fontproperties=desc_prop, color=description.get('color', style['rows']['secondary_color']),
                slot=(desc_origin, desc_right, base+g['legend_y_mm'], height-1))
            if artist is not None:
                artist.set_gid(f'forest-description-{i:02d}')
                description_artists.append(artist)
        grouped = isinstance(legend_config, dict) and legend_config.get('ncols') == 2
        legend_origin = g['grid_lines_mm'][g['legend_start_columns'][0]-1] if grouped else origin
        text(legend_origin, base+g['legend_heading_y_mm'], style['rows']['legend_heading'], fonts['meta_pt'], font_role='caption', color=style['rows']['accent'], slot=(legend_origin, g['plot_right_mm'], base+3.5, base+g['legend_y_mm']-1))
        if isinstance(legend_config, dict):
            legend_config.setdefault('bbox_to_anchor', (legend_origin/width, 1-(base+g['legend_y_mm'])/height))
            legend_config.setdefault('borderpad', 0)
            from ..legend_cards import resolve_legend_config
            from matplotlib.font_manager import FontProperties
            resolved_legend, _ = resolve_legend_config(legend_config, base_dir=self._yaml_dir)
            legend_prop = resolved_legend.get('prop')
            size = (FontProperties._from_any(legend_prop).get_size_in_points() if legend_prop is not None
                    else FontProperties(size=resolved_legend.get('fontsize', 5)).get_size_in_points())
            legend_config['handlelength'] = g['grid_unit_mm']/MM_PER_PT/size-resolved_legend.get('handletextpad', .6)
            legend_config['handler_map'] = {**legend_config.get('handler_map', {}), ForestGlyph: HandlerForestGlyph(), PathCollection: HandlerForestPoint(g['legend_point_key_mm']/MM_PER_PT)}
        legend = apply_legend(ax, legend_config, handles, layer_specs=layout['legend_layers'], base_dir=self._yaml_dir)
        if legend:
            group_glyph_and_points(legend, legend_config, layout['legend_layers'], layout['glyph_object'], base_dir=self._yaml_dir, left_objects=layout.get('legend_left_objects'))
            from ..legend_cards import resolve_axes_legend
            _, entries = resolve_axes_legend(legend_config, layout['legend_layers'], base_dir=self._yaml_dir)
            if legend_config.get('reverse', False):
                entries = list(reversed(entries))
            label_classes = {p['legend_object']: p['style']['color'] for row in layout['rows'] for p in row.get('intervals', [])}
            explicit_text_family = (legend_prop is not None and
                                    (not isinstance(legend_prop, dict) or bool(set(legend_prop) & {'family', 'fname'})))
            for entry, artist in zip(entries, legend.get_texts()):
                native_prop = artist.get_fontproperties().copy()
                if not explicit_text_family:
                    # Keep the forest card's installed literal font face while
                    # preserving the independent math family from legend JSON.
                    native_prop.set_family(fonts['family'])
                families = native_prop.get_family()
                if native_prop.get_file() is None and len(families) == 1:
                    # An explicitly named single family needs the same TTC
                    # face selection as the forest default. Keep native fname
                    # overrides and ordered fallback-family lists untouched.
                    native_prop.set_file(font_properties(artist.get_fontsize(), families[0], native_prop.get_weight(), style=native_prop.get_style()).get_file())
                artist.set_fontproperties(native_prop)
                artist.set_color(label_classes.get(entry['object'], style['rows']['text_color']) if style.get('overlays', {}).get('legend') else style['rows']['text_color'])
                artist.set_ha('left')
                artist.set_parse_math(style.get('legend_math', False))
                artist.set_usetex(False)
            # Keep canonical items, with keys at declared reference columns.
            # Labels start one reference column after their keys.
            ncols = legend._ncols
            if g['grid_columns'] % ncols:
                raise ValueError('Swiss forest legend ncols must divide the 12-column grid')
            legend._legend_handle_box.sep = 0
            spans = ([g['legend_start_columns'][1]-g['legend_start_columns'][0],
                      (g['description_start_column'] if description.get('text', '').strip() else g['grid_columns']+1)-g['legend_start_columns'][1]] if ncols == 2
                     else [g['grid_columns']/ncols]*ncols)
            physical_columns = []
            for column, span in zip(legend._legend_handle_box.get_children(), spans):
                column = PhysicalLegendColumn(span*g['grid_unit_mm'], pad=column.pad, sep=column.sep,
                                              align='left', children=column.get_children())
                physical_columns.append(column)
                span_px = span*g['grid_unit_mm']/MM_PER_PT*self.fig.canvas.get_renderer().points_to_pixels(1)
                for item in column.get_children():
                    if item.get_children():
                        key = item.get_children()[0]
                        if any(isinstance(a, PathCollection) for a in key.get_children()):
                            key.width = g['legend_point_key_mm']/MM_PER_PT
                    if ncols == 2 and item.get_children():
                        label = item.get_children()[-1]._text
                        key_and_gap = (key.width+item.sep)*MM_PER_PT
                        prop = label.get_fontproperties()
                        label.set_text('\n'.join(wrap_text(label.get_text(), label.get_fontsize(), prop.get_family()[0], prop.get_weight(), span*g['grid_unit_mm']-key_and_gap,
                                                          math=style.get('legend_math', False), prop=prop)))
                    if item.get_bbox(self.fig.canvas.get_renderer()).width > span_px:
                        raise ValueError(f'Swiss forest legend item exceeds its modular span; reduce frame.ax.legend.ncols ({ncols}) or fontsize, or shorten labels')
            legend._legend_handle_box._children = physical_columns
            layout['legend_fonts'] = {
                'family': legend.get_texts()[0].get_fontproperties().get_family(),
                'size_pt': legend.get_texts()[0].get_fontsize(),
                'math_fontfamily': legend.get_texts()[0].get_math_fontfamily(),
            }
            checks.append((legend, (0, width, base+3.5, height-1), None))
        tick_labels(base+2)
        if 'axlogo' in self.axes:
            for artist in self.axlogo.get_children():
                from matplotlib.text import Text
                if isinstance(artist, Text) and artist.get_text():
                    checks.append((artist, (0, width, height-g['logo_bottom_mm']-g['logo_size_mm']-1, height), None))
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        def mm_bounds(artist):
            b = artist.get_window_extent(renderer)
            # Text's window bbox reserves the font's unused ascender/descender
            # space (e.g. 'lp' even for B001). Measure actual literal glyph ink
            # against the physical slot; do not shrink type to that padding.
            from matplotlib.text import Text
            if isinstance(artist, Text):
                from matplotlib.textpath import TextToPath
                from matplotlib.path import Path as MplPath
                from matplotlib.transforms import Bbox
                anchor = artist.get_transform().transform(artist.get_position())
                _, lines, _ = artist._get_layout(renderer)
                bounds = []
                for line in lines:
                    label = line[0]
                    ox, oy = line[2] if len(line) == 3 else line[2:4]
                    from matplotlib.cbook import is_math_text
                    vertices, codes = TextToPath().get_text_path(artist.get_fontproperties(), label, ismath=artist.get_parse_math() and is_math_text(label))
                    if len(vertices):
                        ink = MplPath(vertices, codes).get_extents()
                        scale = artist.get_fontsize()/100*self.fig.dpi/72
                        bounds.append(Bbox.from_extents(anchor[0]+ox+ink.x0*scale, anchor[1]+oy+ink.y0*scale,
                                                       anchor[0]+ox+ink.x1*scale, anchor[1]+oy+ink.y1*scale))
                if bounds:
                    b = Bbox.union(bounds)
            elif isinstance(artist, PathCollection):
                # Scatter collections report an empty window extent. Measure
                # the rendered marker path and its offset instead.
                from matplotlib.transforms import Affine2D, Bbox
                transforms = artist.get_transforms()
                marker = artist.get_paths()[0]
                if len(transforms):
                    marker = marker.transformed(Affine2D(transforms[0]))
                marker = marker.transformed(artist.get_transform())
                ink = marker.get_extents()
                offset = artist.get_offset_transform().transform(artist.get_offsets()[0])
                b = Bbox.from_extents(offset[0]+ink.x0, offset[1]+ink.y0, offset[0]+ink.x1, offset[1]+ink.y1)
                pad = max(artist.get_linewidths(), default=0)/2*self.fig.dpi/72
                b = b.padded(pad)
            else:
                from matplotlib.lines import Line2D
                if isinstance(artist, Line2D):
                    stroke = artist.get_linewidth() if artist.get_linestyle() not in ('None', 'none', '') else 0
                    stroke = max(stroke, artist.get_markeredgewidth() if artist.get_marker() not in ('None', 'none', '', None) else 0)
                    b = b.padded(stroke/2*self.fig.dpi/72)
                elif isinstance(artist, Rectangle):
                    b = b.padded(artist.get_linewidth()/2*self.fig.dpi/72)
            return (b.x0/self.fig.dpi*25.4, b.x1/self.fig.dpi*25.4,
                    height-b.y1/self.fig.dpi*25.4, height-b.y0/self.fig.dpi*25.4)
        for artist, slot, row in checks:
            bounds = mm_bounds(artist)
            tolerance = max(0.02, 25.4/self.fig.dpi/2)  # half a raster pixel for font hinting
            if slot and (bounds[0] < slot[0]-tolerance or bounds[1] > slot[1]+tolerance or bounds[2] < slot[2]-tolerance or bounds[3] > slot[3]+tolerance):
                label = artist.get_text() if hasattr(artist, 'get_text') else artist.get_gid() or 'legend'
                raise ValueError(f"figure {self.name!r}, row {row!r}: text/layout overflow for {label!r}; measured mm {bounds}, slot {slot}. Adjust annotation/typography/geometry or frame.ax.legend.")
        from matplotlib.transforms import Bbox
        def physical_box(artist):
            b = mm_bounds(artist)
            return Bbox.from_extents(b[0], b[2], b[1], b[3])
        header_boxes = [physical_box(artist) for artist in header_artists]
        for i, box in enumerate(header_boxes):
            if any(box.overlaps(other) for other in header_boxes[i+1:]):
                raise ValueError('forest header text overlaps; adjust title/header_note/column labels or forest.geometry header baselines')
        description_boxes = [physical_box(artist) for artist in description_artists]
        for i, box in enumerate(description_boxes):
            if any(box.overlaps(other) for other in description_boxes[i+1:]):
                raise ValueError('forest footer description lines overlap; increase forest.geometry.text_line_pitch_mm/footer_mm or adjust description fontsize')
        badge_boxes = ([physical_box(a) for a in [*self.axlogo.images, *self.axlogo.texts]] if 'axlogo' in self.axes else [])
        if any(box.overlaps(badge) for box in description_boxes for badge in badge_boxes):
            raise ValueError('forest footer description overlaps axlogo; adjust description columns/baselines or forest.geometry footer/logo positions')
        if legend:
            for column in legend._legend_handle_box.get_children():
                for item in column.get_children():
                    if any(boxes_overlap(physical_box(item), b) for b in badge_boxes):
                        raise ValueError('forest footer legend overlaps axlogo; adjust forest.geometry footer/legend/logo positions or frame.ax.legend layout')
                    if any(boxes_overlap(physical_box(item), b) for b in description_boxes):
                        raise ValueError('forest footer legend overlaps description; adjust forest.geometry column starts or frame.ax.legend layout')
        for row, artists in annotations:
            boxes = [physical_box(artist) for artist in artists]
            for i, b in enumerate(boxes):
                if any(b.overlaps(other) for other in boxes[i+1:]):
                    raise ValueError(f'row {row!r}: annotation slots collide; enlarge forest.geometry slots or change literal annotations')
                if any(b.overlaps(physical_box(mark)) for marks in row_graphics[row].values() for mark in marks):
                    raise ValueError(f'row {row!r}: annotation overlaps a glyph, point or interval; increase value_column_mm or row/lane spacing')
            groups = [(name, Bbox.union([physical_box(mark) for mark in marks])) for name, marks in row_graphics[row].items() if marks]
            for i, (name, box) in enumerate(groups):
                if any(box.overlaps(other) for _, other in groups[i+1:]):
                    raise ValueError(f'row {row!r}: interval lanes overlap ({name}); adjust glyph_offset_mm/offset_mm and point_band_mm')
        layout['verification'] = {'text_and_legend_bounds': 'passed', 'graphic_bounds': 'passed', 'annotation_collisions': 'passed', 'all_points_once': True, 'all_intervals_once': True, 'packing': 'deterministic_physical_mm', 'numeric_x_preserved': True, 'tick_label_count': len(tick_artists), 'axis_tick_count': len(ticks), 'grid_line_count': len(grid_values)}

    def savefig(self):
        import matplotlib as mpl
        with mpl.rc_context({'svg.fonttype': 'none', 'text.hinting': 'none', 'text.usetex': False}):
            super().savefig()
        path = Path(self.dir) / f'{self.name}.layout.json'
        path.write_text(json.dumps(self.forest_layout, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
