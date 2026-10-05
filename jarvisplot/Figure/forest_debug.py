"""Read-only Swiss grid and physical band references on a separate overlay."""
from matplotlib.patches import Rectangle
from ..forest_text import font_properties


def draw_forest_reference(fig, cfg):
    if not cfg['show'] or any(ax.get_label() == 'forest-design-reference' for ax in fig.fig.axes):
        return
    scene = fig.forest_layout
    g = scene['geometry']
    ax = fig.fig.add_axes([0, 0, 1, 1], label='forest-design-reference', zorder=10000)
    ax.set(xlim=(0, scene['width_mm']), ylim=(scene['height_mm'], 0)); ax.axis('off')
    ax.patch.set_alpha(0)
    if cfg['columns']['show']:
        for index, x in enumerate(g['grid_lines_mm']):
            line, = ax.plot([x]*2, [0, scene['height_mm']], **cfg['columns']['plot'])
            line.set_gid(f'forest-grid-{index:02d}')
    if cfg['labels']['show']:
        opts = dict(cfg['labels']['text'])
        size = opts.pop('fontsize')
        for index, x in enumerate(g['grid_lines_mm'][:-1]):
            ax.text(x, g['top_axis_y_mm']+.35, f'{index+1:02d}', fontproperties=font_properties(size, scene['typography']['family']), **opts)
    if cfg['baselines']['show']:
        ys = [g[key] for key in ('meta_y_mm','title_y_mm','description_y_mm','x_label_y_mm','column_header_y_mm','top_tick_y_mm')]
        ys += [r['top_mm']+g[key] for r in scene['rows'] for key in ('cell_y_mm','plot_y_mm')]
        for index, y in enumerate(ys):
            line, = ax.plot([g['plot_left_mm'], g['plot_right_mm']], [y]*2, **cfg['baselines']['plot'])
            line.set_gid(f'forest-baseline-{index:03d}')
    if cfg['bands']['show']:
        for index, row in enumerate(scene['rows']):
            for name, y, height in (('text', row['top_mm'], g['plot_y_mm']-g['point_band_mm']/2),
                                    ('marks', row['top_mm']+g['plot_y_mm']-g['point_band_mm']/2, g['point_band_mm'])):
                box = Rectangle((g['plot_left_mm'], y), scene['plot_width_mm'], height, **cfg['bands']['Rectangle'])
                box.set_gid(f'forest-band-{name}-{index:03d}')
                ax.add_patch(box)
