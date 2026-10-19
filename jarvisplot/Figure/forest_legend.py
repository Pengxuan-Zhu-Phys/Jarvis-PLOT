"""A native legend handler for the actual forest glyph artist container."""
from matplotlib.container import Container
from matplotlib.legend_handler import HandlerBase, HandlerPathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import VPacker


def boxes_overlap(first, second):
    """Require positive overlap on both physical axes, measured in mm.

    Footer columns and the badge share grid boundaries. Touching those edges
    is valid; Bbox.overlaps also counts contact and floating-point roundoff.
    """
    width = min(first.xmax, second.xmax) - max(first.xmin, second.xmin)
    height = min(first.ymax, second.ymax) - max(first.ymin, second.ymin)
    return min(width, height) > 1e-9


class PhysicalLegendColumn(VPacker):
    """Recompute physical column width for each raster/vector renderer."""
    def __init__(self, width_mm, **kwargs):
        super().__init__(**kwargs)
        self.width_pt = width_mm*72/25.4

    def _get_bbox_and_child_offsets(self, renderer):
        self.width = self.width_pt*renderer.points_to_pixels(1)
        return super()._get_bbox_and_child_offsets(renderer)


class ForestGlyph(Container):
    def __init__(self, artists, *, glyph, legend_color=None):
        super().__init__(artists)
        self.glyph = glyph
        self.legend_color = legend_color


class HandlerForestPoint(HandlerPathCollection):
    """Place one point at the left of its compact native key."""
    def __init__(self, key_width_pt):
        super().__init__(numpoints=1)
        self.key_width_pt = key_width_pt

    def get_xdata(self, legend, xdescent, ydescent, width, height, fontsize):
        x = -xdescent+self.key_width_pt/2
        return [x, x], [x]


def group_glyph_and_points(legend, config, layers, glyph_object, *, base_dir=None, left_objects=None):
    """Use the canonical native items, with glyphs left and point classes right.

    Matplotlib normally balances item counts across columns, which would place
    some point classes beside the box sample. Repack its existing item boxes
    without recreating handles, labels, or their handlers.
    """
    from matplotlib.offsetbox import DrawingArea
    from ..legend_cards import resolve_axes_legend
    kwargs, entries = resolve_axes_legend(config, layers, base_dir=base_dir)
    if kwargs.get('ncols', kwargs.get('ncol', 1)) != 2:
        return
    legend._ncols = 2
    if kwargs.get('reverse', False):
        entries = list(reversed(entries))
    objects = {id(text): entry['object'] for text, entry in zip(legend.get_texts(), entries)}
    columns = legend._legend_handle_box.get_children()
    groups = [[], []]
    left_objects = set(left_objects or [glyph_object])
    for column in columns:
        for item in column.get_children():
            label = item.get_children()[-1]._text
            groups[0 if objects[id(label)] in left_objects else 1].append(item)
    # Preserve the empty semantic column when only one group is present.
    legend._legend_handle_box._children = [
        VPacker(pad=0, sep=legend.labelspacing*legend._fontsize,
                align='left', children=items or [DrawingArea(0, 0)])
        for items in groups
    ]
    legend._legend_handle_box.align = 'top'


class HandlerForestGlyph(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        whisker = orig_handle[0]
        left, right = -xdescent, width-xdescent
        yc = (height-ydescent)/2
        half = height*.32
        stroke = {'color': orig_handle.legend_color or whisker.get_color(), 'linewidth': whisker.get_linewidth(), 'alpha': 1 if orig_handle.legend_color else whisker.get_alpha(), 'transform': trans}
        artists = [Line2D([left, right], [yc, yc], **stroke)]
        for xx in (left, right):
            artists.append(Line2D([xx, xx], [yc-half, yc+half], **stroke))
        if orig_handle.glyph == 'box':
            box, median = orig_handle[1:]
            artists.append(Rectangle((left+width*.25, yc-half), width*.5, half*2,
                                     facecolor=box.get_facecolor(), edgecolor=box.get_edgecolor(),
                                     linewidth=box.get_linewidth(), alpha=box.get_alpha(), transform=trans))
            artists.append(Line2D([(left+right)/2]*2, [yc-half, yc+half], color=median.get_color(), linewidth=median.get_linewidth(), transform=trans))
        else:
            point = orig_handle[1]
            artists.append(Line2D([(left+right)/2], [yc], linestyle='', marker=point.get_marker(), color=orig_handle.legend_color or point.get_color(),
                                  markersize=point.get_markersize(), markeredgewidth=point.get_markeredgewidth(), alpha=1 if orig_handle.legend_color else point.get_alpha(), transform=trans))
        return artists
