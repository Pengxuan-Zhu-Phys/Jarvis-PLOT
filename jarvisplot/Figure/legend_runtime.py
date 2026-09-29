"""Native legends with reusable spacing and keys composed from real layers."""
from __future__ import annotations

from dataclasses import dataclass

from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase, HandlerTuple
from matplotlib.transforms import Affine2D
from ..legend_cards import resolve_axes_legend



@dataclass(frozen=True)
class StackedKey:
    handles: tuple


class HandlerStack(HandlerBase):
    """Place constituent handles in vertical slots inside one native key."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        artists = []
        slot = height / len(orig_handle.handles)
        handler_map = legend.get_legend_handler_map()
        for index, handle in enumerate(orig_handle.handles):
            handler = legend.get_legend_handler(handler_map, handle)
            # Position the whole delegated key, rather than encoding the offset
            # in ydescent: handlers interpret descent differently (Line2D halves
            # it), which compresses the gaps and shifts the stack off center.
            # YAML order is top to bottom inside the original handle box.
            # Native line handlers center at (height - ydescent) / 2.
            # Split the descent equally so increasing handleheight does not
            # shift a stacked center line below adjacent native line keys.
            offset = height - (index + 1) * slot - ydescent / 2
            slot_transform = Affine2D().translate(0, offset) + trans
            artists.extend(handler.create_artists(
                legend, handle, xdescent, 0,
                width, slot, fontsize, slot_transform,
            ))
        return artists


def representative_handle(rendered):
    """Require one supported artist/container; multiple series are ambiguous."""
    from matplotlib.artist import Artist
    from matplotlib.container import Container

    handler_map = Legend.get_default_handler_map()
    def collect(value):
        if isinstance(value, (Artist, Container)):
            return [value] if Legend.get_legend_handler(handler_map, value) is not None else []
        if isinstance(value, (list, tuple)):
            return [handle for child in value for handle in collect(child)]
        return []

    handles = collect(rendered)
    if len(handles) > 1:
        raise ValueError("Layer returned multiple legend handles; split its series into separate layers with explicit legend declarations")
    return handles[0] if handles else None


def apply_legend(ax, config, layer_handles=None, *, base_dir=None, layer_specs=None):
    """Apply frame.<axes>.legend without modifying config or data artists."""
    kw, entries = resolve_axes_legend(config, layer_specs or [], base_dir=base_dir)
    if config is False or (isinstance(config, dict) and config.get("enabled", True) is False):
        return None

    handles, labels = [], []
    layer_handles = layer_handles or {}
    for entry in entries:
        names, key = entry["layers"], entry["key"]
        parts = []
        for name in names:
            if name not in layer_handles:
                raise ValueError(f"Legend object {entry['object']!r} is missing its rendered component {name!r}")
            handle = representative_handle(layer_handles[name])
            if handle is None:
                raise ValueError(f"Legend object {entry['object']!r} component {name!r} has no supported legend handle")
            parts.append(handle)
        if len(parts) == 1:
            handles.append(parts[0])
        elif key == "stack":
            handles.append(StackedKey(tuple(parts)))
        else:
            handles.append(tuple(parts))
        labels.append(entry["label"])

    kw["handler_map"] = {
        tuple: HandlerTuple(ndivide=1),
        StackedKey: HandlerStack(),
        **kw.get("handler_map", {}),
    }
    return ax.legend(handles=handles, labels=labels, **kw)
