#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .utils.pathing import resolve_project_path


DEFAULT_FLOWCHART_CARD = "&JP/jarvisplot/cards/flowchart/default.json"


def _deep_merge(base: dict, override: Mapping | None) -> dict:
    if not isinstance(override, Mapping):
        return base
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_flowchart_card(card_path: str | None = None) -> dict:
    path = str(resolve_project_path(card_path or DEFAULT_FLOWCHART_CARD))
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def render_flowchart_file(
    scene_path: str,
    output_path: str | None = None,
    *,
    card_path: str | None = None,
    style_override: Mapping | None = None,
    logger=None,
) -> str:
    """Render a Jarvis-HEP flowchart scene JSON and return the image path."""
    scene_file = Path(scene_path).expanduser().resolve()
    with open(scene_file, "r", encoding="utf-8") as handle:
        scene = json.load(handle)

    card = _deep_merge(load_flowchart_card(card_path), style_override)
    renderer = FlowchartRenderer(scene, card, logger=logger)

    target = _resolve_output_path(scene_file, output_path, card)
    renderer.render(target)
    return str(target)


def render_flowchart(
    scene_or_path: Mapping[str, Any] | str | os.PathLike[str],
    output_path: str | os.PathLike[str] | None = None,
    *,
    card_path: str | None = None,
    style_override: Mapping | None = None,
    logger=None,
) -> str:
    """Render a flowchart scene mapping or scene JSON path and return the image path."""
    if isinstance(scene_or_path, Mapping):
        scene = dict(scene_or_path)
        card = _deep_merge(load_flowchart_card(card_path), style_override)
        target = _resolve_output_path_from_scene(scene, output_path, card)
        FlowchartRenderer(scene, card, logger=logger).render(target)
        return str(target)
    return render_flowchart_file(
        str(scene_or_path),
        output_path=str(output_path) if output_path is not None else None,
        card_path=card_path,
        style_override=style_override,
        logger=logger,
    )


def _resolve_output_path(scene_file: Path, output_path: str | None, card: Mapping) -> Path:
    fmt = str(card.get("output", {}).get("format", "png")).lstrip(".") or "png"
    if output_path:
        target = Path(output_path).expanduser()
        if not target.is_absolute():
            target = Path.cwd() / target
        if target.suffix:
            return target.resolve()
        return (target / f"{scene_file.stem}.{fmt}").resolve()
    return scene_file.with_suffix(f".{fmt}").resolve()


def _resolve_output_path_from_scene(scene: Mapping[str, Any], output_path: str | os.PathLike[str] | None, card: Mapping) -> Path:
    fmt = str(card.get("output", {}).get("format", "png")).lstrip(".") or "png"
    if output_path:
        target = Path(output_path).expanduser()
        if not target.is_absolute():
            target = Path.cwd() / target
        if target.suffix:
            return target.resolve()
        scene_id = str(scene.get("scene_id") or scene.get("metadata", {}).get("workflow_name") or "flowchart")
        return (target / f"{scene_id}.{fmt}").resolve()
    scene_id = str(scene.get("scene_id") or scene.get("metadata", {}).get("workflow_name") or "flowchart")
    return (Path.cwd() / f"{scene_id}.{fmt}").resolve()


def _ordered_scene_layers(scene: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_layers = [layer for layer in scene.get("layers", []) if isinstance(layer, Mapping)]

    def layer_key(item):
        idx = item[1].get("index", item[0])
        try:
            return (float(idx), item[0])
        except Exception:
            return (float(item[0]), item[0])

    return [layer for _, layer in sorted(enumerate(raw_layers), key=layer_key)]


class FlowchartRenderer:
    def __init__(self, scene: Mapping[str, Any], card: Mapping[str, Any], logger=None):
        self.scene = dict(scene)
        self.card = dict(card)
        self.logger = logger

    def render(self, output_path: str | os.PathLike[str]) -> None:
        self._render_classic(output_path)

    def _validate_scene(self) -> None:
        if self.scene.get("schema") != "jarvisplot.scene/v1":
            raise ValueError("Unsupported flowchart schema; expected jarvisplot.scene/v1")
        if self.scene.get("scene_type") != "flowchart":
            raise ValueError("Unsupported scene_type; expected flowchart")
        for key in ("layers", "nodes", "edges"):
            if key not in self.scene:
                raise ValueError(f"Flowchart scene missing required field: {key}")

    def _render_classic(self, output_path: str | os.PathLike[str]) -> None:
        """Render in the legacy Jarvis-HEP visual grammar.

        Semantic variable nodes are shown as port labels; file nodes and module
        nodes stay visible. This preserves the detailed JSON structure without
        turning every variable into a separate block diagram node.
        """
        self._validate_scene()

        # Persistent cache under ~/.cache so Agg does not rebuild the font
        # cache on every Jarvis2 run (tempdir-based MPLCONFIGDIR was wiped often).
        _cache_root = os.path.join(os.path.expanduser("~"), ".cache", "jarvisplot")
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(_cache_root, "mplconfig"))
        os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_cache_root, "xdg"))
        os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
        os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

        import logging
        import warnings

        # Quiet noisy one-shot font-manager chatter (cache build / missing weights).
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        warnings.filterwarnings(
            "ignore",
            message=r".*findfont:.*",
            category=UserWarning,
            module=r"matplotlib(\..*)?",
        )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        graph = _ClassicGraph(self.scene, self.card)
        graph.layout()
        self._classic_logo_y = graph.logo_y

        fig, ax = plt.subplots(figsize=graph.figure_size())
        ax.set_axis_off()
        ax.set_xlim(graph.xlim)
        ax.set_ylim(graph.ylim)

        self._draw_classic_logo(ax, Image)
        self._draw_classic_edges(ax, graph)
        self._draw_classic_nodes(ax, graph, Image)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        savefig = dict(self.card.get("savefig", {}))
        savefig.setdefault("bbox_inches", "tight")
        savefig.setdefault("pad_inches", 0.05)
        fig.savefig(output, **savefig)
        plt.close(fig)
        if self.logger:
            self.logger.warning(f"JarvisPLOT flowchart rendered -> {output}")

    def _draw_classic_logo(self, ax, Image) -> None:
        icons = self.card.get("icons", {})
        logo_path = icons.get("logo", "&JP/jarvisplot/cards/icons/flowchart/JarvisHEP.png")
        classic = self.card.get("classic", {}) if isinstance(self.card.get("classic"), Mapping) else {}
        logo_size = float(classic.get("logo_size", 0.5))
        logo_half = logo_size / 2.0
        try:
            with Image.open(resolve_project_path(logo_path)) as image:
                arr = image.convert("RGBA")
                ax.imshow(arr, extent=[0.25, 0.25 + logo_size, self._classic_logo_y - logo_half, self._classic_logo_y + logo_half], zorder=100)
                ax.text(0.32 + logo_size, self._classic_logo_y + float(classic.get("logo_text_dy", 0.14)), "Jarvis-HEP", ha="left", va="top", color="#0F66C3", fontsize=7.5, fontweight="bold")
        except Exception:
            ax.text(0.25, self._classic_logo_y, "Jarvis-HEP", ha="left", va="center", color="#0F66C3", fontsize=8, fontweight="bold")

    def _draw_classic_nodes(self, ax, graph, Image) -> None:
        for item in graph.files.values():
            self._draw_classic_file(ax, item, Image)
        for item in graph.mains.values():
            self._draw_classic_main(ax, item, Image)
        for item in graph.bridges.values():
            self._draw_classic_bridge(ax, item)
        for item in graph.variables.values():
            self._draw_classic_variable(ax, item)
        for item in graph.selections.values():
            self._draw_classic_selection(ax, item)

    def _draw_classic_main(self, ax, item, Image) -> None:
        icons = self.card.get("icons", {})
        node = item["node"]
        metadata = node.get("metadata") if isinstance(node.get("metadata"), Mapping) else {}
        module_type = str(metadata.get("module_type") or node.get("role") or node.get("kind") or "").lower()
        if node.get("kind") == "source":
            icon_key = "sampler"
        elif "operas" in module_type or node.get("role") == "operas":
            icon_key = "opera"
        else:
            icon_key = "calculator"
        icon_path = icons.get(icon_key, f"&JP/jarvisplot/cards/icons/flowchart/{icon_key}.png")
        x, y = item["pos"]
        size = float(self.card.get("classic", {}).get("module_icon_size", 0.9))
        try:
            with Image.open(resolve_project_path(icon_path)) as image:
                arr = image.convert("RGBA")
                ax.imshow(arr, extent=[x - size / 2, x + size / 2, y - size / 2, y + size / 2], zorder=90)
        except Exception:
            from matplotlib.patches import FancyBboxPatch

            patch = FancyBboxPatch((x - 0.42, y - 0.42), 0.84, 0.84, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor="#2383D6", edgecolor="none", zorder=90)
            ax.add_patch(patch)
        label_dy = float(self.card.get("classic", {}).get("module_label_dy", -0.55))
        ax.text(x, y + label_dy, str(node.get("label") or node.get("id")), ha="center", va="top", fontfamily="sans-serif", fontsize=8.2, fontweight="bold", zorder=110)

    def _draw_classic_file(self, ax, item, Image) -> None:
        icons = self.card.get("icons", {})
        direction = item.get("direction", "input")
        icon_key = "input_file" if direction == "input" else "output_file"
        icon_path = icons.get(icon_key, f"&JP/jarvisplot/cards/icons/flowchart/{icon_key}.png")
        x, y = item["pos"]
        try:
            with Image.open(resolve_project_path(icon_path)) as image:
                arr = image.convert("RGBA")
                extent = [x - 0.25, x + 0.25, y - 0.25, y + 0.25]
                ax.imshow(arr, extent=extent, zorder=95)
        except Exception:
            from matplotlib.patches import FancyBboxPatch

            patch = FancyBboxPatch((x - 0.25, y - 0.24), 0.5, 0.48, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor="#2D9CDB", edgecolor="none", zorder=95)
            ax.add_patch(patch)
        label_dy = float(self.card.get("classic", {}).get("file_label_dy", -0.34))
        # Use "normal" not "light": many systems have no light weight → findfont noise.
        ax.text(x, y + label_dy, item.get("file_label") or item["node"].get("label") or "", ha="center", va="top", fontfamily="sans-serif", fontsize=6.2, fontweight="normal", zorder=110)

    def _draw_classic_variable(self, ax, item) -> None:
        x, y = item["pos"]
        label = str(item["node"].get("label") or item["node"].get("id", "").replace("var::", ""))
        align = item.get("align", "right")
        if align == "left":
            ha = "left"
            text_x = x + 0.12
            marker_x = x
        else:
            ha = "right"
            text_x = x
            marker_x = x - item.get("width", 0.42) - 0.1
        marker_x += float(item.get("marker_dx", 0.0))
        ax.text(text_x, y, label, ha=ha, va="center", fontfamily="monospace", fontsize=6.4, fontweight="bold", color="#111111", zorder=115)
        marker = item.get("marker_style") or ("s" if item["node"].get("role") == "parameter" else "o")
        ax.plot(marker_x, y, marker, markersize=3.4, color="#3B4DC0", alpha=1.0, zorder=118)
        item["marker"] = (marker_x, y)

    def _draw_classic_bridge(self, ax, item) -> None:
        x, y = item["pos"]
        ax.plot(x, y, "o", markersize=3.2, color="#3B4DC0", zorder=112)
        ax.text(x + 0.08, y, str(item["node"].get("label") or ""), ha="left", va="center", fontfamily="monospace", fontsize=5.8, fontweight="bold", color="#111111", zorder=112)
        item["marker"] = (x, y)

    def _draw_classic_selection(self, ax, item) -> None:
        from matplotlib.patches import FancyBboxPatch
        from matplotlib.patches import Polygon as MplPolygon
        from shapely.geometry import Point, box
        from shapely.ops import unary_union
        x, y = item["pos"] 

        def point_buffer(point, radius, segments=128):
            try:
                return point.buffer(radius, quad_segs=segments)
            except TypeError:
                return point.buffer(radius, resolution=segments)

        circle1 = point_buffer(Point(x-0.15, y), 0.15)
        circle2 = point_buffer(Point(x+0.15, y), 0.15)
        square = box(x - 0.15, y - 0.15, x + 0.15, y + 0.15)
        merged = unary_union([circle1, circle2, square])
        xx, yy = merged.exterior.xy
        ax.fill(xx, yy, alpha=0.95, edgecolor="#F9F9F8", facecolor="#00D86B", linewidth=0.1, zorder=120)
        circle3 = point_buffer(Point(x-0.15, y), 0.12)
        xx, yy = circle3.exterior.xy
        ax.fill(xx, yy, alpha=0.95, edgecolor="#FFFFFF", facecolor="#F9F9F8", linewidth=0.8, zorder=120)
        import matplotlib.patheffects as pe
        ax.text(x+0.12, y-0.01, "SEL", ha="center", va="center", fontsize=5, fontweight="extra bold", color="#008A34", zorder=121, path_effects=[pe.withStroke(linewidth=0.2, foreground="white")])

    def _draw_classic_edges(self, ax, graph) -> None:
        for edge in graph.edges:
            role = str(edge.get("role", "dataflow")).lower()
            source_id = str(edge.get("source", {}).get("node", ""))
            target_id = str(edge.get("target", {}).get("node", ""))
            start = graph.anchor(source_id, "out", role)
            end = graph.anchor(target_id, "in", role)
            if start is None or end is None:
                continue
            if role == "selectionflow":
                self._classic_gradient_curve(ax, start, end, lw=1.7, start_color="#3994E3", end_color="#58F271", alpha=0.95)
                ax.plot(start[0] + 0.06, start[1], "s", color="darkgray", markersize=2.6, zorder=80)
                ax.plot(start[0] + 0.03, start[1], "s", color="gray", markersize=2.0, zorder=79)      
            elif role == "fileflow" and (graph.is_var(source_id) or graph.is_bridge(source_id)) and graph.is_file(target_id):
                self._classic_gradient_curve(ax, start, end, lw=1.7)
                ax.plot(start[0] + 0.06, start[1], "s", color="darkgray", markersize=2.6, zorder=80)
                ax.plot(start[0] + 0.03, start[1], "s", color="gray", markersize=2.0, zorder=79)            
            elif role == "fileflow" and graph.is_file(source_id) and graph.is_main(target_id):
                end = (end[0], start[1])
                ax.plot(start[0] + 0.06, start[1], "s", color="darkgray", markersize=2.6, zorder=80)
                ax.plot(start[0] + 0.03, start[1], "s", color="gray", markersize=2.0, zorder=79)  
                self._classic_curve(ax, start, end, color="#D45040", lw=2.2, alpha=0.95)
            elif role == "fileflow" and graph.is_main(source_id) and graph.is_file(target_id):
                start = (start[0], end[1])
                self._classic_curve(ax, start, end, color="#3B4DC0", lw=2.2, alpha=0.95)
            elif role == "fileflow" and graph.is_file(source_id) and graph.is_var(target_id):
                self._classic_curve(ax, start, end, color="#3B4DC0", lw=0.75, alpha=0.7)
            elif role == "dataflow" and (graph.is_var(source_id) or graph.is_bridge(source_id)) and graph.is_main(target_id):
                self._classic_gradient_curve(ax, start, end, lw=1.7)
                ax.plot(start[0] + 0.06, start[1], "s", color="darkgray", markersize=2.6, zorder=80)
                ax.plot(start[0] + 0.03, start[1], "s", color="gray", markersize=2.0, zorder=79)
            elif role == "parameterflow":
                self._classic_curve(ax, start, end, color="#3B4DC0", lw=0.9, alpha=0.7)
            elif role == "bridgeflow":
                ax.plot(start[0] + 0.06, start[1], "s", color="darkgray", markersize=2.6, zorder=80)
                ax.plot(start[0] + 0.03, start[1], "s", color="gray", markersize=2.0, zorder=79)  
                self._classic_curve(ax, start, end, color="#3B4DC0", lw=1.4, alpha=0.75)
            else:
                self._classic_curve(ax, start, end, color="#3B4DC0", lw=1.0, alpha=0.78)

    def _classic_curve(self, ax, start, end, *, color, lw, alpha=1.0, linestyle="-") -> None:
        import numpy as np

        x0, y0 = start
        x1, y1 = end
        tt = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 80)
        xx = np.linspace(x0, x1, 80)
        yy = y0 + ((np.sin(np.sin(tt)) / 2 / 0.8414709848078965) + 0.5) * (y1 - y0)
        ax.plot(xx, yy, linestyle=linestyle, color=color, lw=lw, alpha=alpha, zorder=30)

    def _classic_gradient_curve(self, ax, start, end, *, lw, start_color=None, end_color=None, alpha=0.88) -> None:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.collections import LineCollection

        x0, y0 = start
        x1, y1 = end
        tt = np.linspace(-0.5 * np.pi, 0.4 * np.pi, 90)
        xx = np.linspace(x0 + 0.1, x1 - 0.15, 90)
        yy = y0 + (np.sin(np.sin(tt)) + 0.8414709848078965) / 1.6555005931675704 * (y1 - y0)
        points = np.array([xx, yy]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if start_color and end_color:
            cmap = LinearSegmentedColormap.from_list("flowchart_edge_gradient", [start_color, end_color])
            colors = cmap(np.linspace(0.0, 1.0, len(segments)))
        else:
            cmap = plt.get_cmap("coolwarm")
            norm = plt.Normalize(-1.5, 1.5)
            colors = cmap(norm(tt[:-1]))
        ax.add_collection(LineCollection(segments, colors=colors, linewidth=lw, alpha=alpha, zorder=28))

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 1)] + "..."


class _ClassicGraph:
    def __init__(self, scene: Mapping[str, Any], card: Mapping[str, Any]):
        self.scene = scene
        self.card = card
        self.nodes = {str(node.get("id")): node for node in scene.get("nodes", []) if node.get("id")}
        self.edges = [edge for edge in scene.get("edges", []) if isinstance(edge, Mapping)]
        self.layers = [self._copy_layer(layer) for layer in _ordered_scene_layers(scene)]
        self.classic = card.get("classic", {}) if isinstance(card.get("classic"), Mapping) else {}
        if bool(self.classic.get("synthesize_missing_bridges", True)):
            self._synthesize_missing_bridges()
        self.layer_gap = float(self.classic.get("layer_gap", 6.0))
        self.group_gap = float(self.classic.get("group_gap", 1.0))
        self.var_gap = float(self.classic.get("variable_gap", 0.22))
        self.input_file_dx = float(self.classic.get("input_file_dx", 1.1))
        self.output_file_dx = float(self.classic.get("output_file_dx", 1.1))
        self.variable_dx = float(self.classic.get("variable_dx", 2.55))
        self.input_variable_dx = float(self.classic.get("input_variable_dx", self.variable_dx))
        self.output_variable_dx = float(self.classic.get("output_variable_dx", self.variable_dx))
        self.bridge_y_offset = float(self.classic.get("bridge_y_offset", 0.45))
        self.bridge_output_pad = float(self.classic.get("bridge_output_pad", 0.18))
        self.module_icon_size = float(self.classic.get("module_icon_size", 0.9))
        self.mains: dict[str, dict] = {}
        self.files: dict[str, dict] = {}
        self.variables: dict[str, dict] = {}
        self.bridges: dict[str, dict] = {}
        self.selections: dict[str, dict] = {}
        self.xlim = (0.0, 1.0)
        self.ylim = (0.0, 1.0)
        self.logo_y = 1.0

    @staticmethod
    def _copy_layer(layer: Mapping[str, Any]) -> dict:
        copied = dict(layer)
        copied["nodes"] = [str(node_id) for node_id in copied.get("nodes", [])]
        return copied

    def _synthesize_missing_bridges(self) -> None:
        layer_positions = {str(layer.get("id")): idx for idx, layer in enumerate(self.layers)}
        if not layer_positions:
            return
        layer_names = [str(layer.get("id")) for layer in self.layers]
        layer_indices = [layer.get("index", idx + 1) for idx, layer in enumerate(self.layers)]
        next_edges: list[Mapping[str, Any]] = []
        seen_edges: set[tuple[str, str, str, str, str]] = set()
        bridgeable_roles = {"fileflow", "dataflow", "selectionflow"}

        for edge in self.edges:
            role = str(edge.get("role", "")).lower()
            source_id = str(edge.get("source", {}).get("node", ""))
            target_id = str(edge.get("target", {}).get("node", ""))
            source = self.nodes.get(source_id)
            target = self.nodes.get(target_id)
            source_layer = str(source.get("layer", "")) if isinstance(source, Mapping) else ""
            target_layer = str(target.get("layer", "")) if isinstance(target, Mapping) else ""
            source_pos = layer_positions.get(source_layer)
            target_pos = layer_positions.get(target_layer)
            source_kind = source.get("kind") if isinstance(source, Mapping) else None
            target_kind = target.get("kind") if isinstance(target, Mapping) else None

            if (
                role not in bridgeable_roles
                or source_kind == "bridge"
                or target_kind == "bridge"
                or source_pos is None
                or target_pos is None
                or target_pos - source_pos <= 1
            ):
                self._append_unique_edge(next_edges, seen_edges, edge)
                continue

            label = self._bridge_label(edge, source_id)
            prev_node = source_id
            prev_port = str(edge.get("source", {}).get("port", "out") or "out")
            for pos in range(source_pos + 1, target_pos):
                bridge_id = f"bridge::{label}::L{layer_indices[pos]}"
                self._ensure_synthetic_bridge(bridge_id, label, layer_names[pos], source_layer, target_layer)
                self._append_unique_edge(
                    next_edges,
                    seen_edges,
                    self._bridge_edge(edge, prev_node, prev_port, bridge_id, "in", "bridgeflow"),
                )
                prev_node = bridge_id
                prev_port = "out"

            self._append_unique_edge(
                next_edges,
                seen_edges,
                self._bridge_edge(
                    edge,
                    prev_node,
                    prev_port,
                    target_id,
                    str(edge.get("target", {}).get("port", "in") or "in"),
                    role,
                ),
            )

        self.edges = next_edges

    @staticmethod
    def _edge_key(edge: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            str(edge.get("source", {}).get("node", "")),
            str(edge.get("source", {}).get("port", "")),
            str(edge.get("target", {}).get("node", "")),
            str(edge.get("target", {}).get("port", "")),
            str(edge.get("role", "")),
        )

    def _append_unique_edge(self, edges: list[Mapping[str, Any]], seen: set[tuple[str, str, str, str, str]], edge: Mapping[str, Any]) -> None:
        key = self._edge_key(edge)
        if key in seen:
            return
        seen.add(key)
        edges.append(edge)

    def _bridge_label(self, edge: Mapping[str, Any], source_id: str) -> str:
        metadata = edge.get("metadata") if isinstance(edge.get("metadata"), Mapping) else {}
        if metadata.get("variable"):
            return str(metadata["variable"])
        source = self.nodes.get(source_id, {})
        return str(source.get("label") or source_id.replace("var::", ""))

    def _ensure_synthetic_bridge(self, bridge_id: str, label: str, layer_id: str, source_layer: str, target_layer: str) -> None:
        if bridge_id not in self.nodes:
            self.nodes[bridge_id] = {
                "id": bridge_id,
                "kind": "bridge",
                "role": "bridge_relay",
                "layer": layer_id,
                "label": label,
                "in_ports": [{"id": "in", "role": "bridge_in"}],
                "out_ports": [{"id": "out", "role": "bridge_out"}],
                "metadata": {
                    "synthetic": True,
                    "source_layer": source_layer,
                    "target_layer": target_layer,
                    "relay_for": label,
                },
            }
        for layer in self.layers:
            if str(layer.get("id")) == str(layer_id):
                layer.setdefault("nodes", [])
                if bridge_id not in layer["nodes"]:
                    layer["nodes"].append(bridge_id)
                break

    @staticmethod
    def _bridge_edge(
        original: Mapping[str, Any],
        source_node: str,
        source_port: str,
        target_node: str,
        target_port: str,
        role: str,
    ) -> dict:
        metadata = deepcopy(original.get("metadata")) if isinstance(original.get("metadata"), Mapping) else {}
        metadata["synthetic_bridge"] = True
        return {
            "source": {"node": source_node, "port": source_port},
            "target": {"node": target_node, "port": target_port},
            "role": role,
            "metadata": metadata,
        }

    def layout(self) -> None:
        layer_groups = self._layer_groups()
        layer_heights: dict[str, float] = {}
        group_heights: dict[str, float] = {}

        for layer in self.layers:
            total = 0.0
            groups = layer_groups.get(str(layer.get("id")), [])
            for main_id in groups:
                height = self._group_height(main_id)
                group_heights[main_id] = height
                total += height
            if groups:
                total += self.group_gap * (len(groups) - 1)
            bridge_count = len(self._standalone_bridges(str(layer.get("id"))))
            if bridge_count:
                total = max(total, bridge_count * self.var_gap + 0.6)
            layer_heights[str(layer.get("id"))] = max(total, 1.5)

        max_height = max(layer_heights.values() or [1.8]) + 0.8
        for idx, layer in enumerate(self.layers):
            layer_id = str(layer.get("id"))
            x = idx * self.layer_gap + 0.85
            cursor = max_height - 0.6 - (max_height - layer_heights[layer_id]) / 2.0
            for main_id in layer_groups.get(layer_id, []):
                height = group_heights[main_id]
                y = cursor - height / 2.0
                self._place_main_group(main_id, x, y, height)
                cursor -= height + self.group_gap
            self._place_standalone_bridges(layer_id, x, max_height, layer_groups.get(layer_id, []))

        self._place_unpositioned_variables(max_height)
        self._compute_bounds(max_height)

    def figure_size(self) -> tuple[float, float]:
        width_units = self.xlim[1] - self.xlim[0]
        height_units = self.ylim[1] - self.ylim[0]
        units_per_inch = float(self.classic.get("units_per_inch", 1.25))
        return (
            min(float(self.classic.get("max_width", 24.0)), max(float(self.classic.get("min_width", 8.0)), width_units / units_per_inch)),
            min(float(self.classic.get("max_height", 16.0)), max(float(self.classic.get("min_height", 3.5)), height_units / units_per_inch)),
        )

    def anchor(self, node_id: str, direction: str, role: str) -> tuple[float, float] | None:
        if role == "selectionflow" and direction == "in" and node_id in self.selections:
            return self.selections[node_id]["pos"]
        if node_id in self.variables:
            item = self.variables[node_id]
            x, y = item["pos"]
            width = item.get("width", 0.35)
            marker = (x - width - 0.1 + float(item.get("marker_dx", 0.0)), y)
            if direction == "in":
                return marker
            return (x + 0.12, y)
        if node_id in self.bridges:
            item = self.bridges[node_id]
            x, y = item["pos"]
            if direction == "in":
                return (x - 0.02, y)
            return (x + float(item.get("width", 0.3)) + self.bridge_output_pad, y)
        if node_id in self.files:
            item = self.files[node_id]
            return item["pos"]
        if node_id in self.mains:
            item = self.mains[node_id]
            return item["pos"]
        return None

    def is_var(self, node_id: str) -> bool:
        return node_id in self.variables or self.nodes.get(node_id, {}).get("kind") == "variable"

    def is_file(self, node_id: str) -> bool:
        return node_id in self.files or self.nodes.get(node_id, {}).get("kind") == "file"

    def is_bridge(self, node_id: str) -> bool:
        return node_id in self.bridges or self.nodes.get(node_id, {}).get("kind") == "bridge"

    def is_main(self, node_id: str) -> bool:
        return node_id in self.mains or self.nodes.get(node_id, {}).get("kind") in {"module", "source"}

    def _layer_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for layer in self.layers:
            layer_id = str(layer.get("id"))
            groups[layer_id] = []
            for node_id in layer.get("nodes", []):
                node = self.nodes.get(str(node_id))
                if node and node.get("kind") in {"module", "source"}:
                    groups[layer_id].append(str(node_id))
        return groups

    def _standalone_bridges(self, layer_id: str) -> list[str]:
        bridges = [
            node_id
            for node_id, node in self.nodes.items()
            if node.get("kind") == "bridge" and str(node.get("layer")) == layer_id
        ]
        return sorted(bridges, key=self._bridge_order_key)

    def _group_height(self, main_id: str) -> float:
        counts = []
        for file_id in self._files_for_main(main_id, "input"):
            counts.append(len(self._incoming_variables(file_id)))
        for file_id in self._files_for_main(main_id, "output"):
            counts.append(len(self._outgoing_variables(file_id)))
        counts.append(len(self._direct_input_variables(main_id)))
        counts.append(len(self._direct_output_variables(main_id)))
        if self.nodes[main_id].get("kind") == "source":
            counts.append(len(self._outgoing_variables(main_id)))
        max_count = max(counts or [1])
        return max(1.25, max_count * self.var_gap + 0.45)

    def _place_main_group(self, main_id: str, x: float, y: float, height: float) -> None:
        node = self.nodes[main_id]
        self.mains[main_id] = {"node": node, "pos": (x, y), "height": height}
        if node.get("selection"):
            selection = node.get("selection") if isinstance(node.get("selection"), Mapping) else {}
            self.selections[main_id] = {
                "node": node,
                "pos": (x, y + 0.58),
                "output_pos": (x, y),
                "expression": selection.get("expression"),
            }

        if node.get("kind") == "source":
            self._place_variables(self._outgoing_variables(main_id), x + self.output_variable_dx, y, align="right")
            return

        input_files = self._files_for_main(main_id, "input")
        output_files = self._files_for_main(main_id, "output")
        input_anchor_y = y + 0.18
        output_anchor_y = y - 0.18
        for file_id, fy in zip(input_files, self._spread_y(input_anchor_y, len(input_files), 0.65)):
            self._place_file(file_id, x - self.input_file_dx, fy, "input")
            incoming = self._incoming_variables(file_id)
            for var_id in incoming:
                if var_id not in self.variables and var_id not in self.bridges:
                    self._place_variable(var_id, x - self.input_variable_dx, fy, align="right")
        for file_id, fy in zip(output_files, self._spread_y(output_anchor_y, len(output_files), 0.65)):
            self._place_file(file_id, x + self.output_file_dx, fy, "output")
            self._place_variables(self._outgoing_variables(file_id), x + self.output_variable_dx, fy, align="right")

        direct_inputs = self._direct_input_variables(main_id)
        for var_id in direct_inputs:
            if var_id not in self.variables and var_id not in self.bridges:
                self._place_variable(var_id, x - self.input_variable_dx, y, align="right")
        self._place_variables(self._direct_output_variables(main_id), x + self.output_variable_dx, y, align="right")

    def _place_standalone_bridges(self, layer_id: str, x: float, max_height: float, main_ids: list[str]) -> None:
        bridges = self._standalone_bridges(layer_id)
        if not bridges:
            return
        main_items = [self.mains[main_id] for main_id in main_ids if main_id in self.mains]
        if main_items:
            module_top = max(item["pos"][1] + self.module_icon_size / 2.0 for item in main_items)
            lowest_bridge_y = module_top + self.bridge_y_offset
            ys = [
                lowest_bridge_y + (len(bridges) - 1 - idx) * self.var_gap
                for idx in range(len(bridges))
            ]
        else:
            ys = self._spread_y(max_height / 2.0 + self.bridge_y_offset, len(bridges), self.var_gap)
        for node_id, y in zip(bridges, ys):
            label = str(self.nodes[node_id].get("label") or "")
            self.bridges[node_id] = {
                "node": self.nodes[node_id],
                "pos": (x, y),
                "width": max(0.25, len(label) * 0.095),
            }

    def _place_unpositioned_variables(self, max_height: float) -> None:
        by_layer: dict[str, list[str]] = {}
        for node_id, node in self.nodes.items():
            if node.get("kind") == "variable" and node_id not in self.variables:
                by_layer.setdefault(str(node.get("layer")), []).append(node_id)
        layer_x = {str(layer.get("id")): idx * self.layer_gap + 0.85 for idx, layer in enumerate(self.layers)}
        for layer_id, node_ids in by_layer.items():
            x = layer_x.get(layer_id, 0.85) + self.output_variable_dx
            for node_id, y in zip(node_ids, self._spread_y(max_height / 2.0, len(node_ids), self.var_gap)):
                self._place_variable(node_id, x, y, align="right")

    def _place_file(self, file_id: str, x: float, y: float, direction: str) -> None:
        node = self.nodes[file_id]
        self.files[file_id] = {
            "node": node,
            "pos": (x, y),
            "direction": direction,
            "file_label": str(node.get("label") or ""),
        }

    def _place_variables(self, node_ids: list[str], x: float, center_y: float, *, align: str) -> None:
        for node_id, y in zip(node_ids, self._spread_y(center_y, len(node_ids), self.var_gap)):
            self._place_variable(node_id, x, y, align=align)

    def _place_variable(self, node_id: str, x: float, y: float, *, align: str) -> None:
        node = self.nodes.get(node_id)
        if not node:
            return
        label = str(node.get("label") or node_id.replace("var::", ""))
        self.variables[node_id] = {
            "node": node,
            "pos": (x, y),
            "align": align,
            "width": max(0.28, len(label) * 0.095),
            # A file's named output is a file handle, not an observable.
            # Render it as a triangular endpoint so it is distinguishable
            # from the circular observable endpoints.
            "marker_style": "<" if self._is_file_output_variable(node_id) else None,
            "marker_dx": 0.04 if self._is_file_output_variable(node_id) else 0.0,
        }

    def _is_file_output_variable(self, variable_id: str) -> bool:
        variable = self.nodes.get(variable_id, {})
        variable_name = str(variable.get("label") or variable_id.removeprefix("var::"))
        for edge in self.edges:
            source_id = str(edge.get("source", {}).get("node", ""))
            source = self.nodes.get(source_id, {})
            if (
                str(edge.get("role", "")).lower() == "fileflow"
                and str(edge.get("target", {}).get("node", "")) == variable_id
                and source.get("kind") == "file"
                and source.get("role") == "output_file"
                and str(source.get("label") or source_id.rsplit("::", 1)[-1]) == variable_name
            ):
                return True
        return False

    def _compute_bounds(self, max_height: float) -> None:
        xs: list[float] = []
        ys: list[float] = []
        for coll in (self.mains, self.files, self.variables, self.bridges, self.selections):
            for item in coll.values():
                x, y = item["pos"]
                xs.append(x)
                ys.append(y)
        if not xs:
            self.xlim = (0, 2)
            self.ylim = (0, 2)
            self.logo_y = 1.6
            return
        left_margin = float(self.classic.get("x_margin_left", 1.0))
        right_margin = float(self.classic.get("x_margin_right", 1.1))
        bottom_margin = float(self.classic.get("y_margin_bottom", 0.45))
        top_margin = float(self.classic.get("y_margin_top", 0.35))
        logo_size = float(self.classic.get("logo_size", 0.5))
        logo_y_offset = float(self.classic.get("logo_y_offset", 0.45))
        logo_top_margin = float(self.classic.get("logo_top_margin", 0.08))
        content_ymin = min(ys)
        content_ymax = max(ys)
        self.logo_y = content_ymax + logo_y_offset
        logo_top = self.logo_y + logo_size / 2.0 + logo_top_margin
        self.xlim = (min(xs) - left_margin, max(xs) + right_margin)
        self.ylim = (content_ymin - bottom_margin, max(content_ymax + top_margin, logo_top))

    @staticmethod
    def _spread_y(center: float, count: int, gap: float) -> list[float]:
        if count <= 0:
            return []
        if count == 1:
            return [center]
        start = center + (count - 1) * gap / 2.0
        return [start - idx * gap for idx in range(count)]

    def _files_for_main(self, main_id: str, direction: str) -> list[str]:
        prefix = f"file::{main_id}::{direction}::"
        ids = [node_id for node_id in self._layer_node_ids(self.nodes[main_id].get("layer")) if node_id.startswith(prefix)]
        if ids:
            return ids
        return [node_id for node_id in self.nodes if node_id.startswith(prefix)]

    def _layer_node_ids(self, layer_id: str) -> list[str]:
        for layer in self.layers:
            if str(layer.get("id")) == str(layer_id):
                return [str(node_id) for node_id in layer.get("nodes", [])]
        return []

    def _bridge_order_key(self, bridge_id: str) -> tuple[float, float, str]:
        source_id = None
        for edge in self.edges:
            if str(edge.get("target", {}).get("node")) == bridge_id:
                source_id = str(edge.get("source", {}).get("node"))
                break

        source_item = self.variables.get(source_id or "") or self.bridges.get(source_id or "")
        if source_item is not None:
            # Preserve the actual top-to-bottom order of the previous layer.
            return (0.0, -float(source_item["pos"][1]), bridge_id)

        source_node = self.nodes.get(source_id or "")
        source_layer = str(source_node.get("layer", "")) if isinstance(source_node, Mapping) else ""
        layer_ids = self._layer_node_ids(source_layer)
        try:
            source_pos = layer_ids.index(source_id) if source_id is not None else len(layer_ids)
        except ValueError:
            source_pos = len(layer_ids)
        layer_pos = len(self.layers)
        for idx, layer in enumerate(self.layers):
            if str(layer.get("id")) == source_layer:
                layer_pos = idx
                break
        return (float(layer_pos + 1), float(source_pos), bridge_id)

    def _incoming_variables(self, target_id: str) -> list[str]:
        return self._unique(
            str(edge.get("source", {}).get("node"))
            for edge in self.edges
            if str(edge.get("target", {}).get("node")) == target_id
            and self.nodes.get(str(edge.get("source", {}).get("node")), {}).get("kind") in {"variable", "bridge"}
        )

    def _outgoing_variables(self, source_id: str) -> list[str]:
        return self._unique(
            str(edge.get("target", {}).get("node"))
            for edge in self.edges
            if str(edge.get("source", {}).get("node")) == source_id
            and self.nodes.get(str(edge.get("target", {}).get("node")), {}).get("kind") == "variable"
        )

    def _direct_input_variables(self, main_id: str) -> list[str]:
        return self._unique(
            str(edge.get("source", {}).get("node"))
            for edge in self.edges
            if str(edge.get("target", {}).get("node")) == main_id
            and str(edge.get("role")) in {"dataflow", "selectionflow"}
            and self.nodes.get(str(edge.get("source", {}).get("node")), {}).get("kind") in {"variable", "bridge"}
        )

    def _direct_output_variables(self, main_id: str) -> list[str]:
        return self._unique(
            str(edge.get("target", {}).get("node"))
            for edge in self.edges
            if str(edge.get("source", {}).get("node")) == main_id
            and str(edge.get("role")) == "dataflow"
            and self.nodes.get(str(edge.get("target", {}).get("node")), {}).get("kind") == "variable"
        )

    @staticmethod
    def _unique(values) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out
