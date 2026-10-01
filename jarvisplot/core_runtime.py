from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set

import yaml

from .cache_store import ProjectCache
from .column_demand import (
    FIXED_COLUMN_METHODS,
    _layer_columns,
    _transform_columns,
    _transform_needs_all_columns,
    _transform_output_columns,
    expression_texts,
)
from .data_loader import JP_ROW_IDX
from .data_loader_hdf5 import scan_hdf5_leaf_metadata
from .Figure.figure_types import expand_figure_types_in_config
from .utils.pathing import resolve_project_path


class _QuotedString(str):
    """Marker string that should always be dumped with double quotes."""


class _QuotedDumper(yaml.SafeDumper):
    pass


def _quoted_string_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style='"')


_QuotedDumper.add_representer(_QuotedString, _quoted_string_representer)


def prepare_project_layout(core) -> None:
    cfg = core.yaml.config or {}
    project = cfg.get("project", {})
    if not isinstance(project, dict):
        project = {}

    raw_workdir = project.get("workdir", core.yaml.dir or ".")
    wp = resolve_project_path(raw_workdir, base_dir=core.yaml.dir)
    core.workdir = str(wp)
    core.workdir and Path(core.workdir).mkdir(parents=True, exist_ok=True)
    project["workdir"] = core.workdir
    cfg["project"] = project

    output = cfg.get("output", {})
    if not isinstance(output, dict):
        output = {}
    raw_outdir = output.get("dir", None)
    if not raw_outdir:
        outdir = (Path(core.workdir) / "plots").resolve()
    else:
        outdir = resolve_project_path(raw_outdir, base_dir=core.workdir)
    output["dir"] = str(outdir)
    cfg["output"] = output
    core.yaml.config = cfg

    core.cache = ProjectCache(
        core.workdir,
        logger=core.logger,
        rebuild=bool(getattr(core.args, "rebuild_cache", False)),
    )
    core.logger.debug(f"Project workdir -> {core.workdir}")
    core.logger.debug(f"Cache dir -> {core.cache.root}")


def expand_figure_types(core) -> None:
    if not isinstance(getattr(core.yaml, "config", None), dict):
        return
    expand_figure_types_in_config(core.yaml.config, logger=getattr(core, "logger", None))


#: Set to 0 / off / false to load every dataset whole, as before the column
#: plan was applied -- the escape hatch if the plan ever misses a column.
COLUMN_PRUNE_ENV = "JP_DATASET_COLUMN_PRUNE"


def column_pruning_enabled() -> bool:
    raw = str(os.getenv(COLUMN_PRUNE_ENV, "on")).strip().lower()
    return raw not in {"0", "false", "no", "off", "disable", "disabled"}


def _transform_exports(transform: Any) -> bool:
    if not isinstance(transform, list):
        return False
    return any(isinstance(step, Mapping) and ("to_csv" in step or "to_parquet" in step) for step in transform)


def _block_runtime_projection(preprocessor, layer: Mapping, entry: Mapping) -> Set[str]:
    """The columns the render pipeline itself will keep for this data block.

    Read from the preprocessor's own projection, so the plan can never be
    narrower than what a layer would have received from an unpruned table.
    """
    if preprocessor is None:
        return set()
    try:
        transform = preprocessor._effective_transform(entry.get("source"), entry.get("transform"))
        projection = preprocessor._runtime_projection(transform, preprocessor.layer_demand_columns(layer))
    except Exception:
        return set()
    return set(projection or ())


def _source_names(source: Any) -> list[str]:
    if isinstance(source, str):
        return [source]
    if isinstance(source, (list, tuple)):
        return [item for item in source if isinstance(item, str)]
    return []


def plan_dataset_required_columns(core) -> None:
    """Decide which columns each dataset reads from disk.

    The demand is an over-approximation: every column any layer, transform or
    render projection could ask for (lineage through share_data / to_df is not
    tracked, so a layer's columns count against every dataset). A dataset whose
    needs cannot be written down as names loads whole.
    """
    if not isinstance(core.yaml.config, dict):
        return
    ds_names = {str(dts.name): dts for dts in core.dataset}
    if not column_pruning_enabled():
        for dts in core.dataset:
            dts.set_required_columns(None, retained=None)
        if core.logger:
            core.logger.info(f"Dataset column pruning disabled by {COLUMN_PRUNE_ENV}.")
        return

    preprocessor = getattr(core, "preprocessor", None)
    demand: Dict[str, Set[str]] = {name: set() for name in ds_names.keys()}

    figures = core.yaml.config.get("Figures", [])
    if not isinstance(figures, list):
        figures = []
    global_layer_cols: Set[str] = set()
    #: Sources whose needs no name set describes -- a step that picks its own
    #: columns out of the table, or a method that reads columns by fixed name.
    #: These load whole.
    unprunable: Set[str] = set()
    #: Such a source that is a published / shared table, whose lineage is not
    #: tracked: no dataset can be pruned safely.
    prune_nothing: Optional[str] = None
    for fig in figures:
        if not isinstance(fig, Mapping):
            continue
        if fig.get("enable", True) is False:
            continue
        layers = fig.get("layers", [])
        if not isinstance(layers, list):
            continue
        for layer in layers:
            if not isinstance(layer, Mapping):
                continue
            layer_cols = _layer_columns(layer) | expression_texts(layer)
            global_layer_cols.update(layer_cols)
            by_name = str(layer.get("method", "")).strip().lower() in FIXED_COLUMN_METHODS
            entries = layer.get("data", [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                cols = set(layer_cols)
                cols.update(_transform_columns(entry.get("transform", None)))
                cols.update(_block_runtime_projection(preprocessor, layer, entry))
                global_layer_cols.update(cols)
                open_ended = by_name or _transform_needs_all_columns(entry.get("transform", None))
                for item in _source_names(entry.get("source")):
                    if item in demand:
                        demand[item].update(cols)
                        if open_ended:
                            unprunable.add(item)
                    elif open_ended and prune_nothing is None:
                        # A published / shared table: which dataset it came
                        # from is not tracked, so every dataset loads whole.
                        prune_nothing = (
                            "layer '{}' reads '{}', which is not a dataset, and {}".format(
                                layer.get("name", ""),
                                item,
                                "its method reads columns by name"
                                if by_name
                                else "a transform step selects its own columns",
                            )
                        )

    if global_layer_cols:
        for name in demand.keys():
            demand[name].update(global_layer_cols)

    for name, dts in ds_names.items():
        dataset_transform = getattr(dts, "transform", None)
        reason = prune_nothing
        if reason is None and name in unprunable:
            reason = "a transform step selects its own columns, or the layer method reads columns by name"
        if reason is None and _transform_needs_all_columns(dataset_transform):
            reason = "a transform step selects its own columns"
        if reason is None and _transform_exports(dataset_transform):
            reason = "a dataset-level to_csv / to_parquet exports the whole table"
        if reason is not None:
            # `None` is this API's way of saying "no restriction": load the
            # table whole and let the step choose from what is actually there.
            dts.set_required_columns(None, retained=None)
            if core.logger:
                core.logger.info(
                    "Dataset required columns planned:\n\t dataset \t-> {}\n\t required \t-> all"
                    "\n\t reason \t-> {}".format(name, reason)
                )
            continue
        cols = set(demand.get(name, set()))
        cols.add(JP_ROW_IDX)
        dataset_inputs = _transform_columns(dataset_transform) | expression_texts(dataset_transform)
        dataset_outputs = _transform_output_columns(getattr(dts, "transform", None))
        retained = set(cols)
        retained.update(dataset_outputs)
        retained.add(JP_ROW_IDX)
        required = set(retained)
        required.update(dataset_inputs)
        dts.set_required_columns(required if required else None, retained=retained if retained else None)
        if core.logger:
            sample = ", ".join(sorted(list(retained))[:12]) if retained else "<none>"
            core.logger.info(
                "Dataset required columns planned:\n\t dataset \t-> {}\n\t required \t-> {}\n\t retained \t-> {}\n\t sample \t-> {}".format(
                    name,
                    len(required),
                    len(retained),
                    sample,
                )
            )


def prepare_usage_plan(core):
    if core.ctx is None:
        return

    counts: Dict[str, int] = {}
    figures = (core.yaml.config or {}).get("Figures", [])
    if not isinstance(figures, list):
        figures = []

    for fig in figures:
        if not isinstance(fig, dict):
            continue
        if fig.get("enable", True) is False:
            continue
        layers = fig.get("layers", [])
        if not isinstance(layers, list):
            continue
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            entries = layer.get("data", [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                source = entry.get("source")
                if isinstance(source, str):
                    counts[source] = counts.get(source, 0) + 1
                elif isinstance(source, (list, tuple)):
                    for item in source:
                        if isinstance(item, str):
                            counts[item] = counts.get(item, 0) + 1

    core.ctx.set_usage_plan(counts)
    core.logger.debug(
        "Source usage plan -> {}".format(", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    )


def default_parse_data_output(yaml_path) -> str:
    """Where ``--parse-data`` writes when neither ``--out`` nor ``--inplace`` is given."""
    path = Path(str(yaml_path))
    return str(path.with_name(f"{path.stem}.parsed{path.suffix or '.yaml'}"))


_PARSE_DATA_DROPPED_KEYS = ("is_gambit", "columnmap")


def _write_parse_data_yaml(core, columns_by_index: Dict[int, dict]) -> None:
    """Write the parsed config, keeping the source file's comments and layout.

    The edits are replayed on a round-trip load of the source file rather than
    dumping the already-parsed dict, which is what used to strip every comment.
    PyYAML is only the fallback when ruamel is unavailable.
    """
    out_path = Path(str(core.args.out))
    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.comments import CommentedMap
        from ruamel.yaml.scalarstring import DoubleQuotedScalarString
        from ruamel.yaml.util import load_yaml_guess_indent
    except Exception:
        with open(out_path, "w", encoding="utf-8") as f1:
            yaml.dump(
                core.yaml.config,
                f1,
                Dumper=_QuotedDumper,
                sort_keys=False,
                default_flow_style=False,
                indent=2,
                allow_unicode=True,
                width=100000,
            )
        return

    def _quoted(value):
        if isinstance(value, _QuotedString):
            return DoubleQuotedScalarString(str(value))
        if isinstance(value, list):
            return [_quoted(item) for item in value]
        if isinstance(value, dict):
            return {key: _quoted(item) for key, item in value.items()}
        return value

    source_text = Path(core.yaml.path).read_text(encoding="utf-8")
    # Sequences are written both ways in the wild (`- item` flush with its key,
    # or indented by two); reuse whichever the source file uses.
    _, seq_indent, seq_offset = load_yaml_guess_indent(source_text)
    seq_offset = int(seq_offset or 0)
    seq_indent = max(int(seq_indent or 2), seq_offset + 2)
    rt = YAML(typ="rt")
    rt.preserve_quotes = True
    rt.width = 100000
    rt.indent(mapping=2, sequence=seq_indent, offset=seq_offset)
    doc = rt.load(source_text)

    datasets = doc.get("DataSet") if isinstance(doc, dict) else None
    for index, entry in enumerate(datasets if isinstance(datasets, list) else []):
        if not isinstance(entry, dict):
            continue
        for key in _PARSE_DATA_DROPPED_KEYS:
            entry.pop(key, None)
        payload = columns_by_index.get(index)
        if payload is None:
            continue
        columns = entry.get("columns")
        if not isinstance(columns, CommentedMap):
            columns = CommentedMap()
            entry["columns"] = columns
        for key, value in payload.items():
            columns[key] = _quoted(value)

    with open(out_path, "w", encoding="utf-8") as f1:
        rt.dump(doc, f1)


def parse_hdf5_metadata_and_renew_yaml(core):
    def _as_quoted_str(value: Any) -> _QuotedString:
        return _QuotedString(str(value))

    def _normalize_whitelist_as_quoted(raw):
        if isinstance(raw, list):
            return [_as_quoted_str(v) for v in raw if v is not None and str(v).strip()]
        if isinstance(raw, str):
            sval = raw.strip()
            if sval:
                return _as_quoted_str(sval)
        return []

    for dcfg in core.yaml.config.get("DataSet", []):
        if isinstance(dcfg, dict):
            for key in _PARSE_DATA_DROPPED_KEYS:
                dcfg.pop(key, None)

    columns_by_index: Dict[int, dict] = {}
    for index, dcfg in enumerate(core.yaml.config.get("DataSet", [])):
        if not isinstance(dcfg, dict):
            continue
        if str(dcfg.get("type", "")).strip().lower() != "hdf5":
            continue

        name = str(dcfg.get("name", "")).strip()
        project_cfg = core.yaml.config.get("project", {})
        if not isinstance(project_cfg, dict):
            project_cfg = {}
        workdir = project_cfg.get("workdir", core.yaml.dir)
        path = resolve_project_path(str(dcfg.get("path", "")).strip(), base_dir=workdir or core.yaml.dir)
        group = str(dcfg.get("dataset", "")).strip() or None
        old_columns = dcfg.get("columns", {})
        if not isinstance(old_columns, dict):
            old_columns = {}

        metadata = scan_hdf5_leaf_metadata(str(path), group=group)
        usable = [item for item in metadata if not str(item.get("path", "")).endswith("_isvalid")]
        if not usable:
            raise RuntimeError(
                "No usable leaf datasets found for HDF5 parse-data: "
                f"dataset='{name}', path='{path}', group='{group or '<root>'}'."
            )

        vmap_list = []
        for ii, item in enumerate(usable):
            source = str(item["path"])
            target = f"Var{ii}@{name}"
            vmap_list.append(
                {
                    "source": _as_quoted_str(source),
                    "target": target,
                }
            )

        columns_payload = {}
        for k, v in old_columns.items():
            if k in {"rename", "load_whitelist"}:
                continue
            columns_payload[k] = v
        changed = {"rename": vmap_list}
        if "load_whitelist" in old_columns:
            changed["load_whitelist"] = _normalize_whitelist_as_quoted(old_columns.get("load_whitelist"))
        columns_payload.update(changed)

        dcfg["columns"] = columns_payload
        columns_by_index[index] = changed

    _write_parse_data_yaml(core, columns_by_index)
    if core.logger:
        core.logger.warning(f"Parsed dataset metadata written -> {core.args.out}")


# --------------------------------------------------------------------------- #
# Correlation matrices: solve the figure before it is built
# --------------------------------------------------------------------------- #
#
# Every other card fixes its figure size and writes its axes as fractions of
# it.  A correlation matrix cannot: its content is n**2 cells and n variable
# names, so the size that keeps the cells legible depends on the data.  This
# pass runs where ``prebuild_profile_pipelines`` runs -- datasets registered,
# nothing drawn yet -- and writes the answer into the config as ordinary frame
# keys.  By the time ``Figure.from_dict`` sees it, it is a figure like any
# other, which is why none of the drawing code knows this pass exists.
#
# It reads column *names*, not rows.  ``DataSet._prepare_lazy_metadata`` has
# already paid for those, so a matrix in its default order costs no data load
# at all.


def _corr_card(core, info: Mapping) -> dict | None:
    """The style card behind this figure, when it is the correlation card."""
    tokens = info.get("style_card", info.get("style", None))
    if isinstance(tokens, str):
        tokens = [tokens]
    if not isinstance(tokens, (list, tuple)) or not tokens:
        return None
    try:
        from .Figure.style_runtime import resolve_style_bundle_payload

        _family, _variant, bundle = resolve_style_bundle_payload(core.style, list(tokens))
    except Exception:
        return None
    contract = bundle.get("Contract") if isinstance(bundle, Mapping) else None
    if not isinstance(contract, Mapping):
        return None
    if str(contract.get("figure_type", "")) != "correlation_matrix":
        return None
    return bundle


def _corr_layer(info: Mapping, axes_name: str):
    """The (layer, data-entry, correlation config) drawing the matrix."""
    from .Figure.correlation_runtime import correlation_config, is_correlation_transform

    for layer in info.get("layers", []) or []:
        if not isinstance(layer, Mapping) or layer.get("axes") != axes_name:
            continue
        for entry in layer.get("data", []) or []:
            if not isinstance(entry, Mapping):
                continue
            steps = entry.get("transform") or []
            if not isinstance(steps, list):
                continue
            for step in steps:
                if is_correlation_transform(step):
                    return layer, entry, step, correlation_config(step)
    return None, None, None, None


def _numeric_predicate(dts):
    """A name -> bool test for 'this column can be correlated', from metadata.

    The renderer answers this from real dtypes.  Here it comes from a cheap
    schema read so the two agree; when neither is available every non-private
    name is accepted and a genuinely non-numeric column fails loudly at render
    rather than quietly shifting every label by one.
    """
    import pandas as pd

    path, kind = getattr(dts, "path", None), str(getattr(dts, "type", "") or "")
    try:
        if kind == "csv" and path:
            head = pd.read_csv(path, nrows=256)
            ok = {
                str(name)
                for name in head.columns
                if pd.api.types.is_numeric_dtype(head[name])
                and not pd.api.types.is_bool_dtype(head[name])
            }
            return lambda name: name in ok
        if kind == "parquet" and path:
            import polars as pl

            schema = pl.scan_parquet(path).collect_schema()
            ok = {str(n) for n, t in schema.items() if t.is_numeric() and t != pl.Boolean}
            return lambda name: name in ok
    except Exception:
        pass
    return None


def _corr_tick_sizes(card: Mapping) -> tuple[float, float, str, float]:
    """Tick text size, colorbar text size, family, and the panel's label pad.

    The pad is returned in **millimetres** because it is part of how much room a
    name needs: the text starts that far from the panel, so under ``margin.fit``
    a margin of exactly the label width would still clip its last glyph.
    """
    frame = card.get("Frame", {}) or {}
    style = (card.get("Style", {}) or {}).get("corrplot", {}) or {}
    ticks = (frame.get("axcorr", {}) or {}).get("ticks", {}) or {}
    panel = ticks.get("both", {}) or {}
    major = ticks.get("major", {}) or {}
    bar = ((frame.get("axccorr", {}) or {}).get("ticks", {}) or {}).get("both", {}) or {}
    label_pt = float(panel.get("labelsize", 6.0)) * float(style.get("tl.cex", 1.0) or 1.0)
    # An outward tick pushes the label out by its own length first; this card
    # draws none (length 0), but a card that did would need the room.
    pad_pt = float(major.get("pad", 3.5) or 0.0)
    if str(panel.get("direction", "out")).strip().lower() == "out":
        pad_pt += float(major.get("length", 0.0) or 0.0)
    return (
        label_pt,
        float(bar.get("labelsize", 6.0)),
        str(panel.get("labelfontfamily", "sans")),
        pad_pt / 72.0 * 25.4,
    )


def _corr_bar_tick_pad_mm(card: Mapping) -> float:
    """How far the colorbar sets its numbers off the bar, in millimetres.

    Needed by the diamond solve and only by it: on that card the numbers can
    end up between the bar and the matrix's diagonal, where their pad is part
    of how much room the bar has left.
    """
    ticks = ((card.get("Frame", {}) or {}).get("axccorr", {}) or {}).get("ticks", {}) or {}
    major = ticks.get("major", {}) or {}
    both = ticks.get("both", {}) or {}
    pad_pt = float(major.get("pad", 3.5) or 0.0)
    if str(both.get("direction", "out")).strip().lower() == "out":
        pad_pt += float(major.get("length", 0.0) or 0.0)
    return pad_pt / 72.0 * 25.4


def _colorbar_label_samples(card: Mapping) -> list[str]:
    """The strings the colorbar will print, for the overrun check.

    An authored tick list is the truth, so it is used as-is.  Without one the
    ticks do not exist until the bar is drawn -- long after the size has to be
    known -- and the fallback guesses from the card's limits instead; the guess
    only has to be as wide as the truth.
    """
    frame = (card.get("Frame", {}) or {}).get("axccorr", {}) or {}
    authored = ((frame.get("ticks", {}) or {}).get("y", {}) or {}).get("labels")
    if authored:
        return [str(label) for label in authored]
    color = frame.get("color", {}) or {}
    try:
        lo, hi = float(color.get("vmin", -1.0)), float(color.get("vmax", 1.0))
    except (TypeError, ValueError):
        lo, hi = -1.0, 1.0
    span = hi - lo
    return ["{:.2f}".format(lo + span * k / 8.0) for k in range(9)]


def _solve_corr_square(card, info, style, columns, axes_name):
    """The square card: an n x n panel with a label band on two sides."""
    from .Figure.corr_layout import solve_corr_geometry

    n = len(columns)
    show_x, show_y = _corr_tick_visibility(style.get("tl.pos", "lt"))
    label_pt, bar_pt, family, label_pad_mm = _corr_tick_sizes(card)
    cb_title, cb_title_pt = _colorbar_title(card, info)
    geom = solve_corr_geometry(
        n,
        geometry=card.get("Geometry", {}) or {},
        # A label nobody prints costs the layout nothing.  Budgeting for it
        # anyway is how a `tl.pos: n` matrix ends up with a blank margin
        # the width of its longest variable name.
        x_labels=columns if show_x else (),
        y_labels=columns if show_y else (),
        colorbar_labels=_colorbar_label_samples(card),
        colorbar_title=cb_title,
        label_size_pt=label_pt,
        label_pad_mm=label_pad_mm,
        colorbar_label_size_pt=bar_pt,
        colorbar_title_size_pt=cb_title_pt,
        family=family,
    )
    solved = geom.as_frame()
    solved[axes_name] = {
        "xlim": [-0.5, n - 0.5],
        # Row 0 on top: a matrix is read from its top-left corner, and the
        # labels have to agree with the cells they name.
        "ylim": [n - 0.5, -0.5],
        "ticks": {
            "both": {"labelbottom": show_x, "labelleft": show_y,
                     "bottom": show_x, "left": show_y},
            "x": {"positions": list(range(n)), "labels": list(columns)},
            "y": {"positions": list(range(n)), "labels": list(columns)},
        },
    }
    return geom, solved


def _solve_corr_diamond(card, info, style, columns, axes_name):
    """The rotated card: one triangle at 45 degrees, names in a column.

    The names stay y tick labels -- item k sits at ``v = k``, which is the
    whole convenience of the map -- so what changes here against the square
    branch is the limits, the side the ticks are on, and that there is no x
    axis at all.
    """
    from .Figure.corr_layout_diamond import solve_diamond_geometry

    n = len(columns)
    # `t` and `lt` name a band this layout does not have: there is one column
    # of names, on the side `side` picks.  Saying so beats printing the names
    # somewhere they cannot line up with anything.
    tl_pos = str(style.get("tl.pos", "l") or "l").strip().lower()
    if tl_pos not in ("l", "n"):
        raise ValueError(
            "figure '{}' asks for corrplot tl.pos: {}. The diamond card has "
            "one band of names, printed horizontally beside the panel, so it "
            "takes l (print them) or n (do not). Which side they are on is "
            "corrplot side: left | right.".format(info.get("name", "?"), tl_pos)
        )
    show_names = tl_pos == "l"
    side = str(style.get("side", "left") or "left").strip().lower()
    diag = _as_corr_bool(style.get("diag", False))

    # `edge.numbers` repeats each variable's position at the far end of both
    # arms of its V, and prints it once more beside the name.  That second copy
    # is *not* glued onto the name here: it is set in the smaller, lighter face
    # the numbers wear on the diagonal, and a tick label is one string in one
    # colour.  The renderer draws it; the solve owns how much room it costs.
    edge_numbers = _as_corr_bool(style.get("edge.numbers", False))

    label_pt, bar_pt, family, label_pad_mm = _corr_tick_sizes(card)
    number_pt = max(label_pt * float(style.get("edge.numbers.cex", 0.7) or 0.7), 3.2)
    cb_title, cb_title_pt = _colorbar_title(card, info)
    geom = solve_diamond_geometry(
        n,
        geometry=card.get("Geometry", {}) or {},
        labels=columns if show_names else (),
        colorbar_labels=_colorbar_label_samples(card),
        colorbar_title=cb_title,
        side=side,
        diag=diag,
        edge_numbers=edge_numbers,
        label_size_pt=label_pt,
        label_pad_mm=label_pad_mm,
        number_size_pt=number_pt,
        colorbar_label_size_pt=bar_pt,
        colorbar_tick_pad_mm=_corr_bar_tick_pad_mm(card),
        colorbar_title_size_pt=cb_title_pt,
        family=family,
    )
    on_left = show_names and geom.side == "left"
    on_right = show_names and geom.side == "right"
    solved = geom.as_frame()
    solved[axes_name] = {
        "xlim": list(geom.xlim),
        "ylim": list(geom.ylim),
        "ticks": {
            "both": {
                "left": on_left, "labelleft": on_left,
                "right": on_right, "labelright": on_right,
                "bottom": False, "labelbottom": False,
                "top": False, "labeltop": False,
            },
            # A name ends at the panel on the left and starts at it on the
            # right.  Anchoring both the same way is how a right-hand column
            # ends up printed back over its own cells.
            "y": {
                "positions": list(range(n)),
                "labels": list(columns),
                "labelha": "left" if on_right else "right",
            },
        },
    }
    if edge_numbers and show_names:
        # Numbered, the band is a table of two columns and the names turn
        # around: they are set flush with the *outer* edge of their column, so
        # the numbers beside them line up instead of stepping in and out with
        # the length of each name.  The cost is a ragged edge against the
        # panel, which is the right side to spend it on -- a column of numbers
        # only reads as one if every number starts in the same place.
        #
        # The turn is a pad, not just an alignment: matplotlib anchors a tick
        # label a pad in from the spine, so `ha: left` alone would set every
        # name running back over its own cells.  The pad is the full width the
        # solve gave the names.
        ticks = solved[axes_name]["ticks"]
        ticks["y"]["labelha"] = "right" if on_right else "left"
        ticks["major"] = {"pad": geom.name_pad_mm / 25.4 * 72.0}
    # The bar stands inside the empty triangle, not beside the panel, so its
    # two pieces of text go on opposite sides of it: numbers left, label right,
    # the same way round on both mirrors.  The card says so (`ticks_position`,
    # and `va` on the label); the solve only has to place the label, because
    # where it goes is `colorbar.label_gap` expressed against a bar the solve
    # sized, and `ticks_position` would otherwise drag it back to the tick
    # side.
    #
    # The offset anchors the edge of the text *away* from the bar, and which
    # edge that is comes from `va`, not `ha`: the label is turned on its side,
    # and under `rotation_mode: anchor` -- which `set_label_position` turns on
    # -- the alignment is applied before the rotation, so `va: top` puts the
    # anchor on the text's left edge.  Matplotlib would take `va` from the tick
    # side, which is the far side from this label; left alone it anchors the
    # near edge and the label grows into the bar and prints on the scale.
    gap = geom.colorbar_label_gap_mm / geom.colorbar_w_mm
    solved["axccorr"] = {"ylabel_coords": {"x": 1.0 + gap, "y": 0.5}}
    # The badge went to whichever bottom corner the names are not in, so the
    # wordmark beside it has to run back toward the middle of the page.
    logo_left = geom.logo_rect[0] < 0.5
    solved["axlogo"] = {"anchor": "left" if logo_left else "right"}
    return geom, solved


def _corr_shaded_positions(style: Mapping, columns, name: str):
    """`stripe.variables` resolved against the order the figure ended up in.

    It has to happen here and not at draw time.  The renderer sees positions,
    and which position a variable holds is `order`'s answer, settled a moment
    ago -- a name resolved against the *unordered* columns would tint a
    different variable on every reordering and never say so.

    A name that is not in the matrix is an error rather than a silent miss, for
    the same reason: a tint that quietly did nothing is indistinguishable from
    a tint that was not asked for.
    """
    stripe = style.get("stripe", "alternate")
    stripe = "none" if stripe is False else str(stripe or "").strip().lower()
    wanted = style.get("stripe.variables")
    if wanted is None:
        return None
    if stripe != "selected":
        raise ValueError(
            "figure '{}' gives corrplot stripe.variables with stripe: {}. The "
            "names are only read by stripe: selected; leaving them here would "
            "draw the other tint and say nothing about them.".format(name, stripe)
        )
    if isinstance(wanted, str) or not isinstance(wanted, (list, tuple, set)):
        raise ValueError(
            "figure '{}' gives corrplot stripe.variables: {!r}. It takes a "
            "list of variable names.".format(name, wanted)
        )
    at = {str(column): position for position, column in enumerate(columns)}
    missing = [str(v) for v in wanted if str(v) not in at]
    if missing:
        raise ValueError(
            "figure '{}' asks corrplot to tint {}, which {} not in the matrix. "
            "The variables drawn are: {}.".format(
                name, ", ".join(missing), "is" if len(missing) == 1 else "are",
                ", ".join(str(c) for c in columns),
            )
        )
    return sorted(at[str(v)] for v in wanted)


def _as_corr_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "t"}
    return bool(value)


def _enforce_corr_contract(info: Mapping, contract: Mapping, name: str) -> None:
    """The reserved card takes exactly one layer, and it draws the matrix.

    ``Contract.exclusive`` is checked here rather than left to the renderer
    because the whole card is solved *from* the matrix: a second layer would
    be drawn into a panel whose size, limits and tick labels were derived
    without it, and an overlay on axes it never asked for is not a picture
    anyone means to make.
    """
    if not contract.get("exclusive", False):
        return
    axes_name = str(contract.get("axes", "axcorr"))
    allowed = {axes_name}
    layers = [ly for ly in (info.get("layers") or []) if isinstance(ly, Mapping)]

    strays = sorted({str(ly.get("axes", "")) for ly in layers} - allowed)
    if strays:
        raise ValueError(
            "figure '{}' uses the reserved [corrplot, matrix] card, which draws "
            "one correlation matrix on '{}' and nothing else. Layers ask for: "
            "{}. Put the overlay on its own figure.".format(
                name, axes_name, ", ".join(strays)
            )
        )
    if len(layers) > 1:
        raise ValueError(
            "figure '{}' puts {} layers on '{}'. The card's figure size, limits "
            "and tick labels are solved from one matrix, so a second layer would "
            "be drawn into a frame that was never measured for it.".format(
                name, len(layers), axes_name
            )
        )
    for layer in layers:
        method = str(layer.get("method", "")).strip().lower()
        if method != "corrplot":
            raise ValueError(
                "figure '{}' draws the correlation table with method '{}'. The "
                "matrix has its own primitive -- `method: corrplot` -- which is "
                "what reads type / diag / order / addgrid.col and sizes its "
                "glyphs from the solved cell. A scatter layer accepts those keys "
                "and discards them.".format(name, method or "scatter")
            )


def _corr_colormap(style: dict, info: dict, name: str) -> None:
    """Apply R's ``col`` to the colorbar, before anything reads the frame.

    R passes the palette to ``corrplot()``; here the colour scale belongs to
    the colorbar axes, and the colorbar is built from the frame long before
    a layer draws.  So ``col`` is translated rather than forwarded -- the
    alternative is a legend that disagrees with the cells it explains.
    """
    col = style.pop("col", None)
    if col is None:
        return
    import matplotlib

    key = str(col).strip()
    if key not in matplotlib.colormaps:
        raise ValueError(
            "figure '{}': corrplot col: {!r} is not a registered colormap. R's "
            "diverging presets carry over by name (RdBu, BrBG, PiYG, PRGn, "
            "PuOr, RdYlBu), as does any matplotlib or Jarvis colormap. Append "
            "_r to reverse it.".format(name, col)
        )
    frame = info.setdefault("frame", {})
    if not isinstance(frame, dict):
        return
    bar = frame.setdefault("axccorr", {})
    if not isinstance(bar, dict):
        return
    color = bar.setdefault("color", {})
    if not isinstance(color, dict):
        return
    written = color.get("cmap")
    if written is not None and str(written) != key:
        raise ValueError(
            "figure '{}' sets the matrix colour twice: corrplot col: {!r} and "
            "frame.axccorr.color.cmap: {!r}. They are the same setting in two "
            "spellings -- keep one.".format(name, col, written)
        )
    color["cmap"] = key


def _corr_style(card, layer) -> dict:
    """The corrplot formals in force: card defaults under the layer's own.

    Read here as well as at render time because the two halves of corrplot are
    split by design -- ``order`` / ``addrect`` decide *where* a variable sits
    and have to be settled before the tick labels are written, while
    everything else decides what a cell looks like and can wait for the draw.
    Both halves read the same merged block, so a card default behaves exactly
    like the same key written in the YAML.
    """
    style = dict((card.get("Style", {}) or {}).get("corrplot", {}) or {})
    layer_style = layer.get("style") if isinstance(layer, Mapping) else None
    if isinstance(layer_style, Mapping):
        style.update(layer_style)
    return style


def _corr_source_table(core, layer, entry, step, columns):
    """The table the correlation is computed on, for ordering only.

    Runs the data block's transforms *up to* the correlation step, so an
    ordering is never computed from rows a filter was going to remove.  Only
    reached when the ordering is data-dependent: ``order: original`` and
    ``order: alphabet`` are answered from the column names alone and touch no
    data at all.
    """
    from .Figure.preprocessor_runtime import run_pipeline

    steps = list(entry.get("transform") or [])
    prefix = []
    for candidate in steps:
        if candidate is step:
            break
        prefix.append(candidate)

    df, _key, _hit = run_pipeline(
        core.preprocessor,
        entry.get("source"),
        prefix or None,
        combine=str(layer.get("combine", "concat")),
        mode="preprofile",
        projection=list(columns),
    )
    return core.preprocessor.ensure_pandas(df, reason="prebuild:correlation-order")


def _corr_tick_visibility(tl_pos: str) -> tuple[bool, bool]:
    """``(show x labels, show y labels)`` for R's ``tl.pos``.

    Handled here rather than in the renderer because it is not only a tick
    setting: a matrix with no y labels does not need the margin they were
    budgeted, and the figure this card solves is only correct if the two
    agree.
    """
    key = str(tl_pos or "lt").strip().lower()
    # R names the sides it prints on, and its variable names go on top.  Here
    # they go underneath, so `t` and `b` select the same band -- `t` is kept
    # because it is the R formal and pasted R calls should not fail, `b` is
    # accepted because it is what the figure actually shows.
    if key in ("lt", "tl", "lb", "bl"):
        return True, True
    if key in ("t", "b"):
        return True, False
    if key == "l":
        return False, True
    if key == "n":
        return False, False
    raise ValueError(
        "corrplot tl.pos must be lt, t (or b), l or n; got {!r}. R's 'd' draws "
        "the names down the diagonal, which this card does not do -- the "
        "diagonal carries cells here.".format(tl_pos)
    )


def _corr_debug_lines(geom, n_columns: int) -> list[str]:
    """The solve, as the design overlay prints it.

    The overlay's caption already reports the size that came out. On this card
    that is the least interesting number: the size is a *result*, and what a
    reader checking the layout needs is what produced it -- how many variables,
    how big a cell that left, and how far the panel ended up from the corner.

    Written here rather than in the overlay because this is the only place that
    still has the solved geometry; by draw time the figure is an ordinary one
    whose rects happen to be fractions.
    """
    import textwrap

    # Short lines on purpose: the block is centred on a figure that can be
    # 49 mm wide, and one long line runs off both edges of the smallest case.
    if hasattr(geom, "pitch_mm"):
        # The diamond reports its own anchor.  The pitch is the number to read
        # here -- it is the row spacing *and* the cell's diagonal -- and the
        # label column is the band that actually moved.
        lines = [
            "solved diamond · {} vars · pitch {:.3f} mm".format(n_columns, geom.pitch_mm),
            "labels {:.3f} mm · panel {:.2f} × {:.2f} mm".format(
                geom.label_block_mm, geom.panel_w_mm, geom.panel_h_mm
            ),
            "figure {:.2f} × {:.2f} mm".format(geom.width_mm, geom.height_mm),
        ]
    else:
        corner_mm = geom.panel_rect[0] * geom.width_mm
        lines = [
            "solved geometry · {} vars · cell {:.3f} mm".format(n_columns, geom.cell_mm),
            "corner {:.3f} mm · panel {:.2f} mm sq".format(corner_mm, geom.panel_mm),
            "figure {:.2f} × {:.2f} mm".format(geom.width_mm, geom.height_mm),
        ]
    for note in geom.notes:
        lines.extend(textwrap.wrap(note, width=44))
    return lines


def _attach_corr_debug(info: dict, lines: list[str]) -> None:
    """Hand the overlay its lines without turning the overlay on.

    ``Figures[].debug`` is both the switch and the per-figure override, and the
    mapping form defaults to *on*. Carrying ``show`` through explicitly is what
    keeps this a delivery channel rather than a second way to enable debug: a
    figure that never asked for the overlay still ends up with ``show: False``.
    """
    from copy import deepcopy

    existing = info.get("debug", False)
    if isinstance(existing, Mapping):
        node = deepcopy(dict(existing))
        show = bool(node.get("show", True))
    else:
        node, show = {}, bool(existing)
    node["show"] = show
    solved = node.get("solved")
    node["solved"] = {**(solved if isinstance(solved, Mapping) else {}), "lines": lines}
    info["debug"] = node


def _colorbar_title(card: Mapping, info: Mapping) -> tuple[str, float]:
    """The colorbar's own axis label and its size, YAML over card.

    Needed *before* the solve because the label is printed outside the bar:
    unbudgeted, a one-character label like ``$\\rho$`` lands past the right
    edge of the page and is silently cropped by every PDF viewer.
    """
    card_label = ((card.get("Frame", {}) or {}).get("axccorr", {}) or {}).get("label", {}) or {}
    yaml_label = ((info.get("frame") or {}).get("axccorr", {}) or {}).get("label", {}) or {}
    merged = {**card_label, **(yaml_label if isinstance(yaml_label, Mapping) else {})}
    try:
        size = float(merged.get("fontsize", 7.0))
    except (TypeError, ValueError):
        size = 7.0
    return str(merged.get("ylabel", "") or ""), size


def prebuild_correlations(core) -> None:
    """Write each correlation figure's solved geometry and axes into the config.

    Everything a correlation matrix needs to know before it can be drawn is
    settled here: which columns are in, what order they sit in, where the
    ``addrect`` boxes fall, how big a cell is and therefore how big the figure
    is, and what the tick labels say.  All of it lands in ``info["frame"]`` and
    in the transform's ``columns``, so by the time a Figure is built the matrix
    is an ordinary, fully-specified figure.

    This is the half of corrplot that cannot happen at render time.  The tick
    labels are resolved as the figure loads; a renderer that reordered the
    matrix afterwards would leave every label naming a different column, and
    nothing downstream can detect that.
    """
    config = getattr(getattr(core, "yaml", None), "config", None)
    if not isinstance(config, dict) or not getattr(core, "style", None):
        return
    figures = config.get("Figures", [])
    if not isinstance(figures, list):
        return

    for info in figures:
        if not isinstance(info, dict) or info.get("enable", True) is False:
            continue
        card = _corr_card(core, info)
        if card is None:
            continue
        try:
            _prebuild_one(core, info, card)
        except ValueError as exc:
            # Left unsolved on purpose.  The figure then fails its own setup
            # ("carries no Frame.figure"), which counts as a render failure and
            # sets the exit code -- where drawing it anyway would mean handing
            # back a matrix ordered differently from the one that was asked
            # for, with nothing on the page saying so.
            if core.logger:
                core.logger.error(
                    "correlation prebuild failed for figure '{}': {}".format(
                        info.get("name", "?"), exc
                    )
                )


def _prebuild_one(core, info: dict, card: Mapping) -> None:
    """Solve one correlation figure. See :func:`prebuild_correlations`."""
    from deepmerge import always_merger

    from .Figure.corr_order import order_columns
    from .Figure.correlation_runtime import pearson_matrix, resolve_correlation_columns

    contract = card.get("Contract", {}) or {}
    axes_name = str(contract.get("axes", "axcorr"))
    layer, entry, step, cfg = _corr_layer(info, axes_name)
    if layer is None:
        return

    name = info.get("name", "?")
    _enforce_corr_contract(info, contract, name)

    source = entry.get("source")
    # A correlation layer may intentionally concatenate several root
    # datasets.  The runtime already supports source lists; the prebuild only
    # needs one member's cheap column metadata to solve the shared geometry.
    # Use the first registered member (the input tables are required to carry
    # the same schema), while leaving the complete source list for the actual
    # correlation transform below.
    registry = getattr(core, "dataset_registry", {}) or {}
    source_names = [source] if isinstance(source, str) else list(source or [])
    dts = next(
        (registry.get(item) for item in source_names if isinstance(item, str) and registry.get(item) is not None),
        None,
    )
    names = list(getattr(dts, "keys", None) or [])
    if not names:
        if core.logger:
            core.logger.warning(
                "correlation prebuild skipped for figure '{}': dataset '{}' "
                "has no column metadata yet.".format(name, source)
            )
        return

    try:
        columns = resolve_correlation_columns(
            names, cfg, correlatable=_numeric_predicate(dts)
        )
    except ValueError as exc:
        if core.logger:
            core.logger.warning(
                "correlation prebuild skipped for figure '{}': {}".format(name, exc)
            )
        return

    style = _corr_style(card, layer)
    _corr_colormap(style, info, name)
    order = str(style.get("order", "original") or "original")
    addrect = style.get("addrect")
    blocks = None
    if order.strip().lower() != "original" or addrect:
        matrix = None
        if order.strip().lower() != "alphabet" or addrect:
            try:
                df = _corr_source_table(core, layer, entry, step, columns)
                matrix = pearson_matrix(
                    df,
                    columns,
                    missing=str(cfg.get("missing", "listwise")),
                    min_periods=int(cfg.get("min_periods", 2)),
                )
            except Exception as exc:
                raise ValueError(
                    "figure '{}' asks for corrplot order: {}, which is a "
                    "property of the correlation matrix and has to be "
                    "resolved before the tick labels are written. Reading "
                    "'{}' to compute it failed: {}".format(name, order, source, exc)
                ) from exc
        columns, blocks = order_columns(
            matrix,
            columns,
            order,
            hclust_method=str(style.get("hclust.method", "complete")),
            addrect=addrect,
        )

    # Which layout the card is, stated once by the card and nowhere else.  The
    # two are not variants of one solve: the square is anchored on the cell and
    # bounded by the page width, the diamond on the row pitch and bounded by
    # its height.  See Figure/corr_layout_diamond.py.
    layout = str(contract.get("layout", "square")).strip().lower()
    solve = _solve_corr_diamond if layout == "diamond" else _solve_corr_square
    geom, solved = solve(card, info, style, columns, axes_name)

    for note in geom.notes:
        if core.logger:
            core.logger.warning("correlation geometry: {}".format(note))

    n = len(columns)
    info["frame"] = always_merger.merge(info.get("frame") or {}, solved)
    _attach_corr_debug(info, _corr_debug_lines(geom, n))

    # Pin the selection *and its order* so the render cannot resolve a
    # different set from the one the labels were measured on.  This is also
    # how the ordering reaches the figure: `x_index` counts positions in
    # `columns`, so writing the order here is the whole of applying it.
    # `regex` has to go with it -- the selector refuses to be given both.
    target = step.get("correlation") if isinstance(step.get("correlation"), dict) else step
    target["columns"] = list(columns)
    target.pop("regex", None)

    # The renderer maps (x_index, y_index) through the same u/v the solver used,
    # so it has to know which layout it is drawing.  Written into the layer for
    # the same reason `__corr_blocks__` is: the card said it once, at config
    # time, and nothing downstream should have to resolve the card again.
    if layout == "diamond":
        layer.setdefault("style", {})["__corr_layout__"] = layout
        # How far out the names go: the one thing about the solve the renderer
        # cannot measure for itself, and the reach of all three things that
        # follow the names -- the tint, the rules between them, and the outline
        # that closes the figure.  It stops at the *names*, so the numbering
        # beside them falls outside all three: a position is a tag on the row,
        # not part of it, and a box drawn round the tag says otherwise.
        layer["style"]["__corr_label_mm__"] = geom.name_pad_mm
        # Same channel, same reason: the number column is drawn by the renderer
        # but placed and sized by the solve.
        layer["style"]["__corr_number_mm__"] = geom.number_pad_mm
        layer["style"]["__corr_number_pt__"] = geom.number_size_pt
        # `stripe.variables` names variables; the renderer draws positions, and
        # only the order settled above knows which is which.
        shaded = _corr_shaded_positions(style, columns, info.get("name", "?"))
        if shaded is not None:
            layer["style"]["__corr_shaded__"] = shaded

    if blocks:
        # The boxes are cuts of the tree the order came from, so they are
        # computed once, here, and drawn from index ranges.  Recomputing
        # them at render would be a second chance to disagree.
        layer.setdefault("style", {})["__corr_blocks__"] = blocks

    if core.logger:
        # Each layout reports the number it is anchored on, because that is the
        # one a reader checking the figure would go looking for.
        anchor, anchor_mm = (
            ("pitch", geom.pitch_mm) if layout == "diamond" else ("cell", geom.cell_mm)
        )
        core.logger.warning(
            "Correlation geometry solved -> figure '{}'\n\t variables \t-> {}"
            "\n\t layout \t-> {}\n\t order \t\t-> {}{}"
            "\n\t {} \t\t-> {:.2f} mm\n\t figure \t-> {:.2f} × {:.2f} mm".format(
                name, n, layout, order,
                " ({} blocks)".format(len(blocks)) if blocks else "",
                anchor, anchor_mm, geom.width_mm, geom.height_mm,
            )
        )
