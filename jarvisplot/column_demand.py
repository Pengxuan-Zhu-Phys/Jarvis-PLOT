#!/usr/bin/env python3

"""Which columns a config asks each data source for, derived from the YAML alone.

Pure analysis: regex over expression strings and a walk over the config mapping.
No pandas, no matplotlib, no file I/O -- which is what lets both the render
planner (``core_runtime.plan_dataset_required_columns``) and ``jplot validate``
share one owner instead of growing two answers to the same question.

Two consumers with different needs:

- **Column pruning** wants an over-approximation: demanding a column that turns
  out to be unnecessary only costs memory.
- **Existence checking** wants precision: a false "column not found" is worse
  than a missed one, so :func:`plan_source_demand` attributes columns only to
  the source a layer actually names, and subtracts anything a transform produces.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Set

__all__ = [
    "SourceDemand",
    "layer_columns",
    "plan_source_demand",
    "transform_columns",
    "transform_output_columns",
]


def _expr_symbols(expr: Any) -> Set[str]:
    """Return the set of identifier tokens referenced in an expression string.

    Attribute names, quoted literals and the shared ignore table are stripped
    by :func:`jarvisplot.expr_names.expr_identifiers`, so neither ``log10`` in
    ``np.log10(x)`` nor ``groupby`` in ``w.groupby(...)`` is ever reported as a
    missing column.
    """
    if expr is None:
        return set()
    if isinstance(expr, (int, float, bool)):
        return set()
    text = str(expr).strip()
    if not text:
        return set()
    # expr_names is stdlib-only so this path never pulls numpy/sympy (B7).
    from .expr_names import expr_identifiers

    return expr_identifiers(text)


def _profile_cfg_columns(cfg: Any) -> Set[str]:
    """Return column names referenced in a profile coordinates block."""
    out: Set[str] = set()
    if not isinstance(cfg, Mapping):
        return out
    coors = cfg.get("coordinates", {})
    if not isinstance(coors, Mapping):
        return out
    for axis_key, axis_cfg in coors.items():
        axis_name = str(axis_key).strip()
        if isinstance(axis_cfg, Mapping):
            expr = axis_cfg.get("expr")
            out.update(_expr_symbols(expr))
            name = axis_cfg.get("name")
            if isinstance(name, str) and name.strip():
                out.add(name.strip())
            elif axis_name in {"x", "y", "z", "left", "right", "bottom"}:
                out.add(axis_name)
        elif isinstance(axis_cfg, str):
            out.update(_expr_symbols(axis_cfg))
            if axis_name in {"x", "y", "z", "left", "right", "bottom"}:
                out.add(axis_name)
    return out


def _density_cell_transform_config(step: Any) -> Mapping[str, Any]:
    if not isinstance(step, Mapping):
        return {}
    for name in ("make_density_core",):
        if name in step:
            cfg = step.get(name)
            return cfg if isinstance(cfg, Mapping) else {}
    if str(step.get("type", "")).strip().lower() == "make_density_core":
        return step
    return {}


def _posterior_density_transform_config(step: Any) -> Mapping[str, Any]:
    if not isinstance(step, Mapping):
        return {}
    if "posterior_density" in step:
        cfg = step.get("posterior_density")
        return cfg if isinstance(cfg, Mapping) else {}
    if str(step.get("type", "")).strip().lower() == "posterior_density":
        return step
    return {}


def _density_cell_cfg_input_columns(cfg: Any) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(cfg, Mapping):
        return out
    coors = cfg.get("coordinates", {})
    if not isinstance(coors, Mapping):
        coors = {}

    def _add_coord(key: str, default: str, *, required: bool = True) -> None:
        spec = coors.get(key, cfg.get(key, None))
        if spec is None:
            if required:
                out.add(default)
            return
        if isinstance(spec, Mapping):
            expr = spec.get("expr")
            if expr is not None:
                out.update(_expr_symbols(expr))
                return
            name = spec.get("name", default)
            if isinstance(name, str) and name.strip():
                out.add(name.strip())
            return
        if isinstance(spec, str) and spec.strip():
            out.add(spec.strip())

    _add_coord("x", "x")
    _add_coord("y", "y")
    _add_coord("weight", "weight", required=False)
    return out


def _density_cell_cfg_output_columns(cfg: Any) -> Set[str]:
    if not isinstance(cfg, Mapping):
        cfg = {}
    coors = cfg.get("coordinates", {})
    if not isinstance(coors, Mapping):
        coors = {}
    output = cfg.get("output", {})
    if not isinstance(output, Mapping):
        output = {}

    def _name(key: str, default: str) -> str:
        if output.get(key) is not None:
            value = output.get(key, default)
        else:
            spec = coors.get(key, cfg.get(key, None))
            if isinstance(spec, Mapping):
                value = spec.get("name", default)
            else:
                value = default
        text = str(value).strip()
        return text or default

    return {_name("x", "x"), _name("y", "y"), _name("weight", "weight")}


def _posterior_density_cfg_output_columns(cfg: Any) -> Set[str]:
    if not isinstance(cfg, Mapping):
        cfg = {}
    output = cfg.get("output", "density")
    if isinstance(output, Mapping):
        x_name = str(output.get("x", "x")).strip() or "x"
        y_name = str(output.get("y", "y")).strip() or "y"
        z_name = str(output.get("z", output.get("density", "density"))).strip() or "density"
        return {x_name, y_name, z_name}
    text = str(output).strip()
    return {"x", "y", text or "density"}


def _interp_2d_transform_config(step: Any) -> Mapping[str, Any]:
    if not isinstance(step, Mapping):
        return {}
    if "make_interp_2d" in step:
        cfg = step.get("make_interp_2d")
        return cfg if isinstance(cfg, Mapping) else {}
    if str(step.get("type", "")).strip().lower() == "make_interp_2d":
        return step
    return {}


def _interp_2d_cfg_input_columns(cfg: Any) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(cfg, Mapping):
        return out
    coors = cfg.get("coordinates", {})
    if not isinstance(coors, Mapping):
        coors = {}

    def _add_coord(key: str, default: str) -> None:
        spec = coors.get(key, cfg.get(key, None))
        if spec is None:
            out.add(default)
            return
        if isinstance(spec, Mapping):
            expr = spec.get("expr")
            if expr is not None:
                out.update(_expr_symbols(expr))
                return
            name = spec.get("name", default)
            if isinstance(name, str) and name.strip():
                out.add(name.strip())
            return
        if isinstance(spec, str) and spec.strip():
            out.add(spec.strip())

    _add_coord("x", "x")
    _add_coord("y", "y")
    _add_coord("z", "z")
    return out


def _interp_2d_cfg_output_columns(cfg: Any) -> Set[str]:
    if not isinstance(cfg, Mapping):
        cfg = {}
    coors = cfg.get("coordinates", {})
    if not isinstance(coors, Mapping):
        coors = {}
    output = cfg.get("output", {})
    if not isinstance(output, Mapping):
        output = {}

    def _name(key: str, default: str) -> str:
        if key == "z" and cfg.get("output_z") is not None:
            value = cfg.get("output_z", default)
        elif output.get(key) is not None:
            value = output.get(key, default)
        else:
            spec = coors.get(key, cfg.get(key, None))
            if isinstance(spec, Mapping):
                value = spec.get("name", default)
            else:
                value = default
        text = str(value).strip()
        return text or default

    return {_name("x", "x"), _name("y", "y"), _name("z", "z")}
def _collect_expr_columns(obj: Any, out: Set[str]) -> None:
    """Recursively collect column names from expressions inside a config dict/list."""
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            key = str(k).strip().lower()
            if key in {"expr", "filter", "sortby"}:
                out.update(_expr_symbols(v))
                continue
            if key == "profile":
                out.update(_profile_cfg_columns(v))
                continue
            if key == "make_density_core":
                out.update(_density_cell_cfg_input_columns(v))
                continue
            if key == "posterior_density":
                out.update(_density_cell_cfg_input_columns(v))
                continue
            if key == "make_interp_2d":
                out.update(_interp_2d_cfg_input_columns(v))
                continue
            if key == "bin_stat":
                out.update(_bin_stat_cfg_columns(v, "input"))
                continue
            _collect_expr_columns(v, out)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_expr_columns(item, out)


def _transform_columns(transform: Any) -> Set[str]:
    """Return all column names referenced as inputs in a transform list."""
    out: Set[str] = set()
    if not isinstance(transform, list):
        return out
    for step in transform:
        if not isinstance(step, Mapping):
            continue
        _collect_expr_columns(step, out)
        dcfg = _density_cell_transform_config(step)
        if dcfg:
            out.update(_density_cell_cfg_input_columns(dcfg))
        pcfg = _posterior_density_transform_config(step)
        if pcfg:
            out.update(_density_cell_cfg_input_columns(pcfg))
        icfg = _interp_2d_transform_config(step)
        if icfg:
            out.update(_interp_2d_cfg_input_columns(icfg))
        if "add_column" in step:
            add_cfg = step.get("add_column", {})
            if isinstance(add_cfg, Mapping):
                name = add_cfg.get("name")
                if isinstance(name, str) and name.strip():
                    out.add(name.strip())
    return out


def _bin_stat_cfg_columns(cfg: Any, which: str) -> Set[str]:
    from .Figure.bin_stat_runtime import bin_stat_input_columns, bin_stat_output_columns

    if not isinstance(cfg, Mapping):
        return set()
    try:
        return bin_stat_input_columns(cfg) if which == "input" else bin_stat_output_columns(cfg)
    except Exception:
        return set()


def _transform_output_columns(transform: Any) -> Set[str]:
    """Return column names produced as outputs by a transform list."""
    out: Set[str] = set()
    if not isinstance(transform, list):
        return out
    for step in transform:
        if not isinstance(step, Mapping):
            continue
        if "bin_stat" in step:
            out.update(_bin_stat_cfg_columns(step.get("bin_stat"), "output"))
        if "add_column" in step:
            add_cfg = step.get("add_column", {})
            if isinstance(add_cfg, Mapping):
                name = add_cfg.get("name")
                if isinstance(name, str) and name.strip():
                    out.add(name.strip())
        if "profile" in step:
            cfg = step.get("profile", {})
            out.update(_profile_cfg_columns(cfg))
            if isinstance(cfg, Mapping) and str(cfg.get("method", "bridson")).lower() == "grid":
                out.update({
                    "__grid_ix__",
                    "__grid_iy__",
                    "__grid_bin__",
                    "__grid_xmin__",
                    "__grid_xmax__",
                    "__grid_ymin__",
                    "__grid_ymax__",
                    "__grid_xscale__",
                    "__grid_yscale__",
                    "__grid_objective__",
                    "__grid_empty_value__",
                })
        dcfg = _density_cell_transform_config(step)
        if dcfg:
            out.update(_density_cell_cfg_output_columns(dcfg))
        pcfg = _posterior_density_transform_config(step)
        if pcfg:
            out.update(_posterior_density_cfg_output_columns(pcfg))
        icfg = _interp_2d_transform_config(step)
        if icfg:
            out.update(_interp_2d_cfg_output_columns(icfg))
    return out


def _layer_columns(layer: Any) -> Set[str]:
    """Return column names referenced in a layer's coordinates, style, and data blocks."""
    out: Set[str] = set()
    if not isinstance(layer, Mapping):
        return out
    _collect_expr_columns(layer.get("coordinates", {}), out)
    _collect_expr_columns(layer.get("style", {}), out)
    _collect_expr_columns(layer.get("data", []), out)
    return out


# ---------------------------------------------------------------------------
# Public names (the leading underscore is historical; these are the module API)
# ---------------------------------------------------------------------------

layer_columns = _layer_columns
transform_columns = _transform_columns
transform_output_columns = _transform_output_columns


class SourceDemand:
    """Columns one named source is asked for, and which of them a step produces."""

    __slots__ = ("required", "produced", "origins")

    def __init__(self) -> None:
        self.required: Set[str] = set()
        self.produced: Set[str] = set()
        #: column -> YAML paths that asked for it. A diagnostic that points at the
        #: expression is worth far more than one pointing at the DataSet entry.
        self.origins: Dict[str, Set[str]] = {}

    def note(self, columns: Set[str], path: str) -> None:
        self.required.update(columns)
        for column in columns:
            self.origins.setdefault(column, set()).add(path)

    @property
    def missing_candidates(self) -> Set[str]:
        """Columns that must already exist in the source file."""
        return {c for c in self.required - self.produced if not c.startswith("__")}

    def where(self, column: str) -> list[str]:
        return sorted(self.origins.get(column, ()))

    def __repr__(self) -> str:  # pragma: no cover
        return f"SourceDemand(required={sorted(self.required)}, produced={sorted(self.produced)})"


def plan_source_demand(config: Mapping[str, Any]) -> Dict[str, SourceDemand]:
    """Map ``source name -> SourceDemand`` without the cross-source union.

    ``core_runtime.plan_dataset_required_columns`` deliberately unions every
    layer's columns into every dataset, because a layer may consume a share_data
    table whose lineage is not tracked. That over-approximation is right for
    pruning and useless for existence checking, so this walk keeps each layer's
    demand attached to the source that layer names.

    Figures using the ``type:`` shorthand contribute demand from macro slots
    (``x`` / ``y`` / ``z`` / ``weight`` / …) attributed to ``data`` sources.
    """
    demand: Dict[str, SourceDemand] = {}

    def slot(name: str) -> SourceDemand:
        return demand.setdefault(name, SourceDemand())

    for index, entry in enumerate(config.get("DataSet") or ()):
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        bucket = slot(name.strip())
        bucket.note(
            _transform_columns(entry.get("transform")),
            f"$.DataSet[{index}].transform",
        )
        bucket.produced.update(_transform_output_columns(entry.get("transform")))

    for fig_index, figure in enumerate(config.get("Figures") or ()):
        if not isinstance(figure, Mapping):
            continue
        if figure.get("enable", True) is False:
            continue
        # type: macros: demand columns from macro slots (x/y/z/weight/…) before expansion.
        if "type" in figure:
            _note_type_figure_demand(figure, fig_index, slot)
            continue
        for layer_index, layer in enumerate(figure.get("layers") or ()):
            if not isinstance(layer, Mapping):
                continue
            layer_path = f"$.Figures[{fig_index}].layers[{layer_index}]"
            wanted = _layer_columns(layer)
            by_path = _layer_columns_by_path(layer, layer_path)

            shared = layer.get("share_data")
            if isinstance(shared, str) and shared.strip():
                # Whatever this layer publishes is produced downstream of its own
                # sources, so it is never a column read from a file.
                slot(shared.strip()).produced.update(wanted)

            for block_index, block in enumerate(layer.get("data") or ()):
                if not isinstance(block, Mapping):
                    continue
                block_path = f"{layer_path}.data[{block_index}]"
                step_inputs = _transform_columns(block.get("transform"))
                step_outputs = _transform_output_columns(block.get("transform"))
                sources = block.get("source")
                sources = sources if isinstance(sources, (list, tuple)) else [sources]
                for source in sources:
                    if not isinstance(source, str) or not source.strip():
                        continue
                    bucket = slot(source.strip())
                    published = _published_name(block)
                    if published is None:
                        for column, path in by_path.items():
                            bucket.note({column}, path)
                        bucket.note(wanted - set(by_path), layer_path)
                    else:
                        # The layer reads this block's table under another name,
                        # so the coordinates are that table's problem, not this
                        # source file's.
                        slot(published).produced.update(step_outputs)
                    bucket.note(step_inputs, f"{block_path}.transform")
                    bucket.produced.update(step_outputs)

    return demand


def _published_name(block: Mapping[str, Any]) -> Any:
    """The ``to_df`` name this block publishes under, if it publishes at all."""
    steps = block.get("transform")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        if "to_ds" in step and step.get("to_ds") is not False:
            source = block.get("source")
            return source.strip() if isinstance(source, str) and source.strip() else None
        if "to_df" not in step:
            continue
        spec = step.get("to_df")
        name = spec.get("name") if isinstance(spec, Mapping) else spec
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def _note_type_figure_demand(
    figure: Mapping[str, Any],
    fig_index: int,
    slot,
) -> None:
    """Attribute type-macro expr symbols to the figure's ``data`` source(s)."""
    base = f"$.Figures[{fig_index}]"
    raw = figure.get("data")
    if isinstance(raw, str) and raw.strip():
        sources = [raw.strip()]
    elif isinstance(raw, (list, tuple)):
        sources = [str(s).strip() for s in raw if str(s).strip()]
    else:
        sources = []
    if not sources:
        return

    for key in ("x", "y", "z", "weight", "c", "color", "w"):
        if key not in figure:
            continue
        field = figure.get(key)
        path = f"{base}.{key}"
        symbols: Set[str] = set()
        if isinstance(field, Mapping):
            symbols |= _expr_symbols(field.get("expr"))
            name = field.get("name")
            if isinstance(name, str) and name.strip() and "expr" not in field:
                symbols.add(name.strip())
            path = f"{path}.expr" if "expr" in field else path
        elif isinstance(field, str):
            symbols |= _expr_symbols(field)
        else:
            continue
        if not symbols:
            continue
        for source in sources:
            slot(source).note(symbols, path)


def _layer_columns_by_path(layer: Mapping[str, Any], layer_path: str) -> Dict[str, str]:
    """Column -> the coordinate expression that named it.

    Only coordinates get this precision, because that is where the mistakes are:
    an agent writing ``x: {expr: aa}`` needs the path of that line, not the path
    of the dataset it happens to resolve against.
    """
    out: Dict[str, str] = {}
    coordinates = layer.get("coordinates")
    if not isinstance(coordinates, Mapping):
        return out
    for axis, spec in coordinates.items():
        if isinstance(spec, Mapping):
            expr = spec.get("expr")
        elif isinstance(spec, str):
            expr = spec
        else:
            continue
        for symbol in _expr_symbols(expr):
            out.setdefault(symbol, f"{layer_path}.coordinates.{axis}.expr")
    return out
