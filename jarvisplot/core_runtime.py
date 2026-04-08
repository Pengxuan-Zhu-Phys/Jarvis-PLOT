from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Set

import yaml

from .cache_store import ProjectCache
from .data_loader import JP_ROW_IDX
from .data_loader_hdf5 import scan_hdf5_leaf_metadata
from .utils.pathing import resolve_project_path


# ---------------------------------------------------------------------------
# Expression-analysis helpers (pure functions, no dependency on core object)
# ---------------------------------------------------------------------------

def _expr_symbols(expr: Any) -> Set[str]:
    """Return the set of identifier tokens referenced in an expression string."""
    if expr is None:
        return set()
    if isinstance(expr, (int, float, bool)):
        return set()
    text = str(expr).strip()
    if not text:
        return set()
    toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    ignore = {
        "np", "math", "True", "False", "None",
        "and", "or", "not", "in", "if", "else", "for", "lambda",
        "abs", "min", "max", "sum", "len", "int", "float", "str", "bool",
        "round", "sin", "cos", "tan", "exp", "log", "sqrt", "pi", "e",
    }
    return {t for t in toks if t not in ignore}


def _profile_cfg_columns(cfg: Any) -> Set[str]:
    """Return column names referenced in a profile/grid_profile coordinates block."""
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
            if key == "grid_profile":
                prof = dict(v) if isinstance(v, dict) else {}
                prof.setdefault("method", "grid")
                out.update(_profile_cfg_columns(prof))
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
        if "add_column" in step:
            add_cfg = step.get("add_column", {})
            if isinstance(add_cfg, Mapping):
                name = add_cfg.get("name")
                if isinstance(name, str) and name.strip():
                    out.add(name.strip())
    return out


def _transform_output_columns(transform: Any) -> Set[str]:
    """Return column names produced as outputs by a transform list."""
    out: Set[str] = set()
    if not isinstance(transform, list):
        return out
    for step in transform:
        if not isinstance(step, Mapping):
            continue
        if "add_column" in step:
            add_cfg = step.get("add_column", {})
            if isinstance(add_cfg, Mapping):
                name = add_cfg.get("name")
                if isinstance(name, str) and name.strip():
                    out.add(name.strip())
        if "profile" in step:
            out.update(_profile_cfg_columns(step.get("profile", {})))
        if "grid_profile" in step:
            cfg = dict(step.get("grid_profile", {})) if isinstance(step.get("grid_profile"), dict) else {}
            cfg.setdefault("method", "grid")
            out.update(_profile_cfg_columns(cfg))
    return out


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


def _layer_columns(layer: Any) -> Set[str]:
    """Return column names referenced in a layer's coordinates, style, and data blocks."""
    out: Set[str] = set()
    if not isinstance(layer, Mapping):
        return out
    _collect_expr_columns(layer.get("coordinates", {}), out)
    _collect_expr_columns(layer.get("style", {}), out)
    _collect_expr_columns(layer.get("data", []), out)
    return out


def plan_dataset_required_columns(core) -> None:
    if not isinstance(core.yaml.config, dict):
        return
    ds_names = {str(dts.name): dts for dts in core.dataset}
    demand: Dict[str, Set[str]] = {name: set() for name in ds_names.keys()}

    for dts in core.dataset:
        name = str(dts.name)
        demand.setdefault(name, set())

    figures = core.yaml.config.get("Figures", [])
    if not isinstance(figures, list):
        figures = []
    global_layer_cols: Set[str] = set()
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
            layer_cols = _layer_columns(layer)
            global_layer_cols.update(layer_cols)
            entries = layer.get("data", [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                cols = set(layer_cols)
                cols.update(_transform_columns(entry.get("transform", None)))
                src = entry.get("source")
                if isinstance(src, str):
                    if src in demand:
                        demand[src].update(cols)
                elif isinstance(src, (list, tuple)):
                    for item in src:
                        if isinstance(item, str) and item in demand:
                            demand[item].update(cols)

    if global_layer_cols:
        for name in demand.keys():
            demand[name].update(global_layer_cols)

    for name, dts in ds_names.items():
        cols = set(demand.get(name, set()))
        cols.add(JP_ROW_IDX)
        dataset_inputs = _transform_columns(getattr(dts, "transform", None))
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


def rename_hdf5_and_renew_yaml(core):
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
            dcfg.pop("is_gambit", None)
            dcfg.pop("columnmap", None)

    for dt in core.dataset:
        core.logger.warning("DataSet -> {}, type -> {}".format(dt.name, dt.type))
        vmap_dict = {}
        vmap_list = []
        if dt.type == "hdf5":
            old_columns = dt.columns if isinstance(dt.columns, dict) else {}
            for ii, kk in enumerate(dt.keys):
                vname = "Var{}@{}".format(ii, dt.name)
                vmap_dict[kk] = vname
                vmap_list.append({
                    "source": _as_quoted_str(kk),
                    "target": vname,
                })
            columns_payload = {}
            if isinstance(old_columns, dict):
                for k, v in old_columns.items():
                    if k in {"rename", "load_whitelist"}:
                        continue
                    columns_payload[k] = v
            columns_payload["rename"] = vmap_list

            if isinstance(old_columns, dict) and "load_whitelist" in old_columns:
                columns_payload["load_whitelist"] = _normalize_whitelist_as_quoted(old_columns.get("load_whitelist"))

            core.yaml.update_dataset(dt.name, {"columns": columns_payload})
            dt.rename_columns(vmap_dict)
            core.logger.debug(f"Dataset '{dt.name}' renamed columns -> {dt.keys}")

    with open(core.args.out, "w", encoding="utf-8") as f1:
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
            dcfg.pop("is_gambit", None)
            dcfg.pop("columnmap", None)

    for dcfg in core.yaml.config.get("DataSet", []):
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
        columns_payload["rename"] = vmap_list

        if "load_whitelist" in old_columns:
            columns_payload["load_whitelist"] = _normalize_whitelist_as_quoted(old_columns.get("load_whitelist"))

        core.yaml.update_dataset(name, {"columns": columns_payload})

    with open(core.args.out, "w", encoding="utf-8") as f1:
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
