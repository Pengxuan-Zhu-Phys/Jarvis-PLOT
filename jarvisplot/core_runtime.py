from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Set

import yaml

from .cache_store import ProjectCache
from .data_loader import JP_ROW_IDX
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


def _layer_columns(core, layer: Any) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(layer, Mapping):
        return out
    core._collect_expr_columns(layer.get("coordinates", {}), out)
    core._collect_expr_columns(layer.get("style", {}), out)
    core._collect_expr_columns(layer.get("data", []), out)
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
            layer_cols = _layer_columns(core, layer)
            global_layer_cols.update(layer_cols)
            entries = layer.get("data", [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                cols = set(layer_cols)
                cols.update(core._transform_columns(entry.get("transform", None)))
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
        dataset_inputs = core._transform_columns(getattr(dts, "transform", None))
        dataset_outputs = core._transform_output_columns(getattr(dts, "transform", None))
        retained = set(cols)
        retained.update(dataset_outputs)
        retained.add(JP_ROW_IDX)
        required = set(retained)
        required.update(dataset_inputs)
        dts.set_required_columns(required if required else None, retained=retained if retained else None)
        if core.logger:
            sample = ", ".join(sorted(list(retained))[:12]) if retained else "<none>"
            core.logger.warning(
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
