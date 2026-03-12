#!/usr/bin/env python3 

from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict, Set, Mapping
from .cli import CLI
from loguru import logger
import os, sys
import yaml
import re
from .config import ConfigLoader
from .data_loader import DataSet, JP_ROW_IDX
import io
from contextlib import redirect_stdout
jppwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import json 
from .Figure.data_pipelines import SharedContent, DataContext
from .cache_store import ProjectCache
from .Figure.preprocessor import DataPreprocessor


class _QuotedString(str):
    """Marker string that should always be dumped with double quotes."""


class _QuotedDumper(yaml.SafeDumper):
    pass


def _quoted_string_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style='"')


_QuotedDumper.add_representer(_QuotedString, _quoted_string_representer)


class JarvisPLOT():
    def __init__(self) -> None:
        self.variables  =   {}
        self.yaml       =   ConfigLoader()
        self.style      =   {}
        self.profiles   =   {}
        self.cli        =   CLI()
        self.logger     =   None
        self.dataset: list[DataSet] = []
        self.shared     =   None
        self.ctx        =   None
        self.interpolators = None
        self.workdir: Optional[str] = None
        self.cache: Optional[ProjectCache] = None
        self.dataset_registry: Dict[str, DataSet] = {}
        self.preprocessor: Optional[DataPreprocessor] = None

    def init(self):
        self.args = self.cli.args.parse_args()

        # Initialize logger early
        self.init_logger()

        self.load_cmaps()

        self.load_yaml()
        self.prepare_project_layout()

        # sys.exit()
        if self.args.parse_data:
            if self.args.out is None and not self.args.inplace:
                self.args.out = self.yaml.path
            elif self.args.out is None and self.args.inplace:
                self.args.out = self.yaml.path
            elif self.args.out is not None and self.args.inplace:
                self.logger.error("Conflicting arguments: --out and --inplace. Please choose only one.")
                sys.exit(2)
            self.load_dataset(eager=True)
            self.rename_hdf5_and_renew_yaml()
        else:
            self.load_dataset(eager=False)
            self.plan_dataset_required_columns()
            if self.shared is None:
                self.shared = SharedContent(logger=self.logger)
            self.ctx = DataContext(self.shared)
            for dts in self.dataset:
                self.dataset_registry[dts.name] = dts
                self.ctx.register(
                    dts.name,
                    lambda _shared, _d=dts: _d.get_data(),
                    release_fn=dts.release,
                )

            # Register external functions (e.g. lazy-loaded interpolators) into the expression runtime.
            self.load_interpolators()
            self.preprocessor = DataPreprocessor(
                self.ctx,
                cache=self.cache,
                dataset_registry=self.dataset_registry,
                logger=self.logger,
            )
            self.prebuild_profile_pipelines()
            self.prepare_usage_plan()

            self.load_styles()
            self.plot()

    @staticmethod
    def _expr_symbols(expr: Any) -> Set[str]:
        if expr is None:
            return set()
        if isinstance(expr, (int, float, bool)):
            return set()
        text = str(expr).strip()
        if not text:
            return set()
        toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
        ignore = {
            "np",
            "math",
            "True",
            "False",
            "None",
            "and",
            "or",
            "not",
            "in",
            "if",
            "else",
            "for",
            "lambda",
            "abs",
            "min",
            "max",
            "sum",
            "len",
            "int",
            "float",
            "str",
            "bool",
            "round",
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sqrt",
            "pi",
            "e",
        }
        return {t for t in toks if t not in ignore}

    def _profile_cfg_columns(self, cfg: Any) -> Set[str]:
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
                out.update(self._expr_symbols(expr))
                name = axis_cfg.get("name")
                if isinstance(name, str) and name.strip():
                    out.add(name.strip())
                elif axis_name in {"x", "y", "z", "left", "right", "bottom"}:
                    out.add(axis_name)
            elif isinstance(axis_cfg, str):
                out.update(self._expr_symbols(axis_cfg))
                if axis_name in {"x", "y", "z", "left", "right", "bottom"}:
                    out.add(axis_name)
        return out

    def _collect_expr_columns(self, obj: Any, out: Set[str]) -> None:
        if isinstance(obj, Mapping):
            for k, v in obj.items():
                key = str(k).strip().lower()
                if key in {"expr", "filter", "sortby"}:
                    out.update(self._expr_symbols(v))
                    continue
                if key == "profile":
                    out.update(self._profile_cfg_columns(v))
                    continue
                if key == "grid_profile":
                    prof = v if isinstance(v, Mapping) else {}
                    if isinstance(prof, dict):
                        prof = dict(prof)
                    else:
                        prof = {}
                    prof.setdefault("method", "grid")
                    out.update(self._profile_cfg_columns(prof))
                    continue
                self._collect_expr_columns(v, out)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                self._collect_expr_columns(item, out)

    def _transform_columns(self, transform: Any) -> Set[str]:
        out: Set[str] = set()
        if not isinstance(transform, list):
            return out
        for step in transform:
            if not isinstance(step, Mapping):
                continue
            self._collect_expr_columns(step, out)
            if "add_column" in step:
                add_cfg = step.get("add_column", {})
                if isinstance(add_cfg, Mapping):
                    name = add_cfg.get("name")
                    if isinstance(name, str) and name.strip():
                        out.add(name.strip())
        return out

    def _transform_output_columns(self, transform: Any) -> Set[str]:
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
                out.update(self._profile_cfg_columns(step.get("profile", {})))
            if "grid_profile" in step:
                cfg = step.get("grid_profile", {})
                if isinstance(cfg, dict):
                    cfg = dict(cfg)
                else:
                    cfg = {}
                cfg.setdefault("method", "grid")
                out.update(self._profile_cfg_columns(cfg))
        return out

    def _layer_columns(self, layer: Any) -> Set[str]:
        out: Set[str] = set()
        if not isinstance(layer, Mapping):
            return out
        self._collect_expr_columns(layer.get("coordinates", {}), out)
        self._collect_expr_columns(layer.get("style", {}), out)
        self._collect_expr_columns(layer.get("data", []), out)
        return out

    def plan_dataset_required_columns(self) -> None:
        if not isinstance(self.yaml.config, dict):
            return
        ds_names = {str(dts.name): dts for dts in self.dataset}
        demand: Dict[str, Set[str]] = {name: set() for name in ds_names.keys()}

        for dts in self.dataset:
            name = str(dts.name)
            demand.setdefault(name, set())

        figures = self.yaml.config.get("Figures", [])
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
                layer_cols = self._layer_columns(layer)
                global_layer_cols.update(layer_cols)
                entries = layer.get("data", [])
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, Mapping):
                        continue
                    cols = set(layer_cols)
                    cols.update(self._transform_columns(entry.get("transform", None)))
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
            dataset_inputs = self._transform_columns(getattr(dts, "transform", None))
            dataset_outputs = self._transform_output_columns(getattr(dts, "transform", None))
            retained = set(cols)
            retained.update(dataset_outputs)
            retained.add(JP_ROW_IDX)
            required = set(retained)
            required.update(dataset_inputs)
            dts.set_required_columns(required if required else None, retained=retained if retained else None)
            if self.logger:
                sample = ", ".join(sorted(list(retained))[:12]) if retained else "<none>"
                self.logger.warning(
                    "Dataset required columns planned:\n\t dataset \t-> {}\n\t required \t-> {}\n\t retained \t-> {}\n\t sample \t-> {}".format(
                        name,
                        len(required),
                        len(retained),
                        sample,
                    )
                )

    def load_cmaps(self):
        """Load and register JarvisPLOT colormaps from the internal JSON bundle."""
        try:
            # Prefer the project's colormap setup helper
            from .utils import cmaps

            json_path = "&JP/jarvisplot/cards/colors/colormaps.json"
            cmap_summary = cmaps.setup(self.load_path(json_path), force=True)

            if self.logger:
                self.logger.debug(f"JarvisPLOT: colormaps registered: {cmap_summary}")
                try:
                    self.logger.debug(
                        f"JarvisPLOT: available colormaps sample: {cmaps.list_available()}"
                    )
                except Exception:
                    pass
        except Exception as e:
            if self.logger:
                self.logger.warning(f"JarvisPLOT: failed to initialize colormaps: {e}")        

    def load_interpolators(self):
        """Parse YAML interpolator specs and register them for lazy use in expressions."""
        cfg = self.yaml.config.get("Functions", None)
        if cfg is not None: 
            from .inner_func import set_external_funcs_getter
            from .utils.interpolator import InterpolatorManager
            mgr = InterpolatorManager.from_yaml(
                cfg,
                yaml_dir=self.yaml.dir,
                shared=self.shared,
                logger=self.logger,
            )
            self.interpolators = mgr
            set_external_funcs_getter(lambda: (mgr.as_eval_funcs() or {}))
            if self.interpolators:
                self.logger.debug(f"JarvisPLOT: Functions registered: {mgr.summary()}")

    def load_styles(self):
        spp = "&JP/jarvisplot/cards/style_preference.json"
        self.logger.debug("Loading internal Format set -> {}".format(self.load_path(spp)))
        with open(self.load_path(spp), 'r') as f1:
            stl = json.load(f1)
            for sty, boudle in stl.items():
                self.style[sty] = {}
                for kk, vv in boudle.items():
                    vpath = self.load_path(vv)
                    if os.path.exists(vpath):
                        self.logger.debug("Loading '{}' boudle, {} Style \n\t-> {}".format(sty, kk, vpath))
                        with open(vpath, 'r') as f2:
                            self.style[sty][kk] = json.load(f2)
                    else:
                        self.logger.error("Style Not Found: '{}' boudle, {} Style \n\t-> {}".format(sty, kk, vpath))

    def prepare_project_layout(self):
        """Resolve workdir/output defaults and initialize local cache."""
        cfg = self.yaml.config or {}
        project = cfg.get("project", {})
        if not isinstance(project, dict):
            project = {}

        raw_workdir = project.get("workdir", self.yaml.dir or ".")
        wp = Path(str(raw_workdir)).expanduser()
        if not wp.is_absolute():
            wp = (Path(self.yaml.dir) / wp).resolve()
        self.workdir = str(wp)
        os.makedirs(self.workdir, exist_ok=True)
        project["workdir"] = self.workdir
        cfg["project"] = project

        output = cfg.get("output", {})
        if not isinstance(output, dict):
            output = {}
        raw_outdir = output.get("dir", None)
        if not raw_outdir:
            outdir = (Path(self.workdir) / "plots").resolve()
        else:
            op = Path(str(raw_outdir)).expanduser()
            if not op.is_absolute():
                op = (Path(self.workdir) / op).resolve()
            outdir = op
        output["dir"] = str(outdir)
        cfg["output"] = output
        self.yaml.config = cfg

        self.cache = ProjectCache(
            self.workdir,
            logger=self.logger,
            rebuild=bool(getattr(self.args, "rebuild_cache", False)),
        )
        self.logger.debug(f"Project workdir -> {self.workdir}")
        self.logger.debug(f"Cache dir -> {self.cache.root}")

    def prebuild_profile_pipelines(self):
        """Traverse figures once and prebuild profile pipelines."""
        if self.preprocessor is None:
            return
        try:
            stats = self.preprocessor.prebuild_profiles(self.yaml.config or {})
            self.logger.warning(
                "Prebuild profile pipelines finished -> tasks={tasks}, hits={hits}, miss={miss}".format(
                    tasks=stats.get("tasks", 0),
                    hits=stats.get("hits", 0),
                    miss=stats.get("miss", 0),
                )
            )
        except Exception as e:
            self.logger.warning(f"Prebuild profile pipelines failed: {e}")

    def prepare_usage_plan(self):
        """Count how many times each shared source is consumed during plotting."""
        if self.ctx is None:
            return

        counts: Dict[str, int] = {}
        figures = (self.yaml.config or {}).get("Figures", [])
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

        self.ctx.set_usage_plan(counts)
        self.logger.debug(
            "Source usage plan -> {}".format(
                ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            )
        )


    def load_path(self, path):
        if "&JP/" == path[0:4]:
            path = os.path.abspath( os.path.join(jppwd, path[4:]) )
        else:
            path = Path(path).expanduser().resolve()
        return path

    def plot(self):
        for fig in self.yaml.config["Figures"]:
            from .Figure.figure import Figure
            figobj = Figure()
            figobj._yaml_dir = self.yaml.dir
            figobj.config = self.yaml.config
            figobj.logger = self.logger
            figobj.jpstyles = self.style
            figobj.context = self.ctx
            figobj.preprocessor = self.preprocessor
            if getattr(self.args, "no_logo", False):
                figobj.print = True

            try:
                if figobj.set(fig):
                    self.logger.warning(f"Succefully loading figure -> {figobj.name} setting")
                    figobj.plot()
                else:
                    self.logger.warning(
                        f"Skip figure {fig.get('name', '<noname>')}: setup failed before plotting."
                    )
            except Exception as e:
                self.logger.warning(f"Figure {fig.get('name', '<noname>')} failed: {e}")
                continue





            # print(fig)



    def load_yaml(self):
        # If no YAML file provided, show a friendly message and help, then return gracefully
        yaml_path = getattr(self.args, 'file', None)
        if not yaml_path:
            self.logger.error("No input YAML file specified. Please provide one.\n")
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    self.cli.args.print_help()
                help_text = buf.getvalue()
                self.logger.warning("JarvisPLOT " + help_text)
                sys.exit(2)
            except Exception:
                pass
            return
        self.parser_yaml(os.path.abspath(yaml_path))

    def init_logger(self) -> None:
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d[%H:%M:%S]")

        # Remove Loguru's default handler to avoid duplicate console lines
        try:
            logger.remove()
        except Exception:
            pass

        def global_log_filter(record):
            return record["extra"].get("JPlot", False)

        def stream_filter(record):
            return record["extra"].get("to_console", False)

        def custom_format(record):
            module = record["extra"].get("module", "No module")
            return f"\n\n<cyan>{module}</cyan> \n\t-> <green>{record['time']:MM-DD HH:mm:ss.SSS}</green> - [<level>{record['level']}</level>] >>> \n<level>{record['message']}</level> "

        logger.add(
            sys.stdout,
            filter=stream_filter,
            format=custom_format,
            colorize=True,
            enqueue=True,
            level="DEBUG" if self.args.debug else "WARNING"
        )
        self.logger = logger.bind(module="JarvisPLOT", to_console=True, JPlot=True)
        self.logger.warning("JarvisPLOT logging system initialized successful!")
        if self.args.debug:
            self.logger.debug("JarvisPLOT run in debug mode!")

    def parser_yaml(self, file):
        self.yaml.file = os.path.abspath(file)
        self.yaml.load()
        self.logger.debug("Resolved YAML file -> {}".format(self.yaml.path))

    def load_dataset(self, eager: bool = False):
        dts = self.yaml.config['DataSet']
        data_root = self.workdir or self.yaml.dir
        for dt in dts:
            dataset = DataSet()
            dataset.logger = self.logger
            dataset.full_load = bool(getattr(self.args, "parse_data", False))
            dataset.setinfo(dt, data_root, eager=eager, cache=self.cache)
            self.dataset.append(dataset)

    def rename_hdf5_and_renew_yaml(self):
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

        for dcfg in self.yaml.config.get("DataSet", []):
            if isinstance(dcfg, dict):
                dcfg.pop("is_gambit", None)
                dcfg.pop("columnmap", None)

        for dt in self.dataset:
            self.logger.warning("DataSet -> {}, type -> {}".format(dt.name, dt.type))
            vmap_dict = {}
            vmap_list = []
            if dt.type == "hdf5":
                old_columns = dt.columns if isinstance(dt.columns, dict) else {}
                for ii, kk in enumerate(dt.keys):
                    vname = "Var{}@{}".format(ii, dt.name)
                    vmap_dict[kk] = vname
                    vmap_list.append({
                        "source":  _as_quoted_str(kk),
                        "target":  vname
                    })
                columns_payload = {}
                if isinstance(old_columns, dict):
                    for k, v in old_columns.items():
                        if k in {"rename", "load_whitelist"}:
                            continue
                        columns_payload[k] = v
                columns_payload["rename"] = vmap_list

                if isinstance(old_columns, dict) and "load_whitelist" in old_columns:
                    columns_payload["load_whitelist"] = _normalize_whitelist_as_quoted(
                        old_columns.get("load_whitelist")
                    )

                self.yaml.update_dataset(dt.name, {"columns": columns_payload})
                dt.rename_columns(vmap_dict)
                print(dt.keys)

        with open(self.args.out, 'w', encoding='utf-8') as f1:
            yaml.dump(
                self.yaml.config,
                f1,
                Dumper=_QuotedDumper,
                sort_keys=False,
                default_flow_style=False,
                indent=2,
                allow_unicode=True,
                width=100000,
            )
