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
from .core_assets import load_cmaps as _load_cmaps
from .core_assets import load_interpolators as _load_interpolators
from .core_assets import load_styles as _load_styles
from .Figure.data_pipelines import SharedContent, DataContext
from .cache_store import ProjectCache
from .Figure.preprocessor import DataPreprocessor
from .utils.pathing import resolve_project_path
from .core_runtime import (
    plan_dataset_required_columns as runtime_plan_dataset_required_columns,
    prepare_project_layout as runtime_prepare_project_layout,
    prepare_usage_plan as runtime_prepare_usage_plan,
    rename_hdf5_and_renew_yaml as runtime_rename_hdf5_and_renew_yaml,
)


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
        return runtime_plan_dataset_required_columns(self)

    def load_cmaps(self):
        _load_cmaps(self)

    def load_interpolators(self):
        _load_interpolators(self)

    def load_styles(self):
        _load_styles(self)

    def prepare_project_layout(self):
        """Resolve workdir/output defaults and initialize local cache."""
        return runtime_prepare_project_layout(self)

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
        return runtime_prepare_usage_plan(self)


    def load_path(self, path):
        return resolve_project_path(path)

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
        return runtime_rename_hdf5_and_renew_yaml(self)
