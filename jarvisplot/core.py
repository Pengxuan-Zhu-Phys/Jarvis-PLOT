#!/usr/bin/env python3 

from __future__ import annotations
from typing import Optional, Dict
from .cli import CLI
from loguru import logger
import os, sys
from .config import ConfigLoader
from .data_loader import DataSet
import io
from contextlib import redirect_stdout
from .core_assets import load_cmaps, load_interpolators, load_styles
from .Figure.data_pipelines import SharedContent, DataContext
from .cache_store import ProjectCache
from .Figure.preprocessor import DataPreprocessor
from .utils.pathing import resolve_project_path
from .core_runtime import (
    plan_dataset_required_columns as runtime_plan_dataset_required_columns,
    prepare_project_layout as runtime_prepare_project_layout,
    prepare_usage_plan as runtime_prepare_usage_plan,
    parse_hdf5_metadata_and_renew_yaml as runtime_parse_hdf5_metadata_and_renew_yaml,
)


def _format_console_record(record):
    module = record["extra"].get("module", "No module")
    message = str(record["message"]).replace("{", "{{").replace("}", "}}").replace("<", "\\<")
    return f"\n\n<cyan>{module}</cyan> \n\t-> <green>{record['time']:MM-DD HH:mm:ss.SSS}</green> - [<level>{record['level']}</level>] >>> \n<level>{message}</level> "


class JarvisPLOT():
    def __init__(self) -> None:
        self.yaml       =   ConfigLoader()
        self.style      =   {}
        self.cli        =   CLI()
        self.logger     =   None
        self.dataset: list[DataSet] = []
        self.shared     =   None
        self.ctx        =   None
        self.workdir: Optional[str] = None
        self.cache: Optional[ProjectCache] = None
        self.dataset_registry: Dict[str, DataSet] = {}
        self.preprocessor: Optional[DataPreprocessor] = None

    def init(self):
        self.args = self.cli.args.parse_args()

        # Initialize logger early
        self.init_logger()

        load_cmaps(self.load_path, logger=self.logger)

        self.load_yaml()

        # sys.exit()
        if self.args.parse_data:
            if self.args.out is None and not self.args.inplace:
                self.args.out = self.yaml.path
            elif self.args.out is None and self.args.inplace:
                self.args.out = self.yaml.path
            elif self.args.out is not None and self.args.inplace:
                self.logger.error("Conflicting arguments: --out and --inplace. Please choose only one.")
                sys.exit(2)
            runtime_parse_hdf5_metadata_and_renew_yaml(self)
            return
        else:
            runtime_prepare_project_layout(self)
            self.load_dataset(eager=False)
            runtime_plan_dataset_required_columns(self)
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
            load_interpolators(
                self.yaml.config,
                yaml_dir=self.yaml.dir,
                shared=self.shared,
                logger=self.logger,
            )
            self.preprocessor = DataPreprocessor(
                self.ctx,
                cache=self.cache,
                dataset_registry=self.dataset_registry,
                logger=self.logger,
                base_dir=self.workdir or self.yaml.dir,
            )
            self.prebuild_profile_pipelines()
            runtime_prepare_usage_plan(self)

            self.style = load_styles(self.load_path, logger=self.logger)
            self.plot()

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
                setup = figobj.from_dict(fig)
                if setup:
                    self.logger.warning(f"Succefully loading figure -> {figobj.name} setting")
                    figobj.plot()
                else:
                    fig_name = figobj.name or fig.get("name", "<noname>")
                    if getattr(figobj, "_setup_status", None) == "disabled":
                        self.logger.warning(f"Skip figure {fig_name}: disabled in YAML.")
                    else:
                        self.logger.warning(
                            f"Skip figure {fig_name}: setup failed before plotting."
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
        resolved = os.path.abspath(yaml_path)
        try:
            self.parser_yaml(resolved)
        except FileNotFoundError:
            self.logger.error(f"YAML file not found: {resolved}")
            sys.exit(2)
        except OSError as e:
            self.logger.error(f"Failed to open YAML file '{resolved}': {e}")
            sys.exit(2)

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
            return _format_console_record(record)

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
