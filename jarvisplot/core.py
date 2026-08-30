#!/usr/bin/env python3 

from __future__ import annotations
from typing import Optional, Dict
from .cli import CLI
from loguru import logger
import os, sys
from .config import ConfigLoader
from .data_loader import DataSet
from .core_assets import load_cmaps, load_interpolators, load_styles
from .Figure.data_pipelines import SharedContent, DataContext
from .cache_store import ProjectCache
from .Figure.preprocessor import DataPreprocessor
from .utils.pathing import resolve_project_path
from .core_runtime import (
    expand_figure_types as runtime_expand_figure_types,
    plan_dataset_required_columns as runtime_plan_dataset_required_columns,
    prebuild_correlations as runtime_prebuild_correlations,
    prepare_project_layout as runtime_prepare_project_layout,
    prepare_usage_plan as runtime_prepare_usage_plan,
    parse_hdf5_metadata_and_renew_yaml as runtime_parse_hdf5_metadata_and_renew_yaml,
)


def _format_console_record(record):
    module = record["extra"].get("module", "No module")
    message = str(record["message"]).replace("{", "{{").replace("}", "}}").replace("<", "\\<")
    return f"\n\n<cyan>{module}</cyan> \n\t-> <green>{record['time']:MM-DD HH:mm:ss.SSS}</green> - [<level>{record['level']}</level>] >>> \n<level>{message}</level> "


class JarvisPLOT():
    def __init__(self, *, prog: str = "jplot", argv=None) -> None:
        self.yaml       =   ConfigLoader()
        self.style      =   {}
        self.cli        =   CLI(prog=prog)
        self.argv       =   argv
        self.logger     =   None
        self.dataset: list[DataSet] = []
        self.shared     =   None
        self.ctx        =   None
        self.workdir: Optional[str] = None
        self.cache: Optional[ProjectCache] = None
        self.dataset_registry: Dict[str, DataSet] = {}
        self.preprocessor: Optional[DataPreprocessor] = None

    def init(self):
        if self.argv is None:
            self.args = self.cli.args.parse_args()
        else:
            self.args = self.cli.args.parse_args(self.argv)

        # Initialize logger early
        self.init_logger()

        if self._is_flowchart_command():
            self.render_flowchart()
            return

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
            runtime_expand_figure_types(self)
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
            self.style = load_styles(self.load_path, logger=self.logger)
            self.prebuild_profile_pipelines()
            runtime_prebuild_correlations(self)
            runtime_prepare_usage_plan(self)

            self.plot()

    def _is_flowchart_command(self) -> bool:
        command = getattr(self.args, "file", None)
        return isinstance(command, str) and command.strip().lower() == "flowchart"

    def render_flowchart(self) -> None:
        flowchart_file = getattr(self.args, "flowchart_file", None)
        if not flowchart_file:
            self.logger.error("No input flowchart JSON file specified. Usage: jplot flowchart path-to-flowchart.json")
            sys.exit(2)
        from .flowchart import render_flowchart_file

        try:
            render_flowchart_file(
                flowchart_file,
                output_path=getattr(self.args, "out", None),
                logger=self.logger,
            )
        except Exception as e:
            self.logger.error(f"Flowchart rendering failed: {e}")
            if getattr(self.args, "debug", False):
                import traceback

                self.logger.debug(traceback.format_exc())
            sys.exit(2)

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
        failures: list[str] = []
        report_figures: list[dict] = []
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

            fig_name = (
                (fig.get("name") if isinstance(fig, dict) else None)
                or getattr(figobj, "name", None)
                or "<noname>"
            )
            try:
                setup = figobj.from_dict(fig)
                if setup:
                    self.logger.warning(f"Successfully loading figure -> {figobj.name} setting")
                    # Collect post-transform observations during layer load.
                    figobj.health_observations = []
                    if isinstance(fig, dict) and isinstance(fig.get("frame"), dict):
                        figobj._yaml_frame = fig.get("frame")
                    figobj.plot()
                    self._evaluate_render_health(figobj)
                    self._maybe_write_agent_digest(fig, figobj)
                    # Always drop temporary expand stash after a successful figure.
                    if isinstance(fig, dict):
                        from .agent_digest import strip_digest_axes_stash

                        strip_digest_axes_stash(
                            fig,
                            yaml_path=getattr(self.yaml, "path", None),
                            figure_name=str(getattr(figobj, "name", None) or fig_name),
                            logger=self.logger,
                        )
                    if getattr(self.args, "report", False):
                        report_figures.append(
                            self._figure_report_block(figobj, fig_name)
                        )
                else:
                    if getattr(figobj, "_setup_status", None) == "disabled":
                        self.logger.warning(f"Skip figure {fig_name}: disabled in YAML.")
                    else:
                        msg = f"Skip figure {fig_name}: setup failed before plotting."
                        self.logger.error(msg)
                        failures.append(str(fig_name))
            except Exception as e:
                self.logger.error(f"Figure {fig_name} failed: {e}")
                failures.append(str(fig_name))
                if getattr(self.args, "debug", False):
                    import traceback

                    self.logger.debug(traceback.format_exc())
                continue
        if report_figures and getattr(self.args, "report", False):
            self._write_render_report(report_figures, failures=failures)
        if failures:
            self.logger.error(
                f"Render failed for {len(failures)} figure(s): {', '.join(failures)}"
            )
            sys.exit(1)
    def _evaluate_render_health(self, figobj) -> None:
        """Run JP-VIZ rules on post-transform observations collected during plot."""
        try:
            from .render_health import evaluate_health

            observations = getattr(figobj, "health_observations", None) or []
            if not observations:
                return
            bag = evaluate_health(observations)
            figobj.health_diagnostics = bag
            for diag in bag:
                level = str(getattr(diag, "level", "") or "").lower()
                msg = f"{diag.code}: {diag.message}"
                if level == "error":
                    self.logger.error(msg)
                elif level == "warning":
                    self.logger.warning(msg)
                else:
                    self.logger.info(msg)
        except Exception as exc:
            self.logger.debug(f"render health evaluation skipped: {exc}")

    def _figure_report_block(self, figobj, fig_name: str) -> dict:
        from .render_health import report_to_dict

        bag = getattr(figobj, "health_diagnostics", None)
        observations = getattr(figobj, "health_observations", None) or []
        block = report_to_dict(observations, diagnostics=bag)
        block["figure"] = str(fig_name)
        return block

    def _write_render_report(self, figures: list, *, failures: list[str]) -> None:
        """Ephemeral JP-VIZ / layer ledger for agents; delete after final plots."""
        import json
        from pathlib import Path

        yaml_path = getattr(self.yaml, "path", None)
        if yaml_path:
            out = Path(str(yaml_path)).expanduser().with_suffix(".render-report.json")
        else:
            out = Path(self.workdir or ".") / "jplot.render-report.json"
        payload = {
            "kind": "jplot_render_report",
            "ephemeral": True,
            "note": (
                "Ephemeral render-health report (JP-VIZ + layer observations). "
                "Delete this file after final plots are accepted — not part of "
                "the deliverable figure set."
            ),
            "yaml_path": str(yaml_path) if yaml_path else None,
            "failures": list(failures),
            "figures": figures,
        }
        try:
            out.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            self.logger.warning(
                f"Render report written -> {out} "
                "(ephemeral: delete after final plots)"
            )
        except Exception as exc:
            self.logger.warning(f"could not write --report file: {exc}")

    def _maybe_write_agent_digest(self, figure_cfg, figobj) -> None:
        """Write figure-level agent_output digest after a successful plot."""
        try:
            from .agent_digest import (
                load_figure_source_dataframe,
                maybe_write_figure_digest,
                parse_agent_output,
            )
            from .cli import JPLOT_VERSION

            if not isinstance(figure_cfg, dict):
                return
            # Do not load the source dataframe just to discover that no agent
            # digest was requested. This matters after rendering releases the
            # last layer's source data.
            root_config = getattr(self.yaml, "config", {}) or {}
            root_output = root_config.get("output", {}) if isinstance(root_config, dict) else {}
            if parse_agent_output(figure_cfg, root_output=root_output) is None:
                return
            # Prefer raw YAML figure block (pre type-expand may have type: fields).
            # After expand_figure_types, config figures are already expanded; still OK.
            df = load_figure_source_dataframe(self, figure_cfg)
            if df is None and hasattr(figobj, "context") and figobj.context is not None:
                # last resort: first registered dataset
                reg = getattr(self, "dataset_registry", {}) or {}
                for ds in reg.values():
                    if hasattr(ds, "get_data"):
                        df = ds.get_data()
                        break
            maybe_write_figure_digest(
                figure_cfg=figure_cfg,
                config=getattr(self.yaml, "config", {}) or {},
                dataframe=df,
                base_dir=getattr(self.yaml, "dir", None) or self.workdir,
                yaml_path=getattr(self.yaml, "path", None),
                logger=self.logger,
                jplot_version=str(JPLOT_VERSION),
            )
        except Exception as exc:
            # Non-strict failures already logged inside helper; strict re-raises.
            self.logger.warning(f"agent_output digest step failed: {exc}")

    def load_yaml(self):
        # If no YAML file provided, keep the message short (full help is jplot -h).
        yaml_path = getattr(self.args, "file", None)
        if not yaml_path:
            prog = getattr(self.cli.args, "prog", None) or "jplot"
            self.logger.error(
                "No input YAML file specified.\n"
                f"  usage: {prog} <file>\n"
                f"         {prog} <command> [args]\n"
                f"         {prog} -h"
            )
            sys.exit(2)
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
            if not record["extra"].get("to_console", False):
                return False
            if self.args.debug:
                return True
            return record["level"].no >= logger.level("WARNING").no

        def custom_format(record):
            return _format_console_record(record)

        # stderr, not stdout: stdout is reserved for machine output so a caller
        # can pipe `--json` straight into a parser while still seeing the log.
        logger.add(
            sys.stderr,
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
