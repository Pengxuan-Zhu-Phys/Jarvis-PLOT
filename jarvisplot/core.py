#!/usr/bin/env python3 

from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Dict
from .cli import CLI
from loguru import logger
import os, sys
from .config import ConfigLoader
from .data_loader import DataSet
import io
from contextlib import redirect_stdout
jppwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import json 
from .Figure.data_pipelines import SharedContent, DataContext

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

    def init(self):
        self.args = self.cli.args.parse_args()

        # Initialize logger early
        self.init_logger()

        self.load_cmaps()

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
            self.load_dataset()
            self.rename_hdf5_and_renew_yaml()
        else:
            self.load_dataset()
            if self.shared is None:
                self.shared = SharedContent(logger=self.logger)
            self.ctx = DataContext(self.shared)
            for dts in self.dataset:
                self.ctx.update(dts.name, dts.data)

            # Register external functions (e.g. lazy-loaded interpolators) into the expression runtime.
            self.load_interpolators()

            self.load_styles()
            self.plot()

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
            if getattr(self.args, "no_logo", False):
                figobj.print = True

            try:
                if figobj.set(fig):
                    self.logger.warning(f"Succefully loading figure -> {figobj.name} setting")
                    figobj.plot()
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

    def load_dataset(self):
        dts = self.yaml.config['DataSet']
        for dt in dts:
            dataset = DataSet()
            dataset.logger = self.logger
            dataset.setinfo(dt, self.yaml.dir)
            self.dataset.append(dataset)

    def rename_hdf5_and_renew_yaml(self):
        for dt in self.dataset:
            self.logger.warning("DataSet -> {}, type -> {}".format(dt.name, dt.type))
            vmap_dict = {}
            vmap_list = []
            if dt.type == "hdf5":
                for ii, kk in enumerate(dt.keys):
                    vname = "Var{}@{}".format(ii, dt.name)
                    vmap_dict[kk] = vname
                    vmap_list.append({
                        "source_name":  r"{}".format(kk),
                        "new_name":     vname
                    })
                self.yaml.update_dataset(dt.name, {"columnmap": {"list": vmap_list}})
                dt.rename_columns(vmap_dict)
                print(dt.keys)

        import yaml
        with open(self.args.out, 'w', encoding='utf-8') as f1:
            yaml.dump(self.yaml.config, f1, sort_keys=False, default_flow_style=False, indent=2)