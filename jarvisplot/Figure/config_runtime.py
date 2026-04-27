from __future__ import annotations

from pathlib import Path
from typing import Mapping


def apply_figure_config(fig, info: Mapping) -> bool:
    """Apply a YAML figure block to a Figure instance."""
    if not isinstance(info, Mapping):
        raise TypeError("from_dict expects a mapping/dict")

    fig._setup_status = "pending"
    fig._setup_error = None

    try:
        if "name" in info:
            fig.name = info["name"]

        if "yaml_dir" in info:
            fig._yaml_dir = info.get("yaml_dir")
        elif "_yaml_dir" in info:
            fig._yaml_dir = info.get("_yaml_dir")
        elif "yaml_path" in info:
            try:
                fig._yaml_dir = str(Path(info.get("yaml_path")).expanduser().resolve().parent)
            except Exception:
                pass

        if "debug" in info:
            fig.debug = info["debug"]
            try:
                fig.logger.debug("Loading plot -> {} in debug mode".format(fig.name))
            except Exception:
                pass

        fig._enable = info.get("enable", True)
        if not fig._enable:
            fig._setup_status = "disabled"
            return False

        if "style" in info:
            style_tokens = info["style"]
        else:
            style_tokens = ["a4paper_2x1"]
        fig.style = style_tokens
        fig.logger.debug("Figure style loaded")
        if style_tokens and "gambit" in str(style_tokens[0]).lower():
            fig.mode = "gambit"

        if "frame" in info:
            fig.frame = info["frame"]
        fig.logger.debug("Figure frame information loaded")

        import matplotlib.pyplot as plt

        plt.rcParams["mathtext.fontset"] = "stix"
        fig.fig = plt.figure(**fig.frame["figure"])

        if fig.print:
            try:
                if isinstance(fig.frame.get("axes"), dict):
                    fig.frame["axes"].pop("axlogo", None)
                fig.frame.pop("axlogo", None)
            except Exception:
                pass

        fig.load_axes()

        if "layers" in info:
            fig.layers = info["layers"]
        elif getattr(fig, "_default_layers", None):
            fig.layers = fig._default_layers

        fig._setup_status = "ok"
        return True
    except Exception as e:
        if fig.logger:
            try:
                import traceback

                fig.logger.error(
                    "Failed to configure figure '{}': {}".format(
                        getattr(fig, "name", "<noname>"), e
                    )
                )
                fig.logger.debug(traceback.format_exc())
            except Exception:
                pass
        fig._setup_status = "failed"
        fig._setup_error = e
        return False
