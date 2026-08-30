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

        fig_type = str(info.get("type", "") or "").strip()
        if fig_type:
            # Every `type:` is lowered to layers before rendering. One still
            # here means the expansion failed and the reason was logged
            # upstream -- and the figure would otherwise render as whatever
            # the default card draws with no layers: a blank page, reported
            # as a success.
            raise ValueError(
                "figure '{}' still carries type: {} at render time, so its "
                "expansion into layers did not happen. The reason was reported "
                "when the config was loaded; `jplot explain {}` shows the "
                "slots this type expects.".format(fig.name, fig_type, fig_type)
            )

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
        if "figure" not in (fig.frame or {}):
            # A card may leave the figure size to be solved from the data
            # (the correlation matrix does).  Say so, rather than letting a
            # bare KeyError('figure') read as a corrupt config.
            raise KeyError(
                "style {} carries no Frame.figure: this card's figure size is "
                "solved from the data, so it can only be rendered through the "
                "figure type that runs the solve.".format(style_tokens)
            )
        fig.fig = plt.figure(**fig.frame["figure"])
        # An interactive backend sizes the figure to whole window pixels: the
        # macosx one calls set_size_inches(round(w * 100) / 100, ...) while it
        # builds the manager, so a size of 3.377953 in is saved as 3.37.  On an
        # authored card figsize is already a round number and nothing moves; on
        # a solved card it silently costs 0.2 mm of width, which is enough to
        # stop the correlation panel being exactly square.  Restate the size the
        # card asked for, without forwarding it back to the window.
        figsize = fig.frame["figure"].get("figsize")
        if figsize is not None:
            fig.fig.set_size_inches(*(float(v) for v in figsize), forward=False)

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
