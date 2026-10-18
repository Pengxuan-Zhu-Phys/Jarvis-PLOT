#!/usr/bin/env python3

import sys


def main(argv=None, *, prog="jplot"):
    """Entry point: verb router in front, bare-path render behind.

    Render is **not** a verb: ``jplot <file>`` (and therefore
    ``Jarvis2 plot <file>``) draws figures. Agent verbs are dispatched without
    touching :mod:`jarvisplot.core`, so they never pay for the render stack.
    See DR-08: there is no ``jplot run``.
    """
    from jarvisplot.verbs import route

    tokens = list(sys.argv[1:] if argv is None else argv)
    if {"-v", "--version"}.intersection(tokens) and set(tokens) <= {"-v", "--version", "--json"}:
        from jarvisplot.version import get_version

        version = get_version()
        if "--json" in tokens:
            import json
            print(json.dumps({"ok": True, "command": "version", "name": "Jarvis-PLOT",
                              "version": version}, indent=2))
        else:
            print(f"Jarvis-PLOT {version}")
        return 0
    handled, code = route(tokens, prog=prog)
    if handled:
        return code

    from jarvisplot.agent_io import system_exit_code
    from jarvisplot.core import JarvisPLOT
    from jarvisplot.cache_registry import CacheUsageError

    jp = JarvisPLOT(prog=prog, argv=tokens)
    try:
        jp.init()
    except SystemExit as exc:
        return system_exit_code(exc)
    except CacheUsageError as exc:
        import os
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        Console(stderr=True, no_color=bool(os.environ.get("NO_COLOR"))).print(
            Panel(Text(str(exc)), title=Text(prog), title_align="left", border_style="red"))
        if "--debug" in tokens or "-d" in tokens:
            import traceback
            traceback.print_exc()
        return 2
    return 0
