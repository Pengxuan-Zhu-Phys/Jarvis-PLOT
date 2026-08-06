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
    handled, code = route(tokens, prog=prog)
    if handled:
        return code

    from jarvisplot.core import JarvisPLOT

    jp = JarvisPLOT(prog=prog, argv=tokens)
    try:
        jp.init()
    except SystemExit as exc:
        return int(exc.code)
    return 0
