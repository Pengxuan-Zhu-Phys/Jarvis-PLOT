from __future__ import annotations

import sys
from types import SimpleNamespace

from rich.cells import cell_len

import jarvisplot.cli as cli_module
from jarvisplot.cli import JPLOT_VERSION, CLI, render_flowchart_help, render_help
from jarvisplot.client import main


def _panel_lines(rendered: str) -> list[str]:
    return [
        line
        for line in rendered.splitlines()
        if line.startswith(("╭", "│", "╰"))
    ]


def test_root_help_uses_one_fixed_panel_width_and_aligned_columns(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    rendered = render_help()

    panel_lines = _panel_lines(rendered)
    assert {cell_len(line) for line in panel_lines} == {80}
    assert "╭─ Commands" in rendered
    assert "╭─ Inputs" in rendered
    assert "╭─ Options" in rendered
    assert "│ validate" in rendered
    assert "│ cap" in rendered
    assert "│ data" in rendered
    assert "│ flowchart" in rendered
    assert "│ run " not in rendered  # DR-08: no jplot run
    assert "│ --help                    -h" in rendered
    assert "│ --parse-data" in rendered
    assert "│ --rebuild-cache" in rendered
    assert "validate <file>" in rendered
    assert "cap [section]" in rendered
    assert "data describe|head|eval" in rendered

    description_starts = {
        rendered.splitlines()[i].index(description)
        for i, line in enumerate(rendered.splitlines())
        for description in (
            "render a Jarvis-HEP flowchart scene JSON",
            "YAML to render (bare path; not a verb)",
            "show this help message and exit",
        )
        if description in line
    }
    assert len(description_starts) == 1
    assert "flowchart_file                    path" in rendered
    assert not rendered.endswith("\n\n")


def test_help_width_expands_with_the_terminal(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    monkeypatch.setattr(
        cli_module.shutil,
        "get_terminal_size",
        lambda: SimpleNamespace(columns=96),
    )

    rendered = render_help()

    assert {cell_len(line) for line in _panel_lines(rendered)} == {96}


def test_tty_help_keeps_rich_colors(monkeypatch):
    tty_stdout = SimpleNamespace(isatty=lambda: True)
    monkeypatch.setattr(cli_module.sys, "stdout", tty_stdout)

    rendered = render_help()

    assert "\x1b[" in rendered


def test_flowchart_help_reuses_the_same_panel_geometry(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    rendered = render_flowchart_help(prog="Jarvis2 plot")

    assert {cell_len(line) for line in _panel_lines(rendered)} == {80}
    assert "│ Jarvis2 plot flowchart <flowchart_file>" in rendered
    assert "╭─ Arguments" in rendered
    assert "╭─ Options" in rendered
    assert "│ flowchart_file                    path" in rendered
    assert "│ --help                    -h" in rendered
    assert not rendered.endswith("\n\n")


def test_cli_help_and_version_are_integer_returning_and_forwardable(capsys):
    assert main(["-h"]) == 0
    help_output = capsys.readouterr()
    assert help_output.err == ""
    assert "jplot flowchart <flowchart_file>" in help_output.out

    assert main(["flowchart", "-h"], prog="Jarvis2 plot") == 0
    subcommand_output = capsys.readouterr()
    assert subcommand_output.err == ""
    assert "Jarvis2 plot flowchart <flowchart_file>" in subcommand_output.out

    assert main(["-v"]) == 0
    version_output = capsys.readouterr()
    assert version_output.out.strip() == f"JarvisPLOT {JPLOT_VERSION}"
    assert version_output.err == ""


def test_cli_parser_preserves_plot_and_flowchart_argument_semantics():
    parser = CLI().args

    plot_args = parser.parse_args(["config.yaml", "--rebuild-cache"])
    assert plot_args.file == "config.yaml"
    assert plot_args.rebuild_cache is True

    flowchart_args = parser.parse_args(["flowchart", "scene.json", "--out", "scene.png"])
    assert flowchart_args.file == "flowchart"
    assert flowchart_args.flowchart_file == "scene.json"
    assert flowchart_args.out == "scene.png"
