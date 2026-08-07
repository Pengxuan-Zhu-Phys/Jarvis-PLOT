"""A2 acceptance: the verb router, and `jplot validate` end to end.

The load-bearing assertion here is the *negative* one: adding verbs must not
change what ``jplot <file>`` does.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from jarvisplot.cli import CLI, render_help
from jarvisplot.client import main
from jarvisplot.verbs import is_verb, route


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def good_config(tmp_path):
    (tmp_path / "samples.csv").write_text("x,y,LogL\n1,2,-3\n", encoding="utf-8")
    return _write(
        tmp_path,
        "good.yaml",
        """
        version: "0.3"
        DataSet:
          - {name: df, path: ./samples.csv, type: csv}
        Figures:
          - name: f1
            layers:
              - name: scatter
                data:
                  - source: df
                axes: ax
                method: scatter
                coordinates:
                  x: {expr: x}
                  y: {expr: y}
        output: {dir: ./plots}
        """,
    )


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #


def test_known_verbs_are_claimed_by_the_router():
    assert is_verb("validate")
    assert is_verb("cap")
    assert is_verb("data")
    assert is_verb("man")
    assert is_verb("config")
    assert not is_verb("flowchart"), "flowchart stays owned by core.py"
    assert not is_verb("config.yaml")


def test_router_falls_through_for_files_and_flowchart():
    assert route(["config.yaml"]) == (False, 0)
    assert route(["flowchart", "scene.json"]) == (False, 0)
    assert route([]) == (False, 0)


def test_unknown_bare_command_is_rejected(capsys):
    handled, code = route(["whaat"])
    assert handled is True
    assert code == 2
    err = capsys.readouterr().err
    assert "unknown command" in err


def test_run_is_rejected_not_aliased(capsys):
    """DR-08: bare path renders; `jplot run` must not silently mean render."""
    assert is_verb("run") is False
    handled, code = route(["run", "x.yaml"])
    assert handled is True
    assert code == 2
    err = capsys.readouterr().err
    assert "jplot <file>" in err or "bare path" in err.lower() or "Jarvis2 plot" in err
    assert "run a scan" in err or "no `jplot run`" in err or "no jplot run" in err.lower()


def test_context_verb_removed(capsys):
    """Aggregated context pack was product-discouraged; command must not exist."""
    assert is_verb("context") is False
    handled, code = route(["context", "--data", "x.csv", "--json"])
    assert handled is True
    assert code == 2
    err = capsys.readouterr().err
    assert "context" in err.lower()
    assert "data describe" in err or "agent_output" in err


def test_main_rejects_run_verb(capsys):
    assert main(["run", "whatever.yaml"]) == 2
    err = capsys.readouterr().err
    assert "Jarvis2 plot" in err or "jplot <file>" in err


def test_legacy_parser_semantics_are_untouched():
    """Same assertions as tests/test_cli_help.py, restated as a router guard."""
    parser = CLI().args
    plot_args = parser.parse_args(["config.yaml", "--rebuild-cache"])
    assert plot_args.file == "config.yaml"
    assert plot_args.rebuild_cache is True

    flowchart_args = parser.parse_args(["flowchart", "scene.json", "--out", "scene.png"])
    assert flowchart_args.file == "flowchart"
    assert flowchart_args.flowchart_file == "scene.json"


def test_root_help_lists_commands_from_the_card():
    rendered = render_help()
    assert "validate" in rendered
    assert "flowchart" in rendered
    # DR-08: render is bare path; never advertise `run` as a command.
    assert "│ run " not in rendered
    assert "jplot run" not in rendered


# --------------------------------------------------------------------------- #
# A4: the --json convention, enforced for every verb that ever gets registered
# --------------------------------------------------------------------------- #


def _accepts_json(parser) -> bool:
    """True if this parser or any of its subparsers takes --json."""
    for action in parser._actions:
        if "--json" in getattr(action, "option_strings", ()):
            return True
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            for sub in choices.values():
                if hasattr(sub, "_actions") and _accepts_json(sub):
                    return True
    return False


@pytest.mark.parametrize("verb", sorted(__import__("jarvisplot.verbs", fromlist=["VERBS"]).VERBS))
def test_every_verb_accepts_json(verb):
    import importlib

    from jarvisplot.verbs import VERBS

    module = importlib.import_module(VERBS[verb].partition(":")[0])
    assert hasattr(module, "build_parser"), f"{verb} must expose build_parser()"
    assert _accepts_json(module.build_parser()), f"{verb} must accept --json"


# --------------------------------------------------------------------------- #
# validate verb
# --------------------------------------------------------------------------- #


def test_validate_clean_config_exits_zero(good_config, capsys):
    assert main(["validate", str(good_config)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "", "human mode must not write to stdout"
    assert "OK" in captured.err


def test_validate_json_stdout_is_pure(good_config, capsys):
    assert main(["validate", str(good_config), "--json"]) == 0
    captured = capsys.readouterr()
    env = json.loads(captured.out)
    assert env["kind"] == "validate"
    assert env["ok"] is True
    assert env["diagnostics"] == []


def test_validate_reports_every_problem_in_one_pass(tmp_path, capsys):
    config = _write(
        tmp_path,
        "bad.yaml",
        """
        DataSet:
          - {name: df, path: ./missing.csv, type: csv}
        Figures:
          - name: f1
            layers:
              - name: a
                data: [{source: dff}]
                method: scatter
                coordinates: {x: {expr: x}, y: {expr: y}}
          - name: f1
            layers: {}
        """,
    )
    assert main(["validate", str(config), "--json"]) == 1
    env = json.loads(capsys.readouterr().out)
    codes = {d["code"] for d in env["diagnostics"]}
    assert codes == {
        "JP-DAT-004",  # missing.csv does not exist
        "JP-REF-001",  # source: dff is not a declared dataset
        "JP-SCH-004",  # layers: {} is a mapping, not a list
        "JP-FIG-003",  # figure name f1 used twice
    }
    assert env["data"]["error_count"] == 4


def test_validate_suggests_the_right_dataset_name(tmp_path, capsys):
    (tmp_path / "samples.csv").write_text("x\n1\n", encoding="utf-8")
    config = _write(
        tmp_path,
        "typo.yaml",
        """
        DataSet:
          - {name: df_samples_0, path: ./samples.csv, type: csv}
        Figures:
          - name: f1
            layers:
              - name: a
                data: [{source: df_smaples_0}]
                method: scatter
                coordinates: {x: {expr: x}, y: {expr: y}}
        """,
    )
    main(["validate", str(config), "--json"])
    env = json.loads(capsys.readouterr().out)
    ref = next(d for d in env["diagnostics"] if d["code"] == "JP-REF-001")
    assert ref["context"]["did_you_mean"] == ["df_samples_0"]
    assert ref["path"] == "$.Figures[0].layers[0].data[0].source"


def test_validate_accepts_share_data_as_a_source(tmp_path, capsys):
    """`share_data` publishes a name; consuming it is not an unknown source."""
    (tmp_path / "samples.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    config = _write(
        tmp_path,
        "shared.yaml",
        """
        DataSet:
          - {name: df, path: ./samples.csv, type: csv}
        Figures:
          - name: f1
            layers:
              - name: a
                data: [{source: df}]
                share_data: prof
                method: scatter
                coordinates: {x: {expr: x}, y: {expr: y}}
              - name: b
                data: [{source: prof}]
                method: voronoi
                coordinates: {x: {expr: x}, y: {expr: y}}
        """,
    )
    assert main(["validate", str(config), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []


def test_validate_reports_yaml_syntax_errors_with_a_location(tmp_path, capsys):
    config = tmp_path / "syntax.yaml"
    config.write_text("Figures:\n  - name: f1\n   layers: []\n", encoding="utf-8")
    assert main(["validate", str(config), "--json"]) == 1
    env = json.loads(capsys.readouterr().out)
    assert env["data"]["parsed"] is False
    assert env["diagnostics"][0]["code"] == "JP-YML-001"
    assert "line" in env["diagnostics"][0]["message"]


def test_validate_missing_file_is_a_diagnostic_not_a_traceback(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "nope.yaml"), "--json"]) == 1
    env = json.loads(capsys.readouterr().out)
    assert env["diagnostics"][0]["code"] == "JP-YML-000"


def test_validate_requires_datasets_and_figures(tmp_path, capsys):
    config = _write(tmp_path, "empty.yaml", "version: '0.3'\n")
    main(["validate", str(config), "--json"])
    env = json.loads(capsys.readouterr().out)
    missing = {d["message"].split("'")[1] for d in env["diagnostics"] if d["code"] == "JP-SCH-002"}
    assert missing == {"DataSet", "Figures"}


def test_validate_bad_usage_exits_two(capsys):
    assert main(["validate"]) == 2
