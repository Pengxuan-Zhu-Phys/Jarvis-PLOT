"""The verb router and the internal configuration checks retained by doctor.

The load-bearing assertion here is the *negative* one: adding verbs must not
change what ``jplot <file>`` does.
"""

from __future__ import annotations

import textwrap

import pytest

from jarvisplot.cli import CLI, render_help
from jarvisplot.client import main
from jarvisplot.verbs import is_verb, route
from jarvisplot.validation import validate_file


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
    assert not is_verb("validate")
    assert is_verb("doctor")
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


def _assert_unknown_cli_message(err: str, name: str) -> None:
    low = err.lower()
    assert "unknown command" in low
    assert name in err
    assert "jplot -h" in err or "jplot man" in err or "jplot cap" in err
    assert "use the jplot cli" in low


def test_unknown_bare_command_is_rejected(capsys):
    handled, code = route(["whaat"])
    assert handled is True
    assert code == 2
    _assert_unknown_cli_message(capsys.readouterr().err, "whaat")


def test_run_is_rejected_not_aliased(capsys):
    """DR-08: bare path renders; `jplot run` is not a verb — same unknown-command shape."""
    assert is_verb("run") is False
    handled, code = route(["run", "x.yaml"])
    assert handled is True
    assert code == 2
    _assert_unknown_cli_message(capsys.readouterr().err, "run")


def test_context_verb_removed(capsys):
    """context is not a verb — same short CLI pointer, no replacement cookbook."""
    assert is_verb("context") is False
    handled, code = route(["context", "--data", "x.csv", "--json"])
    assert handled is True
    assert code == 2
    _assert_unknown_cli_message(capsys.readouterr().err, "context")


def test_main_rejects_run_verb(capsys):
    assert main(["run", "whatever.yaml"]) == 2
    _assert_unknown_cli_message(capsys.readouterr().err, "run")


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
    assert "│ validate" not in rendered
    assert "doctor" in rendered
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


def test_removed_validate_command_cannot_rewrite_yaml(good_config, capsys):
    original = good_config.read_text()
    assert main(["validate", str(good_config), "--fix", "--write"]) == 2
    _assert_unknown_cli_message(capsys.readouterr().err, "validate")
    assert good_config.read_text() == original


def test_internal_validation_accepts_clean_config(good_config):
    config, bag = validate_file(str(good_config))
    assert config is not None
    assert bag.ok and not len(bag)


def test_internal_validation_reports_every_problem_in_one_pass(tmp_path):
    config = _write(tmp_path, "bad.yaml", """
        DataSet:
          - {name: df, path: ./missing.csv, type: csv}
        Figures:
          - name: f1
            layers:
              - name: a
                data: [{source: dff}]
                axes: ax
                method: scatter
                coordinates: {x: {expr: x}, y: {expr: y}}
          - name: f1
            layers: {}
        """)
    _, bag = validate_file(str(config))
    assert {d.code for d in bag} == {"JP-DAT-004", "JP-REF-001", "JP-SCH-004", "JP-FIG-003"}
    assert len(bag.errors) == 4


def test_internal_validation_suggests_dataset_name(tmp_path):
    (tmp_path / "samples.csv").write_text("x\n1\n")
    config = _write(tmp_path, "typo.yaml", """
        DataSet:
          - {name: df_samples_0, path: ./samples.csv, type: csv}
        Figures:
          - name: f1
            layers:
              - data: [{source: df_smaples_0}]
                axes: ax
                method: scatter
                coordinates: {x: {expr: x}, y: {expr: x}}
        """)
    _, bag = validate_file(str(config))
    diagnostic = next(d for d in bag if d.code == "JP-REF-001")
    assert diagnostic.context["did_you_mean"] == ["df_samples_0"]
    assert diagnostic.path == "$.Figures[0].layers[0].data[0].source"


def test_internal_validation_accepts_shared_source(tmp_path):
    (tmp_path / "samples.csv").write_text("x,y\n1,2\n")
    config = _write(tmp_path, "shared.yaml", """
        DataSet:
          - {name: df, path: ./samples.csv, type: csv}
        Figures:
          - name: f1
            layers:
              - name: a
                data: [{source: df}]
                share_data: prof
                axes: ax
                method: scatter
                coordinates: {x: {expr: x}, y: {expr: y}}
              - name: b
                data: [{source: prof}]
                axes: ax
                method: voronoi
                coordinates: {x: {expr: x}, y: {expr: y}}
        """)
    _, bag = validate_file(str(config))
    assert bag.ok and not len(bag)


def test_internal_validation_reports_yaml_location(tmp_path):
    path = tmp_path / "syntax.yaml"
    path.write_text("Figures:\n  - name: f1\n   layers: []\n")
    config, bag = validate_file(str(path))
    assert config is None
    assert bag.errors[0].code == "JP-YML-001"
    assert "line" in bag.errors[0].message


def test_internal_validation_reports_missing_file(tmp_path):
    config, bag = validate_file(str(tmp_path / "nope.yaml"))
    assert config is None
    assert bag.errors[0].code == "JP-YML-000"


def test_internal_validation_requires_datasets_and_figures(tmp_path):
    path = _write(tmp_path, "empty.yaml", "version: '0.3'\n")
    _, bag = validate_file(str(path))
    missing = {d.message.split("'")[1] for d in bag if d.code == "JP-SCH-002"}
    assert missing == {"DataSet", "Figures"}
