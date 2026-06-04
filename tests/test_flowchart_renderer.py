from __future__ import annotations

import json

from jarvisplot import render_flowchart
from jarvisplot.flowchart import _ClassicGraph, render_flowchart_file


def test_render_flowchart_scene_with_selectionflow(tmp_path):
    scene = {
        "schema": "jarvisplot.scene/v1",
        "scene_type": "flowchart",
        "scene_id": "test_flowchart",
        "metadata": {"producer": "Jarvis-HEP"},
        "layers": [
            {"id": "layer_1", "index": 1, "label": "Parameters", "nodes": ["Parameters", "var::x"]},
            {"id": "layer_2", "index": 2, "label": "Selected", "nodes": ["SelectedModule", "var::y"]},
        ],
        "nodes": [
            {
                "id": "Parameters",
                "kind": "source",
                "role": "parameter_source",
                "label": "Parameters",
                "layer": "layer_1",
                "out_ports": [{"id": "x", "role": "parameter"}],
            },
            {
                "id": "var::x",
                "kind": "variable",
                "role": "parameter",
                "label": "x",
                "layer": "layer_1",
                "in_ports": [{"id": "in", "role": "parameter"}],
                "out_ports": [{"id": "out", "role": "parameter"}],
            },
            {
                "id": "SelectedModule",
                "kind": "module",
                "role": "operas",
                "label": "SelectedModule",
                "layer": "layer_2",
                "selection": {"expression": "x > 0", "variables": ["x"]},
                "in_ports": [{"id": "selection::x", "role": "selection", "label": "x"}],
                "out_ports": [{"id": "y", "role": "variable"}],
                "metadata": {"module_type": "Operas"},
            },
            {
                "id": "var::y",
                "kind": "variable",
                "role": "observable",
                "label": "y",
                "layer": "layer_2",
                "in_ports": [{"id": "in", "role": "observable"}],
                "out_ports": [{"id": "out", "role": "observable"}],
            },
        ],
        "edges": [
            {
                "source": {"node": "Parameters", "port": "x"},
                "target": {"node": "var::x", "port": "in"},
                "role": "parameterflow",
            },
            {
                "source": {"node": "var::x", "port": "out"},
                "target": {"node": "SelectedModule", "port": "selection::x"},
                "role": "selectionflow",
            },
            {
                "source": {"node": "SelectedModule", "port": "y"},
                "target": {"node": "var::y", "port": "in"},
                "role": "dataflow",
            },
        ],
    }

    scene_path = tmp_path / "flowchart.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    output_path = tmp_path / "flowchart.png"

    rendered = render_flowchart_file(str(scene_path), output_path=str(output_path))

    assert rendered == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_flowchart_public_api_accepts_scene_mapping(tmp_path):
    scene = {
        "schema": "jarvisplot.scene/v1",
        "scene_type": "flowchart",
        "scene_id": "mapping_api",
        "layers": [{"id": "layer_1", "index": 1, "label": "Parameters", "nodes": ["Parameters"]}],
        "nodes": [
            {
                "id": "Parameters",
                "kind": "source",
                "role": "parameter_source",
                "label": "Parameters",
                "layer": "layer_1",
                "out_ports": [],
            }
        ],
        "edges": [],
    }
    output_path = tmp_path / "mapping_api.png"

    rendered = render_flowchart(scene, output_path=output_path)

    assert rendered == str(output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_flowchart_synthesizes_missing_bridge_for_cross_layer_file_input():
    scene = {
        "schema": "jarvisplot.scene/v1",
        "scene_type": "flowchart",
        "layers": [
            {"id": "layer_1", "index": 1, "nodes": ["Parameters", "var::x"]},
            {"id": "layer_2", "index": 2, "nodes": ["FirstModule", "file::FirstModule::input::first_input"]},
            {"id": "layer_3", "index": 3, "nodes": ["SecondModule", "file::SecondModule::input::second_input"]},
        ],
        "nodes": [
            {"id": "Parameters", "kind": "source", "label": "Parameters", "layer": "layer_1"},
            {"id": "var::x", "kind": "variable", "role": "parameter", "label": "x", "layer": "layer_1"},
            {"id": "FirstModule", "kind": "module", "label": "FirstModule", "layer": "layer_2"},
            {
                "id": "file::FirstModule::input::first_input",
                "kind": "file",
                "role": "input_file",
                "label": "first_input",
                "layer": "layer_2",
            },
            {"id": "SecondModule", "kind": "module", "label": "SecondModule", "layer": "layer_3"},
            {
                "id": "file::SecondModule::input::second_input",
                "kind": "file",
                "role": "input_file",
                "label": "second_input",
                "layer": "layer_3",
            },
        ],
        "edges": [
            {"source": {"node": "Parameters", "port": "x"}, "target": {"node": "var::x", "port": "in"}, "role": "parameterflow"},
            {
                "source": {"node": "var::x", "port": "out"},
                "target": {"node": "file::FirstModule::input::first_input", "port": "in"},
                "role": "fileflow",
                "metadata": {"variable": "x"},
            },
            {
                "source": {"node": "var::x", "port": "out"},
                "target": {"node": "file::SecondModule::input::second_input", "port": "in"},
                "role": "fileflow",
                "metadata": {"variable": "x"},
            },
        ],
    }

    graph = _ClassicGraph(scene, {"classic": {"synthesize_missing_bridges": True}})

    bridge_id = "bridge::x::L2"
    assert bridge_id in graph.nodes
    assert any(
        edge["role"] == "bridgeflow"
        and edge["source"]["node"] == "var::x"
        and edge["target"]["node"] == bridge_id
        for edge in graph.edges
    )
    assert any(
        edge["role"] == "fileflow"
        and edge["source"]["node"] == bridge_id
        and edge["target"]["node"] == "file::SecondModule::input::second_input"
        for edge in graph.edges
    )
    assert not any(
        edge["role"] == "fileflow"
        and edge["source"]["node"] == "var::x"
        and edge["target"]["node"] == "file::SecondModule::input::second_input"
        for edge in graph.edges
    )
