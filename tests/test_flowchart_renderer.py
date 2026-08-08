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

    graph = _ClassicGraph(
        scene,
        {"classic": {"synthesize_missing_bridges": True, "bridge_y_offset": 0.6}},
    )

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

    graph.layout()
    module_top = graph.mains["FirstModule"]["pos"][1] + graph.module_icon_size / 2.0
    bridge_y = graph.bridges[bridge_id]["pos"][1]
    assert abs(bridge_y - module_top - 0.6) < 1e-9


def test_flowchart_orders_bridge_chain_by_previous_layer_y_position():
    scene = {
        "schema": "jarvisplot.scene/v1",
        "scene_type": "flowchart",
        "layers": [
            {"id": "layer_1", "index": 1, "nodes": ["Parameters", "var::top", "var::bottom"]},
            {"id": "layer_2", "index": 2, "nodes": []},
            {"id": "layer_3", "index": 3, "nodes": []},
            {
                "id": "layer_4",
                "index": 4,
                "nodes": ["FinalModule", "file::FinalModule::input::final_input"],
            },
        ],
        "nodes": [
            {"id": "Parameters", "kind": "source", "label": "Parameters", "layer": "layer_1"},
            {"id": "var::top", "kind": "variable", "role": "parameter", "label": "top", "layer": "layer_1"},
            {"id": "var::bottom", "kind": "variable", "role": "parameter", "label": "bottom", "layer": "layer_1"},
            {"id": "FinalModule", "kind": "module", "label": "FinalModule", "layer": "layer_4"},
            {
                "id": "file::FinalModule::input::final_input",
                "kind": "file",
                "role": "input_file",
                "label": "final_input",
                "layer": "layer_4",
            },
        ],
        "edges": [
            {"source": {"node": "Parameters", "port": "top"}, "target": {"node": "var::top", "port": "in"}, "role": "parameterflow"},
            {"source": {"node": "Parameters", "port": "bottom"}, "target": {"node": "var::bottom", "port": "in"}, "role": "parameterflow"},
            {
                "source": {"node": "var::bottom", "port": "out"},
                "target": {"node": "file::FinalModule::input::final_input", "port": "in"},
                "role": "fileflow",
                "metadata": {"variable": "bottom"},
            },
            {
                "source": {"node": "var::top", "port": "out"},
                "target": {"node": "file::FinalModule::input::final_input", "port": "in"},
                "role": "fileflow",
                "metadata": {"variable": "top"},
            },
        ],
    }

    graph = _ClassicGraph(scene, {"classic": {"synthesize_missing_bridges": True}})
    graph.layout()

    layer_2_bridges = graph._standalone_bridges("layer_2")
    layer_3_bridges = graph._standalone_bridges("layer_3")
    assert layer_2_bridges == ["bridge::top::L2", "bridge::bottom::L2"]
    assert layer_3_bridges == ["bridge::top::L3", "bridge::bottom::L3"]
    assert graph.bridges["bridge::top::L3"]["pos"][1] > graph.bridges["bridge::bottom::L3"]["pos"][1]


def test_flowchart_uses_triangle_for_file_output_variables():
    scene = {
        "schema": "jarvisplot.scene/v1",
        "scene_type": "flowchart",
        "layers": [{"id": "layer_1", "index": 1, "nodes": ["Calc", "file::Calc::output::result", "var::result"]}],
        "nodes": [
            {"id": "Calc", "kind": "module", "label": "Calc", "layer": "layer_1"},
            {"id": "file::Calc::output::result", "kind": "file", "role": "output_file", "label": "result", "layer": "layer_1"},
            {"id": "var::result", "kind": "variable", "role": "observable", "label": "result", "layer": "layer_1"},
        ],
        "edges": [
            {"source": {"node": "Calc", "port": "result"}, "target": {"node": "file::Calc::output::result", "port": "in"}, "role": "fileflow"},
            {"source": {"node": "file::Calc::output::result", "port": "out"}, "target": {"node": "var::result", "port": "in"}, "role": "fileflow"},
        ],
    }

    graph = _ClassicGraph(scene, {"classic": {}})
    graph.layout()

    assert graph.variables["var::result"]["marker_style"] == "<"
    assert graph.variables["var::result"]["marker_dx"] == 0.04
    assert graph.anchor("var::result", "in", "fileflow") == (
        graph.variables["var::result"]["pos"][0]
        - graph.variables["var::result"]["width"]
        - 0.1
        + 0.04,
        graph.variables["var::result"]["pos"][1],
    )
