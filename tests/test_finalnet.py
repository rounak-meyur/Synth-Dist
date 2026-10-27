"""
Tests for combining the primary and secondary networks into the final distribution
network (utils/finalnet_utils.py).
"""
import json
import logging

import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString

from utils.dataloader import Home
from utils.finalnet_utils import (
    combine_networks,
    distribution_network_summary,
    load_distribution_network,
    save_distribution_network,
    validate_distribution_network,
)

REGION = "777"
ROAD_GEOM = "LINESTRING (0 0, 0.0005 0.0002, 0.001 0)"  # bent road between road nodes 1 and 2


def home(hid, cord, load=1000.0):
    return Home(id=hid, cord=cord, profile=[load] * 24, peak=load, load=load)


HOMES = [home(11, (0.0, 0.0002)), home(12, (0.0001, 0.0002)), home(21, (0.001, 0.0002))]


def write_primary(prim_dir, sub_id, nodes, edges):
    """nodes: (id, cord, label, load); edges: (u, v, label, length, geometry or None)."""
    prim_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"node_id": n, "cord": json.dumps(list(c)), "label": lab, "load": load, "distance": 0.0}
         for n, c, lab, load in nodes]
    ).to_csv(prim_dir / f"{REGION}_{sub_id}_nodes.csv", index=False)
    pd.DataFrame(
        [{"source": u, "target": v, "geometry": g, "label": lab, "length": length}
         for u, v, lab, length, g in edges]
    ).to_csv(prim_dir / f"{REGION}_{sub_id}_edges.csv", index=False)


def write_secondary(sec_dir, edges):
    sec_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"from_node": u, "to_node": v, "length": length, "road_start": 1, "road_end": 2, "road_edge_key": 0}
         for u, v, length in edges]
    ).to_csv(sec_dir / f"{REGION}_secondary_edges.csv", index=False)


@pytest.fixture
def network_files(tmp_path):
    """
    Substation 4242 -> feeder road node 1 -> T1 -> road node 2 -> T2, along a bent road;
    homes 11 and 12 on T1, home 21 on T2.
    """
    prim, sec = tmp_path / "primnet", tmp_path / "secnet"
    write_primary(prim, "4242", nodes=[
        ("4242", (-0.001, 0.0), "S", 0.0),
        ("1", (0.0, 0.0), "R", 0.0),
        ("T1", (0.0, 0.0001), "T", 2000.0),
        ("2", (0.001, 0.0), "R", 0.0),
        ("T2", (0.001, 0.0001), "T", 1000.0),
    ], edges=[
        ("4242", "1", "feeder", 111.0, "LINESTRING (-0.001 0, 0 0)"),
        ("1", "T1", "primary", 11.0, None),
        ("1", "2", "primary", 120.0, ROAD_GEOM),
        ("2", "T2", "primary", 11.0, None),
    ])
    write_secondary(sec, [("T1", "11", 11.0), ("11", "12", 11.0), ("T2", "21", 11.0)])
    return prim, sec


def combine(network_files, sub_ids=("4242",), homes=HOMES):
    prim, sec = network_files
    return combine_networks(list(sub_ids), REGION, list(homes), str(prim), str(sec))


# combine_networks

def test_combine_builds_one_tree_per_substation(network_files):
    graph = combine(network_files)
    assert set(graph.nodes) == {"4242", "1", "2", "T1", "T2", "11", "12", "21"}
    assert nx.is_tree(graph)
    labels = nx.get_node_attributes(graph, "label")
    assert labels["4242"] == "S" and labels["T1"] == "T" and labels["11"] == "H"
    edge_labels = nx.get_edge_attributes(graph, "label")
    assert edge_labels[("4242", "1")] == "feeder"
    assert edge_labels[("1", "2")] == "primary"
    assert edge_labels[("T1", "11")] == "secondary"


def test_combine_keeps_primary_road_geometry(network_files):
    graph = combine(network_files)
    assert graph.edges["1", "2"]["geometry"].equals(LineString([(0, 0), (0.0005, 0.0002), (0.001, 0)]))
    # missing geometry becomes a straight line between the end nodes
    assert list(graph.edges["1", "T1"]["geometry"].coords) == [(0.0, 0.0), (0.0, 0.0001)]


def test_combine_keeps_loads(network_files):
    graph = combine(network_files)
    assert graph.nodes["T1"]["load"] == 2000.0
    assert graph.nodes["11"]["load"] == 1000.0
    assert graph.nodes["11"]["cord"] == (0.0, 0.0002)


def test_combine_missing_primary_file(network_files):
    with pytest.raises(FileNotFoundError, match="substation 9999"):
        combine(network_files, sub_ids=("4242", "9999"))


def test_combine_missing_secondary_file(network_files):
    prim, sec = network_files
    (sec / f"{REGION}_secondary_edges.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Secondary network file"):
        combine(network_files)


def test_combine_rejects_transformer_without_primary_network(network_files):
    _, sec = network_files
    write_secondary(sec, [("T1", "11", 11.0), ("T9", "21", 11.0)])
    with pytest.raises(ValueError, match="T9 .* not in any primary network"):
        combine(network_files)


def test_combine_rejects_unknown_home(network_files):
    with pytest.raises(ValueError, match="neither a home nor a transformer"):
        combine(network_files, homes=HOMES[:1])


def test_combine_rejects_transformer_in_two_primary_networks(network_files):
    prim, _ = network_files
    write_primary(prim, "5353", nodes=[("5353", (0.002, 0.0), "S", 0.0), ("T1", (0.0, 0.0001), "T", 2000.0)],
                  edges=[("5353", "T1", "feeder", 10.0, None)])
    with pytest.raises(ValueError, match="more than one primary network"):
        combine(network_files, sub_ids=("4242", "5353"))


# validate_distribution_network

def test_validate_accepts_valid_network(network_files):
    report = validate_distribution_network(combine(network_files), HOMES)
    assert report == {
        "is_forest": True, "trees": 1, "trees_without_substation": 0,
        "trees_with_several_substations": 0, "homes_missing": [], "isolated_nodes": [],
    }


def test_validate_reports_problems(network_files, caplog):
    graph = combine(network_files)
    graph.add_edge("12", "21", label="secondary", length=1.0, geometry=None)   # cycle
    graph.add_node("H9", cord=(1, 1), label="H", load=0.0)                       # isolated
    missing = home(99, (2, 2))
    with caplog.at_level(logging.WARNING, logger="finalnet_utils"):
        report = validate_distribution_network(graph, HOMES + [missing])
    assert not report["is_forest"]
    assert report["trees_without_substation"] == 1
    assert report["homes_missing"] == ["99"]
    assert report["isolated_nodes"] == ["H9"]
    assert "cycles" in caplog.text and "not connected" in caplog.text


# save / load / summary

def test_save_and_load_round_trip(tmp_path, network_files):
    graph = combine(network_files)
    nodes_file, edges_file = save_distribution_network(graph, REGION, str(tmp_path / "finalnet"))
    assert nodes_file.name == f"{REGION}_distribution_network_nodes.csv"
    assert not list((tmp_path / "finalnet").glob("*.tmp"))

    loaded = load_distribution_network(nodes_file, edges_file)
    assert set(loaded.nodes) == set(graph.nodes)
    assert {frozenset(e) for e in loaded.edges} == {frozenset(e) for e in graph.edges}
    for n, d in graph.nodes(data=True):
        assert loaded.nodes[n]["label"] == d["label"]
        assert loaded.nodes[n]["load"] == d["load"]
        assert loaded.nodes[n]["cord"] == pytest.approx(d["cord"])
    for u, v, d in graph.edges(data=True):
        assert loaded.edges[u, v]["geometry"].equals(d["geometry"])
        assert loaded.edges[u, v]["label"] == d["label"]


def test_load_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_distribution_network(tmp_path / "n.csv", tmp_path / "e.csv")


def test_summary(network_files):
    summary = distribution_network_summary(combine(network_files))
    assert summary["nodes_by_label"] == {"H": 3, "R": 2, "S": 1, "T": 2}
    assert summary["edges_by_label"] == {"feeder": 1, "primary": 3, "secondary": 3}
    assert summary["length_km_by_label"] == {"feeder": 0.111, "primary": 0.142, "secondary": 0.033}
    assert summary["home_load_total"] == 3000.0
