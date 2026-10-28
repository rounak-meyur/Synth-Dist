"""
Tests for partitioning transformers to the nearest substation by network distance
(utils/partition_utils.py).
"""
import json

import networkx as nx
import pytest
from geopy.distance import geodesic

from conftest import SIZE, geodesic_m, grid_node_coord, grid_node_id, make_combined_grid
from utils.dataloader import Substation
from utils.partition_utils import (
    NetworkPartitioner,
    load_partitioning,
    load_substation_partition,
    read_substation_list,
    save_substation_partitions,
)

def substation_at(sub_id: str, i: int, j: int, offset: float = 0.00005) -> Substation:
    x, y = grid_node_coord(i, j)
    return Substation(id=sub_id, cord=(x + offset, y + offset))


def network_distances(G, node):
    return nx.single_source_dijkstra_path_length(G, node, weight="length")


# find_nearest_road_nodes

def test_find_nearest_road_node_ignores_transformers():
    G = make_combined_grid()
    # right next to transformer T0303 (midpoint of (3,3)-(4,3)), but the answer must be a road node
    t_cord = G.nodes["T0303"]["cord"]
    sub = Substation(id="1", cord=(t_cord[0] + 0.00001, t_cord[1]))
    nearest = NetworkPartitioner(G).find_nearest_road_nodes([sub], search_radius=0.002)
    assert nearest["1"] in (grid_node_id(3, 3), grid_node_id(4, 3))
    assert G.nodes[nearest["1"]]["label"] == "R"


def test_find_nearest_road_nodes_for_each_substation():
    G = make_combined_grid()
    subs = [substation_at("1", 2, 5), substation_at("2", 12, 9)]
    nearest = NetworkPartitioner(G).find_nearest_road_nodes(subs, search_radius=0.001)
    assert nearest == {"1": grid_node_id(2, 5), "2": grid_node_id(12, 9)}


def test_find_nearest_road_nodes_retries_with_larger_radius():
    G = make_combined_grid()
    x, y = grid_node_coord(0, 0)
    far = Substation(id="1", cord=(x - 0.03, y - 0.03))  # outside 0.001 and 0.01, inside 0.1
    nearest = NetworkPartitioner(G).find_nearest_road_nodes([far], search_radius=0.001)
    assert nearest == {"1": grid_node_id(0, 0)}


def test_find_nearest_road_nodes_gives_up():
    G = make_combined_grid()
    x, y = grid_node_coord(0, 0)
    with pytest.raises(ValueError):
        NetworkPartitioner(G).find_nearest_road_nodes([Substation(id="1", cord=(x - 1, y - 1))], search_radius=0.001)


# partition_transformers

def test_partition_assigns_every_node_to_nearest_substation():
    G = make_combined_grid()
    substation_nodes = {"1": grid_node_id(1, 1), "2": grid_node_id(13, 10)}
    assignments = NetworkPartitioner(G).partition_transformers(substation_nodes)

    assert set(assignments) == {"1", "2"}
    cells = {sid: set(nodes) for sid, nodes in assignments.items()}
    assert not cells["1"] & cells["2"], "partitions must be disjoint"
    assert cells["1"] | cells["2"] == set(G.nodes), "every node must be assigned"

    dist = {sid: network_distances(G, node) for sid, node in substation_nodes.items()}
    for sid, nodes in cells.items():
        other = "2" if sid == "1" else "1"
        for n in nodes:
            assert dist[sid][n] <= dist[other][n] + 1e-6, f"{n} is closer to substation {other}"


def test_partition_uses_network_not_geographic_distance():
    """
    A barrier between columns 7 and 8 with a single crossing on the top row. Transformer T0800
    (between columns 8 and 9 on the bottom row) is geographically closer to substation A at
    column 6 than to B at column 15, but by road it is much closer to B.
    """
    G = make_combined_grid(barrier_at=7)
    a, b = grid_node_id(6, 0), grid_node_id(15, 0)
    t = "T0800"
    assert geodesic_m(G.nodes[t]["cord"], G.nodes[a]["cord"]) < geodesic_m(G.nodes[t]["cord"], G.nodes[b]["cord"])
    assert network_distances(G, a)[t] > network_distances(G, b)[t]

    assignments = NetworkPartitioner(G).partition_transformers({"A": a, "B": b})
    assert t in assignments["B"]
    assert t not in assignments["A"]
    # all transformers east of the barrier on the lower rows belong to B
    # (near the bridge on the top rows, A can be closer by road)
    east = [n for n, d in G.nodes(data=True) if d["label"] == "T" and int(n[1:3]) >= 8 and int(n[3:5]) < 8]
    assert set(east) <= set(assignments["B"])


def test_partition_drops_substation_with_small_cell():
    G = make_combined_grid()
    # a 10-node dead-end spur hanging off the corner; substation C at its far end
    prev = grid_node_id(0, 0)
    for k in range(1, 11):
        node = 9_000_000 + k
        x, y = grid_node_coord(0, 0)
        G.add_node(node, cord=(x - 0.001 * k, y), label="R", load=0.0)
        G.add_edge(prev, node, length=90.0, highway="residential")
        prev = node
    substation_nodes = {"A": grid_node_id(2, 2), "B": grid_node_id(13, 13), "C": prev}

    assignments = NetworkPartitioner(G).partition_transformers(substation_nodes)
    assert set(assignments) == {"A", "B"}
    # the spur is re-assigned to A, the nearest remaining substation
    assert prev in assignments["A"]
    assert set(assignments["A"]) | set(assignments["B"]) == set(G.nodes)


def test_partition_raises_when_all_cells_are_small():
    G = make_combined_grid(size=4)
    with pytest.raises(ValueError, match="No valid centers"):
        NetworkPartitioner(G).partition_transformers({"A": grid_node_id(0, 0)})


# save_partitioning / load_partitioning

@pytest.fixture
def saved_partition(tmp_path):
    G = make_combined_grid()
    subs = [substation_at("101", 1, 1), substation_at("202", 13, 10), substation_at("303", 7, 7)]
    partitioner = NetworkPartitioner(G)
    substation_nodes = partitioner.find_nearest_road_nodes(subs[:2], search_radius=0.001)
    assignments = partitioner.partition_transformers(substation_nodes)
    output = tmp_path / "interim" / "assignment.json"  # parent created by save_partitioning
    partitioner.save_partitioning(assignments, subs, output_file=str(output))
    return G, subs, assignments, output


def test_save_partitioning_writes_induced_subgraphs(saved_partition):
    G, subs, assignments, output = saved_partition
    data = json.loads(output.read_text())

    assert set(data) == {"101", "202"}, "substations without an assignment are left out"
    for sub in subs[:2]:
        part = data[sub.id]
        nodes = assignments[sub.id]
        assert set(part["nodelist"]) == {str(n) for n in nodes}
        assert len(part["edgelist"]) == G.subgraph(nodes).number_of_edges()
        for n in nodes:
            attrs = part["nodelist"][str(n)]
            assert attrs["label"] == G.nodes[n]["label"]
            assert attrs["load"] == G.nodes[n]["load"]
            if attrs["label"] == "R":
                assert attrs["distance"] == pytest.approx(geodesic_m(sub.cord, G.nodes[n]["cord"]))
            else:
                assert attrs["distance"] == 999999999


def test_load_partitioning_round_trip(saved_partition):
    _, _, assignments, output = saved_partition
    loaded = load_partitioning(str(output))
    assert set(loaded) == {101, 202}
    transformers = {n for nodes in assignments.values() for n in nodes if str(n).startswith("T")}
    loaded_transformers = {
        n for part in loaded.values() for n, attrs in part["nodelist"].items() if attrs["label"] == "T"
    }
    assert loaded_transformers == transformers


def test_load_partitioning_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_partitioning(str(tmp_path / "missing.json"))

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_partitioning(str(bad_json))

    bad_id = tmp_path / "bad_id.json"
    bad_id.write_text(json.dumps({"sub-A": {"nodelist": {}, "edgelist": {}}}))
    with pytest.raises(ValueError, match="Invalid substation ID"):
        load_partitioning(str(bad_id))



# save_substation_partitions / load_substation_partition

def test_substation_partitions_round_trip(saved_partition, tmp_path):
    G, subs, assignments, output = saved_partition
    data = json.loads(output.read_text())
    partition_dir = tmp_path / "group_partition"
    saved = save_substation_partitions(data, subs, str(partition_dir))

    assert saved == ["101", "202"]
    assert read_substation_list(str(partition_dir)) == saved
    for sub in subs[:2]:
        loaded_sub, part = load_substation_partition(str(partition_dir), sub.id)
        assert loaded_sub == sub
        assert part == data[sub.id], "same partition data as the combined file"


def test_substation_partition_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="Substation list"):
        read_substation_list(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="substation 101"):
        load_substation_partition(str(tmp_path), "101")
