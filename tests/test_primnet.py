"""
Tests for the primary network generation: the heuristic in models/primnet_heuristic.py,
its agreement with the MILP in models/primnet.py, and the method switch in utils/primnet_utils.py.
"""
import logging
from unittest.mock import MagicMock

import cvxpy as cp
import networkx as nx
import pytest
from omegaconf import OmegaConf

from conftest import REPO_ROOT, geodesic_m, grid_node_coord, grid_node_id, make_combined_grid
from models.primnet import optimize_primary_network
from models.primnet_heuristic import create_primary_network_heuristic
from utils import primnet_utils
from utils.dataloader import Substation
from utils.partition_utils import NetworkPartitioner, load_partitioning
from utils.primnet_utils import PRIMNET_METHODS, PrimaryNetworkGenerator, _milp_config

HEURISTIC_CONFIG = REPO_ROOT / "configs/config_heuristic.yaml"
MILP_CONFIG = REPO_ROOT / "configs/config_milp.yaml"


def primnet_args(path=HEURISTIC_CONFIG, **overrides):
    args = OmegaConf.load(path).primnet.primnet_args
    for key, value in overrides.items():
        args[key] = value
    return args


def partition_graph(size=4, load=1000.0, sub=(0, 0)):
    """Combined grid as one substation partition: 'distance' (m) of road nodes to the substation."""
    G = make_combined_grid(size)
    sub_cord = grid_node_coord(*sub)
    for n, d in G.nodes(data=True):
        if d["label"] == "T":
            d["load"] = load
        d["distance"] = geodesic_m(sub_cord, d["cord"]) if d["label"] == "R" else 999999999
    return G


def rooted(tree, root):
    order, parent = [root], {root: None}
    for u, v in nx.bfs_edges(tree, root):
        parent[v] = u
        order.append(v)
    sub = {n: 1e-3 * tree.nodes[n]["load"] for n in order}
    for n in reversed(order[1:]):
        sub[parent[n]] += sub[n]
    return order, parent, sub


def assert_valid_primnet(result, feeders, graph, config, feeder_voltage=1.0):
    """Check every constraint of the primary network problem."""
    transformers = {n for n, d in graph.nodes(data=True) if d["label"] == "T"}
    assert transformers <= set(result.nodes), "every transformer must be covered"
    assert all(graph.has_edge(u, v) for u, v in result.edges()), "only candidate edges may be used"
    assert all(d["label"] == "primary" for _, _, d in result.edges(data=True))
    assert nx.is_forest(result)
    assert len(set(feeders)) == len(feeders)
    assert all(graph.nodes[f]["label"] == "R" for f in feeders), "feeders are road nodes"

    r_per_km = config["conductor_resistance_per_km"] / config["base_impedance"]
    for component in nx.connected_components(result):
        roots = [f for f in feeders if f in component]
        assert len(roots) == 1, f"each tree needs exactly one feeder, got {roots}"
        root = roots[0]
        for n in component:
            if graph.nodes[n]["label"] == "R" and n != root:
                assert result.degree(n) >= 2, f"road node {n} must not be a leaf"
        order, parent, sub = rooted(result.subgraph(component), root)
        assert sub[root] <= config["maximum_branch_flow"] + 1e-6
        voltage = {root: feeder_voltage}
        for n in order[1:]:
            voltage[n] = voltage[parent[n]] - r_per_km * 1e-3 * result[parent[n]][n]["length"] * sub[n]
        assert min(voltage.values()) >= config["voltage_min"] - 1e-9


def network_cost(graph, result, feeders):
    lines = 1e-3 * sum(d["length"] for _, _, d in result.edges(data=True))
    return lines + 1e-3 * sum(graph.nodes[f]["distance"] for f in feeders)


# Heuristic

def test_heuristic_single_feeder_when_limits_allow():
    graph = partition_graph()
    config = primnet_args()
    result, feeders = create_primary_network_heuristic(graph, config)
    assert_valid_primnet(result, feeders, graph, config)
    assert feeders == [grid_node_id(0, 0)], "the feeder is the road node closest to the substation"


def test_heuristic_splits_feeders_on_branch_flow():
    graph = partition_graph(load=50_000.0)  # 12 transformers x 50 kVA = 600 kVA > 400 kVA
    config = primnet_args()
    result, feeders = create_primary_network_heuristic(graph, config)
    assert_valid_primnet(result, feeders, graph, config)
    assert len(feeders) >= 2


def test_heuristic_splits_feeders_on_voltage():
    graph = partition_graph(size=6, load=5000.0)
    config = primnet_args(conductor_resistance_per_km=2000.0)  # exaggerated so voltage binds
    result, feeders = create_primary_network_heuristic(graph, config)
    assert_valid_primnet(result, feeders, graph, config)
    assert len(feeders) >= 2
    one_feeder = primnet_args()
    _, single = create_primary_network_heuristic(graph, one_feeder)
    assert len(single) == 1, "without the voltage limit a single feeder suffices"


def test_heuristic_no_road_leaves():
    graph = partition_graph(size=6)
    result, feeders = create_primary_network_heuristic(graph, primnet_args())
    for n in result:
        if graph.nodes[n]["label"] == "R" and n not in feeders:
            assert result.degree(n) >= 2


def test_heuristic_single_transformer_reaches_road_node():
    graph = nx.MultiGraph()
    graph.add_node(1, label="R", load=0.0, cord=(0, 0), distance=10.0)
    graph.add_node(2, label="R", load=0.0, cord=(0, 1), distance=20.0)
    graph.add_node("T1", label="T", load=1000.0, cord=(0, 0.5), distance=999999999)
    graph.add_edge(1, "T1", length=50.0, geometry=None)
    graph.add_edge("T1", 2, length=50.0, geometry=None)
    result, feeders = create_primary_network_heuristic(graph, primnet_args())
    assert feeders == [1]
    assert set(result.edges()) == {(1, "T1")}


def test_heuristic_infeasible_transformer_still_covered(caplog):
    graph = partition_graph(size=3)
    big = next(n for n, d in graph.nodes(data=True) if d["label"] == "T")
    graph.nodes[big]["load"] = 500_000.0  # 500 kVA on its own
    with caplog.at_level(logging.WARNING, logger="primnet_heuristic"):
        result, feeders = create_primary_network_heuristic(graph, primnet_args())
    assert {n for n, d in graph.nodes(data=True) if d["label"] == "T"} <= set(result.nodes)
    assert "violates the flow or voltage limits" in caplog.text


def test_heuristic_ignores_road_only_components():
    graph = partition_graph(size=3)
    graph.add_node(99, label="R", load=0.0, cord=(1, 1), distance=1.0)
    graph.add_node(98, label="R", load=0.0, cord=(1, 2), distance=2.0)
    graph.add_edge(98, 99, length=10.0, geometry=None)
    result, feeders = create_primary_network_heuristic(graph, primnet_args())
    assert 98 not in result and 99 not in result


# Comparison with the MILP

@pytest.mark.skipif("SCIP" not in cp.installed_solvers(), reason="SCIP solver not installed")
@pytest.mark.parametrize("size,load", [(3, 1000.0), (4, 1000.0), (4, 50_000.0)])
def test_heuristic_close_to_milp(size, load):
    graph = partition_graph(size=size, load=load)
    args = primnet_args(MILP_CONFIG)
    milp_config = _milp_config(args, {**args.parameters, "verbose": False, "relative_gap": 0.0})
    milp, milp_feeders = optimize_primary_network(graph, "scip", milp_config)
    heur, heur_feeders = create_primary_network_heuristic(graph, primnet_args())

    # the MILP may raise the feeder voltage up to voltage_max
    assert_valid_primnet(milp, milp_feeders, graph, milp_config, feeder_voltage=milp_config.voltage_max)
    assert_valid_primnet(heur, heur_feeders, graph, primnet_args())
    milp_cost, heur_cost = network_cost(graph, milp, milp_feeders), network_cost(graph, heur, heur_feeders)
    assert heur_cost >= milp_cost - 1e-6, "the heuristic cannot beat the optimum"
    assert heur_cost <= 1.10 * milp_cost


# PrimaryNetworkGenerator

@pytest.fixture
def assignment(tmp_path):
    """Partition data of one substation as saved by save_partitioning and read by load_partitioning."""
    G = make_combined_grid(11)
    x, y = grid_node_coord(5, 5)
    sub = Substation(id="4242", cord=(x + 0.00005, y + 0.00005))
    partitioner = NetworkPartitioner(G)
    assignments = partitioner.partition_transformers(partitioner.find_nearest_road_nodes([sub], search_radius=0.001))
    partitioner.save_partitioning(assignments, [sub], output_file=str(tmp_path / "assignment.json"))
    return sub, load_partitioning(str(tmp_path / "assignment.json"))[4242], G


def test_generator_heuristic_exports_valid_network(tmp_path, assignment, monkeypatch):
    milp = MagicMock(side_effect=AssertionError("MILP must not be called"))
    monkeypatch.setattr(primnet_utils, "optimize_primary_network", milp)
    sub, part, G = assignment

    generator = PrimaryNetworkGenerator(output_dir=str(tmp_path / "primnet"))
    generator.generate_network_for_substation(sub, part, config=primnet_args())
    generator.export_to_csv(prefix="777_4242")  # asserts a single tree covering all transformers

    network = generator.optimal_graph
    assert network.nodes[sub.id]["label"] == "S"
    feeders = [v if u == sub.id else u for u, v, d in network.edges(data=True) if d["label"] == "feeder"]
    assert feeders and all(network.nodes[f]["label"] == "R" for f in feeders)
    assert (tmp_path / "primnet/777_4242_nodes.csv").exists()
    assert (tmp_path / "primnet/777_4242_edges.csv").exists()


def test_generator_milp_gets_merged_parameters(tmp_path, assignment, monkeypatch):
    sub, part, _ = assignment
    seen = {}

    def fake_milp(graph, solver, config):
        seen.update(solver=solver, config=config)
        return create_primary_network_heuristic(graph, config)

    monkeypatch.setattr(primnet_utils, "optimize_primary_network", fake_milp)
    generator = PrimaryNetworkGenerator(output_dir=str(tmp_path))
    generator.generate_network_for_substation(sub, part, config=primnet_args(MILP_CONFIG))

    assert seen["solver"] == "scip"
    assert seen["config"].relative_gap == 0.05 and seen["config"].verbose is True
    assert seen["config"].maximum_branch_flow == 400
    assert "parameters" not in seen["config"] and "method" not in seen["config"]


def test_generator_rejects_unknown_method(tmp_path, assignment):
    sub, part, _ = assignment
    generator = PrimaryNetworkGenerator(output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Unknown primary network method"):
        generator.generate_network_for_substation(sub, part, config=primnet_args(method="annealing"))


@pytest.mark.parametrize("path,method", [(HEURISTIC_CONFIG, "heuristic"), (MILP_CONFIG, "milp")])
def test_config_declares_primnet_method(path, method):
    assert primnet_args(path).method == method
    assert primnet_args(path).method in PRIMNET_METHODS
