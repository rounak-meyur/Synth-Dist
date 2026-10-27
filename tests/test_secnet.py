"""
Tests for the secondary network generation: the heuristic in models/secnet_heuristic.py,
its agreement with the MILP in models/secnet.py, and the method switch in utils/secnet_utils.py.
"""
import logging
import math
from unittest.mock import MagicMock

import cvxpy as cp
import networkx as nx
import pytest
from omegaconf import OmegaConf
from shapely.geometry import LineString

from conftest import REPO_ROOT
from models.secnet import (
    create_candidate_network,
    create_secondary_distribution_network,
    interpolate_points,
    minimum_transformers,
)
from models.secnet_heuristic import create_secondary_distribution_network_heuristic
from utils import secnet_utils
from utils.dataloader import Home
from utils.secnet_utils import SecondaryNetworkGenerator

ROAD_LINK = (101, 102, 0)
# East-west street of ~350 m
ROAD_GEOM = LineString([(-80.4200, 37.1900), (-80.4160, 37.1900)])


def street_homes(n_per_side: int, load: float = 1000.0, offset: float = 0.0002):
    """Homes evenly spaced along both sides of ROAD_GEOM (~22 m off the road)."""
    (x0, y0), (x1, _) = ROAD_GEOM.coords
    homes = []
    for side, dy in ((0, offset), (1, -offset)):
        for k in range(n_per_side):
            x = x0 + (x1 - x0) * (k + 0.5) / n_per_side + side * 0.00007  # stagger the two sides
            profile = [load + k] * 24  # slightly different loads
            homes.append(Home(id=7000 + side * 100 + k, cord=(x, y0 + dy), profile=profile,
                              peak=max(profile), load=sum(profile) / 24))
    return homes


def candidate(homes, **kwargs):
    graph, _ = create_candidate_network(ROAD_LINK, ROAD_GEOM, homes, **kwargs)
    return graph


def network_cost(network: nx.Graph, penalty: float) -> float:
    return sum(d['length'] + penalty * d['crossing'] for _, _, d in network.edges(data=True))


def assert_valid_secnet(result: nx.Graph, graph: nx.Graph, max_rating: float, max_hops: int):
    """Check every constraint of the secondary network problem."""
    homes = {n for n, d in graph.nodes(data=True) if d['label'] == 'H'}
    assert {n for n, d in result.nodes(data=True) if d['label'] == 'H'} == homes, "every home must be connected"
    assert all(graph.has_edge(u, v) for u, v in result.edges()), "only candidate edges may be used"
    assert nx.is_forest(result)

    for component in nx.connected_components(result):
        roots = [n for n in component if result.nodes[n]['label'] == 'T']
        assert len(roots) == 1, f"each tree needs exactly one transformer root, got {roots}"
        depth = nx.single_source_shortest_path_length(result, roots[0])
        assert max(depth.values()) <= max_hops
        load = sum(result.nodes[n]['load'] for n in component if result.nodes[n]['label'] == 'H')
        assert load <= max_rating + 1e-6

    assert all(result.degree(h) <= 2 for h in homes)


@pytest.mark.parametrize("n_per_side", [1, 3, 8])  # 8 per side uses Delaunay home edges
def test_heuristic_satisfies_constraints(n_per_side):
    graph = candidate(street_homes(n_per_side))
    result = create_secondary_distribution_network_heuristic(graph, penalty=0.5, max_rating=25e3, max_hops=10)
    assert_valid_secnet(result, graph, max_rating=25e3, max_hops=10)


def test_heuristic_result_has_candidate_attributes():
    graph = candidate(street_homes(3))
    result = create_secondary_distribution_network_heuristic(graph)
    for n, d in result.nodes(data=True):
        assert d == graph.nodes[n]
    for u, v, d in result.edges(data=True):
        assert d == graph[u][v]


def test_heuristic_max_hops_one_connects_homes_directly():
    graph = candidate(street_homes(4))
    result = create_secondary_distribution_network_heuristic(graph, max_hops=1)
    assert_valid_secnet(result, graph, max_rating=25e3, max_hops=1)
    for h, d in result.nodes(data=True):
        if d['label'] == 'H':
            assert all(result.nodes[n]['label'] == 'T' for n in result.neighbors(h))


def test_heuristic_respects_transformer_rating():
    homes = street_homes(5, load=1000.0)  # ~10 kW in total
    graph = candidate(homes)
    max_rating = 2500.0
    result = create_secondary_distribution_network_heuristic(graph, max_rating=max_rating)

    assert_valid_secnet(result, graph, max_rating=max_rating, max_hops=10)
    n_transformers = sum(1 for _, d in result.nodes(data=True) if d['label'] == 'T')
    assert n_transformers >= math.ceil(sum(h.load for h in homes) / max_rating)


def test_heuristic_avoids_road_crossings_with_high_penalty():
    graph = candidate(street_homes(5))
    result = create_secondary_distribution_network_heuristic(graph, penalty=1e4)
    assert_valid_secnet(result, graph, max_rating=25e3, max_hops=10)
    # crossing == 2: home-to-home edge across the road; T-home edges always count 1
    assert all(d['crossing'] < 2 for _, _, d in result.edges(data=True))


def test_local_search_does_not_increase_cost():
    graph = candidate(street_homes(8))
    greedy = create_secondary_distribution_network_heuristic(graph, local_search=False)
    improved = create_secondary_distribution_network_heuristic(graph, local_search=True)
    assert_valid_secnet(greedy, graph, max_rating=25e3, max_hops=10)
    assert network_cost(improved, 0.5) <= network_cost(greedy, 0.5) + 1e-9


def test_heuristic_overloaded_home_still_connected(caplog):
    homes = street_homes(2, load=1000.0)
    homes[0].load = 50_000.0  # exceeds max_rating on its own
    graph = candidate(homes)
    with caplog.at_level(logging.WARNING, logger="secnet_heuristic"):
        result = create_secondary_distribution_network_heuristic(graph, max_rating=25e3)
    assert {n for n, d in result.nodes(data=True) if d['label'] == 'H'} == {h.id for h in homes}
    assert "exceeds max_rating" in caplog.text


def test_heuristic_empty_candidate():
    assert create_secondary_distribution_network_heuristic(nx.Graph()).number_of_edges() == 0


# Comparison with the MILP

@pytest.mark.skipif("SCIP" not in cp.installed_solvers(), reason="SCIP solver not installed")
@pytest.mark.parametrize("n_per_side,max_hops", [(2, 10), (3, 10), (3, 2), (6, 3)])
def test_heuristic_close_to_milp(n_per_side, max_hops):
    # max_rating is non-binding: the MILP limits flow per edge, the heuristic per transformer
    graph = candidate(street_homes(n_per_side))
    params = dict(penalty=0.5, max_rating=1e6, max_hops=max_hops)

    milp = create_secondary_distribution_network(graph, solver="scip", verbose=False, relative_gap=0.0, **params)
    heur = create_secondary_distribution_network_heuristic(graph, **params)

    assert_valid_secnet(milp, graph, max_rating=1e6, max_hops=max_hops)
    assert_valid_secnet(heur, graph, max_rating=1e6, max_hops=max_hops)
    milp_cost, heur_cost = network_cost(milp, 0.5), network_cost(heur, 0.5)
    assert heur_cost >= milp_cost - 1e-3, "the heuristic cannot beat the optimum"
    assert heur_cost <= 1.05 * milp_cost


# Method selection in SecondaryNetworkGenerator

HEURISTIC_CONFIG = REPO_ROOT / "configs/config_heuristic.yaml"
MILP_CONFIG = REPO_ROOT / "configs/config_milp.yaml"


@pytest.mark.parametrize("path,method", [(HEURISTIC_CONFIG, "heuristic"), (MILP_CONFIG, "milp")])
def test_config_declares_method(path, method):
    assert OmegaConf.load(path).secnet.secnet_args.method == method


def test_configs_differ_only_in_methods():
    heuristic, milp = OmegaConf.load(HEURISTIC_CONFIG), OmegaConf.load(MILP_CONFIG)
    for conf in (heuristic, milp):
        for args in (conf.secnet.secnet_args, conf.primnet.primnet_args):
            del args.method
            del args.parameters
    assert heuristic == milp


def test_main_defaults_to_heuristic_config():
    assert 'default = "configs/config_heuristic.yaml"' in (REPO_ROOT / "main.py").read_text()


def test_generator_uses_heuristic(tmp_path, monkeypatch):
    milp = MagicMock(side_effect=AssertionError("MILP must not be called"))
    monkeypatch.setattr(secnet_utils, "create_secondary_distribution_network", milp)
    secnet_args = OmegaConf.load(HEURISTIC_CONFIG).secnet.secnet_args

    homes = street_homes(3)
    generator = SecondaryNetworkGenerator(output_dir=str(tmp_path), base_transformer_id=5199900000)
    result = generator.generate_network_for_link(ROAD_LINK, ROAD_GEOM, homes, **secnet_args)

    assert_valid_secnet(result, candidate(homes), max_rating=secnet_args.max_rating, max_hops=secnet_args.max_hops)
    assert generator.all_transformers
    assert sum(t.load for t in generator.all_transformers.values()) == pytest.approx(sum(h.load for h in homes))

    generator.save_results(prefix="test", road_link_id=ROAD_LINK)
    assert (tmp_path / "test_secondary_edges.csv").exists()
    assert (tmp_path / "test_transformers.csv").exists()


def test_generator_passes_milp_parameters(tmp_path, monkeypatch):
    graph = candidate(street_homes(1))
    milp = MagicMock(return_value=create_secondary_distribution_network_heuristic(graph))
    monkeypatch.setattr(secnet_utils, "create_secondary_distribution_network", milp)
    secnet_args = OmegaConf.load(MILP_CONFIG).secnet.secnet_args

    generator = SecondaryNetworkGenerator(output_dir=str(tmp_path), base_transformer_id=1)
    generator.generate_network_for_link(ROAD_LINK, ROAD_GEOM, street_homes(1), **secnet_args)

    kwargs = milp.call_args.kwargs
    for key, value in secnet_args.parameters.items():
        assert kwargs[key] == value
    assert kwargs["max_rating"] == secnet_args.max_rating


def test_generator_defaults_to_milp(tmp_path, monkeypatch):
    graph = candidate(street_homes(1))
    milp = MagicMock(return_value=create_secondary_distribution_network_heuristic(graph))
    monkeypatch.setattr(secnet_utils, "create_secondary_distribution_network", milp)

    generator = SecondaryNetworkGenerator(output_dir=str(tmp_path), base_transformer_id=1)
    generator.generate_network_for_link(ROAD_LINK, ROAD_GEOM, street_homes(1))
    milp.assert_called_once()


def test_generator_rejects_unknown_method(tmp_path):
    generator = SecondaryNetworkGenerator(output_dir=str(tmp_path), base_transformer_id=1)
    with pytest.raises(ValueError, match="Unknown secondary network method"):
        generator.generate_network_for_link(ROAD_LINK, ROAD_GEOM, street_homes(1), method="annealing")


# Candidate transformers on short, crowded road links

SHORT_GEOM = LineString([(-80.4200, 37.1900), (-80.41966, 37.1900)])  # ~30 m


def crowded_homes(n=24, load=4000.0):
    """n homes along both sides of the ~30 m SHORT_GEOM, e.g. an apartment block."""
    (x0, y0), (x1, _) = SHORT_GEOM.coords
    homes = []
    for k in range(n):
        x = x0 + (x1 - x0) * (k // 2 + 0.5) / (n // 2)
        y = y0 + (0.0002 if k % 2 else -0.0002) + 0.00001 * (k // 2)
        profile = [load] * 24
        homes.append(Home(id=8000 + k, cord=(x, y), profile=profile, peak=load, load=load))
    return homes


@pytest.mark.parametrize("loads,rating,expected", [
    ([], 25e3, 1),
    ([1000] * 5, 25e3, 1),
    ([4000] * 24, 25e3, 4),          # 6 homes per transformer
    ([15e3, 15e3, 15e3], 25e3, 3),   # ceil(45/25) = 2 would not fit
    ([20e3, 5e3, 20e3, 5e3], 25e3, 2),
    ([30e3, 1000], 25e3, 2),         # an oversized home gets a transformer of its own
])
def test_minimum_transformers(loads, rating, expected):
    assert minimum_transformers(loads, rating) == expected


def test_interpolate_points_regular_spacing_unchanged():
    # ~350 m at 50 m separation: points at 0, 50, ..., 350 (geodesic) along the road
    points = interpolate_points(ROAD_GEOM, 50.0, min_points=3)
    assert len(points) == 8
    assert points[0] == ROAD_GEOM.coords[0]


def test_interpolate_points_short_link_gets_min_points():
    assert len(interpolate_points(SHORT_GEOM, 50.0)) == 1
    points = interpolate_points(SHORT_GEOM, 50.0, min_points=4)
    assert len(points) == 4
    xs = [p[0] for p in points]
    assert xs == sorted(xs), "ordered along the road"
    (x0, _), (x1, _) = SHORT_GEOM.coords
    assert xs == pytest.approx([x0 + (x1 - x0) * (k + 0.5) / 4 for k in range(4)])


def test_interpolate_points_zero_length_link():
    point = (-80.42, 37.19)
    assert interpolate_points(LineString([point, point]), 50.0, min_points=2) == [point, point]


def test_crowded_short_link_respects_rating(tmp_path, caplog):
    homes = crowded_homes()
    secnet_args = OmegaConf.load(HEURISTIC_CONFIG).secnet.secnet_args
    generator = SecondaryNetworkGenerator(output_dir=str(tmp_path), base_transformer_id=1)
    with caplog.at_level(logging.WARNING, logger="secnet_heuristic"):
        result = generator.generate_network_for_link(ROAD_LINK, SHORT_GEOM, homes, **secnet_args)

    # 4 transformers needed, plus one spare candidate
    graph, _ = create_candidate_network(ROAD_LINK, SHORT_GEOM, homes, min_transformers=5)
    assert_valid_secnet(result, graph, max_rating=secnet_args.max_rating, max_hops=secnet_args.max_hops)
    assert "exceeds max_rating" not in caplog.text
    assert all(t.load <= secnet_args.max_rating for t in generator.all_transformers.values())
    assert len(generator.all_transformers) >= 4


@pytest.mark.parametrize("n_homes,load,expected", [(5, 1000.0, 1), (24, 4000.0, 5), (4, 10000.0, 3)])
def test_generator_requests_enough_candidate_transformers(tmp_path, monkeypatch, n_homes, load, expected):
    """minimum_transformers plus one spare when more than one transformer is needed."""
    requested = []
    create = secnet_utils.create_candidate_network

    def spy(*args, **kwargs):
        requested.append(kwargs["min_transformers"])
        return create(*args, **kwargs)

    monkeypatch.setattr(secnet_utils, "create_candidate_network", spy)
    secnet_args = OmegaConf.load(HEURISTIC_CONFIG).secnet.secnet_args
    generator = SecondaryNetworkGenerator(output_dir=str(tmp_path), base_transformer_id=1)
    generator.generate_network_for_link(ROAD_LINK, SHORT_GEOM, crowded_homes(n_homes, load), **secnet_args)
    assert requested == [expected]
