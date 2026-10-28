"""
Tests for combining the transformer nodes of the secondary network with the road network
(SecondaryNetworkGenerator.combine_networks and helpers in utils/secnet_utils.py).
"""
import networkx as nx
import pandas as pd
import pytest
from geopy.distance import geodesic
from shapely.geometry import LineString

from conftest import grid_node_coord, make_road_grid, make_profile
from utils.dataloader import Home
from utils.mapping import compute_edge_to_homes_map, map_homes_to_edges
from utils.osm_utils import load_road_network_from_files, save_road_network
from utils.secnet_utils import (
    SecondaryNetworkGenerator,
    _get_node_coord,
    get_road_link_geometry,
    load_combined_network,
)

PREFIX = "777"
# A low rating forces one transformer per home, so each link carries several transformers
SECNET_ARGS = dict(
    separation=50.0, penalty=0.5, max_rating=1500, max_hops=10,
    method="heuristic", parameters={"max_iterations": 1000, "local_search": True},
)


def geodesic_m(a, b):
    return geodesic((a[1], a[0]), (b[1], b[0])).meters


def road_network_as_in_main(tmp_path, grid: nx.MultiDiGraph) -> nx.MultiGraph:
    """Save and reload the OSM-like grid the way main.py does (undirected MultiGraph)."""
    for u, v, d in grid.edges(data=True):
        d.setdefault("geometry", LineString([(grid.nodes[n]["x"], grid.nodes[n]["y"]) for n in (u, v)]))
    edges, nodes = tmp_path / "road_edges.csv", tmp_path / "road_nodes.csv"
    save_road_network(grid, str(edges), str(nodes))
    return load_road_network_from_files(str(edges), str(nodes))


@pytest.fixture
def homes():
    """Three homes along each of a few grid edges (~10 m off the road)."""
    homes = []
    for e, ((i1, j1), (i2, j2)) in enumerate([((0, 1), (1, 1)), ((1, 1), (1, 2)), ((2, 2), (3, 2))]):
        p, q = grid_node_coord(i1, j1), grid_node_coord(i2, j2)
        for f in (0.2, 0.5, 0.8):
            x, y = p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1])
            cord = (x, y + 0.0001) if j1 == j2 else (x + 0.0001, y)
            profile = make_profile(1000 / 12.5)  # average load 1000
            homes.append(Home(id=5200000000 + 10 * e + int(f * 10), cord=cord, profile=profile,
                              peak=max(profile), load=sum(profile) / 24))
    return homes


@pytest.fixture
def combined(tmp_path, homes):
    """Run secondary network generation for every mapped road link and combine with the roads."""
    roads = road_network_as_in_main(tmp_path, make_road_grid(4))
    r2h = compute_edge_to_homes_map(map_homes_to_edges(roads, homes, padding_distance=0.001))
    out_dir = tmp_path / "secnet"
    generator = SecondaryNetworkGenerator(output_dir=str(out_dir), base_transformer_id=5177700000)
    for road_link, mapped in r2h.items():
        geometry = get_road_link_geometry(roads, road_link)
        generator.generate_network_for_link(road_link, geometry, mapped, **SECNET_ARGS)
        generator.save_results(prefix=PREFIX, road_link_id=road_link)
    network = generator.combine_networks(road_network=roads, prefix=PREFIX)
    return dict(roads=roads, r2h=r2h, network=network, out_dir=out_dir, homes=homes)


def road_transformer_sequences(out_dir):
    """Parse <prefix>_road_transformer_edges.txt into {road_link: node sequence}."""
    sequences = {}
    for line in (out_dir / f"{PREFIX}_road_transformer_edges.txt").read_text().splitlines():
        u, v, k, *nodes = line.split()
        sequences[(int(u), int(v), int(k))] = [n if n.startswith("T") else int(n) for n in nodes]
    return sequences


# get_road_link_geometry

def test_road_geometry_is_reversed_on_reload(tmp_path):
    """Why orientation matters: the undirected reload keeps the v -> u geometry for two-way roads."""
    roads = road_network_as_in_main(tmp_path, make_road_grid(4))
    starts_at_v = [
        (u, v, k) for u, v, k, d in roads.edges(keys=True, data=True)
        if d["geometry"].coords[0] == (roads.nodes[v]["x"], roads.nodes[v]["y"])
    ]
    assert starts_at_v


def test_get_road_link_geometry_starts_at_first_node(tmp_path):
    roads = road_network_as_in_main(tmp_path, make_road_grid(4))
    for u, v, k, d in roads.edges(keys=True, data=True):
        for a, b in ((u, v), (v, u)):
            geometry = get_road_link_geometry(roads, (a, b, k))
            assert geometry.coords[0] == (roads.nodes[a]["x"], roads.nodes[a]["y"])
            assert geometry.coords[-1] == (roads.nodes[b]["x"], roads.nodes[b]["y"])
            assert geometry.equals(d["geometry"])  # same line, possibly reversed


def test_get_road_link_geometry_missing(tmp_path):
    roads = road_network_as_in_main(tmp_path, make_road_grid(4))
    with pytest.raises(KeyError):
        get_road_link_geometry(roads, (1, 2, 0))
    u, v, k = next(iter(roads.edges(keys=True)))
    roads.edges[u, v, k]["geometry"] = None
    with pytest.raises(KeyError):
        get_road_link_geometry(roads, (u, v, k))


# combine_networks

def test_transformers_replace_mapped_road_links(combined):
    network, r2h = combined["network"], combined["r2h"]
    sequences = road_transformer_sequences(combined["out_dir"])
    assert set(sequences) == set(r2h)

    for (u, v, k), nodes in sequences.items():
        assert not network.has_edge(u, v, key=k), "original road edge must be replaced"
        assert nodes[0] == u and nodes[-1] == v
        assert len(nodes) > 2 and all(isinstance(n, str) and n.startswith("T") for n in nodes[1:-1])
        for a, b in zip(nodes[:-1], nodes[1:]):
            assert network.has_edge(a, b)


def test_transformer_chain_follows_road_geometry(combined):
    """u -> T... -> v must run along the road: transformers ordered by distance from u, same total length."""
    network, roads = combined["network"], combined["roads"]
    for road_link, nodes in road_transformer_sequences(combined["out_dir"]).items():
        coords = list(roads.edges[road_link]["geometry"].coords)
        road_length = sum(geodesic_m(a, b) for a, b in zip(coords[:-1], coords[1:]))
        cords = [network.nodes[n]["cord"] for n in nodes]
        chain_length = sum(geodesic_m(a, b) for a, b in zip(cords[:-1], cords[1:]))
        from_u = [geodesic_m(cords[0], c) for c in cords]
        assert from_u == sorted(from_u), f"transformers out of order on {road_link}"
        assert chain_length == pytest.approx(road_length, rel=0.01)


def test_transformer_nodes_carry_csv_attributes(combined):
    network, out_dir = combined["network"], combined["out_dir"]
    df = pd.read_csv(out_dir / f"{PREFIX}_transformers.csv")
    assert len(df) > 0
    for _, row in df.iterrows():
        attrs = network.nodes[row["transformer_id"]]
        assert attrs["label"] == "T"
        assert attrs["cord"] == pytest.approx((row["longitude"], row["latitude"]))
        assert attrs["load"] == pytest.approx(row["total_load"])
    t_load = sum(d["load"] for _, d in network.nodes(data=True) if d["label"] == "T")
    assert t_load == pytest.approx(sum(h.load for h in combined["homes"]))


def test_road_nodes_are_standardised(combined):
    network, roads = combined["network"], combined["roads"]
    for n, d in roads.nodes(data=True):
        attrs = network.nodes[n]
        assert attrs["label"] == "R"
        assert attrs["load"] == 0.0
        assert attrs["cord"] == (d["x"], d["y"])


def test_unmapped_road_links_are_kept(combined):
    network, roads, r2h = combined["network"], combined["roads"], combined["r2h"]
    for u, v, k, d in roads.edges(keys=True, data=True):
        if (u, v, k) not in r2h:
            assert network.has_edge(u, v, key=k)
            assert network.edges[u, v, k]["geometry"].equals(d["geometry"])


def test_new_edges_have_geometry_length_and_highway(combined):
    network = combined["network"]
    for u, v, k, d in network.edges(keys=True, data=True):
        if network.nodes[u]["label"] == "T" or network.nodes[v]["label"] == "T":
            assert d["geometry"].equals(LineString([network.nodes[u]["cord"], network.nodes[v]["cord"]]))
            assert d["length"] == pytest.approx(geodesic_m(network.nodes[u]["cord"], network.nodes[v]["cord"]))
            assert d["highway"] == "residential"


def test_combined_network_is_connected(combined):
    assert nx.is_connected(combined["network"])


# save_combined_network / load_combined_network

def test_combined_network_round_trip(combined):
    network, out_dir = combined["network"], combined["out_dir"]
    loaded = load_combined_network(
        nodes_file=str(out_dir / f"{PREFIX}_combined_network_nodes.csv"),
        edges_file=str(out_dir / f"{PREFIX}_combined_network_edges.csv"),
    )
    assert set(loaded.nodes) == set(network.nodes)
    assert loaded.number_of_edges() == network.number_of_edges()
    for n, d in loaded.nodes(data=True):
        assert isinstance(n, str) == (d["label"] == "T"), "road ids are int, transformer ids str"
        assert d["label"] == network.nodes[n]["label"]
        assert d["cord"] == pytest.approx(network.nodes[n]["cord"])
        assert d["load"] == pytest.approx(network.nodes[n]["load"])
    for u, v, d in loaded.edges(data=True):
        assert isinstance(d["geometry"], LineString)
        assert d["length"] >= 0  # the first candidate transformer sits on the road node


def test_load_combined_network_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_combined_network(str(tmp_path / "nodes.csv"), str(tmp_path / "edges.csv"))


# connect_network_components

def _labelled_path(graph, ids, y, label_last=None):
    """Add a straight east-west road path at latitude y; optionally label its last node as T."""
    for k, n in enumerate(ids):
        graph.add_node(n, cord=(-80.42 + 0.001 * k, y), label="R", load=0.0)
    for a, b in zip(ids[:-1], ids[1:]):
        graph.add_edge(a, b, length=90.0, highway="residential")
    if label_last:
        graph.nodes[ids[-1]].update(label="T", load=1000.0)
        nx.relabel_nodes(graph, {ids[-1]: label_last}, copy=False)


def test_connect_components_single_component_unchanged(tmp_path):
    graph = nx.MultiGraph()
    _labelled_path(graph, [1, 2, 3], 37.19, label_last="T1")
    generator = SecondaryNetworkGenerator(output_dir=str(tmp_path), base_transformer_id=1)
    assert generator.connect_network_components(graph) is graph


def test_connect_components_drops_transformer_free_and_links_the_rest(tmp_path):
    graph = nx.MultiGraph()
    _labelled_path(graph, [1, 2, 3, 4, "x"], 37.190, label_last="T1")  # largest, with transformer
    _labelled_path(graph, [11, 12, "y"], 37.191, label_last="T2")      # smaller, with transformer
    _labelled_path(graph, [21, 22], 37.195)                            # no transformer

    generator = SecondaryNetworkGenerator(output_dir=str(tmp_path), base_transformer_id=1)
    connected = generator.connect_network_components(graph)

    assert nx.is_connected(connected)
    assert 21 not in connected and 22 not in connected
    # the bridge joins the closest pair of road nodes: 1 and 11 share a longitude
    assert connected.has_edge(1, 11)
    assert connected.edges[1, 11, 0]["highway"] == "unclassified"
    assert connected.edges[1, 11, 0]["length"] == pytest.approx(geodesic_m((-80.42, 37.190), (-80.42, 37.191)))


def test_get_node_coord():
    graph = nx.MultiGraph()
    graph.add_node(1, x=-80.0, y=37.0)
    graph.add_node("T1", cord=(-80.1, 37.1))
    graph.add_node(2)
    assert _get_node_coord(graph, 1) == (-80.0, 37.0)
    assert _get_node_coord(graph, "T1") == (-80.1, 37.1)
    assert _get_node_coord(graph, 2) is None


# Consistency between a fresh and a reloaded combined network

def test_combine_networks_returns_the_saved_network(combined):
    network, out_dir = combined["network"], combined["out_dir"]
    loaded = load_combined_network(
        nodes_file=str(out_dir / f"{PREFIX}_combined_network_nodes.csv"),
        edges_file=str(out_dir / f"{PREFIX}_combined_network_edges.csv"),
    )
    assert set(network.nodes) == set(loaded.nodes)
    assert sorted(map(sorted_edge, network.edges(keys=True))) == sorted(map(sorted_edge, loaded.edges(keys=True)))
    for n, d in loaded.nodes(data=True):
        if d["label"] == "R":
            assert (d["x"], d["y"]) == pytest.approx(d["cord"])


def sorted_edge(edge):
    u, v, k = edge
    return tuple(sorted((str(u), str(v)))) + (k,)


# Resuming per road link

def generate_links(tmp_path, homes):
    roads = road_network_as_in_main(tmp_path, make_road_grid(4))
    r2h = compute_edge_to_homes_map(map_homes_to_edges(roads, homes, padding_distance=0.001))
    generator = SecondaryNetworkGenerator(output_dir=str(tmp_path / "secnet"), base_transformer_id=5177700000)
    return roads, r2h, generator


def test_resume_drops_partially_saved_link(tmp_path, homes):
    roads, r2h, generator = generate_links(tmp_path, homes)
    links = list(r2h)
    for road_link in links[:2]:
        generator.generate_network_for_link(road_link, get_road_link_geometry(roads, road_link), r2h[road_link], **SECNET_ARGS)
        generator.save_results(prefix=PREFIX, road_link_id=road_link)
    out_dir = generator.output_dir
    clean = {f: (out_dir / f).read_text() for f in (
        f"{PREFIX}_transformers.csv", f"{PREFIX}_secondary_edges.csv", f"{PREFIX}_road_transformer_edges.txt")}

    # third link: rows appended to transformers / edges, sequence line cut off mid-write
    road_link = links[2]
    generator.generate_network_for_link(road_link, get_road_link_geometry(roads, road_link), r2h[road_link], **SECNET_ARGS)
    generator.save_results(prefix=PREFIX, road_link_id=road_link)
    seq = out_dir / f"{PREFIX}_road_transformer_edges.txt"
    seq.write_text(seq.read_text()[:-5])

    resumed = SecondaryNetworkGenerator(output_dir=str(out_dir), base_transformer_id=5177700000)
    assert resumed.resume(prefix=PREFIX) == set(links[:2])
    assert {f: (out_dir / f).read_text() for f in clean} == clean
    n_transformers = len(pd.read_csv(out_dir / f"{PREFIX}_transformers.csv"))
    assert resumed._generate_transformer_id() == f"T{5177700000 + n_transformers}"


def test_resume_without_saved_results_starts_fresh(tmp_path):
    generator = SecondaryNetworkGenerator(output_dir=str(tmp_path), base_transformer_id=1)
    (tmp_path / f"{PREFIX}_transformers.csv").write_text("transformer_id\nT9\n")  # orphaned file
    assert generator.resume(prefix=PREFIX) == set()
    assert not (tmp_path / f"{PREFIX}_transformers.csv").exists()
    assert generator._generate_transformer_id() == "T1"


def test_clear_results_removes_all_outputs(combined):
    out_dir = combined["out_dir"]
    generator = SecondaryNetworkGenerator(output_dir=str(out_dir), base_transformer_id=1)
    generator.clear_results(prefix=PREFIX)
    assert not list(out_dir.glob(f"{PREFIX}_*"))
