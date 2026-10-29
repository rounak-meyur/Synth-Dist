from unittest.mock import MagicMock

import networkx as nx
import pytest
from shapely.geometry import LineString, Point

from conftest import grid_node_id
from utils import osm_utils
from utils.osm_utils import (
    load_road_network_from_files,
    load_roads,
    plot_network,
    save_road_network,
)


@pytest.fixture
def mock_osm(monkeypatch, road_grid):
    """Replace the OpenStreetMap download with the synthetic grid."""
    mock = MagicMock(return_value=road_grid)
    monkeypatch.setattr(osm_utils.ox, "graph_from_polygon", mock)
    return mock


# load_roads

def test_load_roads_queries_osm_with_home_hull(mock_osm, homes):
    load_roads(homes)

    mock_osm.assert_called_once()
    (polygon,), kwargs = mock_osm.call_args
    assert kwargs == {"network_type": "all", "retain_all": True, "truncate_by_edge": False}
    for home in homes:
        assert polygon.buffer(1e-9).contains(Point(home.cord))


BUSY = osm_utils.ox._errors.ResponseStatusCodeError(
    "'overpass-api.de' responded: 503 Service Unavailable This Overpass API server is currently at capacity."
)


def test_load_roads_retries_busy_overpass_with_exponential_backoff(monkeypatch, road_grid, homes):
    osm = MagicMock(side_effect=[BUSY, BUSY, road_grid])
    sleep = MagicMock()
    monkeypatch.setattr(osm_utils.ox, "graph_from_polygon", osm)
    monkeypatch.setattr(osm_utils.time, "sleep", sleep)

    G = load_roads(homes, max_retries=3, retry_wait=10)

    assert G is road_grid
    assert osm.call_count == 3
    assert [c.args[0] for c in sleep.call_args_list] == [10, 20]


def test_load_roads_gives_up_after_max_retries(monkeypatch, homes):
    osm = MagicMock(side_effect=BUSY)
    sleep = MagicMock()
    monkeypatch.setattr(osm_utils.ox, "graph_from_polygon", osm)
    monkeypatch.setattr(osm_utils.time, "sleep", sleep)

    with pytest.raises(osm_utils.ox._errors.ResponseStatusCodeError, match="at capacity"):
        load_roads(homes, max_retries=2, retry_wait=1)
    assert osm.call_count == 3
    assert [c.args[0] for c in sleep.call_args_list] == [1, 2]


def test_load_roads_does_not_retry_other_errors(monkeypatch, homes):
    osm = MagicMock(side_effect=KeyError("bug"))
    monkeypatch.setattr(osm_utils.ox, "graph_from_polygon", osm)
    monkeypatch.setattr(osm_utils.time, "sleep", MagicMock())
    with pytest.raises(KeyError):
        load_roads(homes)
    assert osm.call_count == 1


def test_load_roads_adds_missing_geometry(mock_osm, homes):
    G = load_roads(homes)

    assert all("geometry" in d for _, _, d in G.edges(data=True))
    u, v = grid_node_id(0, 0), grid_node_id(1, 0)
    geom = G.edges[u, v, 0]["geometry"]
    assert isinstance(geom, LineString)
    assert list(geom.coords) == [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]


def test_load_roads_no_homes():
    with pytest.raises(ValueError):
        load_roads([])


def test_load_roads_no_edges(monkeypatch, homes):
    monkeypatch.setattr(osm_utils.ox, "graph_from_polygon", MagicMock(return_value=nx.MultiDiGraph()))
    with pytest.raises(ValueError):
        load_roads(homes)


# save_road_network / load_road_network_from_files

@pytest.fixture
def saved_grid(tmp_path, mock_osm, homes):
    G = load_roads(homes)
    edge_file = tmp_path / "nested" / "edges.csv"
    node_file = tmp_path / "nested" / "nodes.csv"
    save_road_network(G, edgelist_file=str(edge_file), nodelist_file=str(node_file))
    return G, edge_file, node_file


def test_save_road_network_creates_files(saved_grid):
    G, edge_file, node_file = saved_grid
    assert edge_file.exists() and node_file.exists()
    # header + one row per edge / node
    assert len(edge_file.read_text().splitlines()) == G.number_of_edges() + 1
    assert len(node_file.read_text().splitlines()) == G.number_of_nodes() + 1


def test_road_network_round_trip(saved_grid):
    G, edge_file, node_file = saved_grid
    H = load_road_network_from_files(edgelist_file=str(edge_file), nodelist_file=str(node_file))

    assert isinstance(H, nx.MultiGraph) and not H.is_directed()
    assert set(H.nodes) == set(G.nodes)
    for n, data in G.nodes(data=True):
        assert H.nodes[n]["x"] == pytest.approx(data["x"])
        assert H.nodes[n]["y"] == pytest.approx(data["y"])

    # Directed (u,v) and (v,u) collapse to one undirected edge per road segment
    undirected_pairs = {frozenset((u, v)) for u, v in G.edges()}
    assert {frozenset((u, v)) for u, v in H.edges()} == undirected_pairs

    for u, v, key, data in H.edges(keys=True, data=True):
        assert isinstance(data["geometry"], LineString)
        assert data["length"] == pytest.approx(222.0)
        assert data["highway"] == "residential"
        assert isinstance(u, int) and isinstance(v, int) and isinstance(key, int)


@pytest.mark.parametrize("missing", ["edges", "nodes"])
def test_load_road_network_missing_file(saved_grid, missing):
    _, edge_file, node_file = saved_grid
    (edge_file if missing == "edges" else node_file).unlink()
    with pytest.raises(FileNotFoundError):
        load_road_network_from_files(edgelist_file=str(edge_file), nodelist_file=str(node_file))


# plot_network

def test_plot_network_saves_into_figs(tmp_path, monkeypatch, homes):
    monkeypatch.chdir(tmp_path)
    fig, ax = MagicMock(), MagicMock()
    monkeypatch.setattr(osm_utils.ox, "plot_graph", MagicMock(return_value=(fig, ax)))
    monkeypatch.setattr(osm_utils.plt, "close", MagicMock())

    plot_network(nx.MultiDiGraph(), homes, "test_plot.png")

    assert (tmp_path / "figs").is_dir()
    fig.savefig.assert_called_once()
    assert fig.savefig.call_args.args[0].as_posix() == "figs/test_plot.png"
    assert ax.scatter.call_count == len(homes)
