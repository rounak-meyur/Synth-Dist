from collections import Counter

import pytest
from shapely.geometry import LineString

from conftest import REPO_ROOT
from utils.mapping import (
    compute_edge_to_homes_map,
    map_homes_to_edges,
    read_mapping_from_file,
    write_mapping_to_file,
)
from utils.osm_utils import load_road_network_from_files, save_road_network


@pytest.fixture
def road_graph(tmp_path, road_grid):
    """The grid as main.py sees it: saved to CSV and re-loaded as an undirected MultiGraph."""
    # fill the one missing geometry the same way load_roads does
    for u, v, d in road_grid.edges(data=True):
        if "geometry" not in d:
            d["geometry"] = LineString([(road_grid.nodes[n]["x"], road_grid.nodes[n]["y"]) for n in (u, v)])
    edges, nodes = tmp_path / "edges.csv", tmp_path / "nodes.csv"
    save_road_network(road_grid, str(edges), str(nodes))
    return load_road_network_from_files(str(edges), str(nodes))


# map_homes_to_edges

def test_map_homes_to_nearest_edge(road_graph, grid_homes):
    homes = [h for h, _ in grid_homes]
    result = map_homes_to_edges(road_graph, homes, padding_distance=0.001)

    mapped = {home.id: edge for home, edge in result}
    for home, expected in grid_homes:
        edge = mapped[home.id]
        assert edge is not None
        assert frozenset(edge[:2]) == expected
        assert road_graph.has_edge(*edge)


def test_map_homes_each_home_once(road_graph, homes):
    result = map_homes_to_edges(road_graph, homes, padding_distance=0.001)
    counts = Counter(home.id for home, _ in result)
    assert set(counts) == {h.id for h in homes}
    assert all(c == 1 for c in counts.values())


def test_map_homes_retries_with_larger_padding(road_graph, grid_homes):
    """
    With a tiny padding no edge bounding box reaches the homes on the first try;
    the retry decorator must widen the padding until every home is mapped, and
    the final result must hold exactly one (home, edge) entry per home.
    """
    homes = [h for h, _ in grid_homes]
    result = map_homes_to_edges(road_graph, homes, padding_distance=1e-6)

    counts = Counter(home.id for home, _ in result)
    assert all(c == 1 for c in counts.values()), f"duplicate entries: {counts}"
    assert all(edge is not None for _, edge in result)
    mapped = {home.id: edge for home, edge in result}
    for home, expected in grid_homes:
        assert frozenset(mapped[home.id][:2]) == expected


# write_mapping_to_file / read_mapping_from_file

def test_mapping_file_round_trip(tmp_path, road_graph, homes):
    result = map_homes_to_edges(road_graph, homes, padding_distance=0.001)
    path = tmp_path / "out" / "map_h2r.txt"
    write_mapping_to_file(result, str(path))

    assert path.exists()
    loaded = read_mapping_from_file(homes, filename=str(path))
    assert sorted((h.id, e) for h, e in loaded) == sorted((h.id, e) for h, e in result)
    # read_mapping_from_file must hand back the original Home objects
    by_id = {h.id: h for h in homes}
    assert all(h is by_id[h.id] for h, _ in loaded)


def test_write_mapping_skips_unmapped(tmp_path, homes):
    path = tmp_path / "map.txt"
    write_mapping_to_file([(homes[0], (1, 2, 0)), (homes[1], None)], str(path))
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(f"Home(id={homes[0].id},")


def test_read_mapping_ignores_unknown_homes_and_bad_lines(tmp_path, homes):
    path = tmp_path / "map.txt"
    h = homes[0]
    path.write_text(
        f"Home(id={h.id}, lon={h.cord[0]}, lat={h.cord[1]}) -> Edge(1, 2, 0)\n"
        "Home(id=999, lon=-80.0, lat=37.0) -> Edge(3, 4, 0)\n"
        "garbage line\n"
    )
    assert read_mapping_from_file(homes, str(path)) == [(h, (1, 2, 0))]


def test_read_mapping_file_not_found(tmp_path, homes):
    with pytest.raises(FileNotFoundError):
        read_mapping_from_file(homes, str(tmp_path / "missing.txt"))


# compute_edge_to_homes_map

def test_compute_edge_to_homes_map(homes):
    e1, e2 = (1, 2, 0), (2, 3, 0)
    h2r = [(homes[0], e1), (homes[1], e1), (homes[2], e2), (homes[3], None)]
    r2h = compute_edge_to_homes_map(h2r)
    assert r2h == {e1: [homes[0], homes[1]], e2: [homes[2]]}


def test_compute_edge_to_homes_map_covers_all_mapped_homes(road_graph, homes):
    h2r = map_homes_to_edges(road_graph, homes, padding_distance=0.001)
    r2h = compute_edge_to_homes_map(h2r)
    assert sorted(h.id for hs in r2h.values() for h in hs) == sorted(h.id for h in homes)


# Real data (only where the road network has been cached locally; skipped in CI)

ROAD_EDGES = REPO_ROOT / "out/interim/999_road_edges.csv"
ROAD_NODES = REPO_ROOT / "out/interim/999_road_nodes.csv"


@pytest.mark.integration
@pytest.mark.skipif(not (ROAD_EDGES.exists() and ROAD_NODES.exists()),
                    reason="cached road network for region 999 not present")
def test_region_999_all_homes_mapped():
    from utils.dataloader import load_homes

    homes = load_homes(REPO_ROOT / "data/load/999-home-load.csv")
    roads = load_road_network_from_files(str(ROAD_EDGES), str(ROAD_NODES))
    h2r = map_homes_to_edges(roads, homes, padding_distance=0.001)

    mapped = {h.id for h, e in h2r if e is not None}
    assert mapped == {h.id for h in homes}
    assert all(roads.has_edge(*e) for _, e in h2r if e is not None)
