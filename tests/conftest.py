"""
Shared pytest fixtures.

All fixtures are generated in code (no network access, no committed CSVs) so the
suite runs the same locally and in CI.
"""
import csv
import logging
from pathlib import Path

import networkx as nx
import pytest
from geopy.distance import geodesic
from shapely.geometry import LineString

from utils.dataloader import Home

REPO_ROOT = Path(__file__).resolve().parent.parent

# Synthetic grid near Montgomery County, VA (region 999 in data/load/)
GRID_ORIGIN = (-80.42, 37.19)
GRID_STEP = 0.002  # degrees (~200 m)
GRID_SIZE = 4      # 4 x 4 nodes


def grid_node_id(i: int, j: int) -> int:
    # Large positive integers, like OSM node ids
    return 1_000_000 + i * 100 + j


def grid_node_coord(i: int, j: int):
    return (GRID_ORIGIN[0] + i * GRID_STEP, GRID_ORIGIN[1] + j * GRID_STEP)


def make_profile(scale: float = 1.0):
    return [scale * h for h in range(1, 25)]


def write_homes_csv(path: Path, homes):
    """Write homes in the same format as data/load/<region>-home-load.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hid", "longitude", "latitude"] + [f"hour{i}" for i in range(1, 25)])
        for home in homes:
            writer.writerow([home.id, home.cord[0], home.cord[1]] + home.profile)


@pytest.fixture
def road_grid() -> nx.MultiDiGraph:
    return make_road_grid(GRID_SIZE)


def make_road_grid(size: int) -> nx.MultiDiGraph:
    """
    An OSMnx-like road graph: a size x size grid with edges in both
    directions. Each edge geometry has a midpoint vertex so a home placed next
    to the midpoint has an unambiguous nearest road segment. One edge is left
    without geometry to mimic OSMnx output for straight segments.
    """
    G = nx.MultiDiGraph(crs="epsg:4326")
    for i in range(size):
        for j in range(size):
            x, y = grid_node_coord(i, j)
            G.add_node(grid_node_id(i, j), x=x, y=y)

    for i in range(size):
        for j in range(size):
            for di, dj in ((1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                if ni >= size or nj >= size:
                    continue
                u, v = grid_node_id(i, j), grid_node_id(ni, nj)
                p, q = grid_node_coord(i, j), grid_node_coord(ni, nj)
                mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
                for a, b, pts in ((u, v, [p, mid, q]), (v, u, [q, mid, p])):
                    G.add_edge(
                        a, b, key=0,
                        geometry=LineString(pts),
                        length=222.0,
                        highway="residential",
                    )

    # Edge without geometry (load_roads should add a straight-line geometry)
    del G.edges[grid_node_id(0, 0), grid_node_id(1, 0), 0]["geometry"]
    return G


@pytest.fixture
def grid_homes():
    """
    Homes placed just off the midpoints of known grid edges.

    Returns a list of (Home, expected undirected edge as frozenset{u, v}).
    """
    targets = [((0, 1), (1, 1)), ((1, 1), (1, 2)), ((2, 2), (3, 2)), ((2, 0), (2, 1)), ((1, 3), (2, 3))]
    offset = 0.0001  # ~10 m off the road
    out = []
    for k, ((i1, j1), (i2, j2)) in enumerate(targets):
        p, q = grid_node_coord(i1, j1), grid_node_coord(i2, j2)
        mid = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
        # Shift perpendicular to the edge direction
        if j1 == j2:   # horizontal edge
            cord = (mid[0], mid[1] + offset)
        else:          # vertical edge
            cord = (mid[0] + offset, mid[1])
        profile = make_profile(k + 1)
        home = Home(id=5100000000 + k, cord=cord, profile=profile,
                    peak=max(profile), load=sum(profile) / 24)
        out.append((home, frozenset({grid_node_id(i1, j1), grid_node_id(i2, j2)})))
    return out


@pytest.fixture
def homes(grid_homes):
    return [h for h, _ in grid_homes]


@pytest.fixture
def restore_logging():
    """LogManager.initialize reconfigures the root logger; undo that after the test."""
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        if handler not in saved_handlers:
            handler.close()
    root.handlers = saved_handlers
    root.setLevel(saved_level)


# Combined road + transformer networks (as produced by combine_networks)

SIZE = 16  # 256 road nodes: NetworkPartitioner keeps only cells with > 100 nodes


def geodesic_m(a, b):
    return geodesic((a[1], a[0]), (b[1], b[0])).meters


def make_combined_grid(size: int = SIZE, barrier_at: int = None) -> nx.MultiGraph:
    """
    A combined road + transformer network as produced by combine_networks: a grid of road
    nodes ('R') where every horizontal road link carries a transformer ('T') at its midpoint.

    barrier_at: remove the horizontal links between columns barrier_at and barrier_at + 1
    except on the top row (e.g. a river with a single bridge).
    """
    G = nx.MultiGraph()
    for i in range(size):
        for j in range(size):
            G.add_node(grid_node_id(i, j), cord=grid_node_coord(i, j), label="R", load=0.0)

    def link(a, b):
        ca, cb = G.nodes[a]["cord"], G.nodes[b]["cord"]
        G.add_edge(a, b, length=geodesic_m(ca, cb), highway="residential", geometry=LineString([ca, cb]))

    for i in range(size):
        for j in range(size):
            u = grid_node_id(i, j)
            if j + 1 < size:
                link(u, grid_node_id(i, j + 1))
            if i + 1 < size and not (barrier_at == i and j != size - 1):
                v = grid_node_id(i + 1, j)
                (x1, y1), (x2, y2) = G.nodes[u]["cord"], G.nodes[v]["cord"]
                t = f"T{i:02d}{j:02d}"
                G.add_node(t, cord=((x1 + x2) / 2, (y1 + y2) / 2), label="T", load=1000.0)
                link(u, t)
                link(t, v)
    return G
