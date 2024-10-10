import osmnx as ox
import networkx as nx
from pyqtree import Index
from geopy.distance import geodesic
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import os
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)


from utils.dataloader import Home
from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize(
        log_file_path="logs/test_osm_utils.log", 
        log_config_path="configs/logging_config.ini"
    )

logger = LogManager.get_logger("osm_utils")

def load_roads(homes: List[Home]):
    """
    Load road data from OpenStreetMap based on the bounding polygon of given homes.

    Args:
        homes (List[Home]): A list of Home objects containing coordinate information.

    Returns:
        networkx.MultiDiGraph: A graph representing the road network.

    Raises:
        ValueError: If no homes are provided or if no roads are found in the area.
    """
    if not homes:
        logger.error("No homes provided to load_roads")
        raise ValueError("No homes provided.")

    logger.info(f"Creating polygon from {len(homes)} homes")
    # Create points from home coordinates
    points = [Point(home.cord[0], home.cord[1]) for home in homes]

    # Create a polygon from the points
    polygon = Polygon(unary_union(points).convex_hull)

    logger.info("Loading road network from OpenStreetMap")
    # Load the road network within the polygon with updated parameters
    G = ox.graph_from_polygon(polygon, network_type="all", retain_all=True, truncate_by_edge=False)

    if not G.edges:
        logger.error("No roads found in the given area")
        raise ValueError("No roads found in the given area.")

    logger.info(f"Loaded road network with {len(G.edges)} edges")

    # Ensure all edges have a geometry
    edges_without_geometry = 0
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'geometry' not in data:
            # Create a straight line geometry if it doesn't exist
            start_node = G.nodes[u]
            end_node = G.nodes[v]
            line = LineString([(start_node['x'], start_node['y']), (end_node['x'], end_node['y'])])
            data['geometry'] = line
            edges_without_geometry += 1

    logger.info(f"Added geometry to {edges_without_geometry} edges without existing geometry")
    return G

def map_homes_to_edges(G: nx.MultiDiGraph, homes: List[Home], padding_distance: float = 0.005) -> List[Tuple[Home, Tuple]]:
    """
    Map each home to the nearest road network edge.

    Args:
        G (nx.MultiDiGraph): The road network graph.
        homes (List[Home]): List of Home objects to map.
        padding_distance (float): Distance in radians to use for padding around edges and homes.

    Returns:
        List[Tuple[Home, Tuple]]: List of tuples containing each home and its nearest edge (u, v, key).
    """
    logger.info(f"Mapping {len(homes)} homes to nearest edges")

    # Calculate and add bounding box to the graph
    node_points = [Point((data['x'], data['y'])) for _, data in G.nodes(data=True)]
    bbox = unary_union(node_points).bounds
    G.graph['bbox'] = bbox

    # Create a spatial index for the edge geometries
    idx = Index(bbox=(G.graph['bbox'][0], G.graph['bbox'][1], G.graph['bbox'][2], G.graph['bbox'][3]))

    # Add padded edge geometries to the index
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_geom = data['geometry']
        edge_box = edge_geom.buffer(padding_distance).bounds
        idx.insert((u, v, key), edge_box)

    mapped_homes = []

    for home_idx, home in enumerate(homes):
        if (home_idx % 20) == 0 and home_idx != 0:
            logger.info(f"{home_idx}/{len(homes)} mapped to nearest road network edge.")
        home_point = Point(home.cord[0], home.cord[1])
        home_box = home_point.buffer(padding_distance).bounds

        # Find potential nearby edges
        nearby_edges = idx.intersect(home_box)

        nearest_edge = None
        min_distance = float('inf')

        for u, v, key in nearby_edges:
            edge_geom = G.edges[u, v, key]['geometry']
            edge_coords = list(edge_geom.coords)
            
            # Calculate distances to all points on the edge
            distances = [geodesic((home.cord[1], home.cord[0]), (y, x)).meters for x, y in edge_coords]
            
            min_edge_distance = min(distances)
            
            if min_edge_distance < min_distance:
                min_distance = min_edge_distance
                nearest_edge = (u, v, key)

        if nearest_edge:
            mapped_homes.append((home, nearest_edge))
        else:
            logger.warning(f"No nearby edge found for home at {home.cord}")

    logger.info(f"Successfully mapped {len(mapped_homes)} homes to edges")
    return mapped_homes

def compute_edge_to_homes_map(home_to_edge_map: List[Tuple[Home, Tuple]]) -> Dict[Tuple, List[Home]]:
    """
    Compute a reverse map from road network edges to homes.

    Args:
        home_to_edge_map (List[Tuple[Home, Tuple]]): List of tuples containing each home and its nearest edge (u, v, key).

    Returns:
        Dict[Tuple, List[Home]]: Dictionary mapping each edge (u, v, key) to a list of homes nearest to it.
    """
    logger.info("Computing reverse map from edges to homes")

    edge_to_homes_map = defaultdict(list)

    for home, edge in home_to_edge_map:
        edge_to_homes_map[edge].append(home)

    logger.info(f"Computed reverse map for {len(edge_to_homes_map)} edges")

    return dict(edge_to_homes_map)

def plot_network(G, homes: List[Home], filename: str = 'network.png'):
    """
    Plot the road network and homes, saving the plot in the 'figs' directory.

    Args:
        G (networkx.MultiDiGraph): The road network graph.
        homes (List[Home]): A list of Home objects to plot.
        filename (str): The filename to save the plot (default: 'network.png').
    """
    # Ensure the 'figs' directory exists
    figs_dir = Path('figs')
    figs_dir.mkdir(exist_ok=True)

    # Create the full path for the file
    filepath = figs_dir / filename

    logger.info(f"Plotting network with {len(G.edges)} edges and {len(homes)} homes")

    # Plot the graph
    fig, ax = ox.plot_graph(G, show=False, close=False)

    # Plot homes
    for home in homes:
        ax.scatter(home.cord[0], home.cord[1], c='red', s=10, zorder=3)

    # Save the plot
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Plot saved as {filepath}")

if __name__ == "__main__":
    from dataloader import load_homes
    test_homes = load_homes("data/load/test-home-load.csv")
    test_roads = load_roads(test_homes)
    plot_network(test_roads, test_homes, filename="test_roads.png")