import osmnx as ox
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
from typing import List
import matplotlib.pyplot as plt
from pathlib import Path
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