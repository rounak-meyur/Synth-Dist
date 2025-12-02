import osmnx as ox
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import shapely.wkt as wkt
from typing import List
import matplotlib.pyplot as plt
import networkx as nx
import csv
from pathlib import Path
import os
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)


from utils.dataloader import Home, Substation
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

def save_road_network(
        G: nx.MultiDiGraph, 
        edgelist_file: str = 'road_network_edges.csv', 
        nodelist_file: str = 'road_network_nodes.csv'
        ):
    """
    Save road network graph as separate edgelist and nodelist files.

    Args:
        G (nx.MultiDiGraph): The road network graph.
        edgelist_file (str): Name of the output file for edges (default: 'road_network_edges.csv').
        nodelist_file (str): Name of the output file for nodes (default: 'road_network_nodes.csv').
    """
    logger.info(f"Saving road network to {edgelist_file} and {nodelist_file}")
    
    # Create output directories if they don't exist
    Path(edgelist_file).parent.mkdir(parents=True, exist_ok=True)
    Path(nodelist_file).parent.mkdir(parents=True, exist_ok=True)

    # Save edges
    with open(edgelist_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write edge header
        writer.writerow(['u', 'v', 'key', 'geometry', 'length', 'highway'])
        
        # Write edges
        for u, v, key, data in G.edges(keys=True, data=True):
            # Convert geometry to WKT format for string representation
            geometry_wkt = data.get('geometry', '').wkt if 'geometry' in data else ''
            
            # Get edge attributes
            length = data.get('length', '')
            highway = data.get('highway', '')
            
            writer.writerow([u, v, key, geometry_wkt, length, highway])
    
    # Save nodes
    with open(nodelist_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write node header
        writer.writerow(['node_id', 'x', 'y'])
        
        # Write nodes
        for node, data in G.nodes(data=True):
            x = data.get('x', '')
            y = data.get('y', '')
            writer.writerow([node, x, y])
    
    logger.info(f"Saved {G.number_of_edges()} edges and {G.number_of_nodes()} nodes")

def load_road_network_from_files(edgelist_file: str = 'road_network_edges.csv',
                               nodelist_file: str = 'road_network_nodes.csv') -> nx.MultiDiGraph:
    """
    Load road network graph from edgelist and nodelist files.

    Args:
        edgelist_file (str): Path to the edgelist file.
        nodelist_file (str): Path to the nodelist file.

    Returns:
        nx.MultiGraph: The loaded road network graph.

    Raises:
        FileNotFoundError: If either file doesn't exist.
    """
    logger.info(f"Loading road network from {edgelist_file} and {nodelist_file}")
    
    # Check if files exist
    if not Path(edgelist_file).exists():
        logger.error(f"Edgelist file not found: {edgelist_file}")
        raise FileNotFoundError(f"The file {edgelist_file} does not exist.")
    
    if not Path(nodelist_file).exists():
        logger.error(f"Nodelist file not found: {nodelist_file}")
        raise FileNotFoundError(f"The file {nodelist_file} does not exist.")
    
    G = nx.MultiGraph()
    
    # Load nodes first
    with open(nodelist_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row['node_id'])
            # Convert coordinates to float, use None if empty
            x = float(row['x']) if row['x'] else None
            y = float(row['y']) if row['y'] else None
            G.add_node(node_id, x=x, y=y)
    
    # Load edges
    with open(edgelist_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert geometry from WKT back to Shapely geometry
            geometry = wkt.loads(row['geometry']) if row['geometry'] else None
            
            # Add edge with attributes
            G.add_edge(
                int(row['u']), 
                int(row['v']), 
                key=int(row['key']),
                geometry=geometry,
                length=float(row['length']) if row['length'] else None,
                highway=row['highway']
            )
    
    logger.info(f"Loaded {G.number_of_edges()} edges and {G.number_of_nodes()} nodes")
    return G

def plot_network(G, homes: List[Home], subs:List[Substation], filename: str = 'network.png'):
    """
    Plot the road network and homes, saving the plot in the 'figs' directory.

    Args:
        G (networkx.MultiDiGraph): The road network graph.
        homes (List[Home]): A list of Home objects to plot.
        subs (List[Substation]): A list of Substation objects to plot.
        filename (str): The filename to save the plot (default: 'network.png').
    """
    # Ensure the 'figs' directory exists
    figs_dir = Path('figs')
    figs_dir.mkdir(exist_ok=True)

    # Create the full path for the file
    filepath = figs_dir / filename

    logger.info(f"Plotting network with {len(G.edges)} edges and {len(homes)} homes")

    # Plot the graph
    fig, ax = ox.plot_graph(
        G, figsize=(20,20), 
        node_size=1, node_color='black', edge_linewidth=1, bgcolor='white',
        show=False, close=False)

    # Plot homes
    for home in homes:
        ax.scatter(home.cord[0], home.cord[1], c='crimson', s=2, zorder=3, label="Residences")
    
    # Plot substations
    for sub in subs:
        ax.scatter(sub.cord[0], sub.cord[1], c='royalblue', s=500, zorder=3, label="Substations")

    # Save the plot
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    logger.info(f"Plot saved as {filepath}")

if __name__ == "__main__":
    from dataloader import load_homes, load_substations
    test_homes = load_homes("data/load/999-home-load.csv")
    test_subs = load_substations("data/substations.csv",test_homes)
    test_roads = load_roads(test_homes)
    plot_network(test_roads, test_homes, test_subs, filename="test_roads.png")