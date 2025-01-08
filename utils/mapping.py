import networkx as nx
from pyqtree import Index
from typing import List, Tuple, Dict, Callable
from geopy.distance import geodesic
from shapely.ops import unary_union
from shapely.geometry import Point
from collections import defaultdict
from pathlib import Path
import functools
import re
import sys
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)


from utils.dataloader import Home
from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize(
        log_file_path="logs/test_mapping.log", 
        log_config_path="configs/logging_config.ini"
    )

logger = LogManager.get_logger("mapping")

def retry_with_increased_padding(max_retries: int = 5, padding_increment: float = 10):
    """
    Decorator to retry mapping homes to edges with increased padding distance.
    
    Args:
        max_retries (int): Maximum number of retry attempts.
        padding_increment (float): Amount to increase padding distance on each retry.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            original_padding = kwargs.get('padding_distance', 0.001)
            unmapped_homes = args[1]  # Assuming homes are the second argument
            all_mapped_homes = []

            while retries < max_retries and unmapped_homes:
                result = func(*args, **kwargs)
                mapped_homes = [home for home, edge in result if edge is not None]
                all_mapped_homes.extend(result)
                unmapped_homes = [home for home in unmapped_homes if home not in mapped_homes]

                if not unmapped_homes:
                    break

                retries += 1
                new_padding = original_padding * (padding_increment**retries)
                logger.info(f"Retry {retries} with padding distance {new_padding}")
                kwargs['padding_distance'] = new_padding
                args = (args[0], unmapped_homes) + args[2:]  # Update homes list in args

            return all_mapped_homes

        return wrapper
    return decorator

@retry_with_increased_padding(max_retries=5, padding_increment=10)
def map_homes_to_edges(
        G: nx.MultiDiGraph, 
        homes: List[Home], 
        padding_distance: float = 0.005
        ) -> List[Tuple[Home, Tuple]]:
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
        x1, y1, x2, y2 = edge_geom.bounds
        edge_box = x1-padding_distance, y1-padding_distance, x2+padding_distance, y2+padding_distance
        idx.insert((u, v, key), edge_box)

    mapped_homes = []

    for home in homes:
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

        mapped_homes.append((home, nearest_edge))
        if nearest_edge is None:
            logger.warning(f"No nearby edge found for home at {home.cord}")

    logger.info(f"Successfully mapped {len([h for h, e in mapped_homes if e is not None])} homes to edges")

    return mapped_homes

def write_mapping_to_file(mapped_homes: List[Tuple[Home, Tuple]], filename: str = 'home_to_edge_mapping.txt'):
    """
    Write the home to edge mapping results to a txt file.

    Args:
        mapped_homes (List[Tuple[Home, Tuple]]): List of tuples containing each home and its nearest edge (u, v, key).
        filename (str): Name of the output file (default: 'home_to_edge_mapping.txt').
    """
    output_file = Path(filename)
    outdir = output_file.parent
    outdir.mkdir(parents=True, exist_ok=True)
    with output_file.open('w') as f:
        for home, edge in mapped_homes:
            if edge is not None:
                f.write(f"Home(id={home.id}, lon={home.cord[0]}, lat={home.cord[1]}) -> Edge{edge}\n")
    
    logger.info(f"Home to edge mapping written to {output_file}")

def read_mapping_from_file(homes: List[Home], filename: str) -> List[Tuple[Home, Tuple]]:
    """
    Read the home to edge mapping from a txt file, using a list of homes to populate full home information.

    Args:
        homes (List[Home]): List of Home objects to use for populating full home information.
        filename (str): Name of the input file (default: 'home_to_edge_mapping.txt').

    Returns:
        List[Tuple[Home, Tuple]]: List of tuples containing each home and its nearest edge (u, v, key).
    """
    input_file = Path(filename)
    if not input_file.exists():
        logger.error(f"Mapping file not found: {input_file}")
        raise FileNotFoundError(f"The file {input_file} does not exist.")

    # Create a dictionary of homes keyed by their id for quick lookup
    home_dict = {home.id: home for home in homes}

    mapped_homes = []
    with input_file.open('r') as f:
        for line in f:
            # Parse the line using regular expressions
            home_match = re.match(r"Home\(id=(\d+), lon=([-\d.]+), lat=([-\d.]+)\) -> Edge\((\d+), (\d+), (\d+)\)", line.strip())
            if home_match:
                home_id, lon, lat, u, v, key = home_match.groups()
                home_id = int(home_id)
                if home_id in home_dict:
                    home = home_dict[home_id]
                    edge = (int(u), int(v), int(key))
                    mapped_homes.append((home, edge))
                else:
                    logger.warning(f"Home with id {home_id} not found in the provided list of homes")
            else:
                logger.warning(f"Skipping invalid line in mapping file: {line.strip()}")

    logger.info(f"Read {len(mapped_homes)} mappings from {input_file}")
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
        if edge is not None:
            edge_to_homes_map[edge].append(home)

    logger.info(f"Computed reverse map for {len(edge_to_homes_map)} edges")

    return dict(edge_to_homes_map)