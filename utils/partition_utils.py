from typing import Dict, List, Tuple, Callable
import networkx as nx
import numpy as np
from geopy.distance import geodesic
from pyqtree import Index
import json
from pathlib import Path
from functools import wraps
import os
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager
from utils.dataloader import Substation

if __name__ == "__main__":
    LogManager.initialize(
        log_file_path="logs/test_partition_utils.log", 
        log_config_path="configs/logging_config.ini"
    )

logger = LogManager.get_logger("partition_utils")

def retry_with_increased_radius(
    max_retries: int = 3,
    radius_multiplier: float = 2.0
):
    """
    Decorator to retry a function with increased search radius.
    
    Args:
        max_retries (int): Maximum number of retries
        radius_multiplier (float): Factor to multiply radius by on each retry
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            radius = kwargs.get('search_radius', 0.001)
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    # Update search radius in kwargs
                    kwargs['search_radius'] = radius
                    logger.info(f"Trying with padding radius {radius}")
                    return func(*args, **kwargs)
                except ValueError as e:
                    last_exception = e
                    # Increase radius for next attempt
                    radius *= radius_multiplier
            
            # If we get here, all retries failed
            raise last_exception or ValueError(
                f"No nodes found after {max_retries} attempts "
                f"with maximum radius {radius/radius_multiplier}"
            )
        return wrapper
    return decorator

class NetworkPartitioner:
    """Class to handle network partitioning based on substations."""
    
    def __init__(self, combined_network: nx.MultiGraph):
        """
        Initialize partitioner with combined network.
        
        Args:
            combined_network (nx.MultiGraph): Network with road and transformer nodes
        """
        self.network = combined_network
        # Get bounds for spatial index
        self.bbox = self._get_network_bounds()
        # Create spatial index for road nodes
        self.spatial_index, self.road_nodes = self._build_spatial_index()
    
    def _get_network_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the bounding box of the network.
        
        Returns:
            Tuple[float, float, float, float]: (min_x, min_y, max_x, max_y)
        """
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for node, attrs in self.network.nodes(data=True):
            if attrs['label'] == 'R':  # Road nodes
                lon, lat = attrs['cord']
                min_x = min(min_x, lon)
                min_y = min(min_y, lat)
                max_x = max(max_x, lon)
                max_y = max(max_y, lat)
        
        # Add small buffer
        buffer = 0.01  # ~1km at equator
        return (min_x - buffer, min_y - buffer, max_x + buffer, max_y + buffer)
    
    def _build_spatial_index(self) -> Tuple[Index, Dict[int, Tuple[float, float]]]:
        """
        Build Pyqtree spatial index for road nodes.
        
        Returns:
            Tuple[Index, Dict]: Spatial index and dictionary of road node coordinates
        """
        # Initialize Pyqtree index with network bounds
        spindex = Index(bbox=self.bbox)
        road_nodes = {}
        
        # Add road nodes to spatial index
        for node, attrs in self.network.nodes(data=True):
            if attrs['label'] == 'R':  # Road nodes
                lon, lat = attrs['cord']
                # Store in dictionary
                road_nodes[node] = (lon, lat)
                # Add to Pyqtree index
                spindex.insert(node, (lon, lat, lon, lat))
        
        return spindex, road_nodes
    
    def _find_nodes_in_radius(
        self,
        coord: Tuple[float, float],
        search_radius: float
    ) -> List[int]:
        """
        Find road nodes within the specified radius of a coordinate.
        
        Args:
            coord (Tuple[float, float]): (longitude, latitude) coordinate
            search_radius (float): Search radius in degrees
            
        Returns:
            List[int]: List of node IDs within radius
            
        Raises:
            ValueError: If no nodes found within radius
        """
        lon, lat = coord
        search_bbox = (
            lon - search_radius,
            lat - search_radius,
            lon + search_radius,
            lat + search_radius
        )
        
        candidate_nodes = self.spatial_index.intersect(search_bbox)
        if not candidate_nodes:
            err_msg = f"No nodes found within radius {search_radius}"
            logger.warning(err_msg)
            raise ValueError(err_msg)
        
        return candidate_nodes
    
    @retry_with_increased_radius(max_retries=3, radius_multiplier=10.0)
    def find_nearest_road_nodes(
        self,
        substations: List[Substation],
        search_radius: float = 0.001
    ) -> Dict[int, int]:
        """
        Find nearest road node for each substation.
        Will retry with increased radius if no nodes found.
        
        Args:
            substations (List[Substation]): List of substations
            search_radius (float): Initial search radius in degrees
            
        Returns:
            Dict[int, int]: Mapping of substation ID to nearest road node ID
        """
        substation_nodes = {}
        
        for substation in substations:
            # Find candidate nodes within radius
            candidate_nodes = self._find_nodes_in_radius(
                substation.cord,
                search_radius
            )
            logger.info(f"Found {len(candidate_nodes)} nearby road nodes for search radius {search_radius}")
            
            # Calculate distances to all candidate nodes
            distances = []
            lon, lat = substation.cord
            for node_id in candidate_nodes:
                node_lon, node_lat = self.road_nodes[node_id]
                dist = np.sqrt((lon - node_lon)**2 + (lat - node_lat)**2)
                distances.append((dist, node_id))
            
            # Select nearest node
            nearest_node = min(distances, key=lambda x: x[0])[1]
            substation_nodes[substation.id] = nearest_node
            logger.info(f"The nearest road network node to substation {substation.id} is {nearest_node}")
        
        return substation_nodes
    
    def partition_transformers(
        self,
        substation_nodes: Dict[int, int]
    ) -> Dict[int, List[str]]:
        """
        Assign transformers to substations using Voronoi partitioning based on network distances.
        
        Args:
            substation_nodes (Dict[int, int]): Mapping of substation ID to road node ID
            
        Returns:
            Dict[int, List[str]]: Mapping of substation ID to list of assigned transformer IDs
        """
        # Extract center nodes for Voronoi partitioning
        centers = list(substation_nodes.values())
        logger.info(f"Number of nodes: {len(self.network.nodes)}")
        
        # First Voronoi partitioning
        cells = nx.voronoi_cells(self.network, centers, 'length')
        
        # Filter centers with significant cells
        valid_centers = [c for c in centers if len(cells[c]) > 100]
        if not valid_centers:
            logger.error("No valid centers found after filtering")
            raise ValueError("No valid centers found after filtering")
        
        # Recompute Voronoi cells with valid centers
        cells = nx.voronoi_cells(self.network, valid_centers, 'length')
        
        # Create assignments dictionary
        assignments = {
            sub_id: []
            for sub_id, node in substation_nodes.items()
            if node in valid_centers
        }
        
        # Assign transformers based on cell membership
        assignment_count = 0
        for center_node in valid_centers:
            # Find corresponding substation ID
            sub_id = next(
                sid for sid, node in substation_nodes.items() 
                if node == center_node
            )
            
            assignments[sub_id] = list(cells[center_node])
            assignment_count += len(assignments[sub_id])
        
        logger.info(f"Number of assignments: {assignment_count}")
        return assignments
    
    def save_partitioning(
        self,
        assignments: Dict[int, List[str]],
        subs: List[Substation],
        output_file: str
    ) -> None:
        """
        Save transformer-substation assignments to JSON file with node and edge information
        from induced subgraphs.
        
        Args:
            assignments (Dict[int, List[str]]): Mapping of substation IDs to transformer and road node lists
            substations (List[Substation]): List of substations
            output_file (str): Path to output JSON file
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create structured data for JSON
        partitioned_data = {}
        
        for sub in subs:
            # Get substation coordinates
            sub_cord = sub.cord
            sub_id = str(sub.id)
            
            if sub_id not in assignments:
                logger.info(f"No nodes assigned to {sub_id}")
            
            else:
                partition_nodes = assignments[sub_id]
            
                # Get induced subgraph
                subgraph = self.network.subgraph(partition_nodes)
                
                # Create node list with attributes
                nodelist = {}
                for node, attrs in subgraph.nodes(data=True):
                    nodelist[str(node)] = {
                        "cord": attrs['cord'],
                        "label": attrs['label'],
                        "load": attrs['load'],
                        "distance": geodesic(
                            (sub_cord[1], sub_cord[0]),
                            (attrs['cord'][1], attrs['cord'][0])
                        ).meters if attrs['label'] == 'R' else 999999999
                    }
                
                # Create edge list with attributes
                edgelist = {}
                for u, v, k, data in subgraph.edges(data=True, keys=True):
                    edge_key = f"({u},{v},{k})"  # Create string key for JSON
                    edgelist[edge_key] = {
                        "highway": data.get('highway', 'unclassified'),
                        "geometry": str(data.get('geometry')),  # Convert LineString to string
                        "length": data.get('length', 0.0)
                    }
                
                # Store partition data
                partitioned_data[str(sub_id)] = {
                    "nodelist": nodelist,
                    "edgelist": edgelist
                }
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(partitioned_data, f, indent=2)


def load_partitioning(
    input_file: str
    ) -> Dict[int, Dict[str, Dict]]:
    """
    Load transformer graph-substation assignments from JSON file.
    
    Args:
        input_file (str): Path to input JSON file
        
    Returns:
        Dict[int, Dict[str, Dict]]: Mapping of substation IDs (int) to transformer graph data (dict)
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If JSON structure is invalid
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Assignment file not found: {input_file}")
    
    try:
        with open(input_path, 'r') as f:
            json_data = json.load(f)
        
        # Convert substation IDs from strings back to integers
        assignments = {}
        for sub_id_str, graph_data in json_data.items():
            # Validate substation ID
            try:
                sub_id = int(sub_id_str)
            except ValueError:
                logger.error(f"Expected integer format for substation ID {sub_id_str}")
                raise ValueError(
                    f"Invalid substation ID format: {sub_id_str}. "
                    "Expected integer."
                )
            
            assignments[sub_id] = graph_data
        
        return assignments
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {input_file}: {str(e)}")