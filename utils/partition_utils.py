from typing import Dict, List, Tuple, Callable
import networkx as nx
import numpy as np
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
            logger.error(err_msg)
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
        Assign transformers to nearest substation based on network distance.
        
        Args:
            substation_nodes (Dict[int, int]): Mapping of substation ID to road node ID
            
        Returns:
            Dict[int, List[str]]: Mapping of substation ID to list of assigned transformer IDs
        """
        # Get all transformer nodes
        transformer_nodes = [
            node for node, attrs in self.network.nodes(data=True)
            if attrs['label'] == 'T'
        ]
        
        # Initialize assignments dictionary
        assignments = {sub_id: [] for sub_id in substation_nodes.keys()}
        
        # Calculate shortest paths from each transformer to all substation nodes
        for transformer in transformer_nodes:
            min_distance = float('inf')
            assigned_substation = None
            
            # Calculate shortest path to each substation
            for sub_id, road_node in substation_nodes.items():
                try:
                    path_length = nx.shortest_path_length(
                        self.network,
                        source=transformer,
                        target=road_node,
                        weight='length'
                    )
                    
                    if path_length < min_distance:
                        min_distance = path_length
                        assigned_substation = str(sub_id)
                
                except nx.NetworkXNoPath:
                    logger.error(f"No path exists between {road_node} and {transformer}")
                    continue
            
            if assigned_substation is not None:
                assignments[assigned_substation].append(transformer)
            else:
                err_msg = f"No valid path found from transformer {transformer} to any substation"
                logger.error(err_msg)
                raise ValueError(err_msg)
        
        return assignments
    
    def save_partitioning(
        self,
        assignments: Dict[int, List[str]],
        output_file: str
    ) -> None:
        """
        Save transformer-substation assignments to JSON file.
        
        Args:
            assignments (Dict[int, List[str]]): Mapping of substation IDs to transformer lists
            output_file (str): Path to output JSON file
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert all keys to strings for JSON compatibility
        json_data = {str(sub_id): transformers 
                    for sub_id, transformers in assignments.items()}
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)

def load_partitioning(
    input_file: str
    ) -> Dict[int, List[str]]:
    """
    Load transformer-substation assignments from JSON file.
    
    Args:
        input_file (str): Path to input JSON file
        
    Returns:
        Dict[int, List[str]]: Mapping of substation IDs (int) to transformer IDs (str)
        
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
        for sub_id_str, transformers in json_data.items():
            # Validate substation ID
            try:
                sub_id = int(sub_id_str)
            except ValueError:
                raise ValueError(
                    f"Invalid substation ID format: {sub_id_str}. "
                    "Expected integer."
                )
            
            # Validate transformer IDs
            if not isinstance(transformers, list):
                raise ValueError(
                    f"Invalid transformer list for substation {sub_id}: "
                    f"{transformers}. Expected list."
                )
            
            # Verify transformer format
            for t_id in transformers:
                if not isinstance(t_id, str) or not t_id.startswith('T'):
                    raise ValueError(
                        f"Invalid transformer ID format: {t_id}. "
                        "Expected string starting with 'T'."
                    )
            
            assignments[sub_id] = transformers
        
        return assignments
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {input_file}: {str(e)}")