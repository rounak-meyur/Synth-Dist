from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Set
from shapely.geometry import LineString
import networkx as nx
import pandas as pd
from pathlib import Path

import os
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager
from utils.dataloader import Home
from models.secnet import create_candidate_network, create_secondary_distribution_network

if __name__ == "__main__":
    LogManager.initialize(
        log_file_path="logs/test_secnet_utils.log", 
        log_config_path="configs/logging_config.ini"
    )

logger = LogManager.get_logger("secnet_utils")

@dataclass
class Transformer:
    """Data class to store transformer information from the secondary distribution network."""
    id: str
    cord: Tuple[float, float]
    load: float

class SecondaryNetworkGenerator:
    """Class to manage sequential generation of secondary networks and results."""
    
    def __init__(self, output_dir: str, base_transformer_id: int):
        """
        Initialize the generator with output directory and base transformer ID.
        
        Args:
            output_dir (str): Directory to save all results
            base_transformer_id (int, optional): Starting ID for transformers.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage for results
        self.all_transformers: Dict[str, Transformer] = {}
        self.secondary_edges = []
        self.transformer_road_edges = []
        self._next_transformer_id = base_transformer_id
    
    def _generate_transformer_id(self) -> str:
        """
        Generate a unique transformer ID and increment the counter.
        
        Returns:
            str: Unique transformer ID in format 'T{number}'
        """
        current_id = f"T{self._next_transformer_id}"
        self._next_transformer_id += 1
        return current_id
    
    def _get_tree_children(
        self,
        network: nx.Graph,
        root: str,
        parent: str = None
    ) -> Set[str]:
        """
        Get all children nodes in a tree by traversing from root to leaves.
        Uses DFS traversal considering the tree structure.
        
        Args:
            network (nx.Graph): The network graph
            root (str): Current node being processed
            parent (str, optional): Parent of current node to avoid going backwards
            
        Returns:
            Set[str]: Set of all child nodes in the subtree
        """
        children = set()
        for neighbor in network.neighbors(root):
            if neighbor != parent:  # Don't go back to parent
                children.add(neighbor)
                # Recursively get children of this neighbor
                children.update(self._get_tree_children(network, neighbor, root))
        return children

    def _calculate_transformer_load(
        self,
        network: nx.Graph,
        transformer_node: str
    ) -> float:
        """
        Calculate total load for a transformer by traversing its tree.
        
        Args:
            network (nx.Graph): The network graph
            transformer_node (str): Root transformer node
            
        Returns:
            float: Total load of all home nodes in the transformer's tree
        """
        # Get all nodes in this transformer's tree
        tree_nodes = self._get_tree_children(network, transformer_node)
        
        # Sum loads of all home nodes in the tree
        total_load = sum(
            network.nodes[node]['load']
            for node in tree_nodes
            if network.nodes[node]['label'] == 'H'
        )
        
        return total_load
    
    def generate_network_for_link(
        self,
        road_link: Tuple,
        road_geom: LineString,
        mapped_homes: List[Home],
        **kwargs
    ) -> nx.Graph:
        """
        Generate a secondary network for a road link and update results.
        
        Args:
            road_link (Tuple): Input road link ID
            road_geom (LineString): Geometry for the road link
            mapped_homes (List[Home]): list of homes mapped to road link
            kwargs : Optional arguments
                nearest_homes (Optional[int]): Minimum number of homes to consider for direct transformer connection.
                minimum_separation (Optional[float]): Minimum distance in meters between points on the road edge geometry.
                penalty (Optional[float]): Edge crossing penalty
                max_rating (Optional[float]): Maximum transformer rating
                max_hops (Optional[int]): Maximum hops from transformer
            
        Returns:
            nx.Graph: Generated secondary network
        """
        # Keyword arguments
        nearest_homes = kwargs.get("nearest_homes", None)
        minimum_separation = kwargs.get("separation", 50.0)
        penalty = kwargs.get("penalty", 0.5)
        max_rating = kwargs.get("max_rating", 25e3)
        max_hops = kwargs.get("max_hops", 10)

        # Generate candidate network
        graph, road_nodes = create_candidate_network(
            road_link, road_geom,mapped_homes,
            nearest_homes=nearest_homes,
            minimum_separation=minimum_separation,
            )
        
        # Generate secondary network
        result = create_secondary_distribution_network(
            graph=graph,
            penalty=penalty,
            max_rating=max_rating,
            max_hops=max_hops
        )
        
        # Process and save results
        self._process_network_results(
            network=result,
            road_nodes=road_nodes
        )
        
        return result
    
    def _process_network_results(
        self,
        network: nx.Graph,
        road_nodes: List[str]
    ) -> None:
        """
        Process network results and store transformer and edge information.
        
        Args:
            network (nx.Graph): Generated secondary network
            road_nodes (List[str]): Ordered list of road nodes including potential transformer locations
        """
        # Create mapping of old transformer IDs to new ones
        transformer_id_mapping = {}
        transformer_nodes = set(n for n, d in network.nodes(data=True) if d['label'] == 'T')
        
        # Process each transformer in the network
        for t_node in transformer_nodes:
            # Generate unique transformer ID
            transformer_id = self._generate_transformer_id()
            transformer_id_mapping[t_node] = transformer_id
            
            # Calculate total load for this transformer's tree
            total_load = self._calculate_transformer_load(network, t_node)
            
            # Create and store transformer
            transformer = Transformer(
                id=transformer_id,
                cord=network.nodes[t_node]['cord'],
                load=total_load
            )
            self.all_transformers[transformer_id] = transformer
        
        # Process road nodes and create edges
        updated_road_nodes = []
        for node in road_nodes:
            if node in transformer_nodes:
                # If this node is a selected transformer, use its new ID
                updated_road_nodes.append(transformer_id_mapping[node])
            elif node not in transformer_nodes and not str(node).startswith('T'):
                # If this is a regular road node (not an unselected transformer)
                updated_road_nodes.append(node)
            # Skip unselected transformer nodes
        
        # Create edges from the ordered list of nodes
        road_edges = list(zip(updated_road_nodes[:-1], updated_road_nodes[1:]))
        self.transformer_road_edges.extend(road_edges)
        
        # Store secondary network edges with new transformer IDs
        for u, v, data in network.edges(data=True):
            # Map old node IDs to new transformer IDs if needed
            u_id = transformer_id_mapping.get(u, str(u))
            v_id = transformer_id_mapping.get(v, str(v))
            
            self.secondary_edges.append({
                'from_node': u_id,
                'to_node': v_id,
                'length': data['length'],
            })
    
    def save_results(self, prefix: str = "network") -> None:
        """
        Append results to existing CSV files or create new ones if they don't exist.
        
        Args:
            prefix (str, optional): Prefix for output files. Defaults to "network"
        """
        def save_df_with_append(df: pd.DataFrame, filepath: Path, index: bool = False) -> None:
            """
            Save DataFrame to CSV, appending if file exists, creating new if it doesn't.
            
            Args:
                df (pd.DataFrame): DataFrame to save
                filepath (Path): Path to save the CSV file
                index (bool, optional): Whether to write row indices. Defaults to False.
            """
            if not filepath.exists():
                # If file doesn't exist, save with header
                df.to_csv(filepath, index=index, mode='w')
            else:
                # If file exists, append without header
                df.to_csv(filepath, index=index, mode='a', header=False)

        # Save transformers
        transformer_data = [
            {
                'transformer_id': t.id,
                'longitude': t.cord[0],
                'latitude': t.cord[1],
                'total_load': t.load
            }
            for t in self.all_transformers.values()
        ]
        if transformer_data:  # Only save if there are transformers
            transformer_df = pd.DataFrame(transformer_data)
            transformer_file = self.output_dir / f"{prefix}_transformers.csv"
            save_df_with_append(transformer_df, transformer_file)
        
        # Save secondary network edges
        if self.secondary_edges:  # Only save if there are edges
            edge_df = pd.DataFrame(self.secondary_edges)
            edge_file = self.output_dir / f"{prefix}_secondary_edges.csv"
            save_df_with_append(edge_df, edge_file)
        
        # Save road network edges with transformers
        if self.transformer_road_edges:  # Only save if there are edges
            road_edge_data = [
                {'from_node': u, 'to_node': v}
                for u, v in self.transformer_road_edges
            ]
            road_edge_df = pd.DataFrame(road_edge_data)
            road_edge_file = self.output_dir / f"{prefix}_road_transformer_edges.csv"
            save_df_with_append(road_edge_df, road_edge_file)
        
        # Clear the current data after saving
        self.all_transformers.clear()
        self.secondary_edges.clear()
        self.transformer_road_edges.clear()

# Example usage:
"""
# Initialize generator
generator = SecondaryNetworkGenerator(
    output_dir="output/networks",
    base_transformer_id=1001
)

# Process road link with ordered nodes
road_nodes = ['R1', 'T1', 'T2', 'R2', 'T3', 'R3']  # Original road nodes with potential transformers
graph = create_input_graph(...)  # Create input graph

result = generator.generate_network(
    graph=graph,
    penalty=1.0,
    max_rating=100,
    max_hops=5,
    road_link_id='RL001',
    road_nodes=road_nodes
)

# If only T1 and T3 are selected in the optimization:
# updated_road_nodes would be ['R1', 'T1001', 'R2', 'T1002', 'R3']
# transformer_road_edges would contain: [('R1', 'T1001'), ('T1001', 'R2'), ('R2', 'T1002'), ('T1002', 'R3')]

generator.save_results(prefix="region1")
"""