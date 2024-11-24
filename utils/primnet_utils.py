
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from geopy.distance import geodesic
from shapely.geometry import LineString, Point
import networkx as nx
import pandas as pd
import numpy as np
import json
import os
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.dataloader import Substation
from models.primnet import PrimaryNetworkConfig, optimize_primary_network
from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize(
        log_file_path="logs/test_primnet_utils.log", 
        log_config_path="configs/logging_config.ini"
    )

logger = LogManager.get_logger("primnet_utils")


class PrimaryNetworkGenerator:
    """Class to manage sequential generation of primary networks and results."""
    
    def __init__(self, output_dir: str):
        """Initialize the generator with the output directory

        Args:
            output_dir (str): Directory to save all results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize candidate graph and optimal graph
        self.candidate_graph = nx.MultiGraph()
        self.optimal_graph = nx.Graph()
    
    def _reconstruct_graph_from_dict(
        self,
        partition_data: Dict[str, Dict]
    ) -> nx.MultiGraph:
        """
        Reconstruct a networkx MultiGraph from dictionary containing nodelist and edgelist.
        
        Args:
            partition_data (Dict[str, Dict]): Dictionary with 'nodelist' and 'edgelist' keys
                nodelist: Dict[str, Dict] with node attributes (cord, label, load)
                edgelist: Dict[str, Dict] with edge attributes (highway, geometry, length)
                
        Returns:
            nx.MultiGraph: Reconstructed graph with all attributes
            
        Raises:
            ValueError: If required attributes are missing or invalid
        """
        
        def parse_geometry(geom_str: str) -> LineString:
            """Parse geometry string into LineString object"""
            try:
                # Remove 'LINESTRING' and parentheses
                coords_str = geom_str.replace('LINESTRING (', '').replace(')', '')
                # Split into coordinate pairs
                coords_pairs = coords_str.split(', ')
                # Convert to list of coordinate tuples
                coords = [
                    tuple(map(float, pair.split()))
                    for pair in coords_pairs
                ]
                return LineString(coords)
            except Exception as e:
                raise ValueError(f"Invalid geometry string format: {geom_str}") from e
        
        # Create new graph
        graph = nx.MultiGraph()
        
        # Add nodes with attributes
        for node_id, attrs in partition_data['nodelist'].items():
            # Convert node_id to proper type (int for road nodes, str for transformers)
            if not isinstance(node_id, str) or not node_id.startswith('T'):
                node_id = int(node_id)
                
            # Validate required attributes
            if not all(key in attrs for key in ['cord', 'label', 'load']):
                raise ValueError(f"Missing required attributes for node {node_id}")
                
            # Process attributes
            node_attrs = {
                'cord': tuple(attrs['cord']),  # Ensure cord is tuple
                'label': attrs['label'],
                'load': attrs['load'],
                'distance': attrs['distance']
            }
            
            graph.add_node(node_id, **node_attrs)
        
        # Add edges with attributes
        for edge_key, attrs in partition_data['edgelist'].items():
            # Parse edge key (from_node,to_node,key)
            try:
                from_node, to_node, key = edge_key.strip('()').split(',')
                # Convert node IDs to proper type
                if not from_node.startswith('T'):
                    from_node = int(from_node)
                if not to_node.startswith('T'):
                    to_node = int(to_node)
                key = int(key)  # Convert key to int
            except ValueError as e:
                raise ValueError(f"Invalid edge key format: {edge_key}") from e
            
            # Validate required attributes
            if not all(key in attrs for key in ['highway', 'geometry', 'length']):
                raise ValueError(f"Missing required attributes for edge {edge_key}")
            
            # Process attributes
            edge_attrs = {
                'highway': attrs['highway'],
                'geometry': parse_geometry(attrs['geometry']),
                'length': attrs['length']
            }
            
            graph.add_edge(from_node, to_node, key=key, **edge_attrs)
        
        return graph
    
    def _get_partitions(
        self, 
        graph_list: Union[nx.MultiGraph, List[nx.MultiGraph]], 
        max_load: float
        ):
        """
        This function handles primary network creation for large number of nodes.
        It divides the network into multiple partitions of small networks such that 
        the optimizer can solve an optimization problem for each sub-network.
        
        Args:
            graph_list (Union[nx.MultiGraph, List[nx.MultiGraph]]): list of existing candidate networkx graphs
            max_load (float): maximum MVA load for each partition
        """
        if not isinstance(graph_list, list): graph_list = [graph_list]
        for g in graph_list:
            total_load = 1e-3*sum(nx.get_node_attributes(g,'load').values())
            if total_load < max_load:
                self.candidate_graph = nx.compose(self.candidate_graph, g)
            else:
                comp = nx.algorithms.community.girvan_newman(g)
                nodes_component = [c for c in next(comp)]
                subgraph_list = [nx.subgraph(g, nodelist) for nodelist in nodes_component]
                subgraph_loads = [1e-3*sum(nx.get_node_attributes(sg,'load').values()) for sg in subgraph_list]
                logger.info(f"Graph with load of {total_load} kVA is partitioned to {subgraph_loads}")
                self._get_partitions(subgraph_list, max_load)
        return
        
    def generate_network_for_substation(
        self,
        sub: Substation,
        assignment: Dict[str, Dict],
        config: PrimaryNetworkConfig
        ):
        
        # get the combined road and transformer network mapped to the substation
        total_graph = self._reconstruct_graph_from_dict(assignment)
        total_load = 1e-3*sum(nx.get_node_attributes(total_graph,'load').values())
        
        # create communities using girvan-newman algorithm
        max_feeder_capacity = max([
            config.get("max_feeder_capacity", 1000), 
            total_load/config.get('max_feeder_number', 10)
            ])
        self._get_partitions(total_graph, max_feeder_capacity)
        
        # create optimal network for each partition
        for nodelist in list(nx.connected_components(self.candidate_graph)):
            graph = nx.subgraph(self.candidate_graph, list(nodelist))
        
            # solve for optimal network
            result, feeder_nodes = optimize_primary_network(graph, config)
            
            for n in feeder_nodes:
                result.add_edge(
                    sub.id, n,
                    geometry=LineString((sub.cord, graph.nodes[n]['cord'])),
                    label="feeder",
                    length=geodesic(
                        (sub.cord[1], sub.cord[0]),
                        (graph.nodes[n]['cord'][1], graph.nodes[n]['cord'][0])
                        ).meters
                    )
            
            # Combine the result with the existing optimal network
            self.optimal_graph = nx.compose(self.optimal_graph, result)
        
        # Add the attributes for substation node
        self.optimal_graph.nodes[sub.id]["load"] = 0.0
        self.optimal_graph.nodes[sub.id]["label"] = 'S'
        self.optimal_graph.nodes[sub.id]["cord"] = sub.cord
        return
    
    def export_to_csv(self, prefix: Optional[str] = None):
        """
        Export graph nodes and edges to separate CSV files.
        
        Args:
            graph: networkx graph to save
            prefix: Optional prefix for filenames
            
        Raises:
            OSError: If directory creation or file writing fails
        """
        
        # Check for tree structure and transformer coverage
        assert self.optimal_graph.number_of_nodes() == self.optimal_graph.number_of_edges() + 1
        transformer_candidate = len([n for n,d in self.candidate_graph.nodes(data=True) if d['label']=='T'])
        transformer_optimal_graph = len([n for n,d in self.optimal_graph.nodes(data=True) if d['label']=='T'])
        assert transformer_candidate == transformer_optimal_graph
        
        # Generate filename components
        components = []
        if prefix:
            components.append(prefix)
        
        # Create base filename
        base_filename = '_'.join(components) if components else 'network'
        
        # Export nodes
        nodes_file = self.output_dir / f"{base_filename}_nodes.csv"
        edges_file = self.output_dir / f"{base_filename}_edges.csv"
        
        # Create DataFrames and export
        nodes_df = self._create_nodes_dataframe(self.optimal_graph)
        edges_df = self._create_edges_dataframe(self.optimal_graph)
        
        nodes_df.to_csv(nodes_file, index=False)
        edges_df.to_csv(edges_file, index=False)
        
        return
    
    def _create_nodes_dataframe(self, graph) -> pd.DataFrame:
        """Create DataFrame from graph nodes with all attributes."""
        rows = []
        
        for node in graph.nodes():
            # Start with node ID
            row = {'node_id': node}
            
            # Add all node attributes with proper handling of special types
            attrs = graph.nodes[node]
            for key, value in attrs.items():
                processed_value = self._process_attribute_value(value)
                row[key] = processed_value
            
            rows.append(row)
        
        # Create DataFrame and handle column ordering
        df = pd.DataFrame(rows)
        
        # Ensure node_id is first column
        cols = ['node_id'] + [col for col in df.columns if col != 'node_id']
        df = df[cols]
        
        return df
    
    def _create_edges_dataframe(self, graph) -> pd.DataFrame:
        """Create DataFrame from graph edges with all attributes."""
        rows = []
        
        # Handle both regular and multi-graphs
        if isinstance(graph, nx.MultiGraph):
            edges = graph.edges(keys=True, data=True)
        else:
            edges = graph.edges(data=True)
        
        for edge in edges:
            # Extract edge information based on graph type
            if isinstance(graph, nx.MultiGraph):
                u, v, key, attrs = edge[0], edge[1], edge[2], edge[3]
                row = {
                    'source': u,
                    'target': v,
                    'key': key
                }
            else:
                u, v, attrs = edge
                row = {
                    'source': u,
                    'target': v
                }
            
            # Add all edge attributes with proper handling of special types
            for key, value in attrs.items():
                processed_value = self._process_attribute_value(value)
                row[key] = processed_value
            
            rows.append(row)
        
        # Create DataFrame and handle column ordering
        df = pd.DataFrame(rows)
        
        # Ensure ID columns are first
        id_cols = ['source', 'target']
        if isinstance(graph, nx.MultiGraph):
            id_cols.append('key')
        
        other_cols = [col for col in df.columns if col not in id_cols]
        df = df[id_cols + other_cols]
        
        return df
    
    def _process_attribute_value(self, value: Any) -> Any:
        """Process attribute values for CSV storage."""
        if isinstance(value, (LineString, Point)):
            # Convert geometry objects to WKT
            return value.wkt
        elif isinstance(value, (list, tuple, set)):
            # Convert sequences to JSON
            return json.dumps(list(value))
        elif isinstance(value, dict):
            # Convert dictionaries to JSON
            return json.dumps(value)
        elif isinstance(value, (np.integer, np.floating)):
            # Convert numpy types to Python types
            return value.item()
        elif pd.isna(value):
            # Handle missing values
            return None
        else:
            return value