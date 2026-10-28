from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Set, Union, Any
from shapely.geometry import LineString, Point
import shapely.wkt as wkt
import networkx as nx
from geopy.distance import geodesic
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
from models.secnet import create_candidate_network, create_secondary_distribution_network, minimum_transformers
from models.secnet_heuristic import create_secondary_distribution_network_heuristic

SECNET_METHODS = ("milp", "heuristic")

if __name__ == "__main__":
    LogManager.initialize(
        log_file_path="logs/test_secnet_utils.log", 
        log_config_path="configs/logging_config.ini"
    )

logger = LogManager.get_logger("secnet_utils")

def get_road_link_geometry(road_network: nx.MultiGraph, road_link: Tuple) -> LineString:
    """
    Get the geometry of a road link oriented to start at its first node.

    The road network is undirected, so a stored geometry may run from v to u. The
    secondary network places transformers along the geometry and records them in the
    order u -> T... -> v, so the geometry must start at u.

    Args:
        road_network (nx.MultiGraph): Road network with node attributes 'x' and 'y'
        road_link (Tuple): Road link (u, v, key)

    Returns:
        LineString: Geometry of the road link starting at u

    Raises:
        KeyError: If the road link or its geometry does not exist
    """
    geometry = road_network.edges[road_link]['geometry']
    if geometry is None:
        raise KeyError(f"Road link {road_link} has no geometry")
    u = road_link[0]
    start = Point(road_network.nodes[u]['x'], road_network.nodes[u]['y'])
    coords = list(geometry.coords)
    if start.distance(Point(coords[-1])) < start.distance(Point(coords[0])):
        return LineString(coords[::-1])
    return geometry

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

        # Track files created in this session
        self._session_files = set()
    
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
                method (Optional[str]): "milp" (default) or "heuristic"
                parameters (Optional[dict]): Parameters for the selected method
                    milp: solver, verbose, time_limit, warm_start, relative_gap
                    heuristic: max_iterations, local_search

        Returns:
            nx.Graph: Generated secondary network
        """
        # Keyword arguments
        nearest_homes = kwargs.get("nearest_homes", None)
        minimum_separation = kwargs.get("separation", 50.0)
        penalty = kwargs.get("penalty", 0.5)
        max_rating = kwargs.get("max_rating", 25e3)
        max_hops = kwargs.get("max_hops", 10)
        method = str(kwargs.get("method", "milp")).lower()
        parameters = kwargs.get("parameters") or {}

        if method not in SECNET_METHODS:
            raise ValueError(f"Unknown secondary network method '{method}'. Choose from {SECNET_METHODS}.")

        # Enough candidate transformers to serve all homes even on short, crowded road links.
        # When more than one transformer is needed, one spare candidate leaves the solver room
        # to find a packing of the home loads that respects max_rating.
        required_transformers = minimum_transformers([home.load for home in mapped_homes], max_rating)
        min_transformers = required_transformers + 1 if required_transformers > 1 else 1

        # Generate candidate network
        graph, road_nodes = create_candidate_network(
            road_link, road_geom,mapped_homes,
            nearest_homes=nearest_homes,
            minimum_separation=minimum_separation,
            min_transformers=min_transformers,
            )
        
        # Generate secondary network
        create_network = (
            create_secondary_distribution_network_heuristic if method == "heuristic"
            else create_secondary_distribution_network
        )
        result = create_network(
            graph=graph,
            penalty=penalty,
            max_rating=max_rating,
            max_hops=max_hops,
            **parameters
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
    
    def _save_road_transformer_sequence(
        self, 
        filepath: Path, 
        road_link_id: Tuple[int, int, str], 
        nodes: List[Union[int, str]]
    ) -> None:
        """
        Save road transformer sequence to a text file.
        If first save in session, overwrite file; otherwise append.
        
        Args:
            filepath (Path): Path to the text file
            road_link_id (Tuple[int, int, str]): Tuple of (start_node, end_node, edge_key)
                where start_node and end_node are integers
            nodes (List[Union[int, str]]): Ordered list of nodes including road nodes (int) 
                and transformer nodes (str)
        """
        mode = 'w' if filepath not in self._session_files else 'a'
        with open(filepath, mode, encoding='utf-8') as f:
            # Format road link ID components with edge key
            start_node, end_node, edge_key = road_link_id
            # Convert node list to strings, road nodes are integers
            node_sequence = ' '.join(str(n) for n in nodes)
            f.write(f"{start_node} {end_node} {edge_key} {node_sequence}\n")
        
        # Track file creation in this session
        self._session_files.add(filepath)

    def _save_dataframe(self, df: pd.DataFrame, filepath: Path) -> None:
        """
        Save DataFrame to CSV, overwriting if first time in session, appending otherwise.
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filepath (Path): Path to save the CSV file
        """
        if filepath not in self._session_files:
            # First time in this session - overwrite the file
            df.to_csv(filepath, index=False, mode='w')
            self._session_files.add(filepath)
        else:
            # Already created in this session - append without header
            df.to_csv(filepath, index=False, mode='a', header=False)

    def save_results(
        self, 
        prefix: str = "network", 
        road_link_id: Tuple[int, int, str] = None
    ) -> None:
        """
        Save results to files. For each file:
        - If first save in this session: overwrite existing file
        - If subsequent save in this session: append to file
        
        Args:
            prefix (str, optional): Prefix for output files. Defaults to "network"
            road_link_id (Tuple[int, int, str]): Tuple of (start_node, end_node, edge_key)
                where start_node and end_node are integers
        """
        # Save transformers
        transformer_file = self.output_dir / f"{prefix}_transformers.csv"
        if self.all_transformers:
            transformer_data = [
                {
                    'transformer_id': t.id,
                    'longitude': t.cord[0],
                    'latitude': t.cord[1],
                    'total_load': t.load
                }
                for t in self.all_transformers.values()
            ]
            transformer_df = pd.DataFrame(transformer_data)
            self._save_dataframe(transformer_df, transformer_file)
        
        # Save secondary network edges
        edge_file = self.output_dir / f"{prefix}_secondary_edges.csv"
        if self.secondary_edges:
            # Include road link components in the edge data
            if road_link_id:
                start_node, end_node, edge_key = road_link_id
                for edge in self.secondary_edges:
                    edge['road_start'] = int(start_node)
                    edge['road_end'] = int(end_node)
                    edge['road_edge_key'] = edge_key
            
            edge_df = pd.DataFrame(self.secondary_edges)
            self._save_dataframe(edge_df, edge_file)
        
        # Save road transformer sequence
        road_seq_file = self.output_dir / f"{prefix}_road_transformer_edges.txt"
        if road_link_id and self.transformer_road_edges:
            # Get ordered list of nodes for this road link
            nodes = []
            for u, v in self.transformer_road_edges:
                if not nodes:  # First edge
                    nodes.extend([u, v])
                else:  # Subsequent edges
                    nodes.append(v)
            
            # Save to text file
            self._save_road_transformer_sequence(road_seq_file, road_link_id, nodes)
        
        # Clear the current data after saving
        self.all_transformers.clear()
        self.secondary_edges.clear()
        self.transformer_road_edges.clear()

    def reset_session(self) -> None:
        """
        Reset the session tracking for file creation.
        Next save will overwrite existing files.
        """
        self._session_files.clear()

    def _result_files(self, prefix: str) -> Dict[str, Path]:
        return {
            'transformers': self.output_dir / f"{prefix}_transformers.csv",
            'edges': self.output_dir / f"{prefix}_secondary_edges.csv",
            'sequence': self.output_dir / f"{prefix}_road_transformer_edges.txt",
            'combined_nodes': self.output_dir / f"{prefix}_combined_network_nodes.csv",
            'combined_edges': self.output_dir / f"{prefix}_combined_network_edges.csv",
        }

    def clear_results(self, prefix: str) -> None:
        """Delete all secondary network outputs for a prefix and start a new session."""
        for path in self._result_files(prefix).values():
            path.unlink(missing_ok=True)
        self.reset_session()

    def resume(self, prefix: str) -> Set[Tuple[int, int, int]]:
        """
        Resume an interrupted run from the per-link results saved by `save_results`.

        A road link counts as finished once its line in the road transformer sequence file
        is complete (that file is written last). Rows that an interrupted link wrote to the
        transformer or secondary edge files are dropped, the transformer ID counter continues
        after the highest saved ID, and later saves append to the existing files.

        Args:
            prefix (str): Prefix of the output files

        Returns:
            Set[Tuple[int, int, int]]: Road links (start_node, end_node, edge_key) already finished
        """
        files = self._result_files(prefix)
        for name in ('combined_nodes', 'combined_edges'):
            files[name].unlink(missing_ok=True)  # rebuilt by combine_networks
        if not files['sequence'].exists():
            self.clear_results(prefix)
            return set()

        completed: Dict[Tuple[int, int, int], List[str]] = {}
        valid_lines = []
        for line in files['sequence'].read_text().splitlines(keepends=True):
            parts = line.split()
            if not line.endswith('\n') or len(parts) < 5:
                continue  # partially written line
            try:
                road_link = (int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError:
                continue
            completed[road_link] = parts[3:]
            valid_lines.append(line)
        files['sequence'].write_text(''.join(valid_lines))

        transformer_ids = {n for nodes in completed.values() for n in nodes if n.startswith('T')}
        if files['transformers'].exists():
            df = pd.read_csv(files['transformers'], dtype={'transformer_id': str})
            df[df['transformer_id'].isin(transformer_ids)].to_csv(files['transformers'], index=False)
        if files['edges'].exists():
            df = pd.read_csv(files['edges'], dtype={'from_node': str, 'to_node': str})
            done = [
                (int(u), int(v), int(k)) in completed
                for u, v, k in zip(df['road_start'], df['road_end'], df['road_edge_key'])
            ]
            df[done].to_csv(files['edges'], index=False)

        if transformer_ids:
            self._next_transformer_id = max(
                self._next_transformer_id, max(int(t[1:]) for t in transformer_ids) + 1
            )
        self.reset_session()
        self._session_files.update(
            files[name] for name in ('transformers', 'edges', 'sequence') if files[name].exists()
        )
        logger.info(f"Resuming secondary network generation: {len(completed)} road links already finished")
        return set(completed)

    def combine_networks(
        self,
        road_network: nx.MultiGraph,
        prefix: str = "network"
    ) -> nx.MultiGraph:
        """
        Combine road network with transformer nodes by replacing road edges with 
        transformer-connected edges.
        
        Args:
            road_network (nx.MultiGraph): Original road network
            prefix (str): Prefix used for input/output files
            
        Returns:
            nx.MultiGraph: Combined network with transformer nodes and edges
        """
        transformer_edges_file = self.output_dir / f"{prefix}_road_transformer_edges.txt"
        transformers_file = self.output_dir / f"{prefix}_transformers.csv"
        
        # Create new graph
        new_graph = road_network.copy()
        
        # First, add all road network nodes with standardized attributes
        for node, attrs in road_network.nodes(data=True):
            new_graph.add_node(
                node,
                cord=(attrs['x'], attrs['y']),  # Convert x,y to cord
                label='R',  # Road node
                load=0.0    # Road nodes have zero load
            )
        
        # Read transformer information
        transformer_info = {}
        if transformers_file.exists():
            df_transformers = pd.read_csv(transformers_file)
            transformer_info = {
                row['transformer_id']: {
                    'cord': (row['longitude'], row['latitude']),
                    'load': row['total_load']
                }
                for _, row in df_transformers.iterrows()
            }
        
        # Read and process transformer edge sequences
        if transformer_edges_file.exists():
            with open(transformer_edges_file, 'r') as f:
                for line in f:
                    # Parse line
                    parts = line.strip().split()
                    if len(parts) < 4:  # Skip invalid lines
                        continue
                    
                    # Extract road link information
                    start_node = int(parts[0])
                    end_node = int(parts[1])
                    edge_key = int(parts[2])
                    node_sequence = [
                        int(n) if not n.startswith('T') else n 
                        for n in parts[3:]
                    ]
                    
                    # Get original edge attributes
                    if not road_network.has_edge(start_node, end_node, key=edge_key):
                        continue
                    
                    edge_attrs = road_network.get_edge_data(start_node, end_node, edge_key)
                    highway_type = edge_attrs.get('highway', 'unclassified')
                    
                    # Remove original edge if it exists in new graph
                    if new_graph.has_edge(start_node, end_node, key=edge_key):
                        new_graph.remove_edge(start_node, end_node, key=edge_key)
                    
                    # Add transformer nodes with standardized attributes
                    for node in node_sequence:
                        if isinstance(node, str) and node.startswith('T'):
                            if node in transformer_info:
                                new_graph.add_node(
                                    node,
                                    cord=transformer_info[node]['cord'],
                                    label='T',  # Transformer node
                                    load=transformer_info[node]['load']
                                )
                    
                    # Add new edges between consecutive nodes
                    for i in range(len(node_sequence) - 1):
                        u, v = node_sequence[i], node_sequence[i + 1]
                        
                        # Get coordinates for nodes
                        u_cord = new_graph.nodes[u]['cord']
                        v_cord = new_graph.nodes[v]['cord']
                        
                        # Create geometry and calculate length
                        if u_cord and v_cord:
                            # Create LineString geometry
                            geometry = LineString([u_cord, v_cord])
                            
                            # Calculate geodesic length
                            length = geodesic(
                                (u_cord[1], u_cord[0]),  # lat, lon
                                (v_cord[1], v_cord[0])   # lat, lon
                            ).meters
                            
                            # Add edge with attributes
                            new_graph.add_edge(
                                u, v,
                                geometry=geometry,
                                highway=highway_type,
                                length=length
                            )
        
        # Connect components
        connected_new_graph = self.connect_network_components(new_graph)
        
        # Save the combined network
        self.save_combined_network(connected_new_graph, prefix)
        
        return connected_new_graph
    
    def connect_network_components(self, graph: nx.MultiGraph) -> nx.MultiGraph:
        """
        Connect components in the network, removing components without transformers
        and connecting remaining components to the largest component.
        
        Args:
            graph (nx.MultiGraph): Input network
            
        Returns:
            nx.MultiGraph: Connected network
        """
        # Get connected components
        components = list(nx.connected_components(graph))
        if len(components) == 1:
            logger.info("The combined network is a single connected component.")
            return graph  # Already connected
            
        # Create a copy to modify
        new_graph = graph.copy()
        
        # Filter components that have transformers
        valid_components = []
        for component in components:
            has_transformer = any(
                new_graph.nodes[n]['label'] == 'T' 
                for n in component
            )
            if has_transformer:
                valid_components.append(component)
            else:
                # Remove nodes from components without transformers
                new_graph.remove_nodes_from(component)
        logger.info("Removed unnecessary components without transformer nodes")
        
        if len(valid_components) <= 1:
            logger.info("Found single connected component after removal of unnecessary components")
            return new_graph  # No need to connect if only one valid component
            
        # Find the largest component
        largest_component = max(valid_components, key=len)
        other_components = [c for c in valid_components if c != largest_component]
        
        # Function to get road nodes from a component
        def get_road_nodes(component):
            return [n for n in component if new_graph.nodes[n]['label'] == 'R']
        
        # Function to calculate edge length between nodes
        def calculate_edge_length(u, v):
            u_cord = new_graph.nodes[u]['cord']
            v_cord = new_graph.nodes[v]['cord']
            # Calculate geodesic length
            length = geodesic(
                (u_cord[1], u_cord[0]),  # lat, lon
                (v_cord[1], v_cord[0])   # lat, lon
            ).meters
            return length
            
        # Function to create edge geometry
        def create_edge_geometry(u, v):
            u_cord = new_graph.nodes[u]['cord']
            v_cord = new_graph.nodes[v]['cord']
            return LineString([u_cord, v_cord])
        
        # Get road nodes for the largest component
        largest_road_nodes = get_road_nodes(largest_component)
        
        # Connect each other component to the largest component
        for component in other_components:
            component_road_nodes = get_road_nodes(component)
            
            # Find the minimum length connection
            min_length = float('inf')
            best_connection = None
            
            for u in largest_road_nodes:
                for v in component_road_nodes:
                    length = calculate_edge_length(u, v)
                    if length < min_length:
                        min_length = length
                        best_connection = (u, v, length)
            
            if best_connection:
                u, v, length = best_connection
                # Add the connecting edge with attributes
                geometry = create_edge_geometry(u, v)
                logger.info(f"Adding edge {(u,v)} to connect components")
                new_graph.add_edge(
                    u, v,
                    geometry=geometry,
                    highway='unclassified',  # Default type for connecting edges
                    length=length
                )
                
        # Verify the graph is now connected
        if not nx.is_connected(new_graph):
            error_message = "Failed to create a connected network after adding connecting edges"
            logger.error(error_message)
            raise ValueError(error_message)
        
        return new_graph

    def save_combined_network(self, graph: nx.MultiGraph, prefix: str) -> None:
        """
        Save combined network nodes and edges to separate CSV files.
        
        Args:
            graph (nx.MultiGraph): Combined network graph
            prefix (str): Prefix for output files
        """
        # Save nodes
        node_data = []
        for node, attrs in graph.nodes(data=True):
            data = {'node_id': node}
            if isinstance(node, str) and node.startswith('T'):
                # Transformer node - already has cord attribute
                data.update({
                    'longitude': attrs['cord'][0],
                    'latitude': attrs['cord'][1],
                    'load': attrs['load'],
                    'label': 'T'  # Use 'T' for transformer nodes
                })
            else:
                # Road node - convert x,y to cord
                data.update({
                    'longitude': attrs.get('x'),
                    'latitude': attrs.get('y'),
                    'load': attrs.get('load', 0.0),
                    'label': 'R'  # Use 'R' for road nodes
                })
            node_data.append(data)
        
        pd.DataFrame(node_data).to_csv(
            self.output_dir / f"{prefix}_combined_network_nodes.csv",
            index=False
        )
        
        # Save edges
        edge_data = []
        for u, v, k, attrs in graph.edges(data=True, keys=True):
            edge_data.append({
                'from_node': u,
                'to_node': v,
                'key': k,
                'highway': attrs.get('highway', 'unclassified'),
                'length': attrs.get('length', 0.0),
                'geometry': attrs.get('geometry', None)
            })
        
        pd.DataFrame(edge_data).to_csv(
            self.output_dir / f"{prefix}_combined_network_edges.csv",
            index=False
        )

def load_combined_network(
    nodes_file: str, 
    edges_file: str
    ) -> nx.MultiGraph:
    """
    Load combined network from saved CSV files with all attributes.
    
    Args:
        nodes_file (str): CSV file with node attributes
        edges_file (str): CSV file with edge attributes
        
    Returns:
        nx.MultiGraph: Reconstructed network with all attributes
        
    Raises:
        FileNotFoundError: If either nodes or edges file is not found
    """
    
    if not Path(nodes_file).exists():
        raise FileNotFoundError(f"Nodes file not found: {nodes_file}")
    if not Path(edges_file).exists():
        raise FileNotFoundError(f"Edges file not found: {edges_file}")
    
    # Create new graph
    graph = nx.MultiGraph()
    
    # Load nodes
    # Read IDs as strings so road (int) and transformer ('T...') IDs are parsed the same way
    df_nodes = pd.read_csv(nodes_file, dtype={'node_id': str})
    for _, row in df_nodes.iterrows():
        node_id = _parse_node_id(row['node_id'])
        
        # Create standardized node attributes (same as combine_networks)
        node_attrs = {
            'cord': (float(row['longitude']), float(row['latitude'])),
            'load': float(row['load']),
            'label': row['label']  # 'R' for road nodes, 'T' for transformer nodes
        }
        if node_attrs['label'] == 'R':
            node_attrs['x'], node_attrs['y'] = node_attrs['cord']
        
        graph.add_node(node_id, **node_attrs)
    
    # Load edges
    df_edges = pd.read_csv(edges_file, dtype={'from_node': str, 'to_node': str})
    for _, row in df_edges.iterrows():
        # Convert node IDs to proper type
        from_node = _parse_node_id(row['from_node'])
        to_node = _parse_node_id(row['to_node'])
        key = int(row['key'])
        
        # Handle geometry conversion from WKT string to LineString
        geometry_str = row['geometry']
        if isinstance(geometry_str, str) and geometry_str not in ('', 'None'):
            geometry = wkt.loads(geometry_str)
        else:
            # If geometry is missing, create from node coordinates
            from_cord = graph.nodes[from_node]['cord']
            to_cord = graph.nodes[to_node]['cord']
            if from_cord and to_cord:
                geometry = LineString([from_cord, to_cord])
            else:
                geometry = None
        
        # Add edge with attributes
        graph.add_edge(
            from_node,
            to_node,
            key=key,
            geometry=geometry,
            highway=row['highway'] if isinstance(row['highway'], str) else 'unclassified',
            length=float(row['length'])
        )
    
    return graph

def merge_combined_networks(
    regions: List[str],
    secnet_dir: str
    ) -> nx.MultiGraph:
    """
    Merge the combined road and transformer networks of several regions into one network.

    Road networks of neighbouring regions overlap near their border and share OSM node IDs,
    so the merged network is connected across regions. A road link that one region replaced
    by its transformer sequence can still be a plain road edge in a neighbouring region (no
    homes of that region along it); such plain edges are removed so that only the road with
    the transformers on it remains.

    Args:
        regions (List[str]): Regions (prefixes of the secondary network files)
        secnet_dir (str): Directory with <region>_combined_network_nodes.csv / _edges.csv
            and <region>_road_transformer_edges.txt

    Returns:
        nx.MultiGraph: Merged network with the attributes of load_combined_network
    """
    merged = nx.MultiGraph()
    replaced = set()
    for region in regions:
        graph = load_combined_network(
            nodes_file=str(Path(secnet_dir) / f"{region}_combined_network_nodes.csv"),
            edges_file=str(Path(secnet_dir) / f"{region}_combined_network_edges.csv"),
        )
        merged.add_nodes_from(graph.nodes(data=True))
        # keep the keys so that a road edge present in several regions is added once
        merged.add_edges_from(graph.edges(keys=True, data=True))

        sequence_file = Path(secnet_dir) / f"{region}_road_transformer_edges.txt"
        if sequence_file.exists():
            for line in sequence_file.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 4:
                    replaced.add((int(parts[0]), int(parts[1]), int(parts[2])))

    removed = 0
    for u, v, k in replaced:
        if merged.has_edge(u, v, key=k):
            merged.remove_edge(u, v, key=k)
            removed += 1
    logger.info(
        f"Merged networks of {len(regions)} regions: {merged.number_of_nodes()} nodes, "
        f"{merged.number_of_edges()} edges ({removed} plain road edges replaced by transformers)"
    )
    return merged

def _parse_node_id(node_id: str) -> Union[int, str]:
    """Road node IDs are integers, transformer IDs are strings starting with 'T'."""
    return node_id if node_id.startswith('T') else int(node_id)

def _get_node_coord(
    graph: nx.MultiGraph,
    node: Union[int, str]
) -> Tuple[float, float]:
    """
    Get coordinate tuple for a node from its attributes.
    
    Args:
        graph (nx.MultiGraph): Network graph
        node (Union[int, str]): Node identifier
        
    Returns:
        Tuple[float, float]: (longitude, latitude) coordinates or None if not found
    """
    attrs = graph.nodes[node]
    if isinstance(node, str) and node.startswith('T'):
        return attrs.get('cord')
    else:
        x = attrs.get('x')
        y = attrs.get('y')
        if x is not None and y is not None:
            return (x, y)
    return None