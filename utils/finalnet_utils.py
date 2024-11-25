from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
import networkx as nx
import pandas as pd
from shapely.geometry import LineString, Point
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
        log_file_path="logs/test_finalnet_utils.log", 
        log_config_path="configs/logging_config.ini"
    )

logger = LogManager.get_logger("finalnet_utils")

def _parse_cord_string(cord_str: str) -> Tuple[float, float]:
    """
    Parse coordinate string into a tuple of floats.
    
    Args:
        cord_str (str): String representation of coordinates
        
    Returns:
        Tuple[float, float]: Tuple of (longitude, latitude)
    """
    try:
        # Remove brackets and split
        cord_str = cord_str.strip('[]()').replace(' ', '')
        lon, lat = map(float, cord_str.split(','))
        return (lon, lat)
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing coordinate string: {cord_str}")
        raise ValueError(f"Invalid coordinate format: {cord_str}") from e

def _create_edge_geometry(node1_cord: Tuple[float, float], node2_cord: Tuple[float, float]) -> LineString:
    """
    Create a LineString geometry from two node coordinates.
    
    Args:
        node1_cord (Tuple[float, float]): Coordinates of first node
        node2_cord (Tuple[float, float]): Coordinates of second node
        
    Returns:
        LineString: LineString geometry connecting the nodes
    """
    return LineString([node1_cord, node2_cord])

def combine_networks(
    sub_ids: List[str], 
    region_name: str, 
    homes: List[Home], 
    prim_dir: str,
    sec_dir: str
    ) -> nx.Graph:
    """
    Combine primary and secondary distribution networks into a single graph.
    
    Args:
        sub_ids (List[str]): List of substation IDs
        region_name (str): Name of the region for secondary network files
        homes (List[Home]): List of Home objects containing coordinate information
        prim_dir (str): Directory containing primary network files
        sec_dir (str): Directory containing secondary network files
        
    Returns:
        nx.Graph: Combined network graph
        
    Raises:
        FileNotFoundError: If required files are not found
        ValueError: If data format is invalid
    """
    logger.info(f"Combining networks for {len(sub_ids)} substations in region {region_name}")
    
    # Initialize combined graph
    combined_graph = nx.Graph()
    
    # Create home ID to coordinates mapping
    home_coords = {str(home.id): home.cord for home in homes}
    
    # First, load and combine primary networks for each substation
    for sub_id in sub_ids:
        node_file = Path(prim_dir) / f"{sub_id}_nodes.csv"
        edge_file = Path(prim_dir) / f"{sub_id}_edges.csv"
        
        if not node_file.exists() or not edge_file.exists():
            logger.warning(f"Files not found for substation {sub_id}")
            continue
            
        try:
            # Load nodes
            nodes_df = pd.read_csv(node_file)
            for _, row in nodes_df.iterrows():
                node_id = str(row['node_id'])
                # Parse coordinates if they exist
                try:
                    cord = _parse_cord_string(str(row['cord']))
                except ValueError:
                    logger.warning(f"Invalid coordinate format for node {node_id}")
                    continue
                
                # Add node with attributes
                combined_graph.add_node(
                    node_id,
                    cord=cord,
                    label=row['label']
                )
            
            # Load edges
            edges_df = pd.read_csv(edge_file)
            for _, row in edges_df.iterrows():
                source = str(row['source'])
                target = str(row['target'])
                
                if source in combined_graph and target in combined_graph:
                    # Get node coordinates
                    source_cord = combined_graph.nodes[source]['cord']
                    target_cord = combined_graph.nodes[target]['cord']
                    
                    # Create geometry
                    geometry = _create_edge_geometry(source_cord, target_cord)
                    
                    # Add edge with attributes
                    combined_graph.add_edge(
                        source,
                        target,
                        geometry=geometry,
                        label=row['label'],
                        length=row['length']
                    )
                
        except Exception as e:
            logger.error(f"Error processing files for substation {sub_id}: {str(e)}")
            continue
    
    # Load secondary network
    secondary_file = Path(sec_dir) / f"{region_name}_secondary_edges.csv"
    
    if secondary_file.exists():
        try:
            secondary_df = pd.read_csv(secondary_file)
            
            # Process secondary edges
            for _, row in secondary_df.iterrows():
                from_node = str(row['from_node'])
                to_node = str(row['to_node'])
                
                # Add nodes if they don't exist (should be homes)
                for node in [from_node, to_node]:
                    if node not in combined_graph:
                        if node in home_coords:
                            combined_graph.add_node(
                                node,
                                cord=home_coords[node],
                                label='H'
                            )
                        else:
                            logger.warning(f"Home coordinates not found for node {node}")
                            continue
                
                # Add edge if both nodes exist
                if from_node in combined_graph and to_node in combined_graph:
                    # Get node coordinates
                    from_cord = combined_graph.nodes[from_node]['cord']
                    to_cord = combined_graph.nodes[to_node]['cord']
                    
                    # Create geometry
                    geometry = _create_edge_geometry(from_cord, to_cord)
                    
                    # Add edge with attributes
                    combined_graph.add_edge(
                        from_node,
                        to_node,
                        geometry=geometry,
                        label='secondary',
                        length=row['length']
                    )
                    
        except Exception as e:
            logger.error(f"Error processing secondary network file: {str(e)}")
    else:
        logger.warning(f"Secondary network file not found: {secondary_file}")
    
    # Log summary statistics
    logger.info(f"Combined network has {combined_graph.number_of_nodes()} nodes and {combined_graph.number_of_edges()} edges")
    
    return combined_graph

def save_combined_network(
    graph: nx.Graph,
    region_name: str,
    out_dir: str = None
) -> None:
    """
    Save the combined network to CSV files.
    
    Args:
        graph (nx.Graph): Combined network graph
        region_name (str): Name of the region for output files
        out_dir (str): Output directory
    """
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save nodes
    nodes_data = []
    for node, attrs in graph.nodes(data=True):
        nodes_data.append({
            'node_id': node,
            'cord': attrs['cord'],
            'label': attrs['label']
        })
    
    
    pd.DataFrame(nodes_data).to_csv(
        out_path / f"{region_name}_combined_nodes.csv",
        index=False
    )
    
    # Save edges
    edges_data = []
    for u, v, attrs in graph.edges(data=True):
        edges_data.append({
            'source': u,
            'target': v,
            'geometry': attrs['geometry'].wkt,
            'label': attrs['label'],
            'length': attrs['length']
        })
    
    
    pd.DataFrame(edges_data).to_csv(
        out_path / f"{region_name}_combined_edges.csv",
        index=False
    )
    
    logger.info(f"Saved combined network files for region {region_name}")