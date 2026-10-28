"""
Combine the primary networks (one per substation) and the secondary network into the
final synthetic distribution network of a region.

Node labels: 'S' substation, 'R' road node, 'T' transformer, 'H' home.
Edge labels: 'feeder' (substation to feeder road node), 'primary', 'secondary'.
"""
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import json
import os
import sys

import networkx as nx
import pandas as pd
import shapely.wkt as wkt
from shapely.geometry import LineString

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


def _parse_cord(value: Any) -> Tuple[float, float]:
    """Parse a coordinate saved as a JSON list "[lon, lat]" (or a tuple string "(lon, lat)")."""
    try:
        if isinstance(value, str):
            value = json.loads(value.replace("(", "[").replace(")", "]"))
        lon, lat = value
        return (float(lon), float(lat))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid coordinate format: {value}") from e


def _parse_geometry(value: Any, u_cord, v_cord) -> LineString:
    """Parse a WKT geometry; missing geometries become straight lines between the end nodes."""
    if isinstance(value, str) and value not in ("", "None"):
        return wkt.loads(value)
    return LineString([u_cord, v_cord])


def combine_networks(
    sub_ids: List[str],
    region_name: str,
    homes: List[Home],
    prim_dir: str,
    sec_dir: str,
    secondary_regions: List[str] = None
    ) -> nx.Graph:
    """
    Combine primary and secondary distribution networks into a single graph.

    Args:
        sub_ids (List[str]): List of service substation IDs
        region_name (str): Name of the region or group of regions (prefix of the network files)
        homes (List[Home]): List of Home objects (coordinates and loads)
        prim_dir (str): Directory with <region>_<substation>_nodes.csv / _edges.csv
        sec_dir (str): Directory with <region>_secondary_edges.csv
        secondary_regions (List[str]): Regions whose secondary networks are combined
            (default: [region_name])

    Returns:
        nx.Graph: Combined network graph. Node IDs are strings; nodes carry 'cord', 'label'
            and 'load', edges carry 'geometry', 'label' and 'length'.

    Raises:
        FileNotFoundError: If a primary or the secondary network file is missing
        ValueError: If a file has an invalid format, or a secondary edge refers to a node
            that is neither a home nor a transformer of the primary networks
    """
    logger.info(f"Combining networks for {len(sub_ids)} substations in region {region_name}")
    graph = nx.Graph()

    # Primary networks of every substation
    for sub_id in sub_ids:
        node_file = Path(prim_dir) / f"{region_name}_{sub_id}_nodes.csv"
        edge_file = Path(prim_dir) / f"{region_name}_{sub_id}_edges.csv"
        for path in (node_file, edge_file):
            if not path.exists():
                raise FileNotFoundError(f"Primary network file not found for substation {sub_id}: {path}")

        nodes_df = pd.read_csv(node_file, dtype={"node_id": str})
        for _, row in nodes_df.iterrows():
            node_id = row["node_id"]
            if node_id in graph and graph.nodes[node_id]["label"] != "R":
                raise ValueError(f"Node {node_id} appears in more than one primary network")
            graph.add_node(node_id, cord=_parse_cord(row["cord"]), label=row["label"], load=float(row["load"]))

        edges_df = pd.read_csv(edge_file, dtype={"source": str, "target": str})
        for _, row in edges_df.iterrows():
            u, v = row["source"], row["target"]
            if u not in graph or v not in graph:
                raise ValueError(f"Primary edge ({u}, {v}) of substation {sub_id} refers to an unknown node")
            graph.add_edge(
                u, v,
                geometry=_parse_geometry(row.get("geometry"), graph.nodes[u]["cord"], graph.nodes[v]["cord"]),
                label=row["label"],
                length=float(row["length"]),
            )

    # Secondary network: homes hang from the transformers of the primary networks
    secondary_files = [
        Path(sec_dir) / f"{region}_secondary_edges.csv" for region in (secondary_regions or [region_name])
    ]
    for secondary_file in secondary_files:
        if not secondary_file.exists():
            raise FileNotFoundError(f"Secondary network file not found: {secondary_file}")
    home_by_id = {str(home.id): home for home in homes}
    secondary_df = pd.concat(
        [pd.read_csv(f, dtype={"from_node": str, "to_node": str}) for f in secondary_files],
        ignore_index=True,
    )
    for _, row in secondary_df.iterrows():
        u, v = row["from_node"], row["to_node"]
        for node in (u, v):
            if node in graph:
                continue
            if node in home_by_id:
                home = home_by_id[node]
                graph.add_node(node, cord=tuple(home.cord), label="H", load=float(home.load))
            elif node.startswith("T"):
                raise ValueError(f"Transformer {node} of the secondary network is not in any primary network")
            else:
                raise ValueError(f"Secondary network node {node} is neither a home nor a transformer")
        graph.add_edge(
            u, v,
            geometry=LineString([graph.nodes[u]["cord"], graph.nodes[v]["cord"]]),
            label="secondary",
            length=float(row["length"]),
        )

    logger.info(f"Combined network has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph


def validate_distribution_network(graph: nx.Graph, homes: List[Home] = ()) -> Dict[str, Any]:
    """
    Check the structure of the combined distribution network.

    The network must be a forest with exactly one substation per tree, and every home must
    be connected. Problems are logged as warnings and returned.

    Returns:
        Dict[str, Any]: is_forest, trees, trees_without_substation, trees_with_several_substations,
            homes_missing (IDs of homes not in the network) and isolated_nodes
    """
    components = list(nx.connected_components(graph))
    substations_per_tree = [sum(1 for n in c if graph.nodes[n]["label"] == "S") for c in components]
    home_ids = {str(h.id) for h in homes}
    report = {
        "is_forest": nx.is_forest(graph) if graph.number_of_nodes() else True,
        "trees": len(components),
        "trees_without_substation": sum(1 for k in substations_per_tree if k == 0),
        "trees_with_several_substations": sum(1 for k in substations_per_tree if k > 1),
        "homes_missing": sorted(home_ids - set(graph.nodes)),
        "isolated_nodes": sorted(n for n in graph if graph.degree(n) == 0),
    }
    if not report["is_forest"]:
        logger.warning("The distribution network contains cycles")
    if report["trees_without_substation"] or report["trees_with_several_substations"]:
        logger.warning(
            f"{report['trees_without_substation']} trees without a substation and "
            f"{report['trees_with_several_substations']} trees with several substations"
        )
    if report["homes_missing"]:
        logger.warning(f"{len(report['homes_missing'])} homes are not connected to the distribution network")
    if report["isolated_nodes"]:
        logger.warning(f"{len(report['isolated_nodes'])} isolated nodes in the distribution network")
    return report


def save_distribution_network(
    graph: nx.Graph,
    region_name: str,
    out_dir: str
) -> Tuple[Path, Path]:
    """
    Save the combined network to <region>_distribution_network_nodes.csv / _edges.csv.

    Returns:
        Tuple[Path, Path]: Paths of the nodes and edges files
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    nodes_file = out_path / f"{region_name}_distribution_network_nodes.csv"
    edges_file = out_path / f"{region_name}_distribution_network_edges.csv"

    nodes_df = pd.DataFrame([
        {
            "node_id": node,
            "longitude": attrs["cord"][0],
            "latitude": attrs["cord"][1],
            "label": attrs["label"],
            "load": attrs.get("load", 0.0),
        }
        for node, attrs in graph.nodes(data=True)
    ], columns=["node_id", "longitude", "latitude", "label", "load"])
    edges_df = pd.DataFrame([
        {
            "source": u,
            "target": v,
            "label": attrs["label"],
            "length": attrs["length"],
            "geometry": attrs["geometry"].wkt,
        }
        for u, v, attrs in graph.edges(data=True)
    ], columns=["source", "target", "label", "length", "geometry"])

    # write to temporary files first so an interrupted run never leaves a partial file
    for df, path in ((nodes_df, nodes_file), (edges_df, edges_file)):
        tmp = path.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)

    logger.info(f"Saved distribution network files for region {region_name}")
    return nodes_file, edges_file


def load_distribution_network(nodes_file: Union[str, Path], edges_file: Union[str, Path]) -> nx.Graph:
    """Load a network saved by `save_distribution_network`."""
    for path in (nodes_file, edges_file):
        if not Path(path).exists():
            raise FileNotFoundError(f"Distribution network file not found: {path}")
    graph = nx.Graph()
    for _, row in pd.read_csv(nodes_file, dtype={"node_id": str}).iterrows():
        graph.add_node(
            row["node_id"],
            cord=(float(row["longitude"]), float(row["latitude"])),
            label=row["label"],
            load=float(row["load"]),
        )
    for _, row in pd.read_csv(edges_file, dtype={"source": str, "target": str}).iterrows():
        u, v = row["source"], row["target"]
        graph.add_edge(
            u, v,
            geometry=_parse_geometry(row["geometry"], graph.nodes[u]["cord"], graph.nodes[v]["cord"]),
            label=row["label"],
            length=float(row["length"]),
        )
    return graph


def distribution_network_summary(graph: nx.Graph) -> Dict[str, Any]:
    """Counts of nodes and edges by label and line lengths (km) by edge label."""
    nodes = Counter(d["label"] for _, d in graph.nodes(data=True))
    edges = Counter(d["label"] for _, _, d in graph.edges(data=True))
    length = Counter()
    for _, _, d in graph.edges(data=True):
        length[d["label"]] += d["length"]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "nodes_by_label": dict(sorted(nodes.items())),
        "edges_by_label": dict(sorted(edges.items())),
        "length_km_by_label": {k: round(v / 1000, 3) for k, v in sorted(length.items())},
        "home_load_total": round(sum(d["load"] for _, d in graph.nodes(data=True) if d["label"] == "H"), 3),
    }
