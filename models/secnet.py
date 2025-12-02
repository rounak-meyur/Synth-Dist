import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, LinearRing
from scipy.spatial import Delaunay
from geopy.distance import geodesic
from typing import List, Tuple, Optional, Union
import cvxpy as cp
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
        log_file_path="logs/test_secnet.log", 
        log_config_path="configs/logging_config.ini"
    )

logger = LogManager.get_logger("secnet")


def create_candidate_network(
        road_edge: Tuple,
        road_edge_geometry: LineString,
        mapped_homes: List[Home],
        nearest_homes: Optional[int] = None,
        minimum_separation: float = 50.0
    ) -> Tuple[nx.Graph, List[Union[int, str]]]:
    """
    Create an undirected graph representing the candidate set of edges for the optimal secondary distribution network.

    Args:
        road_edge_geometry (LineString): The geometry of the road edge.
        mapped_homes (List[Home]): List of homes mapped to the road edge.
        nearest_homes (Optional[int]): Minimum number of homes to consider for direct transformer connection.
        minimum_separation (float): Minimum distance in meters between points on the road edge geometry.

    Returns:
        Tuple containing:
        - nx.Graph: The candidate network graph (without road nodes)
        - List[Union[int, str]]: Node IDs in order (start node, transformer nodes, end node)
    """
    logger.info(f"Creating candidate network for {len(mapped_homes)} homes mapped to {road_edge}")

    # Get road end points
    start_node = road_edge[0]  
    end_node = road_edge[1]

    # 1. Interpolate points along the road edge geometry
    probable_transformers = interpolate_points(road_edge_geometry, minimum_separation)
    logger.info(f"Interpolated {len(probable_transformers)} probable transformer locations")

    # 2. Label homes and probable transformers
    labeled_points = label_points(road_edge_geometry, mapped_homes, probable_transformers)
    logger.info("Labeled homes and probable transformers")

    # Create the graph
    G = nx.Graph()

    # Add nodes to the graph
    for point, side in labeled_points:
        if isinstance(point, Home):
            G.add_node(point.id, cord=point.cord, load=point.load, label='H', side=side)
        else:
            G.add_node(f"T{probable_transformers.index(point)}", cord=point, load=0.0, label='T', side=side)

    # 3. Add edges between homes
    if len(mapped_homes) > 10:
        logger.info("Using Delaunay triangulation for home connections")
        home_edges = delaunay_edges(mapped_homes)
    else:
        logger.info("Using all possible edges for home connections")
        home_edges = all_pairs(mapped_homes)

    for edge in home_edges:
        add_edge(G, edge[0].id, edge[1].id)

    # 4. Add edges between probable transformers and homes
    for i, transformer in enumerate(probable_transformers):
        transformer_id = f"T{i}"
        if nearest_homes is None:
            for home in mapped_homes:
                add_edge(G, transformer_id, home.id)
        else:
            nearest = sorted(mapped_homes, key=lambda h: geodesic(transformer[::-1], h.cord[::-1]).meters)[:nearest_homes]
            for home in nearest:
                add_edge(G, transformer_id, home.id)

    logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Create ordered lists of nodes and coordinates, including road nodes but without modifying original IDs
    ordered_nodes = [start_node] + [f"T{i}" for i in range(len(probable_transformers))] + [end_node]
    return G, ordered_nodes

def interpolate_points(geometry: LineString, min_separation: float) -> List[Tuple[float, float]]:
    """Interpolate points along the geometry at the specified minimum separation using geodesic distance."""
    points = []
    coords = list(geometry.coords)
    total_length = sum(geodesic(coord[::-1], coords[i+1][::-1]).meters for i, coord in enumerate(coords[:-1]))
    distance = 0
    while distance < total_length:
        point = geometry.interpolate(distance / total_length, normalized=True)
        points.append((point.x, point.y))
        distance += min_separation
    return points

def label_points(geometry: LineString, homes: List[Home], transformers: List[Tuple[float, float]]) -> List[Tuple[Union[Home, Tuple[float, float]], int]]:
    """Label points as being on one side or the other of the geometry."""
    link_coords = list(geometry.coords)
    side = {home.id: 1 if LinearRing(link_coords + [tuple(home.cord), link_coords[0]]).is_ccw else -1 for home in homes}
    
    labeled_points = []
    for home in homes:
        labeled_points.append((home, side[home.id]))
    
    for transformer in transformers:
        labeled_points.append((transformer, 0))
    
    return labeled_points

def delaunay_edges(homes: List[Home]) -> List[Tuple[Home, Home]]:
    """Get edges from Delaunay triangulation of home locations."""
    points = np.array([home.cord for home in homes])
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[2], simplex[0]))))
    return [(homes[i], homes[j]) for i, j in edges]

def all_pairs(homes: List[Home]) -> List[Tuple[Home, Home]]:
    """Get all possible pairs of homes."""
    return [(homes[i], homes[j]) for i in range(len(homes)) for j in range(i+1, len(homes))]

def add_edge(G: nx.Graph, node1: str, node2: str):
    """Add an edge to the graph with length and crossing attributes."""
    cord1 = G.nodes[node1]['cord']
    cord2 = G.nodes[node2]['cord']
    length = geodesic(cord1[::-1], cord2[::-1]).meters
    crossing = abs(G.nodes[node1]['side'] - G.nodes[node2]['side'])
    G.add_edge(node1, node2, length=length, crossing=crossing)


def create_secondary_distribution_network(
        graph: nx.Graph,
        penalty: float = 0.5,
        max_rating: float = 25e3,
        max_hops: int = 10,
        solver: str = "scip",
        **kwargs
    ) -> nx.Graph:

    # Step 1: Prepare the input data
    nodes = list(graph.nodes())
    edges = list(graph.edges())
    home_nodes = [n for n, d in graph.nodes(data=True) if d['label'] == 'H']
    
    # Create parameter vectors and matrices
    p = np.array([graph.nodes[n]['load'] for n in nodes])
    c = np.array([graph[u][v]['length'] + (penalty * graph[u][v]['crossing']) for u, v in edges])
    y = np.ones(len(home_nodes))
    
    # Create incidence matrices
    A = nx.incidence_matrix(graph, oriented=True).todense()
    I = np.abs(A)
    
    # Create submatrices for home nodes
    home_indices = [nodes.index(n) for n in home_nodes]
    Ar = A[home_indices, :]
    Ir = I[home_indices, :]
    
    # Step 2: Define the optimization variables
    x = cp.Variable(len(edges), boolean=True)
    f = cp.Variable(len(edges))
    z = cp.Variable(len(edges))
    
    # Step 3: Define the objective function
    objective = cp.Minimize(c @ x)
    
    # Step 4: Define the constraints
    constraints = [
        cp.sum(x) == len(home_nodes),
        Ar @ f == -p[home_indices],
        f - (max_rating * x) <= 0,
        f + (max_rating * x) >= 0,
        Ar @ z == -y,
        z - (max_hops * x) <= 0,
        z + (max_hops * x) >= 0,
        Ir @ x <= 2 * y
    ]
    
    # Step 5: Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    logger.info("Starting optimization")

    try:
        if solver.lower() == "scip":
            solver_params = {
                'limits/time': kwargs.get('time_limit', 2147483647),
                'display/verblevel': 4 if kwargs.get('verbose',True) else 0,
                'limits/gap': kwargs.get('relative_gap',0.01),  # Stop when gap is reached
                'presolving/maxrounds': 0 if kwargs.get('warm_start',True) else -1,  # Disable presolving if warm start
                }
            problem.solve(solver=cp.SCIP, verbose=False, **solver_params)
            logger.info(f"Optimization completed. Status: {problem.status}, Optimal value: {problem.value}")
        
        elif solver.lower() == "gurobi":
            problem.solve(solver=cp.GUROBI, verbose=False, **solver_params)
            logger.info(f"Optimization completed. Status: {problem.status}, Optimal value: {problem.value}")
        else:
            logger.error(f"Unsupported solver specified: {solver}")
            return nx.Graph()
    except Exception as e:
        logger.error(f"Optimization failed while solving optimization problem: {str(e)}")
        return nx.Graph()
    
    # Step 6: Construct the result graph
    result = nx.Graph()
    
    # Add the selected edges to the result graph
    for i, (u, v) in enumerate(edges):
        if x.value[i] > 0.5:  # Consider the edge selected if x > 0.5
            result.add_edge(u, v, **graph[u][v])
    
    # Add node attributes for nodes in the result graph
    for node in result.nodes():
        result.nodes[node].update(graph.nodes[node])
    
    logger.info(f"Result graph created with {result.number_of_nodes()} nodes and {result.number_of_edges()} edges")
    return result


if __name__ == "__main__":
    logger.info("SecNet module executed directly")
    # Add any direct execution code here if needed

    from utils.dataloader import load_homes
    from utils.osm_utils import load_roads
    input_home_csv = "data/load/test-home-load.csv"
    homes = load_homes(file_path=input_home_csv)
    roads = load_roads(homes)

    from utils.mapping import(
        read_mapping_from_file, 
        compute_edge_to_homes_map
    )
    h2r = read_mapping_from_file(homes, filename="out/mapping/test_map_h2r.txt")
    r2h = compute_edge_to_homes_map(h2r)

    test_road = (896654257, 896654257, 0)
    logger.info(f"Creating secondary network for road link: {test_road}")
    test_geom = roads.edges(keys=True)[test_road]['geometry']
    candidate_g, road_nodes = create_candidate_network(
        test_road, test_geom,
        r2h[test_road],
    )
    logger.info(f"Nodes along the road link in order: {road_nodes}")

    result_secnet = create_secondary_distribution_network(
        graph=candidate_g,
    )

    from utils.drawings import plot_candidate, plot_secnet
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(20,20))
    plot_candidate(candidate_g, test_geom, ax=ax[0], fontsize=18)
    plot_secnet(result_secnet, test_geom, ax=ax[1], fontsize=18)
    fig.savefig("figs/test_candidate_graph.png", bbox_inches='tight')
    