from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import networkx as nx
import cvxpy as cp
import numpy as np
import os
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.dataloader import Substation
from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize(
        log_file_path="logs/test_primnet.log", 
        log_config_path="configs/logging_config.ini"
    )

logger = LogManager.get_logger("primnet")


@dataclass
class PrimaryNetworkConfig:
    """Configuration parameters for primary network optimization."""
    voltage_max: float                  # Maximum allowed voltage (pu)
    voltage_min: float                  # Minimum allowed voltage (pu)
    maximum_voltage_drop: float         # maximum voltage drop allowable
    conductor_resistance_per_km: float  # Conductor resistance per km (ohm/km)
    base_impedance: float               # base impedance for pu computation (ohm)
    maximum_branch_flow: float          # Maximum power flow per branch (kVA)
    max_feeder_number: float            # Maximum number of feeders
    max_feeder_capacity: float          # Maximum feeder capacity
    
    # SCIP solver parameters
    relative_gap: float                 # relative optimality gap
    time_limit: float                   # maximum time limit
    verbose: bool                       # display verbose level
    seed: int                           # randomization seed
    warm_start: bool                    # pre-solve or warm start
    threads: int                        # number of threads to use

def optimize_primary_network(
    graph: nx.MultiGraph,
    config: PrimaryNetworkConfig
    ) -> Tuple[nx.Graph, List[int]]:
    """
    Optimize primary distribution network.
    
    Args:
        graph (nx.Graph): combined road and transformer graph partitioned to nearest substation.
    
    """
    
    # Load graph data
    edges = list(graph.edges(keys=True))
    nodes = list(graph.nodes())
    tnodes = [n for n, d in graph.nodes(data=True) if d['label']=='T']
    rnodes = [n for n, d in graph.nodes(data=True) if d['label']=='R']
    n_edges = len(edges)
    n_nodes = len(nodes)
    n_rnodes = len(rnodes)
    
    # Setup parameter values
    p = np.array([1e-3 * graph.nodes[n]['load'] for n in nodes])
    d = np.array([1e-3 * graph.nodes[n]['distance'] for n in nodes])
    c = np.array([1e-3 * graph[u][v][k]['length'] for u, v, k in edges])
    
    # Create incidence matrices
    A = nx.incidence_matrix(graph, oriented=True).todense()
    I = np.abs(A)
    
    # Create submatrices for home nodes
    road_indices = [nodes.index(n) for n in rnodes]
    transformer_indices = [nodes.index(n) for n in tnodes]
    Ar = A[road_indices, :]
    Ir = I[road_indices, :]
    At = A[transformer_indices, :]
    pt = p[transformer_indices]
    dr = d[road_indices]
    
    # R matrix
    R = (config.conductor_resistance_per_km / config.base_impedance) * np.diag(c)
    
    # Setup optimization variables
    x = cp.Variable(n_edges, boolean=True)
    t = cp.Variable(n_edges, boolean=True)
    f = cp.Variable(n_edges)
    g = cp.Variable(n_edges)
    y = cp.Variable(n_rnodes, boolean=True)
    z = cp.Variable(n_rnodes, boolean=True)
    v = cp.Variable(n_nodes)
    
    # Vectors of ones
    a = np.ones(shape=(n_nodes-1,))
    b = np.ones(shape=(n_nodes,))
    r = np.ones(shape=(n_rnodes,))
    e = np.ones(shape=(n_edges,))
    
    # Setup constraints
    constraints = [
        # master tree connectivity constraint
        A[1:,:] @ g == -a, 
        g - (n_nodes * t) <= 0,
        g + (n_nodes * t) >= 0,
        x <= t,
        
        # radial network constraints
        cp.sum(x) == n_nodes - (2 * n_rnodes) + cp.sum(y) + cp.sum(z),
        Ir @ x <= n_edges * y,
        Ir @ x >= 2 * (y + z - r),
        r - z <= y,
        
        # flow constraints
        Ar @ f - config.maximum_branch_flow * (r - z) <= 0,
        Ar @ f + config.maximum_branch_flow * (r - z) >= 0,
        At @ f == -pt,
        f - (config.maximum_branch_flow * x) <= 0,
        f + (config.maximum_branch_flow * x) >= 0,
        
        # feeder number upper bound
        n_rnodes - cp.sum(z) <= config.max_feeder_number,
        
        # voltage constraints
        (A.T @ v) - (R @ f) - config.maximum_voltage_drop * (e - x) <= 0,
        (A.T @ v) - (R @ f) + config.maximum_voltage_drop * (e - x) >= 0,
        r - z <= v[road_indices],
        v <= config.voltage_max * b,
        v >= config.voltage_min * b
        
    ]
    
    # Objective function
    objective = cp.Minimize(
        (c @ x) + (dr @ (r - z))
    )
    
    # Solve optimization problem
    solver_params = {
        
        # Logging/output control
        'limits/time': config.get('time_limit', 2147483647),
        'display/verblevel': 4 if config.verbose else 0,
        'limits/gap': config.relative_gap,  # Stop when gap is reached
        
        # # Additional performance settings
        # 'presolving/maxrounds': 0 if config.warm_start else -1,  # Disable presolving if warm start
        # 'lp/threads': config.threads,  # Single thread for deterministic behavior
    }
    problem = cp.Problem(objective, constraints)
    logger.info("Starting optimization")
    problem.solve(solver=cp.SCIP, verbose=True, **solver_params)
    logger.info(f"Optimization completed. Status: {problem.status}, Optimal value: {problem.value}")
    
    # Build optimized network
    result = nx.Graph()
    
    # Add the selected edges to the result graph
    for i, (u, v, k) in enumerate(edges):
        if x.value[i] > 0.5:  # Consider the edge selected if x > 0.5
            result.add_edge(
                u, v, 
                geometry=graph[u][v][k]['geometry'],
                label="primary",
                length=graph[u][v][k]["length"]
                )
    
    # Add node attributes for nodes in the result graph
    for node in result.nodes():
        result.nodes[node].update(graph.nodes[node])
        
    # Add feeder connections from substation to feeder nodes
    feeder_nodes = []
    for i, n in enumerate(rnodes):
        if z.value[i] < 0.2:
            if n not in result.nodes():
                logger.error(f"Feeder node {n} is not present in the result network.")
                raise ValueError("Identified feeder node is not in output network.")
            else:
                feeder_nodes.append(n)
            
    
    return result, feeder_nodes
