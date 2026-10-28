"""
Heuristic alternative to the MILP in models/primnet.py for generating the primary
distribution network of a substation partition.

Problem: on the combined road ('R') and transformer ('T') network of a partition, select
a forest of trees such that
    - every transformer node is covered,
    - every tree is rooted at a road node (feeder) that connects to the substation,
    - road nodes only connect transformers: a road node that is not a feeder is never a leaf,
    - the power flow on every branch is at most `maximum_branch_flow`,
    - the voltage at every node stays at or above `voltage_min` (linearized drop
      r / Z_base * length_km * flow along each branch, starting at `feeder_voltage` at the feeder),
    - at most `max_feeder_number` feeders are used (best effort, logged when exceeded),
minimising total line length (km) plus the distance (km) from each feeder to the substation,
the same objective as the MILP.

Method:
    1. add a virtual substation node joined to every road node by an edge of the road node's
       distance to the substation, and connect all transformers and the substation with an
       approximate Steiner tree (Mehlhorn), improved by a local search that inserts or removes
       road nodes when the pruned minimum spanning tree gets shorter; removing the substation node gives a forest whose
       feeders are the substation's neighbours;
    2. a tree that violates the flow or voltage limits is re-rooted at the cheapest feasible
       road node, or cut into two load-balanced subtrees at an edge where both sides contain a
       road node, recursively;
    3. prune road-node leaves that are not feeders.
"""
from typing import Dict, Hashable, List, Optional, Tuple

import networkx as nx
from networkx.algorithms.approximation import steiner_tree
from shapely.geometry import LineString

from utils.logging_utils import LogManager

logger = LogManager.get_logger("primnet_heuristic")

Tree = nx.Graph
SUBSTATION = "__substation__"  # virtual substation node of the Steiner formulation


class _PrimaryForestSolver:

    def __init__(self, graph: nx.MultiGraph, config, feeder_voltage: float = 1.0, max_passes: int = 20):
        self.graph = graph
        self.max_passes = max_passes
        self.max_flow = float(config["maximum_branch_flow"])
        self.voltage_min = float(config["voltage_min"])
        self.feeder_voltage = feeder_voltage
        self.r_per_km = float(config["conductor_resistance_per_km"]) / float(config["base_impedance"])

        # loads in kVA (W * 1e-3) and distances in km, as in the MILP
        self.load = {n: 1e-3 * float(d['load']) for n, d in graph.nodes(data=True)}
        self.is_road = {n: d['label'] == 'R' for n, d in graph.nodes(data=True)}
        self.feeder_cost = {
            n: 1e-3 * float(d.get('distance', 0.0)) for n, d in graph.nodes(data=True) if self.is_road[n]
        }

        # simple graph keeping the shortest of parallel edges (and its key)
        self.simple = nx.Graph()
        self.simple.add_nodes_from(graph.nodes)
        for u, v, k, d in graph.edges(keys=True, data=True):
            if u == v:
                continue
            if not self.simple.has_edge(u, v) or d['length'] < self.simple[u][v]['length']:
                self.simple.add_edge(u, v, length=float(d['length']), key=k)

    # Feasibility of a rooted tree

    def _orders(self, tree: Tree, root) -> Tuple[List, Dict]:
        parent = {root: None}
        order = [root]
        for u, v in nx.bfs_edges(tree, root):
            parent[v] = u
            order.append(v)
        return order, parent

    def _subtree_loads(self, tree: Tree, root) -> Tuple[List, Dict, Dict]:
        order, parent = self._orders(tree, root)
        sub = {n: self.load[n] for n in order}
        for n in reversed(order[1:]):
            sub[parent[n]] += sub[n]
        return order, parent, sub

    def is_feasible(self, tree: Tree, root) -> bool:
        order, parent, sub = self._subtree_loads(tree, root)
        if sub[root] > self.max_flow + 1e-9:
            return False
        voltage = {root: self.feeder_voltage}
        for n in order[1:]:
            p = parent[n]
            voltage[n] = voltage[p] - self.r_per_km * 1e-3 * tree[p][n]['length'] * sub[n]
            if voltage[n] < self.voltage_min - 1e-9:
                return False
        return True

    def best_root(self, tree: Tree) -> Optional[Hashable]:
        """Feasible road node with the smallest feeder cost, or None."""
        roads = sorted((n for n in tree if self.is_road[n]), key=lambda n: self.feeder_cost[n])
        return next((r for r in roads if self.is_feasible(tree, r)), None)

    # Construction

    def _steiner_forest(self) -> Tuple[Tree, List[Hashable]]:
        """
        Connect all transformers to the substation at minimum line + feeder length.

        A virtual substation node is joined to every road node by an edge as long as that road
        node's distance to the substation. A Steiner tree over the transformers and the substation
        then minimises the MILP objective; removing the substation node leaves the forest, whose
        feeders are the substation's neighbours. Steiner tree leaves are terminals, so road nodes
        are never leaves.
        """
        aug = self.simple.copy()
        for n, cost in self.feeder_cost.items():
            aug.add_edge(SUBSTATION, n, length=1e3 * cost)

        reachable = nx.node_connected_component(aug, SUBSTATION) if self.feeder_cost else {SUBSTATION}
        transformers = [n for n in self.graph if not self.is_road[n]]
        unreachable = [n for n in transformers if n not in reachable]
        if unreachable:
            logger.warning(f"Transformers {sorted(map(str, unreachable))} cannot reach a road node; left unconnected")
        terminals = [n for n in transformers if n in reachable] + [SUBSTATION]
        if len(terminals) == 1:
            return nx.Graph(), []

        tree = nx.Graph(steiner_tree(aug.subgraph(reachable), terminals, weight='length', method='mehlhorn'))
        tree = self._improve(aug, tree, set(terminals))
        feeders = list(tree.neighbors(SUBSTATION))
        tree.remove_node(SUBSTATION)
        for u, v in tree.edges():
            tree[u][v].update(self.simple[u][v])
        return tree, feeders

    @staticmethod
    def _mst_pruned(aug: nx.Graph, nodes, terminals: set) -> Tuple[float, Optional[Tree]]:
        """Minimum spanning tree over `nodes` with non-terminal leaves pruned; (inf, None) if disconnected."""
        sub = aug.subgraph(nodes)
        mst = nx.minimum_spanning_tree(sub, weight='length')
        if mst.number_of_edges() != mst.number_of_nodes() - 1:
            return float('inf'), None
        leaves = [n for n in mst if mst.degree(n) <= 1 and n not in terminals]
        while leaves:
            mst.remove_nodes_from(leaves)
            leaves = [n for n in mst if mst.degree(n) <= 1 and n not in terminals]
        return sum(d['length'] for _, _, d in mst.edges(data=True)), mst

    def _improve(self, aug: nx.Graph, tree: Tree, terminals: set) -> Tree:
        """
        Steiner tree local search over the road (Steiner) nodes: insert a road node adjacent to the
        tree or remove one from it when the pruned minimum spanning tree gets shorter.
        """
        best_cost, best = self._mst_pruned(aug, tree, terminals)
        if best is None:
            best_cost, best = sum(d['length'] for _, _, d in tree.edges(data=True)), tree
        for _ in range(self.max_passes):
            improved = False
            steiner = [n for n in best if n not in terminals]
            outside = {m for n in best for m in aug.neighbors(n) if m not in best and m != SUBSTATION}
            for node in steiner + sorted(outside, key=str):
                nodes = set(best) - {node} if node in best else set(best) | {node}
                cost, mst = self._mst_pruned(aug, nodes, terminals)
                if cost < best_cost - 1e-9:
                    best_cost, best, improved = cost, mst, True
            if not improved:
                break
        return best

    def _split(self, tree: Tree) -> List[Tuple[Tree, Hashable]]:
        """Recursively split a tree until every part has a feasible feeder root."""
        root = self.best_root(tree)
        if root is not None:
            return [(tree, root)]

        roads = [n for n in tree if self.is_road[n]]
        root = min(roads, key=lambda n: self.feeder_cost[n])
        order, parent, sub = self._subtree_loads(tree, root)
        # road nodes below each node: both parts of a cut need a road node for their feeder
        roads_below = {n: int(self.is_road[n]) for n in order}
        for n in reversed(order[1:]):
            roads_below[parent[n]] += roads_below[n]

        total = sub[root]
        cuts = [n for n in order[1:] if roads_below[n] > 0]
        if not cuts:
            logger.warning(
                f"Cannot split a tree with load {total:.1f} kVA into feasible feeders; "
                f"keeping it on feeder {root} although it violates the flow or voltage limits"
            )
            return [(tree, root)]
        cut = min(cuts, key=lambda n: max(sub[n], total - sub[n]))

        # nodes in the subtree below the cut (BFS order visits parents before children)
        below = {cut}
        for n in order:
            if parent[n] in below:
                below.add(n)
        lower = tree.subgraph(below).copy()
        upper = tree.subgraph(set(tree) - below).copy()
        return self._split(lower) + self._split(upper)

    def _prune(self, tree: Tree, root) -> Tree:
        """Remove road-node leaves that are not the feeder (they carry no transformer)."""
        tree = tree.copy()
        leaves = [n for n in tree if tree.degree(n) <= 1 and self.is_road[n] and n != root]
        while leaves:
            tree.remove_nodes_from(leaves)
            leaves = [n for n in tree if tree.degree(n) <= 1 and self.is_road[n] and n != root]
        return tree

    def solve(self) -> List[Tuple[Tree, Hashable]]:
        tree, feeders = self._steiner_forest()
        forest = []
        for root in feeders:
            part = tree.subgraph(nx.node_connected_component(tree, root)).copy()
            parts = [(part, root)] if self.is_feasible(part, root) else self._split(part)
            for piece, piece_root in parts:
                piece = self._prune(piece, piece_root)
                if any(not self.is_road[n] for n in piece):
                    forest.append((piece, piece_root))
        return forest


def create_primary_network_heuristic(
        graph: nx.MultiGraph,
        config,
        feeder_voltage: float = 1.0,
        local_search_passes: int = 20,
        **kwargs
    ) -> Tuple[nx.Graph, List[Hashable]]:
    """
    Heuristic substitute for `optimize_primary_network`.

    Args:
        graph (nx.MultiGraph): Combined road and transformer network of a substation partition.
            Nodes carry 'label' ('R' or 'T'), 'load' (W) and 'distance' (m, to the substation);
            edges carry 'length' (m) and 'geometry'.
        config: Primary network parameters: maximum_branch_flow (kVA), voltage_min,
            conductor_resistance_per_km, base_impedance and max_feeder_number.
        feeder_voltage (float): Voltage (pu) at the feeder nodes used in the voltage check.
        local_search_passes (int): Maximum passes of the Steiner node insertion/removal local search
            (0 disables it).
        kwargs: Ignored; accepted so MILP solver parameters can be passed unchanged.

    Returns:
        Tuple[nx.Graph, List]: The primary network (edges labelled 'primary' with 'geometry' and
            'length', node attributes copied from `graph`) and the feeder road nodes, in the same
            format as the MILP.
    """
    solver = _PrimaryForestSolver(graph, config, feeder_voltage=feeder_voltage, max_passes=local_search_passes)
    forest = solver.solve()

    result = nx.Graph()
    feeder_nodes = []
    for tree, root in forest:
        feeder_nodes.append(root)
        result.add_node(root)
        for u, v in tree.edges():
            k = solver.simple[u][v]['key']
            geometry = graph[u][v][k].get('geometry')
            if geometry is None:
                geometry = LineString([graph.nodes[u]['cord'], graph.nodes[v]['cord']])
            result.add_edge(
                u, v,
                geometry=geometry,
                label="primary",
                length=graph[u][v][k]['length'],
            )
    for node in result.nodes():
        result.nodes[node].update(graph.nodes[node])

    max_feeders = config.get("max_feeder_number") if hasattr(config, "get") else None
    if max_feeders is not None and len(feeder_nodes) > max_feeders:
        logger.warning(f"Heuristic needs {len(feeder_nodes)} feeders, more than max_feeder_number={max_feeders}")
    length_km = 1e-3 * sum(d['length'] for _, _, d in result.edges(data=True))
    feeder_km = sum(solver.feeder_cost[n] for n in feeder_nodes)
    logger.info(
        f"Primary network heuristic: {len(feeder_nodes)} feeders, {result.number_of_edges()} edges, "
        f"cost {length_km + feeder_km:.3f} km (lines {length_km:.3f} + feeders {feeder_km:.3f})"
    )
    return result, feeder_nodes
