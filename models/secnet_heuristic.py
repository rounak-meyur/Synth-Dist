"""
Heuristic alternative to the MILP in models/secnet.py for generating the secondary
distribution network on a candidate graph built by `create_candidate_network`.

Problem: connect every home node ('H') into a forest of trees where
    - each tree is rooted at a probable transformer node ('T'),
    - the total load served by a transformer is at most `max_rating`,
    - the depth of each tree (hops from the transformer) is at most `max_hops`,
    - a home node has degree at most 2,
minimising the total edge cost  length + penalty * crossing  (same as the MILP).

Because a home has degree <= 2 (one parent, at most one child), every tree is a
transformer with one or more chains of homes hanging from it. The solution is
therefore represented as a set of routes [t, h1, h2, ..., hk] (an open,
multi-depot vehicle routing problem) and solved in two phases:
    1. greedy construction: Prim-like growth from all transformers, only adding
       edges that keep the solution feasible;
    2. local search: relocate, swap, 2-opt, tail exchange and re-root moves,
       accepted only when they lower the cost and keep the solution feasible.
"""
import heapq
import itertools
from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Tuple

import networkx as nx

from utils.logging_utils import LogManager

logger = LogManager.get_logger("secnet_heuristic")

EPS = 1e-9
Route = List[Hashable]


class _ForestSolver:
    """Construction and local search for the chain-forest secondary network problem."""

    def __init__(self, graph: nx.Graph, penalty: float, max_rating: float, max_hops: int):
        self.graph = graph
        self.max_rating = max_rating
        self.max_hops = max_hops

        self.homes = [n for n, d in graph.nodes(data=True) if d['label'] == 'H']
        self.transformers = [n for n, d in graph.nodes(data=True) if d['label'] == 'T']
        self.load = {n: graph.nodes[n]['load'] for n in graph.nodes}

        # Edge cost in both orientations; missing candidate edges are infeasible
        self.cost: Dict[Tuple, float] = {}
        for u, v, d in graph.edges(data=True):
            c = d['length'] + penalty * d['crossing']
            self.cost[(u, v)] = c
            self.cost[(v, u)] = c

        self.routes: List[Route] = []
        self.route_costs: List[float] = []
        self.route_loads: List[float] = []
        self.t_load: Dict[Hashable, float] = defaultdict(float)

    # Route bookkeeping

    def route_cost(self, route: Route) -> float:
        return sum(self.cost.get((a, b), float('inf')) for a, b in zip(route[:-1], route[1:]))

    def route_load(self, route: Route) -> float:
        return sum(self.load[h] for h in route[1:])

    def total_cost(self) -> float:
        return sum(self.route_costs)

    def _add_route(self, route: Route):
        self.routes.append(route)
        self.route_costs.append(self.route_cost(route))
        self.route_loads.append(self.route_load(route))
        self.t_load[route[0]] += self.route_loads[-1]

    def _replace(self, remove: List[int], add: List[Route]):
        for i in sorted(set(remove), reverse=True):
            self.t_load[self.routes[i][0]] -= self.route_loads[i]
            del self.routes[i], self.route_costs[i], self.route_loads[i]
        for route in add:
            if len(route) > 1:  # drop transformers left without homes
                self._add_route(route)

    def _try_move(self, remove: List[int], add: List[Route]) -> bool:
        """Apply the move (replace routes `remove` with `add`) if it is feasible and improving."""
        remove = sorted(set(remove))
        old = sum(self.route_costs[i] for i in remove)
        new = sum(self.route_cost(r) for r in add)
        if not new < old - EPS:
            return False
        if any(len(r) - 1 > self.max_hops for r in add):
            return False
        delta = defaultdict(float)
        for i in remove:
            delta[self.routes[i][0]] -= self.route_loads[i]
        for r in add:
            delta[r[0]] += self.route_load(r)
        for t, d in delta.items():
            # A transformer may only gain load while it stays within its rating
            if d > EPS and self.t_load[t] + d > self.max_rating + EPS:
                return False
        self._replace(remove, add)
        return True

    # Phase 1: greedy construction

    def construct(self):
        """Grow chains from all transformers, always taking the cheapest feasible edge."""
        route_of: Dict[Hashable, int] = {}  # home -> index of its route
        attached = set()
        counter = itertools.count()
        heap = []

        def push_from(u):
            for w in self.graph.neighbors(u):
                if self.graph.nodes[w]['label'] == 'H' and w not in attached:
                    heapq.heappush(heap, (self.cost[(u, w)], next(counter), u, w))

        for t in self.transformers:
            push_from(t)

        while heap and len(attached) < len(self.homes):
            _, _, u, v = heapq.heappop(heap)
            if v in attached:
                continue
            if self.graph.nodes[u]['label'] == 'T':
                if self.t_load[u] + self.load[v] > self.max_rating + EPS:
                    continue
                self._add_route([u, v])
                route_of[v] = len(self.routes) - 1
            else:
                r = route_of[u]
                route = self.routes[r]
                # u must still be the chain tail (degree <= 2) and the chain must have room
                if route[-1] != u or len(route) - 1 >= self.max_hops:
                    continue
                if self.t_load[route[0]] + self.load[v] > self.max_rating + EPS:
                    continue
                route.append(v)
                self.route_costs[r] += self.cost[(u, v)]
                self.route_loads[r] += self.load[v]
                self.t_load[route[0]] += self.load[v]
                route_of[v] = r
            attached.add(v)
            push_from(v)

        for h in self.homes:
            if h not in attached:
                self._attach_relaxed(h)

    def _attach_relaxed(self, home):
        """
        Fallback when no feasible attachment exists: connect the home directly to the
        least overloaded adjacent transformer, exceeding its rating.
        """
        options = [t for t in self.graph.neighbors(home) if self.graph.nodes[t]['label'] == 'T']
        if not options:
            logger.warning(f"Home {home} has no candidate edge to a transformer and is left unconnected")
            return
        t = min(options, key=lambda t: (self.t_load[t], self.cost[(t, home)]))
        logger.warning(
            f"No feasible attachment for home {home}; connecting it to transformer {t} "
            f"which exceeds max_rating ({self.t_load[t] + self.load[home]:.1f} > {self.max_rating})"
        )
        self._add_route([t, home])

    # Phase 2: local search

    def _relocate(self) -> bool:
        for a, route_a in enumerate(self.routes):
            for i in range(1, len(route_a)):
                h = route_a[i]
                reduced = route_a[:i] + route_a[i + 1:]
                # reinsert into the same route
                for j in range(1, len(reduced) + 1):
                    if j != i and self._try_move([a], [reduced[:j] + [h] + reduced[j:]]):
                        return True
                # insert into another route
                for b, route_b in enumerate(self.routes):
                    if b == a:
                        continue
                    for j in range(1, len(route_b) + 1):
                        if self._try_move([a, b], [reduced, route_b[:j] + [h] + route_b[j:]]):
                            return True
                # start a new chain at any transformer
                for t in self.transformers:
                    if self._try_move([a], [reduced, [t, h]]):
                        return True
        return False

    def _swap(self) -> bool:
        for a, b in itertools.combinations(range(len(self.routes)), 2):
            route_a, route_b = self.routes[a], self.routes[b]
            for i in range(1, len(route_a)):
                for j in range(1, len(route_b)):
                    new_a = route_a[:i] + [route_b[j]] + route_a[i + 1:]
                    new_b = route_b[:j] + [route_a[i]] + route_b[j + 1:]
                    if self._try_move([a, b], [new_a, new_b]):
                        return True
        return False

    def _two_opt(self) -> bool:
        for a, route in enumerate(self.routes):
            for i in range(1, len(route) - 1):
                for j in range(i + 1, len(route)):
                    if self._try_move([a], [route[:i] + route[i:j + 1][::-1] + route[j + 1:]]):
                        return True
        return False

    def _exchange_tails(self) -> bool:
        for a, b in itertools.combinations(range(len(self.routes)), 2):
            route_a, route_b = self.routes[a], self.routes[b]
            for i in range(1, len(route_a) + 1):
                for j in range(1, len(route_b) + 1):
                    if i == len(route_a) and j == len(route_b):
                        continue
                    if self._try_move([a, b], [route_a[:i] + route_b[j:], route_b[:j] + route_a[i:]]):
                        return True
        return False

    def _reroot(self) -> bool:
        for a, route in enumerate(self.routes):
            chain = route[1:]
            for t in self.transformers:
                for c in (chain, chain[::-1]):
                    if (t != route[0] or c is not chain) and self._try_move([a], [[t] + c]):
                        return True
        return False

    def improve(self, max_iterations: int = 1000) -> int:
        """Run first-improvement local search; returns the number of improving moves applied."""
        moves = (self._relocate, self._two_opt, self._exchange_tails, self._swap, self._reroot)
        for iteration in range(max_iterations):
            if not any(move() for move in moves):
                return iteration
        logger.info(f"Local search stopped after reaching max_iterations={max_iterations}")
        return max_iterations

    # Result

    def to_graph(self) -> nx.Graph:
        result = nx.Graph()
        for route in self.routes:
            for u, v in zip(route[:-1], route[1:]):
                result.add_edge(u, v, **self.graph[u][v])
        for node in result.nodes():
            result.nodes[node].update(self.graph.nodes[node])
        return result


def create_secondary_distribution_network_heuristic(
        graph: nx.Graph,
        penalty: float = 0.5,
        max_rating: float = 25e3,
        max_hops: int = 10,
        max_iterations: int = 1000,
        local_search: bool = True,
        **kwargs
    ) -> nx.Graph:
    """
    Heuristic substitute for `create_secondary_distribution_network`.

    Args:
        graph (nx.Graph): Candidate network from `create_candidate_network`. Nodes carry
            'label' ('H' or 'T') and 'load'; edges carry 'length' and 'crossing'.
        penalty (float): Cost added per unit of road crossing on an edge.
        max_rating (float): Maximum total load served by a single transformer.
        max_hops (int): Maximum depth of a tree (hops from the transformer to a home).
        max_iterations (int): Maximum number of improving local search moves.
        local_search (bool): Run the local search phase after the greedy construction.
        kwargs: Ignored; accepted so solver parameters for the MILP can be passed unchanged.

    Returns:
        nx.Graph: The secondary network (selected edges with node and edge attributes copied
            from the candidate graph), in the same format as the MILP result.
    """
    solver = _ForestSolver(graph, penalty=penalty, max_rating=max_rating, max_hops=max_hops)
    logger.info(
        f"Starting heuristic for {len(solver.homes)} homes and "
        f"{len(solver.transformers)} probable transformers"
    )

    solver.construct()
    initial_cost = solver.total_cost()
    moves = solver.improve(max_iterations=max_iterations) if local_search else 0
    logger.info(
        f"Heuristic completed. Construction cost: {initial_cost:.2f}, final cost: "
        f"{solver.total_cost():.2f} after {moves} improving moves"
    )

    result = solver.to_graph()
    logger.info(f"Result graph created with {result.number_of_nodes()} nodes and {result.number_of_edges()} edges")
    return result
