"""
Run statistics for main.py: compute time and data sizes for every pipeline stage,
written to a JSON file that is updated after each stage (so it survives failures).
"""
import json
import os
import platform
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx

from utils.logging_utils import LogManager

logger = LogManager.get_logger("stats_utils")


def _peak_memory_mb() -> float:
    """Peak resident memory of this process in MB (None where unsupported)."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in bytes on macOS and in kilobytes on Linux
    return round(peak / (1024 * 1024 if platform.system() == "Darwin" else 1024), 1)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class RunStats:
    """Collect per-stage statistics and save them as JSON."""

    def __init__(self, output_file: str, keep_stages: bool = False, **run_info: Any):
        """
        Args:
            output_file (str): Path of the JSON statistics file
            keep_stages (bool): Keep the statistics of stages already saved in the file
                (stages run by an earlier job); stages run now replace them
            run_info: Run level details (region, config file, method, ...)
        """
        self.output_file = Path(output_file)
        self._t0 = time.perf_counter()
        previous = {}
        if keep_stages and self.output_file.exists():
            try:
                previous = json.loads(self.output_file.read_text()).get("stages", {})
            except (json.JSONDecodeError, AttributeError):
                logger.warning(f"Ignoring unreadable statistics file {self.output_file}")
        self.data: Dict[str, Any] = {
            **run_info,
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "total_seconds": None,
            "peak_memory_mb": None,
            "stages": previous,
        }
        self.save()

    @contextmanager
    def stage(self, name: str):
        """
        Time a stage. The yielded dict is stored as the stage's statistics; set its
        'status' to 'computed', 'reused' or 'resumed' and add data sizes to it.
        """
        entry: Dict[str, Any] = {"status": "computed"}
        self.data["stages"][name] = entry
        start = time.perf_counter()
        try:
            yield entry
        except BaseException as e:
            entry["status"] = "failed"
            entry["error"] = f"{type(e).__name__}: {e}"
            self.data["status"] = "failed"
            self.data["failed_stage"] = name
            raise
        finally:
            entry["seconds"] = round(time.perf_counter() - start, 3)
            self._update_totals()
            self.save()

    def finish(self, status: str = "complete"):
        self.data["status"] = status
        self._update_totals()
        self.save()
        logger.info(f"Run statistics written to {self.output_file}")

    def _update_totals(self):
        self.data["finished_at"] = _now()
        self.data["total_seconds"] = round(time.perf_counter() - self._t0, 3)
        self.data["peak_memory_mb"] = _peak_memory_mb()

    def save(self):
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.output_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        os.replace(tmp, self.output_file)


# Data size summaries

def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0, "mean": 0, "max": 0}
    return {"min": min(values), "mean": round(mean(values), 3), "max": max(values)}


def road_network_stats(roads: nx.MultiGraph) -> Dict[str, Any]:
    lengths = [d.get("length") or 0.0 for _, _, d in roads.edges(data=True)]
    return {
        "road_nodes": roads.number_of_nodes(),
        "road_edges": roads.number_of_edges(),
        "road_length_km": round(sum(lengths) / 1000, 3),
        "road_components": nx.number_connected_components(roads) if roads.number_of_nodes() else 0,
    }


def mapping_stats(homes: List, h2r: List[Tuple], r2h: Dict[Tuple, List]) -> Dict[str, Any]:
    mapped = {home.id for home, edge in h2r if edge is not None}
    return {
        "homes": len(homes),
        "homes_mapped": len(mapped),
        "homes_unmapped": len(homes) - len(mapped),
        "road_links_with_homes": len(r2h),
        "homes_per_road_link": _summary([len(h) for h in r2h.values()]),
    }


def secnet_stats(
        combined_network: nx.MultiGraph,
        transformers: Iterable[Dict[str, Any]],
        secondary_edges: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
    """
    Args:
        combined_network: Road network combined with transformer nodes
        transformers: Rows of <region>_transformers.csv
        secondary_edges: Rows of <region>_secondary_edges.csv
    """
    transformers, secondary_edges = list(transformers), list(secondary_edges)
    homes_per_transformer: Dict[str, int] = {t["transformer_id"]: 0 for t in transformers}
    homes = set()
    for e in secondary_edges:
        for node in (str(e["from_node"]), str(e["to_node"])):
            if not node.startswith("T"):
                homes.add(node)
    # every home has exactly one edge towards its transformer; count homes per tree
    graph = nx.Graph()
    graph.add_edges_from((str(e["from_node"]), str(e["to_node"])) for e in secondary_edges)
    for component in nx.connected_components(graph):
        roots = [n for n in component if n.startswith("T")]
        if len(roots) == 1 and roots[0] in homes_per_transformer:
            homes_per_transformer[roots[0]] = len(component) - 1
    return {
        "transformers": len(transformers),
        "transformer_load_total": round(sum(float(t["total_load"]) for t in transformers), 3),
        "transformer_load": _summary([float(t["total_load"]) for t in transformers]),
        "homes_per_transformer": _summary(list(homes_per_transformer.values())),
        "homes_served": len(homes),
        "secondary_edges": len(secondary_edges),
        "secondary_length_km": round(sum(float(e["length"]) for e in secondary_edges) / 1000, 3),
        "combined_nodes": combined_network.number_of_nodes(),
        "combined_edges": combined_network.number_of_edges(),
        "combined_components": nx.number_connected_components(combined_network),
    }


def partition_stats(substations: List, partition_data: Dict[int, Dict[str, Dict]]) -> Dict[str, Any]:
    per_substation = {}
    for sub_id, part in partition_data.items():
        nodes = part["nodelist"].values()
        transformers = [n for n in nodes if n["label"] == "T"]
        per_substation[str(sub_id)] = {
            "nodes": len(part["nodelist"]),
            "edges": len(part["edgelist"]),
            "transformers": len(transformers),
            "road_nodes": sum(1 for n in nodes if n["label"] == "R"),
            "load": round(sum(float(n["load"]) for n in transformers), 3),
        }
    return {
        "substations_in_region": len(substations),
        "substations_assigned": len(partition_data),
        "transformers_assigned": sum(s["transformers"] for s in per_substation.values()),
        "per_substation": per_substation,
    }


def primnet_stats(
        edges_by_substation: Dict[str, List[Dict[str, Any]]],
        seconds_by_substation: Dict[str, float] = None,
    ) -> Dict[str, Any]:
    """
    Args:
        edges_by_substation: Rows of <region>_<substation>_edges.csv for every service substation
        seconds_by_substation: Generation time of the substations generated in this run
    """
    seconds_by_substation = seconds_by_substation or {}
    per_substation = {}
    for sub_id, edges in edges_by_substation.items():
        primary = [e for e in edges if e.get("label") == "primary"]
        feeders = [e for e in edges if e.get("label") == "feeder"]
        entry = {
            "feeders": len(feeders),
            "primary_edges": len(primary),
            "primary_length_km": round(sum(float(e["length"]) for e in primary) / 1000, 3),
            "feeder_length_km": round(sum(float(e["length"]) for e in feeders) / 1000, 3),
        }
        if sub_id in seconds_by_substation:
            entry["seconds"] = seconds_by_substation[sub_id]
        per_substation[str(sub_id)] = entry
    return {
        "service_substations": len(per_substation),
        "feeders": sum(s["feeders"] for s in per_substation.values()),
        "primary_edges": sum(s["primary_edges"] for s in per_substation.values()),
        "primary_length_km": round(sum(s["primary_length_km"] for s in per_substation.values()), 3),
        "feeder_length_km": round(sum(s["feeder_length_km"] for s in per_substation.values()), 3),
        "per_substation": per_substation,
    }
