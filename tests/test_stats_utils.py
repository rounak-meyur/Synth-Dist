import json

import networkx as nx
import pytest

from utils.stats_utils import RunStats, mapping_stats, road_network_stats, secnet_stats


def test_run_stats_saved_after_each_stage(tmp_path):
    path = tmp_path / "stats" / "999_run_stats.json"
    stats = RunStats(str(path), region="999")
    assert json.loads(path.read_text())["status"] == "running"

    with stats.stage("roads") as s:
        s["road_nodes"] = 3
    saved = json.loads(path.read_text())
    assert saved["stages"]["roads"]["road_nodes"] == 3
    assert saved["stages"]["roads"]["seconds"] >= 0

    stats.finish()
    saved = json.loads(path.read_text())
    assert saved["status"] == "complete" and saved["finished_at"]
    assert saved["peak_memory_mb"] is None or saved["peak_memory_mb"] > 0


def test_run_stats_records_interruption(tmp_path):
    path = tmp_path / "stats.json"
    stats = RunStats(str(path))
    with pytest.raises(KeyboardInterrupt):
        with stats.stage("secnet"):
            raise KeyboardInterrupt
    saved = json.loads(path.read_text())
    assert saved["status"] == "failed" and saved["failed_stage"] == "secnet"
    assert saved["stages"]["secnet"]["error"].startswith("KeyboardInterrupt")


def test_road_network_stats():
    G = nx.MultiGraph()
    G.add_edge(1, 2, length=1500.0)
    G.add_edge(3, 4, length=500.0)
    assert road_network_stats(G) == {"road_nodes": 4, "road_edges": 2, "road_length_km": 2.0, "road_components": 2}


def test_mapping_stats(homes):
    h2r = [(homes[0], (1, 2, 0)), (homes[1], (1, 2, 0)), (homes[2], (2, 3, 0))]
    r2h = {(1, 2, 0): homes[:2], (2, 3, 0): homes[2:3]}
    stats = mapping_stats(homes, h2r, r2h)
    assert stats["homes"] == len(homes) and stats["homes_mapped"] == 3
    assert stats["homes_unmapped"] == len(homes) - 3
    assert stats["homes_per_road_link"] == {"min": 1, "mean": 1.5, "max": 2}


def test_secnet_stats_counts_homes_per_transformer():
    combined = nx.MultiGraph([(1, "T1"), ("T1", 2), (2, "T2"), ("T2", 3)])
    transformers = [{"transformer_id": "T1", "total_load": 3000}, {"transformer_id": "T2", "total_load": 1000}]
    edges = [
        {"from_node": "T1", "to_node": 11, "length": 20.0},
        {"from_node": 11, "to_node": 12, "length": 30.0},
        {"from_node": "T2", "to_node": 21, "length": 50.0},
    ]
    stats = secnet_stats(combined, transformers, edges)
    assert stats["transformers"] == 2 and stats["homes_served"] == 3
    assert stats["homes_per_transformer"] == {"min": 1, "mean": 1.5, "max": 2}
    assert stats["secondary_length_km"] == pytest.approx(0.1)
    assert stats["transformer_load_total"] == 4000
    assert stats["combined_components"] == 1
