"""
End-to-end check of main.py: config + logging, home loading, road loading/caching,
home-to-road mapping, secondary network generation and transformer partitioning,
including checkpointing (reuse, invalidation, resume and --from-scratch).

OpenStreetMap is replaced by a synthetic road grid, and every output goes to tmp_path.
"""
import json
import sys
from unittest.mock import MagicMock

import networkx as nx
import pytest
from omegaconf import OmegaConf

import main as main_module
from conftest import REPO_ROOT, make_road_grid, write_homes_csv
from utils import osm_utils, secnet_utils

REGION = "777"
SUBSTATION_ID = "4242"
STAGE_LOGS = {
    "roads": "Reusing saved road network",
    "mapping": "Reusing saved home to road mapping",
    "secnet": "Reusing saved secondary network",
    "partition": "Reusing saved transformer partitioning",
    "primnet": "Reusing saved primary networks",
    "finalnet": "Reusing saved distribution network",
}


@pytest.fixture
def road_grid():
    # NetworkPartitioner drops substations whose cell has <= 100 nodes, so the
    # road network must be large enough to keep one substation
    return make_road_grid(12)


class Pipeline:
    """main.py configured to read and write everything below `base`."""

    def __init__(self, base, homes, monkeypatch):
        self.base, self.monkeypatch = base, monkeypatch
        self.conf = OmegaConf.load(REPO_ROOT / "configs/config_heuristic.yaml")
        # main.py builds paths by string concatenation, so directories need a trailing slash
        self.conf.inputs.home_csv_dir = f"{base}/data/load/"
        self.conf.inputs.substation_csv = str(base / "substations.csv")
        self.conf.miscellaneous.log_dir = f"{base}/logs/"
        self.conf.miscellaneous.intermediate_dir = f"{base}/out/interim/"
        self.conf.miscellaneous.mapping_dir = f"{base}/out/mapping/"
        self.conf.miscellaneous.stats_dir = f"{base}/out/stats/"
        self.conf.secnet.out_dir = f"{base}/out/secnet/"
        self.conf.primnet.out_dir = f"{base}/out/primnet/"
        self.conf.finalnet.out_dir = f"{base}/out/finalnet/"
        self.conf.figures.directory = f"{base}/figs"

        write_homes_csv(base / f"data/load/{REGION}-home-load.csv", homes)
        # one substation inside the convex hull of the homes
        lon = sum(h.cord[0] for h in homes) / len(homes)
        lat = sum(h.cord[1] for h in homes) / len(homes)
        (base / "substations.csv").write_text(f"ID,X,Y\n{SUBSTATION_ID},{lon},{lat}\n")

        self.out = {
            "edges": base / f"out/interim/{REGION}_road_edges.csv",
            "nodes": base / f"out/interim/{REGION}_road_nodes.csv",
            "h2r": base / f"out/mapping/{REGION}_map_h2r.txt",
            "transformers": base / f"out/secnet/{REGION}_transformers.csv",
            "secnet_edges": base / f"out/secnet/{REGION}_combined_network_edges.csv",
            "secnet_nodes": base / f"out/secnet/{REGION}_combined_network_nodes.csv",
            "assignment": base / f"out/interim/{REGION}_transformer_assignment.json",
            "primnet_nodes": base / f"out/primnet/{REGION}_{SUBSTATION_ID}_nodes.csv",
            "primnet_edges": base / f"out/primnet/{REGION}_{SUBSTATION_ID}_edges.csv",
            "final_nodes": base / f"out/finalnet/{REGION}_distribution_network_nodes.csv",
            "final_edges": base / f"out/finalnet/{REGION}_distribution_network_edges.csv",
            "checkpoints": base / f"out/interim/{REGION}_checkpoints.json",
            "stats": base / f"out/stats/{REGION}_run_stats.json",
            "log": base / f"logs/application_{REGION}.log",
        }

    def run(self, osm_mock, *extra_args):
        config_path = self.base / "config.yaml"
        OmegaConf.save(self.conf, config_path)
        # main.py reads configs/logging_config.ini relative to the working directory
        self.monkeypatch.chdir(REPO_ROOT)
        self.monkeypatch.setattr(sys, "argv", ["main.py", "-c", str(config_path), "-r", REGION, *extra_args])
        self.monkeypatch.setattr(osm_utils.ox, "graph_from_polygon", osm_mock)
        log = self.out["log"]
        start = len(log.read_text()) if log.exists() else 0
        main_module.main()
        return log.read_text()[start:]  # log of this run only

    def snapshot(self, *names):
        return {name: self.out[name].read_text() for name in names}


@pytest.fixture
def pipeline(tmp_path, homes, monkeypatch, restore_logging):
    return Pipeline(tmp_path, homes, monkeypatch)


def no_osm():
    return MagicMock(side_effect=AssertionError("OSM should not be queried when roads are reused"))


def reused_stages(log):
    return {stage for stage, msg in STAGE_LOGS.items() if msg in log}


# First run

def test_main_first_run_creates_outputs(pipeline, road_grid, homes):
    osm = MagicMock(return_value=road_grid)
    log = pipeline.run(osm)

    osm.assert_called_once()
    out = pipeline.out
    for name, path in out.items():
        assert path.exists(), f"{name} output missing: {path}"
    assert reused_stages(log) == set()

    # every home is mapped to a road edge
    lines = out["h2r"].read_text().splitlines()
    assert len(lines) == len(homes)
    assert {int(line.split("id=")[1].split(",")[0]) for line in lines} == {h.id for h in homes}

    assert "Residence and road data loading complete" in log
    assert "Residence and road data mapping complete" in log
    assert "Transformer partitioning and partition data loading complete" in log

    # every transformer of the combined network is assigned to the single substation
    from utils.partition_utils import load_partitioning
    from utils.secnet_utils import load_combined_network
    combined = load_combined_network(str(out["secnet_nodes"]), str(out["secnet_edges"]))
    transformers = {str(n) for n, d in combined.nodes(data=True) if d["label"] == "T"}
    partition = load_partitioning(str(out["assignment"]))
    assert list(partition) == [int(SUBSTATION_ID)]
    assert transformers
    assert transformers <= set(partition[int(SUBSTATION_ID)]["nodelist"])

    stages = json.loads(out["checkpoints"].read_text())["stages"]
    assert list(stages) == ["roads", "mapping", "secnet", "partition", "primnet", "finalnet"]
    assert all(s["status"] == "complete" for s in stages.values())


def test_main_missing_home_csv(pipeline, road_grid):
    (pipeline.base / f"data/load/{REGION}-home-load.csv").unlink()
    with pytest.raises(FileNotFoundError):
        pipeline.run(MagicMock(return_value=road_grid))


def test_main_skips_road_link_without_geometry(pipeline, road_grid, homes, monkeypatch):
    """A road link missing from the road network is skipped, not built with the previous link's geometry."""
    from utils import mapping

    missing_link = (1, 2, 0)
    compute = mapping.compute_edge_to_homes_map

    def with_missing_link(h2r):
        r2h = compute(h2r)
        return {missing_link: homes[:1], **r2h}  # first in iteration order

    generated = []
    generate = secnet_utils.SecondaryNetworkGenerator.generate_network_for_link

    def spy(self, road_link, road_geom, mapped_homes, **kwargs):
        generated.append((road_link, road_geom))
        return generate(self, road_link, road_geom, mapped_homes, **kwargs)

    monkeypatch.setattr(mapping, "compute_edge_to_homes_map", with_missing_link)
    monkeypatch.setattr(secnet_utils.SecondaryNetworkGenerator, "generate_network_for_link", spy)
    log = pipeline.run(MagicMock(return_value=road_grid))

    assert missing_link not in [link for link, _ in generated]
    assert len(generated) == len(compute(mapping.read_mapping_from_file(homes, pipeline.out["h2r"])))
    assert f"Cannot extract geometry for road link {missing_link}; skipping it" in log


def test_main_partition_failure_raises_original_error(pipeline, road_grid):
    # a 4 x 4 grid is too small: NetworkPartitioner drops cells with <= 100 nodes
    with pytest.raises(ValueError, match="No valid centers"):
        pipeline.run(MagicMock(return_value=make_road_grid(4)))
    stages = json.loads(pipeline.out["checkpoints"].read_text())["stages"]
    assert stages["partition"]["status"] == "in_progress"


# Reusing checkpoints

ALL_OUTPUTS = (
    "edges", "nodes", "h2r", "transformers", "secnet_edges", "secnet_nodes", "assignment",
    "primnet_nodes", "primnet_edges", "final_nodes", "final_edges",
)


def test_main_second_run_reuses_every_stage(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid))
    before = pipeline.snapshot(*ALL_OUTPUTS)

    osm = no_osm()
    log = pipeline.run(osm)

    osm.assert_not_called()
    assert reused_stages(log) == set(STAGE_LOGS)
    assert pipeline.snapshot(*ALL_OUTPUTS) == before


@pytest.mark.parametrize("use_flag", ["cli", "config"])
def test_main_from_scratch_reruns_every_stage(pipeline, road_grid, use_flag):
    pipeline.run(MagicMock(return_value=road_grid))
    before = pipeline.snapshot(*ALL_OUTPUTS)

    osm = MagicMock(return_value=road_grid)
    if use_flag == "cli":
        log = pipeline.run(osm, "--from-scratch")
    else:
        pipeline.conf.miscellaneous.from_scratch = True
        log = pipeline.run(osm)

    osm.assert_called_once()
    assert reused_stages(log) == set()
    assert "Running all stages from scratch" in log
    assert pipeline.snapshot(*ALL_OUTPUTS) == before, "recomputing must reproduce the same outputs"


def test_main_changed_secnet_parameters_rerun_secnet_and_partition(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid))
    before = pipeline.snapshot("edges", "nodes", "h2r")

    pipeline.conf.secnet.secnet_args.max_hops = 1
    log = pipeline.run(no_osm())

    assert reused_stages(log) == {"roads", "mapping"}
    assert pipeline.snapshot("edges", "nodes", "h2r") == before


def test_main_changed_padding_reruns_mapping_and_later_stages(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid))
    pipeline.conf.miscellaneous.padding = 0.002
    log = pipeline.run(no_osm())
    assert reused_stages(log) == {"roads"}


def test_main_changed_homes_refetch_roads(pipeline, road_grid, homes):
    pipeline.run(MagicMock(return_value=road_grid))
    homes[0].profile = [x + 1 for x in homes[0].profile]
    write_homes_csv(pipeline.base / f"data/load/{REGION}-home-load.csv", homes)

    osm = MagicMock(return_value=road_grid)
    log = pipeline.run(osm)
    osm.assert_called_once()
    assert reused_stages(log) == set()


def test_main_changed_substations_rerun_partition_only(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid))
    path = pipeline.base / "substations.csv"
    path.write_text(path.read_text().replace(SUBSTATION_ID, "5353"))

    log = pipeline.run(no_osm())
    assert reused_stages(log) == {"roads", "mapping", "secnet"}
    assert list(json.loads(pipeline.out["assignment"].read_text())) == ["5353"]
    assert (pipeline.base / f"out/primnet/{REGION}_5353_edges.csv").exists()
    assert not pipeline.out["primnet_edges"].exists(), "outputs of the old substation are removed"


def test_main_missing_output_reruns_that_stage(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid))
    pipeline.out["h2r"].unlink()
    log = pipeline.run(no_osm())
    assert reused_stages(log) == {"roads"}
    assert pipeline.out["h2r"].exists()


def test_main_outputs_without_checkpoints_are_regenerated(pipeline, road_grid):
    """
    Outputs from runs before checkpointing (e.g. with the old edge-direction bug) are not
    trusted, except the raw road download, which is adopted to avoid querying OSM again.
    """
    pipeline.run(MagicMock(return_value=road_grid))
    pipeline.out["secnet_nodes"].write_text("stale\n")
    pipeline.out["checkpoints"].unlink()

    log = pipeline.run(no_osm())
    assert "Adopting existing outputs of stage 'roads'" in log
    assert reused_stages(log) == {"roads"}
    assert pipeline.out["secnet_nodes"].read_text() != "stale\n"


def test_main_from_scratch_does_not_adopt_road_files(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid))
    pipeline.out["checkpoints"].unlink()
    osm = MagicMock(return_value=road_grid)
    pipeline.run(osm, "--from-scratch")
    osm.assert_called_once()


# Resuming an interrupted secondary network stage

def test_main_resumes_interrupted_secnet_stage(tmp_path, homes, monkeypatch, restore_logging, road_grid):
    clean = Pipeline(tmp_path / "clean", homes, monkeypatch)
    clean.run(MagicMock(return_value=road_grid))

    run = Pipeline(tmp_path / "resumed", homes, monkeypatch)
    n_links = len(clean.out["h2r"].read_text().splitlines())  # one home per road link here
    assert n_links >= 3

    # Interrupt while saving the 3rd road link: its transformers and edges are written,
    # its road transformer sequence (written last) is not
    calls = {"n": 0}
    save_sequence = secnet_utils.SecondaryNetworkGenerator._save_road_transformer_sequence

    def crash_on_third(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return save_sequence(self, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(secnet_utils.SecondaryNetworkGenerator, "_save_road_transformer_sequence", crash_on_third)
        with pytest.raises(KeyboardInterrupt):
            run.run(MagicMock(return_value=road_grid))
    assert json.loads(run.out["checkpoints"].read_text())["stages"]["secnet"]["status"] == "in_progress"

    generated = []
    generate = secnet_utils.SecondaryNetworkGenerator.generate_network_for_link

    def spy(self, road_link, *args, **kwargs):
        generated.append(road_link)
        return generate(self, road_link, *args, **kwargs)

    monkeypatch.setattr(secnet_utils.SecondaryNetworkGenerator, "generate_network_for_link", spy)
    log = run.run(no_osm())

    assert reused_stages(log) == {"roads", "mapping"}
    assert "Resuming secondary network generation: 2 road links already finished" in log
    secnet_stats = json.loads(run.out["stats"].read_text())["stages"]["secnet"]
    assert secnet_stats["status"] == "resumed"
    assert secnet_stats["road_links_finished_before"] == 2
    assert secnet_stats["road_links_generated"] == n_links - 2
    assert len(generated) == n_links - 2, "finished road links must not be generated again"
    outputs = ("transformers", "secnet_edges", "secnet_nodes", "assignment", "primnet_nodes", "primnet_edges")
    assert run.snapshot(*outputs) == clean.snapshot(*outputs)


def test_main_does_not_resume_with_changed_parameters(pipeline, road_grid, monkeypatch):
    calls = {"n": 0}
    save_sequence = secnet_utils.SecondaryNetworkGenerator._save_road_transformer_sequence

    def crash_on_second(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return save_sequence(self, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(secnet_utils.SecondaryNetworkGenerator, "_save_road_transformer_sequence", crash_on_second)
        with pytest.raises(KeyboardInterrupt):
            pipeline.run(MagicMock(return_value=road_grid))

    pipeline.conf.secnet.secnet_args.max_hops = 2
    log = pipeline.run(no_osm())
    assert "Resuming secondary network generation" not in log


# Run statistics

def read_stats(pipeline):
    return json.loads(pipeline.out["stats"].read_text())


def test_main_writes_run_statistics(pipeline, road_grid, homes):
    pipeline.run(MagicMock(return_value=road_grid))
    stats = read_stats(pipeline)

    assert stats["status"] == "complete"
    assert stats["region"] == REGION and stats["secnet_method"] == "heuristic"
    assert stats["total_seconds"] >= sum(s["seconds"] for s in stats["stages"].values()) - 1e-3
    assert list(stats["stages"]) == ["roads", "mapping", "secnet", "partition", "primnet", "finalnet"]
    assert all(s["status"] == "computed" for s in stats["stages"].values())

    roads, mapping = stats["stages"]["roads"], stats["stages"]["mapping"]
    assert roads["homes"] == len(homes)
    assert roads["road_nodes"] == 144 and roads["road_edges"] == 2 * 12 * 11
    assert mapping["homes_mapped"] == len(homes) and mapping["homes_unmapped"] == 0
    assert mapping["road_links_with_homes"] == len(homes)  # one home per link in this fixture

    secnet = stats["stages"]["secnet"]
    assert secnet["road_links_generated"] == len(homes)
    assert secnet["homes_served"] == len(homes)
    assert secnet["transformer_load_total"] == pytest.approx(sum(h.load for h in homes), rel=1e-6)
    assert secnet["combined_components"] == 1
    assert secnet["combined_nodes"] == roads["road_nodes"] + secnet["transformers"]

    partition = stats["stages"]["partition"]
    assert partition["substations_assigned"] == 1
    assert partition["transformers_assigned"] == secnet["transformers"]
    assert partition["per_substation"][SUBSTATION_ID]["load"] == pytest.approx(secnet["transformer_load_total"])

    primnet = stats["stages"]["primnet"]
    assert stats["primnet_method"] == primnet["primnet_method"] == "heuristic"
    assert primnet["service_substations"] == 1 and primnet["substations_generated"] == 1
    assert primnet["feeders"] >= 1 and primnet["primary_length_km"] > 0
    assert primnet["per_substation"][SUBSTATION_ID]["seconds"] >= 0


def test_main_statistics_mark_reused_stages(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid))
    first = read_stats(pipeline)
    pipeline.run(no_osm())
    second = read_stats(pipeline)

    assert all(s["status"] == "reused" for s in second["stages"].values())
    # data sizes are computed from the saved outputs, so they match the first run
    run_specific = (
        "status", "road_links_generated", "road_links_finished_before",
        "substations_generated", "substations_finished_before",
    )

    def sizes(stage):
        out = {k: v for k, v in stage.items() if "seconds" not in k and k not in run_specific}
        if "per_substation" in out:
            out["per_substation"] = {
                sid: {k: v for k, v in sub.items() if k != "seconds"} for sid, sub in out["per_substation"].items()
            }
        return out

    for name, stage in first["stages"].items():
        assert sizes(second["stages"][name]) == sizes(stage)


def test_main_statistics_record_failure(pipeline):
    with pytest.raises(ValueError):
        pipeline.run(MagicMock(return_value=make_road_grid(4)))
    stats = read_stats(pipeline)
    assert stats["status"] == "failed" and stats["failed_stage"] == "partition"
    assert stats["stages"]["partition"]["status"] == "failed"
    assert "No valid centers" in stats["stages"]["partition"]["error"]
    assert stats["stages"]["secnet"]["status"] == "computed"



# Primary network stage

def test_main_primary_network_covers_all_transformers(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid))
    import pandas as pd
    nodes = pd.read_csv(pipeline.out["primnet_nodes"], dtype={"node_id": str})
    edges = pd.read_csv(pipeline.out["primnet_edges"], dtype={"source": str, "target": str})
    transformers = set(pd.read_csv(pipeline.out["transformers"])["transformer_id"])
    assert transformers <= set(nodes["node_id"])
    assert set(edges["label"]) == {"primary", "feeder"}
    assert len(edges) == len(nodes) - 1, "one tree rooted at the substation"
    assert (nodes["label"] == "S").sum() == 1


def test_main_changed_primnet_parameters_rerun_primnet_only(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid))
    pipeline.conf.primnet.primnet_args.maximum_branch_flow = 300
    log = pipeline.run(no_osm())
    assert reused_stages(log) == {"roads", "mapping", "secnet", "partition"}


def test_main_resumes_interrupted_primnet_stage(pipeline, road_grid, monkeypatch):
    from utils import primnet_utils

    with monkeypatch.context() as m:
        m.setattr(primnet_utils.PrimaryNetworkGenerator, "export_to_csv", MagicMock(side_effect=KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            pipeline.run(MagicMock(return_value=road_grid))
    assert json.loads(pipeline.out["checkpoints"].read_text())["stages"]["primnet"]["status"] == "in_progress"

    log = pipeline.run(no_osm())
    assert reused_stages(log) == {"roads", "mapping", "secnet", "partition"}
    assert pipeline.out["final_edges"].exists()
    primnet = json.loads(pipeline.out["stats"].read_text())["stages"]["primnet"]
    assert primnet["status"] == "resumed" and primnet["substations_generated"] == 1
    assert pipeline.out["primnet_edges"].exists()



# Final distribution network

def test_main_final_network_connects_every_home(pipeline, road_grid, homes):
    pipeline.run(MagicMock(return_value=road_grid))
    from utils.finalnet_utils import load_distribution_network
    network = load_distribution_network(pipeline.out["final_nodes"], pipeline.out["final_edges"])

    assert nx.is_tree(network)
    labels = [d["label"] for _, d in network.nodes(data=True)]
    assert labels.count("S") == 1
    assert {str(h.id) for h in homes} <= set(network.nodes)
    assert {d["label"] for _, _, d in network.edges(data=True)} == {"feeder", "primary", "secondary"}

    final = json.loads(pipeline.out["stats"].read_text())["stages"]["finalnet"]
    assert final["is_forest"] and final["trees"] == 1
    assert final["homes_missing"] == 0 and final["trees_without_substation"] == 0
    assert final["nodes_by_label"]["H"] == len(homes)


def test_main_plot_saved_in_figures_directory(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid), "-p")
    figdir = pipeline.base / "figs"
    assert (figdir / f"{REGION}_distribution_network.png").exists()
    assert (figdir / f"{REGION}_combined_network.png").exists()
