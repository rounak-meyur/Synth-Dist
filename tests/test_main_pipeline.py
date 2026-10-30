"""
End-to-end check of main.py: config + logging, home loading, road loading/caching,
home-to-road mapping, secondary network generation and transformer partitioning,
including checkpointing (reuse, invalidation, resume and --from-scratch), groups of
several regions and running the stages as separate jobs.

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
REGION_STAGES = ["roads", "mapping", "secnet"]
GROUP_STAGES = ["partition", "primnet", "finalnet"]
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
    """main.py configured to read and write everything below `base`, for a group of one region."""

    def __init__(self, base, homes, monkeypatch, group=REGION, regions=None):
        self.base, self.monkeypatch = base, monkeypatch
        regions = regions or {REGION: homes}
        self.conf = OmegaConf.load(REPO_ROOT / "configs/config_heuristic.yaml")
        # main.py builds paths by string concatenation, so directories need a trailing slash
        self.conf.inputs.substation_csv = str(base / "substations.csv")
        self.conf.miscellaneous.log_dir = f"{base}/logs/"
        self.conf.miscellaneous.intermediate_dir = f"{base}/out/interim/"
        self.conf.miscellaneous.mapping_dir = f"{base}/out/mapping/"
        self.conf.miscellaneous.stats_dir = f"{base}/out/stats/"
        self.conf.secnet.out_dir = f"{base}/out/secnet/"
        self.conf.primnet.out_dir = f"{base}/out/primnet/"
        self.conf.finalnet.out_dir = f"{base}/out/finalnet/"
        self.conf.figures.directory = f"{base}/figs"

        base.mkdir(parents=True, exist_ok=True)
        self.group_file = base / "group.yaml"
        OmegaConf.save(OmegaConf.create({
            "state": 51,
            "inputs": {"home_csv_dir": f"{base}/data/load/"},
            "group": {"name": group, "regions": list(regions)},
        }), self.group_file)
        for region, region_homes in regions.items():
            write_homes_csv(base / f"data/load/{region}-home-load.csv", region_homes)
        # one substation inside the convex hull of the homes
        all_homes = [h for region_homes in regions.values() for h in region_homes]
        lon = sum(h.cord[0] for h in all_homes) / len(all_homes)
        lat = sum(h.cord[1] for h in all_homes) / len(all_homes)
        (base / "substations.csv").write_text(f"ID,X,Y\n{SUBSTATION_ID},{lon},{lat}\n")

        self.out = {
            "edges": base / f"out/interim/{REGION}_road_edges.csv",
            "nodes": base / f"out/interim/{REGION}_road_nodes.csv",
            "h2r": base / f"out/mapping/{REGION}_map_h2r.txt",
            "transformers": base / f"out/secnet/{REGION}_transformers.csv",
            "secnet_edges": base / f"out/secnet/{REGION}_combined_network_edges.csv",
            "secnet_nodes": base / f"out/secnet/{REGION}_combined_network_nodes.csv",
            "assignment": base / f"out/interim/{group}_partition/{SUBSTATION_ID}.json",
            "substation_list": base / f"out/interim/{group}_partition/substations.txt",
            "primnet_nodes": base / f"out/primnet/{group}_{SUBSTATION_ID}_nodes.csv",
            "primnet_edges": base / f"out/primnet/{group}_{SUBSTATION_ID}_edges.csv",
            "final_nodes": base / f"out/finalnet/{group}_distribution_network_nodes.csv",
            "final_edges": base / f"out/finalnet/{group}_distribution_network_edges.csv",
            "checkpoints": base / f"out/interim/{REGION}_checkpoints.json",
            "group_checkpoints": base / f"out/interim/{group}_group_checkpoints.json",
            "primnet_checkpoints": base / f"out/interim/{group}_primnet/{SUBSTATION_ID}_checkpoints.json",
            "stats": base / f"out/stats/{REGION}_run_stats.json",
            "group_stats": base / f"out/stats/{group}_group_run_stats.json",
            "primnet_stats": base / f"out/stats/{group}_primnet/{SUBSTATION_ID}_run_stats.json",
            "log": base / f"logs/application_{group}.log",
        }

    def run(self, osm_mock, *args):
        """Run main.py with `args` (stage first, if any); returns the log of this run."""
        config_path = self.base / "config.yaml"
        OmegaConf.save(self.conf, config_path)
        # main.py reads configs/logging_config.ini relative to the working directory
        self.monkeypatch.chdir(REPO_ROOT)
        self.monkeypatch.setattr(
            sys, "argv", ["main.py", *args, "-c", str(config_path), "-g", str(self.group_file)]
        )
        self.monkeypatch.setattr(osm_utils.ox, "graph_from_polygon", osm_mock)
        # every stage and task writes its own log file
        logs = self.base / "logs"
        before = {p: len(p.read_text()) for p in logs.glob("*.log")} if logs.exists() else {}
        main_module.main()
        return "".join(p.read_text()[before.get(p, 0):] for p in sorted(logs.glob("*.log")))

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
    from utils.partition_utils import load_substation_partition
    from utils.secnet_utils import load_combined_network
    combined = load_combined_network(str(out["secnet_nodes"]), str(out["secnet_edges"]))
    transformers = {str(n) for n, d in combined.nodes(data=True) if d["label"] == "T"}
    assert out["substation_list"].read_text().split() == [SUBSTATION_ID]
    sub, partition = load_substation_partition(str(out["assignment"].parent), SUBSTATION_ID)
    assert sub.id == SUBSTATION_ID
    assert transformers
    assert transformers <= set(partition["nodelist"])

    for name, expected in (
        ("checkpoints", ["roads", "mapping", "secnet"]),
        ("group_checkpoints", ["partition", "finalnet"]),
        ("primnet_checkpoints", ["primnet"]),
    ):
        stages = json.loads(out[name].read_text())["stages"]
        assert list(stages) == expected
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
    stages = json.loads(pipeline.out["group_checkpoints"].read_text())["stages"]
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


def test_main_changed_secnet_parameters_rerun_secnet(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid))
    before = pipeline.snapshot("edges", "nodes", "h2r")

    pipeline.conf.secnet.secnet_args.max_hops = 1
    log = pipeline.run(no_osm())

    # one home per road link: the secondary network is the same, so the group stages are reused
    assert reused_stages(log) == {"roads", "mapping", "partition", "primnet", "finalnet"}
    assert read_stats(pipeline)["stages"]["secnet"]["road_links_generated"] > 0
    assert pipeline.snapshot("edges", "nodes", "h2r") == before


def test_main_changed_padding_reruns_mapping_and_later_stages(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid))
    pipeline.conf.miscellaneous.padding = 0.002
    log = pipeline.run(no_osm())
    # the larger padding maps the homes the same way here, so the substation's partition is
    # unchanged and its primary network is reused
    assert reused_stages(log) == {"roads", "primnet"}


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
    assert pipeline.out["substation_list"].read_text().split() == ["5353"]
    assert not pipeline.out["assignment"].exists()
    assert (pipeline.base / f"out/primnet/{REGION}_5353_edges.csv").exists()
    assert not pipeline.out["primnet_edges"].exists(), "outputs of the old substation are removed"


def test_main_partition_rerun_keeps_unchanged_primary_networks(pipeline, road_grid):
    """Rerunning the partition stage from scratch does not discard primary networks it would rebuild unchanged."""
    pipeline.run(MagicMock(return_value=road_grid))
    pipeline.run(no_osm(), "partition", "-f")
    assert pipeline.out["primnet_edges"].exists()
    log = pipeline.run(no_osm(), "primnet")
    assert f"Reusing saved primary network of substation {SUBSTATION_ID}" in log


def test_main_missing_output_reruns_that_stage(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid))
    pipeline.out["h2r"].unlink()
    log = pipeline.run(no_osm())
    # the region's later stages are rerun; the group stages see identical inputs and are reused
    assert reused_stages(log) == {"roads", "partition", "primnet", "finalnet"}
    assert pipeline.out["h2r"].exists()


def test_main_group_stages_rerun_when_their_inputs_change(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid))
    edges = pipeline.out["secnet_edges"]
    edges.write_text(edges.read_text().replace("residential", "tertiary"))
    log = pipeline.run(no_osm())
    reused = reused_stages(log)
    assert {"roads", "mapping", "secnet"} <= reused
    assert "partition" not in reused, "the partition reads the changed combined network"

    pipeline.out["primnet_checkpoints"].unlink()
    log = pipeline.run(no_osm())
    assert reused_stages(log) == {"roads", "mapping", "secnet", "partition", "finalnet"}, (
        "the primary network is regenerated; it is identical, so the final network is reused"
    )


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
    # the regenerated combined network is the one the saved partition was made from
    assert reused_stages(log) == {"roads", "partition", "primnet", "finalnet"}
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

def read_stats(pipeline, name="stats"):
    return json.loads(pipeline.out[name].read_text())


def test_main_writes_run_statistics(pipeline, road_grid, homes):
    pipeline.run(MagicMock(return_value=road_grid))
    region_stats, group_stats = read_stats(pipeline), read_stats(pipeline, "group_stats")

    for stats, stages in ((region_stats, REGION_STAGES), (group_stats, GROUP_STAGES)):
        assert stats["status"] == "complete"
        assert stats["total_seconds"] >= sum(s["seconds"] for s in stats["stages"].values()) - 1e-3
        assert list(stats["stages"]) == stages
        assert all(s["status"] == "computed" for s in stats["stages"].values())
    assert region_stats["region"] == REGION and region_stats["secnet_method"] == "heuristic"
    assert group_stats["group"] == REGION and group_stats["regions"] == [REGION]
    stats = {**group_stats, "stages": {**region_stats["stages"], **group_stats["stages"]}}

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
    first = {name: read_stats(pipeline, name) for name in ("stats", "group_stats", "primnet_stats")}
    pipeline.run(no_osm())
    second = {name: read_stats(pipeline, name) for name in ("stats", "group_stats", "primnet_stats")}
    first = {"stages": {f"{n}/{k}": v for n, st in first.items() for k, v in st["stages"].items()}}
    second = {"stages": {f"{n}/{k}": v for n, st in second.items() for k, v in st["stages"].items()}}

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
    stats = read_stats(pipeline, "group_stats")
    assert stats["status"] == "failed" and stats["failed_stage"] == "partition"
    assert stats["stages"]["partition"]["status"] == "failed"
    assert "No valid centers" in stats["stages"]["partition"]["error"]
    region_stats = read_stats(pipeline)
    assert region_stats["status"] == "complete"
    assert region_stats["stages"]["secnet"]["status"] == "computed"



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
    reused = reused_stages(log)
    assert {"roads", "mapping", "secnet", "partition"} <= reused
    assert "primnet" not in reused
    assert read_stats(pipeline, "primnet_stats")["stages"]["primnet"]["status"] == "computed"


def test_main_resumes_interrupted_primnet_stage(pipeline, road_grid, monkeypatch):
    from utils import primnet_utils

    with monkeypatch.context() as m:
        m.setattr(primnet_utils.PrimaryNetworkGenerator, "export_to_csv", MagicMock(side_effect=KeyboardInterrupt))
        with pytest.raises(KeyboardInterrupt):
            pipeline.run(MagicMock(return_value=road_grid))
    stages = json.loads(pipeline.out["primnet_checkpoints"].read_text())["stages"]
    assert stages["primnet"]["status"] == "in_progress"

    log = pipeline.run(no_osm())
    assert reused_stages(log) == {"roads", "mapping", "secnet", "partition"}
    assert pipeline.out["final_edges"].exists()
    primnet = read_stats(pipeline, "group_stats")["stages"]["primnet"]
    assert primnet["status"] == "computed" and primnet["substations_generated"] == 1
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

    final = read_stats(pipeline, "group_stats")["stages"]["finalnet"]
    assert final["is_forest"] and final["trees"] == 1
    assert final["homes_missing"] == 0 and final["trees_without_substation"] == 0
    assert final["nodes_by_label"]["H"] == len(homes)


def test_main_plot_saved_in_figures_directory(pipeline, road_grid):
    pipeline.run(MagicMock(return_value=road_grid), "-p")
    figdir = pipeline.base / "figs"
    assert (figdir / f"{REGION}_distribution_network.png").exists()
    assert (figdir / f"{REGION}_combined_network.png").exists()


# Groups of several regions

GROUP = "pair"
OTHER_REGION = "778"


@pytest.fixture
def pair_regions(homes):
    """Two regions on the same road grid: the second region's homes are 6 columns to the east."""
    from conftest import GRID_STEP
    from utils.dataloader import Home
    shifted = [
        Home(id=h.id + 1000, cord=(h.cord[0] + 6 * GRID_STEP, h.cord[1]), profile=h.profile, peak=h.peak, load=h.load)
        for h in homes
    ]
    return {REGION: homes, OTHER_REGION: shifted}


def pair_pipeline(base, pair_regions, monkeypatch):
    return Pipeline(base, None, monkeypatch, group=GROUP, regions=pair_regions)


@pytest.fixture
def pair(tmp_path, pair_regions, monkeypatch, restore_logging):
    return pair_pipeline(tmp_path, pair_regions, monkeypatch)


def run_stages_as_jobs(pipeline, road_grid):
    """The jobs of scripts/slurm/submit.sh, one after the other; returns their logs."""
    log = ""
    for task in range(2):
        log += pipeline.run(MagicMock(return_value=road_grid), "secnet", "--task-index", str(task), "--task-count", "2")
    log += pipeline.run(no_osm(), "partition")
    for task in range(3):  # more tasks than substations: the extra tasks have nothing to do
        log += pipeline.run(no_osm(), "primnet", "--task-index", str(task), "--task-count", "3")
    return log + pipeline.run(no_osm(), "finalnet")


def test_main_group_combines_all_regions(pair, road_grid, pair_regions):
    pair.run(MagicMock(return_value=road_grid))
    from utils.finalnet_utils import load_distribution_network
    network = load_distribution_network(pair.out["final_nodes"], pair.out["final_edges"])

    assert nx.is_tree(network), "one substation serves the homes of both regions"
    all_homes = [h for region_homes in pair_regions.values() for h in region_homes]
    assert {str(h.id) for h in all_homes} <= set(network.nodes)
    for region in pair_regions:
        assert (pair.base / f"out/stats/{region}_run_stats.json").exists()
    final = read_stats(pair, "group_stats")["stages"]["finalnet"]
    assert final["homes_missing"] == 0 and final["nodes_by_label"]["H"] == len(all_homes)


def test_main_group_merge_drops_plain_copies_of_replaced_road_links(pair, road_grid):
    """A road link with transformers of one region is a plain road edge in the other region's network."""
    pair.run(MagicMock(return_value=road_grid))
    from utils.secnet_utils import merge_combined_networks
    secnet_dir = pair.base / "out/secnet"
    merged = merge_combined_networks([REGION, OTHER_REGION], str(secnet_dir))

    replaced = [
        tuple(int(x) for x in line.split()[:3])
        for region in (REGION, OTHER_REGION)
        for line in (secnet_dir / f"{region}_road_transformer_edges.txt").read_text().splitlines()
    ]
    assert len(replaced) == 10
    for u, v, k in replaced:
        assert not merged.has_edge(u, v, key=k)
        assert nx.has_path(merged, u, v), "the road with the transformers on it remains"
    transformers = [n for n, d in merged.nodes(data=True) if d["label"] == "T"]
    assert transformers and nx.is_connected(merged)


def test_main_stages_as_separate_jobs_match_a_single_run(tmp_path, pair_regions, monkeypatch, restore_logging, road_grid):
    single = pair_pipeline(tmp_path / "single", pair_regions, monkeypatch)
    single.run(MagicMock(return_value=road_grid))

    jobs = pair_pipeline(tmp_path / "jobs", pair_regions, monkeypatch)
    run_stages_as_jobs(jobs, road_grid)

    outputs = ("assignment", "primnet_nodes", "primnet_edges", "final_nodes", "final_edges")
    assert jobs.snapshot(*outputs) == single.snapshot(*outputs)
    logs = {p.name for p in (jobs.base / "logs").glob("*.log")}
    assert {f"application_{GROUP}_secnet_0.log", f"application_{GROUP}_secnet_1.log",
            f"application_{GROUP}_partition.log", f"application_{GROUP}_finalnet.log"} <= logs

    # the finalnet job keeps the statistics of the partition job
    stats = read_stats(jobs, "group_stats")
    assert list(stats["stages"]) == GROUP_STAGES
    assert stats["stages"]["primnet"]["substations_generated"] == 1

    # running the jobs again reuses everything
    log = run_stages_as_jobs(jobs, road_grid)
    assert reused_stages(log) == set(STAGE_LOGS)


def test_main_group_stages_need_the_earlier_stages(pair, road_grid):
    pair.run(MagicMock(return_value=road_grid), "secnet", "-r", REGION)
    with pytest.raises(RuntimeError, match=f"run the secnet stage for region {OTHER_REGION}"):
        pair.run(no_osm(), "partition")

    pair.run(MagicMock(return_value=road_grid), "secnet", "-r", OTHER_REGION)
    with pytest.raises(FileNotFoundError, match="run the partition stage first"):
        pair.run(no_osm(), "primnet")
    pair.run(no_osm(), "partition")
    with pytest.raises(RuntimeError, match="run the primnet stage"):
        pair.run(no_osm(), "finalnet")


def test_main_list_tasks(pair, road_grid, capsys):
    def listed(*args):
        capsys.readouterr()
        pair.run(no_osm(), *args, "--list")
        return capsys.readouterr().out.split()

    assert listed("secnet") == [REGION, OTHER_REGION]
    assert listed("secnet", "--task-index", "1", "--task-count", "2") == [OTHER_REGION]
    assert listed("secnet", "-r", "999") == ["999"]

    pair.run(MagicMock(return_value=road_grid), "secnet")
    pair.run(no_osm(), "partition")
    assert listed("primnet") == [SUBSTATION_ID]
    assert listed("primnet", "--task-index", "1", "--task-count", "2") == []


def test_main_regions_option_runs_a_group_of_those_regions(pair, road_grid):
    """-r replaces the group's regions; group outputs are named after them."""
    pair.run(MagicMock(return_value=road_grid), "-r", OTHER_REGION, REGION)
    assert (pair.base / f"out/finalnet/{OTHER_REGION}_{REGION}_distribution_network_edges.csv").exists()
    assert not pair.out["final_edges"].exists()

    # the substation lies between the two regions, outside the homes of either one
    with pytest.raises(ValueError, match=f"No substation .* group {REGION}"):
        pair.run(no_osm(), "-r", REGION)


@pytest.mark.parametrize("args", [
    ["partition", "--task-count", "2", "--task-index", "1"],
    ["secnet", "--task-count", "2", "--task-index", "2"],
])
def test_main_rejects_invalid_task_options(pair, args):
    with pytest.raises(SystemExit):
        pair.run(no_osm(), *args)


def test_main_requires_state_in_group_config(pair):
    group = OmegaConf.load(pair.group_file)
    del group["state"]
    OmegaConf.save(group, pair.group_file)
    with pytest.raises(SystemExit, match="must set the state"):
        pair.run(no_osm())


def test_main_requires_home_directory_in_group_config(pair):
    group = OmegaConf.load(pair.group_file)
    del group["inputs"]
    OmegaConf.save(group, pair.group_file)
    with pytest.raises(SystemExit, match="must set the directory of the home load files"):
        pair.run(no_osm())
