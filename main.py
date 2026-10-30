"""
Generate synthetic distribution networks for a group of regions (counties).

The regions of a group are listed in a group configuration file (configs/regions/*.yaml).
The pipeline has four stages:

    secnet     per region      road network, home to road mapping and secondary network
    partition  per group       merge the secondary networks of the regions and assign every
                               transformer to a substation serving the group
    primnet    per substation  primary network of the substation
    finalnet   per group       combine the primary and secondary networks

`all` (the default) runs every stage in order in one process. The stages can also run as
separate jobs: secnet jobs of different regions, and primnet jobs of different substations,
are independent of each other (see scripts/slurm/submit.sh).
"""
import argparse
import json
import shutil
import time
from pathlib import Path

from omegaconf import OmegaConf

STAGES = ["all", "secnet", "partition", "primnet", "finalnet"]
REGION_STAGES = ["roads", "mapping", "secnet"]
GROUP_STAGES = ["partition", "finalnet"]
DEFAULT_GROUP = "configs/regions/virginia.yaml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "stage", nargs="?", default="all", choices=STAGES,
        help = "stage to run (default: all)"
    )
    parser.add_argument(
        "-c", "--configPath",
        help = "path to configuration file",
        default = "configs/config_heuristic.yaml"
    )
    parser.add_argument(
        "-g", "--group",
        help = "path to group configuration file with the regions to run",
        default = DEFAULT_GROUP
    )
    parser.add_argument(
        "-r", "--region", type=str, nargs="+",
        help = "county IDs of the regions to run instead of the regions of the group; "
               "group outputs are then named after these regions"
    )
    parser.add_argument(
        "-s", "--substation", type=str, nargs="+",
        help = "primnet: substation IDs to run (default: every substation of the group)"
    )
    parser.add_argument(
        "--task-index", type=int, default=0,
        help = "secnet/primnet: run every task-count-th region/substation starting at this index"
    )
    parser.add_argument(
        "--task-count", type=int, default=1,
        help = "secnet/primnet: number of tasks the regions/substations are split into"
    )
    parser.add_argument(
        "--list",
        help = "print the regions (secnet) or substations (primnet) this task runs and exit",
        action = "store_true"
    )
    parser.add_argument(
        "-p", "--generate_plot",
        help = "generate plots",
        action = "store_true"
    )
    parser.add_argument(
        "-f", "--from-scratch",
        help = "ignore saved outputs and run the stages from scratch "
               "(same as miscellaneous.from_scratch: true in the config)",
        action = "store_true"
    )
    args = parser.parse_args(argv)
    if not 0 <= args.task_index < args.task_count:
        parser.error("--task-index must be between 0 and --task-count - 1")
    if args.task_count > 1 and args.stage not in ("secnet", "primnet"):
        parser.error("--task-index/--task-count apply to the secnet and primnet stages only")
    return args


class Run:
    """Configuration, regions and file locations of one invocation of the pipeline."""

    def __init__(self, conf, args):
        self.conf = conf
        self.args = args
        self.configpath = args.configPath
        self.regions = [str(r) for r in (args.region or conf["group"]["regions"])]
        self.group = "_".join(self.regions) if args.region else str(conf["group"]["name"])
        self.from_scratch = bool(args.from_scratch or conf["miscellaneous"].get("from_scratch", False))
        misc = conf["miscellaneous"]
        self.interim_dir = misc["intermediate_dir"]
        self.stats_dir = misc["stats_dir"]
        self.partition_dir = Path(f"{self.interim_dir}{self.group}_partition")
        self.primnet_dir = Path(conf["primnet"]["out_dir"])
        self.secnet_dir = Path(conf["secnet"]["out_dir"])

    @property
    def label(self):
        """Name of this invocation's log file."""
        if self.args.stage == "all":
            return self.group
        suffix = f"_{self.args.task_index}" if self.args.task_count > 1 else ""
        return f"{self.group}_{self.args.stage}{suffix}"

    def task_items(self):
        """Regions (secnet) or substations (primnet) run by this task."""
        if self.args.stage == "primnet":
            items = self.args.substation or self.substations()
        else:
            items = self.regions
        return items[self.args.task_index::self.args.task_count]

    def substations(self):
        from utils.partition_utils import read_substation_list
        try:
            return read_substation_list(self.partition_dir)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No partitioning found for group {self.group}; run the partition stage first"
            ) from None

    def home_csv(self, region):
        return f"{self.conf['inputs']['home_csv_dir']}{region}-home-load.csv"

    def load_homes(self):
        """Homes of every region of the group."""
        from utils.dataloader import load_homes
        return [home for region in self.regions for home in load_homes(file_path=self.home_csv(region))]

    def region_manifest(self, region):
        return f"{self.interim_dir}{region}_checkpoints.json"

    def secnet_params(self, region):
        secnet_args = self.conf["secnet"]["secnet_args"]
        base_tsfr_id = int(f"{self.conf['state']}{region}{0:012d}")
        return {
            "secnet_args": OmegaConf.to_container(secnet_args, resolve=True),
            "base_transformer_id": base_tsfr_id,
        }

    def secnet_files(self, region):
        """Outputs of the secnet stage used by the group stages."""
        return [
            self.secnet_dir / f"{region}_combined_network_nodes.csv",
            self.secnet_dir / f"{region}_combined_network_edges.csv",
            self.secnet_dir / f"{region}_secondary_edges.csv",
        ]

    def group_manifest(self):
        return f"{self.interim_dir}{self.group}_group_checkpoints.json"

    def group_stats_file(self):
        return f"{self.stats_dir}{self.group}_group_run_stats.json"

    def primnet_files(self, sub_id):
        return (
            self.primnet_dir / f"{self.group}_{sub_id}_nodes.csv",
            self.primnet_dir / f"{self.group}_{sub_id}_edges.csv",
        )

    def primnet_manifest(self, sub_id):
        return f"{self.interim_dir}{self.group}_primnet/{sub_id}_checkpoints.json"

    def primnet_stats_file(self, sub_id):
        return f"{self.stats_dir}{self.group}_primnet/{sub_id}_run_stats.json"

    def primnet_params(self, sub_id):
        from utils.checkpoint_utils import file_digest
        primnet_config = self.conf["primnet"]["primnet_args"]
        return {
            "primnet_args": OmegaConf.to_container(primnet_config, resolve=True),
            "partition": file_digest(self.partition_dir / f"{sub_id}.json"),
        }


def run_region(run, region):
    """Road network, home to road mapping and secondary network of one region."""
    from tqdm import tqdm
    from utils.logging_utils import LogManager, progress_logging
    from utils.checkpoint_utils import CheckpointManager, COMPLETE, IN_PROGRESS, file_digest
    from utils.stats_utils import RunStats, road_network_stats, mapping_stats, secnet_stats
    conf = run.conf
    logger = LogManager.get_logger("__main__")
    logger.info(f"Generating secondary network of region {region}")

    # Checkpoints: a stage is reused only if it finished with the same inputs/parameters
    checkpoints = CheckpointManager(
        manifest_file=run.region_manifest(region),
        stages=REGION_STAGES,
        from_scratch=run.from_scratch,
    )

    # Compute statistics and data sizes of every stage, updated after each stage
    stats = RunStats(
        output_file=f"{run.stats_dir}{region}_run_stats.json",
        region=region,
        state=conf["state"],
        config=run.configpath,
        secnet_method=conf["secnet"]["secnet_args"].get("method", "milp"),
        from_scratch=run.from_scratch,
    )

    with stats.stage("roads") as stage_stats:
        # Load inputs
        ts = time.time()
        from utils.dataloader import load_homes
        from utils.osm_utils import load_roads, save_road_network, load_road_network_from_files

        input_home_csv = run.home_csv(region)
        homes = load_homes(file_path=input_home_csv)

        road_edge_file = f"{run.interim_dir}{region}_road_edges.csv"
        road_node_file = f"{run.interim_dir}{region}_road_nodes.csv"
        roads_params = {"homes": file_digest(input_home_csv)}
        # road files are raw OSM downloads: adopt ones saved before checkpointing existed
        roads_status = checkpoints.status(
            "roads", roads_params, [road_edge_file, road_node_file], adopt_existing=True
        )
        if roads_status == COMPLETE:
            logger.info("Reusing saved road network")
            stage_stats["status"] = "reused"
        else:
            checkpoints.start("roads", roads_params)
            osm_conf = conf.get("osm") or {}
            if osm_conf.get("overpass_url"):
                import osmnx as ox
                ox.settings.overpass_url = osm_conf["overpass_url"]
            roads = load_roads(
                homes,
                max_retries=osm_conf.get("max_retries", 6),
                retry_wait=osm_conf.get("retry_wait", 30.0),
            )
            save_road_network(
                roads,
                edgelist_file=road_edge_file,
                nodelist_file=road_node_file
                )
            checkpoints.complete("roads", roads_params)
        # load the road from the file to load as an undirected graph
        roads = load_road_network_from_files(
            edgelist_file=road_edge_file,
            nodelist_file=road_node_file
            )
        stage_stats.update(homes=len(homes), **road_network_stats(roads))
        logger.info(f"Residence and road data loading complete in {time.time()-ts} seconds.")

    with stats.stage("mapping") as stage_stats:
        # Map roads and homes
        ts = time.time()
        from utils.mapping import(
            map_homes_to_edges,
            write_mapping_to_file,
            read_mapping_from_file,
            compute_edge_to_homes_map
        )

        h2r_map_file = f"{conf['miscellaneous']['mapping_dir']}{region}_map_h2r.txt"
        mapping_params = {"padding": conf["miscellaneous"]["padding"]}
        if checkpoints.status("mapping", mapping_params, [h2r_map_file]) == COMPLETE:
            logger.info("Reusing saved home to road mapping")
            stage_stats["status"] = "reused"
        else:
            checkpoints.start("mapping", mapping_params)
            h2r = map_homes_to_edges(
                roads, homes,
                padding_distance=conf["miscellaneous"]["padding"])
            write_mapping_to_file(h2r, h2r_map_file)
            checkpoints.complete("mapping", mapping_params)
        # always read the mapping back so fresh and resumed runs use identical inputs
        h2r = read_mapping_from_file(homes, filename=h2r_map_file)

        r2h = compute_edge_to_homes_map(h2r)
        stage_stats.update(mapping_stats(homes, h2r, r2h))
        logger.info(f"Residence and road data mapping complete in {time.time()-ts} seconds.")

    with stats.stage("secnet") as stage_stats:
        # Generate secondary network and combine road network with transformers
        ts = time.time()
        import pandas as pd
        from utils.secnet_utils import SecondaryNetworkGenerator, get_road_link_geometry, load_combined_network
        combined_nodes, combined_edges, secondary_edges = run.secnet_files(region)
        secnet_args = conf["secnet"]["secnet_args"]
        secnet_params = run.secnet_params(region)
        secnet_status = checkpoints.status("secnet", secnet_params, [combined_edges, combined_nodes])
        if secnet_status == COMPLETE:
            logger.info("Reusing saved secondary network")
            stage_stats["status"] = "reused"
        else:
            generator = SecondaryNetworkGenerator(
                output_dir=conf["secnet"]["out_dir"],
                base_transformer_id=secnet_params["base_transformer_id"],
            )
            if secnet_status == IN_PROGRESS:
                # continue an interrupted run from the road links that already finished
                finished = generator.resume(prefix=region)
                stage_stats["status"] = "resumed"
            else:
                generator.clear_results(prefix=region)
                finished = set()
            checkpoints.start("secnet", secnet_params)

            remaining = [road_link for road_link in r2h if road_link not in finished]
            stage_stats.update(road_links_finished_before=len(finished), road_links_generated=0)
            t_links = time.perf_counter()
            with progress_logging():
                for road_link in tqdm(
                    remaining, initial=len(r2h) - len(remaining), total=len(r2h),
                    desc="Generating secondary network for road links", unit="link",
                ):
                    try:
                        road_geometry = get_road_link_geometry(roads, road_link)
                    except KeyError:
                        logger.warning(f"Cannot extract geometry for road link {road_link}; skipping it")
                        continue
                    generator.generate_network_for_link(
                        road_link=road_link,
                        road_geom=road_geometry,
                        mapped_homes=r2h[road_link],
                        **secnet_args
                    )
                    generator.save_results(prefix=region, road_link_id=road_link)
                    stage_stats["road_links_generated"] += 1
            stage_stats["generation_seconds"] = round(time.perf_counter() - t_links, 3)
            if stage_stats["road_links_generated"]:
                stage_stats["seconds_per_road_link"] = round(
                    stage_stats["generation_seconds"] / stage_stats["road_links_generated"], 4
                )

            # Combine road network with transformers
            generator.combine_networks(road_network=roads, prefix=region)
            checkpoints.complete("secnet", secnet_params)
        combined_network = load_combined_network(
            nodes_file=combined_nodes,
            edges_file=combined_edges
            )
        stage_stats.update(secnet_stats(
            combined_network,
            transformers=pd.read_csv(run.secnet_dir / f"{region}_transformers.csv").to_dict("records"),
            secondary_edges=pd.read_csv(secondary_edges).to_dict("records"),
        ))
        logger.info(f"Secondary network generation/loading complete in {time.time()-ts} seconds.")

    stats.finish()


def check_secnet_complete(run, region):
    """The group stages need the up to date secondary network of every region."""
    from utils.checkpoint_utils import CheckpointManager, COMPLETE
    checkpoints = CheckpointManager(manifest_file=run.region_manifest(region), stages=REGION_STAGES)
    if checkpoints.status("secnet", run.secnet_params(region), run.secnet_files(region)) != COMPLETE:
        raise RuntimeError(
            f"Secondary network of region {region} is missing or outdated; "
            f"run the secnet stage for region {region} first"
        )


def clear_partition_outputs(run):
    """
    Remove the partitioning of the group. Returns the substations it had, whose primary
    networks are kept: they are reused if the partition of their substation is unchanged.
    """
    old_substations = run.substations() if (run.partition_dir / "substations.txt").exists() else []
    shutil.rmtree(run.partition_dir, ignore_errors=True)
    return old_substations


def clear_primnet_outputs(run, sub_ids):
    """Remove the primary networks, checkpoints and statistics of the given substations."""
    for sub_id in sub_ids:
        for path in [*run.primnet_files(sub_id), Path(run.primnet_manifest(sub_id)), Path(run.primnet_stats_file(sub_id))]:
            path.unlink(missing_ok=True)


def run_partition(run, checkpoints, stats):
    """Merge the networks of the regions and assign every transformer to a substation."""
    from utils.logging_utils import LogManager
    from utils.checkpoint_utils import COMPLETE, file_digest
    from utils.dataloader import load_substations
    from utils.stats_utils import partition_stats
    conf = run.conf
    logger = LogManager.get_logger("__main__")
    input_sub_csv = conf["inputs"]["substation_csv"]
    figdir = conf["figures"]["directory"]

    with stats.stage("partition") as stage_stats:
        # Partition transformer nodes to the nearest reachable substation
        ts = time.time()
        from utils.partition_utils import NetworkPartitioner, save_substation_partitions
        from utils.secnet_utils import merge_combined_networks
        for region in run.regions:
            check_secnet_complete(run, region)
        sequence_files = [run.secnet_dir / f"{region}_road_transformer_edges.txt" for region in run.regions]
        partition_params = {
            "padding": conf["miscellaneous"]["padding"],
            "substations": file_digest(input_sub_csv),
            "networks": {
                str(path): file_digest(path)
                for region, sequence_file in zip(run.regions, sequence_files)
                for path in [*run.secnet_files(region)[:2], sequence_file] if path.exists()
            },
        }
        stats_file = run.partition_dir / "partition_stats.json"
        network = homes = subs = None
        if checkpoints.status("partition", partition_params, [stats_file]) == COMPLETE:
            logger.info("Reusing saved transformer partitioning")
            stage_stats["status"] = "reused"
        else:
            checkpoints.start("partition", partition_params)
            old_substations = clear_partition_outputs(run)
            homes = run.load_homes()
            subs = load_substations(file_path=input_sub_csv, homes=homes)
            if not subs:
                raise ValueError(f"No substation of {input_sub_csv} lies within the homes of group {run.group}")
            network = merge_combined_networks(run.regions, run.secnet_dir)
            partitioner = NetworkPartitioner(network)
            try:
                # Find nearest road nodes - will automatically retry with increased radius if needed
                substation_nodes = partitioner.find_nearest_road_nodes(
                    subs,
                    search_radius=conf["miscellaneous"]["padding"]
                )
                assignments = partitioner.partition_transformers(substation_nodes)
            except ValueError as e:
                logger.error(f"Error occurred while partitioning: {e}")
                raise
            partition_data = partitioner.build_partitioning(assignments, subs)
            for sub in subs:
                if str(sub.id) not in partition_data:
                    logger.info(f"No nodes mapped to the substation {sub.id}")
            new_substations = save_substation_partitions(partition_data, subs, run.partition_dir)
            clear_primnet_outputs(run, set(old_substations) - set(new_substations))
            stats_file.write_text(json.dumps(partition_stats(subs, partition_data), indent=2))
            checkpoints.complete("partition", partition_params)

        stage_stats.update(json.loads(stats_file.read_text()))
        logger.info(f"Transformer partitioning and partition data loading complete in {time.time()-ts} seconds.")

    if run.args.generate_plot:
        from utils.secnet_utils import merge_combined_networks
        if network is None:
            homes = run.load_homes()
            subs = load_substations(file_path=input_sub_csv, homes=homes)
            network = merge_combined_networks(run.regions, run.secnet_dir)
        Path(figdir).mkdir(parents=True, exist_ok=True)
        combined_network_filename = conf["figures"]["combined_network"]["filename"]
        import matplotlib.pyplot as plt
        from utils.drawings import plot_combined_road_transformer, plot_substations, plot_homes
        fig, ax = plt.subplots(1, 1, figsize=(30,18))
        plot_substations(subs, ax=ax)
        plot_homes(homes, ax=ax)
        plot_combined_road_transformer(network, ax=ax)
        fig.suptitle("Road network with transformers and substations", fontsize=55)
        fig.savefig(f"{figdir}/{run.group}_{combined_network_filename}", bbox_inches='tight')
        plt.close(fig)


def run_substation(run, sub_id):
    """Primary network of one substation."""
    import pandas as pd
    from utils.logging_utils import LogManager
    from utils.checkpoint_utils import CheckpointManager, COMPLETE
    from utils.stats_utils import RunStats, primnet_stats
    from utils.partition_utils import load_substation_partition
    from utils.primnet_utils import PrimaryNetworkGenerator
    logger = LogManager.get_logger("__main__")
    primnet_config = run.conf["primnet"]["primnet_args"]
    nodes_file, edges_file = run.primnet_files(sub_id)

    checkpoints = CheckpointManager(
        manifest_file=run.primnet_manifest(sub_id), stages=["primnet"], from_scratch=run.from_scratch,
    )
    stats = RunStats(
        output_file=run.primnet_stats_file(sub_id),
        group=run.group,
        substation=sub_id,
        config=run.configpath,
        primnet_method=primnet_config.get("method", "milp"),
        from_scratch=run.from_scratch,
    )
    with stats.stage("primnet") as stage_stats:
        primnet_params = run.primnet_params(sub_id)
        if checkpoints.status("primnet", primnet_params, [nodes_file, edges_file]) == COMPLETE:
            logger.info(f"Reusing saved primary network of substation {sub_id}")
            stage_stats["status"] = "reused"
        else:
            checkpoints.start("primnet", primnet_params)
            logger.info(f"Creating primary network for service substation {sub_id}")
            sub, assignment = load_substation_partition(run.partition_dir, sub_id)
            generator = PrimaryNetworkGenerator(output_dir=str(run.primnet_dir))
            generator.generate_network_for_substation(
                sub,
                assignment = assignment,
                config=primnet_config,
                )
            generator.export_to_csv(prefix=f"{run.group}_{sub_id}")
            checkpoints.complete("primnet", primnet_params)
        stage_stats.update(primnet_stats({sub_id: pd.read_csv(edges_file).to_dict("records")}))
    stats.finish()


def run_substations(run, sub_ids):
    """Primary networks of the given substations, with a progress bar and a log line per substation."""
    from tqdm import tqdm
    from utils.logging_utils import LogManager, progress_logging
    logger = LogManager.get_logger("__main__")
    sub_ids = list(sub_ids)
    with progress_logging():
        for done, sub_id in enumerate(
            tqdm(sub_ids, desc="Generating primary networks", unit="substation"), start=1
        ):
            run_substation(run, sub_id)
            logger.info(f"Primary networks finished for {done} of {len(sub_ids)} substations")


def summarize_primnet(run, stats):
    """Statistics of the primary networks of all substations of the group."""
    import pandas as pd
    from utils.logging_utils import LogManager
    from utils.checkpoint_utils import CheckpointManager, COMPLETE
    from utils.stats_utils import primnet_stats
    logger = LogManager.get_logger("__main__")
    primnet_config = run.conf["primnet"]["primnet_args"]

    with stats.stage("primnet") as stage_stats:
        sub_ids = run.substations()
        outdated = [
            sub_id for sub_id in sub_ids
            if CheckpointManager(manifest_file=run.primnet_manifest(sub_id), stages=["primnet"]).status(
                "primnet", run.primnet_params(sub_id), run.primnet_files(sub_id)
            ) != COMPLETE
        ]
        if outdated:
            raise RuntimeError(
                f"Primary networks of substations {outdated} are missing or outdated; "
                "run the primnet stage for them first"
            )
        generated = {}
        for sub_id in sub_ids:
            stats_file = Path(run.primnet_stats_file(sub_id))
            sub_stats = json.loads(stats_file.read_text())["stages"]["primnet"] if stats_file.exists() else {}
            if sub_stats.get("status") == "computed":
                generated[sub_id] = sub_stats["seconds"]
        if not generated:
            logger.info("Reusing saved primary networks")
            stage_stats["status"] = "reused"
        stage_stats.update(
            primnet_method=primnet_config.get("method", "milp"),
            substations_generated=len(generated),
            **primnet_stats({
                sub_id: pd.read_csv(run.primnet_files(sub_id)[1]).to_dict("records") for sub_id in sub_ids
            }, generated),
        )
        logger.info(f"Primary networks of {len(sub_ids)} substations are complete")


def run_finalnet(run, checkpoints, stats):
    """Combine the primary networks and the secondary networks of all regions."""
    from utils.logging_utils import LogManager
    from utils.checkpoint_utils import COMPLETE, file_digest
    from utils.finalnet_utils import (
        combine_networks, validate_distribution_network, save_distribution_network,
        load_distribution_network, distribution_network_summary,
    )
    conf = run.conf
    logger = LogManager.get_logger("__main__")

    with stats.stage("finalnet") as stage_stats:
        # Create the network by combining primary and secondary network
        ts = time.time()
        service_substations = run.substations()
        for region in run.regions:
            check_secnet_complete(run, region)
        finalnet_dir = Path(conf["finalnet"]["out_dir"])
        final_nodes = finalnet_dir / f"{run.group}_distribution_network_nodes.csv"
        final_edges = finalnet_dir / f"{run.group}_distribution_network_edges.csv"
        finalnet_params = {
            "substations": sorted(service_substations),
            "networks": {
                str(path): file_digest(path)
                for path in [
                    *(f for sub_id in service_substations for f in run.primnet_files(sub_id)),
                    *(run.secnet_files(region)[2] for region in run.regions),
                    *(run.home_csv(region) for region in run.regions),  # home locations and loads
                ]
            },
        }
        homes = run.load_homes()
        if checkpoints.status("finalnet", finalnet_params, [final_nodes, final_edges]) == COMPLETE:
            logger.info("Reusing saved distribution network")
            stage_stats["status"] = "reused"
        else:
            checkpoints.start("finalnet", finalnet_params)
            distribution_network = combine_networks(
                service_substations,
                region_name=run.group,
                homes=homes,
                prim_dir=str(run.primnet_dir),
                sec_dir=str(run.secnet_dir),
                secondary_regions=run.regions,
            )
            save_distribution_network(distribution_network, region_name=run.group, out_dir=finalnet_dir)
            checkpoints.complete("finalnet", finalnet_params)
        distribution_network = load_distribution_network(final_nodes, final_edges)
        validation = validate_distribution_network(distribution_network, homes)
        stage_stats.update(
            **distribution_network_summary(distribution_network),
            is_forest=validation["is_forest"],
            trees=validation["trees"],
            trees_without_substation=validation["trees_without_substation"],
            trees_with_several_substations=validation["trees_with_several_substations"],
            homes_missing=len(validation["homes_missing"]),
            isolated_nodes=len(validation["isolated_nodes"]),
        )

        if run.args.generate_plot:
            import matplotlib.pyplot as plt
            from utils.drawings import plot_distribution_network
            figdir = conf["figures"]["directory"]
            final_network_filename = conf["figures"]["final_network"]["filename"]
            Path(figdir).mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(1, 1, figsize=(30,18))
            plot_distribution_network(distribution_network, ax=ax)
            fig.suptitle(f"Synthetic distribution network for region: {run.group}", fontsize=55)
            fig.savefig(f"{figdir}/{run.group}_{final_network_filename}", bbox_inches='tight')
            plt.close(fig)
        logger.info(f"Distribution network combination/loading complete in {time.time()-ts} seconds.")


def main(argv=None):
    args = parse_args(argv)
    # the group file (regions, state, input directories) overrides the configuration file
    conf = OmegaConf.merge(OmegaConf.load(args.configPath), OmegaConf.load(args.group))
    if "state" not in conf:
        raise SystemExit(f"Group configuration {args.group} must set the state (FIPS code), e.g. state: 51")
    if not conf.get("inputs", {}).get("home_csv_dir"):
        raise SystemExit(
            f"Group configuration {args.group} must set the directory of the home load files, "
            'e.g. inputs: {home_csv_dir: "data/load_51/"}'
        )
    run = Run(conf, args)
    stage = args.stage

    if args.list:
        if stage not in ("secnet", "primnet"):
            raise SystemExit("--list applies to the secnet and primnet stages only")
        print("\n".join(run.task_items()))
        return

    from utils.logging_utils import LogManager
    LogManager.initialize(
        log_file_path=f"{conf['miscellaneous']['log_dir']}application_{run.label}.log",
        log_config_path="configs/logging_config.ini"
    )
    logger = LogManager.get_logger("__main__")
    logger.info(f"Running stage '{stage}' for group {run.group} with regions {run.regions}")
    if run.from_scratch:
        logger.info("Running all stages from scratch")

    if stage == "secnet":
        for region in run.task_items():
            run_region(run, region)
        return
    if stage == "primnet":
        run_substations(run, run.task_items())
        return

    from utils.checkpoint_utils import CheckpointManager
    from utils.stats_utils import RunStats
    checkpoints = CheckpointManager(
        manifest_file=run.group_manifest(), stages=GROUP_STAGES, from_scratch=run.from_scratch,
    )
    # a finalnet job adds its statistics to those saved by the partition job
    stats = RunStats(
        output_file=run.group_stats_file(),
        keep_stages=(stage == "finalnet"),
        group=run.group,
        regions=run.regions,
        state=conf["state"],
        config=run.configpath,
        primnet_method=conf["primnet"]["primnet_args"].get("method", "milp"),
        from_scratch=run.from_scratch,
    )
    if stage == "all":
        for region in run.regions:
            run_region(run, region)
    if stage in ("all", "partition"):
        run_partition(run, checkpoints, stats)
    if stage == "all":
        run_substations(run, run.substations())
    if stage in ("all", "finalnet"):
        summarize_primnet(run, stats)
        run_finalnet(run, checkpoints, stats)
    stats.finish()


if __name__ == "__main__":
    main()
