

def main():
    import os
    import time
    import argparse
    from omegaconf import OmegaConf
    from tqdm import tqdm
    from utils.logging_utils import progress_logging
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--configPath",
        help = "path to configuration file",
        default = "configs/config_heuristic.yaml"
    )
    parser.add_argument(
        "-r", "--region", type=str,
        help = "county ID for the region",
        default = "999"
    )
    parser.add_argument(
        "-p", "--generate_plot",
        help = "generate plots",
        action = "store_true"
    )
    parser.add_argument(
        "-f", "--from-scratch",
        help = "ignore saved outputs and run every stage from scratch "
               "(same as miscellaneous.from_scratch: true in the config)",
        action = "store_true"
    )
    args = parser.parse_args()
    configpath = args.configPath
    conf = OmegaConf.load(configpath)
    state = conf["state"]
    region = args.region
    conf["region"] = region

    from utils.logging_utils import LogManager
    LogManager.initialize(
        log_file_path=f"{conf['miscellaneous']['log_dir']}application_{region}.log", 
        log_config_path="configs/logging_config.ini"
    )
    logger = LogManager.get_logger("__main__")

    # Checkpoints: a stage is reused only if it finished with the same inputs/parameters
    from utils.checkpoint_utils import CheckpointManager, COMPLETE, IN_PROGRESS, file_digest
    from_scratch = args.from_scratch or conf["miscellaneous"].get("from_scratch", False)
    checkpoints = CheckpointManager(
        manifest_file=f"{conf['miscellaneous']['intermediate_dir']}{region}_checkpoints.json",
        stages=["roads", "mapping", "secnet", "partition", "primnet", "finalnet"],
        from_scratch=from_scratch,
    )
    if from_scratch:
        logger.info("Running all stages from scratch")

    # Compute statistics and data sizes of every stage, updated after each stage
    from utils.stats_utils import (
        RunStats, road_network_stats, mapping_stats, secnet_stats, partition_stats, primnet_stats
    )
    stats = RunStats(
        output_file=f"{conf['miscellaneous']['stats_dir']}{region}_run_stats.json",
        region=region,
        state=state,
        config=configpath,
        secnet_method=conf["secnet"]["secnet_args"].get("method", "milp"),
        primnet_method=conf["primnet"]["primnet_args"].get("method", "milp"),
        from_scratch=bool(from_scratch),
    )

    with stats.stage("roads") as stage_stats:
        # Load inputs
        ts = time.time()
        from utils.dataloader import load_homes, load_substations
        from utils.osm_utils import load_roads, save_road_network, load_road_network_from_files

        input_home_csv = f"{conf['inputs']['home_csv_dir']}{region}-home-load.csv"
        homes = load_homes(file_path=input_home_csv)
    
        road_edge_file = f"{conf['miscellaneous']['intermediate_dir']}{region}_road_edges.csv"
        road_node_file = f"{conf['miscellaneous']['intermediate_dir']}{region}_road_nodes.csv"
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
            roads = load_roads(homes)
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
        from utils.secnet_utils import SecondaryNetworkGenerator, get_road_link_geometry, load_combined_network
        combined_edges = f"{conf['secnet']['out_dir']}{region}_combined_network_edges.csv"
        combined_nodes = f"{conf['secnet']['out_dir']}{region}_combined_network_nodes.csv"
        secnet_args = conf["secnet"]["secnet_args"]
        base_tsfr_id = int(f"{state}{region}{0:012d}")
        secnet_params = {
            "secnet_args": OmegaConf.to_container(secnet_args, resolve=True),
            "base_transformer_id": base_tsfr_id,
        }
        secnet_status = checkpoints.status("secnet", secnet_params, [combined_edges, combined_nodes])
        if secnet_status == COMPLETE:
            logger.info("Reusing saved secondary network")
            stage_stats["status"] = "reused"
        else:
            generator = SecondaryNetworkGenerator(
                output_dir=conf["secnet"]["out_dir"],
                base_transformer_id=base_tsfr_id,
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
        import pandas as pd
        secnet_dir = Path(conf["secnet"]["out_dir"])
        stage_stats.update(secnet_stats(
            combined_network,
            transformers=pd.read_csv(secnet_dir / f"{region}_transformers.csv").to_dict("records"),
            secondary_edges=pd.read_csv(secnet_dir / f"{region}_secondary_edges.csv").to_dict("records"),
        ))
        logger.info(f"Secondary network generation/loading complete in {time.time()-ts} seconds.")

    input_sub_csv = conf["inputs"]["substation_csv"]
    subs = load_substations(file_path=input_sub_csv, homes=homes)

    if args.generate_plot:
        figdir = conf["figures"]["directory"]
        Path(figdir).mkdir(parents=True, exist_ok=True)
        combined_network_filename = conf["figures"]["combined_network"]["filename"]
        import matplotlib.pyplot as plt
        from utils.drawings import plot_combined_road_transformer, plot_substations, plot_homes
        fig, ax = plt.subplots(1, 1, figsize=(30,18))
        plot_substations(subs, ax=ax)
        plot_homes(homes, ax=ax)
        plot_combined_road_transformer(combined_network, ax=ax)
        fig.suptitle("Road network with transformers and substations", fontsize=55)
        fig.savefig(f"{figdir}/{region}_{combined_network_filename}", bbox_inches='tight')

    with stats.stage("partition") as stage_stats:
        # Partition transformer nodes to the nearest reachable substation
        ts = time.time()
        from utils.partition_utils import NetworkPartitioner, load_partitioning
        assignment_json = f"{conf['miscellaneous']['intermediate_dir']}{region}_transformer_assignment.json"
        partition_params = {
            "padding": conf["miscellaneous"]["padding"],
            "substations": file_digest(input_sub_csv),
        }
        if checkpoints.status("partition", partition_params, [assignment_json]) == COMPLETE:
            logger.info("Reusing saved transformer partitioning")
            stage_stats["status"] = "reused"
        else:
            checkpoints.start("partition", partition_params)
            partitioner = NetworkPartitioner(combined_network)
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
            partitioner.save_partitioning(
                assignments, 
                subs, 
                output_file=assignment_json
                )
            checkpoints.complete("partition", partition_params)
    
        partition_data = load_partitioning(assignment_json)
        stage_stats.update(partition_stats(subs, partition_data))
        logger.info(f"Transformer partitioning and partition data loading complete in {time.time()-ts} seconds.")

    
    with stats.stage("primnet") as stage_stats:
        # Create primary network sequentially for all substation partitioned data
        ts = time.time()
        import pandas as pd
        from utils.primnet_utils import PrimaryNetworkGenerator
        primnet_config = conf["primnet"]["primnet_args"]
        primnet_dir = Path(conf["primnet"]["out_dir"])
        primnet_params = {"primnet_args": OmegaConf.to_container(primnet_config, resolve=True)}

        service_substations = [sub for sub in subs if int(sub.id) in partition_data]
        for sub in subs:
            if int(sub.id) not in partition_data:
                logger.info(f"No nodes mapped to the substation {sub.id}")
        primnet_files = {
            sub.id: (primnet_dir / f"{region}_{sub.id}_nodes.csv", primnet_dir / f"{region}_{sub.id}_edges.csv")
            for sub in service_substations
        }
        all_files = [f for files in primnet_files.values() for f in files]

        primnet_status = checkpoints.status("primnet", primnet_params, all_files)
        if primnet_status == COMPLETE:
            logger.info("Reusing saved primary networks")
            stage_stats["status"] = "reused"
            finished = set(primnet_files)
        else:
            if primnet_status == IN_PROGRESS:
                # continue an interrupted run: substations whose files were both written are done
                finished = {sid for sid, files in primnet_files.items() if all(f.exists() for f in files)}
                stage_stats["status"] = "resumed"
            else:
                finished = set()
                for f in primnet_dir.glob(f"{region}_*_nodes.csv"):
                    f.unlink()
                for f in primnet_dir.glob(f"{region}_*_edges.csv"):
                    f.unlink()
            checkpoints.start("primnet", primnet_params)

        substation_seconds = {}
        for sub in service_substations:
            if sub.id in finished:
                continue
            logger.info(f"Creating primary network for service substation {sub.id}")
            t_sub = time.perf_counter()
            generator = PrimaryNetworkGenerator(output_dir=conf["primnet"]["out_dir"])
            generator.generate_network_for_substation(
                sub, 
                assignment = partition_data[int(sub.id)],
                config=primnet_config,
                )
            generator.export_to_csv(prefix=f"{region}_{str(sub.id)}")
            substation_seconds[sub.id] = round(time.perf_counter() - t_sub, 3)
        if primnet_status != COMPLETE:
            checkpoints.complete("primnet", primnet_params)

        stage_stats.update(
            primnet_method=primnet_config.get("method", "milp"),
            substations_generated=len(substation_seconds),
            substations_finished_before=len(finished) if primnet_status != COMPLETE else 0,
            **primnet_stats({
                sid: pd.read_csv(files[1]).to_dict("records") for sid, files in primnet_files.items()
            }, substation_seconds),
        )
        service_substations = [sub.id for sub in service_substations]
        logger.info(f"Primary network generation/loading complete in {time.time()-ts} seconds.")

    with stats.stage("finalnet") as stage_stats:
        # Create the network by combining primary and secondary network
        ts = time.time()
        from utils.finalnet_utils import (
            combine_networks, validate_distribution_network, save_distribution_network,
            load_distribution_network, distribution_network_summary,
        )
        finalnet_dir = Path(conf["finalnet"]["out_dir"])
        final_nodes = finalnet_dir / f"{region}_distribution_network_nodes.csv"
        final_edges = finalnet_dir / f"{region}_distribution_network_edges.csv"
        finalnet_params = {"substations": sorted(service_substations)}
        if checkpoints.status("finalnet", finalnet_params, [final_nodes, final_edges]) == COMPLETE:
            logger.info("Reusing saved distribution network")
            stage_stats["status"] = "reused"
        else:
            checkpoints.start("finalnet", finalnet_params)
            distribution_network = combine_networks(
                service_substations, 
                region_name=region, 
                homes=homes,
                prim_dir=conf["primnet"]["out_dir"],
                sec_dir=conf["secnet"]["out_dir"]
            )
            save_distribution_network(distribution_network, region_name=region, out_dir=finalnet_dir)
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

        if args.generate_plot:
            import matplotlib.pyplot as plt
            from utils.drawings import plot_distribution_network
            figdir = conf["figures"]["directory"]
            final_network_filename = conf["figures"]["final_network"]["filename"]
            Path(figdir).mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(1, 1, figsize=(30,18))
            plot_distribution_network(distribution_network, ax=ax)
            fig.suptitle(f"Synthetic distribution network for region: {region}", fontsize=55)
            fig.savefig(f"{figdir}/{region}_{final_network_filename}", bbox_inches='tight')
            plt.close(fig)
        logger.info(f"Distribution network combination/loading complete in {time.time()-ts} seconds.")

    stats.finish()

    
if __name__ == "__main__":
    main()