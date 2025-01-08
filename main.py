

def main():
    import os
    import time
    import argparse
    from omegaconf import OmegaConf
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--configPath",
        help = "path to configuration file",
        default = "configs/test.yaml"
    )
    parser.add_argument(
        "-p", "--generate_plot",
        help = "generate plots",
        action = "store_true"
    )
    args = parser.parse_args()
    configpath = args.configPath
    conf = OmegaConf.load(configpath)
    region = conf["region"]

    from utils.logging_utils import LogManager
    LogManager.initialize(
        log_file_path=f"{conf['miscellaneous']['log_dir']}application_{region}.log", 
        log_config_path="configs/logging_config.ini"
    )
    logger = LogManager.get_logger("__main__")

    # Load inputs
    ts = time.time()
    from utils.dataloader import load_homes, load_substations
    from utils.osm_utils import load_roads, save_road_network, load_road_network_from_files

    input_home_csv = f"{conf['inputs']['home_csv_dir']}{region}-home-load.csv"
    input_sub_csv = conf["inputs"]["substation_csv"]
    homes = load_homes(file_path=input_home_csv)
    subs = load_substations(file_path=input_sub_csv, homes=homes)
    road_edge_file = f"{conf['miscellaneous']['intermediate_dir']}{region}_road_edges.csv"
    road_node_file = f"{conf['miscellaneous']['intermediate_dir']}{region}_road_nodes.csv"
    if not os.path.exists(road_edge_file) or not os.path.exists(road_node_file):
        roads = load_roads(homes)
        save_road_network(
            roads, 
            edgelist_file=road_edge_file,
            nodelist_file=road_node_file
            )
    # load the road from the file to load as an undirected graph
    roads = load_road_network_from_files(
        edgelist_file=road_edge_file,
        nodelist_file=road_node_file
        )
    logger.info(f"Residence and road data loading complete in {time.time()-ts} seconds.")

    # Map roads and homes
    ts = time.time()
    from utils.mapping import(
        map_homes_to_edges, 
        write_mapping_to_file, 
        read_mapping_from_file, 
        compute_edge_to_homes_map
    )

    h2r_map_file = f"{conf['miscellaneous']['mapping_dir']}{region}_map_h2r.txt"
    if not os.path.exists(h2r_map_file):
        h2r = map_homes_to_edges(
            roads, homes, 
            padding_distance=conf["miscellaneous"]["padding"])
        write_mapping_to_file(h2r, h2r_map_file)
    else:
        h2r = read_mapping_from_file(homes, filename=h2r_map_file)

    r2h = compute_edge_to_homes_map(h2r)
    logger.info(f"Residence and road data mapping complete in {time.time()-ts} seconds.")

    # Generate secondary network and combine road network with transformers
    ts = time.time()
    combined_edges = f"{conf['secnet']['out_dir']}{region}_combined_network_edges.csv"
    combined_nodes = f"{conf['secnet']['out_dir']}{region}_combined_network_nodes.csv"
    if not os.path.exists(combined_edges) or not os.path.exists(combined_nodes):
        from utils.secnet_utils import SecondaryNetworkGenerator
        generator = SecondaryNetworkGenerator(
            output_dir=conf["secnet"]["out_dir"],
            base_transformer_id=conf["secnet"]["base_transformer_id"],
        )
        secnet_args = conf["secnet"]["secnet_args"]
        for road_link in r2h:
            try:
                road_geometry = roads.edges(keys=True)[road_link]['geometry']
            except:
                logger.info(f"Cannot extract geometry for road link {road_link}")
            generator.generate_network_for_link(
                road_link=road_link,
                road_geom=road_geometry,
                mapped_homes=r2h[road_link],
                **secnet_args
            )
            generator.save_results(prefix=region, road_link_id=road_link)

        # Combine road network with transformers
        combined_network = generator.combine_networks(road_network=roads, prefix=conf["region"])
    else:
        from utils.secnet_utils import load_combined_network
        combined_network = load_combined_network(
            nodes_file=combined_nodes,
            edges_file=combined_edges
            )
    logger.info(f"Secondary network generation/loading complete in {time.time()-ts} seconds.")
    
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

    # Partition transformer nodes to the nearest reachable substation
    ts = time.time()
    assignment_json = f"{conf['miscellaneous']['intermediate_dir']}{region}_transformer_assignment.json"
    if not os.path.exists(assignment_json):
        from utils.partition_utils import NetworkPartitioner
        partitioner = NetworkPartitioner(combined_network)
        try:
            # Find nearest road nodes - will automatically retry with increased radius if needed
            substation_nodes = partitioner.find_nearest_road_nodes(
                subs,
                search_radius=conf["miscellaneous"]["padding"]
            )
            assignments = partitioner.partition_transformers(substation_nodes)
            partitioner.save_partitioning(
                assignments, 
                subs, 
                output_file=assignment_json
                )
        except ValueError as e:
            logger.error(f"Error occurred while partitioning: {e}")
    
    from utils.partition_utils import load_partitioning
    partition_data = load_partitioning(assignment_json)
    logger.info(f"Transformer partitioning and partition data loading complete in {time.time()-ts} seconds.")
    
    # Create primary network sequentially for all substation partitioned data
    from utils.primnet_utils import PrimaryNetworkGenerator
    primnet_config = conf["primnet"]["primnet_args"]
    
    service_substations = []
    for sub in subs:
        if int(sub.id) not in partition_data:
            logger.info(f"No nodes mapped to the substation {sub.id}")
        else:
            service_substations.append(sub.id)
            primnet_edge_csv = f"{conf['primnet']['out_dir']}{region}_{sub.id}_edges.csv"
            primnet_node_csv = f"{conf['primnet']['out_dir']}{region}_{sub.id}_nodes.csv"
            if not os.path.exists(primnet_edge_csv) or not os.path.exists(primnet_node_csv):
                logger.info(f"Creating primary network for service substation {sub.id}")
                generator = PrimaryNetworkGenerator(output_dir=conf["primnet"]["out_dir"])
                generator.generate_network_for_substation(
                    sub, 
                    assignment = partition_data[int(sub.id)],
                    config=primnet_config
                    )
                generator.export_to_csv(prefix=f"test_{str(sub.id)}")
            else:
                logger.info(f"Primary network already generated and saved for substation {sub.id}")
    
    # Create the network by combining primary nd secondary network
    from utils.finalnet_utils import combine_networks
    from utils.drawings import plot_distribution_network
    distribution_network = combine_networks(
        service_substations, 
        region_name=conf["region"], 
        homes=homes,
        prim_dir=conf["primnet"]["out_dir"],
        sec_dir=conf["secnet"]["out_dir"]
    )
    if args.generate_plot:
        figdir = conf["figures"]["directory"]
        final_network_filename = conf["figures"]["final_network"]["filename"]
        Path(figdir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(30,18))
        plot_distribution_network(distribution_network, ax=ax)
        fig.suptitle(f"Synthetic distribution network for region: {region}", fontsize=55)
        fig.savefig(f"figs/{region}_{final_network_filename}", bbox_inches='tight')
    
    
if __name__ == "__main__":
    main()