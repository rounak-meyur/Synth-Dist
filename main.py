

def main():
    import os
    import time
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--configPath",
        help = "path to configuration file",
        default = "configs/test.yaml"
    )
    args = parser.parse_args()
    configpath = args.configPath
    conf = OmegaConf.load(configpath)

    from utils.logging_utils import LogManager
    LogManager.initialize(
        log_file_path=conf["logging"]["log_file"], 
        log_config_path="configs/logging_config.ini"
    )
    logger = LogManager.get_logger("__main__")

    # Load inputs
    ts = time.time()
    from utils.dataloader import load_homes, load_substations
    from utils.osm_utils import load_roads, save_road_network, load_road_network_from_files

    input_home_csv = conf["inputs"]["home_csv"]
    input_sub_csv = conf["inputs"]["substation_csv"]
    homes = load_homes(file_path=input_home_csv)
    subs = load_substations(file_path=input_sub_csv, homes=homes)
    if not os.path.exists(conf["intermediate"]["road_edges"]) or not os.path.exists(conf["intermediate"]["road_nodes"]):
        roads = load_roads(homes)
        save_road_network(
            roads, 
            edgelist_file=conf["intermediate"]["road_edges"],
            nodelist_file=conf["intermediate"]["road_nodes"]
            )
    else:
        logger.info(f"Road network has been loaded earlier. Loading from edgelist {conf['intermediate']['road_edges']}")
        roads = load_road_network_from_files(
            edgelist_file=conf["intermediate"]["road_edges"],
            nodelist_file=conf["intermediate"]["road_nodes"]
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

    h2r_map_file = conf["mapping"]["home_to_road"]
    if not os.path.exists(h2r_map_file):
        h2r = map_homes_to_edges(
            roads, homes, 
            padding_distance=conf["mapping"]["padding"])
        write_mapping_to_file(h2r, h2r_map_file)
    else:
        h2r = read_mapping_from_file(homes, filename=h2r_map_file)

    r2h = compute_edge_to_homes_map(h2r)
    logger.info(f"Residence and road data mapping complete in {time.time()-ts} seconds.")

    # Generate secondary network and combine road network with transformers
    combined_edges = f"{conf['secnet']['out_dir']}{conf['secnet']['out_prefix']}_combined_network_edges.csv"
    combined_nodes = f"{conf['secnet']['out_dir']}{conf['secnet']['out_prefix']}_combined_network_nodes.csv"
    if not os.path.exists(combined_edges) or not os.path.exists(combined_nodes):
        ts = time.time()
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
            generator.save_results(prefix=conf["secnet"]["out_prefix"], road_link_id=road_link)
        logger.info(f"Secondary network generation complete in {time.time()-ts} seconds.")

        # Combine road network with transformers
        combined_network = generator.combine_networks(road_network=roads, prefix=conf["secnet"]["out_prefix"])
    else:
        from utils.secnet_utils import load_combined_network
        combined_network = load_combined_network(
            nodes_file=combined_nodes,
            edges_file=combined_edges
            )
    
    import matplotlib.pyplot as plt
    from utils.drawings import plot_combined_road_transformer, plot_substations, plot_homes
    fig, ax = plt.subplots(1, 1, figsize=(30,18))
    plot_substations(subs, ax=ax)
    plot_homes(homes, ax=ax)
    plot_combined_road_transformer(combined_network, ax=ax)
    fig.suptitle("Road network with transformers and substations", fontsize=55)
    fig.savefig("figs/test_combined_network.png", bbox_inches='tight')
    
if __name__ == "__main__":
    main()