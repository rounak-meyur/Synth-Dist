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
        default = "configs/config.yaml"
    )
    parser.add_argument(
        "-r", "--region",
        help = "county ID for the region",
        default = 999
    )
    parser.add_argument(
        "-s", "--state",
        help = "state ID for the region",
        default = 999
    )
    
    args = parser.parse_args()
    configpath = args.configPath
    conf = OmegaConf.load(configpath)
    state = args.state
    region = args.region

    from utils.logging_utils import LogManager
    LogManager.initialize(
        log_file_path=f"{conf['miscellaneous']['log_dir']}secnet_{state}_{region}.log", 
        log_config_path="utils/logging_config.ini"
    )
    logger = LogManager.get_logger("__main__")

    # Load inputs
    ts = time.time()
    from utils.dataloader import load_homes
    from utils.osm_utils import load_roads, save_road_network, load_road_network_from_files

    input_home_csv = f"{region}-home-load.csv"
    homes = load_homes(file_path=input_home_csv)
    road_edge_file = f"out_{state}_{region}/{conf['miscellaneous']['intermediate_dir']}{region}_road_edges.csv"
    road_node_file = f"out_{state}_{region}/{conf['miscellaneous']['intermediate_dir']}{region}_road_nodes.csv"
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

    h2r_map_file = f"out_{state}_{region}/{conf['miscellaneous']['mapping_dir']}{region}_map_h2r.txt"
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
    combined_edges = f"out_{state}_{region}/{conf['secnet']['out_dir']}{region}_combined_network_edges.csv"
    combined_nodes = f"out_{state}_{region}/{conf['secnet']['out_dir']}{region}_combined_network_nodes.csv"
    
    
    from utils.secnet_utils import SecondaryNetworkGenerator
    base_tsfr_id = int(f"{state}{region}{0:012d}")
    generator = SecondaryNetworkGenerator(
        output_dir=f"out_{state}_{region}/{conf["secnet"]["out_dir"]}",
        base_transformer_id=base_tsfr_id,
    )
    secnet_args = conf["secnet"]["secnet_args"]
    total_links = len(r2h)
    count = 0
    for road_link in r2h:
        count += 1
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
        logger.info(f"Completed secondary network generation for {count}/{total_links} road links.")

    # Combine road network with transformers
    combined_network = generator.combine_networks(road_network=roads, prefix=region)
    
    logger.info(f"Secondary network generation/loading complete in {time.time()-ts} seconds.")


if __name__ == "__main__":
    main()