

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
    from utils.dataloader import load_homes
    from utils.osm_utils import load_roads

    input_home_csv = conf["inputs"]["home_csv"]
    homes = load_homes(file_path=input_home_csv)
    roads = load_roads(homes)
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

    # Generate secondary network
    ts = time.time()
    from utils.secnet_utils import SecondaryNetworkGenerator
    generator = SecondaryNetworkGenerator(
        output_dir=conf["secnet"]["out_dir"],
        base_transformer_id=conf["secnet"]["base_transformer_id"],
    )
    secnet_args = conf["secnet"]["secnet_args"]
    for road_link in r2h:
        road_geometry = roads.edges(keys=True)[road_link]['geometry']
        generator.generate_network_for_link(
            road_link=road_link,
            road_geom=road_geometry,
            mapped_homes=r2h[road_link],
            **secnet_args
        )
        generator.save_results(prefix=conf["secnet"]["out_prefix"])
    logger.info(f"Secondary network generation complete in {time.time()-ts} seconds.")

if __name__ == "__main__":
    main()