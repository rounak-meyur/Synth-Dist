

def main():
    import os
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
    from utils.dataloader import load_homes
    from utils.osm_utils import load_roads

    input_home_csv = conf["inputs"]["home_csv"]
    homes = load_homes(file_path=input_home_csv)
    roads = load_roads(homes)

    # Map roads and homes
    from utils.mapping import(
        map_homes_to_edges, 
        write_mapping_to_file, 
        read_mapping_from_file, 
        compute_edge_to_homes_map
    )

    h2r_map_file = conf["outputs"]["home_to_road"]
    if not os.path.exists(h2r_map_file):
        h2r = map_homes_to_edges(
            roads, homes, 
            padding_distance=conf["mapping"]["padding"])
        write_mapping_to_file(h2r, h2r_map_file)
    else:
        h2r = read_mapping_from_file(homes, filename=h2r_map_file)

    r2h = compute_edge_to_homes_map(h2r)

    # Generate secondary network

if __name__ == "__main__":
    main()