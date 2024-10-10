

def main():
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
    from utils.osm_utils import load_roads, map_homes_to_edges

    input_home_csv = conf["inputs"]["home_csv"]
    homes = load_homes(file_path=input_home_csv)
    roads = load_roads(homes)
    h2r = map_homes_to_edges(roads, homes, padding_distance=conf["mapping"]["padding"])
    logger.info(h2r)


if __name__ == "__main__":
    main()