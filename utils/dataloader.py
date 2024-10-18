from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path
import pandas as pd
from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize(
        log_file_path="logs/test_osm_utils.log", 
        log_config_path="configs/logging_config.ini"
    )
logger = LogManager.get_logger("dataloader")

@dataclass
class Home:
    """
    Represents a home with its location and energy consumption data.

    Attributes:
        id (int): Unique identifier for the home.
        cord (Tuple[float, float]): The coordinates (longitude, latitude) of the home.
        profile (List[float]): The 24-hour energy consumption profile.
        peak (float): The peak energy consumption over the 24-hour period.
        load (float): The average energy consumption over the 24-hour period.
    """
    id: int
    cord: Tuple[float, float]
    profile: List[float]
    peak: float
    load: float

def load_homes(file_path: str) -> List[Home]:
    """
    Load home data from a CSV file and return a list of Home objects.

    Args:
        file_path (str): The path to the CSV file containing home data.

    Returns:
        List[Home]: A list of Home objects created from the CSV data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the CSV file has an invalid format.
    """
    
    file_path = Path(file_path)
    logger.info(f"Loading homes data from {file_path}")

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    homes = []
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows from CSV")

    for index, row in df.iterrows():
        try:
            id = int(row['hid'])
            cord = (float(row['longitude']), float(row['latitude']))
            profile = [float(row[f'hour{i}']) for i in range(1, 25)]
            peak = max(profile)
            load = sum(profile) / 24

            homes.append(Home(id=id, cord=cord, profile=profile, peak=peak, load=load))
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid data format in row {index + 2}: {e}")
            raise ValueError(f"Invalid data format in row {index + 2}: {e}")

    logger.info(f"Successfully created {len(homes)} Home objects")
    return homes
