from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
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

@dataclass
class Substation:
    """
    Represents a substation with its location.

    Attributes:
        id (str): Unique identifier for the substation.
        cord (Tuple[float, float]): The coordinates (longitude, latitude) of the substation.
    """
    id: str
    cord: Tuple[float, float]

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

def load_substations(file_path: str, homes: List[Home]) -> List[Substation]:
    """
    Load substation data from a CSV file and filter based on homes convex hull.
    The CSV file should have the columns: 'ID', 'X' and 'Y'

    Args:
        file_path (str): Path to the CSV file containing substation data.
        homes (List[Home]): List of Home objects to determine the convex hull.

    Returns:
        List[Substation]: List of Substation objects within the convex hull of homes.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the CSV file has an invalid format or no homes are provided.
    """
    if not homes:
        logger.error("No homes provided to determine convex hull")
        raise ValueError("List of homes cannot be empty")

    file_path = Path(file_path)
    logger.info(f"Loading substations data from {file_path}")

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Create convex hull from home coordinates
    home_points = [Point(home.cord[0], home.cord[1]) for home in homes]
    convex_hull = unary_union(home_points).convex_hull
    logger.info("Created convex hull from home locations")

    # Load and filter substations
    substations = []
    df = pd.read_csv(
        file_path, 
        usecols=['ID', 'X', 'Y'], 
        dtype = {'ID': str, 'X': np.float64, 'Y': np.float64}
        )
    logger.info(f"Loaded {len(df)} substations from CSV")

    for index, row in df.iterrows():
        try:
            sub_id = str(row['ID'])
            cord = (float(row['X']), float(row['Y']))
            
            # Check if substation is within convex hull
            if convex_hull.contains(Point(cord)):
                substations.append(Substation(id=sub_id, cord=cord))
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid data format in row {index + 2}: {e}")
            raise ValueError(f"Invalid data format in row {index + 2}: {e}")

    logger.info(f"Found {len(substations)} substations within the convex hull")
    return substations