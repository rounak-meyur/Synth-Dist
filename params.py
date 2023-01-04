# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:20:23 2022

Author: Rounak Meyur

Description: Settings and parameters for the synthetic network generation frame-
-work
"""


import os
from pathlib import Path

from dotenv import load_dotenv


# Not convinced this is the best way to set folder paths but it works!
synthdist_path = Path(__file__).parent
project_path = synthdist_path.parent

load_dotenv(dotenv_path=synthdist_path / ".env")

DATA_PATHS = {}
DATA_PATHS["results"] = project_path / "results"
DATA_PATHS["synthdist"] = project_path / "synthdist"
DATA_PATHS["data"] = project_path / "data"
DATA_PATHS["osm"] = DATA_PATHS["data"] / "osm"
DATA_PATHS["load"] = DATA_PATHS["data"] / "load"
DATA_PATHS["tests"] = project_path / "tests"


# IPM_SHAPEFILE_PATH = DATA_PATHS["ipm_shapefiles"] / "IPM_Regions_201770405.shp"
# IPM_GEOJSON_PATH = DATA_PATHS["data"] / "ipm_regions_simple.geojson"

# SETTINGS = {}
# SETTINGS["PUDL_DB"] = os.environ.get("PUDL_DB")
# SETTINGS["PG_DB"] = os.environ.get("PG_DB")
# SETTINGS["EIA_API_KEY"] = os.environ.get("EIA_API_KEY")
# SETTINGS["RESOURCE_GROUPS"] = os.environ.get("RESOURCE_GROUPS")


