# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:11:06 2022

Author: Rounak Meyur
"""

import logging
logger = logging.getLogger(__name__)

from shapely.geometry import Point, LineString, MultiPoint
import pandas as pd
import numpy as np
from collections import namedtuple as nt
import osmnx as ox

from params import DATA_PATHS



def load_homes(filename: str) -> pd.DataFrame:
    """
    Gets residence data from the file

    Parameters
    ----------
    filename : str
        Name of the residence file.

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing the residence hourly energy consumption data.

    """
    data_dir = DATA_PATHS["load"]
    data_dir.mkdir(exist_ok=True)
    if (data_dir / f"{filename}.csv").exists():
        df = pd.read_csv(data_dir / f"{filename}.csv")
        
    else:
        logger.error("File f{filename}.csv not present!!!")
    
    return df

def get_homes(df_home : pd.DataFrame) -> nt:
    """
    Prepares the residential data for the synthetic network creation process

    Parameters
    ----------
    df_home : pd.DataFrame
        pandas dataframe for the input residential load usage data.

    Returns
    -------
    nt
        named tuple of residential data with location, average and peak demands.

    """
    df_home['average'] = pd.Series(np.mean(df_home.iloc[:,3:27].values,axis=1))
    df_home['peak'] = pd.Series(np.max(df_home.iloc[:,3:27].values,axis=1))
    
    home = nt("home",field_names=["cord","profile","peak","average"])
    dict_load = df_home.iloc[:,[0]+list(range(3,27))].set_index('hid').T.to_dict('list')
    dict_cord = df_home.iloc[:,0:3].set_index('hid').T.to_dict('list')
    dict_peak = dict(zip(df_home.hid,df_home.peak))
    dict_avg = dict(zip(df_home.hid,df_home.average))
    return home(cord=dict_cord,profile=dict_load,peak=dict_peak,average=dict_avg)
    

def load_substations(columns: list = ["ID","LATITUDE","LONGITUDE"]) -> pd.DataFrame:
    """
    Load EIA substation data either from file (if it exists) or from the API.

    Parameters
    ----------
    columns : list, optional
        The expected output dataframe columns. 
        The default is ["ID","LATITUDE","LONGITUDE"].

    Returns
    -------
    df : pd.DataFrame
        Data from substation CSV file.

    """
    data_dir = DATA_PATHS["data"]
    data_dir.mkdir(exist_ok=True)
    if (data_dir / "substations.csv").exists():
        df = pd.read_csv(data_dir / "substations.csv", usecols=columns)
        
    else:
        logger.error("File substations.csv not present!!!")

    return df


def get_roads(homes):
    points = [Point(homes.cord[n]) for n in homes.cord]
    bound_polygon = MultiPoint(points).convex_hull
    
    # Get the OSM links within the county polygon
    osm_graph = ox.graph_from_polygon(bound_polygon, retain_all=True,
                                  truncate_by_edge=False)
    
    # Add geometries for links without it
    edge_nogeom = [e for e in osm_graph.edges(keys=True) \
                   if 'geometry' not in osm_graph.edges[e]]
    for e in edge_nogeom:
        pts = [(osm_graph.nodes[e[0]]['x'],osm_graph.nodes[e[0]]['y']),
               (osm_graph.nodes[e[1]]['x'],osm_graph.nodes[e[1]]['y'])]
        link_geom = LineString(pts)
        osm_graph.edges[e]['geometry'] = link_geom
    
    return osm_graph



df = load_homes("test-homes")
homes = get_homes(df)
roads = get_roads(homes)
