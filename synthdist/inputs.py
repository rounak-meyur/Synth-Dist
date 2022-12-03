# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:11:06 2022

Author: Rounak Meyur
"""

import os
import logging
logger = logging.getLogger(__name__)

from shapely.geometry import Point, LineString, MultiPoint
import pandas as pd
import numpy as np
from collections import namedtuple as nt
import osmnx as ox
import networkx as nx

from params import DATA_PATHS



def load_homes(filepath: str) -> nt:
    """
    Gets residence data from the file

    Parameters
    ----------
    filename : str
        Name of the residence file.

    Returns
    -------
    nt
        named tuple of residential data with location, average and peak demands.

    """
    if os.path.exists(f"{filepath}.csv"):
        df_home = pd.read_csv(f"{filepath}.csv")
        
    else:
        logger.error(f"File {filepath}.csv not present!!!")
        raise ValueError(f"{filepath}.csv doesn't exist!")
    
    df_home['average'] = pd.Series(np.mean(df_home.iloc[:,3:27].values,axis=1))
    df_home['peak'] = pd.Series(np.max(df_home.iloc[:,3:27].values,axis=1))
    
    home = nt("home",field_names=["cord","profile","peak","load"])
    dict_load = df_home.iloc[:,[0]+list(range(3,27))].set_index('hid').T.to_dict('list')
    dict_cord = df_home.iloc[:,0:3].set_index('hid').T.to_dict('list')
    dict_peak = dict(zip(df_home.hid,df_home.peak))
    dict_avg = dict(zip(df_home.hid,df_home.average))
    return home(cord=dict_cord,profile=dict_load,peak=dict_peak,load=dict_avg)   
    

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


def get_roads(homes, to_filepath=None):
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
    
    # Save the road graph as a gpickle file
    if to_filepath:
        nx.write_gpickle(osm_graph, to_filepath)
    
    return osm_graph

def read_roads_from_gpickle(filepath):
    if not os.path.exists(filepath):
        logger.error(f"{filepath} not present!!!")
        raise ValueError(f"{filepath} doesn't exist!")
    
    roads = nx.read_gpickle(filepath)
    return roads

def load_map(filename):
    if not os.path.exists(filename):
        logger.error(f"{filename} not present!!!")
        raise ValueError(f"{filename} doesn't exist!")
    else:
        df_map = pd.read_csv(
            filename, 
            sep = " ", header = None, 
            names = ["hid", "source", "target", "key"]
            )
        map_h2r = dict([(t.hid, (t.source, t.target, t.key)) \
                        for t in df_map.itertuples()])
    return map_h2r

def load_reverse_map(filename):
    if not os.path.exists(filename):
        logger.error(f"{filename} not present!!!")
        raise ValueError(f"{filename} doesn't exist!")
    else:
        with open(filename) as f:
            lines = f.readlines()
        map_r2h = {}
        for line in lines:
            temp = line.strip('\n').split(' ')
            link = tuple([int(m) for m in temp[:3]])
            reslist = [int(m) for m in temp[3:]]
            map_r2h[link] = reslist
    return map_r2h
