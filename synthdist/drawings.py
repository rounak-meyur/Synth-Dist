# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:53:11 2022

Author: Rounak Meyur
"""

from shapely.geometry import Point
import geopandas as gpd




def plot_roads(roads, ax, **kwargs):
    # ------ parameters for plot ------
    nodecolor = kwargs.get("nodecolor", 'black')
    nodesize = kwargs.get("nodesize", 5)
    edgecolor = kwargs.get("edgecolor", 'black')
    width = kwargs.get("width", 1.0)
    alpha = kwargs.get("alpha", 1.0)
    style = kwargs.get("linestyle", 'dashed')
    
    # ------ plot the nodes -----------
    d = {'nodes':[n for n in roads.nodes],
         'geometry':[Point(roads.nodes[n]["x"],roads.nodes[n]["y"]) \
                     for n in roads.nodes]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax = ax, color = nodecolor, markersize = nodesize, 
                  alpha = alpha, label = "road nodes")
    
    # ------ plot the edges -----------
    d = {'edges':[e for e in roads.edges],
         'geometry':[roads.edges[e]['geometry'] for e in roads.edges]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax = ax, edgecolor = edgecolor, linewidth = width,
                  linestyle = style, alpha = alpha, label = "road edges")
    return


def plot_homes(homes, ax, **kwargs):
    # ------ parameters for plot ------
    color = kwargs.get("color", 'red')
    size = kwargs.get("size", 20)
    alpha = kwargs.get("alpha", 1.0)
    
    # ------ plot the nodes -----------
    d = {'nodes':[h for h in homes.cord],
         'geometry':[Point(homes.cord[h]) for h in homes.cord]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax = ax, color = color, markersize = size, 
                  alpha = alpha, label = "residences")
    return