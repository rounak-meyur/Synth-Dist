# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 13:53:11 2022

Author: Rounak Meyur
"""

from shapely.geometry import Point, LineString
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



def plot_candidate(graph, linkgeom, ax, **kwargs):
    # ------ parameters for plot ------
    homecolor = kwargs.get("home_color", 'red')
    homesize = kwargs.get("home_size", 200)
    tsfrcolor = kwargs.get("tsfr_color", 'green')
    tsfrsize = kwargs.get("tsfr_size", 200)
    linkcolor = kwargs.get("link_color", 'black')
    linkwidth = kwargs.get("link_width", 3)
    edgecolor = kwargs.get("edge_color", 'green')
    edgewidth = kwargs.get("edge_width", 1)
    alpha = kwargs.get("alpha", 1.0)
    
    show_candidate = kwargs.get("show_candidate", True)
    
    # ------ plot the residence nodes -----------
    hnodes = [n for n in graph if graph.nodes[n]['label']=='H']
    d = {'nodes':hnodes,
         'geometry':[Point(graph.nodes[n]["cord"]) for n in hnodes]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax = ax, color = homecolor, markersize = homesize, 
                  alpha = alpha, label = "residences")
    
    # ------ plot the probable transformer nodes -----------
    tnodes = [n for n in graph if graph.nodes[n]['label']=='T']
    d = {'nodes':tnodes,
         'geometry':[Point(graph.nodes[n]["cord"]) for n in tnodes]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax = ax, color = tsfrcolor, markersize = tsfrsize, 
                  alpha = alpha, label = "probable transformers")
    
    # ------ plot the candidate edges -----------
    if show_candidate:
        d = {'edges':[e for e in graph.edges],
             'geometry':[LineString((Point(graph.nodes[e[0]]["cord"]),
                                     Point(graph.nodes[e[1]]["cord"]))) \
                         for e in graph.edges]}
        df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
        df_edges.plot(ax = ax, edgecolor = edgecolor, linewidth = edgewidth,
                      linestyle = "dashed", alpha = alpha, 
                      label = "candidate edges")
    
    # ------ plot the road link -----------
    d = {'edges':[0],
         'geometry':[linkgeom]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax = ax, edgecolor = linkcolor, linewidth = linkwidth,
                  linestyle = "dashed", alpha = alpha, label = "road link")
    
    # ----- Legend handler ------
    fontsize = kwargs.get('fontsize', 30)
    ax.legend(loc='best', markerscale=3, fontsize=fontsize)
    ax.tick_params(left=False, bottom=False, 
                    labelleft=False, labelbottom=False)
    return

def plot_secnet(graph, linkgeom, ax, **kwargs):
    # ------ parameters for plot ------
    homecolor = kwargs.get("home_color", 'red')
    homesize = kwargs.get("home_size", 200)
    tsfrcolor = kwargs.get("tsfr_color", 'green')
    tsfrsize = kwargs.get("tsfr_size", 200)
    linkcolor = kwargs.get("link_color", 'black')
    linkwidth = kwargs.get("link_width", 3)
    edgecolor = kwargs.get("edge_color", 'red')
    edgewidth = kwargs.get("edge_width", 1)
    alpha = kwargs.get("alpha", 1.0)
    
    # ------ plot the residence nodes -----------
    hnodes = [n for n in graph if graph.nodes[n]['label']=='H']
    d = {'nodes':hnodes,
         'geometry':[Point(graph.nodes[n]["cord"]) for n in hnodes]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax = ax, color = homecolor, markersize = homesize, 
                  alpha = alpha, label = "residences")
    
    # ------ plot the transformer nodes -----------
    tnodes = [n for n in graph if graph.nodes[n]['label']=='T']
    d = {'nodes':tnodes,
         'geometry':[Point(graph.nodes[n]["cord"]) for n in tnodes]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax = ax, color = tsfrcolor, markersize = tsfrsize, 
                  alpha = alpha, label = "local transformers")
    
    # ------ plot the secondary edges -----------
    d = {'edges':[e for e in graph.edges],
         'geometry':[LineString((Point(graph.nodes[e[0]]["cord"]),
                                 Point(graph.nodes[e[1]]["cord"]))) \
                     for e in graph.edges]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax = ax, edgecolor = edgecolor, linewidth = edgewidth,
                  linestyle = "solid", alpha = alpha, 
                  label = "secondary edges")
    
    # ------ plot the road link -----------
    d = {'edges':[0],
         'geometry':[linkgeom]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax = ax, edgecolor = linkcolor, linewidth = linkwidth,
                  linestyle = "dashed", alpha = alpha, label = "road link")
    
    # ----- Legend handler ------
    fontsize = kwargs.get('fontsize', 30)
    ax.legend(loc='best', markerscale=3, fontsize=fontsize)
    ax.tick_params(left=False, bottom=False, 
                    labelleft=False, labelbottom=False)
    return





























