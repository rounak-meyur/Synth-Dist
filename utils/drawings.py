from shapely.geometry import Point, LineString
import geopandas as gpd
import os
import sys

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize(
        log_file_path="logs/test_drawing.log", 
        log_config_path="configs/logging_config.ini"
    )

logger = LogManager.get_logger("drawing")


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
    d = {'nodes':[home.id for home in homes],
         'geometry':[Point(home.cord[0], home.cord[1]) for home in homes]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax = ax, color = color, markersize = size, 
                  alpha = alpha, label = "residences")
    return

def plot_substations(subs, ax, **kwargs):
    # ------ parameters for plot ------
    color = kwargs.get("color", 'royalblue')
    size = kwargs.get("size", 200)
    alpha = kwargs.get("alpha", 1.0)
    
    # ------ plot the nodes -----------
    d = {'nodes':[sub.id for sub in subs],
         'geometry':[Point(sub.cord[0], sub.cord[1]) for sub in subs]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax = ax, color = color, markersize = size, 
                  alpha = alpha, label = "substations")
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
    
    # ------ plot the residence nodes on side A -----------
    hnodes = [n for n in graph if graph.nodes[n]['label']=='H' and graph.nodes[n]['side']==1]
    d = {'nodes':hnodes,
         'geometry':[Point(graph.nodes[n]["cord"]) for n in hnodes]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax = ax, color = homecolor, marker = "*", 
                  markersize = homesize, alpha = alpha, label = "residences on side A")
    
    # ------ plot the residence nodes on side B -----------
    hnodes = [n for n in graph if graph.nodes[n]['label']=='H' and graph.nodes[n]['side']==-1]
    d = {'nodes':hnodes,
         'geometry':[Point(graph.nodes[n]["cord"]) for n in hnodes]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax = ax, color = homecolor, marker = "^", 
                  markersize = homesize, alpha = alpha, label = "residences on side B")
    
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
    ax.legend(loc='upper left', markerscale=1.5, fontsize=fontsize)
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
    ax.legend(loc='upper left', markerscale=1.5, fontsize=fontsize)
    ax.tick_params(left=False, bottom=False, 
                    labelleft=False, labelbottom=False)
    return

def plot_combined_road_transformer(combined_network, ax, **kwargs):
    # ------ parameters for plot ------
    roadnodecolor = kwargs.get("roadnodecolor", 'black')
    roadnodesize = kwargs.get("roadnodesize", 5)
    tsfrnodecolor = kwargs.get("transformercolor", 'seagreen')
    tsfrnodesize = kwargs.get("transformersize", 50)
    edgecolor = kwargs.get("edgecolor", 'black')
    width = kwargs.get("width", 1.0)
    alpha = kwargs.get("alpha", 1.0)
    style = kwargs.get("linestyle", 'dashed')
    
    # ------ plot the road nodes -----------
    d = {'nodes':[n for n in combined_network.nodes if combined_network.nodes[n]['label']=='R'],
         'geometry':[Point(combined_network.nodes[n]["cord"][0],
                           combined_network.nodes[n]["cord"][1]) \
                     for n in combined_network.nodes if combined_network.nodes[n]['label']=='R']}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax = ax, color = roadnodecolor, markersize = roadnodesize, 
                  alpha = alpha, label = "road nodes")
    
    # ------ plot the transformer nodes -----------
    d = {'nodes':[n for n in combined_network.nodes if combined_network.nodes[n]['label']=='T'],
         'geometry':[Point(combined_network.nodes[n]["cord"][0],
                           combined_network.nodes[n]["cord"][1]) \
                     for n in combined_network.nodes if combined_network.nodes[n]['label']=='T']}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax = ax, color = tsfrnodecolor, markersize = tsfrnodesize, 
                  alpha = alpha, label = "transformers")
    
    # ------ plot the edges -----------
    d = {'edges':[e for e in combined_network.edges],
         'geometry':[combined_network.edges[e]['geometry'] for e in combined_network.edges]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax = ax, edgecolor = edgecolor, linewidth = width,
                  linestyle = style, alpha = alpha, label = "road edges")
    
    # ----- Legend handler ------
    fontsize = kwargs.get('fontsize', 30)
    ax.legend(loc='upper left', markerscale=1.5, fontsize=fontsize)
    ax.tick_params(left=False, bottom=False, 
                    labelleft=False, labelbottom=False)
    return