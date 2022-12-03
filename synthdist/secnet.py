# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:37:38 2022

Author: Rounak Meyur
"""

import sys
import gurobipy as grb
import networkx as nx
import numpy as np
from geodesic import Link, geodist
from shapely.geometry import LinearRing
from scipy.spatial import Delaunay
from itertools import combinations

def candidate(linkgeom, homes, **kwargs):
    """
    Creates the base network to carry out the optimization problem. The base graph
    may be a Delaunay graph or a full graph depending on the size of the problem.
    
    Inputs: 
        linkgeom: shapely geometry of road link.
        homes: dictionary of residence data
        sep: minimum separation in meters between the transformers.
        penalty: penalty factor for crossing the link.
        heuristic: join transformers to nearest few nodes given by heuristic.
        Used to create the dummy graph
    Outputs:graph: the generated base graph also called the dummy graph
            transformers: list of points along link which are probable locations 
            of transformers.
    """
    # Get the keyword arguments
    sep = kwargs.get("separation", 50)
    penalty = kwargs.get("penalty", 0.5)
    heuristic = kwargs.get("heuristic", None)
    
    # Interpolate points along link for probable transformer locations
    interpolated_points = Link(linkgeom).interpolate_points(sep)
    prob_tsfr = {i:pt for i,pt in enumerate(interpolated_points)}
    
    # Identify which side of road each home is located
    link_cords = list(linkgeom.coords)
    sides = {h:1 if LinearRing(link_cords+[tuple(homes[h]['cord']),
            link_cords[0]]).is_ccw else -1 for h in homes}
    
    # Node attributes
    homelist = [h for h in homes]
    cord = {h:homes[h]['cord'] for h in homes}
    load = {h:homes[h]['load']/1000.0 for h in homes}
    
    # Create the base graph
    graph = nx.Graph()
    if len(homes)>10:
        # create a Delaunay graph from the home coordinates
        points = np.array([cord[h] for h in homes])
        triangles = Delaunay(points).simplices
        edgelist = []
        for t in triangles:
            edges = [(homelist[t[0]],homelist[t[1]]),
                     (homelist[t[1]],homelist[t[2]]),
                     (homelist[t[2]],homelist[t[0]])]
            edgelist.extend(edges)
        graph.add_edges_from(edgelist)
    else:
        # create a complete graph from the home coordinates
        edges = combinations(homelist,2)
        graph.add_edges_from(edges)
    
    # Add the new edges
    if heuristic != None:
        # select candidate edges between nearby points
        new_edges = []
        for t in prob_tsfr:
            distlist = [geodist(prob_tsfr[t],cord[h]) for h in cord]
            imphomes = np.array(homelist)[np.argsort(distlist)[:heuristic]]
            new_edges.extend([(t,n) for n in imphomes])
    else:
        # candidate edges from all possible pairs of points
        new_edges = [(t,n) for t in prob_tsfr for n in homes]
    graph.add_edges_from(new_edges)
    
    # Update the attributes of nodes with transformer attributes
    for n in graph:
        if n in prob_tsfr:
            graph.nodes[n]['cord'] = (prob_tsfr[n].x, prob_tsfr[n].y)
            graph.nodes[n]['load'] = 0.0
            graph.nodes[n]['label'] = 'T'
            sides[n] = 0
        else:
            graph.nodes[n]['cord'] = cord[n]
            graph.nodes[n]['load'] = load[n]
            graph.nodes[n]['label'] = 'H'
    
    # get cost of candidate edges
    for e in graph.edges:
        length = geodist(graph.nodes[e[0]]['cord'],graph.nodes[e[1]]['cord'])
        factor = 1 + ( penalty * abs( sides[e[0]] - sides[e[1]] ) )
        graph.edges[e]['cost'] = length * factor
    return graph


def mycallback(model, where):
    if where == grb.GRB.Callback.MIP:
        # General MIP callback
        objbst = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        time = model.cbGet(grb.GRB.Callback.RUNTIME)
        if(time>300 and abs(objbst - objbnd) < 0.005 * (1.0 + abs(objbst))):
            print('Stop early - 0.50% gap achieved time exceeds 5 minutes')
            model.terminate()
        elif(time>60 and abs(objbst - objbnd) < 0.0025 * (1.0 + abs(objbst))):
            print('Stop early - 0.25% gap achieved time exceeds 1 minute')
            model.terminate()
        elif(time>300 and abs(objbst - objbnd) < 0.01 * (1.0 + abs(objbst))):
            print('Stop early - 1.00% gap achieved time exceeds 5 minutes')
            model.terminate()
        elif(time>480 and abs(objbst - objbnd) < 0.05 * (1.0 + abs(objbst))):
            print('Stop early - 5.00% gap achieved time exceeds 8 minutes')
            model.terminate()
        elif(time>600 and abs(objbst - objbnd) < 0.1 * (1.0 + abs(objbst))):
            print('Stop early - 10.0% gap achieved time exceeds 10 minutes')
            model.terminate()
        elif(time>1500 and abs(objbst - objbnd) < 0.15 * (1.0 + abs(objbst))):
            print('Stop early - 15.0% gap achieved time exceeds 25 minutes')
            model.terminate()
        elif(time>3000 and abs(objbst - objbnd) < 0.2 * (1.0 + abs(objbst))):
            print('Stop early - 20.0% gap achieved time exceeds 50 minutes')
            model.terminate()
        elif(time>6000 and abs(objbst - objbnd) < 0.3 * (1.0 + abs(objbst))):
            print('Stop early - 30.0% gap achieved time exceeds 100 minutes')
            model.terminate()
        elif(time>12000 and abs(objbst - objbnd) < 0.4 * (1.0 + abs(objbst))):
            print('Stop early - 40.0% gap achieved time exceeds 200 minutes')
            model.terminate()
    return


def secnet_milp(graph, path, **kwargs):
    # Get the keyword arguments
    H = kwargs.get("max_hops", 10)
    M = kwargs.get("max_rating", 25)
    
    edges = list(graph.edges)
    nodes = list(graph.nodes)
    hindex = [i for i,n in enumerate(nodes) if graph.nodes[n]['label'] == 'H']
    
    A = nx.incidence_matrix(graph,nodelist=nodes,edgelist=edges,oriented=True)
    I = nx.incidence_matrix(graph,nodelist=nodes,edgelist=edges,oriented=False)
    
    # get coefficients and parameters from graph attributes
    COST = nx.get_edge_attributes(graph,name='cost')
    LOAD = nx.get_node_attributes(graph,name='load')
    c = np.array([1e-3*COST[e] for e in edges])
    p = np.array([1e-3*LOAD[nodes[i]] for i in hindex])
    y = np.ones(shape=(len(hindex),))
    
    # Initialize the model
    model = grb.Model(name="Get SecNet")
    model.ModelSense = grb.GRB.MINIMIZE
    
    # Define variables
    x = model.addMVar(len(edges), vtype=grb.GRB.BINARY, name='x')
    f = model.addMVar(len(edges), vtype=grb.GRB.CONTINUOUS,
                      lb=-grb.GRB.INFINITY, name='f')
    z = model.addMVar(len(edges), vtype=grb.GRB.CONTINUOUS,
                      lb=-grb.GRB.INFINITY, name='z')
    
    # Add radiality constraint
    model.addConstr( x.sum() == len(hindex) )
    model.addConstr( (A[hindex,:] @ f == -p) )
    model.addConstr( f - (M * x) <= 0 )
    model.addConstr( f + (M * x) >= 0 )
    
    # Add hop constraint as a heuristic constraint
    model.addConstr( A[hindex,:] @ z == -y )
    model.addConstr( z - (H * x) <= 0 )
    model.addConstr( z + (H * x) >= 0 )
    model.addConstr( I[hindex,:] @ x <= (2 * y) )
    
    # add objective function
    model.setObjective( c @ x )
    
    # write the model
    model.update()
    model.write(f"{path}-secondary.lp")
    
    # Turn off display and heuristics
    grb.setParam('OutputFlag', 0)
    grb.setParam('Heuristics', 0)
    
    # Open log file
    logfile = open(f'{path}-gurobi.log', 'w')
    
    # Pass data into my callback function
    model._lastiter = -grb.GRB.INFINITY
    model._lastnode = -grb.GRB.INFINITY
    model._logfile = logfile
    model._vars = model.getVars()
    
    # Solve model and capture solution information
    model.optimize(mycallback)
    # Close log file
    logfile.close()
    if model.SolCount == 0:
        print(f'No solution found, optimization status = {model.Status}')
        sys.exit(0)
    else:
        x_optimal = x.getAttr("x").tolist()
        optimal_edges = [e for i,e in enumerate(edges) if x_optimal[i]>0.5]
    
    # Generate the forest of trees
    forest = nx.Graph()
    forest.add_edges_from(optimal_edges)
    
    # Label the nodes and edges
    for n in forest:
        forest.nodes[n]['cord'] = graph.nodes[n]['cord']
        forest.nodes[n]['label'] = graph.nodes[n]['label']
        forest.nodes[n]['load'] = graph.nodes[n]['load']
    
    return forest

def data_secnet(forest, link, start_count):
    tnodes = [n for n in forest if forest.nodes[n]['label']=='T']
    t_data = []
    count = start_count
    for i,t in enumerate(tnodes):
        t_id = start_count + i
        cord = list(forest.nodes[t]['cord'])
        homes = list(nx.descendants(forest, t))
        load = sum([forest.nodes[h]['load'] for h in homes])
        
        # Data record for transformers
        t_data.append(' '.join([str(x) for x in [t_id, load] + cord]))
    return



