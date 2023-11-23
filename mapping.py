# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:40:23 2022

Author: Rounak Meyur
"""

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

from collections import defaultdict
from pyqtree import Index
from shapely.geometry import Point

def map_home_to_road(homes, road, to_filepath=None):
    M = MapOSM(road)
    H2Link = M.map_point(homes)
    
    # write the mapping in a txt file
    if to_filepath:
        data_map = '\n'.join(
            [' '.join([str(h),str(H2Link[h][0]),str(H2Link[h][1]),
                       str(H2Link[h][2])]) for h in H2Link])
        with open(to_filepath,'w') as f:
            f.write(data_map)
    
    return H2Link

class MapOSM:
    """
    Class consisting of attributes and methods to map OSM links to the residences
    """
    def __init__(self,road,radius=0.001):
        """
        Initializes the class object by creating a bounding box of known radius
        around each OSM road link.

        Parameters
        ----------
        road : networkx Multigraph
            The Open Street Map multigraph with node and edge attributes.
        radius : float, optional
            The radius of the bounding box around each road link. 
            The default is 0.01.

        Returns
        -------
        None.

        """
        longitudes = [road.nodes[n]['x'] for n in road.nodes()]
        latitudes = [road.nodes[n]['y'] for n in road.nodes()]
        xmin = min(longitudes); xmax = max(longitudes)
        ymin = min(latitudes); ymax = max(latitudes)
        bbox = (xmin,ymin,xmax,ymax)
        
        # keep track of edges so we can recover them later
        all_link = list(road.edges(keys=True))
        self.links = []
    
        # initialize the quadtree index
        self.idx = Index(bbox)
        
        # add edge bounding boxes to the index
        for i, link in enumerate(all_link):
            # create line geometry
            link_geom = road.edges[link]['geometry']
        
            # bounding boxes, with padding
            x1, y1, x2, y2 = link_geom.bounds
            bounds = x1-radius, y1-radius, x2+radius, y2+radius
        
            # add to quadtree
            self.idx.insert(i, bounds)
        
            # save the line for later use
            self.links.append((link_geom, bounds, link))
        return
    
    def map_point(self,points,radius=0.001):
        '''
        Finds the nearest link to the residence under consideration.
        '''
        Map2Link = {}
        for h in points.cord:
            pt = Point(points.cord[h])
            pt_bounds = pt.x-radius, pt.y-radius, pt.x+radius, pt.y+radius
            matches = self.idx.intersect(pt_bounds)
            
            # find closest path
            try:
                closest_path = min(matches, 
                                   key=lambda i: self.links[i][0].distance(pt))
                Map2Link[h] = self.links[closest_path][-1]
            except:
                Map2Link[h] = None
        
        # Delete unmapped points
        unmapped = [p for p in Map2Link if Map2Link[p]==None]
        for p in unmapped:
            del Map2Link[p]
        
        return Map2Link

def reverse_map(many_to_one, to_filepath = None):
    """
    Converts a many-to-one mapping into a one-to-many mapping.
    
    Parameters
    ----------
    many_to_one : dict
        Input dictionary.

    Returns
    -------
    dict
        a dictionary mapping values from `many_to_one`
        to sets of keys from `many_to_one` that have that value.

    """
    one_to_many = defaultdict(set)
    for v, k in many_to_one.items():
        one_to_many[k].add(v)
    D = dict(one_to_many)
    output_dict = {k:list(D[k]) for k in D}
    
    if to_filepath:
        data_map = '\n'.join([' '.join([str(i) for i in list(r)] +\
                                       [str(h) for h in output_dict[r]]) \
                            for r in output_dict if len(output_dict[r]) > 0])
        with open(to_filepath,'w') as f:
            f.write(data_map)
        
    return output_dict