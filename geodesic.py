# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:52:37 2022

Author: Rounak Meyur
"""
from typing import Union
from shapely.geometry import Point, LineString
from geographiclib.geodesic import Geodesic
import numpy as np



def geodist(geomA : Union[tuple,list,Point], 
            geomB : Union[tuple,list,Point]) -> float:
    """
    Computes the geodesic distance between the two points embedded on the 
    geographic plane

    Parameters
    ----------
    geomA : Union[tuple,list,Point]
        Tuple, list or Point object defining the geometry of 
        the first coordinate.
    geomB : Union[tuple,list,Point]
        Tuple, list or Point object defining the geometry of 
        the second coordinate.

    Returns
    -------
    float
        The geodesic length between two points embedded on the geoegraphic
        plane in meters.

    """
    if type(geomA) != Point: geomA = Point(geomA)
    if type(geomB) != Point: geomB = Point(geomB)
    geod = Geodesic.WGS84
    return geod.Inverse(geomA.y, geomA.x, geomB.y, geomB.x)['s12']


def interpolate(line_geom : LineString, sep : float = 20):
    """
    Interpolate points along a link embedded on geographic plane. 

    Parameters
    ----------
    line_geom : LineString
        LineString geometry representing the input linear shape.
    sep : float, optional
        Minimum separation between a pair of transformers. 
        The default is 20.

    Returns
    -------
    points : list of shapely Point objects
        shapely point objects denoting the interpolated geometry.

    """
    # compute the geodesic length of the linestring object
    length = sum([geodist(pt1,pt2) \
                for pt1,pt2 in zip(line_geom.coords,
                                   line_geom.coords[1:])])
        
    # interpolate points
    num_vert = int(round(length / sep))
    if num_vert == 0:
        num_vert = 1
    geom_interpolated = LineString(
        [line_geom.interpolate(float(n) / num_vert, normalized=True)\
         for n in range(num_vert + 1)])
    points = [Point(pt) for pt in geom_interpolated.coords]
    return points