# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 15:52:37 2022

Author: Rounak Meyur
"""

import logging
from typing import Union
logger = logging.getLogger(__name__)

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


class Link(LineString):
    """
    Derived class from Shapely LineString to compute metric distance based on 
    geographical coordinates over geometric coordinates.
    """
    def __init__(self,line_geom):
        super().__init__(line_geom)
        return
    
    
    def geod_length(self) -> float:
        """
        Computes the geographical length in meters between the extreme ends of 
        the link.

        Returns
        -------
        float
            The geodesic length of the link in meters.

        """
        if self.geom_type != 'LineString':
            print("Cannot compute length!!!")
            return None
        return sum([geodist(pt1,pt2) \
                    for pt1,pt2 in zip(self.coords,self.coords[1:])])
    
    def interpolate_points(self, sep:float=20) -> list:
        """
        Interpolate points along a link embedded on geographic plane. 

        Parameters
        ----------
        sep : float, optional
            Separation between interpolated points in meters. 
            The default is 20.

        Returns
        -------
        list
            List of point objects denoting the interpolated points.

        """
        points = []
        length = self.geod_length()
        for i in np.arange(0,length,sep):
            x,y = self.interpolate(i/length,normalized=True).xy
            xy = (x[0],y[0])
            points.append(Point(xy))
        if len(points)==0: 
            points.append(Point((self.xy[0][0],self.xy[1][0])))
        return points