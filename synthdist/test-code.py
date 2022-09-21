# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:42:22 2022

@author: rm5nz
"""

import logging
logger = logging.getLogger(__name__)


from inputs import load_homes, get_homes, get_roads
from mapping import map_home_to_road


df = load_homes("test-homes")
homes = get_homes(df)
roads = get_roads(homes)

H2L = map_home_to_road(homes,roads)
