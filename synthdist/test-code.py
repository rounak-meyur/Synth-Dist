# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:42:22 2022

Author: Rounak Meyur
"""

import logging
logger = logging.getLogger(__name__)



import os
import unittest
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from datetime import timedelta

import csv
import pandas as pd
import networkx as nx
from shapely.geometry import Point, MultiPoint, LineString, base as sg
from collections import namedtuple as nt


from inputs import load_homes, get_roads
from mapping import map_home_to_road, groups
from drawings import plot_roads, plot_homes
from secnet import candidate, secnet_milp


def get_fig_from_ax(ax, figsize, **kwargs):
    if not ax:
        no_ax = True
        ndim = kwargs.get('ndim', (1, 1))
        fig, ax = plt.subplots(*ndim, figsize=figsize)
    else:
        no_ax = False
        if not isinstance(ax, matplotlib.axes.Axes):
            getter = kwargs.get('ax_getter', lambda x: x[0])
            ax = getter(ax)
        fig = ax.get_figure()

    return fig, ax, no_ax


def close_fig(fig, to_file=None, show=True, **kwargs):
    if to_file:
        fig.savefig(to_file, **kwargs)
    if show:
        plt.show()
    plt.close(fig)
    pass


def timeit(f, *args, **kwargs):
    start = timer()
    outs = f(*args, **kwargs)
    end = timer()
    return outs, end - start








class SynthDist(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.home_path = "../data/load"
        
        self._out_dir = "../out"
        self._fig_dir = "../figs"
        self.fis = "test"
        pass
    
    # Out directory setter/ if not, create a directory
    @property
    def out_dir(self):
        return self._out_dir

    @out_dir.setter
    def out_dir(self, out):
        self._out_dir = out
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        pass
    
    # Figures directory setter/ if not, create a directory
    @property
    def fig_dir(self):
        return self._fig_dir

    @fig_dir.setter
    def fig_dir(self, fig):
        self._fig_dir = fig
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        pass
    
    
    def read_inputs(self, fis=None, filename=None):
        if not fis:
            fis = self.fis
        if not filename:
            filename = f"{fis}-home-load"
        
        # Get the input homes file path
        home_filepath = f"{self.home_path}/{filename}"

        # Retreive the load data
        homes = load_homes(home_filepath)
        
        # Retrieve the underlying road network from OpenStreetMaps
        roads = get_roads(homes)
        return homes, roads
    
    def map_inputs(self, 
                   homes=None, roads=None, 
                   to_file=None
                   ):
        if not homes:
            homes, _  = self.read_inputs()
        if not roads:
            roads = get_roads(homes)
        
        # map homes to nearest road link
        map_h2r = map_home_to_road(homes, roads)
        
        # write the mapping in a txt file
        if to_file:
            data_map = '\n'.join([' '.join([str(h),str(map_h2r[h][0]),
                                            str(map_h2r[h][1])]) \
                                  for h in map_h2r])
        with open(f"{self.out_dir}/{to_file}",'w') as f:
            f.write(data_map)
        return map_h2r
    
    def get_reverse_map(self, 
                        map_file=None, 
                        homes=None, roads=None, 
                        to_file=None
                        ):
        if not map_file:
            homes, roads = self.read_inputs()
            map_h2r = self.map_inputs(homes, roads)
        
        else:
            if not os.path.exists(f"{self.out_dir}/{map_file}"):
                logger.error(f"File {map_file} not present!!!")
                raise ValueError(f"{map_file} doesn't exist!")
            else:
                df_map = pd.read_csv(
                    f"{self.out_dir}/{map_file}", 
                    sep = " ", header = None, 
                    names = ["hid", "source", "target"]
                    )
                map_h2r = dict([(t.hid, (t.source, t.target)) \
                                for t in df_map.itertuples()])
        
        # get the reverse mapping
        map_r2h = groups(map_h2r)
        
        # write the mapping in a txt file
        if to_file:
            data_map = '\n'.join([' '.join([str(r[0]), str(r[1])] \
                                + [str(h) for h in map_r2h[r]]) \
                                for r in map_r2h if len(map_r2h[r]) > 0])
        with open(f"{self.out_dir}/{to_file}",'w') as f:
            f.write(data_map)
        return map_r2h
    
    def get_candidate_graph(self, 
                            )
                
                
            
    
    
    
    
    ##### -------- Plot functions ------------
    def plot_input_data(
            self, roads, homes, fis=None,
            ax=None, to_file=None, show=True,
            **kwargs
            ):
        kwargs.setdefault('figsize', (40, 20))
        fontsize = kwargs.get('fontsize', 30)
        do_return = kwargs.get('do_return', False)
        if not fis:
            fis = self.fis

        # ---- PLOT ----
        fig, ax, no_ax = get_fig_from_ax(ax, **kwargs)
        plot_roads(roads, ax, **kwargs)
        plot_homes(homes, ax, **kwargs)
        
        # ----- Legend handler ------
        ax.legend(loc='upper left', markerscale=3, fontsize=fontsize)
        ax.tick_params(left=False, bottom=False, 
                       labelleft=False, labelbottom=False)
        
        # ---- Edit the title of the plot ----
        title = f"${self.fis}$"
        if title_sfx := kwargs.get('title_sfx'):
            title = f"{title} : {title_sfx}"
        ax.set_title(title, fontsize=fontsize)

        if file_name_sfx := kwargs.get('file_name_sfx'):
            if not to_file:
                to_file = f"{fis}-inputs"
            to_file = f"{to_file}_{file_name_sfx}"

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = f"{to_file}"
            if suptitle_sfx := kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle} : {suptitle_sfx}"

            fig.suptitle(suptitle, fontsize=fontsize+3)
            close_fig(fig, to_file, show)

        if do_return:
            return fig, ax
        pass
    
    
    



class SynthDist_Montgomery(SynthDist):
    
    
    def __init__(self, methodName:str = ...) -> None:
        super().__init__(methodName)
        self.out_dir = "../out/test"
        self.fig_dir = "../figs/test"
        self.fis = "test"
        return
    
    # def test_input(self):
    #     # Get the homes and roads
    #     homes,roads = self.read_inputs()
        
    #     # check if function returns properly
    #     self.assertIsNotNone(homes)
    #     self.assertIsNotNone(roads)
        
    #     # Plot the input data
    #     self.plot_input_data(
    #         roads, homes, 
    #         title_sfx = "homes and road network in the geographic region",
    #         suptitle_sfx = "homes and roads",
    #         to_file = f"{self.fis}-inputs",
    #         )
    #     pass
    
    # def test_map_inputs(self):
    #     homes, roads = self.read_inputs()
    #     map_h2r = self.map_inputs(
    #         homes=homes, 
    #         roads=roads,
    #         to_file = f"{self.fis}-h2r_map.txt"
    #         )
    #     self.assertIsNotNone(map_h2r)
        
    #     pass

    def test_reverse_map(self):
        map_r2h = self.get_reverse_map(
            map_file = "test-h2r_map.txt",
            to_file = "test-r2h_map.txt")
        
        self.assertIsNotNone(map_r2h)
        pass
































if __name__ == '__main__':
    unittest.main()
