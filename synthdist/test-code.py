# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 12:42:22 2022

Author: Rounak Meyur
"""

import logging
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")



import os
import unittest
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm import tqdm

import csv
import pandas as pd
import networkx as nx
from shapely.geometry import Point, MultiPoint, LineString, base as sg
from collections import namedtuple as nt


from inputs import load_homes, get_roads, read_roads_from_gpickle
from inputs import load_map, load_reverse_map
from mapping import map_home_to_road, reverse_map
from drawings import plot_roads, plot_homes, plot_candidate, plot_secnet
from secnet import candidate, secnet_milp, data_secnet


def get_fig_from_ax(ax, **kwargs):
    if not ax:
        no_ax = True
        ndim = kwargs.get('ndim', (1, 1))
        figsize = kwargs.get('figsize', (10, 10))
        constrained_layout = kwargs.get('constrained_layout', False)
        fig, ax = plt.subplots(*ndim, figsize=figsize, 
                               constrained_layout=constrained_layout)
    else:
        no_ax = False
        if not isinstance(ax, matplotlib.axes.Axes):
            if isinstance(ax, list):
                getter = kwargs.get('ax_getter', lambda x: x[0])
                ax = getter(ax)
            if isinstance(ax, dict):
                getter = kwargs.get('ax_getter', lambda x: next(iter(ax.values())))
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
        self._grb_dir = "../gurobi"
        self.state_ID = 51
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
    
    # Gurobi directory setter/ if not, create a directory
    @property
    def grb_dir(self):
        return self._grb_dir

    @grb_dir.setter
    def grb_dir(self, grb):
        self._grb_dir = grb
        if not os.path.exists(self.grb_dir):
            os.makedirs(self.grb_dir)
        pass
    
    def read_homes(self, fis=None, filename=None):
        if not fis:
            fis = self.fis
        if not filename:
            filename = f"{fis}-home-load"
        
        # Get the input homes file path
        home_filepath = f"{self.home_path}/{filename}"

        # Retreive the load data
        homes = load_homes(home_filepath)
        return homes
    
    def read_roads(self, fis=None, filename=None):
        if not fis:
            fis = self.fis
        
        if not filename:
            filename = f"{fis}-roads.gpickle"
        
        # Get the roads gpickle file path
        roads = read_roads_from_gpickle(f"{self.out_dir}/{filename}")
        return roads
    
    
    def read_inputs(self, fis=None, filename=None):
        # Retrieve the homes
        homes = self.read_homes(
            fis=fis, 
            filename=filename
            )
        
        # Retrieve the underlying road network from OpenStreetMaps
        roads = get_roads(
            homes, 
            to_filepath = f"{self.out_dir}/{self.fis}-roads.gpickle"
            )
        return homes, roads
    
    def map_inputs(self, homes, roads, to_file=None):
        map_h2r = map_home_to_road(
            homes, roads, 
            to_filepath = f"{self.out_dir}/{to_file}"
            )
        return map_h2r
    
    def get_reverse_map(self, map_h2r=None, map_file=None, to_file=None):
        if not map_h2r:
            if not map_file:
                map_file = f"{self.fis}-map_h2r.txt"
            
            map_h2r = load_map(f"{self.out_dir}/{map_file}")
        
        # get the reverse mapping
        map_r2h = reverse_map(
            map_h2r, 
            to_filepath = f"{self.out_dir}/{self.fis}-map_r2h.txt"
            )
        return map_r2h
    
    def read_reverse_map(self, reverse_mapfile=None):
        if not reverse_mapfile:
            reverse_mapfile = f"{self.fis}-map_r2h.txt"
        map_r2h = load_reverse_map(f"{self.out_dir}/{reverse_mapfile}")
        return map_r2h
    
    def get_candidate_graph(self, link, homes):
        # change the heuristic based on the number of residences mapped
        if len(homes) > 350: 
            heuristic = 3
        elif len(homes) > 150:
            heuristic = 5
        elif len(homes) > 80:
            heuristic = 15
        else:
            heuristic = None
        
        # construct the candidate network
        graph = candidate(
            link, homes, 
            heuristic = heuristic)
        return graph
    
    def get_secnet_link(self, candidate_graph, temp_path):
        # change the parameters based on the number of residences mapped
        nhomes = len([n for n in candidate_graph \
                      if candidate_graph.nodes[n]['label'] == 'H'])
        if nhomes > 350: 
            maximum_hops = 100
            maximum_rating = 150
        elif nhomes > 150:
            maximum_hops = 80
            maximum_rating = 100
        elif nhomes > 80:
            maximum_hops = 40
            maximum_rating = 60
        else:
            maximum_hops = 10
            maximum_rating = 25
        
        # construct the optimal secondary network
        forest = secnet_milp(
            candidate_graph, temp_path,
            max_hops = maximum_hops, max_rating = maximum_rating)
        
        return forest
    
    def get_secnet_region(self, map_r2h, homes, roads):
        for k, link in tqdm(enumerate(map_r2h), 
                            ncols = 100,
                            desc = "Created secondary network"):
            linkgeom = roads.edges[link]['geometry']
            homelist = map_r2h[link]
            dict_homes = {h:{'cord': homes.cord[h], 
                              'load': homes.load[h]} for h in homelist}
            
            
            cand_graph = self.get_candidate_graph(linkgeom, dict_homes)
            secnet = self.get_secnet_link(cand_graph, 
                                          temp_path=f"{self.grb_dir}/{k}")
            
            if isinstance(self.fis, str):
                counter = int(f"{self.state_ID}500500000")
            else:
                counter = int(f"{self.state_ID}{self.fis}500000")
            counter = data_secnet(secnet, counter, 
                                     f"{self.out_dir}/{self.fis}")
            
            
        return
      
    
    
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
        if file_name_sfx := kwargs.get('file_name_sfx'):
            if not to_file:
                to_file = f"{fis}-inputs"
            to_file = f"{to_file}_{file_name_sfx}"

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = "inputs to the model"
            if suptitle_sfx := kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle} : {suptitle_sfx}"

            fig.suptitle(suptitle, fontsize=fontsize+3)
            close_fig(fig, to_file, show)

        if do_return:
            return fig, ax
        pass
    
    def plot_secnet_results(
            self, output_graph, link_geom,
            candidate_graph = None,
            plot_list = ["input", "candidate", "output"],
            fis=None,
            ax=None, to_file=None, show=True,
            **kwargs
            ):
        kwargs.setdefault('figsize', (40, 20))
        fontsize = kwargs.get('fontsize', 30)
        do_return = kwargs.get('do_return', False)
        if not fis:
            fis = self.fis

        # ---- PLOT ----
        if not candidate_graph:
            print("Candidate network not available to be plotted!!!")
            fig, axs, no_ax = get_fig_from_ax(ax, ndim=(1,1), **kwargs)
            plot_secnet(output_graph, link_geom, axs)
        
        else:
            fig, axs, no_ax = get_fig_from_ax(
                ax, ndim=(1,len(plot_list)), **kwargs)
            for i,item in enumerate(plot_list):
                if item == "input":
                    plot_candidate(candidate_graph, link_geom, axs[i], 
                                   show_candidate=False, **kwargs)
                    axs[i].set_title("road link and mapped homes", 
                                     fontsize=fontsize)
                
                elif item == "candidate":
                    plot_candidate(candidate_graph, link_geom, axs[i], **kwargs)
                    axs[i].set_title("candidate edges for secondary", 
                                     fontsize=fontsize)
                
                elif item == "output":
                    plot_secnet(output_graph, link_geom, axs[i])
                    axs[i].set_title("optimal secondary network", 
                                     fontsize=fontsize)
                
                else:
                    logger.error(f"{item} not recognized!!!")
                    raise ValueError(f"{item} not recognized!")
        
        # ---- Edit the title of the plot ----
        if file_name_sfx := kwargs.get('file_name_sfx'):
            if not to_file:
                to_file = f"{fis}-link-secnet"
            to_file = f"{to_file}_{file_name_sfx}"

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle = "optimal secondary network construction"
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
        self.grb_dir = "../gurobi/test"
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
    #         suptitle_sfx = "homes and roads in the geographic region",
    #         file_name_sfx = "inputs",
    #         )
    #     pass
    
    # def test_map_inputs(self):
    #     homes, roads = self.read_inputs()
    #     map_h2r = self.map_inputs(
    #         homes=homes, 
    #         roads=roads,
    #         to_file = f"{self.fis}-map_h2r.txt"
    #         )
    #     self.assertIsNotNone(map_h2r)
        
    #     pass

    # def test_reverse_map(self):
    #     map_r2h = self.get_reverse_map(
    #         map_file = "test-map_h2r.txt",
    #         to_file = "test-map_r2h.txt")
        
    #     self.assertIsNotNone(map_r2h)
    #     pass
    
    
    def test_create_secnet_link(self):
        map_r2h = self.read_reverse_map()
        self.assertIsNotNone(map_r2h)
        
        # get the homes and roads
        homes = self.read_homes()
        roads = self.read_roads()
        self.assertIsNotNone(homes)
        self.assertIsNotNone(roads)
        
        # Choose an example link
        link = [r for r in map_r2h if 10 < len(map_r2h[r]) < 20][10]
        linkgeom = roads.edges[link]['geometry']
        link_index = list(map_r2h.keys()).index(link)
        homelist = map_r2h[link]
        dict_homes = {h:{'cord': homes.cord[h], 
                          'load': homes.load[h]} for h in homelist}
        
        # Generate the candidate graph
        cand_graph = self.get_candidate_graph(linkgeom, dict_homes)
        self.assertIsNotNone(cand_graph)
        
        # Solve the optimization problem
        temp_path=f"{self.grb_dir}/{link_index}"
        secnet = self.get_secnet_link(
            cand_graph, temp_path
            )
        self.assertIsNotNone(secnet)
        
        if isinstance(self.fis, str):
            counter = int(f"{self.state_ID}500500000")
        else:
            counter = int(f"{self.state_ID}{self.fis}500000")
        counter = data_secnet(secnet, link, linkgeom, counter, 
                                 f"{self.out_dir}/{self.fis}",
                                 write_mode="w")
        
        # Plot the candidate graph
        self.plot_secnet_results(
            secnet, linkgeom, candidate_graph=cand_graph,
            plot_list = ["input", "candidate", "output"],
            suptitle_sfx = "input, candidate and output",
            file_name_sfx = "input_candidate_output",
            figsize = (50,20),
            )
        pass
    
    # def test_create_secnet_region(self):
    #     map_r2h = self.read_reverse_map()
    #     self.assertIsNotNone(map_r2h)
        
    #     # get the homes and roads
    #     homes = self.read_homes()
    #     roads = self.read_roads()
    #     self.assertIsNotNone(homes)
    #     self.assertIsNotNone(roads)
        
    #     self.get_secnet_region(map_r2h, homes, roads)
    #     pass
































if __name__ == '__main__':
    unittest.main()
