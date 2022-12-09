# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 17:04:22 2022

Author: Rounak Meyur
"""

import networkx as nx


g = nx.Graph()
g.add_edges_from(
    [(0,1), (1,2), (2,3), (0,4), (4,5), (0,6), 
     (10,11), (11,12), (12,13), (10,14), (14,15)]
    )

for n in g:
    if n in [0,10]:
        g.nodes[n]['label'] = 'T'
    else:
        g.nodes[n]['label'] = 'H'


tnodes = [n for n in g if g.nodes[n]['label']=='T']
lnodes = [n for n in g if nx.degree(g,n)==1]


paths = [' '.join([str(x) for x in nx.shortest_path(g, l, t)[::-1]]) \
         for l in lnodes for t in tnodes if nx.has_path(g,l,t)]

print(paths)

# for t in tnodes:
#     t_degree = nx.degree(g,t)
#     branches = [[t]] * t_degree
#     t_descendants = list(nx.descendants(g,t))
#     while (t_descendants != []):
#         for i in range(len(branches)):
#             branch = [n for n in branches[i]]
#             neighbors = [n for n in t_descendants if (n,branch[-1]) in g.edges]
#             if len(neighbors)!=0:
#                 node = neighbors[0]
#                 branch.append(node)
#                 t_descendants.remove(node)
#                 branches[i] = branch
#     print(branches)

    