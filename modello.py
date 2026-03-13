#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:12:52 2023

@author: andreamarin
"""

import numpy as np
import math
import networkx as nx
from numpy import linalg as LA

from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import rc
import random
import tqdm

nodes = 10

def sedge(s,d=None):
    """ just return a sorted tuple to be used as a dictionary index.
        accepts an edge or a pair of nodes """
    if d:
        return tuple(sorted([s,d]))
    return tuple(sorted(s))

def create_graph(nodes=10):
    """ for testing purposes """
    lambdas = {}
    g = nx.path_graph(nodes) # a path graph
    for _,_,d in g.edges(data=True):
        d['SKR'] = 1
    for n in g:
        for m in g:
            if n == m:
                continue
            lambdas[(n,m)] = 4/(math.pow(nodes,2)-nodes)
    return g, lambdas

def relabel_graph(g):
    relabels = {}
    for node in g.nodes():
        relabels[node] = len(relabels)
    return nx.relabel_nodes(g, relabels)
    
def psi(s, d, rhos, lambdas, g):
    """ returns the key demand of link (s,d), given a previous iteration on
    rho, and the graph g from which we obtain gammas """
    t = 0.0 
    for start in g.nodes():
        for stop in g.nodes():
            if start == stop:
                continue
            path = nx.shortest_path(g, start, stop)
            edges = [(path[i],path[i+1]) for i in range(len(path)-1)]
            if (s,d) not in edges and (d,s) not in edges:
                continue
            fact  = 1.0
            for edge in edges:
                if edge == (s,d) or edge == (d,s):
                    break
                fact = fact * rhos[sedge(edge)] # edges are duplicated, but we want 
                                                 # to deduplicate. So b->a is mapped to a->b 
            t = t + lambdas[(start,stop)]*fact 
    return t

def compute_rhos(g, lambdas, verbose=True):
    rhos = dict.fromkeys([sedge(e) for e in g.edges()], 0.1)
    oldrhos = dict.fromkeys(rhos.keys(), 0)
    #relaxation coefficient
    alpha = 0.1

    # this is approximating rho, eq. 2
    last_alpha = 1
    if verbose:
        print('Starting graph analysis. Alpha progress: ', end='', flush=True)
    precision = 6
    while (LA.norm([oldrhos[k]-rhos[k] for k in rhos],2)>1/10**6):
        curr_alpha = LA.norm([oldrhos[k]-rhos[k] for k in rhos],2)
        if curr_alpha < last_alpha/10 and verbose:
            last_alpha = curr_alpha
            print(f'{curr_alpha:.6f}', end=', ', flush=True)

        oldrhos = rhos.copy()
        for s,d in g.edges():
            # see the comment on sorted edges in psi()
                rhos[sedge(s,d)]  = alpha * min(1.0, g.edges[s,d]['SKR']/psi(s, d, oldrhos, lambdas, g)) + (1-alpha)*oldrhos[sedge(s,d)]
    if verbose:
        print()
    r = []
    all_paths = dict(nx.all_pairs_shortest_path(g))
    prob_dict = defaultdict(list)
    for source in all_paths:
        for dest in all_paths[source]:
            path = all_paths[source][dest]
            lpath = len(path)
            edges = [(path[i],path[i+1]) for i in range(lpath-1)]
            prob = 1
            for edge in edges:
                prob = prob*rhos[sedge(edge)]
            prob_dict[lpath].append(prob) 
    return rhos, prob_dict

def plot_graphs(g, rhos, prob_dict):

    for edge in g.edges():
        g.edges[edge]['rho'] = rhos[sedge(edge)]
    
    fix, ax = plt.subplots()
    ax.bar([str(x) for x in rhos.keys()], rhos.values())
    ax.set_xlabel("Link")
    ax.set_ylabel("Prob. of filled") # prob that link i is non empty?
    plt.yscale('log')
    ax.set_title("Model")
    ax.tick_params(axis='x', labelrotation=90)
    ax.grid()
    #ax.legend(loc = 'lower right')

    fix, ax = plt.subplots()
    max_len = max(prob_dict.keys())
    lens = sorted(prob_dict.keys())
    ax.plot(lens,[np.average(prob_dict[k]) for k in lens])

    ax.set_xlabel("Path length")
    ax.set_ylabel("Prob. of success")
    ax.set_title("Model")
    ax.grid()
    #ax.legend(loc = 'lower right')
    edge_colors = []      
    for edge in g.edges():
        edge_colors.append(g.edges[edge]['rho'])
    fix, ax = plt.subplots()
    nx.draw(g, edge_color=edge_colors, edge_cmap=plt.cm.autumn, with_labels=True)
    plt.show()

if __name__ == '__main__':
    g, lambdas = create_graph()
    rhos, prob_dict = compute_rhos(g, lambdas)
    plot_graphs(g, rhos, prob_dict)