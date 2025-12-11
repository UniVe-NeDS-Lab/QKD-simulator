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

nodes = 10

def create_graph(nodes=10):
    """ for testing purposes """
    lambdas = {}
    g = nx.path_graph(nodes) # a path graph
    for _,_,d in g.edges(data=True):
        d['gamma'] = 1
    for n in g:
        for m in g:
            if n == m:
                continue
            lambdas[(n,m)] = 4/(math.pow(nodes,2)-nodes)
    return g, lambdas
    
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
                fact = fact * rhos[tuple(sorted(edge))] # edges are duplicated, but we want 
                                                 # to deduplicate. So b->a is mapped to a->b 
            t = t + lambdas[(start,stop)]*fact 
    return t

def compute_rhos(g, lambdas):
    rhos = dict.fromkeys(g.edges(), 0.1)
    oldrhos = dict.fromkeys(rhos.keys(), 0)

    #relaxation coefficient
    alpha = 0.1

    # this is approximating rho, eq. 2
    while (LA.norm([oldrhos[k]-rhos[k] for k in rhos],2)>0.000001):
        oldrhos = rhos.copy()
        for s,d in g.edges():
            # see the comment on sorted edges in psi()
            rhos[tuple(sorted([s,d]))]  = alpha * min(1.0, g.edges[s,d]['gamma']/psi(s, d, oldrhos, lambdas, g)) + (1-alpha)*oldrhos[(s,d)]
    return rhos

def plot_graphs(g, rhos):
    fix, ax = plt.subplots()
    ax.bar([str(x) for x in rhos.keys()], rhos.values())
    ax.set_xlabel("Link")
    ax.set_ylabel("Prob. of filled") # prob that link i is non empty?
    plt.yscale('log')
    ax.set_title("Model")
    ax.grid()
    #ax.legend(loc = 'lower right')

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
                prob = prob*rhos[tuple(sorted(edge))]
            prob_dict[lpath].append(prob) 

    fix, ax = plt.subplots()
    max_len = max(prob_dict.keys())
    lens = sorted(prob_dict.keys())
    ax.plot(lens,[np.average(prob_dict[k]) for k in lens])
    ax.set_xlabel("Path length")
    ax.set_ylabel("Prob. of success")
    ax.set_title("Model")
    ax.grid()
    #ax.legend(loc = 'lower right')      
    plt.show()

if __name__ == '__main__':
    g, lambdas = create_graph()
    rhos = compute_rhos(g, lambdas)
    plot_graphs(g, rhos)