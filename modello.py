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
from scipy import stats
from collections import defaultdict
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rc
import random
import tqdm
import argparse 
from pathlib import Path

nodes = 10

def sedge(s,d=None):
    """ just return a sorted tuple to be used as a dictionary index.
        accepts an edge or a pair of nodes """
    if d:
        return tuple(sorted([s,d]))
    return tuple(sorted(s))

def create_graph(nodes=10):
    """ for testing purposes """
    g = nx.path_graph(nodes) # a path graph
    for _,_,d in g.edges(data=True):
        d['SKR'] = 1
    lambdas = set_lambda(g)
    return g, lambdas


def set_lambda(g, lbd=1):
    lambdas = {}
    for n in g:
        for m in g:
            if n == m:
                continue
            lambdas[(n,m)] = lbd
    return lambdas

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

def compute_rhos(g, lambdas, verbose=True, directed=False):
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
    couple_set = set()
    prob_dict = defaultdict(list)
    for source in all_paths:
        for dest in all_paths[source]:
            tup = (min(source, dest), max(source, dest))
            if source == dest:
                continue
            if not directed and tup in couple_set:
                continue
            couple_set.add(tup)
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
    
    
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a) # Mean and Standard Error
    
    # Calculate the CI
    h = se * stats.t.ppf((1 + confidence) / 2, n-1)
    
    return m, h

def parse_all_graphs(files, lbd, d):
    areas = set()
    sizes = set()
    graphs = defaultdict(list)
    for fpath in files:
        f = Path(fpath).name
        area = f.split('-')[0]
        size = f.split('-')[2][4:]
        areas.add(area)
        sizes.add(int(size))
        graphs[area+size].append(fpath)
    for area in areas:
        for size in sorted(sizes):
            probs = []
            glist = graphs[area+str(size)]
            for g in glist:
                g = nx.read_graphml(g)
                l = set_lambda(g, lbd)
                _, prob_dict = compute_rhos(g, l)
                probs.extend([p for plist in prob_dict.values() for p in plist])
            m, h = mean_confidence_interval(probs) 
        d.loc[len(d)] = [area, size, lbd, m, h, len(probs), len(glist)]
    print(d)
    
if __name__ == '__main__':
    """ graph g is expected to have a link attribute 'SKR' that contains 
    the secrete key rate. While lambda[(n,m)] is the traffic intensity 
    from node n do m. Units are arbitrary, they must be consistent """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', help='a list of .graphml files to parse', 
                        nargs='+', required=True)
    parser.add_argument('--lbd', help='Traffic intensity (lambda), in Mb/s', 
                        type=float, default=1, nargs='+')
    args = parser.parse_args()
    d = pd.DataFrame(columns=['area', 'size', 'lambda', 'avg', 'CI', 'paths', 
                              'graphs'])   
    for lbd in args.lbd:
            parse_all_graphs(args.files, lbd, d)
    exit()
    for fname in args.files:
        g = nx.read_graphml(fname)
        rhos, prob_dict = compute_rhos(g, set_lambda(g, args.lbd))
        plot_graphs(g, rhos, prob_dict)    
    #g, lambdas = create_graph()
    