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
from tqdm.contrib import itertools as tqiter
import os
from qber_2021 import weather_profile

import matplotlib.pyplot as plt
from matplotlib import rc
import random
import tqdm
import argparse 
from pathlib import Path
import multiprocessing
import time
import cProfile

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
        paths = nx.single_source_shortest_path(g, start)
        for stop in g.nodes():
            if start == stop:
                continue
            path = paths[stop] #nx.shortest_path(g, start, stop)
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

def compute_rhos(g, lambdas, verbose=False, directed=False):
    rhos = dict.fromkeys([sedge(e) for e in g.edges()], 0.1)
    oldrhos = dict.fromkeys(rhos.keys(), 0)
    #relaxation coefficient
    alpha = 0.1
    #profiler = cProfile.Profile()
    #profiler.enable()
    skr_string = f'SKR-{args.weather}'
    # this is approximating rho, eq. 2
    last_alpha = 1
    if verbose:
        print('Starting graph analysis. Alpha progress: ', end='', flush=True)
    while (LA.norm([oldrhos[k]-rhos[k] for k in rhos],2)>1/args.relax_thr):
        curr_alpha = LA.norm([oldrhos[k]-rhos[k] for k in rhos],2)
        if curr_alpha < last_alpha/10 and verbose:
            last_alpha = curr_alpha
            print(f'{curr_alpha:.6f}', end=', ', flush=True)

        oldrhos = rhos.copy()
        for s,d in g.edges():
            # see the comment on sorted edges in psi()
                rhos[sedge(s,d)]  = alpha * min(1.0, g.edges[s,d][skr_string]/psi(s, d, oldrhos, lambdas, g)) + (1-alpha)*oldrhos[sedge(s,d)]
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
    #profiler.disable()
    # Save stats with the Process ID (PID) in the filename
    stats_file = f"profile_pid_{os.getpid()}.stats"
    #profiler.dump_stats(stats_file)
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

def compute_combined_estimate(means, variances, confidence=0.95):
    """
    Takes a list of means, and variances measured on different graphs. 
    Computes the grand mean and confidence interval using Inverse-Variance 
    Weighting:https://en.wikipedia.org/wiki/Inverse-variance_weighting
    That is, compute the comulative mean and variance, then compute the CI.
    
    Default z_score is 1.96 (for a 95% Confidence Interval).
    Use 1.645 for 90% CI, or 2.576 for 99% CI.
    
    Actually, I am not using this anymore. Since every average is weighted
    by its variance, the runs with high variance count less in the weighted
    average. This is OK if you imagine these are runs with less confidence
    but in this case they are just runs with reasonably varying results. 
    """
    # 1. Calculate the weight for each run, adds a minimum variance
    # because there could be rare full-meshes, in which the probs 
    # are all the same, so zero variance
    epsilon = 1e-10  # A very small number to avoid zero division
    weights = 1.0 / (np.array(variances) + epsilon)
    # 2. Compute the weighted Grand Mean
    grand_mean = np.sum(weights * np.array(means)) / np.sum(weights)
    
    # 3. Compute the combined variance and Standard Error
    combined_variance = 1.0 / np.sum(weights) # this decreases with len(means)
    standard_error = np.sqrt(combined_variance)
    
    h = standard_error * stats.t.ppf((1 + confidence) / 2, len(means)-1)
    
    return grand_mean, h 

def max_demand_per_link(avg_SKR, key_size, rekey_interval, rate):
    """ This function computes the allowed demand per link in average,
    Returns Mb/s"""
    keys_per_sec = avg_SKR/key_size
    bps_per_link = rate*keys_per_sec*rekey_interval*10**9*8 
    return bps_per_link/1_000_000 # returns Mb/s

def parse_all_graphs(files, lbd):
    areas = set()
    sizes = set()
    results = []
    graphs = defaultdict(list)
    tot_nodes = 0
    for fpath in files:
        f = Path(fpath).name
        area = f.split('-')[0]
        size = f.split('-')[2][4:] 
        areas.add(area)
        sizes.add(int(size))
        graphs[area+size].append(fpath)
        tot_nodes+=int(size)

    # add TQDM
    args_list = []
    res_list = []
    f_args = []
    res = []
    skr_weather = f'SKR-{args.weather}'
    with tqdm.tqdm(total=tot_nodes, desc="Processed graphs", unit="graphs") as pbar:
        """ prepare the list of results that will be put in a DF """
        for area in areas:
            for size in sorted(sizes):
                pbar.set_description(f"Area: {area}, Size: {size}")
                probs = []
                glist = graphs[area+str(size)]
                for g in glist:
                    g = nx.read_graphml(g)
                    l = set_lambda(g, lbd)
                    f_args.append((g,l))
                    mean_SKR = np.mean([e[2][skr_weather] for e in g.edges(data=True)])
                    res_list.append([area, size, lbd, 0, 0,  
                                    len(glist), mean_SKR, 
                                    g.graph['max_link_len']])
                    if args.processes == 1:
                        res.append(compute_rhos(g,l))
                        pbar.update(len(g))
    if args.processes > 1:
        with multiprocessing.Pool(processes=args.processes) as pool:
            res = pool.starmap_async(compute_rhos, f_args)
        
            with tqdm.tqdm(total=tot_nodes, desc="Processed graphs", 
                           unit="graphs") as pbar:
                while not res.ready():
                    remaining = res._number_left # hack
                    pbar.n = len(f_args) - remaining
                    pbar.refresh()
                    area = res_list[pbar.n][0]
                    size = res_list[pbar.n][1]
                    pbar.set_description(f"Area: {area}, Size: {size}")
                    time.sleep(1)
            res = res.get()
    temp_df = pd.DataFrame(columns=df_columns) 
    
    for i in range(len(res_list)):
        _, prob_dict = res[i]
        line = res_list[i]
        new_res = []
        
        probs = [p for plist in prob_dict.values() for p in plist]
        #print(prob_dict)
        for p in probs:
            new_res.append(res_list[i].copy())
            new_res[-1][3] = p
        run_df = pd.DataFrame(new_res, columns=temp_df.columns)
        temp_df = pd.concat([temp_df, run_df], ignore_index=True)

    
    res_df = pd.DataFrame(columns=df_columns)
    group_by_fields = ['area', 'size', 'lambda', 'max_link_len']
    grouped = temp_df.groupby(group_by_fields)
    print(temp_df.to_string())
    for (condition, group) in grouped:
        new_row_dict = dict(zip(group_by_fields, condition))
        m, ci = mean_confidence_interval(group['avg'])
        new_row_dict['avg'] = m
        new_row_dict['CI'] = ci
        new_row_dict['graphs'] = len(group)
        new_row_dict['avg_SKR'] = np.mean(group['avg_SKR'])        
    
        res_df = pd.concat([res_df, pd.DataFrame([new_row_dict])])
    return res_df
    
if __name__ == '__main__':
    """ graph g is expected to have a link attribute 'SKR' that contains 
    the secrete key rate. While lambda[(n,m)] is the traffic intensity 
    from node n do m. Units are arbitrary, they must be consistent """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', help='a list of .graphml files to parse', 
                        nargs='+', required=True)
    #parser.add_argument('--lbd', help='Traffic intensity (lambda), in Mb/s', 
    #                    type=float, default=1, nargs='+')
    parser.add_argument('--gen_rate', type=int, default=1_000_000_000)
    parser.add_argument('--traffic_demand', type=int, default=[100],
                        help="The actual traffic demand between every couple "
                        "of nodes (Mb/s). List is supported", nargs='+')
    parser.add_argument('--rekey_interval', type=float, default=100,
                        help="The interval between rekeys (GB).")
    parser.add_argument('--key_size', type=int, default=256)
    parser.add_argument('--processes', help='number of parallel processes', 
                        type=int, default=1)
    parser.add_argument('--save_to', help='dump the dataframe to a file')
    parser.add_argument('--relax_thr', help=' inverse of the threshold used'
                        ' for the equation relaxation', default=10**6, 
                        type=int)
    parser.add_argument('--weather', choices=weather_profile.keys(), 
                        default='AVG')
    
    args = parser.parse_args()
    
    df_columns = ['area', 'size', 'lambda', 'avg', 'CI', 
                  'graphs', 'avg_SKR', 'max_link_len']
    res_d = pd.DataFrame()
    
    if args.save_to:
        # Remove the file if it exists to start fresh
        if os.path.exists(args.save_to):
            os.remove(args.save_to)
            print(f"Existing {args.save_to} removed. Starting fresh.")
    for dem in args.traffic_demand:
        # every rekey_interval Bytes, we need a key of size key_size, so for 
        # every QKD bit sent, we can carry key_size/rekey_interval*8 
        # thus to sustain a certain traffic_demand we need a QKD load of:
        lbd = dem*1_000_000*args.key_size/\
          (args.rekey_interval*1_000_000_000*8) # lambda in bit/s
           
        lbd = lbd/(args.gen_rate)   # the SKR in the graphs is 
                                    # b/s/pulse, so we normalize
                                    # by the gen_rate 
        d = parse_all_graphs(args.files, lbd)
        d['gen_rate (Gb/s)'] = args.gen_rate/1_000_000_000
        d['traffic_demand (Mb/s)'] = dem
        d['rekey_interval (GB)'] = args.rekey_interval
        d['relax_thr'] = args.relax_thr
        d['key_size'] = args.key_size
        d['max_demand_per_link (Mb/s)'] = d.apply(lambda r:\
                                        max_demand_per_link(r['avg_SKR'], 
                                            r['key_size'], 
                                            r['rekey_interval (GB)'], 
                                            args.gen_rate), axis=1)
        d['weather'] = args.weather
        res_d = pd.concat([res_d, d], ignore_index=True)
            
        if args.save_to:
            write_header = not os.path.exists(args.save_to)
    
            res_d.to_csv(args.save_to, 
                      mode='a',             # append
                      index=False, 
                      header=write_header,  # Only True for the first write
                      encoding='utf-8')
    print(d)
           