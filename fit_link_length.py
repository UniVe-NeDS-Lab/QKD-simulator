#!/usr/bin/env python3

from distfit import distfit
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import argparse
import multiprocessing
import time
import os
import glob
from qber import FSOQKD
from qber_2021 import calculate_skr
import random

rural_list = ['visibility_graphs/rural1/', 'visibility_graphs/rural2/', 
               'visibility_graphs/rural3/']

suburban_list = ['visibility_graphs/suburban1/',
               'visibility_graphs/suburban2/', 'visibility_graphs/suburban3/']

urban_list = ['visibility_graphs/urban1/', 'visibility_graphs/urban2/',
               'visibility_graphs/urban3/']

def distance(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def filter_edges(adj_list_file, max_len, coord_dict, sample_size):
    """ return a list of graphs that are connected components that 
    satisfy a maximum link lenght and a minimum number of nodes """
    g = nx.read_adjlist(adj_list_file, delimiter=',')
    if not max_len:
        return [g]
    
    def filter_edge(frm, to):
        return distance(coord_dict[int(frm)], 
                        coord_dict[int(to)]) < max_len
    
    # remove edges shorter than the threshold
    if max_len:
        g = nx.subgraph_view(g, filter_edge=filter_edge)
    # filter out too small components
    large_components = [g.subgraph(c) for c in nx.connected_components(g)
                        if len(c) >= sample_size]
    return large_components
   

def MDRW(components, sample_size, coord_dict, ngraphs=1, 
         use_cent=False, save_to=''):
    """ multi-dimensional random walk, with an optional variant to use 
        eigenvector centrality to tune the probabilities """
    core_nodes = 5
   
    graphs = []
    while len(graphs) < ngraphs:      
        # if the graph is split in components (this happens when we filter 
        # the edges for their length) pick one at random
        g = random.sample(components, 1)[0]
        tot_cent = 0
        c_dict = {}
    
        if use_cent:
            print('Computing  centrality')
            for (nlist,cent) in nx.eigenvector_centrality(g).items():
                for n in nlist.split(','):
                    c_dict[n] = cent
                    tot_cent += cent
            print('Done with centrality')
        print('Starting to sample')

        nodes = set()
        if use_cent:
            core = list(np.random.choice([x for x in c_dict], 
                        p=[c/tot_cent for c in c_dict.values()], 
                        size=core_nodes))
        else:
            core = list(np.random.choice(g.nodes(), size=core_nodes))
        iterations = 1000
        # if we are unlucky in the initial choice of core nodes, we may get 
        # stuck in a disconnected partition and loop forever. 
        # If we didn't reach the desired size in 1000 iterations stop and 
        # try again 
        while len(nodes) != sample_size:
            if use_cent:
                tot_sum = sum([c_dict[x] for x in core])
                start = np.random.choice([x for x in core], 
                                         p=[c_dict[x]/tot_sum for x in core])
                tot_sum = sum([c_dict[x] for x in g[start]])
                newnode = np.random.choice([x for x in g[start]], 
                                    p=[c_dict[x]/tot_sum for x in g[start]])
            else:
                tot_sum = sum([g.degree(x) for x in core])
                start = np.random.choice([x for x in core], 
                                         p=[g.degree(x)/tot_sum for x in core])
                newnode = np.random.choice([x for x in g[start]])
            nodes.add(start)
            nodes.add(newnode)
            core.append(newnode)
            core.remove(start)
            iterations -= 1
            if not iterations:
                break
        if not iterations:
            # got suck in a partition, try again
            continue
        subg = g.subgraph(nodes)
        # we want to consider only graphs that are connected
        if nx.is_connected(subg):
            graphs.append(subg)
    
    print('Done with graph sampling')
    return graphs


def gen_graphs(folder, runs=5000, size=30, fname='', gnumber=0):
    adj_file = folder + "/intervisibility.adj" 
    position_file  = folder + "/best_p.csv"
    print('Opening folder:', folder)
    coord_dict = {}
    edges = []
    nodes = []
    with open(position_file) as coords:
        for line in coords:
            if line[0] == '#':
                continue
            oid , x, y = [int(i) for i in line.split(',')]  
            if oid in coord_dict:
                print("Error! double building id!", oid)
            else:
                coord_dict[oid] = (x,y)
    save_folder = args.results_folder + './graphs/'
    try:  
        os.mkdir(save_folder)
    except OSError:
        pass
    #FIXME
    count = 0
    components = filter_edges(adj_file, args.max_len, coord_dict, size)
    for g in MDRW(components, size, coord_dict, runs):
        g.graph.update({'scenario':folder})
        for frm, to, data in g.edges(data=True):
            data['length'] =  distance(coord_dict[int(frm)], 
                                      coord_dict[int(to)])
            #data['SKR'] = fso.get_rate(data['length'], 1)[3]
            data['SKR'] = calculate_skr(0, data['length']/1000) # convert to km
            edges.append(data['length'])
        nodes.extend(g.nodes())
        if not args.no_save:
            nx.write_graphml(g, save_folder + f'/{fname}-ZONE{gnumber}-SIZE{size}-G{str(count)}.graphml')
        count += 1
    print("Sub graph density:", len(edges)/len(nodes))

    return edges

def fit(values, fname, target='length', comments=''):
    print(f'Fitting {target} on data for', fname)
    d = distfit()
    if no_bootstrap:
        # disable the bootstrap test
        d = distfit()  
    else:
        d = distfit(n_boots=1000)
    pd.set_option('display.precision', 16) # print with more decimals
    res = d.fit_transform(np.array(values), verbose = 0)
    d.plot()
    print(f'-------------{fname}-{target}-------------\n{res['summary']}')
    fprefix = f'{args.results_folder}/{target}-{fname}-{size}-{comments}'
    plt.savefig(fprefix +'.png')
    res['summary'].to_csv(fprefix + '.csv')
    func = res['model']['model'].pdf
    with open(fprefix + "-data.gnuplot", 'w') as pfile:
        pfile.write(f'# {target}, histogram \n')
        y = res['histdata'][0]
        x = res['histdata'][1]
        for i in range(len(x)):
            pfile.write(f'{x[i]} {y[i]}\n')
        pfile.write('\n\n')
        # add some initial x values to make all the fitted curves start from 10
        x = list(range(10,int(x[0]),20)) + list(res['histdata'][1])
        pfile.write(f'# {target}, fitted\n')
        if comments:
            pfile.write(f'# {comments}')
        pfile.write(f'# {res["model"]["name"]} (params[] loc, scale) '
                    f'{res["model"]["params"]} '
                    f'mean={res["model"]["model"].mean()}\n')
        for i in range(len(x)):
            pfile.write(f'{x[i]} {func(x[i])} \n')
        
        pfile.write('\n\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=10000)
    parser.add_argument('--processes', help='number of parallel processes', type=int, default=1)
    parser.add_argument('--gen_only', action='store_true', default=False)
    parser.add_argument('--fit_only', action='store_true', default=False)
    parser.add_argument('--gen_rate', type=int, default=1_000_000_000)
    parser.add_argument('--graph_size', type=int, default=30)
    parser.add_argument('--no_bootstrap', action='store_true', default=False)
    parser.add_argument('--no_save', action='store_true', default=False)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max_len', type=int, help='Maximum link length (m)', 
                        default = 0)
    parser.add_argument('--results_folder', default='./results/')


    args = parser.parse_args()


    try:
        os.mkdir(args.results_folder)
    except OSError:
        pass

    if args.seed is not None:
        np.random.seed(args.seed)
        np.random.default_rng(args.seed)
        print(f'Set random seed to {args.seed}')


                
    runs = args.runs
    size = args.graph_size
    no_bootstrap = args.no_bootstrap
    fit_processes = []
    fso = FSOQKD()


    #for area in [rural_list]:#, suburban_list, urban_list]:

    if args.fit_only: # do not regenerate the graphs, use the existing ones
        for area in ['rural', 'suburban', 'urban']:
            edges = []
            nodes = 0
            graphs_folder = args.results_folder + 'graphs'
            file_list = glob.glob(f'{graphs_folder}/{area}*.graphml')[:args.runs]
            if not file_list:
                print(f'I could not find files for the {area} areas in {graphs_folder}. Please generate them')
                exit()
            print(f'Using {len(file_list)} existing graphs for {area} areas ', end='')
            # Open and read each file
            for file_name in file_list:
                with open(file_name, 'r') as f:
                    g = nx.read_graphml(f)
                    nodes += len(g)
                    edges.extend([e['length'] for _,_,e in g.edges(data=True)])
            print(f'containing {len(edges)} edges and {nodes} nodes')
            comments = str(args.runs)
            p = multiprocessing.Process(target=fit, args=(edges, area, 'length', comments))
            p.start()
            fit_processes.append(p)
            # get the rate in Mb/s
            rates = [fso.get_rate(x, args.gen_rate)[3]/1_000_000 for x in edges if x>0]
            p = multiprocessing.Process(target=fit, args=(rates, area, 'rate', comments+'-'+str(args.gen_rate)))
            p.start()
            fit_processes.append(p)

    else:
        for area in [rural_list, suburban_list, urban_list]:
            edges = []
            pool = multiprocessing.Pool(processes=args.processes)
            arguments = []
            gnumber = 0
            for folder in area:
                fname = ''.join(folder.split('/')[-2][:-1])
                arguments.append([folder, int(runs/len(area)), size, fname, gnumber])
                gnumber += 1
            for e in pool.starmap(gen_graphs, arguments):
                edges.extend(e)
                print('Received', len(edges), 'edges')
            print('All workers provided edges')
            if args.gen_only:
                continue
            comments = str(args.runs)
            p = multiprocessing.Process(target=fit, args=(edges, fname, 'length', comments))
            p.start()
            fit_processes.append(p)
            # get the rate in Mb/s
            rates = [fso.get_rate(x, args.gen_rate)[3]/1_000_000 for x in edges if x>0]
            p = multiprocessing.Process(target=fit, args=(rates, fname, 'rate', comments + '-' + str(args.gen_rate)))
            p.start()
            fit_processes.append(p)

    for p in fit_processes:
        p.join()
    

    
            
