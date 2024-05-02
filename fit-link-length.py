# use this to convert .adj files into edgelist with annotated distances
  
from distfit import distfit
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import argparse
import multiprocessing
import time
import os

rural_list = ['visibility_graphs/rural1/', 'visibility_graphs/rural2/', 
               'visibility_graphs/rural3/']

suburban_list = ['visibility_graphs/suburban1/',
               'visibility_graphs/suburban2/', 'visibility_graphs/suburban3/']

urban_list = ['visibility_graphs/urban1/', 'visibility_graphs/urban2/',
               'visibility_graphs/urban3/']

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=10000)
parser.add_argument('--graph_size', type=int, default=30)
parser.add_argument('--no_bootstrap', action='store_true', default=False)
parser.add_argument('--no_save', action='store_true', default=False)
parser.add_argument('--seed', type=int)
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

def distance(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def MDRW(adj_list_file, sample_size, ngraphs=1, use_cent=False, save_to=''):
    """ multi-dimensional random walk, with an optional variant to use 
        eigenvector centrality to tune the probabilities """
    core_nodes = 5
    g = nx.read_adjlist(adj_list_file, delimiter=',')
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
    graphs = []
    while len(graphs) < ngraphs:
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
    


def process_folder(folder, runs=5000, size=30, fname=''):
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
    count = 0
    for g in MDRW(adj_file, size, runs):
        for (frm, to , data) in g.edges(data=True):
            data['length'] = distance(coord_dict[int(frm)], 
                                      coord_dict[int(to)])
        edges.extend([data['length'] for _, _, data in g.edges(data=True)])
        nodes.extend(g.nodes())
        if not args.no_save:
            nx.write_graphml(g, save_folder + '/'  + fname + str(count) + \
                         '.graphml')
        count += 1
    print("Sub graph density:", len(edges)/len(nodes))
    return edges

def fit_length(edges, fname):
    print('Fitting the data for', fname)
    d = distfit()
    if no_bootstrap:
        # disable the bootstrap test
        d = distfit()  
    else:
        d = distfit(n_boots=100)
    pd.set_option('display.precision', 16) # print with more decimals
    res = d.fit_transform(np.array(edges), verbose = 0)
    d.plot()
    print(res['summary'])
    plt.savefig(args.results_folder + fname + '-' + str(size) + '.png')
    res['summary'].to_csv(args.results_folder + fname + '-' + str(size) \
                          + '.csv')
    func = res['model']['model'].pdf
    with open(args.results_folder + fname + "-data.gnuplot", 'w') as pfile:
        pfile.write('# length, histogram \n')
        y = res['histdata'][0]
        x = res['histdata'][1]
        for i in range(len(x)):
            pfile.write(f'{x[i]} {y[i]}\n')
        pfile.write('\n\n')
        # add some initial x values to make all the fitted curves start from 10
        x = list(range(10,int(x[0]),20)) + list(res['histdata'][1])
        pfile.write('# length, fitted \n')
        pfile.write(f'# {res["model"]["name"]} (params[] loc, scale) '
                    f'{res["model"]["params"]} '
                    f'mean={res["model"]["model"].mean()}\n')
        for i in range(len(x)):
            pfile.write(f'{x[i]} {func(x[i])}\n')
            
            
runs = args.runs
size = args.graph_size
no_bootstrap = args.no_bootstrap
fit_processes = []

for area in [rural_list]:#, suburban_list, urban_list]:
#for area in [rural_list, suburban_list, urban_list]:
    edges = []
    pool = multiprocessing.Pool(processes=3)
    arguments = []
    for folder in area:
        fname = ''.join(folder.split('/')[-2][:-1])
        arguments.append([folder, runs, size, fname])
    for e in pool.starmap(process_folder, arguments):
        edges.extend(e)
        print('Received', len(edges), 'edges')
    print('All workers provided edges')
    p = multiprocessing.Process(target=fit_length, args=(edges, fname,))
    p.start()
    fit_processes.append(p)
for p in fit_processes:
    p.join()
    

    
            
