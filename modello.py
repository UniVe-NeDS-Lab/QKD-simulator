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

#rhos = np.array([1.0]*(nodes-1))


# this assumes 
gammas = [1.0]*(nodes-1)


def create_graph(nodes):
    lambdas = {}
    #g = nx.connected_watts_strogatz_graph(nodes, 2 , 0.1) # just a random graph
    g = nx.path_graph(nodes) # a path graph
    for _,_,d in g.edges(data=True):
        #d['gamma'] = random.random()
        d['gamma'] = 1
    print(g)
    for n in g:
        for m in g:
            if n == m:
                continue
            #lambdas[(n,m)] = random.random()
            lambdas[(n,m)] = 4/(math.pow(nodes,2)-nodes)
    return g, lambdas

g, lambdas = create_graph(nodes)
rhos = dict.fromkeys(g.edges(), 0.1)
print(lambdas)
print(gammas)
# pij is the fraction of traffic that goes from node i to not j. All nodes
# are chosen with equal probaiblity.
# since each node can chose among (n-1) possible destination with equal
# probability, then the total traffic lambda is divided by n(n-1) for each
# route
 
# pij = np.ones((nodes,nodes))*1/(math.pow(nodes,2)-nodes)
# for i in range(0, nodes):
#     pij[i][i] = 0.0


# ltot = 4.0

# # lambij is the rate at which a communication from node i to node j takes place  
# lambij = ltot * pij

# what is psi??

def psi_old(i, rhos, lamb):
    t = 0.0
    for k in range(0, i+1): # for each source node
        for j in range(i+1, nodes): # to each destination node
            fact  = 1.0
            for f in range(k,i):
                fact = fact * rhos[f]
            t = t + lamb[k][j]*fact
                
    for k in range(i+1, nodes): # same thing, on the other direction
        for j in range(0, i+1):
            fact  = 1.0
            for f in range(i+1,k):
                fact = fact * rhos[f]
            t = t + lamb[k][j]*fact
    return t
    
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

#g=create_graph(nodes)

#oldrho = np.array([0]*(len(g.edges())))
oldrhos = dict.fromkeys(rhos.keys(), 0)



#relaxation coefficient
alpha = 0.1

# this is approximating rho, so eq. 2?
while (LA.norm([oldrhos[k]-rhos[k] for k in rhos],2)>0.000001):
    print(LA.norm([oldrhos[k]-rhos[k] for k in rhos],2))
    oldrhos = rhos.copy()
    for s,d in g.edges():
        # see the comment on sorted edges in psi()
        rhos[tuple(sorted([s,d]))]  = alpha * min(1.0, g.edges[s,d]['gamma']/psi(s, d, oldrhos, lambdas, g)) + (1-alpha)*oldrhos[(s,d)]
    #for i in range(0,nodes-1):
    #    rhos[i] = alpha * min(1.0, gammas[i]/psi(i, oldrho, lambij)) + (1-alpha)*oldrho[i]


fix, ax = plt.subplots()
ax.bar([str(x) for x in rhos.keys()], rhos.values())
ax.set_xlabel("Link")
ax.set_ylabel("Prob. of filled") # prob that link i is non empty?
plt.yscale('log')
ax.set_title("Model")
ax.grid()
ax.legend(loc = 'lower right')

r = []
all_paths = dict(nx.all_pairs_shortest_path(g))
prob_dict = defaultdict(list)
for source in all_paths:
    for dest in all_paths[source]:
        path = all_paths[source][dest]
        lpath = len(path)
        if lpath == 10:
            print(path)
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
ax.legend(loc = 'lower right')      
plt.show()