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

nodes = 10

rhos = np.array([1.0]*(nodes-1))

gammas = [1.0]*(nodes-1)


def create_graph(nodes):
    graph=nx.path_graph(nodes)
    return graph

pij = np.ones((nodes,nodes))*1/(math.pow(nodes,2)-nodes)
for i in range(0, nodes):
    pij[i][i] = 0.0

#pij = np.zeros((nodes,nodes))
#for i in range(0,nodes-1):
#    pij[i][nodes-1] = 1.0
    
    
ltot = 4.0

def psi(i, rhos, lamb):
    t = 0.0
    for k in range(0, i+1):
        for j in range(i+1, nodes):
            fact  = 1.0
            for f in range(k,i):
                fact = fact * rhos[f]
            t = t + lamb[k][j]*fact
                
    for k in range(i+1, nodes):
        for j in range(0, i+1):
            fact  = 1.0
            for f in range(i+1,k):
                fact = fact * rhos[f]
            t = t + lamb[k][j]*fact
    return t
    


g=create_graph(nodes)

oldrho = np.array([0]*(nodes-1))
lambij = ltot * pij
alpha = 0.1

while (LA.norm(oldrho-rhos,2)>0.000001):
    print(LA.norm(oldrho-rhos,2))
    oldrho = np.copy(rhos)
    for i in range(0,nodes-1):
        rhos[i] = alpha * min(1.0, gammas[i]/psi(i, oldrho, lambij)) + (1-alpha)*oldrho[i]

fix, ax = plt.subplots()
ax.plot(range(0,len(rhos)),rhos)
ax.set_xlabel("Link")
ax.set_ylabel("Prob. of filled")
plt.yscale('log')
ax.set_title("Model")
ax.grid()
ax.legend(loc = 'lower right')

r = []
l = dict(nx.all_pairs_shortest_path(g))
for k in l:
    for j in l[k]:
        
        print(k,"-",j,':',l[k][j])
        if len(l[k][j])>1:
            if l[k][j][0]<l[k][j][1]:
                r.append(l[k][j][0:-1]) 
            
psucc=[1.0]*len(r)            
for e in range(0,len(r)):
    for k in r[e]:
        psucc[e] = psucc[e]*rhos[k]
    
dist = defaultdict(list)

for i in range(len(psucc)):
    dist[len(r[i])].append(psucc[i])

avg_p = []
for k in dist:
    avg_p.append(sum(dist[k])/len(dist[k]))    
  

fix, ax = plt.subplots()
ax.plot(range(1,len(rhos)+1),avg_p)
ax.set_xlabel("Path length")
ax.set_ylabel("Prob. of success")
ax.set_title("Model")
ax.grid()
ax.legend(loc = 'lower right')      