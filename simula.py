#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 12:05:03 2023

@author: andreamarin
"""

import numpy as np
import math
from numpy import linalg as LA

import matplotlib.pyplot as plt
from matplotlib import rc
import random

nodes = 30

rhos = np.array([1.0]*(nodes-1))

gammas = [1.0]*(nodes-1)

#pij = np.ones((nodes,nodes))*1/(math.pow(nodes,2)-nodes)
#for i in range(0, nodes):
#    pij[i][i] = 0.0
    
pij = np.zeros((nodes,nodes))
for i in range(0,nodes-1):
    pij[i][nodes-1] = 1.0
    

ltot = nodes*0.005

def sampleexp(rate):
    if rate == 0.0:
        return math.inf
    else:
        return -math.log(random.uniform(0,1))/rate
    

def selectroute():
    x = random.uniform(0, 1)
    acc = 0.0
    for r in range(0,len(pij)):
        for c in range(0,len(pij)):
            acc = acc + pij[r][c]
            if x < acc:
                return r,c
    return len(pij)-1,len(pij)-1
    

    
    
fel = []    
statelinks = [0]*(nodes-1)
for i in range(0, nodes-1):
    fel.append(sampleexp(gammas[i]))
    
fel.append(sampleexp(ltot))

#fel: posizioni 0..nodes-2 generazione delle chiavi, possizione n-1 prossimo arrivo


time = 0.0
maxbuf = 10000000
simlen = 1000000

failures = 0
successes = 0
transmissions = 0
meansuccess = 0

rhos= [0.0]*(nodes-1)

for events in range(0, simlen):
    nextev = min(fel)
    posnextev = fel.index(nextev)
    
    for i in range(0, nodes-1):
        if statelinks[i]>0:
            rhos[i] = rhos[i] + nextev
            
    time = time + nextev
    
    if posnextev == nodes-1: #arrival
        transmissions = transmissions + 1
        route = selectroute()
        entire = True
        if route[1]>route[0]:
            for i in range(route[0], route[1]):
                if statelinks[i] > 0:
                    statelinks[i] = statelinks[i]-1
                else:
                    entire = False
                    break
        else:
            for i in range(route[0]-1, route[1]-1, -1):
                if statelinks[i] >0 :
                    statelinks[i] = statelinks[i]-1
                else:
                    entire = False
                    break
        if entire:
            successes = successes + 1
            meansuccess = meansuccess + abs(route[1]-route[0])
        else:
            failures = failures + 1
                
        
    else:
        if statelinks[posnextev] < maxbuf:
            statelinks[posnextev] = statelinks[posnextev] + 1
    fel = []    
    for i in range(0, nodes-1):
        fel.append(sampleexp(gammas[i]))
        
    fel.append(sampleexp(ltot))
        

rhos= [r/time for r in rhos]                
meansuccess = meansuccess / successes

print("Mean length of successful transmission:" + str(meansuccess))
print("Throughput: " + str(successes/time))
print("Blocking rate :" + str(failures/time))

    
fix, ax = plt.subplots()
ax.plot(range(0,len(rhos)),rhos)
ax.set_xlabel("Link")
ax.set_ylabel("Prob. of filled")
ax.set_title("Simulation")
ax.grid()
ax.legend(loc = 'lower right')


