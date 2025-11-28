#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:58:50 2023

@author: leonardo
"""
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


class LinkProperties():
    def __init__(self, scenario='urban'):
        self.distr = getattr(stats, self.params[scenario][0])(*self.params[scenario][1])

    def get_values(self, n=1):
        return self.distr.rvs(size=n)
    
    def plot_pdf(self):
        fit_x = np.linspace(10,16000, 1000)
        fit_y = [self.distr.pdf(x) for x in fit_x]
        p, fig = plt.subplots()
        fig.plot(fit_x,fit_y)
        plt.show()

class LinkLength(LinkProperties):
    params = {'rural': ('gamma', (np.float64(0.8789882695507861), np.float64(0.6113185470495653), np.float64(1779.981731504433))),
              'suburban' :('beta', (np.float64(0.9926921384298317), np.float64(131.16374649754158), np.float64(0.9999999999999999), np.float64(247575.76974684666))),
              'urban': ('lognorm', (np.float64(0.4228235530029076), -2503.344405574453, np.float64(6201.235802227958)))}

    def plot_pdf(self):
        fit_x = np.linspace(10,16000, 1000)
        fit_y = [self.distr.pdf(x) for x in fit_x]
        p, fig = plt.subplots()
        fig.plot(fit_x,fit_y)
        fig.set_xlabel('m')
        fig.set_title('Link length PDF')
        plt.show()

class LinkRate(LinkProperties):
    params = {'rural': ('genextreme', (np.float64(-4.93940734617113), np.float64(0.25537813764694905), np.float64(1.2615103544161017))),
              'suburban' :('genextreme', (np.float64(-4.755475693810034), np.float64(0.10774273011304142), np.float64(0.5124206359573793))),
              'urban': ('genextreme', (np.float64(-3.699590179231103), np.float64(0.0009470518652798229), np.float64(0.0035030122044944914)))}
    
    def plot_pdf(self):
        fit_x = np.linspace(10,16000, 1000)
        fit_y = [self.distr.pdf(x) for x in fit_x]
        p, fig = plt.subplots()
        fig.set_xscale('log')
        fig.plot(fit_x,fit_y)
        fig.set_xlabel('Rate Mb/s')
        fig.set_title('Link rate PDF')
        plt.show()


if __name__ == "__main__":
    g = LinkRate(scenario='urban')
    g.plot_pdf()
