#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 07:48:14 2026

@author: leonardo
"""


import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from qber_2021 import get_weather_mapping

def plot_len_stats(flist):

    df = pd.read_csv(flist[0])    
    for fname in flist[1:]:
        df = pd.concat([df, pd.read_csv(fname)])
    # Extract the data: Group by 'area' and 'size', then calculate mean and standard deviation
    # This creates a new dataframe with columns: area, size, mean, std
    stats_df = [df.groupby(['area', 'size', 'max_link_len'])['path_len_m'].agg(['mean', 'std']).reset_index()]
    stats_df.append(df.groupby(['area', 'size', 'max_link_len'])['area_covered'].agg(['mean', 'std']).reset_index())
    stats_df.append(df.groupby(['area', 'size', 'max_link_len'])['degree'].agg(['mean', 'std']).reset_index())
    stats_df.append(df.groupby(['area', 'size', 'max_link_len'])['effective_degree'].agg(['mean', 'std']).reset_index())
    # stats_df.to_csv('extracted_network_stats.csv', index=False)
    
    # 3. Set up the Matplotlib figure
    fig, axes = plt.subplots(4, 3, figsize=(18, 6), sharex=True, sharey="row")
    scenarios = ['urban', 'suburban', 'rural']
    
    wmap = get_weather_mapping()
    subset_labels = ['L0_OPTIMAL', 'L4_MODERATE', 'L8_EXTREME']
    wmap = get_weather_mapping().items()
    subset_len = []
    for v, k in wmap:
        if k in subset_labels:
            subset_len.append(v) 
    # 4. Loop through and plot using plain Matplotlib
    for idx, label in enumerate(['Averge Path Length (m)', 'Covered Area (sq. m)', 'degree', 'effective_degree']):
        for i, area in enumerate(scenarios):
            ax = axes[idx][i]
            lines = []
            for link_len in subset_len:
                
                # Filter the pre-calculated stats for the current scenario
                subset = stats_df[idx][(stats_df[idx]['area'] == area) & (stats_df[idx]['max_link_len'] == link_len)]
                
                # Extract the X, Y, and Error arrays
                x_sizes = subset['size']
                y_means = subset['mean']
                y_errors = subset['std']
                
                # Plot using Matplotlib's errorbar function
                line = ax.errorbar(
                    x_sizes, 
                    y_means, 
                    yerr=y_errors, 
                    fmt='-o',          # Line ('-') with circle markers ('o')
                    capsize=5,         # Width of the caps on the error bars
                    elinewidth=1.5,    # Thickness of the error bars
                    capthick=1.5,      # Thickness of the caps
                    label = get_weather_mapping()[link_len]
                )
                if len(lines) < len(subset_len):
                    lines.append(line)
            # Formatting
            ax.set_title(f'Scenario: {area.capitalize()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Network Size (Number of Nodes)', fontsize=12)
            
            # Only label the Y-axis on the first plot
            if i == 0:
                ax.set_ylabel(label, fontsize=12)
                
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Force X-axis to only show integer ticks (since network size is discrete)
            ax.set_xticks(x_sizes.unique())
    fig.legend(lines, subset_labels, loc='lower center', ncol=len(subset_labels), 
           bbox_to_anchor=(0.5, 0.02), title="Weather Conditions")

    plt.tight_layout()
    plt.show()
    
    
def plot_probs():
    
    res = pd.DataFrame()
    for f in args.files:
        res = pd.concat([res, pd.read_csv(f)])    
    areas = res['area'].unique()
    
    
    fig1, all_g = plt.subplots(nrows=len(areas), ncols=10, figsize=(100, 15), sharex=True, sharey=True)
    fig2, match_g = plt.subplots(nrows=len(areas), ncols=10, figsize=(100, 15), sharex=True, sharey=True)
    fig1_lines = []
    fig1_labels = []
    size_list = []
    
    for i, area_name in enumerate(areas):
        subset = res[res['area'] == area_name]
        counter = 0
        for weather_type in subset['weather'].unique():
           ax1 = all_g[i][counter]
           ax2 = match_g[i][counter]
           ax1.set_title(weather_type)
           ax2.set_title(weather_type)
           if counter == 0:
               ax1.set_ylabel(area_name, rotation=90, size='large')
               ax2.set_ylabel(area_name, rotation=90, size='large')   
           counter += 1

           for max_link in sorted(subset['max_link_len'].unique()):
               data_plot = subset[(subset['weather'] == weather_type) & 
                                   (subset['max_link_len'] == max_link)].sort_values('size')
               line, = ax1.plot(data_plot['size'], data_plot['avg'], marker='o', label=weather_type)
                                  
               ax1.fill_between(
                    data_plot['size'], 
                    data_plot['avg'] - data_plot['CI'], 
                    data_plot['avg'] + data_plot['CI'], 
                    alpha=0.2)
               if len(fig1_lines) < len(get_weather_mapping()):
                   fig1_lines.append(line)
                   fig1_labels.append(get_weather_mapping()[max_link])
               if get_weather_mapping()[max_link] == weather_type:
 
                   data_plot = subset[(subset['weather'] == weather_type) & 
                                      (subset['max_link_len'] == max_link)].sort_values('size')
                   line, = ax2.plot(data_plot['size'], data_plot['avg'], marker='o', label=weather_type)
                    
                   ax2.fill_between(
                        data_plot['size'], 
                        data_plot['avg'] - data_plot['CI'], 
                        data_plot['avg'] + data_plot['CI'], 
                        alpha=0.2)
    fig1.legend(fig1_lines, fig1_labels, loc='lower center', ncol=len(fig1_labels), 
           bbox_to_anchor=(0.5, 0.02), title="Weather Conditions")
    
    
   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', help='a list of .csv files to parse', 
                        nargs='+')
    parser.add_argument('--stats', help='a dataframe with graph stats', 
                        nargs='+')
    args = parser.parse_args()

    if args.files:
        plot_probs()
    if args.stats:
        plot_len_stats(args.stats)
    #if args.graph_stats:
    #    plot_area_stats(args.graph_stats)
    plt.show()
            