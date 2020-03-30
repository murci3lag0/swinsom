#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:04:29 2020

@author: amaya
"""

import matplotlib.pyplot as plt

def set_figure(size, latex=False):
    plt.rcParams.update({'font.size': 11})
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    if latex:
        plt.rc('text', usetex=True)
        
    fig = plt.figure(figsize=(size))
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax
   
def fig_datacoverage(data, cols, fname=None, latex=False):    
    fig, ax = set_figure((4,3))
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of entries')
    ax = data[cols[0]].groupby(data.index.year).count().plot(kind="bar")
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', transparent=True)