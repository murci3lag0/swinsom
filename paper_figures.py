#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:04:29 2020

@author: amaya
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
        
def fig_dimreduc(data, x1, x2, cmap='Set1', fname=None, latex=False):
    cmap = plt.cm.get_cmap(cmap, 5)
    fig, ax = plt.subplots(2,6,figsize=(16,6),sharex='row', sharey='row')
    alpha = 0.6
    size = 0.1
    
    ax[0][0].scatter(x1[:,0], x1[:,1], c=data['Xu_SW_type'], s=size, alpha=alpha, vmin=0, vmax=5, cmap=cmap)
    ax[0][1].scatter(x1[:,2], x1[:,1], c=data['Xu_SW_type'], s=size, alpha=alpha, vmin=0, vmax=5, cmap=cmap)
    ax[0][2].scatter(x1[:,0], x1[:,1], c=data['Zhao_SW_type'], s=size, alpha=alpha, vmin=0, vmax=5, cmap=cmap)
    ax[0][3].scatter(x1[:,2], x1[:,1], c=data['Zhao_SW_type'], s=size, alpha=alpha, vmin=0, vmax=5, cmap=cmap)
    ax[0][4].hist2d (x1[:,0], x1[:,1], bins=50, cmap='magma_r', norm=mcolors.PowerNorm(0.75))
    ax[0][5].hist2d (x1[:,2], x1[:,1], bins=50, cmap='magma_r', norm=mcolors.PowerNorm(0.75))
    ax[1][0].scatter(x2[:,0], x2[:,1], c=data['Xu_SW_type'], s=size, alpha=alpha, vmin=0, vmax=5, cmap=cmap)
    ax[1][1].scatter(x2[:,2], x2[:,1], c=data['Xu_SW_type'], s=size, alpha=alpha, vmin=0, vmax=5, cmap=cmap)
    ax[1][2].scatter(x2[:,0], x2[:,1], c=data['Zhao_SW_type'], s=size, alpha=alpha, vmin=0, vmax=5, cmap=cmap)
    sct = ax[1][3].scatter(x2[:,2], x2[:,1], c=data['Zhao_SW_type'], s=size, alpha=alpha, vmin=0, vmax=5, cmap=cmap)
    ax[1][4].hist2d (x2[:,0], x2[:,1], bins=50, cmap='magma_r', norm=mcolors.PowerNorm(0.75))
    ax[1][5].hist2d (x2[:,2], x2[:,1], bins=50, cmap='magma_r', norm=mcolors.PowerNorm(0.75))

    ax[0][0].text(0.05, 0.9, 'a)', fontsize=11, transform=ax[0][0].transAxes)
    ax[0][1].text(0.05, 0.9, 'b)', fontsize=11, transform=ax[0][1].transAxes)
    ax[0][2].text(0.05, 0.9, 'c)', fontsize=11, transform=ax[0][2].transAxes)
    ax[0][3].text(0.05, 0.9, 'd)', fontsize=11, transform=ax[0][3].transAxes)
    ax[0][4].text(0.05, 0.9, 'e)', fontsize=11, transform=ax[0][4].transAxes)
    ax[0][5].text(0.05, 0.9, 'f)', fontsize=11, transform=ax[0][5].transAxes)
    ax[1][0].text(0.05, 0.9, 'g)', fontsize=11, transform=ax[1][0].transAxes)
    ax[1][1].text(0.05, 0.9, 'h)', fontsize=11, transform=ax[1][1].transAxes)
    ax[1][2].text(0.05, 0.9, 'i)', fontsize=11, transform=ax[1][2].transAxes)
    ax[1][3].text(0.05, 0.9, 'j)', fontsize=11, transform=ax[1][3].transAxes)
    ax[1][4].text(0.05, 0.9, 'k)', fontsize=11, transform=ax[1][4].transAxes)
    ax[1][5].text(0.05, 0.9, 'l)', fontsize=11, transform=ax[1][5].transAxes)
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
    fig.colorbar(sct, cax=cbar_ax, ticks=range(5))
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', transparent=True)