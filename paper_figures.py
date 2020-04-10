#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:04:29 2020

@author: amaya
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib_hex_map import matplotlib_hex_map as map_plot
import numpy as np
import pandas as pd
from som import *
    
cpalette = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#42d4f4', '#f032e6', '#469990', '#e6beff', '#9A6324', '#800000', '#000075']

def set_figure():
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rc('font', family='serif')
   
def fig_datacoverage(data, cols, fname=None):    
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    set_figure()
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of entries')
    ax = data[cols[0]].groupby(data.index.year).count().plot(kind="bar")
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', transparent=True)
        
def fig_dimreduc(data, x1, x2, ncls, cmap='Set1', fname=None):
    cmap = mcolors.ListedColormap(cpalette[:ncls])
    fig, ax = plt.subplots(2,6, figsize=(16,6), sharex='none', sharey='row')
    set_figure()
    alpha = 0.6
    size = 0.1
    
    ax[0][0].scatter(x1[:,0], x1[:,1], c=data['Xu_SW_type'], s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[0][1].scatter(x1[:,2], x1[:,1], c=data['Xu_SW_type'], s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[0][2].scatter(x1[:,0], x1[:,1], c=data['Zhao_SW_type'], s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[0][3].scatter(x1[:,2], x1[:,1], c=data['Zhao_SW_type'], s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[0][4].hist2d (x1[:,0], x1[:,1], bins=50, cmap='BuGn', norm=mcolors.PowerNorm(0.3))
    ax[0][5].hist2d (x1[:,2], x1[:,1], bins=50, cmap='BuGn', norm=mcolors.PowerNorm(0.3))
    ax[1][0].scatter(x2[:,0], x2[:,1], c=data['Xu_SW_type'], s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[1][1].scatter(x2[:,2], x2[:,1], c=data['Xu_SW_type'], s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[1][2].scatter(x2[:,0], x2[:,1], c=data['Zhao_SW_type'], s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    sct = ax[1][3].scatter(x2[:,2], x2[:,1], c=data['Zhao_SW_type'], s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[1][4].hist2d (x2[:,0], x2[:,1], bins=50, cmap='BuGn', norm=mcolors.PowerNorm(0.3))
    hst = ax[1][5].hist2d (x2[:,2], x2[:,1], bins=50, cmap='BuGn', norm=mcolors.PowerNorm(0.3))

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
    
    fig.subplots_adjust(right=0.9, left=0.1)
    cbar1 = fig.add_axes([0.04, 0.11, 0.02, 0.77])
    fig.colorbar(sct, cax=cbar1, ticks=range(ncls+1))
    cbar1.yaxis.set_ticks_position('left')
    cbar2 = fig.add_axes([0.92, 0.11, 0.02, 0.77])
    fig.colorbar(hst[3], cax=cbar2)
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', transparent=True)
        
def fig_clustering(data, x1, x2, y1, y2, y3, y4, y5, y6, ncls, cmap='Set1', fname=None):
    cmap = mcolors.ListedColormap(cpalette[:ncls])
    fig, ax = plt.subplots(3,4,figsize=(16,9),sharex='col', sharey='col')
    set_figure()
    alpha = 0.6
    size = 0.1
    
    ax[0][0].scatter(x1[:,0], x1[:,1], c=y1, s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[0][1].scatter(x1[:,2], x1[:,1], c=y1, s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[1][0].scatter(x1[:,0], x1[:,1], c=y2, s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[1][1].scatter(x1[:,2], x1[:,1], c=y2, s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[2][0].scatter(x1[:,0], x1[:,1], c=y3, s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[2][1].scatter(x1[:,2], x1[:,1], c=y3, s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    
    sct = ax[0][2].scatter(x2[:,0], x2[:,1], c=y4, s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[0][3].scatter(x2[:,2], x2[:,1], c=y4, s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[1][2].scatter(x2[:,0], x2[:,1], c=y5, s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[1][3].scatter(x2[:,2], x2[:,1], c=y5, s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[2][2].scatter(x2[:,0], x2[:,1], c=y6, s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    ax[2][3].scatter(x2[:,2], x2[:,1], c=y6, s=size, alpha=alpha, vmin=0, vmax=ncls, cmap=cmap)
    
    ax[0][0].text(0.05, 0.9, 'a)', fontsize=11, transform=ax[0][0].transAxes)
    ax[0][1].text(0.05, 0.9, 'b)', fontsize=11, transform=ax[0][1].transAxes)
    ax[1][0].text(0.05, 0.9, 'c)', fontsize=11, transform=ax[1][0].transAxes)
    ax[1][1].text(0.05, 0.9, 'd)', fontsize=11, transform=ax[1][1].transAxes)
    ax[2][0].text(0.05, 0.9, 'e)', fontsize=11, transform=ax[2][0].transAxes)
    ax[2][1].text(0.05, 0.9, 'f)', fontsize=11, transform=ax[2][1].transAxes)
    
    ax[0][2].text(0.05, 0.9, 'g)', fontsize=11, transform=ax[0][2].transAxes)
    ax[0][3].text(0.05, 0.9, 'h)', fontsize=11, transform=ax[0][3].transAxes)
    ax[1][2].text(0.05, 0.9, 'i)', fontsize=11, transform=ax[1][2].transAxes)
    ax[1][3].text(0.05, 0.9, 'j)', fontsize=11, transform=ax[1][3].transAxes)
    ax[2][2].text(0.05, 0.9, 'k)', fontsize=11, transform=ax[2][2].transAxes)
    ax[2][3].text(0.05, 0.9, 'l)', fontsize=11, transform=ax[2][3].transAxes)
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.77])
    fig.colorbar(sct, cax=cbar_ax, ticks=range(ncls+1))
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', transparent=True)
        
def fig_maps(m, n, som, x, data, ftr_name, px, py, hits, dist, W, wmix, scaler, feat, pcomp=None, fname=None):
    fig, ax = plt.subplots(2 , 4, figsize=(16,7))
    set_figure()
    
    #-- Histogram plot in [0,0]
    color = W.sum(axis=2)
    cmin = color.min() #np.min(x, axis=0)
    cmax = color.max() #np.max(x, axis=0)
    color = (color - cmin) / (cmax - cmin)

    add_data = np.arange(m*n).reshape((m,n))
    add_name = 'node'
    hbin = ax[0][0].hexbin(x[:,0], x[:,1], bins='log', gridsize=30, cmap='BuGn')  
    ax[0][0].scatter(W[:,:,0].flatten(), W[:,:,1].flatten(), c=color.reshape((m*n)), cmap='inferno_r', s=10, marker='o', label='nodes')
    
    f = lambda p, q: p-1 if (q%2 == 0) else p
    i = f(px, py)
    j = py
    ax[0][0].plot([W[px,py,0], W[i +1,j+1,0]], [W[px,py,1], W[i +1,j+1,1]], 'r-', lw=2)
    ax[0][0].plot([W[px,py,0], W[px+1,j+0,0]], [W[px,py,1], W[px+1,j+0,1]], 'r-', lw=2)
    ax[0][0].plot([W[px,py,0], W[i +1,j-1,0]], [W[px,py,1], W[i +1,j-1,1]], 'r-', lw=2)
    ax[0][0].plot([W[px,py,0], W[i +0,j-1,0]], [W[px,py,1], W[i +0,j-1,1]], 'r-', lw=2)
    ax[0][0].plot([W[px,py,0], W[px-1,j+0,0]], [W[px,py,1], W[px-1,j+0,1]], 'r-', lw=2)
    ax[0][0].plot([W[px,py,0], W[i +0,j+1,0]], [W[px,py,1], W[i +0,j+1,1]], 'r-', lw=2)
    
    ax[0][0].xaxis.set_ticks_position("top")
    
    #-- hit map in [0,1]
    size=hits # np.ones_like(hits)
    
    map_plot(ax[0][1], dist, color, m, n, size=size, scale=3, cmap='inferno_r', lcolor='black', title='Hit map')
    ax[0][1].set_aspect('equal')
    ax[0][1].set_xlim(-1, m-0.5)
    ax[0][1].set_ylim(-0.5, n*0.75-0.25)
    
    f = lambda p, q: p-0.5 if (q%2 == 0) else p
    
    i = f(px, py)
    j = py
    ax[0][1].plot([i,i+0.5], [j*0.75,j*0.75+0.75], 'r-', lw=2)
    ax[0][1].plot([i,i+1  ], [j*0.75,j*0.75     ], 'r-', lw=2)
    ax[0][1].plot([i,i+0.5], [j*0.75,j*0.75-0.75], 'r-', lw=2)
    ax[0][1].plot([i,i-0.5], [j*0.75,j*0.75-0.75], 'r-', lw=2)
    ax[0][1].plot([i,i-1  ], [j*0.75,j*0.75     ], 'r-', lw=2)
    ax[0][1].plot([i,i-0.5], [j*0.75,j*0.75+0.75], 'r-', lw=2) 
            
    #-- Oxygen ratio in [0,2]
    ftr = feat.index(ftr_name)
    size=np.ones_like(hits)
    WW = W.reshape(m*n, -1)
    if (pcomp):
        WW = pcomp.inverse_transform(WW)
    WW = scaler.inverse_transform(WW)
    WW = WW.reshape(m, n, len(feat))
    
    color = WW[:,:,ftr]
    cmin = color.min()
    cmax = color.max()
    color = (color - cmin) / (cmax - cmin)

    map_plot(ax[0][2], dist, color, m, n, size=size, scale=3, cmap='inferno_r', title=ftr_name)
    
    #-- Xu solar wind type int [0,3]
    K = 'avqO'
    Q = 'Xu_SW_type'
    V = 2
    color = np.zeros((m, n))
    size  = np.zeros((m, n))
    for x in range(m):
        for y in range(n):
            color[x,y] = data[K].iloc[wmix[x,y]].mean()
            vc = data[Q].iloc[wmix[x,y]].value_counts()
            size [x,y] = vc.loc[V] if V in vc else 0
    color = np.nan_to_num(color)
    cbmin = color.min()
    cbmax = color.max()
    color = (color - cbmin)/(cbmax - cbmin)
    
    sbmin = size.min()
    sbmax = size.max()
    size  = (size - sbmin)/(sbmax - sbmin) if sbmax>sbmin else np.zeros((m, n))
    map_plot(ax[0][3], dist, color, m, n, size=size, scale=3, cmap='inferno_r', lcolor='black', title=K+' ['+Q+'='+str(V)+']')

    #-- Three components in row [1,0:3]
    maxk = min(3, len(feat))
    size  = np.ones_like(hits)
    color = W[:,:,:maxk]
    cmin  = color.min()
    cmax  = color.max()
    color = (color - cmin) / (cmax - cmin)
    map_plot(ax[1][0], dist, color, m, n, size=size, scale=3, cmap='inferno_r', title='Feature map')
    ax[1][0].set_aspect('equal')
    ax[1][0].set_xlim(-1, m-0.5)
    ax[1][0].set_ylim(-0.5, n*0.75-0.25)
        
    for i in range(3):
        if i>=maxk:
            ax[1][i+1].axis('off')
        else:
            color = W[:,:,i]
            cmin  = color.min()
            cmax  = color.max()
            color = (color - cmin) / (cmax - cmin)
            map_plot(ax[1][i+1], dist, color, m, n, size=size, scale=3, cmap='inferno_r', title='Component '+str(i+1))
            ax[1][i+1].set_aspect('equal')
            ax[1][i+1].set_xlim(-1, m-0.5)
            ax[1][i+1].set_ylim(-0.5, n*0.75-0.25)
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', transparent=True)
        
def fig_datarange(data, fname=None):
    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    set_figure()
    plt.violinplot(data, showextrema=False)
    boxprops = dict(linestyle='-', linewidth=2, color='red', facecolor='red', alpha=0.3)
    medianprops = dict(linestyle='-', linewidth=1, color='k')
    meanprops = dict(linestyle='--', color='k')
    whiskerprops = dict(linestyle='-', linewidth=2, color='k')
    plt.boxplot(data, notch=True, showfliers=False, showmeans=True, patch_artist=True, 
                boxprops=boxprops, meanline=True, medianprops=medianprops, meanprops=meanprops,
                whiskerprops=whiskerprops, capprops=whiskerprops)
    plt.xticks(range(1,data.shape[1]+1))
    plt.xlabel('Feature number')
    plt.ylabel('Normalized feature value')
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', transparent=True)
        
def fig_classesdatarange(data, ftr, scaler, nclasses, classname, c, fname=None):
    import matplotlib.ticker as plticker    

    fig, ax = plt.subplots(nclasses, 1, figsize=(4, nclasses), sharex='all', sharey='all')
    set_figure()
    intervals = 0.25 
    loc = plticker.MultipleLocator(base=intervals)
        
    for cl in range(nclasses):
        boxprops = dict(linestyle='-', linewidth=1, color=cpalette[cl], facecolor=cpalette[cl])
        medianprops = dict(linestyle='-', linewidth=1, color='k')
        meanprops = dict(linestyle='--', color='k')
        df = data[data[classname]==cl][ftr]
        raw = scaler.transform(df.values)
        k = raw.mean(axis=0)
        box = ax[cl].boxplot(raw, notch=True, showfliers=False, showmeans=True, meanline=True, patch_artist=True,
                           boxprops=boxprops,
                           #capprops=dict(color=c),
                           #whiskerprops=dict(color=c),
                           #flierprops=dict(color=c, markeredgecolor=c),
                           medianprops=medianprops,
                           meanprops=meanprops,)
        ax[cl].yaxis.set_major_locator(loc)
        ax[cl].grid(which='major', axis='y', alpha=0.5)
        for i, patch in enumerate(box['boxes']):
            # color = np.append(cpalette[cl], k[i])
            patch.set_facecolor(cpalette[cl])
        plt.xticks(range(1,raw.shape[1]+1))
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', transparent=True)
        
def fig_timeseries(data, beg, end, n_clusters, fname=None):
    fig, ax = plt.subplots(6,1, figsize=(16,7), sharex='all')
    set_figure()
    
    ax[0].set_xlim(pd.to_datetime(beg), pd.to_datetime(end))
    cmap = mcolors.ListedColormap(cpalette[:n_clusters])
    ax[0].scatter(data[beg:end].index, data[beg:end]['proton_speed'], c=data[beg:end]['class-kmeans'], cmap=cmap, s=5, zorder=3)
    ax[1].scatter(data[beg:end].index, data[beg:end]['proton_speed'], c=data[beg:end]['class-gmm'], cmap=cmap, s=5, zorder=3)
    ax[2].scatter(data[beg:end].index, data[beg:end]['proton_speed'], c=data[beg:end]['class-som'], cmap=cmap, s=5, zorder=3)
        
    shck = pd.read_csv('catalogs/HSCA_Shock_cat.csv', comment="#", parse_dates={'Datetime' : ['Year','Month','Day','UT']}, index_col='Datetime')
    aces = pd.read_csv('catalogs/ACE_Shocks_cat.csv', comment="#", parse_dates={'Datetime' : ['Month','Day','Year','Time']}, index_col='Datetime')
    icme = pd.read_csv('catalogs/Richarson_Cane_ICME_cat.csv', comment="#", parse_dates=['Datetime'], index_col='Datetime')
        
    for a in range(3):
        for idx in shck[beg:end].index:
            ax[a].axvline(idx, color='blue', alpha=0.3, zorder=1)
        for idx in aces[beg:end].index:
            ax[a].axvline(idx, ls='--', color='blue', alpha=0.3, zorder=1)
        icme['Start'] = pd.to_datetime(icme['Start'])
        icme['End'] = pd.to_datetime(icme['End'])
        for i in range(len(icme[beg:end])):
            x1 = icme[beg:end].iloc[i]['Start']
            x2 = icme[beg:end].iloc[i]['End']
            ax[a].axvspan(x1, x2, color='grey', edgecolor='none', alpha=0.2, zorder=1)
    
    from datetime import timedelta
    for date in pd.date_range(pd.to_datetime(beg), pd.to_datetime(end), freq='52D'):
        ax[3].axvspan(date, date+timedelta(days=27), color='grey', edgecolor='none', alpha=0.2, zorder=1)
    y1 = data[beg:end]['Bgsm_x']
    y2 = data[beg:end]['Bgsm_y']
    # ax[3].plot(y1, 'r-')
    # ax[3].plot(y2, 'g-')
    ax[3].fill_between(y1.index, y1, y2, where=y1>y2, interpolate=True, color='red')
    ax[3].fill_between(y1.index, y1, y2, where=y1<y2, interpolate=True, color='green')
    
    y1 = data[beg:end]['Bgsm_z']
    # ax[4].plot(y1, 'b-')
    
    ax[4].fill_between(y1.index, y1, 0.0, where=y1>0.0, interpolate=True, color='b')
    ax[4].fill_between(y1.index, y1, 0.0, where=y1<0.0, interpolate=True, color='r')

    y1 = data[beg:end]['log_O7to6']
    y2  = np.log(6.008)-0.00578*data[beg:end]['proton_speed']
    y3 = np.log(0.145)    
    ax[5].plot(y1, 'k:')
    # ax2b = ax[2].twinx()
    ax[5].fill_between(y1.index, y2, y3, where=y2>y3, interpolate=True, color='red', alpha=0.3)
    ax[5].plot(y2, 'b-')
    # ax[5].hlines(y3, beg, end, color='blue', linestyles='dashed')

    
    ax[0].set_ylabel(r'$V_{sw}$ $[km/s]$')
    ax[1].set_ylabel(r'$V_{sw}$ $[km/s]$')
    ax[2].set_ylabel(r'$V_{sw}$ $[km/s]$')
    ax[3].set_ylabel(r'$B_{x,y}$ $[nT]$')
    ax[4].set_ylabel(r'$B_z$ $[nT]$')
    ax[5].set_ylabel(r'$\log O^{7+}/O^{6+}$')
    
    ax[0].text(0.01, 0.7, 'a)', fontsize=11, transform=ax[0].transAxes)
    ax[1].text(0.01, 0.7, 'b)', fontsize=11, transform=ax[1].transAxes)
    ax[2].text(0.01, 0.7, 'c)', fontsize=11, transform=ax[2].transAxes)
    ax[3].text(0.01, 0.7, 'd)', fontsize=11, transform=ax[3].transAxes)
    ax[4].text(0.01, 0.7, 'e)', fontsize=11, transform=ax[4].transAxes)
    ax[5].text(0.01, 0.7, 'f)', fontsize=11, transform=ax[5].transAxes)
        
    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter('%b-%Y')
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(fmt)
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', transparent=True)
    
def fig_tsfeatures(data, ftr, cl, beg, end, n_clusters, fname=None):
    nfeat = len(ftr)
    fig, ax = plt.subplots(nfeat, 1, figsize=(16,2*(nfeat+1)), sharex='all')
    set_figure()
    
    shck = pd.read_csv('catalogs/HSCA_Shock_cat.csv', comment="#", parse_dates={'Datetime' : ['Year','Month','Day','UT']}, index_col='Datetime')
    aces = pd.read_csv('catalogs/ACE_Shocks_cat.csv', comment="#", parse_dates={'Datetime' : ['Month','Day','Year','Time']}, index_col='Datetime')
    icme = pd.read_csv('catalogs/Richarson_Cane_ICME_cat.csv', comment="#", parse_dates=['Datetime'], index_col='Datetime')
    
    ax[0].set_xlim(pd.to_datetime(beg), pd.to_datetime(end))
    cmap = mcolors.ListedColormap(cpalette[:n_clusters])
    
    for a, f in enumerate(ftr):
        ax[a].scatter(data[beg:end].index, data[beg:end][f], c=data[beg:end][cl], cmap=cmap, s=5, zorder=3)
        ax[a].set_ylabel(f)
        for idx in shck[beg:end].index:
            ax[a].axvline(idx, color='blue', alpha=0.3, zorder=1)
        for idx in aces[beg:end].index:
            ax[a].axvline(idx, ls='--', color='blue', alpha=0.3, zorder=1)
        icme['Start'] = pd.to_datetime(icme['Start'])
        icme['End'] = pd.to_datetime(icme['End'])
        for i in range(len(icme[beg:end])):
            x1 = icme[beg:end].iloc[i]['Start']
            x2 = icme[beg:end].iloc[i]['End']
            ax[a].axvspan(x1, x2, color='blue', edgecolor='none', alpha=0.15, zorder=2)
    
        from datetime import timedelta
        for date in pd.date_range(pd.to_datetime(beg), pd.to_datetime(end), freq='52D'):
            ax[a].axvspan(date, date+timedelta(days=27), color='grey', edgecolor='none', alpha=0.2, zorder=1)

    locator = mdates.MonthLocator()
    fmt = mdates.DateFormatter('%b-%Y')
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(fmt)
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', transparent=True)

def fig_anyftmap(data, ftr_name, dist, hits, m, n, wmix, lcolor='white', fname=None): 
    fig, ax = plt.subplots(1,1, figsize=(m/2,n/2))
    set_figure()
    
    color = np.zeros((m, n))
    size=hits
    for x in range(m):
        for y in range(n):
            color[x,y] = data[ftr_name].iloc[wmix[x,y]].mean()
    vmax = data[ftr_name].max()
    vmin = data[ftr_name].min()
    color = np.nan_to_num(color)
    cbmin = color.min()
    cbmax = color.max()
    color = (color - color.min())/(color.max() - color.min())

    cmap='magma_r'
    map_plot(ax, dist, color, m, n, size=size, scale=3, title='Mean('+ftr_name+')', lcolor=lcolor, cmap=cmap)
    norm = mcolors.Normalize(vmin=vmax,vmax=vmin)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.75, left=0.1)
    cbar1 = fig.add_axes([0.8, 0.25, 0.05, 0.45])
    cb = plt.colorbar(sm, cax=cbar1)
    cb.ax.tick_params(labelsize='x-small')
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', transparent=True)
        
def fig_componentmap(data, W, feat, nfeat, case, ftr_name, dist, hits, m, n, bneck, wmix, scaler, pca=False, acode=False, pcomp=None, ae=None, lcolor='white', fname=None): 
    import torch
    fig, ax = plt.subplots(1,1, figsize=(m/2,n/2))
    set_figure()
    
    size=np.ones_like(hits)
    ftr = feat[case].index(ftr_name)
    WW = W.reshape(m*n, bneck)
    if pca and not acode:
        WW = pcomp.inverse_transform(WW)
    if acode:
        WW = ae.decode(torch.Tensor(WW)).detach().numpy()
    WW = scaler.inverse_transform(WW)
    WW = WW.reshape(m, n, nfeat)
    
    color = WW[:,:,ftr]
    if ftr_name.startswith('log'):
        color = np.exp(color)
        ftr_name = ftr_name[4:]
    
    vmax = data[ftr_name].max()
    vmin = data[ftr_name].min()
    color = np.nan_to_num(color)
    cbmin = color.min()
    cbmax = color.max()
    color = (color - color.min())/(color.max() - color.min())

    cmap='magma_r'
    map_plot(ax, dist, color, m, n, size=size, scale=3, title=ftr_name, lcolor=lcolor, cmap=cmap)
    norm = mcolors.Normalize(vmin=vmax,vmax=vmin)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.75, left=0.1)
    cbar1 = fig.add_axes([0.8, 0.25, 0.05, 0.45])
    cb = plt.colorbar(sm, cax=cbar1)
    cb.ax.tick_params(labelsize='x-small')
    
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', transparent=True)
        
def fig_swtypes(data, ftr, cname, cnumber, m, n, dist, wmix, fname=None):
    fig, ax = plt.subplots(1,1, figsize=(m/2,n/2))
    set_figure()
    
    K = ftr
    Q = cname
    V = cnumber
    color = np.zeros((m, n))
    size  = np.zeros((m, n))
    for x in range(m):
        for y in range(n):
            color[x,y] = data[K].iloc[wmix[x,y]].mean()
            vc = data[Q].iloc[wmix[x,y]].value_counts()
            size [x,y] = vc.loc[V] if V in vc else 0
    color = np.nan_to_num(color)
    cbmin = data[ftr].min()
    cbmax = data[ftr].max()
    color = (color - cbmin)/(cbmax - cbmin)
    
    sbmin = size.min()
    sbmax = size.max()
    size  = (size - sbmin)/(sbmax - sbmin) if sbmax>sbmin else np.zeros((m, n))

    cmap='inferno_r'
    map_plot(ax, dist, color, m, n, size=size, scale=3, cmap=cmap, lcolor='black', title=K+' ['+Q+'='+str(V)+', max hits:'+str(int(sbmax))+']')
    
    norm = mcolors.Normalize(vmin=cbmin,vmax=cbmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(right=0.75, left=0.1)
    cbar1 = fig.add_axes([0.8, 0.25, 0.05, 0.45])
    cb = plt.colorbar(sm, cax=cbar1)
    cb.ax.tick_params(labelsize='x-small')

    if fname is not None:
        plt.savefig(fname, bbox_inches='tight', transparent=True)