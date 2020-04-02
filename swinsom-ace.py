#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:35:27 2020

@author: amaya
"""

'''
    ---------------------
    Read and process data
    ---------------------
'''

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from acedata import *
from matplotlib_hex_map import matplotlib_hex_map as map_plot 

## Seting up the path and range -----------------------------------------------
# acedir  : directory containing the hdf5 ACE data
# ybeg    : year of start of the analysis
# yend    : year of ending of the analysis
#acedir = '/home/amaya/Data/ACE'
#outdir = '/home/amaya/Sources/swinsom-git/papers/2020-Frontiers/figures/'
acedir = '/home/amaya/Workdir/MachineLearning/Data/ACE'
outdir = '/home/amaya/Workdir/MachineLearning/swinsom-git/papers/2020-Frontiers/figures/'
# ybeg  = 2002
# yend  = 2004
ybeg  = 1998
yend  = 2011
optim = False
calculate_som = False
clustering = False

## Code options ---------------------------------------------------------------
# acode   : use autoencoding to generate the training data
# case    : selection of features from the available options in the dict
case = 'Amaya'
dynamic = True
params = {'Roberts' :
              {'ybeg' : 2002,
               'yend' : 2004,
               'autoencode' : False,
               'pca' : True,
               'm' : 12,
               'n' : 12,
               'maxiter' : 50000,
               'batch size' : 32,
               'nepochs' : 30,
               'sigma' : 5.0,
               'learning_rate' : 0.25,
               'init_method' : 'rand_points',
               'bottle_neck' : 8,},
          'XuBorovsky' :
              {'ybeg' : 2002,
               'yend' : 2004,
               'autoencode' : False,
               'pca' : True,
               'm' : 12,
               'n' : 12,
               'maxiter' : 50000,
               'batch size' : 32,
               'nepochs' : 30,
               'sigma' : 5.0,
               'learning_rate' : 0.25,
               'init_method' : 'rand_points',
               'bottle_neck' : 4,},
          'ZhaZuFi' :
              {'autoencode' : False,
               'pca' : True,
               'm' : 12,
               'n' : 12,
               'maxiter' : 50000,
               'batch size' : 32,
               'nepochs' : 30,
               'sigma' : 5.0,
               'learning_rate' : 0.25,
               'init_method' : 'rand_points',
               'bottle_neck' : 3,},
          'Amaya' :
              {'autoencode' : False,
               'pca' : False,
               'm' : 12,
               'n' : 12,
               'maxiter' : 50000,
               'batch size' : 32,
               'nepochs' : 30,
               'sigma' : 5.0,
               'learning_rate' : 0.25,
               'init_method' : 'rand_points',
               'bottle_neck' : 8,
              },
         }

## Selecting the features to be used ------------------------------------------
# cols    : columns to load from the file
# xcols   : extra columns to add by additional processing. This will add columns
#         to cols, as some of the values required to calculate the extra
#         columns need to be loaded.
# feat    : Feature dictionary of the selected innputs for the training
cols = allacecols
xcols = ['sigmac',
         'sigmar',
         'Ma',
         'Zhao_SW_type',
         'proton_speed_range',
         'proton_density_range',
         'proton_temp_range',
         'Bgsm_x_range',
         'Bgsm_y_range',
         'Bgsm_z_range',
         'Bmag_acor',
         'Bmag_range',
         'Bmag_mean',
         'Bmag_std',
         'proton_density_range',
         'Sp',
         'Va',
         'Texp',
         'Tratio',
         'Xu_SW_type',
         'log_proton_speed',
         'log_proton_density',
         'log_O7to6',
         'log_FetoO',
         'log_avqFe',
         'log_Bmag',
         'log_Sp',
         'log_Va',
         'log_Texp',
         'log_Tratio']

feat = {'Roberts' :
            ['log_O7to6',
             'log_proton_speed',
             'log_proton_density',
             'sigmac',
             'sigmar',
             'log_FetoO',
             'log_avqFe',
             'log_Bmag'],
        'XuBorovsky' :
            ['log_Sp',
             'log_Va',
             'log_Tratio'],
        'ZhaZuFi' : 
            ['O7to6',
             'log_proton_speed'],
        'Amaya' : 
            ['O7to6',
             'log_proton_speed',
             'log_Sp',
             'log_Va',
             'log_Tratio',
             'log_Bmag',
             'proton_temp',
             'proton_density',
             'Ma',
             'He4toprotons',
             'FetoO',
             'C6to5',
             'avqFe',
             'sigmac',
             'sigmar',
             'proton_speed_range',
             'proton_density_range',
             'proton_temp_range',
             'Bgsm_x_range',
             'Bgsm_y_range',
             'Bgsm_z_range',
             'Bmag_range',
             'Bmag_acor',
             'Bmag_mean',
             'Bmag_std'],
       }


## Seting up the options, given the case
acode   = params[case]['autoencode']
pca     = params[case]['pca']
mmax    = params[case]['m']
nmax    = params[case]['n']
maxiter = params[case]['maxiter']
sg      = params[case]['sigma']
lr      = params[case]['learning_rate']
init    = params[case]['init_method']
bneck   = params[case]['bottle_neck']
nfeat   = len(feat[case])

if not dynamic:
    sg = min(sg, int(max(m,n)/2))

if acode:
    batch_size = params[case]['batch size']
    num_epochs = params[case]['nepochs']

## Loading the data
data, nulls = acedata(acedir, cols, ybeg, yend)
if case=='Roberts':
    data['2002-11','2004-05']
print('Data set size after reading files:', len(data))
data = aceaddextra(data, nulls, xcols=xcols, window='6H', center=False)
print('Data set size after adding extras:', len(data))

'''
    ----------------
    Data compression
    ----------------
'''
scaler = MinMaxScaler()

raw = data[feat[case]].values
raw = scaler.fit_transform(raw)

if acode:
    from autoencoder import autoencoder
    nodes = [raw.shape[1], 18, 12, bneck]
    ae = autoencoder(nodes)
    L, T = ae.fit(torch.Tensor(raw), batch_size, num_epochs)
    x = ae.encode(torch.Tensor(raw)).detach().numpy()
    if pca:
        pcomp = PCA(n_components=bneck, whiten=True)
        xpca = pcomp.fit_transform(raw)
else:
    if pca:
        pcomp = PCA(n_components=bneck, whiten=True)
        x = pcomp.fit_transform(raw)
    else:
        x = raw

'''
    ---------------------------------------
    Perform classical clustering techniques
    ---------------------------------------
'''

if clustering:
    from sklearn import cluster, mixture
    kms = cluster.MiniBatchKMeans(n_clusters=4)
    spc = cluster.SpectralClustering(n_clusters=4, eigen_solver='arpack', affinity="nearest_neighbors")
    gmm = mixture.GaussianMixture(n_components=4, covariance_type='full')
    
    y_kms = kms.fit_predict(x)
    y_spc = spc.fit_predict(x)
    y_gmm = gmm.fit_predict(x)
    if acode and pca:
        y_kms_pca = kms.fit_predict(xpca)
        y_spc_pca = spc.fit_predict(xpca)
        y_gmm_pca = gmm.fit_predict(xpca)

'''
    -------------
    Train the SOM
    -------------
'''

## Hyperparameter optimization using optuna
if optim:
    from som import *
    import optuna
    def objective(trial):
        m = trial.suggest_int('m', 5, mmax)
        n = trial.suggest_int('n', 5, nmax)
        lr = trial.suggest_uniform('lr', 0.01, 1.0)
        sg = trial.suggest_uniform('sg', 1.0, 6.0)
        som = selfomap(x, m, n, 1000, sigma=sg, learning_rate=lr, init=init, dynamic=dynamic)
        dist = som_distances(som)
        print(" Mean distance: ", dist.mean())
        return som.quantization_error(x) + dist.mean() + 0.05*sg
    
    study = optuna.create_study()
    study.optimize(objective, n_trials=200)
    
    ## Launch the model sing the optial hyperparameters
    lr = study.best_params['lr']
    sg = study.best_params['sg']
    m  = study.best_params['m']
    n  = study.best_params['n']

if calculate_som:
    from som import *

    ## Run the model   
    som = selfomap(x, m, n, maxiter, sigma=sg, learning_rate=lr, init=init, dynamic=dynamic)
    
    ## processing of the SOM
    # dist : matrix of distances between map nodes
    # hits : total hits for each one of the map nodes
    # wmix : indices of the elements of x that hit each one of the map nodes
    # W    : SOM weights
    dist = som_distances(som)
    hits = som_hits(som, x, m, n, log=True)
    wmix = som.win_map_index(x)
    W    = som.get_weights() 
    
    print(" Mean distance: ", dist.mean())
    
    '''
        ----------------
        Plot the results
        ----------------
    '''
    
    ## Switch on/off plots
    plot_hitmap = False         # Plots the SOM hit map
    plot_neighbors = False      # Plots lines connecting neighbors in the maps and feature space
    plot_featurespace = False   # Plots the feature space
    plot_components = False     # Plot maps of the three components
    plot_features = False       # plot maps of the corresponding features
    plot_datamean = False
    
    ## Select the neighbour to visualize
    if plot_neighbors:
        px = 3
        py = 3
    
    # plt.close('all')

    color = W.sum(axis=2)
    cmin = color.min() #np.min(x, axis=0)
    cmax = color.max() #np.max(x, axis=0)
    color = (color - cmin) / (cmax - cmin)
        
    if plot_hitmap:            
        size=hits # np.ones_like(hits)
        
        fig, ax = plt.subplots(1,1)
        map_plot(ax, dist, color, m, n, size=size, scale=8, cmap='inferno_r', lcolor='black')
        plt.title('Hit map')
        
        if plot_neighbors:
            f = lambda p, q: p-0.5 if (q%2 == 0) else p
            
            i = f(px, py)
            j = py
            plt.plot([i,i+0.5], [j*0.75,j*0.75+0.75], 'k-')
            plt.plot([i,i+1  ], [j*0.75,j*0.75     ], 'k-')
            plt.plot([i,i+0.5], [j*0.75,j*0.75-0.75], 'k-')
            plt.plot([i,i-0.5], [j*0.75,j*0.75-0.75], 'k-')
            plt.plot([i,i-1  ], [j*0.75,j*0.75     ], 'k-')
            plt.plot([i,i-0.5], [j*0.75,j*0.75+0.75], 'k-')
    
    if plot_featurespace:
        plt.figure()
        add_data = np.arange(m*n).reshape((m,n))
        add_name = 'node'
        finaldata = som_addinfo(som, data, x, add_data, add_name)
        hbin = plt.hexbin(x[:,0], x[:,1], bins='log', gridsize=30, cmap='BuGn')  
        plt.colorbar(hbin)
        plt.scatter(W[:,:,0].flatten(), W[:,:,1].flatten(), c=color.reshape((m*n)), cmap='inferno_r', s=50, marker='o', label='nodes')
        plt.title('Code words')
        
        if plot_neighbors:
            f = lambda p, q: p-1 if (q%2 == 0) else p
            i = f(px, py)
            j = py
            plt.plot([W[px,py,0], W[i +1,j+1,0]], [W[px,py,1], W[i +1,j+1,1]], 'k-')
            plt.plot([W[px,py,0], W[px+1,j+0,0]], [W[px,py,1], W[px+1,j+0,1]], 'k-')
            plt.plot([W[px,py,0], W[i +1,j-1,0]], [W[px,py,1], W[i +1,j-1,1]], 'k-')
            plt.plot([W[px,py,0], W[i +0,j-1,0]], [W[px,py,1], W[i +0,j-1,1]], 'k-')
            plt.plot([W[px,py,0], W[px-1,j+0,0]], [W[px,py,1], W[px-1,j+0,1]], 'k-')
            plt.plot([W[px,py,0], W[i +0,j+1,0]], [W[px,py,1], W[i +0,j+1,1]], 'k-')
            
    if plot_components:
        size  = np.ones_like(hits)
        color = W
        cmin  = color.min()
        cmax  = color.max()
        color = (color - cmin) / (cmax - cmin)
        fig, ax = plt.subplots(1,1)
        map_plot(ax, dist, color, m, n, size=size, scale=8, cmap='inferno_r')
        plt.title('Components')
        
        for i in range(3):
            color = W[:,:,i]
            cmin  = color.min()
            cmax  = color.max()
            color = (color - cmin) / (cmax - cmin)
            fig, ax = plt.subplots(1,1)
            map_plot(ax, dist, color, m, n, size=size, scale=8, cmap='inferno_r')
            plt.title('Component '+str(i))
    
    def plt_features(ftr_name):
        ftr = feat[case].index(ftr_name)
        size=np.ones_like(hits)
        WW = W.reshape(m*n, bneck)
        if pca and not acode:
            WW = pcomp.inverse_transform(WW)
        if acode:
            WW = ae.decode(torch.Tensor(WW)).detach().numpy()
        WW = scaler.inverse_transform(WW)
        WW = WW.reshape(m, n, nfeat)
        
        color = WW[:,:,ftr]
        cmin = color.min()
        cmax = color.max()
        color = (color - cmin) / (cmax - cmin)
    
        fig, ax = plt.subplots(1,1)
        map_plot(ax, dist, color, m, n, size=size, scale=8, cmap='inferno_r')
        plt.title(ftr_name)
             
    if plot_features:
        plt_features('proton_speed')
        plt_features('O7to6')
       
    def plt_mapdatamean(K):
        color = np.zeros((m, n))
        size=hits
        for x in range(m):
            for y in range(n):
                color[x,y] = data[K].iloc[wmix[x,y]].mean()
        color = np.nan_to_num(color)
        cbmin = color.min()
        cbmax = color.max()
        color = (color - cbmin)/(cbmax - cbmin)
        fig, ax = plt.subplots(1,1)
        cmap = plt.cm.get_cmap('jet_r', 5)
        map_plot(ax, dist, color, m, n, size=size, scale=6, cmap=cmap, lcolor='black')
        
    if plot_datamean:
        plt_mapdatamean('Xu_SW_type')
        plt_mapdatamean('Zhao_SW_type')
        

'''
    ------------------------
    Generate the paper plots
    ------------------------
'''

import paper_figures as pfig

# fig_path = outdir+case
# pfig.fig_datacoverage(data, cols, fname=fig_path+'/datacoverage.png')
# pfig.fig_dimreduc(data, xpca, x, cmap='jet_r', fname=fig_path+'/dimreduc.png')
# pfig.fig_clustering(data, x, xpca, y_kms, y_spc, y_gmm, y_kms_pca, y_spc_pca, y_gmm_pca, cmap='jet', fname=fig_path+'/clustering.png')
# pfig.fig_maps(m, n, som, x, data, 2, 4, hits, dist, W, wmix, pcomp, scaler, feat[case], fname=fig_path+'/maps.png')
pfig.fig_datarange(raw, fname=fig_path+'/datarange.png')
