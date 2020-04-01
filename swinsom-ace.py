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
from autoencoder import autoencoder
from matplotlib_hex_map import matplotlib_hex_map as map_plot
import optuna 

## Seting up the path and range -----------------------------------------------
# acedir  : directory containing the hdf5 ACE data
# ybeg    : year of start of the analysis
# yend    : year of ending of the analysis
acedir = '/home/amaya/Workdir/MachineLearning/Data/ACE'
ybeg  = 1998
yend  = 2011
optim = False
calculate_som = True
clustering = True

## Code options ---------------------------------------------------------------
# acode   : use autoencoding to generate the training data
# case    : selection of features from the available options in the dict
case = 'Amaya'
params = {'Roberts' :
              {'autoencode' : False,
               'pca' : True,
               'm' : 7,
               'n' : 9,
               'maxiter' : 50000,
               'dynamic' : False,
               'sigma' : 4.0,
               'learning_rate' : 0.1,
               'init_method' : 'random',
               'bottle_neck' : 3,
              },
          'Amaya' :
              {'autoencode' : True,
               'pca' : True,
               'm' : 7,
               'n' : 9,
               'maxiter' : 50000,
               'batch size' : 32,
               'nepochs' : 100,
               'dynamic' : True,
               'sigma' : 5.0,
               'learning_rate' : 0.25,
               'init_method' : 'random',
               'bottle_neck' : 3,
              },
         }

## Selecting the features to be used ------------------------------------------
# cols    : columns to load from the file
# xcols   : extra columns to add by additional processing. This will add columns
#         to cols, as some of the values required to calculate the extra
#         columns need to be loaded.
# logcols : columns that need to be processed in log scale
# feat    : Feature dictionary of the selected innputs for the training
cols = allacecols
xcols = ['sigmac',
         'sigmar',
         'Ma',
         'Zhao_SW_type',
         'proton_speed_range',
         'proton_density_range',
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
         'Xu_SW_type']
logcols = ['proton_speed',
           'proton_density',
           'sigmac',
           'sigmar',
           'O7to6',
           'FetoO',
           'avqFe',
           'Bmag',
           'Sp',
           'Va',
           'Texp',
           'Tratio']

feat = {'Roberts' : ['log_proton_speed',
                     'log_proton_density',
                     'log_sigmac',
                     'log_sigmar',
                     'log_O7to6',
                     'log_FetoO',
                     'log_avqFe',
                     'log_Bmag'],
       'Amaya' : ['proton_speed',
                  'proton_temp',
                  'proton_density',
                  'Ma',
                  'He4toprotons',
                  'O7to6',
                  'FetoO',
                  'C6to5',
                  'sigmac',
                  'sigmar',
                  'proton_speed_range',
                  'proton_density_range',
                  'Bgsm_x_range',
                  'Bgsm_y_range',
                  'Bgsm_z_range',
                  'Bmag_range',
                  'Bmag_acor',
                  'Bmag_mean',
                  'Bmag_std',
                  'log_Sp',
                  'log_Va',
                  'log_Texp',
                  'log_Tratio'],
       }


## Seting up the options, given the case
acode   = params[case]['autoencode']
pca     = params[case]['pca']
m       = params[case]['m']
n       = params[case]['n']
maxiter = params[case]['maxiter']
dynamic = params[case]['dynamic']
sg      = params[case]['sigma']
lr      = params[case]['learning_rate']
init    = params[case]['init_method']
bneck   = params[case]['bottle_neck']

if not dynamic:
    sg = min(sg, int(max(m,n)/2))

if acode:
    batch_size = params[case]['batch size']
    num_epochs = params[case]['nepochs']

## Loading the data
data, nulls = acedata(acedir, cols, ybeg, yend)
print('Data set size after reading files:', len(data))
data = aceaddextra(data, nulls, xcols=xcols, window='6H', center=False)
print('Data set size after adding extras:', len(data))
data = addlogs(data, logcols)

'''
    ----------------
    Data compression
    ----------------
'''
scaler = MinMaxScaler()

raw = data[feat[case]].values
raw = scaler.fit_transform(raw)

if acode:
    nodes = [raw.shape[1], 13, 7, bneck]
    ae = autoencoder(nodes)
    L = ae.fit(torch.Tensor(raw), batch_size, num_epochs)
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
    def objective(trial):
        m = trial.suggest_int('m', 5, 12)
        n = trial.suggest_int('n', 5, 12)
        lr = trial.suggest_uniform('lr', 0.05, 1.0)
        sg = trial.suggest_uniform('sg', 1.0, 10.0)
        som = selfomap(x, m, n, 1000, sigma=sg, learning_rate=lr, init=init, dynamic=dynamic)
        dist = som_distances(som)
        print(" Mean distance: ", dist.mean())
        return som.quantization_error(x) + dist.mean()
    
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    
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
    plot_neighbors = True      # Plots lines connecting neighbors in the maps and feature space
    plot_featurespace = True   # Plots the feature space
    
    ## Select the neighbour to visualize
    if plot_neighbors:
        px = 3
        py = 5
    
    # plt.close('all')

    color = W.sum(axis=2)
    cmin = color.min() #np.min(x, axis=0)
    cmax = color.max() #np.max(x, axis=0)
    color = (color - cmin) / (cmax - cmin)
        
    if plot_hitmap:            
        size=hits # np.ones_like(hits)
        
        map_plot(dist, color, m, n, size=size, scale=5, cmap='autumn')
        
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
        plt.hexbin(x[:,0], x[:,1], bins=None, gridsize=30, cmap='BuGn')    
        plt.scatter(W[:,:,0].flatten(), W[:,:,1].flatten(), c=color.reshape((m*n)), cmap='autumn', s=50, marker='o', label='nodes')
        
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

'''
    ------------------------
    Generate the paper plots
    ------------------------
'''

import paper_figures as pfig

fig_path = '/home/amaya/Workdir/MachineLearning/swinsom-git/papers/2020-Frontiers/figures'
# pfig.fig_datacoverage(data, cols, fname=fig_path+'/datacoverage.png')
pfig.fig_dimreduc(data, xpca, x, cmap='jet_r', fname=fig_path+'/dimreduc.png')
pfig.fig_clustering(data, x, xpca, y_kms, y_spc, y_gmm, y_kms_pca, y_spc_pca, y_gmm_pca, cmap='jet', fname=fig_path+'/clustering.png')
