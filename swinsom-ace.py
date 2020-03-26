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
from som import *
from autoencoder import autoencoder
from matplotlib_hex_map import matplotlib_hex_map as map_plot

## Seting up the path and range -----------------------------------------------
# acedir  : directory containing the hdf5 ACE data
# ybeg    : year of start of the analysis
# yend    : year of ending of the analysis
acedir = '/home/amaya/Workdir/MachineLearning/Data/ACE'
ybeg = 2009
yend = 2011

## Code options ---------------------------------------------------------------
# acode   : use autoencoding to generate the training data
# case    : selection of features from the available options in the dict
case = 'Roberts'
params = {'Roberts' :
              {'autoencode' : False,
               'pca' : True,
               'm' : 7,
               'n' : 9,
               'maxiter' : 15000,
              },
          'Amaya' :
              {'autoencode' : True,
               'pca' : False,
               'm' : 7,
               'n' : 9,
               'maxiter' : 15000,
               'batch size' : 32,
               'nepochs' : 100
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
         'SW_type',
         'proton_speed_delta',
         'proton_density_delta',
         'Bgsm_x_delta',
         'Bgsm_y_delta',
         'Bgsm_z_delta',
         'Bmag_acor',
         'Bmag_delta',
         'Bmag_mean',
         'Bmag_std',
         'proton_density_delta']
logcols = ['proton_speed',
           'proton_density',
           'sigmac',
           'sigmar',
           'O7to6',
           'FetoO',
           'avqFe',
           'Bmag']

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
                  'He4toprotons',
                  'O7to6',
                  'FetoO',
                  'C6to5',
                  'sigmac',
                  'sigmar',
                  'proton_speed_delta',
                  'proton_density_delta',
                  'Bgsm_x_delta',
                  'Bgsm_y_delta',
                  'Bgsm_z_delta',
                  'Bmag_delta',
                  'Bmag_acor',
                  'Bmag_mean',
                  'Bmag_std'],
       }


## Seting up the options, given the case
acode   = params[case]['autoencode']
pca     = params[case]['pca']
m       = params[case]['m']
n       = params[case]['n']
maxiter = params[case]['maxiter']
if acode:
    batch_size = params[case]['batch size']
    num_epochs = params[case]['nepochs']

## Loading the data
data, nulls = acedata(acedir, cols, ybeg, yend)
data = aceaddextra(data, nulls, xcols=xcols, window=7, center=False)
data = addlogs(data, logcols)

'''
    ---------------
    Autoencode data
    ---------------
'''

raw = data[feat[case]].values

scaler = MinMaxScaler()
raw = scaler.fit_transform(raw)

if acode:
    nodes = [raw.shape[1], 7, 2]
    ae = autoencoder(nodes)
    L = ae.fit(torch.Tensor(raw), batch_size, num_epochs)
    x = ae.encode(torch.Tensor(raw)).detach().numpy()
else:
    ## Transform the data
    if pca:
        pcomp = PCA(n_components=2, whiten=True)
        x = pcomp.fit_transform(raw)

'''
    -------------
    Train the SOM
    -------------
'''

som = selfomap(x, m, n, 20000, sigma=4.0, learning_rate=0.2, init='random', dynamic=True)
# som = selfomap(x, m, n, 20000, sigma=5.0, learning_rate=2.0, init='random', dynamic=False)

## processing of the SOM
# dist : matrix of distances between map nodes
# hits : total hits for each one of the map nodes
# wmix : indices of the elements of x that hit each one of the map nodes
dist = som_distances(som)
hits = som_hits(som, x, m, n, log=False)
wmix = som.win_map_index(x)

'''
    ----------------
    Plot the results
    ----------------
'''

plt.close('all')

color = som.get_weights()[:,:,:2].sum(axis=2)
cmin = color.min() #np.min(x, axis=0)
cmax = color.max() #np.max(x, axis=0)
color = (color - cmin) / (cmax - cmin)

map_plot(dist, color, m, n, size=np.ones_like(hits), scale=6)
plt.plot([1,1.5], [0.75,1.5], 'k-')
plt.plot([1,2], [0.75,0.75], 'k-')
plt.plot([1,1.5], [0.75,0], 'k-')
plt.plot([1,0.5], [0.75,0], 'k-')
plt.plot([1,0], [0.75,0.75], 'k-')
plt.plot([1,0.5], [0.75,1.5], 'k-')

plt.figure()
add_data = np.arange(m*n).reshape((m,n))
add_name = 'node'
finaldata = som_addinfo(som, data, x, add_data, add_name)
#plt.scatter(x[:,0],x[:,1],c=finaldata['node'].values, label='data', cmap='prism', s=10, edgecolors='none', alpha=0.5)
#plt.scatter(x[:,1],x[:,2],c=x[:,0], label='data', cmap='jet', s=10, edgecolors='none', alpha=0.5)
plt.hexbin(x[:,0], x[:,1], bins=None, gridsize=30, cmap='hot_r')
#plt.scatter(som.get_weights()[:,:,1].flatten(), som.get_weights()[:,:,2].flatten(), c=color.reshape((m*n,3)), s=50, marker='o', label='nodes')

W = som.get_weights()
plt.scatter(W[:,:,0].flatten(), W[:,:,1].flatten(), c=color.reshape((m*n)), cmap='jet', s=50, marker='o', label='nodes')
plt.plot([W[1,1,0], W[2,2,0]], [W[1,1,1], W[2,2,1]], 'k-')
plt.plot([W[1,1,0], W[2,1,0]], [W[1,1,1], W[2,1,1]], 'k-')
plt.plot([W[1,1,0], W[2,0,0]], [W[1,1,1], W[2,0,1]], 'k-')
plt.plot([W[1,1,0], W[1,0,0]], [W[1,1,1], W[1,0,1]], 'k-')
plt.plot([W[1,1,0], W[0,1,0]], [W[1,1,1], W[0,1,1]], 'k-')
plt.plot([W[1,1,0], W[1,2,0]], [W[1,1,1], W[1,2,1]], 'k-')
