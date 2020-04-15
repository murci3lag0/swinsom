#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:52:32 2020

@author: amaya
"""

import pickle
import joblib 
import numpy as np
import torch

np.random.seed(12345)
torch.manual_seed(56789)

case='Roberts'
Tfeat =['log_O7to6',
        'log_proton_speed',
        'log_proton_density',
        'sigmac',
        'sigmar',
        'log_FetoO',
        'log_avqFe',
        'log_Bmag']

with open(case+'.pkl', 'rb') as f:
    Tx = pickle.load(f)
    Txpca = pickle.load(f)
    Ty_kms = pickle.load(f)
    Ty_gmm = pickle.load(f)
    Ty_kms_pca = pickle.load(f)
    Ty_gmm_pca = pickle.load(f)
    Tlr = pickle.load(f)
    Tsg = pickle.load(f)
    Tm = pickle.load(f)
    Tn = pickle.load(f)
    Tdata = pickle.load(f)
print('Pickled data re-loaded')
Tae = joblib.load(case+'-autoencoder.pkl')
Tpcomp = joblib.load(case+'-pcacomponents.pkl')
Tscaler = joblib.load(case+'-scaler.pkl')
Tscaler_pca = joblib.load(case+'-scaler_pca.pkl')
Tscaler_aec = joblib.load(case+'-scaler_aec.pkl')
print('joblib models re-loaded')

Traw = Tdata[Tfeat].values
Traw = Tscaler.transform(Traw)

from som import *

## Run the model 
print('SOM training...',Tm,Tn,Tlr,Tsg)
Tsom = selfomap(Tx, Tm, Tn, 30000, sigma=Tsg, learning_rate=Tlr, init='random_points', dynamic=True)

## processing of the SOM
# dist : matrix of distances between map nodes
# hits : total hits for each one of the map nodes
# wmix : indices of the elements of x that hit each one of the map nodes
# W    : SOM weights
Tdist = som_distances(Tsom)
Thits = som_hits(Tsom, Tx, Tm, Tn, log=True)
Twmix = Tsom.win_map_index(Tx)
TW    = Tsom.get_weights() 

print(" Mean distance: ", Tdist.mean())

'''
    ---------------------
    Cluster the SOM nodes
    ---------------------
'''
from sklearn import cluster
TC1 = cluster.MiniBatchKMeans(n_clusters=8, n_init=500).fit(TW.reshape(Tm*Tn,-1))
TC1 = np.array(TC1.labels_)
TC1 = TC1.reshape((Tm,Tn))
Tdata = som_addinfo(Tsom, Tdata, Tx, TC1, 'class-som')
Tbdry,_ = som_boundaries(TC1, Tm, Tn)

Tcolor = TC1
Tcmin = Tcolor.min() #np.min(x, axis=0)
Tcmax = Tcolor.max() #np.max(x, axis=0)
Tcolor = (Tcolor - Tcmin) / (Tcmax - Tcmin)

import matplotlib.pyplot as plt
Tfig, Tax = plt.subplots(1,1)
Tcmap = plt.cm.get_cmap('jet', 8)
from matplotlib_hex_map import matplotlib_hex_map as Tmap_plot 
Tmap_plot(Tax, Tbdry, Tcolor, Tm, Tn, usezero=True,  lcolor='black', size=np.ones((Tm,Tn)), scale=1, cmap=Tcmap)
