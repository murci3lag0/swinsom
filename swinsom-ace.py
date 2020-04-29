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

import sys
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import joblib
from acedata import *
from matplotlib_hex_map import matplotlib_hex_map as map_plot 

np.random.seed(12345)
torch.manual_seed(56789)

## Seting up the path and range -----------------------------------------------
# acedir : directory containing the hdf5 ACE data
# outdir : figure output directory
acedir = '/home/amaya/Data/ACE'
outdir = '/home/amaya/Sources/swinsom-git/papers/2020-Frontiers/figures/'

optim = False
calculate_som = True
clustering = False
generate_paper_figures = False

## Code options ---------------------------------------------------------------
# acode   : use autoencoding to generate the training data
# case    : selection of features from the available options in the dict
cases = ['Roberts', 'XuBorovsky', 'ZhaZuFi', 'Amaya']
modes = ['load', 'new']
if len(sys.argv)!=3:
    print('ERROR! Number of arguments')
    print('       [1] Must be one of: ', cases)
    print('       [2] Must be one of: ', modes)
    sys.exit("Number of arguments error.")
if str(sys.argv[1]) not in cases:
    print('ERROR! Incorrect case name')
    print('       [1] Must be one of: ', cases)
    sys.exit("Arguments error.")
if str(sys.argv[2]) not in modes:
    print('ERROR! Incorrect method of initialization')
    print('       [2] Must be one of:', modes)
    sys.exit("Arguments error.")
    
case = str(sys.argv[1])
mode = str(sys.argv[2])

params = {'Roberts' :
              {'ybeg' : 2002,
               'yend' : 2004,
               'autoencode' : True,
               'nodes' : [8, 13, 7, 3],
               'pca' : True,
               'm' : 12,
               'n' : 12,
               'maxiter' : 20000,
               'batch size' : 32,
               'nepochs' : 50,
               'sigma' : 9.0,
               'learning_rate' : 1.0,
               'init_method' : 'rand_points',
               'bottle_neck' : 3,
               'nbr_clusters' : 8,
               'dynamic' : True},
          'XuBorovsky' :
              {'ybeg' : 1998,
               'yend' : 2008,
               'autoencode' : False,
               'pca' : True,
               'm' : 12,
               'n' : 12,
               'maxiter' : 100000,
               'batch size' : 32,
               'nepochs' : 30,
               'sigma' : 5.0,
               'learning_rate' : 1.0,
               'init_method' : 'rand_points',
               'bottle_neck' : 3,
               'nbr_clusters' : 4,
               'dynamic' : True},
          'ZhaZuFi' :
              {'ybeg' : 1998,
               'yend' : 2008,
               'autoencode' : False,
               'pca' : False,
               'm' : 12,
               'n' : 12,
               'maxiter' : 100000,
               'batch size' : 32,
               'nepochs' : 30,
               'sigma' : 6.0,
               'learning_rate' : 2.0,
               'init_method' : 'rand_points',
               'bottle_neck' : 2,
               'nbr_clusters' : 4,
               'dynamic' : False},
          'Amaya' :
              {'ybeg' : 1998,
               'yend' : 2011,
               'autoencode' : True,
               'nodes' : [21, 17, 9, 3],
               'pca' : True,
               'm' : 12,
               'n' : 12,
               'maxiter' : 100000,
               'batch size' : 32,
               'nepochs' : 30,
               'sigma' : 9.0,
               'learning_rate' : 1.0,
               'init_method' : 'rand_points',
               'bottle_neck' : 3,
               'nbr_clusters' : 8,
               'dynamic' : True
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
         'Br_range',
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
         'log_C6to5',
         'log_Bmag',
         'log_Sp',
         'log_Va',
         'log_Texp',
         'log_Tratio',
         'log_Bmag_std',
         'log_Ma',
         'log_He4toprotons',
         'log_Bgsm_x_range',
         'log_Bmag_range',
         'log_Bmag_mean',
         'log_Bmag_acor',
         'log_Bgsm_z_range',
         'log_proton_temp',
         'log_proton_temp_range',
         'log_proton_speed_range',
         'log_Bgsm_y_range',
         'log_proton_density_range']

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
            ['log_O7to6',
             'log_proton_speed'],
        'Amaya' : 
            ['log_O7to6',
      		'log_proton_speed',
      		'log_proton_density',
      		'log_FetoO',
      		'log_avqFe',
      		'sigmac',
      		'sigmar',
      		'log_Bmag',
      		'log_Sp',
      		'log_Va',
      		'log_Tratio',
      		'log_proton_temp',
      		'log_Ma',
      		'log_C6to5',
      		'log_proton_density_range',
      		'log_proton_temp_range',
      		'log_Bgsm_x_range',
      		'log_Bgsm_z_range',
      		'log_Bmag_range',
      		'Bmag_acor',
      		'Delta',]
       }


## Seting up the options, given the case
ybeg    = params[case]['ybeg']
yend    = params[case]['yend']
acode   = params[case]['autoencode']
pca     = params[case]['pca']
mmax    = params[case]['m']
nmax    = params[case]['n']
maxiter = params[case]['maxiter']
sgmax   = params[case]['sigma']
lrmax   = params[case]['learning_rate']
init    = params[case]['init_method']
bneck   = params[case]['bottle_neck']
n_clstr = params[case]['nbr_clusters']
dynamic = params[case]['dynamic']
nfeat   = len(feat[case])

if not dynamic:
    sg = min(sgmax, int(max(mmax,nmax)/2))
    lr = lrmax/2

if acode:
    batch_size = params[case]['batch size']
    num_epochs = params[case]['nepochs']

## Default SOM HP
m  = 7
n  = 9
sg = 5.0
lr = 0.5
## Default scalers
scaler = MinMaxScaler()
scaler_pca = MinMaxScaler()
scaler_aec = MinMaxScaler()

        
'''
    ------------------------------
    Load data if it already exists
    ------------------------------
'''

if mode=='load':
    with open(case+'.pkl', 'rb') as f:
        x = pickle.load(f)
        xpca = pickle.load(f)
        y_kms = pickle.load(f)
        y_gmm = pickle.load(f)
        y_kms_pca = pickle.load(f)
        y_gmm_pca = pickle.load(f)
        lr = pickle.load(f)
        sg = pickle.load(f)
        m = pickle.load(f)
        n = pickle.load(f)
        data = pickle.load(f)
    print('Pickled data re-loaded')
    ae = joblib.load(case+'-autoencoder.pkl')
    pcomp = joblib.load(case+'-pcacomponents.pkl')
    scaler = joblib.load(case+'-scaler.pkl')
    scaler_pca = joblib.load(case+'-scaler_pca.pkl')
    scaler_aec = joblib.load(case+'-scaler_aec.pkl')
    print('joblib models re-loaded')
    
    raw = data[feat[case]].values
    raw = scaler.transform(raw)

else:    
    ## Loading the data
    data, nulls = acedata(acedir, cols, ybeg, yend)
    if case=='Roberts':
        data=data['2002-11':'2004-05']
    print('Data set size after reading files:', len(data))
    data = aceaddextra(data, nulls, xcols=xcols, window='4H', center=False)
    print('Data set size after adding extras:', len(data))
    
    '''
        ----------------
        Data compression
        ----------------
    '''
    
    raw = data[feat[case]].values
    raw = scaler.fit_transform(raw)
    
    pcomp = None
    ae = None
    if acode:
        print('Autoencoder...')
        from autoencoder import autoencoder
        nodes = params[case]['nodes']
        ae = autoencoder(nodes)
        print('Autoencoder fit...')
        L, T = ae.fit(torch.Tensor(raw), batch_size, num_epochs)
        print('Autoencoder encode...')
        X = ae.encode(torch.Tensor(raw)).detach().numpy()
        x = scaler_aec.fit_transform(X)
        if pca:
            print('PCA for comparison...')
            pcomp = PCA(n_components=bneck, whiten=True)
            Xpca = pcomp.fit_transform(raw)
            xpca = scaler_pca.fit_transform(Xpca)
    else:
        if pca:
            print('PCA transormation...')
            pcomp = PCA(n_components=bneck, whiten=True)
            X = pcomp.fit_transform(raw)
            x = scaler_pca.fit_transform(X)
        else:
            print('No data reduction...')
            x = raw
    
    '''
        ---------------------------------------
        Perform classical clustering techniques
        ---------------------------------------
    '''
    
    if clustering:
        print('Clustering...')
        from sklearn import cluster, mixture
        from sklearn.neighbors import kneighbors_graph
        print('Loading clustering methods...')
        kms = cluster.MiniBatchKMeans(verbose=1, n_clusters=n_clstr, n_init=500)
        gmm = mixture.GaussianMixture(verbose=1, n_components=n_clstr, covariance_type='full', n_init=500)
        
        print('Cluster by k-means...')
        y_kms = kms.fit_predict(x)
        print('Cluster by GMM...')
        y_gmm = gmm.fit_predict(x)
        if acode and pca:
            print('Cluster again but for PCA for comparison...')
            y_kms_pca = kms.fit_predict(xpca)
            print('Done with spc...')
            y_gmm_pca = gmm.fit_predict(xpca)
            
        data['class-kmeans'] = y_kms
        data['class-gmm'] = y_gmm
    
    '''
        ----------------------------
        Hyper-Parameter Optimization
        ----------------------------
    '''
    if optim:
        from som import *
        import optuna        
        def objective(trial):
            lr = trial.suggest_uniform('lr', 0.1, lrmax)
            sg = trial.suggest_uniform('sg', 4.0, sgmax)
            m = trial.suggest_int('m', 5, mmax)
            n = trial.suggest_int('n', 5, nmax)

            som = selfomap(x, m, n, 1000, sigma=sg, learning_rate=lr, init=init, dynamic=dynamic)
            QE = som.quantization_error(x)
            ncrit = 0.05*m/mmax
            mcrit = 0.05*n/nmax
            mncrit= 0.01*(abs(m-n))
            print('QE   :', QE)
            print('ncrit:', ncrit)
            print('mcrit:', mcrit)
            print('mncrt:', mncrit)
            return QE + ncrit + mcrit + mncrit

        study = optuna.create_study()
        study.optimize(objective, n_trials=100)
        for op in ['lr','sg','m','n']:
            exec(op+'='+str(study.best_params[op]))
            
if calculate_som:
    from som import *

    ## Run the model 
    print('SOM training...',m,n,lr,sg)
    som = selfomap(x, m, n, 20000, sigma=sg, learning_rate=lr, init=init, dynamic=dynamic)
    
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
        ---------------------
        Cluster the SOM nodes
        ---------------------
    '''
    from sklearn import cluster
    C1 = cluster.MiniBatchKMeans(n_clusters=n_clstr, n_init=500).fit(W.reshape(m*n,-1))
    C1 = np.array(C1.labels_)
    C1 = C1.reshape((m,n))
    data = som_addinfo(som, data, x, C1, 'class-som')
    bdry,_ = som_boundaries(C1, m, n)
    
    '''
        ----------------
        Plot the results
        ----------------
    '''
    
    ## Switch on/off plots
    plots_on = False
    plot_hitmap = plots_on         # Plots the SOM hit map
    plot_neighbors = plots_on      # Plots lines connecting neighbors in the maps and feature space
    plot_featurespace = plots_on   # Plots the feature space
    plot_components = plots_on     # Plot maps of the three components
    plot_features = plots_on       # plot maps of the corresponding features
    plot_datamean = plots_on
    plot_somclustering = plots_on
    plot_timeseries = plots_on
    plot_classbdy = plots_on
    
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
        map_plot(ax, dist, color, m, n, size=size, scale=3, cmap='inferno_r', lcolor='black')
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
        import matplotlib.colors as mcolors
        plt.figure()
        add_data = np.arange(m*n).reshape((m,n))
        add_name = 'node'
        finaldata = som_addinfo(som, data, x, add_data, add_name)
        hbin = plt.hexbin(x[:,0], x[:,1], norm=mcolors.PowerNorm(gamma=0.5), gridsize=30, cmap='cubehelix_r')  
        plt.colorbar(hbin)
        plt.scatter(W[:,:,0].flatten(), W[:,:,1].flatten(), c=color.reshape((m*n)), cmap='inferno_r', s=50, marker='.', label='nodes')
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
        plt_features('log_proton_speed')
        plt_features('log_O7to6')
        
    def plt_anyfeature_mean(ftr_name): 
        import matplotlib.colors as mcolors
        color = np.zeros((m, n))
        size=np.ones_like(hits)
        for x in range(m):
            for y in range(n):
                color[x,y] = data[ftr_name].iloc[wmix[x,y]].mean()
        vmax = data[ftr_name].max()
        vmin = data[ftr_name].min()
        color = np.nan_to_num(color)
        cbmin = color.min()
        cbmax = color.max()
        color = (color - color.min())/(color.max() - color.min())
        fig, ax = plt.subplots(1,1, figsize=(m/2,n/2))
        cmap='inferno_r'
        map_plot(ax, dist, color, m, n, size=size, scale=4, title=ftr_name+' mean', cmap=cmap)
        norm = mcolors.Normalize(vmin=vmax,vmax=vmin)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.subplots_adjust(right=0.75, left=0.1)
        cbar1 = fig.add_axes([0.8, 0.25, 0.05, 0.45])
        cb = plt.colorbar(sm, cax=cbar1)
        cb.ax.tick_params(labelsize='x-small')
    
    if plot_datamean:
        for f in data.columns:
            if (f.startswith('log_') or f.startswith('class')):
                plt_anyfeature_mean(f)

       
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
        
    if plot_somclustering:
        size  = np.ones_like(hits)
        color = C1
        cmin  = color.min()
        cmax  = color.max()
        color = (color - cmin) / (cmax - cmin)
    
        fig, ax = plt.subplots(1,1)
        cmap = plt.cm.get_cmap('jet', n_clstr)
        map_plot(ax, dist, color, m, n, usezero=False, size=size, scale=6, cmap=cmap, lcolor='black')
        plt.title('Clustering SOM')
        
        color = C2
        cmin  = color.min()
        cmax  = color.max()
        color = (color - cmin) / (cmax - cmin)
        fig, ax = plt.subplots(1,1)
        cmap = plt.cm.get_cmap('jet', n_clstr)
        map_plot(ax, dist, color, m, n, size=size, scale=6, cmap=cmap, lcolor='white')
        plt.title('Clustering SOM')
        
        color = C3
        cmin  = color.min()
        cmax  = color.max()
        color = (color - cmin) / (cmax - cmin)
        fig, ax = plt.subplots(1,1)
        cmap = plt.cm.get_cmap('jet', n_clstr)
        map_plot(ax, dist, color, m, n, size=size, scale=6, cmap=cmap, lcolor='white')
        plt.title('Clustering SOM')
        
    if plot_timeseries:
        beg = '2003-05-01'
        end = '2003-09-01'
        
        fig, ax = plt.subplots(6,1, sharex='all') #, gridspec_kw = {'height_ratios':[4, 4, 4, 1, 1, 4]})
        ax[0].set_xlim(pd.to_datetime(beg), pd.to_datetime(end))
        cmap = plt.cm.get_cmap('jet', n_clstr)
        ax[0].scatter(data[beg:end].index, data[beg:end]['proton_speed'], c=data[beg:end]['class-kmeans-8'], cmap=cmap, s=5)
        ax[1].scatter(data[beg:end].index, data[beg:end]['proton_speed'], c=data[beg:end]['class-agglo-8'], cmap=cmap, s=5)
        ax[2].scatter(data[beg:end].index, data[beg:end]['proton_speed'], c=data[beg:end]['class-birch-8'], cmap=cmap, s=5)
        
        ax[3].plot(data[beg:end]['Bgsm_x'], 'r-')
        ax[3].plot(data[beg:end]['Bgsm_y'], 'g-')
        
        ax[4].plot(data[beg:end]['Bgsm_z'], 'b-')
        
        ax[5].plot(data[beg:end]['log_O7to6'], 'r-')
        # ax2b = ax[2].twinx()
        ax[5].plot(np.log(6.008)-0.00578*data[beg:end]['proton_speed'], 'b-.')
        ax[5].hlines(np.log(0.145), beg, end, color='blue', linestyles='dashed')
        
    if plot_classbdy:
        fig, ax = plt.subplots(1,1)
        cmap = plt.cm.get_cmap('jet', n_clstr)
        from matplotlib_hex_map import matplotlib_hex_map as map_plot 
        map_plot(ax, bdry, color, m, n, usezero=True,  lcolor='black', size=np.ones((m,n)), scale=1, cmap=cmap)


'''
    ------------------------
    Generate the paper plots
    ------------------------
'''

if generate_paper_figures:
    plt.ioff()
    import paper_figures as pfig
    
    fig_path = outdir+case
    
    print('Plotting data coverage...')
    pfig.fig_datacoverage(data, cols, fname=fig_path+'/datacoverage.png')
    
    print('Plotting clouds of points...')
    if acode and pca:
        pfig.fig_dimreduc(data, xpca, x, n_clstr, cmap='jet_r', fname=fig_path+'/dimreduc.png')
        pfig.fig_clustering(data, xpca, x, y_kms, y_gmm, data['class-som'].values, y_kms_pca, y_gmm_pca, data['class-som'].values, n_clstr, cmap='jet', fname=fig_path+'/clustering.png')
    
    print('Plotting SOMs...')
    pfig.fig_maps(m, n, som, x, data, feat[case][0], 3, 3, hits, dist, bdry, W, C1, n_clstr, wmix, scaler, scaler_pca, scaler_aec, feat[case], pcomp=pcomp, ae=ae, fname=fig_path+'/maps.png')
    
    print('Plotting fingerprints...')
    pfig.fig_datarange(raw, fname=fig_path+'/datarange.png')
    
    print('Plotting class fingerprints...')
    pfig.fig_classesdatarange(data, feat[case], scaler, n_clstr, 'class-kmeans', [1,0,0], fname=fig_path+'/classesdatarange-kmeans.png')
    pfig.fig_classesdatarange(data, feat[case], scaler, n_clstr, 'class-gmm', [0,1,0], fname=fig_path+'/classesdatarange-gmm.png')
    pfig.fig_classesdatarange(data, feat[case], scaler, n_clstr, 'class-som', [0,0,1], fname=fig_path+'/classesdatarange-som.png')
    
    print('Plotting time series...')
    beg = '2003-05-01'
    end = '2003-09-01'
    pfig.fig_timeseries(data, beg, end, n_clstr, fname=fig_path+'/timeseries.png')
    
    print('Plotting features in the time series...')
    pfeat = ['log_O7to6',
             'C6to5',
             'proton_speed',
             'log_Sp',
             'log_Va',
             'log_Tratio',
             'sigmac',
             'sigmar',
             'FetoO',
             'Bmag_range']
    pfig.fig_tsfeatures(data, pfeat, 'class-kmeans', beg, end, n_clstr, fname=fig_path+'/tsfeatures-kmeans.png')
    pfig.fig_tsfeatures(data, pfeat, 'class-gmm', beg, end, n_clstr, fname=fig_path+'/tsfeatures-gmm.png')
    pfig.fig_tsfeatures(data, pfeat, 'class-som', beg, end, n_clstr, fname=fig_path+'/tsfeatures-som.png')
    
    print('Plotting maps of each component used for the SOM training...')
    for f in feat[case]:
        pfig.fig_componentmap(data, W, feat, nfeat, case, f, dist, bdry, hits, m, n, bneck, wmix, scaler, scaler_pca, scaler_aec, pca=pca, acode=acode, pcomp=pcomp, ae=ae, lcolor='white', fname=fig_path+'/comp-map-'+f+'.png')
    
    print('Plotting any other field not used for the SOM training...')
    # plotcols = ['proton_density',
    #             'proton_temp',
    #             'He4toprotons',
    #             'proton_speed',
    #             'nHe2',
    #             'vHe2',
    #             'vthHe2',
    #             'vthC5',
    #             'vthO6',
    #             'vthFe10',
    #             'avqC',
    #             'avqO',
    #             'Br',
    #             'Bt',
    #             'Bn',
    #             'Lambda',
    #             'Delta',
    #             'dBrms',
    #             'sigma_B']
    # for f in plotcols:
    #     pfig.fig_anyftmap(data, f, dist, np.ones_like(hits), m, n, wmix, lcolor='white', fname=fig_path+'/ftmap-'+f+'.png')
    
    print('Plotting solar wind type hits colored by features...')
    for colorby in ['log_O7to6','proton_speed','log_Sp','log_Va','log_Tratio']:
        for swclass in ['Xu_SW_type','Zhao_SW_type',]:
            smin = int(data[swclass].min())
            smax = int(data[swclass].max())+1
            for c in range(smin, smax):
                pfig.fig_swtypes(data, colorby, swclass, c, m, n, dist, bdry, wmix, fname=fig_path+'/SWtype-'+swclass+'-'+str(c)+'-'+colorby+'.png')
    
    print('Plotting the SOM clustering...')
    pfig.fig_classmap(C1, m, n, dist, hits , n_clstr, fname=fig_path+'/classmap.png')

'''
    Save ALL session variables
'''
if mode=='new':
    with open(case+'.pkl', 'wb') as f:
        pickle.dump(x, f)
        pickle.dump(xpca, f)
        pickle.dump(y_kms, f)
        pickle.dump(y_gmm, f)
        pickle.dump(y_kms_pca, f)
        pickle.dump(y_gmm_pca, f)
        pickle.dump(lr, f)
        pickle.dump(sg, f)
        pickle.dump(m, f)
        pickle.dump(n, f)
        pickle.dump(data, f)
        
    print('Data pickled to:', case+'.pkl')
    joblib.dump(ae    , case+'-autoencoder.pkl')
    joblib.dump(pcomp , case+'-pcacomponents.pkl')
    joblib.dump(scaler, case+'-scaler.pkl')
    joblib.dump(scaler_pca, case+'-scaler_pca.pkl')
    joblib.dump(scaler_aec, case+'-scaler_aec.pkl')
    print('Autoencoder and slcaers joblib-ed to mutiple files.')
