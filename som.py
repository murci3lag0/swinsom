#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:06:31 2019

@author: amaya
"""

import numpy as np

from collections import defaultdict
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

def selfomap(data, nrow, ncol, niter,
          neighborhood_function='gaussian', 
          sigma = 2.0, 
          learning_rate=2.0, 
          random_seed=123,
          pcainit=True,
          verbose=True,
          seed=123,
          dynamic=False):
    
    som = MiniSom(x = nrow,
                  y = ncol,
                  input_len = data.shape[1],
                  neighborhood_function=neighborhood_function, 
                  sigma = sigma, 
                  learning_rate=learning_rate, 
                  random_seed=random_seed)
    
    MiniSom._max = 0
    
    if (pcainit):
        som.pca_weights_init(data)
    else:
        som.random_weights_init(data)
    if (neighborhood_function!='gaussian'):
        raise ValueError('Distance function can only be gaussian') 
    
    def fully_random_weights_init(data):
        return
        
    def _hexaneigfunc(c, sigma):
        xx, yy = np.meshgrid(som._neigx, som._neigy)
        xx = xx.astype(float)
        yy = yy.astype(float)
        xx[::2] -= 0.5
        d = 2*np.pi*sigma*sigma
        ax = np.exp(-np.power(xx-c[0], 2)/d)
        ay = np.exp(-np.power(yy-c[1], 2)/d)
        return (ax*ay).T
    
    def _win_map_index(data):
        """Returns a dictionary wm where wm[(i,j)] is the index of
        all the patterns that have been mapped in the position i,j."""
        winidx = defaultdict(list)
        for idx, x in enumerate(data):
            winidx[som.winner(x)].append(idx)
        return winidx
    
    def _dynupdate(self, x, win, t, max_iteration):
        D = ((x - self._weights)**2).sum(axis=-1)
        self._max = max(D.max(), self._max)
        d = np.sqrt(D/self._max)
        sig = self._sigma * d[win]
        eta = self._learning_rate * d
        g = self.neighborhood(win, sig+1e-7)*eta
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += np.einsum('ij, ijk->ijk', g, x-self._weights)
     
    #MiniSom.update = _dynupdate
    som.neighborhood = _hexaneigfunc
    som.win_map_index = _win_map_index

    som.train_random(data=data, num_iteration=niter, verbose=True)
    return som

def som_distances(som):
    um = np.zeros((som.get_weights().shape[0], som.get_weights().shape[1], 6))
    W = som.get_weights()
    norm = lambda x : np.sqrt(np.dot(x, x.T))
    even = lambda x: x%2 == 0
    ii = [[1, 1, 1, 0, -1, 0],[0, 1, 0, -1, -1, -1]]
    jj = [[1, 0, -1, -1, 0, 1],[1, 0, -1, -1, 0, 1]]
    for x in range(W.shape[0]):
        for y in range(W.shape[1]):
            w_2 = W[x,y]
            e = 1 if (even(y)) else 0
            for k,(i,j) in enumerate(zip(ii[e],jj[e])):
                if (x+i>=0 and x+i<W.shape[0] and y+j>=0 and y+j<W.shape[1]):
                    w_1 = W[x+i, y+j]
                    um[x,y,k] = norm(w_1-w_2)
        
    return um

def som_hits(som, data, m, n, log=False):
    print("Data shape: ", data.shape)
    hits = np.zeros((m, n))
    hitmap = som.win_map(data)
    for pos, val in hitmap.items():
        hits[pos[0], pos[1]] = len(val)

    if log is True: hits = np.log(hits+1)

    scaler = MinMaxScaler()
    hits = scaler.fit_transform(hits)
    hits = np.clip(hits, a_min=0.01, a_max=0.999)
    return hits

def som_colorize(som, data, m, n, somcols, cols, log=False):
    print("Data shape: ", data.shape)
    print("Search data shape:", data.values.shape)
    hits = np.zeros((m, n))
    color = np.zeros((m, n, len(cols)))
    hitmap = som.win_map(data[somcols].values)
    for pos, val in hitmap.items():
        hits[pos[0], pos[1]] = len(val)
        print("val:" , val)
        color[pos[0], pos[1]] = val[cols]
        
    color /= hits

    if log is True: hits = np.log(hits+1)

    scaler = MinMaxScaler()
    hits = scaler.fit_transform(hits)
    hits = np.clip(hits, a_min=0.1, a_max=0.999)
    return hits, color

def som_colortest(som_m, som_n, test='primary', seed=123):
    np.random.seed(seed)
    print("Performing test: "+test)
    if test=='rainbow':
        test_data = [[148, 0, 211],
                     [75, 0, 130],
                     [0, 0, 255],
                     [0, 255, 0],
                     [255, 255, 0],
                     [255, 127, 0],
                     [255, 0 , 0]]

    if test=='random':
        test_data = []
        for i in range(10000):
            test_data.append(np.random.random_integers(0, 255, 3))

    if test=='primary':
        test_data = []
        mean = [150,100,100]
        cov = [[50,0,0],[0,100,0],[0,0,100]]
        test_data.extend(np.random.multivariate_normal(mean, cov, 2000))
        mean = [100,100,150]
        cov = [[100,0,0],[0,50,0],[0,0,100]]
        test_data.extend(np.random.multivariate_normal(mean, cov, 2000))
        mean = [100,150,100]
        cov = [[100,0,0],[0,100,0],[0,0,50]]
        test_data.extend(np.random.multivariate_normal(mean, cov, 2000))
    
    scaler = MinMaxScaler()
    som_data = scaler.fit_transform(test_data)
    som_model = selfomap(som_data, som_m, som_n, 5000)
    return som_model, som_data

def som_adddata(som, data, mapdata):
    data_added = []
    for i in range(len(data)):
        x, y = som.winner(data[i])
        data_added.append(mapdata[x,y])
    data_added = np.array(data_added)
    return data_added

def som_addinfo(som, df, data, mapdata, mapname):
    assert(som.get_weights().shape[0]==mapdata.shape[0])
    assert(som.get_weights().shape[1]==mapdata.shape[1])
    
    data_added = som_adddata(som, data, mapdata)
    df[mapname]=data_added
    return df

if __name__ == "__main__":
    from matplotlib_hex_map import matplotlib_hex_map as map_plot
    m = 7
    n = 9
    model, data = som_colortest(m, n)
    d = som_distances(model)
    hits = som_hits(model, data, m, n, log=False)
    wmi = model.win_map_index(data)

    color = model.get_weights()[:,:,:3]
    color = (color - color.min()) / (color.max() - color.min())
    
    map_plot(d, color, m, n, size=hits, scale=6)