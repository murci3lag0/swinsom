#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Nov 22 18:34:21 2017

@author: amaya
'''

import pandas as pd
import numpy as np
import h5py

allacecols  = [
            'proton_density',
            'proton_temp',
            'He4toprotons',
            'proton_speed',
            'x_dot_RTN',
            'y_dot_RTN',
            'z_dot_RTN',
            'x_dot_GSE',
            'y_dot_GSE',
            'z_dot_GSE',
            'x_dot_GSM',
            'y_dot_GSM',
            'z_dot_GSM',
            'nHe2',
            'vHe2',
            'vC5',
            'vO6',
            'vFe10',
            'vthHe2',
            'vthC5',
            'vthO6',
            'vthFe10',
            'C6to5',
            'O7to6',
            'avqC',
            'avqO',
            'avqFe',
            'FetoO',
            'Br',
            'Bt',
            'Bn',
            'Bgse_x',
            'Bgse_y',
            'Bgse_z',
            'Bgsm_x',
            'Bgsm_y',
            'Bgsm_z',
            'Bmag',
            'Lambda',
            'Delta',
            'dBrms',
            'sigma_B']

def acereaddata(acedir, ybeg, yend, cols):
    
    for elem in ['year','day','hr']:
        if elem not in cols: cols.append(elem)
        
    raw = pd.DataFrame()
        
    for i in range(ybeg, yend+1):
        fname = acedir+'/multi_data_1hr_year'+str(i)+'.h5'
        print("Reading: ", fname)
        new=h5py.File(fname, 'r')
        new=np.array(new['/VG_MULTI_data_1hr/MULTI_data_1hr']).byteswap().newbyteorder()
        new=pd.DataFrame(new)

        new['Datetime'] = pd.to_datetime(new['year'].apply('{:0>4}'.format)+' '
               + new['day'].apply('{:0>3}'.format)+' '
               + new['hr'].apply('{:0>2}'.format),
               format='%Y %j %H', errors='ignore')
        new = new.set_index('Datetime')        
        raw = pd.concat([raw, new])
    
    cols.remove('year')
    cols.remove('day')
    cols.remove('hr')
    
    return raw[cols]

def acedata(acedir, cols, ybeg, yend):
    
    cols_needed = ['proton_speed','proton_density','O7to6','x_dot_GSM','y_dot_GSM','z_dot_GSM','Bgsm_x','Bgsm_y','Bgsm_z','Bmag']
    for elem in cols_needed:
        if elem not in cols: cols.append(elem)

    data = acereaddata(acedir, ybeg, yend, cols)

    nulls = pd.DataFrame([])
    nulls['Null values'] = pd.Series()
        
    for i in cols:
        if i.endswith('_qual') or i=='SW_type':
            nulls.loc[i] = [-1]
        else:
            nulls.loc[i] = [-9999.9]

    #Delete nulls
    for c in cols:
        data = data[data[c]!=nulls.loc[c][0]]
        
    #Keep only good quality data
    for e in data.columns:
        if e.startswith('qf_'):
            data = data[data[e]==0]

    return data, nulls

def aceaddextra(data, nulls, xcols, window=5, center=False):
 
    if 'SW_type' in xcols:
        '''
            see Zhao, L., Zurbuchen, T. H., & Fisk, L. A. (2009).
            Global distribution of the solar wind during solar cycle 23: ACE 
            observations. Geophysical research letters, 36(14).
            1: Coronal hole
            2: ICME
            4: Non-coronal hole
        '''
        data['SW_type']=4
        data.loc[data.O7to6<0.145,'SW_type'] = 1 
        data.loc[data.O7to6>6.008*np.exp(-0.00578*data.proton_speed),'SW_type'] = 2

    if 'Ma' in xcols:
        Va = 21.82915036515064 * data['Bmag'] / np.sqrt(data['proton_density'])
        data['Ma'] = data['proton_speed']/Va
        
    if (('sigmac' in xcols) or ('sigmar' in xcols)):
        V = data[['x_dot_GSM','y_dot_GSM','z_dot_GSM']]
        B = 21.82915036515064 * data[['Bgsm_x','Bgsm_y','Bgsm_z']].div(np.sqrt(data['proton_density']), axis=0)
      
        v  = V - V.rolling(window, center=center).mean()
        b  = B - B.rolling(window, center=center).mean()

        zp = v + b.values
        zn = v - b.values

        v2  = (v * v).sum(axis=1)
        b2  = (b * b).sum(axis=1)
        zp2 = (zp * zp).sum(axis=1)
        zn2 = (zn * zn).sum(axis=1)
        
        bdotv = (b * v.values).sum(axis=1).rolling(window, center=center).mean()
        bnorm = (b2 + v2.values).rolling(window, center=center).mean()

        zdotz = (zp * zn.values).sum(axis=1).rolling(window, center=center).mean()
        znorm = (zp2 + zn2.values).rolling(window, center=center).mean()

        sigc = 2 * bdotv / bnorm
        sigr = 2 * zdotz / znorm
        
        if 'sigmac' in xcols : data['sigmac'] = sigc
        if 'sigmar' in xcols : data['sigmar'] = sigr
    
    for end in ['min','max','mean','std','var']:
        func = getattr(pd.core.window.Rolling, end)
        for c in xcols:
            if c.endswith(end):
                var = c[:-len(end)-1]
                varfunc = func(data[var].rolling(window, center=center))
                data[c] = varfunc
                
    for end in ['delta']:
        fmax = pd.core.window.Rolling.max
        fmin = pd.core.window.Rolling.min
        for c in xcols:
            if c.endswith(end):
                var = c[:-len(end)-1]
                vmax = fmax(data[var].rolling(window, center=center))
                vmin = fmin(data[var].rolling(window, center=center))
                data[c] = vmax - vmin
            
    for end in ['acor']:
        func = lambda x: pd.Series(x).autocorr()
        for c in xcols:
            if c.endswith(end):
                var = c[:-len(end)-1]
                data[c] = data[var].rolling(window, center=center).apply(func, raw=False)
                
    data = data.dropna(axis=0)
    
    #Appending new nulls using the mutable argument reference
    for i in xcols:
        if i == 'SW_type':
            nulls.loc[i] = -1
        else:
            nulls.loc[i] = -9999.9
    
    return data

def addlogs(data, cols):
    for c in cols:
        data['log_'+c] = np.log((data[c] - data[c].min()) + 1.0)
    return data

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    cols = ['O7to6','FetoO','proton_temp','C6to5','Bmag']
    acedir = '/home/amaya/Workdir/MachineLearning/Data/ACE'
    ybeg = 2010
    yend = 2011
    
    data, nulls = acedata(acedir, cols, ybeg, yend)
    
    print(data.columns)
    
    xcols = ['sigmac','sigmar','SW_type','Bgsm_z_min','Bgsm_z_max','Bgsm_z_delta','Bgsm_z_acor']
    data = aceaddextra(data, nulls, xcols=xcols, window=7, center=False)
    
    data = addlogs(data, ['C6to5','O7to6','FetoO','proton_density','sigmar'])
    
    tdata = (data - data.min(axis=0))/(data.max(axis=0) - data.min(axis=0))
    
    pcols = ['C6to5','log_C6to5','O7to6','log_O7to6','FetoO','log_FetoO','proton_speed','proton_temp','proton_density','log_proton_density','Bmag','Bgsm_x','Bgsm_y','Bgsm_z','sigmac','sigmar','log_sigmar']
    tdata = np.array([tdata[c].values for c in pcols]).T
    plt.violinplot(tdata, showextrema=False)
    plt.boxplot(tdata, notch=True, showfliers=False, showmeans=True)
    plt.xticks(range(1,len(pcols)+1), labels=pcols)

