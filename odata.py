#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:01:18 2020

@author: amaya
"""

import pandas as pd
import numpy as np

omniheaders = ['year',
          'day',
          'hour',
          'Bartels',
          'IMF_spacecraft',
          'plasma_spacecraft',
          'IMF_av_npoints',
          'plasma_av_npoints',
          'av_|B|',
          '|av_B|',
          'lat_av_B_GSE',
          'lon_av_B_GSE',
          'Bx',
          'By_GSE',
          'Bz_GSE',
          'By_GSM',
          'Bz_GSM',
          'sigma_|B|',
          'sigma_B',
          'sigma_Bx',
          'sigma_By',
          'sigma_Bz',
          'Tp',
          'Np',
          'V_plasma',
          'phi_V_angle',
          'theta_V_angle',
          'Na/Np',
          'P_dyn',
          'sigma_Tp',
          'sigma_Np',
          'sigma_V',
          'sigma_phi_V',
          'sigma_theta_V',
          'sigma_Na/Np',
          'E',           
          'beta',
          'Ma',
          'Kp',
          'R',
          'Dst',
          'AE',
          'p_flux_>1MeV',
          'p_flux_>2MeV',
          'p_flux_>4MeV',
          'p_flux_>10MeV',
          'p_flux_>30MeV',
          'p_flux_>60MeV',
          'flag',
          'Ap',
          'f10.7',
          'PC',
          'AL',
          'AU',
          'M_ms']

omninulls = [
       None,
       None,
       None,
       9999,
       0,
       0,
       999,
       999,
       999.9,
       999.9,
       999.9,
       999.9,
       999.9,
       999.9,
       999.9,
       999.9,
       999.9,
       999.9,
       999.9,
       999.9,
       999.9,
       999.9,
       9999999.,
       999.9,
       9999.,
       999.9,
       999.9,
       9.999,
       99.99,
       9999999.,
       999.9,
       9999.,
       999.9,
       999.9,
       9.999,
       999.99,
       999.99,
       999.9,
       99,
       999,
       99999,
       9999,
       999999.99,
       99999.99,
       99999.99,
       99999.99,
       99999.99,
       99999.99,
       0,
       999,
       999.9,
       999.9,
       99999,
       99999,
       99.9]

def omnireaddata(fname, cols):   
    raw = pd.read_csv(fname,header=None,
                      names=omniheaders,
                      delim_whitespace=True,
                      index_col=False,
                      usecols=cols)
    raw['Datetime'] = pd.to_datetime(raw['year'].astype(str)+' '
       + raw['day'].astype(str)+' '
       + raw['hour'].astype(str),
       format='%Y %j %H')
    
    raw = raw.set_index('Datetime')
    cols.remove('year')
    cols.remove('day')
    cols.remove('hour')
    
    return raw[cols]

def omnigetnulls(data):
    nulls = pd.DataFrame(omninulls,
                     index=omniheaders, 
                     columns=['Null values'], 
                     dtype=object)
    return nulls.loc[data.columns]

def omnidata(omnidir, cols, ybeg, yend):
    cols_needed = ['year','day','hour','V_plasma','Ma','Tp','Np','Bx','By_GSM','Bz_GSM']
    for elem in cols_needed:
        if elem not in cols: cols.append(elem)
    
    fname = omnidir+"/omni2_"+str(ybeg)+"-"+str(yend)+".dat"    
    data = omnireaddata(fname, cols)
    
    return data
    
def omniorigin(raw, nulls):
    data = raw
        
    for c in raw.columns:
        data = data[data[c]!=nulls.loc[c][0]]
    
    k_b = 8.617333262145e-5
    
    Va=data['V_plasma']/data['Ma']
    Sp=data['Tp']*k_b/data['Np']**(2./3.)
    Texp=np.power(data['V_plasma']/258.0, 3.113)
    Tratio=Texp/(data['Tp']*k_b)
    
    data['Va']=Va
    data['Sp']=Sp
    data['Texp']=Texp
    data['Tratio']=Tratio
        
    e_plane=np.array([0.277  * np.log10(data['Sp'])+
                      0.055  * np.log10(data['Tratio'])+
                      1.83   < np.log10(data['Va'])]).T
    c_plane=np.array([-0.525 * np.log10(data['Tratio'])-
                      0.676  * np.log10(data['Va'])+
                      1.74   < np.log10(data['Sp'])]).T
    s_plane=np.array([-0.125 * np.log10(data['Tratio'])-
                      0.658  * np.log10(data['Va'])+
                      1.04   > np.log10(data['Sp'])]).T
    
    ejecta= pd.Series((v[0] for v in e_plane), name='bools').values
    chole = pd.Series((v[0] for v in c_plane), name='bools').values
    srev  = pd.Series((v[0] for v in s_plane), name='bools').values
    
    data.loc[ejecta, 'SW_type'] = 2
    data.loc[~ejecta&chole, 'SW_type'] = 1
    data.loc[~ejecta&srev, 'SW_type'] = 3
    data.loc[~ejecta&~chole&~srev, 'SW_type'] = 0
   
    raw = raw.merge(pd.DataFrame(data[['Va','Sp','Texp','Tratio','SW_type']]),
                    how='right',left_index=True, right_index=True)

    nulls.loc['Va']=nulls.loc['V_plasma'].values
    nulls.loc['Sp']=nulls.loc['V_plasma'].values
    nulls.loc['Texp']=nulls.loc['V_plasma'].values
    nulls.loc['Tratio']=nulls.loc['V_plasma'].values
    nulls.loc['SW_type']=-999
    raw['SW_type'].fillna(4, inplace=True)
    
    return raw

def omniderived(raw, nulls):
    data = raw
    xcols = ['BI','Viscous_function','Newell']
    cols_needed = ['V_plasma','Np','Bx','By_GSM','Bz_GSM']
    
    for c in cols_needed:
        data = data[data[c]!=nulls.loc[c][0]]

    B_GSM = np.sqrt(data['Bx']*data['Bx'] + 
                    data['By_GSM']*data['By_GSM'] + 
                    data['Bz_GSM']*data['Bz_GSM'])
    
    data = data[B_GSM>0]
    B_GSM = B_GSM[B_GSM>0]
    theta = np.arcsin(np.abs(data['By_GSM'])/B_GSM)
    
    BI=1e-4*data['V_plasma']**2 + 11.7*B_GSM*np.sin(theta/2.0)**3
    Vf=np.sqrt(data['Np'])*data['V_plasma']**2
    Nf=data['V_plasma']**(4./3.) * B_GSM**(2./3.) * np.sin(theta/2.0)**(8./3.)
 
    data['BI']=BI
    data['Viscous_function']=Vf
    data['Newell']=Nf
    
    raw = raw.merge(pd.DataFrame(data[xcols]),
                    how='right', left_index=True, right_index=True)
    
    for i in xcols:
        nulls.loc[i] = nulls.loc['V_plasma'].values
    
    return data

def omniaddshift(raw, nulls, ftime, ftdelta):  
    data = raw
    onulls = nulls.copy()
    for i in range(-ftime,-ftime-ftdelta,-1):
        data = data.join(raw.shift(i).add_suffix("_"+str(abs(i))))
        for c in onulls.index:
            nulls.loc[c+"_"+str(abs(i))] = onulls.loc[c]

    data = data.dropna(axis=0)
    return data

def omniaddextra(data, nulls, xcols, window=5, center=False):
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
                print('Calculating rolling autocorrelation for '+var+'...')
                data[c] = data[var].rolling(window, center=center).apply(func, raw=False)
                
    data = data.dropna(axis=0)

    #Appending new nulls using the mutable argument reference
    for i in xcols:
        nulls.loc[i] = -9999.9
    
    return data

if __name__ == "__main__":
    omnidir = "/home/amaya/Workdir/MachineLearning/Data/OMNI"
    cols = ['av_|B|',
          '|av_B|',
          'By_GSE',
          'Bz_GSE',
          'By_GSM',
          'Bz_GSM',
          'sigma_|B|',
          'sigma_B',
          'Tp',
          'Np',
          'V_plasma',
          'phi_V_angle',
          'theta_V_angle',
          'Na/Np',
          'P_dyn',
          'E',           
          'beta',
          'Ma',
          'Dst',
          'AE',
          'f10.7']
    
    data = omnidata(omnidir, cols, 2000, 2011)
    nulls = omnigetnulls(data)
    data = omniorigin(data, nulls)
    data = omniderived(data, nulls)
    
    xcols = ['Bz_GSE_delta','Bz_GSE_acor']
    data = omniaddextra(data, nulls, xcols)
