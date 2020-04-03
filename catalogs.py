#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:23:38 2020

@author: amaya
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

new = pd.read_csv('catalogs/HSCA_Shock_cat.csv', comment="#")
new['Datetime'] = pd.to_datetime(new['Year'].astype(str)+
                                 '-'+new['Month'].astype(str)+
                                 '-'+new['Day'].astype(str)+
                                 ' '+new['UT'].apply('{:0>4}'.format))
new['Datetime']=new['Datetime'].dt.round('1h')
new = new.set_index('Datetime')
new = new[~new.index.duplicated()]
new['Shock'] = 1

idx = pd.date_range('1998','2012', freq='H')
new=new.reindex(idx).fillna(0.0).rename_axis('Datetime').reset_index().set_index('Datetime')

new['data'] = np.random.rand(len(new))

beg = '2002-05-01'
end = '2002-09-01'
fig, ax = plt.subplots(1,1)
ax.plot(new[beg:end]['data'])

ax.axvline('2002-06-15', color='red')

##

dparser = lambda x : pd.datetime.strptime(x, '%Y/%m/%d %H%M')
icme = pd.read_csv('catalogs/Richarson_Cane_ICME_cat.csv', comment="#", date_parser=dparser, parse_dates=['Datetime'])
icme['Start'] = pd.to_datetime(icme['Start'])
icme['End'] = pd.to_datetime(icme['End'])
icme = icme.set_index('Datetime')
idx = pd.date_range('1998','2012', freq='H')
icmw=new.reindex(idx).fillna(0.0).rename_axis('Datetime').reset_index().set_index('Datetime')

fig, ax = plt.subplots(1,1)
ax.plot(icme[beg:end].index, icme[beg:end]['V_ICME'])
for i, x in enumerate(icme[beg:end]):
    print(i, x['Start'])
