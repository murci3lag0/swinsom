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

shck = pd.read_csv('catalogs/HSCA_Shock_cat.csv', comment="#", parse_dates={'Datetime' : ['Year','Month','Day','UT']}, index_col='Datetime')

icme = pd.read_csv('catalogs/Richarson_Cane_ICME_cat.csv', comment="#", parse_dates=['Datetime'], index_col='Datetime')
icme['Start'] = pd.to_datetime(icme['Start'])
icme['End'] = pd.to_datetime(icme['End'])

aces = pd.read_csv('catalogs/ACE_Shocks_cat.csv', comment="#", parse_dates={'Datetime' : ['Month','Day','Year','Time']}, index_col='Datetime')

beg = '2002-05-01'
end = '2002-09-01'


fig, ax = plt.subplots(1,1)
ax.plot(icme[beg:end].index, icme[beg:end]['V_ICME'])
for idx in aces[beg:end].index:
    ax.axvline(idx, color='red')
