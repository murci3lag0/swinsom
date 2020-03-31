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

new = pd.read_csv('HSCA_Shock_cat.csv', comment="#")
new['Datetime'] = pd.to_datetime(new['Year'].apply('{:0>4}'.format)+' '
       + new['Month'].apply('{:0>2}'.format)+' '
       + new['Day'].apply('{:0>2}'.format)+' '
       + new['UT'].apply('{:0>4}'.format),
       format='%Y %M %d %H%m', errors='ignore')
new = new.set_index('Datetime')

new['data'] = np.random.rand(len(new))
ax = new['data'].plot()
ax.axvspan(*mdates.datestr2num(["10/02/2003", "10/03/2005"]),facecolor='red', alpha=0.5)
