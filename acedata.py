#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Nov 22 18:34:21 2017

@author: amaya
'''

import pandas as pd
import numpy as np

def acedata(acedir, ybeg, yend, cols):
    
    for elem in ['year','day','hr','min','sec']:
        if elem not in cols: cols.append(elem)
        
    raw = pd.DataFrame()
        
    for i in range(ybeg, yend+1):
        fname = acedir+'/ACE_SWICS_Data_'+str(i)+'.txt'
        print("Reading: ", fname)
        new=pd.read_csv(fname,
                        header=41,
                        delim_whitespace=True,
                        index_col=False,
                        comment='B',
                        usecols=cols)
        new.drop(0, inplace=True)
        
        new['Datetime'] = pd.to_datetime(new['year'].apply('{:0>4}'.format)+' '
               + new['day'].apply('{:0>3}'.format)+' '
               + new['hr'].apply('{:0>2}'.format)+' '
               + new['min'].apply('{:0>2}'.format)+' '
               + new['sec'].apply(np.round).apply('{:0>2}'.format)+' ',
               format='%Y %j %H %M %S', errors='ignore')
        new = new.set_index('Datetime')
        new.drop(['year','day','hr','min','sec'], axis=1, inplace=True)
        
        raw = pd.concat([raw, new])
    
    return raw

if __name__ == "__main__":
    cols = ['O7to6','SW_type','FetoO']
    data = acedata('/home/amaya/Workdir/MachineLearning/Data/ACE', 2005, 2010, cols)