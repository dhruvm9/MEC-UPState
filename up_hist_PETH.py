#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:11:09 2022

@author: Dhruv
"""

import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import os, sys
import pynapple as nap 
import time 
import matplotlib.pyplot as plt 
from scipy.stats import kendalltau, pearsonr, wilcoxon, mannwhitneyu
import pickle


data_directory = '/mnt/DataNibelungen/Dhruv/'
rwpath = '/mnt/DataNibelungen/Dhruv/MEC-UPState'
datasets = np.loadtxt(os.path.join(rwpath,'MEC_dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')


allcoefs_up = []
allcoefs_up_ex = []
allspeeds_up = []
allspeeds_up_ex = []
pvals = []
pvals_ex = []

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

    data = nap.load_session(path, 'neurosuite')
    data.load_neurosuite_xml(path)
    channelorder = data.group_to_channel[0]
    spikes = data.spikes
    epochs = data.epochs
    
    depth = np.arange(0, -800, -12.5)
    
   
    

# ############################################################################################### 
#     # LOAD UP AND DOWN STATE, NEW SWS AND NEW WAKE EPOCHS
# ###############################################################################################   
    
    file = os.path.join(path, name +'.sws.evt')
    sws_ep = data.read_neuroscope_intervals(name = 'SWS', path2file = file)
    
    file = os.path.join(path, name +'.evt.py.dow')
    down_ep = data.read_neuroscope_intervals(name = 'DOWN', path2file = file)
    
    file = os.path.join(path, name +'.evt.py.upp')
    up_ep = data.read_neuroscope_intervals(name = 'UP', path2file = file)
    
############################################################################################### 
    # COMPUTE EVENT CROSS CORRS
###############################################################################################  
 
    
    #_,maxch = data.load_mean_waveforms() 
    # pickle.dump(maxch,'maxch')
    
    f = open(os.path.join(path,'maxch.csv'),'rb')
    maxch = pickle.load(f)
    
    depth_final = []
     
    for i in range(len(spikes)):
        tmp = depth[maxch[i]] 
        depth_final.append(tmp)
    
    cc = nap.compute_eventcorrelogram(spikes, nap.Tsd(up_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = sws_ep )    
    dd = cc[0:0.105]
    tmp = pd.DataFrame(dd)
    tmp = tmp.rolling(window=4, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
         
    
    if len(dd.columns) > 0:
        indexplot = []
        depths_keeping = []
        
    for i in range(len(tmp.columns)):
        a = np.where(tmp.iloc[:,i] > 0.5)
            
        if len(a[0]) > 0:
            depths_keeping.append(depth_final[tmp.columns[i]])
            res = tmp.iloc[:,i].index[a]
            indexplot.append(res[0])
            
    y_est = np.zeros(len(indexplot))
    m, b = np.polyfit(indexplot, depths_keeping, 1)
   
         
    for i in range(len(indexplot)):
        y_est[i] = m*indexplot[i]
        
    coef, p = kendalltau(indexplot,depths_keeping)
    
    ###PLOTS
    plt.figure()
    plt.scatter(indexplot, depths_keeping, color = 'cornflowerblue', alpha = 0.8, label = 'R = ' + str(round(coef,4)))
    plt.plot(indexplot, y_est + b, color = 'cornflowerblue')
    plt.title('Bin where FR > 50% baseline rate_' + s)
    plt.ylabel('Depth from top of probe (um)')
    plt.yticks([0, -400, -800])
    plt.xlabel('Lag (s)')
    plt.legend(loc = 'upper right')
    
############################################################################################### 
    # FOR UD TRANSITION
###############################################################################################  
    
    cc2 = nap.compute_eventcorrelogram(spikes, nap.Tsd(down_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = sws_ep )    
    dd2 = cc2[-0.25:0.25]
    tmp2 = pd.DataFrame(dd2)
    tmp2 = tmp2.rolling(window=4, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    

    if len(dd2.columns) > 0:
        tmp = dd2.loc[0.005:] > 0.5
        
        tokeep = tmp.columns[tmp.sum(0) > 0]
        ends = np.array([tmp.index[np.where(tmp[i])[0][0]] for i in tokeep])
        es = pd.Series(index = tokeep, data = ends)
        
        tmp2 = dd2.loc[-0.1:-0.005] > 0.5
    
        tokeep2 = tmp2.columns[tmp2.sum(0) > 0]
        start = np.array([tmp2.index[np.where(tmp2[i])[0][-1]] for i in tokeep2])
        st = pd.Series(index = tokeep2, data = start)
            
        ix = np.intersect1d(tokeep,tokeep2)
        ix = [int(i) for i in ix]
        
        
        depths_keeping = np.array(depth_final)[ix]
        stk = st[ix]
        
    coef, p = kendalltau(stk,depths_keeping)
    
    y_est = np.zeros(len(stk))
    m, b = np.polyfit(stk, depths_keeping, 1)
   
       
    for i in range(len(stk)):
        y_est[i] = m*stk.values[i]
    
    ####PLOT        
    plt.figure()
    plt.scatter(stk.values, depths_keeping, color = 'cornflowerblue', alpha = 0.8, label = 'R = ' + str(round(coef,4)))
    plt.plot(stk, y_est + b, color = 'cornflowerblue')
    plt.title('Last bin before DOWN where FR > 50% baseline rate_' + s)
    plt.xlabel('Depth from top of probe (um)')
    plt.yticks([0, -400, -800])
    plt.xlabel('Lag (s)')
    plt.legend(loc = 'upper right')