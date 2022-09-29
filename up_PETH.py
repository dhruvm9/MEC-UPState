#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 15:30:18 2022

@author: Dhruv
"""

#loading the dataset
import numpy as np 
import pandas as pd 
import scipy.io
from functions import *
from wrappers import *
import os, sys
import pynapple as nap 
import time 
import matplotlib.pyplot as plt 
import pickle
from scipy.stats import kendalltau, pearsonr, wilcoxon

data_directory = '/mnt/DataNibelungen/Dhruv/'
rwpath = '/mnt/DataNibelungen/Dhruv/MEC-UPState'
datasets = np.genfromtxt(os.path.join(rwpath,'MEC_dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')

allcoefs_up = []
allcoefs_dn = []
alldurs = []
updurs = []

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
    
###True Depth for A3707    
    # filepath = os.path.join(path, 'Analysis')
    # listdir    = os.listdir(filepath)
    
    # file = [f for f in listdir if 'CellDepth' in f]
    # celldepth = scipy.io.loadmat(os.path.join(filepath,file[0]))
    # truedepth = celldepth['cellDep']

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
    dd = cc[-0.05:0.25]
    tmp = pd.DataFrame(dd)
    tmp = tmp.rolling(window=4, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
    n = len(depth_final)
    t2 = np.argsort(depth_final)
    desc = t2[::-1][:n]
        
    
    finalRates = tmp[desc]
        
    fig, ax = plt.subplots()
    cax = ax.imshow(finalRates.T,extent=[-50 , 250, len(spikes) , 1],aspect = 'auto', cmap = 'inferno')
    cbar = fig.colorbar(cax, ticks=[0, finalRates.values.max()], label = 'Norm. Firing Rate')
    cbar.ax.set_yticklabels(['0', str(round(finalRates.values.max(),2))])
    plt.title('Event-related Xcorr, aligned to UP state onset_' + s)
    ax.set_ylabel('Neuron number')
    ax.set_xlabel('Lag (ms)')
    
    
    dur = (down_ep['end'] - down_ep['start']) * 1e3
    updur = (up_ep['end'] - up_ep['start']) * 1e3
    updurs.append(updur)
    alldurs.append(dur)

# alldurs = np.concatenate(alldurs)

#Aligned to DOWN onset

    cc2 = nap.compute_eventcorrelogram(spikes, nap.Tsd(down_ep['start'].values), binsize = 0.005, windowsize = 0.255, ep = sws_ep )    
    dd2 = cc2[-0.25:0.05]
    tmp2 = pd.DataFrame(dd2)
    tmp2 = tmp2.rolling(window=4, win_type='gaussian',center=True,min_periods=1).mean(std = 2)
    
    finalRates2 = tmp2[channelorder]
        
    fig, ax = plt.subplots()
    cax = ax.imshow(finalRates2.T,extent=[-250 , 50, len(spikes) , 1],aspect = 'auto', cmap = 'inferno')
    cbar = fig.colorbar(cax, ticks=[0, finalRates2.values.max()], label = 'Norm. Firing Rate')
    cbar.ax.set_yticklabels(['0', str(round(finalRates2.values.max(),2))])
    plt.title('Event-related Xcorr, aligned to DOWN state onset_' + s)
    ax.set_ylabel('Neuron number')
    ax.set_xlabel('Lag (ms)')
    
