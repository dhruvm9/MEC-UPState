#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:37:29 2022

@author: dhruv
"""

#import libraries
import numpy as np
import pandas as pd
from pylab import *
import os,sys
import pynapple as nap
from functions import *
from wrappers import * 
from scipy.signal import hilbert

s = 'B0707-220816'

#loading general info
data_directory = '/mnt/DataNibelungen/Dhruv/B0707-220816'
# datasets = np.loadtxt(os.path.join(data_directory,'dataset_Hor_DM.list'), delimiter = '\n', dtype = str, comments = '#')
#datasets = np.loadtxt(os.path.join(data_directory,'dataset_test.list'), delimiter = '\n', dtype = str, comments = '#')
rwpath = '/mnt/DataNibelungen/Dhruv/MEC-UPState'

for s in datasets:
    print(s)
    name = s.split('/')[-1]
    path = os.path.join(data_directory, s)
    rawpath = os.path.join(rwpath,s)

############################################################################################### 
    # LOADING DATA
###############################################################################################
    data = nap.load_session(data_directory, 'neurosuite')
    data.load_neurosuite_xml(data_directory)
    channelorder = data.group_to_channel[0]
    spikes = data.spikes
    epochs = data.epochs


#fishing out the sleep epoch
    sleep_ep = epochs['sleep']

#get acceleration data
    acceleration = loadAuxiliary(data_directory, 1, fs = 20000) #second argument = 1, third argument is fs in some cases

#make new sleep epoch
    newsleep_ep = refineSleepFromAccel(acceleration, sleep_ep)

#Load LFP
    lfp = data.load_lfp(channel = int(channelorder[0]))
    downsample = 5
    lfp = lfp[::downsample]
    
    lfp_filt_theta = nap.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 4, 12, 1250/5, 2))
    power_theta = nap.Tsd(lfp_filt_theta.index.values, np.abs(hilbert(lfp_filt_theta.values)))
    tmp = power_theta.as_series()
    power_theta = tmp.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=80)

    lfp_filt_delta = nap.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 0.5, 4, 1250/5, 2))
    power_delta = nap.Tsd(lfp_filt_delta.index.values, np.abs(hilbert(lfp_filt_delta.values)))
    tmp = power_delta.as_series()
    power_delta = tmp.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=80)

    ratio = nap.Tsd(t = power_theta.index.values, d = np.log(power_theta.values/power_delta.values))
    tmp = ratio.as_series()
    ratio2 = tmp.rolling(window=10000,win_type='gaussian',center=True,min_periods=1).mean(std=200)
    ratio2 = nap.Tsd(t = ratio2.index.values, d = ratio2.values)

    index = (ratio2 > 0).values*1.0
    start_cand = np.where((index[1:] - index[0:-1]) == 1)[0]+1
    end_cand = np.where((index[1:] - index[0:-1]) == -1)[0]
    if end_cand[0] < start_cand[0]: end_cand = end_cand[1:]    
    if end_cand[-1] < start_cand[-1]: start_cand = start_cand[0:-1]
    tmp = np.where(end_cand != start_cand)
    start_cand = ratio2.index.values[start_cand[tmp]]
    end_cand = ratio2.index.values[end_cand[tmp]]

    good_ep = nap.IntervalSet(start_cand, end_cand)
    good_ep = newsleep_ep.intersect(good_ep)
    good_ep = good_ep.merge_close_intervals(10, time_units = 's')
    good_ep = good_ep.drop_short_intervals(20, time_units = 's')
    good_ep = good_ep.reset_index(drop=True)
    theta_rem_ep = good_ep
    sws_ep = newsleep_ep.set_diff(theta_rem_ep)
    sws_ep = sws_ep.merge_close_intervals(0).drop_short_intervals(0)

    data.write_neuroscope_intervals('.new_sws.evt' , sws_ep, 'SWS')
    data.write_neuroscope_intervals('.new_rem.evt' , theta_rem_ep, 'Theta')
 
    figure()
    ax = subplot(311)
    plt.title('LFP trace')
    [plot(lfp.restrict(sws_ep.loc[[i]]), color = 'blue') for i in sws_ep.index]
    plot(lfp_filt_delta.restrict(sws_ep), color = 'orange')
    subplot(312, sharex = ax)
    plt.title('Theta/Delta ratio')
    [plot(ratio.restrict(sws_ep.loc[[i]]), color = 'blue') for i in sws_ep.index]
    plot(nap.Tsd(ratio2).restrict(sws_ep), color = 'orange')
    axhline(0)
    subplot(313, sharex = ax)
    plt.title('Acceleration')
    plot(acceleration[0].restrict(sws_ep))
    show()












