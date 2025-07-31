#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 13:06:49 2023

@author: glavrent
"""
#load libraries
import numpy as np
from numpy.matlib import repmat

def geomean(iterable):
    return np.exp(np.log(iterable).mean())


def movingmean(y_array, x_array, x_bin, rm_nan=False):
    '''
    Moving Mean Statistics

    Parameters
    ----------
    y_array : np.array
        Response variable.
    x_array : np.array()
        Conditional variable.
    x_bin : np.array()
        Conditional variable bins.
    rm_nan : bool
        Flag remove nan value.

    Returns
    -------
    x_mid : np.array()
        Mid-point of conditional variable bins.
    y_mmed : np.array()
        Moving median.
    y_mmean : np.array()
        Moving mean.
    y_mstd : np.array()
        Moving standard deviation.
    y_m16prc : np.array()
        Moving 16th percentile.
    y_m84prc : np.array()
        Moving 84th percentile.
    '''
    
    #flaten input arrays
    if y_array.ndim == 2:
        x_array = repmat(x_array, y_array.shape[1], 1).T.flatten()
        y_array = y_array.flatten()
    
    #remove nan values
    if rm_nan:
        i_nan = np.isnan(y_array)
        x_array = x_array[~i_nan]
        y_array = y_array[~i_nan]
    
    #bins' mid point
    x_mid = np.array([(x_bin[j]+x_bin[j+1])/2  for j in range(len(x_bin)-1)])
    
    #binned residuals
    y_mmed   = np.full(len(x_mid), np.nan)
    y_mmean  = np.full(len(x_mid), np.nan)
    y_mstd   = np.full(len(x_mid), np.nan)
    y_m16prc = np.full(len(x_mid), np.nan)
    y_m84prc = np.full(len(x_mid), np.nan)
    
    #iterate over residual bins
    for k in range(len(x_mid)):
        #binned residuals
        i_bin = np.logical_and(x_array >= x_bin[k], x_array < x_bin[k+1])
        #summarize statistics
        y_mmed[k]   = np.median( y_array[i_bin] )
        y_mmean[k]  = np.mean( y_array[i_bin] )
        y_mstd[k]   = np.std( y_array[i_bin] )
        y_m16prc[k] = np.quantile( y_array[i_bin], 0.16) if i_bin.sum() else np.nan 
        y_m84prc[k] = np.quantile( y_array[i_bin], 0.84) if i_bin.sum() else np.nan
    
    #remove unavailable bins
    i_bin = ~np.isnan(y_mstd)
    x_mid    = x_mid[i_bin]
    y_mmed   = y_mmed[i_bin]
    y_mmean  = y_mmean[i_bin]
    y_mstd   = y_mstd[i_bin]
    y_m16prc = y_m16prc[i_bin]
    y_m84prc = y_m84prc[i_bin]
     
    return x_mid, y_mmed, y_mmean, y_mstd, y_m16prc, y_m84prc