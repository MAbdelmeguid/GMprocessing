#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 05:56:06 2023

@author: glavrent
"""
#load libraries
import time
import numpy as np
import pandas as pd
import scipy
import scipy.integrate
from obspy import Trace
import pykooh
import pyrotd

#scipy version
scipy_version = tuple(map(int, scipy.__version__.split('.')))

# %% Simulation Functions
# ======================================
# Read Time-Histories
def read_sim_seismogram_txt(fname=None):
    #read ground motion trace
    t = np.genfromtxt(fname, usecols=0)
    trace = np.genfromtxt(fname, usecols=1)
    
    return t, trace

def read_sim_seismogram_txt2(fname=None):
    #read ground motion trace
    data = np.genfromtxt(fname)
    t     = data[:,0]
    trace = data[:,1]
    
    return t, trace

def read_sim_seismogram_txt_oserror(fname=None):
    #read ground motion trace (IO Error proof)
    iter = 0
    while True:
        iter += 1
        try:
            data = np.genfromtxt(fname)
        except OSError as e:
            print(r'Unable to read %s'%fname)
            time.sleep(2*iter)
            if iter>5:
                raise(e)
        else:
            break
    
    #parse data
    t     = data[:,0]
    trace = data[:,1]
    
    return t, trace
    
# Filter Seismograms
def filter_seismogram(time, data, tr_inp=None, fmin=None,fmax=5.0, f_np=4):
    #define gm trace if not provided
    if tr_inp is None:
        tr = Trace()
        tr.data = data
        tr.stats.delta = np.round(time[1]-time[0], 5)
        tr.stats.starttime = time[0]
    else:
    	tr = tr_inp.copy()
    
    #filtering
    tr.filter(type='lowpass', freq=fmax, corners=f_np, zerophase=True)
    if fmin != None: tr.filter(type='highpass', freq=fmin, corners=f_np, zerophase=True)
    
    #return
    if tr_inp is None: 
    	return tr.data
    else:
    	return tr

# Source Information
def read_cmt_sol(fname_cmtsol, xy_offset=None):
    
    #read source file
    df_source = pd.read_csv(fname_cmtsol, delimiter=':', skiprows=1, index_col=0,
                            header=None, names=['property','value']).transpose()

    #convert data types to numeric values    
    df_source = df_source.astype({'time shift':float,'half duration':float,
                                  'latorUTM':float,'longorUTM':float,'depth':float,
                                  'Mpp':float,'Mtt':float,'Mtp':float,
                                  'Mrr':float,'Mrt':float,'Mrp':float})

    if not xy_offset is None:
        df_source.latorUTM  += xy_offset[1]
        df_source.longorUTM += xy_offset[0]
    
    return df_source

# Processing Ground Motions
def calc_acc2vel(time, acc):
    #define gm trace
    tr = Trace()
    tr.data = acc
    tr.stats.delta = np.round(np.diff(time).mean(), 5)
    tr.stats.starttime = time[0]
    
    #integrate to velocity
    tr.integrate()
    
    return tr.data

def differentiate_timehist(time, th1):
    #integrate time history
    th2 =  np.gradient(th1, time)

    return th2
    
def integrate_timehist(time, th1):
    #integrate time history
    if scipy_version >= (1, 14, 0):
        th2 = scipy.integrate.cumulative_trapezoid(th1, x=time, initial=0)
    else:
        th2 =  scipy.integrate.cumtrapz(th1, x=time, initial=0)

    return th2

def halfsine_taper(signal, taper_fraction=0.05):
    """
    Applies a half sine wave taper to the first and last `taper_fraction` of the signal.
    The signal starts and ends smoothly with zeros.

    Parameters:
    - signal (np.ndarray): Input 1D signal array.
    - taper_fraction (float): Fraction of the signal length to apply the taper (default is 0.05 for 5%).

    Returns:
    - tapered (np.ndarray): The taper function.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal must be a NumPy array.")
    
    N = len(signal)
    if N == 0:
        raise ValueError("Input signal is empty.")
    
    N_taper = int(np.floor(taper_fraction * N))
    
    if N_taper < 1:
        raise ValueError("Signal too short to apply the specified taper fraction.")
    
    # Create the taper using half sine waves
    taper = np.ones(N)
    
    # First taper_fraction of the signal
    t = np.linspace(0, np.pi / 2, N_taper)
    taper_start = np.sin(t * 2 - np.pi/2)  # sin(pi * t / N_taper)
    taper[:N_taper] = taper_start
    
    # Last taper_fraction of the signal
    taper_end = np.sin(t[::-1] * 2 - np.pi/2)  # reverse the taper
    taper[-N_taper:] = taper_end
    
    taper = 0.5 + 0.5*taper
    
    return taper

def halfsine_taperend(signal, taper_fraction=0.05):
    """
    Applies a half sine wave taper to the last `taper_fraction` of the signal.
    The signal starts and ends smoothly with zeros.

    Parameters:
    - signal (np.ndarray): Input 1D signal array.
    - taper_fraction (float): Fraction of the signal length to apply the taper (default is 0.05 for 5%).

    Returns:
    - tapered (np.ndarray): The taper function.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal must be a NumPy array.")
    
    N = len(signal)
    if N == 0:
        raise ValueError("Input signal is empty.")
    
    N_taper = int(np.floor(taper_fraction * N))
    
    if N_taper < 1:
        raise ValueError("Signal too short to apply the specified taper fraction.")
    
    # Create the taper using half sine waves
    taper = np.ones(N)
    
    # Last taper_fraction of the signal
    t = np.linspace(0, np.pi / 2, N_taper)
    taper_end = np.sin(t[::-1] * 2 - np.pi/2)  # reverse the taper
    taper[-N_taper:] = taper_end
    
    taper = 0.5 + 0.5*taper
    
    return taper

def calc_ai(time, acc, grav=9.80665):
    
    #calculate arias instensity
    ai = np.pi/(2*grav) * np.trapz(y=acc**2, x=time)
       
    return ai

def calc_dur(time, acc, dur_s=0.05, dur_e=0.75):
    #calculate arias instensity
    if scipy_version >= (1, 14, 0):
        ai_norm = scipy.integrate.cumulative_trapezoid(acc**2, x=time, initial=0)
    else:
        ai_norm = scipy.integrate.cumtrapz(acc**2, x=time, initial=0)
    ai_norm = ai_norm/ai_norm[-1]

    #start and end duration
    dur_s = time[ np.argmin( np.abs(ai_norm-dur_s) ) ]
    dur_e = time[ np.argmin( np.abs(ai_norm-dur_e) ) ]    
           
    return float(dur_e - dur_s)

def calc_dur2(time, acc, acc2, dur_s=0.05, dur_e=0.75):
    #calculate arias instensity
    if scipy_version >= (1, 14, 0):
        ai_normx = scipy.integrate.cumulative_trapezoid(acc**2, x=time, initial=0)
        ai_normy = scipy.integrate.cumulative_trapezoid(acc2**2, x=time, initial=0)
    else:
        ai_normx = scipy.integrate.cumtrapz(acc**2, x=time, initial=0)
        ai_normy = scipy.integrate.cumtrapz(acc2**2, x=time, initial=0)

    intensity_gm = np.sqrt(np.maximum(ai_normx*ai_normy, 0))
    intensity_norm = intensity_gm/intensity_gm[-1]

    #start and end duration
    dur_s = time[ np.argmin( np.abs(intensity_norm-dur_s) ) ]
    dur_e = time[ np.argmin( np.abs(intensity_norm-dur_e) ) ]    
           
    return float(dur_e - dur_s)

def calc_fas(time, acc, freq=np.logspace(-1, 2, 91)):
    #time step 
    dt = np.round(np.diff(time).mean(), 5)
    
    #power-mean spectrum
    fft_fas  = np.abs(np.fft.rfft(acc))*dt
    fft_freq = np.fft.rfftfreq(len(time), d=dt)
    fft_fas  = fft_fas[fft_freq>0] 
    fft_freq = fft_freq[fft_freq>0]
    
    #interpolated fas
    fas = np.exp( np.interp(np.log(freq), np.log(fft_freq), np.log(fft_fas)) )

    return freq, fas
    
def calc_eas(time, acc, freq=np.logspace(-1, 2, 91), bw=0.0333):
    #time step 
    dt = np.round(np.diff(time).mean(), 5)
    
    #power-mean spectrum
    fft_fas  = np.abs(np.fft.rfft(acc))*dt
    fft_freq = np.fft.rfftfreq(len(time), d=dt)
    fft_fas  = fft_fas[fft_freq>0] 
    fft_freq = fft_freq[fft_freq>0]
    
    #KO - smoothed
    #eas = pykooh.smooth(freq, fft_freq, fft_fas, 2*np.pi/bw)
    eas = pykooh.smooth(fft_freq, fft_freq, fft_fas, 2*np.pi/bw)
    eas = np.exp( np.interp(np.log(freq), np.log(fft_freq), np.log(eas)) )

    return freq, eas

def calc_psa(time, acc, freq=np.logspace(-1, 2, 91), damp=0.05):
    #time step 
    dt = np.round(np.diff(time).mean(), 5)
    
    #compute response spectrum
    psa = pyrotd.calc_spec_accels(dt, acc, freq, damp).spec_accel

    return freq, psa


def calc_fas_horiz(time, acc_horiz, freq=np.logspace(-1, 2, 91)):
    #time step 
    dt = np.round(np.diff(time).mean(), 5)
    
    #power-mean spectrum
    fft_fas  = np.linalg.norm( [np.abs(np.fft.rfft(a))*dt for a in acc_horiz], axis=0) / np.sqrt(2)
    fft_freq = np.fft.rfftfreq(len(time), d=dt)
    fft_fas  = fft_fas[fft_freq>0] 
    fft_freq = fft_freq[fft_freq>0]
    
    #interpolated fas
    fas = np.exp( np.interp(np.log(freq), np.log(fft_freq), np.log(fft_fas)) )

    return freq, fas
    
def calc_eas_horiz_med(time, acc_horiz, freq=np.logspace(-1, 2, 91), bw=0.0333):
    #time step 
    dt = np.round(np.diff(time).mean(), 5)
    
    #power-mean spectrum
    fft_fas  = np.linalg.norm( [np.abs(np.fft.rfft(a))*dt for a in acc_horiz], axis=0) / np.sqrt(2)
    fft_freq = np.fft.rfftfreq(len(time), d=dt)
    fft_fas  = fft_fas[fft_freq>0] 
    fft_freq = fft_freq[fft_freq>0]
    
    #KO - smoothed
    #eas = pykooh.smooth(freq, fft_freq, fft_fas, 2*np.pi/bw)
    eas = pykooh.smooth(fft_freq, fft_freq, fft_fas, 2*np.pi/bw)
    eas = np.exp( np.interp(np.log(freq), np.log(fft_freq), np.log(eas)) )

    return freq, eas

def calc_eas_horiz2_med(time, acc_horiz, freq=np.logspace(-1, 2, 91), bw=0.0333):
    #time step 
    dt = np.round(np.diff(time).mean(), 5)

    #concat acceleration time histories
    acc_horiz = np.column_stack(acc_horiz)

    #power-mean spectrum
    fft_fas = np.fft.rfft(acc_horiz*dt, axis=0)
    fft_freq = np.fft.rfftfreq(acc_horiz.shape[0], d=dt)

    #KO - smoothed
    eas_freq, eas = pykooh.effective_ampl(fft_freq, fft_fas[:, 0], fft_fas[:, 1], missing='nan')
    eas = np.exp( np.interp(np.log(freq), np.log(eas_freq), np.log(eas)) )

    return freq, eas

# def calc_psa_horiz_quantile(time, acc_horiz, freq=np.logspace(-1, 2, 91), damp=0.05, quantile=0.5, return_angle=False):
    
#     #time step 
#     dt = np.diff(time).mean()
    
#     #compute response spectrum
#     psa_rotd = pyrotd.calc_rotated_spec_accels(dt, acc_horiz[0], acc_horiz[1], 
#                                                freq, damp, percentiles=[100*quantile])

#     if return_angle is False:
#         return freq, psa_rotd.spec_accel
#     else:
#         return freq, psa_rotd.spec_accel, np.deg2rad(psa_rotd.angle)
    
def calc_psa_horiz_quantile(time, acc_horiz, freq=np.logspace(-1, 2, 91), damp=0.05, quantile=0.5, return_angle=False):
    #time step 
    dt = np.diff(time).mean()
    
    #compute fourier amplitude spectra
    fourier_amps = [np.fft.rfft(acc_horiz[0]), np.fft.rfft(acc_horiz[1])]
    freqs = np.linspace(0, 1.0 / (2 * dt), num=fourier_amps[0].size)
    angles = np.arange(0, 180, step=1)
    percentiles = [100*quantile]
    
    #compute rotated response spectrum without parallelization
    groups = [
        pyrotd.calc_rotated_oscillator_resp(
            angles,
            percentiles,
            freqs,
            fourier_amps,
            damp,
            f,
            max_freq_ratio=5,
            osc_type='psa',
            method='optimized'
        )
        for f in freq
    ]
    records = [g for group in groups for g in group]
    
    #organize results
    rotated_resp = np.rec.fromrecords(records, names="osc_freq,percentile,spec_accel,angle")

    if return_angle is False:
        return freq, rotated_resp.spec_accel
    else:
        return freq, rotated_resp.spec_accel, np.deg2rad(rotated_resp.angle)
    
def calc_psa_horiz_med(time, acc_horiz, freq=np.logspace(-1, 2, 91), damp=0.05, return_angle=False):
    
    return calc_psa_horiz_quantile(time, acc_horiz, freq=freq, damp=damp, quantile=0.5, return_angle=return_angle)

def calc_psa_horiz_max(time, acc_horiz, freq=np.logspace(-1, 2, 91), damp=0.05, return_angle=False):
    
    return calc_psa_horiz_quantile(time, acc_horiz, freq=freq, damp=damp, quantile=1.0, return_angle=return_angle)

