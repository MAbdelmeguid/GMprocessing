#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 16:42:36 2024

@author: glavrent
"""

# libraries
# - - - - - - - - - -
#general libraries
import os
import pathlib
import sys
#arithmetic libraries
import numpy  as np
import pandas as pd
# multiprocessing libraries
import multiprocessing as mp
#plotting libraries
import matplotlib
import matplotlib.pyplot as plt
#user libraries
# - - - - - - - - - -
#general libraries
from pylib.general import str_replace_nth_int
from pylib.general import get_evenly_spaced_elements as evenly_spaced_elements
from pylib.general import polar_to_cartesian
from pylib.stats import geomean
from pylib.geometry import rectangle_to_point_distance
#plotting libraries
from pylib.contour_plots import plot_contour
from pylib.contour_plots import plot_contour_quiver
#ground motion libraries
from pylib.gm_processing import halfsine_taper, halfsine_taperend
from pylib.gm_processing import differentiate_timehist as diff_th
from pylib.gm_processing import integrate_timehist     as intg_th
from pylib.gm_processing import calc_ai   as calc_ai
from pylib.gm_processing import calc_dur  as calc_dur
from pylib.gm_processing import calc_dur2 as calc_dur2
from pylib.gm_processing import calc_fas  as calc_fas
from pylib.gm_processing import calc_eas  as calc_eas
from pylib.gm_processing import calc_psa  as calc_psa
from pylib.gm_processing import calc_fas_horiz  as calc_fas_horiz
from pylib.gm_processing import calc_eas_horiz_med as calc_eas_horiz_med
from pylib.gm_processing import calc_psa_horiz_med as calc_psa_horiz_med
from pylib.gm_processing import calc_psa_horiz_max as calc_psa_horiz_max

# user functions
# - - - - - - - - - -
def CalcSimMetaData(eqid, scnid, eqname, flag_ss, mag, sof, dip, ztor, rwidth, rup_geom, rup_hyp, staid, x, y, vs30, z1):
    """Create Simulation Metadata file"""

    #define rupture geometry
    rup_orgin = rup_geom[['x','y','z_top']].iloc[0,:].values
    rup_end   = rup_geom[['x','y','z_top']].iloc[-1,:].values
    rup_edge1 = rup_end - rup_orgin
    rup_edge2 = rup_geom[['x','y','z_bottom']].iloc[0,:].values - rup_orgin
    
    #rupture termination forward direction
    rup_fw = rup_orgin[:2] if np.linalg.norm(rup_orgin[:2]-rup_hyp[:2]) > np.linalg.norm(rup_orgin[:2]-rup_hyp[:2]) else rup_end[:2]
    
    #compute rupture distance and projection point
    out = [rectangle_to_point_distance(rup_orgin, rup_edge1, rup_edge2, np.array([xi, yi, 0.]))
           for xi, yi in zip(x, y)]
    
    #closest point
    eq_clstpt = np.array([o[1] for o in out]) 
        
    #rupture lenght
    rup_len = np.linalg.norm(rup_end[:2] - rup_orgin[:2])
        
    #rupture distance
    rrup = np.array([o[0] for o in out]) 
    
    #Joyner-Boore distance
    if abs(dip-90. <1e-3) and abs(ztor) <1e-3:
        rjb = rrup
    
    #hanging wall metrics
    if abs(dip-90. <1e-3) and abs(ztor) <1e-3:
        ry0 = np.array([np.dot(rup_edge1[:2]/rup_len, np.array([xi, yi])-eq_clstpt[j,:2]) for j, (xi, yi) in enumerate(zip(x, y))])
        ry0 = np.abs( ry0 )
        rx  = np.sqrt(rjb**2 - ry0**2)
   
    #hypcenter distance
    rhyp = np.array([np.linalg.norm(rup_hyp - np.array([xi, yi, 0.])) for (xi, yi) in zip(x, y)])
    
    #fault normal and parallel hypocenter distance
    rhy = np.array([np.dot(rup_edge1[:2]/rup_len, np.array([xi, yi])-rup_hyp[:2]) for (xi, yi) in zip(x, y)])
    rhx = np.sqrt(rhyp**2 - rhy**2)

    #propagation length
    prop_len = np.array([np.linalg.norm(eq_c[:2] - rup_hyp[:2]) for eq_c in eq_clstpt])
    
    #forward length
    fwd_len = prop_len
    #backward length    
    bwd_len = np.abs(rup_len - prop_len)
    
    #forward/backward location
    flag_fw = np.array([np.dot(rup_fw - rup_hyp[:2], eq_c[:2] - rup_hyp[:2]) > 0 for eq_c in eq_clstpt])
    
    #create flatfile
    df_fltfl = pd.DataFrame({'eqid':eqid, 'scenid':scnid, 'eqname':eqname, 'ss':flag_ss,
                             'mag':mag, 'rup_len':rup_len, 'sof': sof, 'dip': dip,
                             'ztor': ztor, 'rwidth': rwidth,
                             'eqx':rup_hyp[0], 'eqy':rup_hyp[1], 'eqz':rup_hyp[2], 
                             'eqcltx':eq_clstpt[:,0], 'eqclty':eq_clstpt[:,1], 'eqcltz':eq_clstpt[:,2], 
                             'stx':x, 'sty':y,
                             'rrup':rrup, 'rjb':rjb, 'rx':rx, 'ry0':ry0,
                             'rhyp':rhyp, 'rhx':rhx, 'rhy':rhy, 
                             'prplen':prop_len, 'fwdlen':fwd_len, 'bwdlen':bwd_len,
                             'fwrd': flag_fw,
                             'vs30':vs30, 'z1.0':z1})
                             
    return df_fltfl

def IMColNames(freq4psa, freq4eas):
    """Column Names For: psa, eas, and fas"""

    #psa and eas columns
    psa_col  = ['psa_f%.4fhz'%(f) for f in freq4psa]
    eas_col  = ['eas_f%.4fhz'%(f) for f in freq4eas]
    #fault normal and parallel components
    psaFN_col  = ['psaFN_f%.4fhz'%(f) for f in freq4psa]
    psaFP_col  = ['psaFP_f%.4fhz'%(f) for f in freq4psa]
    easFN_col  = ['easFN_f%.4fhz'%(f) for f in freq4eas]
    easFP_col  = ['easFP_f%.4fhz'%(f) for f in freq4eas]

    return psa_col, eas_col, psaFN_col, psaFP_col, easFN_col, easFP_col

def IMPSAMAXColNames(freq4psa):
    
    #psa-max and angle columns
    psaMAX_col       = ['psaMAX_f%.4fhz'%(f) for f in freq4psa]
    psaMAX_angle_col = ['psaMAXangle_f%.4fhz'%(f) for f in freq4psa]
    
    return psaMAX_col, psaMAX_angle_col

def IMPSAColNames(freq4psa):

    psa_col, _, psaFN_col, psaFP_col, _, _ = IMColNames(freq4psa, [0])
    
    return psa_col, psaFN_col, psaFP_col

def IMEASColNames(freq4eas):

    _, eas_col, _, _, easFN_col, easFP_col = IMColNames([0], freq4eas)
    
    return eas_col, easFN_col, easFP_col

def ReadSimVel(dir_sim):
    """Read Simulation Velocity Time Histories"""
    
    #time 
    time_array = np.load(os.path.join(dir_sim, 'T.npy'))
   
    #time step
    dt = np.mean(np.diff(time_array[1:]))
    #updated time array    
    time_array = dt * np.arange(len(time_array))
    
    #velocity signals
    vel_x = np.load(os.path.join(dir_sim, 'Vx.npy'))
    vel_y = np.load(os.path.join(dir_sim, 'Vy.npy'))
    #set begining and end of x-vel time history at zero
    vel_x[ 0,:] = .0
    # vel_x[-1,:] = .0
    #set begining and end of y-vel time history at zero
    vel_y[ 0,:] = .0
    # vel_y[-1,:] = .0
    
    #taper function
    taper = halfsine_taperend(time_array, taper_fraction=0.1)
    #apply taper
    vel_x *= taper[:,np.newaxis]
    vel_y *= taper[:,np.newaxis]
      
    #summarize ground motion
    sim_vel = [pd.DataFrame({'time':time_array, 'X':vel_x[:,j], 'Y':vel_y[:,j]}) for j in range(vel_x.shape[1])]
    
    return sim_vel

def CalcSimAcc(sim_vel, grav=9.80665):
    """Compute Simulation Acceleration Time Histories"""
    
    #compute accelaration time history for every station
    sim_acc = [pd.DataFrame({'time':v.time, 'X':grav**-1*diff_th(v.time, v.X), 'Y':grav**-1*diff_th(v.time, v.Y)}) 
               for v in sim_vel]
    
    return sim_acc

def ProcessGM(gm_acc, gm_vel, freq4psa, freq4eas, cmp=['X','Y'], grav=9.80665):
    """ """
    
    # fault normal/parallel
    # - - - - - - - - - - - - 
    #compute pga
    pga_fnfp = 0
    #compute pgv
    pgv_fnfp = 0
    #compute eas values
    eas_fnfp = 0
    #compute psa values
    psa_fnfp = 0
    #compute arias intenisty
    ai_fnfp  = 0
    #significant duration
    d575_fnfp = 0
    d595_fnfp = 0
    d575_mean = calc_dur2(gm_acc.time.values, gm_acc['X'].values, gm_acc['Y'].values, dur_s=0.05, dur_e=0.75)
    
    # geometric mean
    # - - - - - - - - - - - - 
    #compute pga
    pga = 0
    #compute pgv
    pgv = 0
    #compute eas values
    eas = 0
    #compute psa values
    psa = 0
    #compute arias intenisty
    ai = 0
    #significant duration
    d575 = 0
    d595 = 0

    # max direction
    # - - - - - - - - - - - - 
    #compute psa values
    psa_max = 0
    psa_max_angle = 0

    return (pga, pgv, ai, d575, d595, d575_mean, eas, psa, 
            pga_fnfp, pgv_fnfp, ai_fnfp, d575_fnfp, d595_fnfp, eas_fnfp, psa_fnfp,
            psa_max, psa_max_angle)

def ProcessGMMP(j, inp_arg, ret_dict):
    """Ground Motion Processing, Parallel Implementation"""
    
    #process ground motion
    out = ProcessGM(inp_arg['gm_acc'], inp_arg['gm_vel'],
                    inp_arg['freq4psa'], inp_arg['freq4eas'], 
                    inp_arg['cmp'], grav=inp_arg['grav'])
    
    #parse output
    #pga, pgv, ai, d575, d595, d5575_mean, eas, psa, pga_fnfp, pgv_fnfp, ai_fnfp, d575_fnfp, d595_fnfp, eas_fnfp, psa_fnfp, psa_max, psa_max_angle = out
    pga, pgv, ai, d575, d595, d575_mean, eas, psa = out[0:8]
    pga_fnfp, pgv_fnfp, ai_fnfp, d575_fnfp, d595_fnfp, eas_fnfp, psa_fnfp = out[8:15]
    psa_max, psa_max_angle = out[15:17]
    
    #summarize output arguments
    ret_dict[j] = {'idx':j, 'pga':pga, 'pgv':pgv, 'ai':ai, 'd575':d575, 'd595':d595, 'd575_mean':d575_mean, 'eas':eas, 'psa':psa,
                   'psa_max':psa_max, 'psa_max_angle':psa_max_angle,
                   'pga_fnfp': pga_fnfp, 'pgv_fnfp':pgv_fnfp, 'ai_fnfp':ai_fnfp,
                   'd575_fnfp':d575_fnfp, 'd595_fnfp':d595_fnfp, 'eas_fnfp':eas_fnfp, 'psa_fnfp':psa_fnfp}

def ProcessSimGMs(dir_sim, eqid, scnid,  eqname, flag_ss, mag, sof, dip, ztor, rwidth,
                  rup_geom, rup_hyp, vs30, z1,
                  freq4psa, freq4eas, 
                  cmp=['X','Y'], grav=9.80665, n_batch_size=64):

    #station coordinates
    x = np.load(os.path.join(dir_sim, 'x.npy'))
    y = np.load(os.path.join(dir_sim, 'y.npy'))
    #station id
    stid = np.arange(len(x))

    print("Process Simulation Ground Motions: %s"%eqname)
    #read velocity time histories
    sim_vel = ReadSimVel(dir_sim)
    #compute acceleration time histories
    sim_acc = CalcSimAcc(sim_vel)
    
    #create simulation flatfile
    df_fltfile_sim = CalcSimMetaData(eqid, scnid, eqname, flag_ss, mag, sof, dip, ztor, rwidth,
                                     rup_geom, rup_hyp, stid, x, y, vs30, z1)
    
    #gm flatfile column names
    psa_col, eas_col, psa_col_fn, psa_col_fp, eas_col_fn, eas_col_fp = IMColNames(freq4psa, freq4eas)
    psa_max_col, psa_max_angle_col = IMPSAMAXColNames(freq4psa)
    
    #initialize ground motion columns
    df_fltfile_sim.loc[:,['pga','pgv','ai','dur0.05-0.75','dur0.05-0.95','dur0.05-0.75_mean']] = np.nan
    df_fltfile_sim.loc[:,eas_col+psa_col]                                  = np.nan
    #maximum 
    df_fltfile_sim.loc[:,psa_max_col+psa_max_angle_col]                    = np.nan
    #fault normal/parallel component
    df_fltfile_sim.loc[:,['pga_fn','pgv_fn','ai_fn','dur0.05-0.75_fn','dur0.05-0.95_fn']] = np.nan
    df_fltfile_sim.loc[:,['pga_fp','pgv_fp','ai_fp','dur0.05-0.75_fp','dur0.05-0.95_fp']] = np.nan
    df_fltfile_sim.loc[:,eas_col_fn+psa_col_fn+eas_col_fp+psa_col_fp]                     = np.nan
  
    #process ground motions sequentially
    print("  store processed ground motions")
    for j, gm in df_fltfile_sim.iterrows():
        if (j+1) % 100 == 0: print("\tprogress: %i of %i"%(j+1,len(df_fltfile_sim)))
        
        #process ground motion
        out = ProcessGM(sim_acc[j], sim_vel[j],
                        freq4psa, freq4eas, 
                        cmp=cmp, grav=grav)
        
        #parse output
        pga, pgv, ai, d575, d595, d575_mean, eas, psa = out[0:8]
        pga_fnfp, pgv_fnfp, ai_fnfp, d575_fnfp, d595_fnfp, eas_fnfp, psa_fnfp = out[8:15]
        psa_max, psa_max_angle = out[15:17]
        
        #geometric mean
        # - - - - - - - - - - 
        #store pga and pgv
        df_fltfile_sim.loc[j,'pga'] = 0
        df_fltfile_sim.loc[j,'pgv'] = 0
        #store Arias Intensity
        df_fltfile_sim.loc[j,'ai'] = 0
        #store significant duration
        df_fltfile_sim.loc[j,'dur0.05-0.75'] = 0
        df_fltfile_sim.loc[j,'dur0.05-0.95'] = 0
        df_fltfile_sim.loc[j,'dur0.05-0.75_mean'] = d575_mean
        #store eas and psa
        df_fltfile_sim.loc[j,eas_col] = 0
        df_fltfile_sim.loc[j,psa_col] = 0

        #maximum component
        # - - - - - - - - - - 
        #psa value and angle
        df_fltfile_sim.loc[j,psa_max_col]       = 0
        df_fltfile_sim.loc[j,psa_max_angle_col] = 0

        #fault normal/parallel
        # - - - - - - - - - - 
        #store pga and pgv
        df_fltfile_sim.loc[j,'pga_fn'] = 0
        df_fltfile_sim.loc[j,'pga_fp'] = 0
        df_fltfile_sim.loc[j,'pgv_fn'] = 0
        df_fltfile_sim.loc[j,'pgv_fp'] = 0
        #store Arias Intensity
        df_fltfile_sim.loc[j,'ai_fn'] = 0
        df_fltfile_sim.loc[j,'ai_fp'] = 0
        #store significant duration
        df_fltfile_sim.loc[j,'dur0.05-0.75_fn'] = 0
        df_fltfile_sim.loc[j,'dur0.05-0.75_fp'] = 0
        df_fltfile_sim.loc[j,'dur0.05-0.95_fn'] = 0
        df_fltfile_sim.loc[j,'dur0.05-0.95_fp'] = 0
        #store eas and psa
        df_fltfile_sim.loc[j,eas_col_fn] = 0
        df_fltfile_sim.loc[j,eas_col_fp] = 0
        df_fltfile_sim.loc[j,psa_col_fn] = 0
        df_fltfile_sim.loc[j,psa_col_fp] = 0

    return df_fltfile_sim

def ProcessSim(dir_sim, freq4psa, freq4eas, 
               cmp=['X','Y'], grav=9.80665, n_batch_size=64):

    #read simulation info
    df_info_sim = pd.read_csv(dir_sim + "sim_info.csv")
                 
    #read fault geometry
    df_geom_sim = pd.read_csv(dir_sim + "fault_geom.csv")

    #parse simulation information
    eqid   = df_info_sim.eqid[0].astype(int)
    scnid  = df_info_sim.scenid[0].astype(int)
    eqname = df_info_sim.eqname[0]
    flag_ss = True if df_info_sim.eqname[0] == 1 else False
    rup_hyp = df_info_sim.loc[0,['hyp_x','hyp_y','hyp_z']].values.astype(float)
    mag     = df_info_sim.mag[0].astype(float)
    sof     = df_info_sim.sof[0].astype(float)
    dip     = df_info_sim.dip[0].astype(float)
    vs30    = df_info_sim.vs30[0].astype(float)
    z1      = df_info_sim.loc[0,'z1.0'].astype(float)
    #geometry information
    ztor    = df_geom_sim.z_top.abs().mean()
    rwidth  = (df_geom_sim.z_top-df_geom_sim.z_bottom).abs().mean()
        
    #ground motion flatfile
    df_fltfile_sim = ProcessSimGMs(dir_sim, eqid, scnid, eqname, flag_ss, mag, sof, dip, ztor, rwidth,
                                   df_geom_sim, rup_hyp, vs30, z1,
                                   freq4psa, freq4eas, 
                                   cmp=cmp, grav=grav, n_batch_size=n_batch_size)

    return eqname, mag, df_info_sim, df_geom_sim, df_fltfile_sim


#%% ****************************
### *    Process Simulation    *
### *      Ground Motions      *
### ****************************
if __name__ == '__main__':
    #%% Define Variables
    ### ======================================
    #directories for simulations
    dir_sims = [
        '/Volumes/G-Drive_PD_1/GM_Characterization/SS_525/SS_12_45_O4/VelData_test/',
        # Add more directories here as needed
    ]
    
    #frequency range
    f_min = 0.05
    f_max = 25
    #number of frequencies
    n_freq4psa = 31
    n_freq4eas = 31
    
    #number of batch size
    n_batch_size = mp.cpu_count() // 1 if os.getenv('BATCH_SIZE') is None else int(os.getenv('BATCH_SIZE'))

    print(f"batch_size = {n_batch_size}")
    
    #stations for comparsion
    sta2cmp = {'sta1':[5,-20], 'sta2':[5,0], 'sta3':[5,20], 'sta4':[5,40], 'sta5':[5,60], 'sta6':[5,80]}
    
    #output directory
    dir_out = 'Data/simulations/fast/'
    dir_fig = dir_out + '/figures/'
    
    #%% Process Ground Motions 
    ### ======================================
    #define frequency content
    #psa frequencies
    freq4psa = np.sort( 1/np.array([ 0.333, 0.4, 0.5, 0.7,
                                    1.0, 2.0, 4.0, 10]) )
    #eas frequency content
    freq4eas = np.sort( 1/np.array([ 0.333, 0.4, 0.5 , 0.7,
                                    1.0, 2, 4.0, 10]) )
    
    #psa and eas frequencies if not explicitly defined
    freq4psa = freq4psa if 'freq4psa' in locals() else np.logspace(np.log10(f_min), np.log10(f_max), n_freq4psa)
    freq4eas = freq4eas if 'freq4eas' in locals() else np.logspace(np.log10(f_min), np.log10(f_max), n_freq4eas)
    
    print(f"frequencies studied {freq4psa}")

    #process each simulation directory
    for dir_sim in dir_sims:
        print(f"\nProcessing simulation directory: {dir_sim}")
        
        #process ground motion
        eqname, mag, df_info_sim, df_geom_sim, df_fltfile_sim = ProcessSim(dir_sim, freq4psa, freq4eas, n_batch_size=n_batch_size)
        
        #%% Output
        ### ======================================
        #simulation subdirectory
        sim_dir_fig = dir_fig + '%s/'%eqname
        #create output directory
        if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
        if not os.path.isdir(sim_dir_fig): pathlib.Path(sim_dir_fig).mkdir(parents=True, exist_ok=True)
        
        #save flatfile
        fn_flt = "%s_flatfile"%eqname
        df_fltfile_sim.to_csv(dir_out + fn_flt + '.csv', index=False)
        
        #simulation information
        fn_info = "%s_info"%eqname
        df_info_sim.to_csv(dir_out + fn_info + '.csv', index=False)
        #simulation geometry
        fn_geom = "%s_fltgeom"%eqname
        df_geom_sim.to_csv(dir_out + fn_geom + '.csv', index=False)
        
        print(f"Completed processing for {eqname}")


