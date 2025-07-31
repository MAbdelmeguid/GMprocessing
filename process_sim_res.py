#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:21:08 2024

@author: glavrent
"""

# libraries
# - - - - - - - - - -
#general libraries
import os
import pathlib
import sys
import logging
import warnings
#regular expressions
import re
#arithmetic libraries
import scipy
import numpy  as np
import pandas as pd
#ground motion libraries
import pygmm
#plotting libraries
import matplotlib
import matplotlib.pyplot as plt
#user libraries
# - - - - - - - - - -
from pylib.stats import movingmean
from pylib.contour_plots import plot_contour
# user functions
# - - - - - - - - - -
def IMResColNames(freq_psa, freq_eas):
    """Column Names for: psa and eas residuals"""

    #psa and eas columns
    psa_res_col  = ['psa_res_f%.4fhz'%(f) for f in freq_psa]
    eas_res_col  = ['eas_res_f%.4fhz'%(f) for f in freq_eas]
    #fault normal and parallel components
    psaFN_res_col  = ['psaFN_res_f%.4fhz'%(f) for f in freq_psa]
    psaFP_res_col  = ['psaFP_res_f%.4fhz'%(f) for f in freq_psa]
    easFN_res_col  = ['easFN_res_f%.4fhz'%(f) for f in freq_eas]
    easFP_res_col  = ['easFP_res_f%.4fhz'%(f) for f in freq_eas]

    return psa_res_col, eas_res_col, psaFN_res_col, psaFP_res_col, easFN_res_col, easFP_res_col

#supress pygmm warnings
class WarningFilter(logging.Filter):
    def filter(self, record):
        # Suppress specific warning messages
        return "v_s30" not in record.getMessage()

# Apply the filter to the root logger
logging.getLogger().addFilter(WarningFilter())

#%% Define Variables
### ======================================
#simulation flatfile filename 
dir_sim_gm  = 'Data/simulations/test/' 

#simulation name
n_sim = 'sim40_sr' if os.getenv('NAME_SIM') is None else os.getenv('NAME_SIM')
xmax,xmin,ymax,ymin = [30,-30,45,-20]

#filenames
fn_siminfo_gm = "%s_info"%n_sim
fn_simgeom_gm = "%s_fltgeom"%n_sim
fn_simflt_gm  = "%s_flatfile"%n_sim

#output directory
dir_out = 'Data/simulations/residuals/'
dir_fig = dir_out + '/figures/'





#%% Read Data
### ======================================
#read simulation information
df_sim_info  = pd.read_csv(dir_sim_gm + fn_siminfo_gm + '.csv')
#read simulation geometry
df_sim_geom  = pd.read_csv(dir_sim_gm + fn_simgeom_gm + '.csv')
#read simulation flatfile
df_sim_gmflt = pd.read_csv(dir_sim_gm + fn_simflt_gm + '.csv')

#earthquake name
eq_name = df_sim_gmflt.eqname.unique()[0]

#%% Process Residuals 
### ======================================
print("Process Simulation Residuals: %s"%eq_name)

#hypocenter
eq_hyp   = df_sim_info.loc[0,['hyp_x','hyp_y']].values
#fault geometry
flt_geom = df_sim_geom.loc[:,['x','y']].values

#identify eas and psa values

### introduced by Mohamed
i_pga          = np.array([bool(re.match('^pga', c)) for c in df_sim_gmflt.columns])
i_pgv         = np.array([bool(re.match('^pgv', c)) for c in df_sim_gmflt.columns])  

####

i_eas           = np.array([bool(re.match('^eas_f(.*)hz', c)) for c in df_sim_gmflt.columns])
i_psa           = np.array([bool(re.match('^psa_f(.*)hz', c)) for c in df_sim_gmflt.columns])
i_eas_fn        = np.array([bool(re.match('^easFN_f(.*)hz', c)) for c in df_sim_gmflt.columns])
i_eas_fp        = np.array([bool(re.match('^easFP_f(.*)hz', c)) for c in df_sim_gmflt.columns])
i_psa_fn        = np.array([bool(re.match('^psaFN_f(.*)hz', c)) for c in df_sim_gmflt.columns])
i_psa_fp        = np.array([bool(re.match('^psaFP_f(.*)hz', c)) for c in df_sim_gmflt.columns])
i_psa_max       = np.array([bool(re.match('^psaMAX_f(.*)hz', c)) for c in df_sim_gmflt.columns])
i_psa_max_angle = np.array([bool(re.match('^psaMAX_angle_f(.*)hz', c)) for c in df_sim_gmflt.columns])
#indices other gm values
i_gm_other = np.isin(df_sim_gmflt.columns,['ai','dur0.05-0.75','dur0.05-0.95',
                                           'pga_fn','pgv_fn','ai_fn','dur0.05-0.75_fn','dur0.05-0.95_fn',
                                           'pga_fp','pgv_fp','ai_fp','dur0.05-0.75_fp','dur0.05-0.95_fp'])
#indices all gm
i_gm = np.any([i_eas,i_psa,i_eas_fn,i_eas_fp,i_psa_fn,i_psa_fp,i_psa_max,i_psa_max_angle,i_gm_other],axis=0)

#column names
cn_pga        = df_sim_gmflt.columns[i_pga]
cn_pgv        = df_sim_gmflt.columns[i_pgv]
cn_eas           = df_sim_gmflt.columns[i_eas]
cn_psa           = df_sim_gmflt.columns[i_psa]
cn_eas_fn        = df_sim_gmflt.columns[i_eas_fn]
cn_eas_fp        = df_sim_gmflt.columns[i_eas_fp]
cn_psa_fn        = df_sim_gmflt.columns[i_psa_fn]
cn_psa_fp        = df_sim_gmflt.columns[i_psa_fp]
cn_psa_max       = df_sim_gmflt.columns[i_psa_max]
cn_psa_max_angle = df_sim_gmflt.columns[i_psa_max_angle]

#eas and psa freq
freq_eas = np.array([float(re.findall('^eas_f(.*)hz', c)[0]) for c in df_sim_gmflt.columns[i_eas]])
freq_psa = np.array([float(re.findall('^psa_f(.*)hz', c)[0]) for c in df_sim_gmflt.columns[i_psa]])

pga = df_sim_gmflt.loc[:,cn_pga].values.astype(float)
pgv = df_sim_gmflt.loc[:,cn_pgv].values.astype(float)

#initialize residual dataframe
df_simflt_res = df_sim_gmflt.loc[:,~i_gm].copy()
#residual columns
cn_psa_res, cn_eas_res, cn_psa_fn_res, cn_psa_fp_res, cn_eas_fn_res, cn_eas_fp_res = IMResColNames(freq_psa, freq_eas)
#initialize residual columns
df_simflt_res.loc[:,['pga1','pgv1','pga2','pgv2']+cn_eas_res+cn_eas_fn_res+cn_eas_fp_res+cn_psa_res+cn_psa_fn_res+cn_psa_fp_res] = np.nan

#collect output
print("Compute Residuals")
for j, gm in df_sim_gmflt.iterrows():
    if (j+1) % 100 == 0: print("\tprocess ground motion residual %i of %i"%(j+1,len(df_sim_gmflt)))

    #define ground motion scenario
    # - - - - - - - - - - 
    #scenario
    mag   = gm.mag
    sof   = gm.sof
    mech  = str(np.select([abs(sof)<0.5, sof>=0.5, sof<=-0.5], ['SS','R','N'], 'None'))
    ztor  = gm.ztor
    dip   = gm.dip
    rw    = gm.rwidth
    rrup  = gm.rrup
    rjb   = gm.rrup
    rx    = gm.rrup
    ry0   = gm.rrup
    vs30  = gm.vs30
    z1    = gm['z1.0']

    # print(mag)
    
    #ground motion scenario
    gmm_scen = pygmm.Scenario(mag=mag, dip=dip, mechanism=mech, width=rw, depth_tor=ztor, 
                              dist_rup=rrup, dist_x=rx, dist_y0=ry0, dist_jb=rjb, 
                              v_s30=vs30, depth_1_0=z1,
                              region="global", on_hanging_wall=False, is_aftershock=False)
    
    #define eas and psa gmms
    gmm_eas = pygmm.BaylessAbrahamson2019(gmm_scen)
    gmm_psa = pygmm.BooreStewartSeyhanAtkinson2014(gmm_scen)
    gmm_psa2 = pygmm.AbrahamsonSilvaKamai2014(gmm_scen)

    #compute median ground motions
    eas_med = np.exp( np.interp( np.log(freq_eas), np.log(gmm_eas.freqs),   np.log(gmm_eas.eas)) )
    psa_med = np.exp( np.interp(-np.log(freq_psa), np.log(gmm_psa.periods), np.log(gmm_psa.spec_accels)) )
    pga_gmm    = gmm_psa.pga
    pgv_gmm    = gmm_psa.pgv
    pgv_gmm2   = gmm_psa2.pgv
    pga_gmm2   = gmm_psa2.pga

    

    #compute residuals
    # - - - - - - - - - - 
    #eas
    df_simflt_res.loc[j,cn_eas_res]    = np.log(gm[cn_eas].values.astype(float)) - np.log(eas_med)
    df_simflt_res.loc[j,cn_eas_fn_res] = np.log(gm[cn_eas_fn].values.astype(float)) - np.log(eas_med)
    df_simflt_res.loc[j,cn_eas_fp_res] = np.log(gm[cn_eas_fp].values.astype(float)) - np.log(eas_med)
    #psa
    df_simflt_res.loc[j,cn_psa_res]    = np.log(gm[cn_psa].values.astype(float)) - np.log(psa_med)
    df_simflt_res.loc[j,cn_psa_fn_res] = np.log(gm[cn_psa_fn].values.astype(float)) - np.log(psa_med)
    df_simflt_res.loc[j,cn_psa_fp_res] = np.log(gm[cn_psa_fp].values.astype(float)) - np.log(psa_med)
    df_simflt_res.loc[j,'pga1'] = pga_gmm
    df_simflt_res.loc[j,'pgv1'] = pgv_gmm
    df_simflt_res.loc[j,'pga2'] = pga_gmm2
    df_simflt_res.loc[j,'pgv2'] = pgv_gmm2


#residiual summary    
msg_res_summary = "Resiual Summary\n=========================\n"
#PSA bias summary
msg_res_summary += "PSA\n-----------------\n"
for f, c_res in zip(freq_psa, cn_psa_res): 
    msg_res_summary += "\tf=%.3fhz (mean,std) = %.2f, %.2f\n"%(f, df_simflt_res.loc[:,c_res].mean(), df_simflt_res.loc[:,c_res].std())
#PSA-FN bias summary
msg_res_summary += "PSA-FN\n-----------------\n"
for f, c_res in zip(freq_psa, cn_psa_fn_res): 
    msg_res_summary += "\tf=%.3fhz (mean,std) = %.2f, %.2f\n"%(f, df_simflt_res.loc[:,c_res].mean(), df_simflt_res.loc[:,c_res].std())
#PSA-FP bias summary
msg_res_summary += "PSA-FP\n------------------\n"
for f, c_res in zip(freq_psa, cn_psa_fp_res): 
    msg_res_summary += "\tf=%.3fhz (mean,std) = %.2f, %.2f\n"%(f, df_simflt_res.loc[:,c_res].mean(), df_simflt_res.loc[:,c_res].std())

#EAS bias summary
msg_res_summary += "\nEAS\n------------------\n"
for f, c_res in zip(freq_eas, cn_eas_res): 
    msg_res_summary += "\tf=%.3fhz (mean,std) = %.2f, %.2f\n"%(f, df_simflt_res.loc[:,c_res].mean(), df_simflt_res.loc[:,c_res].std())
#EAS-FN bias summary
msg_res_summary += "EAS-FN\n-----------------\n"
for f, c_res in zip(freq_eas, cn_eas_fn_res): 
    msg_res_summary += "\tf=%.3fhz (mean,std) = %.2f, %.2f\n"%(f, df_simflt_res.loc[:,c_res].mean(), df_simflt_res.loc[:,c_res].std())
#EAS-FP bias summary
msg_res_summary += "EAS-FP\n------------------\n"
for f, c_res in zip(freq_eas, cn_eas_fp_res): 
    msg_res_summary += "\tf=%.3fhz (mean,std) = %.2f, %.2f\n"%(f, df_simflt_res.loc[:,c_res].mean(), df_simflt_res.loc[:,c_res].std())



#%%% Truncate data
### ======================================
#truncate data
df_simflt_res = df_simflt_res[(df_simflt_res['sty'] <= ymax) & (df_simflt_res['sty'] >= ymin) & 
                              (df_simflt_res['stx'] <= xmax) & (df_simflt_res['stx'] >= xmin)]
df_simflt_res.to_csv(dir_out + 'test.csv', index=False)
    
#%% Output
### ======================================
#simulation subdirectory
dir_fig += '%s/'%eq_name
#save file
if not os.path.isdir(dir_out): pathlib.Path(dir_out).mkdir(parents=True, exist_ok=True)
if not os.path.isdir(dir_fig): pathlib.Path(dir_fig).mkdir(parents=True, exist_ok=True)

#save flatfile
fn_flt = "%s_flatfile_res"%eq_name
df_simflt_res.to_csv(dir_out + fn_flt + '.csv', index=False)

#residuals summary
f = open(dir_out + '%s_summary_res'%eq_name + '.txt', 'w')
f.write(msg_res_summary)
f.close()




# #%% Figures
# ### ======================================
# cn_psa_res, cn_eas_res, cn_psa_fn_res, cn_psa_fp_res, cn_eas_fn_res, cn_eas_fp_res = IMResColNames(freq_psa, freq_eas)

# #Residuals versus distance
# #----------------------------------
# #rupture distance
# rrup = df_simflt_res.loc[:,'rrup'].values

# #bined rupture distance
# rrup_bins = np.linspace(0.5, 30., 21)


# # pga
# print("Plot PGA vs Distance")
# fig_fname = "%s_pga_rrup"%(eq_name)
# pga = df_simflt_res.loc[:,'pga'].values
# pga_gmm = df_simflt_res.loc[:,'pga1'].values
# pga2_gmm = df_simflt_res.loc[:,'pga2'].values
# pga_rrup,    _, pga_mu,    _, pga_16prc,    pga_84prc    = movingmean(pga, rrup, rrup_bins)
# fig, ax = plt.subplots(figsize=(10,10))
# ax.plot(rrup,pga*10, 'o',color = 'green', label='Simulated')
# ax.plot(pga_rrup, pga_mu*10, 'x', color='blue', label='mean')
# ax.plot(rrup, pga_gmm*10, 's', color='black', label='BSSA14')
# ax.plot(rrup, pga2_gmm*10, 's', color='red', label='ASK14')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel('Rupture Distance (km)')
# ax.set_ylabel('PGA (m/s/s)')
# ax.set_ylim(5e-2, 5e1)
# ax.set_xlim(0.05,50)
# ax.legend()
# ax.grid(True)
# fig.savefig(dir_fig+fig_fname+'.png', bbox_inches='tight')

# # pgv
# print("Plot PGV vs Distance")
# fig_fname = "%s_pgv_rrup"%(eq_name)
# pgv = df_simflt_res.loc[:,'pgv'].values
# pgv_rrup,    _, pgv_mu,    _, pgv_16prc,    pgv_84prc    = movingmean(pgv, rrup, rrup_bins)
# pgv_gmm = df_simflt_res.loc[:,'pgv1'].values
# pgv_gmm2 = df_simflt_res.loc[:,'pgv2'].values
# pgv_gmm = pgv_gmm/100 # convert to m/s
# pgv_gmm2 = pgv_gmm2/100 # convert to m/s
# fig, ax = plt.subplots(figsize=(10,10))
# ax.plot(rrup, pgv, 'o', label='Simulated')
# ax.plot(pgv_rrup, pgv_mu, 'x', color='blue', label='mean')
# ax.plot(rrup, pgv_gmm2, 's', color='red', label='ASK14')
# ax.plot(rrup, pgv_gmm, 's', color='black', label='BSSA14')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel('Rupture Distance (km)')
# ax.set_ylabel('PGV (m/s)')
# ax.set_xlim(0.05,50)
# ax.legend()
# ax.grid(True)
# fig.savefig(dir_fig+fig_fname+'.png', bbox_inches='tight')
# plt.close()


# # psa
# # - - - - - - - - - -
# print("Plot PSA Residuals vs Distance")
# for j, f_p in enumerate(freq_psa):
#     if (j) % 5 == 0 or (j+1) == len(freq_psa): print("\tplotting PSA frequency f=%.3fhz (%i of %i)"%(f_p,j+1,len(freq_psa)))
#     #parse residuals
#     res_psa    = df_simflt_res.loc[:,cn_psa_res[j]].values
#     res_psa_fn = df_simflt_res.loc[:,cn_psa_fn_res[j]].values
#     res_psa_fp = df_simflt_res.loc[:,cn_psa_fp_res[j]].values
    
#     #compute binned residuals
#     bres_psa_rrup,    _, bres_psa_mu,    _, bres_psa_16prc,    bres_psa_84prc    = movingmean(res_psa, rrup, rrup_bins)
#     bres_psa_fn_rrup, _, bres_psa_fn_mu, _, bres_psa_fn_16prc, bres_psa_fn_84prc = movingmean(res_psa_fn, rrup, rrup_bins)
#     bres_psa_fp_rrup, _, bres_psa_fp_mu, _, bres_psa_fp_16prc, bres_psa_fp_84prc = movingmean(res_psa_fp, rrup, rrup_bins)
    
#     #figure name
#     fig_fname  = "%s_psa_res_f%.4fhz_rrup"%(eq_name,f_p)
#     fig_title  = "%s, Residuals $PSA$(f=%.2fhz)"%(eq_name.replace('_',' ').upper(),f_p)
#     #figure axes
#     fig, ax = plt.subplots(figsize = (20,10), nrows=1, ncols=1)
#     #psa
#     hl = ax.plot(rrup, res_psa, 'o')
#     hl = ax.plot(bres_psa_rrup, bres_psa_mu, 's', color='black', label='Mean')
#     hl = ax.axhline(0, color='black', linestyle='--')

#     hl = ax.plot(bres_psa_rrup, bres_psa_mu-bres_psa_mu[0], 'x', color='blue', label='Mean-Mean[0]')
#     hl = ax.errorbar(bres_psa_rrup, bres_psa_mu, yerr=np.abs(np.vstack((bres_psa_16prc,bres_psa_84prc)) - bres_psa_mu),
#                      capsize=8, fmt='none', ecolor='black', linewidth=2, label=r'$16-84^{th}$'+'\n Percentile')
#     ax.set_ylabel(r'$\delta T$', fontsize=28)
#     ax.legend(loc='lower right', fontsize=28, ncols=2)
#     ax.grid(which='both')
#     ax.tick_params(axis='x', labelsize=25)
#     ax.tick_params(axis='y', labelsize=25)
#     ax.set_ylim([-3, 3])
#     ax.set_xlim([0.1,30])
#     ax.set_yticks([-3.,-1.5,0.,1.5,3.])
#     # #psa-fn
#     # hl = ax[1].plot(rrup, res_psa_fn, 'o')
#     # hl = ax[1].plot(bres_psa_fn_rrup, bres_psa_fn_mu, 's', color='black', label='Mean')
#     # hl = ax[1].errorbar(bres_psa_fn_rrup, bres_psa_fn_mu, yerr=np.abs(np.vstack((bres_psa_fn_16prc,bres_psa_fn_84prc)) - bres_psa_fn_mu),
#     #                     capsize=8, fmt='none', ecolor='black', linewidth=2, label=r'$16-84^{th}$'+'\n Percentile')
#     # ax[1].set_ylabel(r'$\delta T_{PSA-FN}$', fontsize=28)
#     # ax[1].grid(which='both')
#     # ax[1].tick_params(axis='x', labelsize=25)
#     # ax[1].tick_params(axis='y', labelsize=25)
#     # ax[1].set_ylim([-3, 3])
#     # ax[1].set_yticks([-3.,-1.5,0.,1.5,3.])
#     # #psa-fp
#     # hl = ax[2].plot(rrup, res_psa_fp, 'o')
#     # hl = ax[2].plot(bres_psa_fp_rrup, bres_psa_fp_mu, 's', color='black', label='Mean')
#     # hl = ax[2].errorbar(bres_psa_fp_rrup, bres_psa_fp_mu, yerr=np.abs(np.vstack((bres_psa_fp_16prc,bres_psa_fp_84prc)) - bres_psa_fp_mu),
#     #                     capsize=8, fmt='none', ecolor='black', linewidth=2, label=r'$16-84^{th}$'+'\n Percentile')
#     # ax[2].set_xlabel(r'$R_{rup}$ (km)',        fontsize=28)
#     # ax[2].set_ylabel(r'$\delta T_{PSA-FP}$', fontsize=28)
#     # ax[2].grid(which='both')
#     # ax[2].tick_params(axis='x', labelsize=25)
#     # ax[2].tick_params(axis='y', labelsize=25)
#     # ax[2].set_ylim([-3, 3])
#     # ax[2].set_yticks([-3.,-1.5,0.,1.5,3.])
#     # #save figure
#     # ax[0].set_title(fig_title, fontsize=30)
#     # fig.tight_layout()
#     fig.savefig(dir_fig+fig_fname+'.png', bbox_inches='tight')
#     plt.close()

# # combined subplots for psa residuals for first 5 frequencies
# fig, ax = plt.subplots(figsize = (20,5), nrows=1, ncols=5)
# axes = ax.flatten()
# for j, f_p in enumerate([freq_psa[0], freq_psa[1], freq_psa[2], freq_psa[4], freq_psa[6]]):
#     res_psa    = df_simflt_res.loc[:,cn_psa_res[j]].values
#     res_psa_fn = df_simflt_res.loc[:,cn_psa_fn_res[j]].values
#     res_psa_fp = df_simflt_res.loc[:,cn_psa_fp_res[j]].values
    
#     #compute binned residuals
#     bres_psa_rrup,    _, bres_psa_mu,    _, bres_psa_16prc,    bres_psa_84prc    = movingmean(res_psa, rrup, rrup_bins)
#     bres_psa_fn_rrup, _, bres_psa_fn_mu, _, bres_psa_fn_16prc, bres_psa_fn_84prc = movingmean(res_psa_fn, rrup, rrup_bins)
#     bres_psa_fp_rrup, _, bres_psa_fp_mu, _, bres_psa_fp_16prc, bres_psa_fp_84prc = movingmean(res_psa_fp, rrup, rrup_bins)
    
#     #figure name
#     fig_fname  = "%s_psa_res_rrup"%(eq_name)
#     fig_title  = "%s, Residuals $PSA$(f=%.2fhz)"%(eq_name.replace('_',' ').upper(),f_p)

#     #psa
#     hl = axes[j].plot(rrup, res_psa, 'o')
#     hl = axes[j].plot(bres_psa_rrup, bres_psa_mu, 's', color='black', label='Mean')
#     hl = axes[j].axhline(0, color='black', linestyle='--')
#     hl = axes[j].errorbar(bres_psa_rrup, bres_psa_mu, yerr=np.abs(np.vstack((bres_psa_16prc,bres_psa_84prc)) - bres_psa_mu),
#                      capsize=8, fmt='none', ecolor='black', linewidth=2, label=r'$16-84^{th}$'+'\n Percentile')
#     axes[0].set_ylabel(r'$\delta T$', fontsize=28)
#     axes[0].legend(loc='lower right', fontsize=18, ncols=1)
#     axes[j].grid(which='both')
#     axes[j].tick_params(axis='x', labelsize=25)
#     axes[j].tick_params(axis='y', labelsize=25)
#     axes[j].set_ylim([-3, 3])
#     axes[j].set_xlim([0.1,30])
#     axes[0].set_yticks([-3.,-1.5,0.,1.5,3.])
#     axes[j].annotate(f'f={f_p:.2f}hz', (3, 2.0), fontsize=15, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'))
#     axes[0].set_title(f'Hypo = {eq_hyp[1]}, Mw = {gm.mag}', fontsize = 20)
    
#     if j > 0:
#         axes[j].set_yticks([])
    
    
# fig.savefig(dir_fig+fig_fname+'combined.png', bbox_inches='tight')
# # plt.show()
# plt.close()












# # # eas
# # # - - - - - - - - - -
# # print("Plot EAS Residuals vs Distance")
# # for j, f_e in enumerate(freq_eas):
# #     if (j) % 25 == 0 or (j+1) == len(freq_eas): print("\tplotting EAS frequency f=%.3fhz (%i of %i)"%(f_e,j+1,len(freq_eas)))
# #     #parse residuals 
# #     res_eas    = df_simflt_res.loc[:,cn_eas_res[j]].values
# #     res_eas_fn = df_simflt_res.loc[:,cn_eas_fn_res[j]].values
# #     res_eas_fp = df_simflt_res.loc[:,cn_eas_fp_res[j]].values
    
# #     #compute binned residuals
# #     bres_eas_rrup,    _, bres_eas_mu,    _, bres_eas_16prc,    bres_eas_84prc    = movingmean(res_eas, rrup, rrup_bins)
# #     bres_eas_fn_rrup, _, bres_eas_fn_mu, _, bres_eas_fn_16prc, bres_eas_fn_84prc = movingmean(res_eas_fn, rrup, rrup_bins)
# #     bres_eas_fp_rrup, _, bres_eas_fp_mu, _, bres_eas_fp_16prc, bres_eas_fp_84prc = movingmean(res_eas_fp, rrup, rrup_bins)
    
# #     #figure name
# #     fig_fname  = "%s_eas_f%.4fhz_rrup"%(eq_name,f_e)
# #     fig_title  = "%s, $EAS$(f=%.2fhz)"%(eq_name.replace('_',' ').upper(),f_e)
# #     #figure axes
# #     fig, ax = plt.subplots(figsize = (20,3*10), nrows=3, ncols=1)
# #     #eas
# #     hl = ax[0].plot(rrup, res_eas, 'o')
# #     hl = ax[0].plot(bres_eas_rrup, bres_eas_mu, 's', color='black', label='Mean')
# #     hl = ax[0].errorbar(bres_eas_rrup, bres_eas_mu, yerr=np.abs(np.vstack((bres_eas_16prc,bres_eas_84prc)) - bres_eas_mu),
# #                         capsize=8, fmt='none', ecolor='black', linewidth=2, label=r'$16-84^{th}$'+'\n Percentile')
# #     ax[0].set_ylabel(r'$\delta T_{EAS}$',      fontsize=28)
# #     ax[0].legend(loc='lower right', fontsize=28, ncols=2)
# #     ax[0].grid(which='both')
# #     ax[0].tick_params(axis='x', labelsize=25)
# #     ax[0].tick_params(axis='y', labelsize=25)
# #     ax[0].set_ylim([-3, 3])
# #     ax[0].set_yticks([-3.,-1.5,0.,1.5,3.])
# #     #eas-fn
# #     hl = ax[1].plot(rrup, res_eas_fn, 'o')
# #     hl = ax[1].plot(bres_eas_fn_rrup, bres_eas_fn_mu, 's', color='black', label='Mean')
# #     hl = ax[1].errorbar(bres_eas_fn_rrup, bres_eas_fn_mu, yerr=np.abs(np.vstack((bres_eas_fn_16prc,bres_eas_fn_84prc)) - bres_eas_fn_mu),
# #                         capsize=8, fmt='none', ecolor='black', linewidth=2, label=r'$16-84^{th}$'+'\n Percentile')
# #     ax[1].set_ylabel(r'$\delta T_{FASKO-FN}$', fontsize=28)
# #     ax[1].grid(which='both')
# #     ax[1].tick_params(axis='x', labelsize=25)
# #     ax[1].tick_params(axis='y', labelsize=25)
# #     ax[1].set_ylim([-3, 3])
# #     ax[1].set_yticks([-3.,-1.5,0.,1.5,3.])
# #     #eas-fp
# #     hl = ax[2].plot(rrup, res_eas_fp, 'o')
# #     hl = ax[2].plot(bres_eas_fp_rrup, bres_eas_fp_mu, 's', color='black', label='Mean')
# #     hl = ax[2].errorbar(bres_eas_fp_rrup, bres_eas_fp_mu, yerr=np.abs(np.vstack((bres_eas_fp_16prc,bres_eas_fp_84prc)) - bres_eas_fp_mu),
# #                         capsize=8, fmt='none', ecolor='black', linewidth=2, label=r'$16-84^{th}$'+'\n Percentile')
# #     ax[2].set_xlabel(r'$R_{rup}$ (km)',        fontsize=28)
# #     ax[2].set_ylabel(r'$\delta T_{FASKO-FP}$', fontsize=28)
# #     ax[2].grid(which='both')
# #     ax[2].tick_params(axis='x', labelsize=25)
# #     ax[2].tick_params(axis='y', labelsize=25)
# #     ax[2].set_ylim([-3, 3])
# #     ax[2].set_yticks([-3.,-1.5,0.,1.5,3.])
# #     #save figure
# #     ax[0].set_title(fig_title, fontsize=30)
# #     fig.tight_layout()
# #     fig.savefig(dir_fig+fig_fname+'.png', bbox_inches='tight')
# #     plt.close()


# #Spatial Extent of Residuals
# #----------------------------------
# #residual contour levels
# res_levs = np.linspace(-3., 3.,22).tolist()
# #color map
# cmap = 'bwr'

# #hypocenter location
# hyp_pt = df_sim_info.loc[0,['hyp_x','hyp_y']].values

# # psa
# # - - - - - - - - - -
# print("Plot PSA Residuals Contour")
# for j, f_p in enumerate(freq_psa):
#     if (j) % 5 == 0 or (j+1) == len(freq_psa): print("\tplotting PSA frequency f=%.3fhz (%i of %i)"%(f_p,j+1,len(freq_psa)))
#     #psa-rotd50
#     # - - - - - - - - - - - - - -
#     fig_fname  = "%s_res_psa_f%.4fhz_med_contour"%(eq_name,f_p)
#     fig_title  = "%s, Residual $PSA_{rotD50}$(f=%.2fhz)"%(eq_name.replace('_',' ').upper(),f_p)
#     cbar_label = "Residual $PSA_{rotD50}(f=%.2fhz)$"%(f_p)
#     #contour data
#     cont_data = df_simflt_res.loc[:,['stx','sty']+[cn_psa_res[j]]].values
#     #create figure
#     fig, ax, cs, cbar = plot_contour(cont_data, flt_geom, pt_data=None, clevs=res_levs, contour_alpha=1.,
#                                      flag_grid=True, title=fig_title, cbar_label=cbar_label, frmt_clb = '%.3f',
#                                      log_cbar=False, cmap=cmap, flag_contour_labels=False)
#     #add hypocenter location
#     ax.plot(eq_hyp[0],eq_hyp[1],'*',markersize=40, markeredgewidth=3, 
#             markeredgecolor='red', markerfacecolor='yellow')
#     #save figure
#     fig.tight_layout()
#     # fig.savefig(dir_fig+fig_fname+'.png', bbox_inches='tight')
#     #add contour labels
#     ax.clabel(cs, inline=False, fontsize=22, colors='k')
#     fig.savefig(dir_fig+fig_fname+'_labels'+'.png', bbox_inches='tight')
#     plt.close()
    
    # #psa-fn
    # # - - - - - - - - - - - - - -
    # fig_fname  = "%s_res_psa_f%.4fhz_fn_contour"%(eq_name,f_p)
    # fig_title  = "%s, Residual $PSA_{FN}$(f=%.2fhz)"%(eq_name.replace('_',' ').upper(),f_p)
    # cbar_label = "Residual $PSA_{FN}(f=%.2fhz)$"%(f_p)
    # #contour data
    # cont_data = df_simflt_res.loc[:,['stx','sty']+[cn_psa_fn_res[j]]].values
    # #create figure
    # fig, ax, cs, cbar = plot_contour(cont_data, flt_geom, pt_data=None, clevs=res_levs, contour_alpha=1.,
    #                                  flag_grid=True, title=fig_title, cbar_label=cbar_label, frmt_clb = '%.3f',
    #                                  log_cbar=False, cmap=cmap, flag_contour_labels=False)
    # #add hypocenter location
    # ax.plot(eq_hyp[0],eq_hyp[1],'*',markersize=40, markeredgewidth=3, 
    #         markeredgecolor='red', markerfacecolor='yellow')
    # #save figure
    # fig.tight_layout()
    # fig.savefig(dir_fig+fig_fname+'.png', bbox_inches='tight')
    # #add contour labels
    # ax.clabel(cs, inline=False, fontsize=22, colors='k')
    # fig.savefig(dir_fig+fig_fname+'_labels'+'.png', bbox_inches='tight')
    # plt.close()
    
    # #psa-fp
    # # - - - - - - - - - - - - - -
    # fig_fname  = "%s_res_psa_f%.4fhz_fp_contour"%(eq_name,f_p)
    # fig_title  = "%s, Residual $PSA_{FP}$(f=%.2fhz)"%(eq_name.replace('_',' ').upper(),f_p)
    # cbar_label = "Residual $PSA_{FP}(f=%.2fhz)$"%(f_p)
    # #contour data
    # cont_data = df_simflt_res.loc[:,['stx','sty']+[cn_psa_fp_res[j]]].values
    # #create figure
    # fig, ax, cs, cbar = plot_contour(cont_data, flt_geom, pt_data=None, clevs=res_levs, contour_alpha=1.,
    #                                  flag_grid=True, title=fig_title, cbar_label=cbar_label, frmt_clb = '%.3f',
    #                                  log_cbar=False, cmap=cmap, flag_contour_labels=False)
    # #add hypocenter location
    # ax.plot(eq_hyp[0],eq_hyp[1],'*',markersize=40, markeredgewidth=3, 
    #         markeredgecolor='red', markerfacecolor='yellow')
    # #save figure
    # fig.tight_layout()
    # # fig.savefig(dir_fig+fig_fname+'.png', bbox_inches='tight')
    # #add contour labels
    # ax.clabel(cs, inline=False, fontsize=22, colors='k')
    # fig.savefig(dir_fig+fig_fname+'_labels'+'.png', bbox_inches='tight')
    # plt.close()

# eas
# - - - - - - - - - -
# print("Plot PSA Residuals Contour")
# for j, f_e in enumerate(freq_eas):
#     if (j) % 25 == 0 or (j+1) == len(freq_eas): print("\tplotting PSA frequency f=%.3fhz (%i of %i)"%(f_e,j+1,len(freq_eas)))
#     #eas
#     # - - - - - - - - - - - - - -
#     fig_fname  = "%s_res_eas_f%.4fhz_med_contour"%(eq_name,f_e)
#     fig_title  = "%s, Residual $EAS$(f=%.2fhz)"%(eq_name.replace('_',' ').upper(),f_e)
#     cbar_label = "Residual $EAS$(f=%.2fhz)$"%(f_p)
#     #contour data
#     cont_data = df_simflt_res.loc[:,['stx','sty']+[cn_eas_res[j]]].values
#     #create figure
#     fig, ax, cs, cbar = plot_contour(cont_data, flt_geom, pt_data=None, clevs=res_levs, contour_alpha=1.,
#                                      flag_grid=True, title=fig_title, cbar_label=cbar_label, frmt_clb = '%.3f',
#                                      log_cbar=False, cmap=cmap, flag_contour_labels=False)
#     #add hypocenter location
#     ax.plot(eq_hyp[0],eq_hyp[1],'*',markersize=40, markeredgewidth=3, 
#             markeredgecolor='red', markerfacecolor='yellow')
#     #save figure
#     fig.tight_layout()
#     fig.savefig(dir_fig+fig_fname+'.png', bbox_inches='tight')
#     #add contour labels
#     ax.clabel(cs, inline=False, fontsize=22, colors='k')
#     fig.savefig(dir_fig+fig_fname+'_labels'+'.png', bbox_inches='tight')
#     plt.close()
    
#     #fasko-fn
#     # - - - - - - - - - - - - - -
#     fig_fname  = "%s_res_eas_f%.4fhz_fn_contour"%(eq_name,f_p)
#     fig_title  = "%s, Residual $FASKO_{FN}$(f=%.2fhz)"%(eq_name.replace('_',' ').upper(),f_p)
#     cbar_label = "Residual $FASKO_{FN}(f=%.2fhz)$"%(f_p)
#     #contour data
#     cont_data = df_simflt_res.loc[:,['stx','sty']+[cn_eas_fn_res[j]]].values
#     #create figure
#     fig, ax, cs, cbar = plot_contour(cont_data, flt_geom, pt_data=None, clevs=res_levs, contour_alpha=1.,
#                                      flag_grid=True, title=fig_title, cbar_label=cbar_label, frmt_clb = '%.3f',
#                                      log_cbar=False, cmap=cmap, flag_contour_labels=False)
#     #add hypocenter location
#     ax.plot(eq_hyp[0],eq_hyp[1],'*',markersize=40, markeredgewidth=3, 
#             markeredgecolor='red', markerfacecolor='yellow')
#     #save figure
#     fig.tight_layout()
#     fig.savefig(dir_fig+fig_fname+'.png', bbox_inches='tight')
#     #add contour labels
#     ax.clabel(cs, inline=False, fontsize=22, colors='k')
#     fig.savefig(dir_fig+fig_fname+'_labels'+'.png', bbox_inches='tight')
#     plt.close()
    
#     #fasko-fp
#     # - - - - - - - - - - - - - -
#     fig_fname  = "%s_res_eas_f%.4fhz_fp_contour"%(eq_name,f_p)
#     fig_title  = "%s, Residual $FASKO_{FP}$(f=%.2fhz)"%(eq_name.replace('_',' ').upper(),f_p)
#     cbar_label = "Residual $FASKO_{FP}(f=%.2fhz)$"%(f_p)
#     #contour data
#     cont_data = df_simflt_res.loc[:,['stx','sty']+[cn_eas_fp_res[j]]].values
#     #create figure
#     fig, ax, cs, cbar = plot_contour(cont_data, flt_geom, pt_data=None, clevs=res_levs, contour_alpha=1.,
#                                      flag_grid=True, title=fig_title, cbar_label=cbar_label, frmt_clb = '%.3f',
#                                      log_cbar=False, cmap=cmap, flag_contour_labels=False)
#     #add hypocenter location
#     ax.plot(eq_hyp[0],eq_hyp[1],'*',markersize=40, markeredgewidth=3, 
#             markeredgecolor='red', markerfacecolor='yellow')
#     #save figure
#     fig.tight_layout()
#     fig.savefig(dir_fig+fig_fname+'.png', bbox_inches='tight')
#     #add contour labels
#     ax.clabel(cs, inline=False, fontsize=22, colors='k')
#     fig.savefig(dir_fig+fig_fname+'_labels'+'.png', bbox_inches='tight')
#     plt.close()

