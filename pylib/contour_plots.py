#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:00:50 2024

@author: glavrent
"""
## load libraries
#arithmetic libraries
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
#plotting libraries
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib.ticker import FormatStrFormatter
from matplotlib import ticker
#user libraries
from pylib.general import polar_to_cartesian

class FormatScalarFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, fformat="%1.1f", offset=True, mathText=True):
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,
                                                        useMathText=mathText)
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

def plot_contour(cont_data, line_loc=None, pt_data=None, clevs=None, flag_grid=False, title=None, cbar_label=None, 
                 ptlevs = None, pt_label = None, log_cbar = False, frmt_clb = '%.2f', **kwargs):
    
    #additional input arguments
    flag_contour_labels = kwargs['flag_contour_labels'] if 'flag_contour_labels' in kwargs else False
    flag_smooth         = kwargs['flag_smooth'] if 'flag_smooth' in kwargs else False
    sig_smooth          = kwargs['smooth_sig'] if 'smooth_sig' in kwargs else 0.1
    contour_cmap        = kwargs['cmap'] if 'cmap' in kwargs else 'viridis'
    contour_alpha       = kwargs['alpha'] if 'alpha' in kwargs else 0.75

    #number of interpolation points, x & y direction
    ngridx = 1000
    ngridy = 1000
    
    #create figure
    # fig, ax = plt.subplots(figsize=(10, 10))
    fig, ax = plt.subplots(figsize=(15, 15))
    
    #project contour data
    x_cont = cont_data[:,0] 
    y_cont = cont_data[:,1]
    #interpolation grid
    x_int = np.linspace(x_cont.min(), x_cont.max(), ngridx)
    y_int = np.linspace(y_cont.min(), y_cont.max(), ngridy)
    X_grid, Y_grid = np.meshgrid(x_int, y_int)
    #interpolate contour data on grid
    data_grid = griddata((x_cont, y_cont) , cont_data[:,2], (X_grid, Y_grid), method='linear')

    #smooth 
    if flag_smooth: 
        data_grid = gaussian_filter(data_grid, sigma=sig_smooth)   
    #data colorbar
    if clevs is None:
        if not log_cbar:
            clevs = np.linspace(cont_data[:,2].min(),cont_data[:,2].max(),11).tolist()    
        else:
            clevs = np.logspace(np.log10(cont_data[:,2].min()),np.log10(cont_data[:,2].max()),11).tolist()    
    
    #plot interpolated data
    if not log_cbar:
        cs =  ax.contourf(X_grid, Y_grid, data_grid, levels = clevs, zorder=1, alpha=contour_alpha, cmap=contour_cmap, extend='max')
    else:
        cs =  ax.contourf(X_grid, Y_grid, data_grid, levels = clevs, zorder=1, alpha=contour_alpha, cmap=contour_cmap, 
                          locator=ticker.LogLocator())
        
    #color bar
    fmt_clb = ticker.FormatStrFormatter(frmt_clb)
    if not log_cbar:
        cbar = fig.colorbar(cs, boundaries = clevs, pad=0.1, orientation="horizontal", format=fmt_clb)
    else:
        cbar = fig.colorbar(cs, boundaries = clevs, pad=0.1, orientation="horizontal", format=fmt_clb)
    cbar.ax.tick_params(labelsize=25) 
    if (not cbar_label is None): cbar.set_label(cbar_label, size=28)
    
    #add contour labels
    if flag_contour_labels:
        ax.clabel(cs, inline=False, fontsize=22, colors='k')
    
    #plot line
    if not line_loc is None:
       ax.plot(line_loc[:,0], line_loc[:,1], linewidth=3, color='k', zorder= 2 )

    #plot points
    if not pt_data is None:
        if np.size(pt_data,1) == 2:
            ax.plot(pt_data[:,0], pt_data[:,1],  'o', color = 'k', markersize = 4, zorder = 3)
        elif np.size(pt_data,1) == 2:
            raise ValueError('Unimplemented plotting option')
           
    #add figure title
    if (not title == None): ax.set_title(title, fontsize=30)

    ax.set_xlabel('X', fontsize=28)
    ax.set_ylabel('Y', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    
    #grid lines
    if flag_grid: ax.grid()
    
    #tight layout    
    fig.tight_layout()
       
    return fig, ax, cs, cbar

def plot_contour_quiver(contquiver_data, line_loc=None, pt_data=None, clevs=None, degrees=False,
                        flag_grid=False, title=None, cbar_label=None, 
                        ptlevs = None, pt_label = None, log_cbar = False, log_quiv = False, 
                        frmt_clb = '%.2f', **kwargs):
    
    #additional input arguments
    flag_contour_labels = kwargs['flag_contour_labels'] if 'flag_contour_labels' in kwargs else False
    flag_smooth         = kwargs['flag_smooth'] if 'flag_smooth' in kwargs else False
    sig_smooth          = kwargs['smooth_sig'] if 'smooth_sig' in kwargs else 0.1
    contour_cmap        = kwargs['cmap'] if 'cmap' in kwargs else 'viridis'
    contour_alpha       = kwargs['alpha'] if 'alpha' in kwargs else 0.75

    #number of interpolation points, x & y direction
    ncontx = 1000
    nconty = 1000
    nquivx = 25
    nquivy = 25
    
    #create figure
    # fig, ax = plt.subplots(figsize=(10, 10))
    fig, ax = plt.subplots(figsize=(15, 15))
    
    #interpolation contour grid
    x_int = np.linspace(contquiver_data[:,0].min(), contquiver_data[:,0].max(), ncontx)
    y_int = np.linspace(contquiver_data[:,1].min(), contquiver_data[:,1].max(), nconty)
    X_contour, Y_contour = np.meshgrid(x_int, y_int)
    #interpolate contour data on grid
    data_contour = griddata((contquiver_data[:,0], contquiver_data[:,1]), contquiver_data[:,2], 
                            (X_contour, Y_contour), method='linear')
    
    #interpolation quiver grid
    x_int = np.linspace(contquiver_data[:,0].min(), contquiver_data[:,0].max(), nquivx)
    y_int = np.linspace(contquiver_data[:,1].min(), contquiver_data[:,1].max(), nquivy)
    X_quiver, Y_quiver = np.meshgrid(x_int, y_int)
    #interpolate contour data on grid
    data_quiver  = griddata((contquiver_data[:,0], contquiver_data[:,1]), contquiver_data[:,2], 
                            (X_quiver, Y_quiver), method='linear')
    angle_quiver = griddata((contquiver_data[:,0], contquiver_data[:,1]), contquiver_data[:,3], 
                            (X_quiver, Y_quiver), method='linear')
    #quiver cartesian
    data_quiver = data_quiver/np.nanmin(data_quiver)
    if log_quiv: data_quiver = np.log(data_quiver)
    data_quiver = polar_to_cartesian(data_quiver, angle_quiver, degrees=degrees)

    #smooth 
    if flag_smooth: 
        data_contour = gaussian_filter(data_contour, sigma=sig_smooth)   
    #data colorbar
    if clevs is None:
        if not log_cbar:
            clevs = np.linspace(contquiver_data[:,2].min(),contquiver_data[:,2].max(),11).tolist()    
        else:
            clevs = np.logspace(np.log10(contquiver_data[:,2].min()),np.log10(contquiver_data[:,2].max()),11).tolist()    
    
    #plot contour data
    if not log_cbar:
        cs =  ax.contourf(X_contour, Y_contour, data_contour, levels = clevs, zorder=1, alpha=contour_alpha, cmap=contour_cmap)
    else:
        cs =  ax.contourf(X_contour, Y_contour, data_contour, levels = clevs, zorder=1, alpha=contour_alpha, cmap=contour_cmap,
                          locator=ticker.LogLocator())
    #plot quiver
    ax.quiver(X_quiver, Y_quiver, data_quiver[0], data_quiver[1], zorder=1, width=0.002, color='black')
    
    #color bar
    fmt_clb = ticker.FormatStrFormatter(frmt_clb)
    if not log_cbar:
        cbar = fig.colorbar(cs, boundaries = clevs, pad=0.1, orientation="horizontal", format=fmt_clb) # add colorbar
    else:
        cbar = fig.colorbar(cs, boundaries = clevs, pad=0.1, orientation="horizontal", format=fmt_clb) # add colorbar
    cbar.ax.tick_params(labelsize=25) 
    if (not cbar_label is None): cbar.set_label(cbar_label, size=28)
    
    #add contour labels
    if flag_contour_labels:
        ax.clabel(cs, inline=False, fontsize=22, colors='k')
    
    #plot line
    if not line_loc is None:
       ax.plot(line_loc[:,0], line_loc[:,1], linewidth=3, color='k', zorder= 2 )

    #plot points
    if not pt_data is None:
        if np.size(pt_data,1) == 2:
            ax.plot(pt_data[:,0], pt_data[:,1],  'o', color = 'k', markersize = 4, zorder = 3)
        elif np.size(pt_data,1) == 2:
            raise ValueError('Unimplemented plotting option')
           
    #add figure title
    if (not title == None): ax.set_title(title, fontsize=30)

    ax.set_xlabel('X', fontsize=28)
    ax.set_ylabel('Y', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=28)
    
    #grid lines
    if flag_grid: ax.grid()
    
    #tight layout    
    fig.tight_layout()
       
    return fig, ax, cs, cbar
    