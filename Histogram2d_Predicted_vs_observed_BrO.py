#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:32:08 2020

@author: ncp532
"""

# Drawing packages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates            
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatter 

# Data handing packages
import numpy as np
import pandas as pd
from scipy import signal, stats

# Date and Time handling package
import datetime as dt
from datetime import datetime,time, timedelta		# functions to handle date and time

#------------------------------------------------------------------------------
# DEFINE THE DATASET

# BrO predictions without SIPEXII
PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_Test2.csv', index_col=0) # GridBox (BestChoice)

#------------------------------------------------------------------------------
# SET THE DATE

PC_BrO.index = (pd.to_datetime(PC_BrO.index, dayfirst=True))

#------------------------------------------------------------------------------
# SET THE VARIABLES

Date    = PC_BrO.index

#-------------
# Predictions
#-------------
# Swanson
BrO_SP_OLS  = PC_BrO['Swanson_BrOSurf']/1e13  # Surf BrO using Swanson coefficients
BrO_LTP_OLS = PC_BrO['Swanson_BrOLTcol']/1e13 # LTcol BrO using Swanson coefficients

# PCR (OLS)
BrO_SP_Met  = PC_BrO['OLS_BrOSurf']/1e13  # Surf BrO using OLS coefficients
BrO_LTP_Met = PC_BrO['OLS_BrOLTcol']/1e13 # Ltcol BrO using OLS coefficients

#-------------
# Observations
#-------------
BrO_SO  = PC_BrO['Obs_BrOSurf']/1e13
BrO_LTO = PC_BrO['Obs_BrOLTcol']/1e13

# #-------------
# # Observational error
# #-------------
#ErrorS  = np.mean(PC_BrO['BrO_SurfObs_Err']/1e13)
#ErrorLT = np.mean(PC_BrO['BrO_SurfObs_Err']/1e13)
ErrorS   = np.std(BrO_SO)
ErrorLT  = np.std(BrO_LTO)

#-------------
# StDev error
#-------------
StDev_SO = np.std(BrO_SO)
StDev_LTO = np.std(BrO_LTO)

#------------------------------------------------------------------------------
# CONSTRUCT A 2D HISTOGRAM

#-------------
# LTcol BrO
#-------------
# Define the bin edges
#xedgesLT = np.arange(0,2.6,0.2) # Swanson
#yedgesLT = np.arange(0,3.8,0.2) # Swanson

xedgesLT = np.arange(0,2.6,0.1) # My coefficient
yedgesLT = np.arange(0,2.0,0.1) # My coefficient

# Create a histogram (H)
HLT, xedgesLT, yedgesLT = np.histogram2d(BrO_LTO, BrO_LTP_Met, bins=(xedgesLT, yedgesLT))
HLT = HLT.T  # Let each row list bins with common y range.
MaxLT = np.argmax(HLT) 

#-------------
# Surface BrO
#-------------
# Define the bin edges
#xedgesS = np.arange(0,1.0,0.05) # Swanson
#yedgesS = np.arange(0,1.4,0.05) # Swanson

xedgesS = np.arange(0,1.0,0.025) # My coefficient
yedgesS = np.arange(0,1.4,0.025) # My coefficient

# Create a histogram (H)
HS, xedgesS, yedgesS = np.histogram2d(BrO_SO, BrO_SP_Met, bins=(xedgesS, yedgesS))
HS = HS.T  # Let each row list bins with common y range.
MaxS = np.argmax(HS) 

#------------------------------------------------------------------------------
# PLOT THE GRAPH

fig1 = plt.figure()
plt.subplots_adjust(hspace=0.5)

#-----------------------------------
# Graph 1 (LTcol BrO)
ax=plt.subplot(111) # options graph 1 (vertical no, horizontal no, graph no)

# Plot Predictions vs Observations
cmap=plt.cm.Reds
plt.hist2d(BrO_LTO, BrO_LTP_Met, bins=(xedgesLT, yedgesLT),norm=LogNorm(),cmap=cmap,vmax=MaxLT)  # Swanson
#plt.hist2d(BrO_LTO, BrO_LTP_OLS, bins=(xedgesLT, yedgesLT),norm=LogNorm(),cmap=cmap,vmax=MaxLT) # My coefficient
plt.plot([0,4],[0,4],c='black') # Plot 1:1 line
plt.plot([0-ErrorLT,4-ErrorLT],[0,4],c='black', ls='--') # Plot lower error limit
plt.plot([0+ErrorLT,4+ErrorLT],[0,4],c='black',ls='--') # Plot lower error limit

# Format x-axis
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#ax.set_xlim(0.1, 34.5)
ax.xaxis.labelpad = 10

# Format y-axis 1
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#ax.set_ylim(0.1, 22.5)
ax.yaxis.labelpad = 10

# Plot the axis labels
ax.set_xlabel('Observed BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=15)
ax.set_ylabel('Predicted BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=15)

# Plot the title
#plt.title('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=15, pad=10)

# Plot the colorbar (CAMMPCAN)
formatter = LogFormatter(10, labelOnlyBase=False) 
cbar1 = plt.colorbar(ticks=[1,5,10,50], format=formatter)
cbar1.ax.set_ylabel('Number of observations in Bin', fontsize=15)
cbar1.ax.set_yticklabels(['0','5','10','20'], fontsize=15)  # vertically oriented colorbar

## Plot the colorbar (SIPEXII)
#formatter = LogFormatter(10, labelOnlyBase=False) 
#cbar1 = plt.colorbar(ticks=[1,2,5,10,20], format=formatter)
#cbar1.ax.set_ylabel('Number of observations in Bin')
#cbar1.ax.set_yticklabels(['0','2','5','10','20'])  # vertically oriented colorbar

# Format axis labels
ax.tick_params(labelsize=15)

# #-----------------------------------
# # Graph 2 (Surface BrO)
# ax=plt.subplot(212) # options graph 1 (vertical no, horizontal no, graph no)

# # Plot Predictions vs Observations
# cmap=plt.cm.Reds
# plt.hist2d(BrO_SO, BrO_SP_Met, bins=(xedgesS, yedgesS),norm=LogNorm(),cmap=cmap,vmax=MaxS)  # Swanson
# #plt.hist2d(BrO_SO, BrO_SP_OLS, bins=(xedgesS, yedgesS),norm=LogNorm(),cmap=cmap,vmax=MaxS) # My coefficient
# plt.plot([0,4],[0,4],c='black') # Plot 1:1 line
# plt.plot([0-ErrorS,4-ErrorS],[0,4],c='black', ls='--') # Plot lower error limit
# plt.plot([0+ErrorS,4+ErrorS],[0,4],c='black',ls='--') # Plot lower error limit

# # Format x-axis
# ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
# #ax.set_xlim(0.1, 34.5)
# ax.xaxis.labelpad = 10

# # Format y-axis 1
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
# #ax.set_ylim(0.1, 22.5)
# ax.yaxis.labelpad = 10

# # Plot the axis labels
# ax.set_xlabel('Observed BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
# ax.set_ylabel('Predicted BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)

# # Plot the title
# plt.title('Surface BrO', fontsize=15, pad=10)

# # Plot the colorbar
# formatter = LogFormatter(10, labelOnlyBase=False) 
# cbar2 = plt.colorbar(ticks=[1,5,10,50], format=formatter)
# cbar2.ax.set_ylabel('Number of observations in Bin')
# cbar2.ax.set_yticklabels(['0','5','10','100'])  # vertically oriented colorbar

# ## Plot the colorbar (SIPEXII)
# #formatter = LogFormatter(10, labelOnlyBase=False) 
# #cbar1 = plt.colorbar(ticks=[1,2,5,10,20], format=formatter)
# #cbar1.ax.set_ylabel('Number of observations in Bin')
# #cbar1.ax.set_yticklabels(['0','2','5','10','20'])  # vertically oriented colorbar

# # Format axis labels
# ax.tick_params(labelsize=10)
