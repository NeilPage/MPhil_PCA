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
PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_PCA_PCR.csv', index_col=0) # PCA & PCR
#PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_GridBox_BestChoice.csv', index_col=0) # GridBox (BestChoice)
#PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_GridBox_Actual.csv', index_col=0) # GridBox (Actual)

#------------------------------------------------------------------------------
# SET THE DATE

PC_BrO.index = (pd.to_datetime(PC_BrO.index, dayfirst=True))

#------------------------------------------------------------------------------
# SET THE VARIABLES

Date    = PC_BrO.index

#-------------
# Predictions
#-------------
# Met
BrO_SP_Met  = PC_BrO['BrO_SurfPred_Met']/1e13  # Surf BrO using Swanson coefficients
BrO_LTP_Met = PC_BrO['BrO_LtColPred_Met']/1e13 # LTcol BrO using Swanson coefficients

BrO_SP_OLS  = PC_BrO['BrO_SurfPred_OLS']/1e13 # Surf BrO using My coefficients
BrO_LTP_OLS = PC_BrO['BrO_LTPred_OLS']/1e13   # Ltcol BrO using My coefficients

#-------------
# Observations
#-------------
BrO_SO  = PC_BrO['BrO_SurfObs']/1e13
BrO_LTO = PC_BrO['BrO_LtColObs']/1e13

#-------------
# Observational error
#-------------
ErrorS  = np.mean(PC_BrO['BrO_SurfObs_Err']/1e13)
ErrorLT = np.mean(PC_BrO['BrO_SurfObs_Err']/1e13)

#------------------------------------------------------------------------------
# CALCULATE THE CORRELATION COEFFICIENTS (R) AND (p)

#--------------------------------
# Linear correlation coefficient (R)
#--------------------------------
# Met
slope_SMet,  intercept_SMet,  rval_SMet,  pval_SMet,  stderr_SMet  = stats.linregress(BrO_SP_Met,  BrO_SO)
slope_LTMet, intercept_LTMet, rval_LTMet, pval_LTMet, stderr_LTMet = stats.linregress(BrO_LTP_Met, BrO_LTO)

slope_SOLS,  intercept_SOLS,  rval_SOLS,  pval_SOLS,  stderr_SOLS  = stats.linregress(BrO_SP_OLS,  BrO_SO)
slope_LTOLS, intercept_LTOLS, rval_LTOLS, pval_LTOLS, stderr_LTOLS = stats.linregress(BrO_LTP_OLS, BrO_LTO)

#--------------------------------
# Nonparametric Spearman rank correlation coefficient (p)
#--------------------------------
# Met
rho_SMet,  pval_SMet  = stats.spearmanr(BrO_SP_Met,  BrO_SO)
rho_LTMet, pval_LTMet = stats.spearmanr(BrO_LTP_Met, BrO_LTO)

rho_SOLS,  pval_SOLS  = stats.spearmanr(BrO_SP_OLS,  BrO_SO)
rho_LTOLS, pval_LTOLS = stats.spearmanr(BrO_LTP_OLS, BrO_LTO)

#------------------------------------------------------------------------------
#BUILD DATAFRAME FOR CORRELATION COEFFICIENTS

# Build a pandas dataframe
dfCorrCoef = {'Linear (R)': [rval_SMet,rval_LTMet,rval_SOLS,rval_LTOLS],#,rval_SM2_SurfPres,rval_LTM2_SurfPres],
              'Spearman (p)': [rho_SMet,rho_LTMet,rho_SOLS,rho_LTOLS]}#,rho_SM2_SurfPres,rho_LTM2_SurfPres]}
dfCorrCoef = pd.DataFrame(dfCorrCoef, columns = ['Linear (R)','Spearman (p)'],index = ['Surface Swanson','LTcol Swanson','Surface OLS','LTcol OLS'])#,'Surface M2 SurfPres','LTcol M2 SurfPres'])
dfCorrCoef.to_csv('/Users/ncp532/Documents/Data/MERRA2/CorrCoef.csv')

#------------------------------------------------------------------------------
# PLOT THE GRAPH

fig1 = plt.figure()
plt.subplots_adjust(hspace=0.5)

#-----------------------------------
# Graph 1 (LTcol BrO)
ax=plt.subplot(211) # options graph 1 (vertical no, horizontal no, graph no)

# Plot Predictions vs Observations
#plt.plot(BrO_LTO, BrO_LTP_Met, ls='none', marker='o', markersize='2', c='red')
plt.plot(BrO_LTO, BrO_LTP_OLS, ls='none', marker='o', markersize='2', c='red')
plt.plot([0,4],[0,4],c='black') # Plot 1:1 line

# Format x-axis
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.set_xlim(0, 3.6)
ax.xaxis.labelpad = 10

# Format y-axis 1
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.set_ylim(0, 3.6)
ax.yaxis.labelpad = 10

# Plot the axis labels
ax.set_xlabel('Observed BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
ax.set_ylabel('Predicted BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)

# Plot the title
plt.title('BrO$_L$$_T$$_c$$_o$$_l$', fontsize=15, pad=10)

# Format axis labels
ax.tick_params(labelsize=10)

#-----------------------------------
# Graph 2 (Surface BrO)
ax=plt.subplot(212) # options graph 1 (vertical no, horizontal no, graph no)

# Plot Predictions vs Observations
#plt.plot(BrO_SO, BrO_SP_Met, ls='none', marker='o', markersize='2', c='blue')
plt.plot(BrO_SO, BrO_SP_OLS, ls='none', marker='o', markersize='2', c='blue')
plt.plot([0,4],[0,4],c='black') # Plot 1:1 line

# Format x-axis
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
ax.set_xlim(0, 1.2)
ax.xaxis.labelpad = 10

# Format y-axis 1
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.set_ylim(0, 1.2)
ax.yaxis.labelpad = 10

# Plot the axis labels
ax.set_xlabel('Observed BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
ax.set_ylabel('Predicted BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)

# Plot the title
plt.title('BrO$_s$$_u$$_r$$_f$', fontsize=15, pad=10)

# Format axis labels
ax.tick_params(labelsize=10)
