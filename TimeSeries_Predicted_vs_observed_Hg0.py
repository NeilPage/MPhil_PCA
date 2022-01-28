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

# Data handing packages
import numpy as np
import pandas as pd
from scipy import signal, stats

# Date and Time handling package
import datetime as dt
from datetime import datetime,time, timedelta		# functions to handle date and time

#------------------------------------------------------------------------------
# DEFINE THE DATASET

# Hg0 predictions & observations
PC_Hg0 = pd.read_csv('/Users/ncp532/Documents/Data/PCA_Hg0/Hg0_Pred_CAMMPCAN_Test.csv', index_col=0) # BrO
#PC_Hg0 = pd.read_csv('/Users/ncp532/Documents/Data/PCA_Hg0/Without_BrO/Hg0_Pred_CAMMPCAN_noBrO.csv', index_col=0) # Without BrO

#------------------------------------------------------------------------------
# SET THE DATE

PC_Hg0.index = (pd.to_datetime(PC_Hg0.index, dayfirst=True))

#------------------------------------------------------------------------------
# SET THE VARIABLES

Date = np.array(PC_Hg0.index)

#-------------
# Predictions
#-------------

Hg0_Pred = PC_Hg0['OLS_Hg0']

#-------------
# Observations
#-------------
Hg0_Obs  = PC_Hg0['Obs_Hg0']

#-------------
# StDev error
#-------------
StDev_Obs = np.std(Hg0_Obs)

#------------------------------------------------------------------------------    
# CALCULATE THE MONTHLY MEAN

# Define the formula
def monthly(x, date):
    df = pd.DataFrame({'X':x}, index=date) 
    df = df.resample('MS').mean()
    #Reset the index
    df =df.reset_index()
    #extract the values
    x=df['X']
    date=df['index']  
    #convert the pandas series date to list
    date = date.tolist()
    return x,date 

# Retrive the values 
Obs_Mavg,  date_avg = monthly(Hg0_Obs,  Date) # Hg0 observations
Pred_Mavg, date_avg = monthly(Hg0_Pred, Date) # Hg0 predictions

#------------------------------------------------------------------------------
# CONVERT MONTHLY MEAN FROM (1 value per month) TO (1 value per hour)

#-----------------------------------
# Hg0 observations
#-----------------------------------
df = pd.DataFrame((Obs_Mavg), index=date_avg) 
df = pd.DataFrame({'X':Hg0_Obs}, index=Date)
df = df.resample('MS').mean()
# Set the start and end dates
start_date = df.index.min() - pd.DateOffset(day=1)
end_date = df.index.max() + pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq='H')
dates.name = 'date'
df = df.reindex(dates, method='ffill')
df = df.reset_index()
df.index = (pd.to_datetime(df['date'], dayfirst=True))
df.rename(columns={ df.columns[1]: 'Obs_Mavg' }, inplace = True)
df = pd.concat([PC_Hg0,df],axis=1,join='inner')
# Extract the monthly mean (hourly values)
Obs_MAvgH = np.array(df['Obs_Mavg'])

#-----------------------------------
# Hg0 predictions
#-----------------------------------
df = pd.DataFrame((Pred_Mavg), index=date_avg) 
df = pd.DataFrame({'X':Hg0_Pred}, index=Date)
df = df.resample('MS').mean()
# Set the start and end dates
start_date = df.index.min() - pd.DateOffset(day=1)
end_date = df.index.max() + pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq='H')
dates.name = 'date'
df = df.reindex(dates, method='ffill')
df = df.reset_index()
df.index = (pd.to_datetime(df['date'], dayfirst=True))
df.rename(columns={ df.columns[1]: 'Pred_Mavg' }, inplace = True)
df = pd.concat([PC_Hg0,df],axis=1,join='inner')
# Extract the monthly mean (hourly values)
Pred_MAvgH = np.array(df['Pred_Mavg'])

#------------------------------------------------------------------------------
# Calculate the hourly variability (from Monthly Mean)
# (Hourly Values) - (Monthly Mean) 

HV_Obs  = Hg0_Obs  - Obs_MAvgH  # Hg0 observations
HV_Pred = Hg0_Pred - Pred_MAvgH # Hg0 predictions

#------------------------------------------------------------------------------
# CALCULATE THE MEAN NORMALISED BIAS (between predictions and observation)
# MNB = Mean((predictions-observations)/observations)*100

MNB_Hg0 = (np.mean((Hg0_Pred - Hg0_Obs) / Hg0_Obs)) * 100

#------------------------------------------------------------------------------
# CALCULATE THE MEAN NORMALISED ERROR (between predictions and observation)
# MNE = Mean((predictions-observations)/observations)*100

MNE_Hg0 = (np.mean(abs(Hg0_Pred - Hg0_Obs) / Hg0_Obs)) * 100

#------------------------------------------------------------------------------
# CORRECT THE PREDICTIONS FOR THE MEAN NORMALISED BIAS 

Corr_Pred_MNB = Hg0_Pred * (100/MNB_Hg0)

#------------------------------------------------------------------------------
# CALCULATE THE CORRELATION COEFFICIENTS (R) AND (p)

#--------------------------------
# Linear correlation coefficient (R)
#--------------------------------

slope_LR_Hg0, intercept_LR_Hg0, rval_LR_Hg0, pval_LR_Hg0, stderr_LR_Hg0 = stats.linregress(Hg0_Pred, Hg0_Obs)

#--------------------------------
# Nonparametric Spearman rank correlation coefficient (p)
#--------------------------------

rho_Spear_Hg0, pval_Spear_Hg0 = stats.spearmanr(Hg0_Pred, Hg0_Obs)

#--------------------------------
# Linear determination coefficient (R^2)
#--------------------------------

R2_LR_Hg0 = np.square(rval_LR_Hg0)

#------------------------------------------------------------------------------
#BUILD DATAFRAME FOR CORRELATION COEFFICIENTS

# Build a pandas dataframe
dfCorrCoef = {'Linear (R)':   [rval_LR_Hg0],
              'Linear (R2)':  [R2_LR_Hg0],
              'Spearman (p)': [rho_Spear_Hg0]}
dfCorrCoef = pd.DataFrame(dfCorrCoef, columns = ['Linear (R)','Linear (R2)','Spearman (p)'],index = ['OLS_Hg0'])
dfCorrCoef.to_csv('/Users/ncp532/Documents/Data/PCA_Hg0/CorrCoef.csv')

#------------------------------------------------------------------------------
# PLOT THE GRAPH

fig = plt.figure()

gs = gridspec.GridSpec(nrows=3,
                       ncols=4, 
                       figure=fig, 
                       width_ratios= [0.25,0.25,0.25,0.25],
                       height_ratios=[0.25, 0.25, 0.25],
                       hspace=0.3, wspace=0.35)

#-------------------------------------
# Graph 1 (V1 2017-18)
ax = plt.subplot(gs[0,0:2])

# Plot Hg0
ax.plot(Date,     Hg0_Pred, marker='o',    c='r',  markersize = 2.0, linestyle='none', label='Hg0 Predictions')  # Hg0 prediction
ax.plot(Date,     Hg0_Obs,  marker='o',    c='b',  markersize = 2.0, linestyle='none', label='Hg0 Observations') # Hg0 observations
ax.errorbar(Date, Hg0_Obs,  yerr=StDev_Obs,  c='b', alpha=0.5, capsize=None, lw=1.0, ls='none') # Error Hg0 observations 

# Format x-axis
plt.xlim(datetime(2017,11,14),datetime(2017,11,23)) # V1 17 (14-22 Nov 2017)
xmajor_formatter = mdates.DateFormatter('%b %Y') # format how the date is displayed
ax.xaxis.set_major_formatter(xmajor_formatter)
xminor_formatter = mdates.DateFormatter('%d') # format how the date is displayed
ax.xaxis.set_minor_formatter(xminor_formatter)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # set the interval between dispalyed dates
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.tick_params(axis='x',pad=15)

# Format y-axis 1
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V1 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('Hg$^0$ (ng/m$^3$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 2 (V2 2017-18)
ax = plt.subplot(gs[1,0:2])

# Plot Hg0
ax.plot(Date,     Hg0_Pred, marker='o',    c='r',  markersize = 2.0, linestyle='none', label='Hg0 Predictions')  # Hg0 prediction
ax.plot(Date,     Hg0_Obs,  marker='o',    c='b',  markersize = 2.0, linestyle='none', label='Hg0 Observations') # Hg0 observations
ax.errorbar(Date, Hg0_Obs,  yerr=StDev_Obs,  c='b', alpha=0.5, capsize=None, lw=1.0, ls='none') # Error Hg0 observations 

# Format x-axis
plt.xlim(datetime(2017,12,21),datetime(2018,1,6)) # V2 17 (21-22 Dec 2017 & 26 Dec - 5 Jan 2018)
xmajor_formatter = mdates.DateFormatter('%b %Y') # format how the date is displayed
ax.xaxis.set_major_formatter(xmajor_formatter)
xminor_formatter = mdates.DateFormatter('%d') # format how the date is displayed
ax.xaxis.set_minor_formatter(xminor_formatter)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # set the interval between dispalyed dates
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.tick_params(axis='x',pad=15)

# Format y-axis 1
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V2 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('Hg$^0$ (ng/m$^3$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 3 (V3 2017-18)
ax = plt.subplot(gs[2,0:2])

# Plot Hg0
ax.plot(Date,     Hg0_Pred, marker='o',    c='r',  markersize = 2.0, linestyle='none', label='Hg0 Predictions')  # Hg0 prediction
ax.plot(Date,     Hg0_Obs,  marker='o',    c='b',  markersize = 2.0, linestyle='none', label='Hg0 Observations') # Hg0 observations
ax.errorbar(Date, Hg0_Obs,  yerr=StDev_Obs,  c='b', alpha=0.5, capsize=None, lw=1.0, ls='none') # Error Hg0 observations 

# Format x-axis
plt.xlim(datetime(2018,1,27),datetime(2018,2,22)) # V3 17 (27 Jan - 21 Feb 2018)
xmajor_formatter = mdates.DateFormatter('%b %Y') # format how the date is displayed
ax.xaxis.set_major_formatter(xmajor_formatter)
xminor_formatter = mdates.DateFormatter('%d') # format how the date is displayed
ax.xaxis.set_minor_formatter(xminor_formatter)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # set the interval between dispalyed dates
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.tick_params(axis='x',pad=15)

# Format y-axis 1
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V3 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('Hg$^0$ (ng/m$^3$)', fontsize=10)
ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 4 (V1 2018-19)
ax = plt.subplot(gs[0,2:4])

# Plot Hg0
ax.plot(Date,     Hg0_Pred, marker='o',    c='r',  markersize = 2.0, linestyle='none', label='Hg$^0$ Predictions\n (R: '+str("%6.4f"%(rval_LR_Hg0))+')\n (R$^2$: '+str("%6.4f"%(R2_LR_Hg0))+')\n (p: '+str("%6.4f"%(rho_Spear_Hg0))+')\n (MNB: '+str("%6.1f"%(MNB_Hg0))+'%)\n (MNE: '+str("%6.1f"%(MNE_Hg0))+'%)')  # Hg0 prediction
ax.plot(Date,     Hg0_Obs,  marker='o',    c='b',  markersize = 2.0, linestyle='none', label='Hg$^0$ Observations') # Hg0 observations
ax.errorbar(Date, Hg0_Obs,  yerr=StDev_Obs,  c='b', alpha=0.5, capsize=None, lw=1.0, ls='none') # Error Hg0 observations 

# Format x-axis
plt.xlim(datetime(2018,11,7),datetime(2018,11,16)) # V1 18 (7-15 Nov 2018)
xmajor_formatter = mdates.DateFormatter('%b %Y') # format how the date is displayed
ax.xaxis.set_major_formatter(xmajor_formatter)
xminor_formatter = mdates.DateFormatter('%d') # format how the date is displayed
ax.xaxis.set_minor_formatter(xminor_formatter)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # set the interval between dispalyed dates
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.tick_params(axis='x',pad=15)

# Format y-axis 1
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V1 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('Hg$^0$ (ng/m$^3$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 5 (V2 2018-19)
ax = plt.subplot(gs[1,2:4])

# Plot Hg0
ax.plot(Date,     Hg0_Pred, marker='o',    c='r',  markersize = 2.0, linestyle='none', label='Hg0 Predictions')  # Hg0 prediction
ax.plot(Date,     Hg0_Obs,  marker='o',    c='b',  markersize = 2.0, linestyle='none', label='Hg0 Observations') # Hg0 observations
ax.errorbar(Date, Hg0_Obs,  yerr=StDev_Obs,  c='b', alpha=0.5, capsize=None, lw=1.0, ls='none') # Error Hg0 observations 

# Format x-axis
plt.xlim(datetime(2018,12,15),datetime(2018,12,31)) # V2 18 (15-30 Dec 2018)
xmajor_formatter = mdates.DateFormatter('%b %Y') # format how the date is displayed
ax.xaxis.set_major_formatter(xmajor_formatter)
xminor_formatter = mdates.DateFormatter('%d') # format how the date is displayed
ax.xaxis.set_minor_formatter(xminor_formatter)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # set the interval between dispalyed dates
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.tick_params(axis='x',pad=15)

# Format y-axis 1
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V2 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('Hg$^0$ (ng/m$^3$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 6 (V3 2018-19)
ax = plt.subplot(gs[2,2:4])

# Plot Hg0
ax.plot(Date,     Hg0_Pred, marker='o',    c='r',  markersize = 2.0, linestyle='none', label='Hg0 Predictions')  # Hg0 prediction
ax.plot(Date,     Hg0_Obs,  marker='o',    c='b',  markersize = 2.0, linestyle='none', label='Hg0 Observations') # Hg0 observations
ax.errorbar(Date, Hg0_Obs,  yerr=StDev_Obs,  c='b', alpha=0.5, capsize=None, lw=1.0, ls='none') # Error Hg0 observations 

# Format x-axis
plt.xlim(datetime(2019,1,26),datetime(2019,2,21)) # V3 18 (26 Jan 2019 - 20 Feb 2019)
xmajor_formatter = mdates.DateFormatter('%b %Y') # format how the date is displayed
ax.xaxis.set_major_formatter(xmajor_formatter)
xminor_formatter = mdates.DateFormatter('%d') # format how the date is displayed
ax.xaxis.set_minor_formatter(xminor_formatter)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # set the interval between dispalyed dates
ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
ax.tick_params(axis='x',pad=15)

# Format y-axis 1
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V3 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('Hg$^0$ (ng/m$^3$)', fontsize=10)
ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 1.35), loc=2, borderaxespad=0.)

