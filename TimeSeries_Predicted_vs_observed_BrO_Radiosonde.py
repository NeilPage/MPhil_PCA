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

# BrO predictions & observations
PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_Radiosonde.csv', index_col=0) # Radiosonde
#PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_My.csv', index_col=0) # My Loadings

#------------------------------------------------------------------------------
# SET THE DATE

PC_BrO.index = (pd.to_datetime(PC_BrO.index, dayfirst=True))

#------------------------------------------------------------------------------
# SET THE VARIABLES

Date    = np.array(PC_BrO.index)

#-------------
# Predictions
#-------------
# Met
BrO_SP_Met   = PC_BrO['BrO_SurfPred_Met']/1e13  # Surf BrO using Swanson coefficients
BrO_LTP_Met  = PC_BrO['BrO_LtColPred_Met']/1e13 # LTcol BrO using Swanson coefficients

BrO_SP_OLS  = PC_BrO['BrO_SurfPred_OLS']/1e13   # Surf BrO using My coefficients
BrO_LTP_OLS = PC_BrO['BrO_LTPred_OLS']/1e13     # Ltcol BrO using My coefficients

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

#-------------
# StDev error
#-------------
StDev_SO = np.std(BrO_SO)
StDev_LTO = np.std(BrO_LTO)

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
SO_Mavg,      date_avg = monthly(BrO_SO,      Date) # BrO surface observations
LTO_Mavg,     date_avg = monthly(BrO_LTO,     Date) # BrO LTcol observations
SP_OLS_Mavg,  date_avg = monthly(BrO_SP_Met,  Date) # BrO surface predictions Swanson coefficientsn
LTP_OLS_Mavg, date_avg = monthly(BrO_LTP_Met, Date) # BrO LTcol predictions Swansonn coefficients
SP_Met_Mavg,  date_avg = monthly(BrO_SP_OLS,  Date) # BrO surface predictions My coefficients
LTP_Met_Mavg, date_avg = monthly(BrO_LTP_OLS, Date) # BrO LTcol predictions My coefficients

#------------------------------------------------------------------------------
# CONVERT MONTHLY MEAN FROM (1 value per month) TO (1 value per hour)

#-----------------------------------
# BrO surface observations (SO_Mavg)
#-----------------------------------
df = pd.DataFrame((SO_Mavg), index=date_avg) 
df = pd.DataFrame({'X':BrO_SO}, index=Date)
df = df.resample('MS').mean()
# Set the start and end dates
start_date = df.index.min() - pd.DateOffset(day=1)
end_date = df.index.max() + pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq='H')
dates.name = 'date'
df = df.reindex(dates, method='ffill')
df = df.reset_index()
df.index = (pd.to_datetime(df['date'], dayfirst=True))
df.rename(columns={ df.columns[1]: 'SO_Mavg' }, inplace = True)
df = pd.concat([PC_BrO,df],axis=1,join='inner')
# Extract the monthly mean (hourly values)
SO_MAvgH = np.array(df['SO_Mavg'])

#-----------------------------------
# BrO LTcol observations (LTO_Mavg)
#-----------------------------------
df = pd.DataFrame((LTO_Mavg), index=date_avg) 
df = pd.DataFrame({'X':BrO_LTO}, index=Date)
df = df.resample('MS').mean()
# Set the start and end dates
start_date = df.index.min() - pd.DateOffset(day=1)
end_date = df.index.max() + pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq='H')
dates.name = 'date'
df = df.reindex(dates, method='ffill')
df = df.reset_index()
df.index = (pd.to_datetime(df['date'], dayfirst=True))
df.rename(columns={ df.columns[1]: 'LTO_Mavg' }, inplace = True)
df = pd.concat([PC_BrO,df],axis=1,join='inner')
# Extract the monthly mean (hourly values)
LTO_MAvgH = np.array(df['LTO_Mavg'])

#-----------------------------------
# BrO surface predictions Swanson coefficient (SP_Met_Mavg)
#-----------------------------------
df = pd.DataFrame((SP_Met_Mavg), index=date_avg) 
df = pd.DataFrame({'X':BrO_SP_Met}, index=Date)
df = df.resample('MS').mean()
# Set the start and end dates
start_date = df.index.min() - pd.DateOffset(day=1)
end_date = df.index.max() + pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq='H')
dates.name = 'date'
df = df.reindex(dates, method='ffill')
df = df.reset_index()
df.index = (pd.to_datetime(df['date'], dayfirst=True))
df.rename(columns={ df.columns[1]: 'SP_Met_Mavg' }, inplace = True)
df = pd.concat([PC_BrO,df],axis=1,join='inner')
# Extract the monthly mean (hourly values)
SP_Met_MAvgH = np.array(df['SP_Met_Mavg'])

#-----------------------------------
# BrO LTcol predictions Swanson coefficient (LTP_Met_Mavg)
#-----------------------------------
df = pd.DataFrame((LTP_Met_Mavg), index=date_avg) 
df = pd.DataFrame({'X':BrO_LTP_Met}, index=Date)
df = df.resample('MS').mean()
# Set the start and end dates
start_date = df.index.min() - pd.DateOffset(day=1)
end_date = df.index.max() + pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq='H')
dates.name = 'date'
df = df.reindex(dates, method='ffill')
df = df.reset_index()
df.index = (pd.to_datetime(df['date'], dayfirst=True))
df.rename(columns={ df.columns[1]: 'LTP_Met_Mavg' }, inplace = True)
df = pd.concat([PC_BrO,df],axis=1,join='inner')
# Extract the monthly mean (hourly values)
LTP_Met_MAvgH = np.array(df['LTP_Met_Mavg'])

#-----------------------------------
# BrO surface predictions New coefficient (SP_OLS_Mavg)
#-----------------------------------
df = pd.DataFrame((SP_OLS_Mavg), index=date_avg) 
df = pd.DataFrame({'X':BrO_SP_OLS}, index=Date)
df = df.resample('MS').mean()
# Set the start and end dates
start_date = df.index.min() - pd.DateOffset(day=1)
end_date = df.index.max() + pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq='H')
dates.name = 'date'
df = df.reindex(dates, method='ffill')
df = df.reset_index()
df.index = (pd.to_datetime(df['date'], dayfirst=True))
df.rename(columns={ df.columns[1]: 'SP_OLS_Mavg' }, inplace = True)
df = pd.concat([PC_BrO,df],axis=1,join='inner')
# Extract the monthly mean (hourly values)
SP_OLS_MAvgH = np.array(df['SP_OLS_Mavg'])

#-----------------------------------
# BrO LTcol predictions New coefficient (LTP_OLS_Mavg)
#-----------------------------------
df = pd.DataFrame((LTP_OLS_Mavg), index=date_avg) 
df = pd.DataFrame({'X':BrO_LTP_OLS}, index=Date)
df = df.resample('MS').mean()
# Set the start and end dates
start_date = df.index.min() - pd.DateOffset(day=1)
end_date = df.index.max() + pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq='H')
dates.name = 'date'
df = df.reindex(dates, method='ffill')
df = df.reset_index()
df.index = (pd.to_datetime(df['date'], dayfirst=True))
df.rename(columns={ df.columns[1]: 'LTP_OLS_Mavg' }, inplace = True)
df = pd.concat([PC_BrO,df],axis=1,join='inner')
# Extract the monthly mean (hourly values)
LTP_OLS_MAvgH = np.array(df['LTP_OLS_Mavg'])

#------------------------------------------------------------------------------
# Calculate the hourly variability (from Monthly Mean)
# (Hourly Values) - (Monthly Mean) 

HV_SO      = BrO_SO      - SO_MAvgH      # Surface observations
HV_LTO     = BrO_LTO     - LTO_MAvgH     # LTcol observations
HV_SP_Met   = BrO_SP_Met - SP_Met_MAvgH  # Surface predictions Swanson coefficient
HV_LTP_Met = BrO_LTP_Met - LTP_Met_MAvgH # LTcol predictions Swanson coefficient
HV_SP_OLS  = BrO_SP_OLS  - SP_OLS_MAvgH  # Surface predictions OLS coefficient
HV_LTP_OLS = BrO_LTP_OLS - LTP_OLS_MAvgH # LTcol predictions OLS coefficient

#------------------------------------------------------------------------------
# CALCULATE THE MEAN NORMALISED BIAS (between predictions and observation)
# MNB = Mean((predictions-observations)/observations)*100

MNB_SP_Met  = (np.mean((BrO_SP_Met  - BrO_SO)  / BrO_SO))  * 100 # Surface predictions Swanson coefficient
MNB_LTP_Met = (np.mean((BrO_LTP_Met - BrO_LTO) / BrO_LTO)) * 100 # LTcol predictions Swanson coefficient
MNB_SP_OLS  = (np.mean((BrO_SP_OLS  - BrO_SO)  / BrO_SO))  * 100 # Surface predictions OLS coefficient
MNB_LTP_OLS = (np.mean((BrO_LTP_OLS - BrO_LTO) / BrO_LTO)) * 100 # LTcol predictions OLS coefficient

#------------------------------------------------------------------------------
# CALCULATE THE MEAN NORMALISED ERROR (between predictions and observation)
# MNE = Mean((predictions-observations)/observations)*100

MNE_SP_Met  = (np.mean(abs(BrO_SP_Met  - BrO_SO)  / BrO_SO))  * 100 # Surface predictions Swanson coefficient
MNE_LTP_Met = (np.mean(abs(BrO_LTP_Met - BrO_LTO) / BrO_LTO)) * 100 # LTcol predictions Swanson coefficient
MNE_SP_OLS  = (np.mean(abs(BrO_SP_OLS  - BrO_SO)  / BrO_SO))  * 100 # Surface predictions OLS coefficient
MNE_LTP_OLS = (np.mean(abs(BrO_LTP_OLS - BrO_LTO) / BrO_LTO)) * 100 # LTcol predictions OLS coefficient

#------------------------------------------------------------------------------
# CORRECT THE PREDICTIONS FOR THE MEAN NORMALISED BIAS 

SP_MMNB_Met  = BrO_SP_Met  * (100/MNB_SP_Met)   # Surface predictions Swanson coefficient
LTP_MMNB_Met = BrO_LTP_Met * (100/MNB_LTP_Met)  # LTcol predictions Swanson coefficient
SP_MMNB_OLS  = BrO_SP_OLS  * (100/MNB_SP_OLS)  # Surface predictions OLS coefficient
LTP_MMNB_OLS = BrO_LTP_OLS * (100/MNB_LTP_OLS) # LTcol predictions OLS coefficient

#------------------------------------------------------------------------------
# CALCULATE THE CORRELATION COEFFICIENTS (R) AND (p)

#--------------------------------
# Linear correlation coefficient (R)
#--------------------------------
# Met
slope_SMet,  intercept_SMet,  rval_SMet,  pval_SMet,  stderr_SMet  = stats.linregress(BrO_SP_Met,  BrO_SO)
slope_LTMet, intercept_LTMet, rval_LTMet, pval_LTMet, stderr_LTMet = stats.linregress(BrO_LTP_Met, BrO_LTO)

# OLS
slope_SOLS,  intercept_SOLS,  rval_SOLS,  pval_SOLS,  stderr_SOLS  = stats.linregress(BrO_SP_OLS,  BrO_SO)
slope_LTOLS, intercept_LTOLS, rval_LTOLS, pval_LTOLS, stderr_LTOLS = stats.linregress(BrO_LTP_OLS, BrO_LTO)

#--------------------------------
# Nonparametric Spearman rank correlation coefficient (p)
#--------------------------------
# Met
rho_SMet,  pval_SMet  = stats.spearmanr(BrO_SP_Met,  BrO_SO)
rho_LTMet, pval_LTMet = stats.spearmanr(BrO_LTP_Met, BrO_LTO)

# OLS
rho_SOLS,  pval_SOLS  = stats.spearmanr(BrO_SP_OLS,  BrO_SO)
rho_LTOLS, pval_LTOLS = stats.spearmanr(BrO_LTP_OLS, BrO_LTO)

#--------------------------------
# Linear determination coefficient (R^2)
#--------------------------------

R2_SMet  = np.square(rval_SMet)  # Surface predictions Swanson coefficient
R2_LTMet = np.square(rval_LTMet) # Surface predictions Swanson coefficient
R2_SOLS  = np.square(rval_SOLS)  # Surface predictions OLS coefficient
R2_LTOLS = np.square(rval_LTOLS) # Surface predictions OLS coefficient

#------------------------------------------------------------------------------
#BUILD DATAFRAME FOR CORRELATION COEFFICIENTS

# Build a pandas dataframe
dfCorrCoef = {'Linear (R)':   [rval_SMet,rval_LTMet,rval_SOLS,rval_LTOLS],#,rval_SM2_SurfPres,rval_LTM2_SurfPres],
              'Linear (R2)':  [R2_SMet,R2_LTMet,R2_SOLS,R2_LTOLS],
              'Spearman (p)': [rho_SMet,rho_LTMet,rho_SOLS,rho_LTOLS]}#,rho_SM2_SurfPres,rho_LTM2_SurfPres]}
dfCorrCoef = pd.DataFrame(dfCorrCoef, columns = ['Linear (R)','Linear (R2)','Spearman (p)'],index = ['Surface Met','LTcol Met','Surface OLS','LTcol OLS'])#,'Surface M2 SurfPres','LTcol M2 SurfPres'])
dfCorrCoef.to_csv('/Users/ncp532/Documents/Data/MERRA2/CorrCoef.csv')

#------------------------------------------------------------------------------
# PLOT THE GRAPH (Surface BrO)

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

# Plot BrO
ax.plot(Date, BrO_SP_Met,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, BrO_SP_OLS,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (OLS coefficients)
ax.plot(Date, BrO_SO,       marker='o', c='b',      markersize = 2.0, linestyle='none', label='BrO Surface Observations') # Surf BrO observations
ax.errorbar(Date, BrO_SO, yerr=StDev_SO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                                 # Error surf observations 

# Plot the corrected BrO prediction
#ax.plot(Date, SP_MMNB_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='Corrected BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)

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
ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 2 (V2 2017-18)
ax = plt.subplot(gs[1,0:2])

# Plot BrO
ax.plot(Date, BrO_SP_Met,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, BrO_SP_OLS,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (OLS coefficients)
ax.plot(Date, BrO_SO,       marker='o', c='b',      markersize = 2.0, linestyle='none', label='BrO Surface Observations') # Surf BrO observations
ax.errorbar(Date, BrO_SO, yerr=StDev_SO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                                 # Error surf observations 

# Plot the corrected BrO prediction
#ax.plot(Date, SP_MMNB_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='Corrected BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)

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
ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 3 (V3 2017-18)
ax = plt.subplot(gs[2,0:2])

# Plot BrO
ax.plot(Date, BrO_SP_Met,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, BrO_SP_OLS,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (OLS coefficients)
ax.plot(Date, BrO_SO,       marker='o', c='b',      markersize = 2.0, linestyle='none', label='BrO Surface Observations') # Surf BrO observations
ax.errorbar(Date, BrO_SO, yerr=StDev_SO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                                 # Error surf observations 

# Plot the corrected BrO prediction
#ax.plot(Date, SP_MMNB_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='Corrected BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)

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
ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

# #-------------------------------------
# # Graph 4 (V1 2018-19)
# ax = plt.subplot(gs[0,2:4])

# # Plot BrO
# ax.plot(Date, BrO_SP_Met,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions\n (R: '+str("%6.4f"%(rval_SMet))+')\n (R$^2$: '+str("%6.4f"%(R2_SMet))+')\n (p: '+str("%6.4f"%(rho_SMet))+')\n (MNB: '+str("%6.1f"%(MNB_SP_Met))+'%)\n (MNE: '+str("%6.1f"%(MNE_SP_Met))+'%)') # Surf BrO prediction (Swanson coefficients)
# #ax.plot(Date, BrO_SP_OLS,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (OLS coefficients)
# ax.plot(Date, BrO_SO,       marker='o', c='b',      markersize = 2.0, linestyle='none', label='BrO Surface Observations') # Surf BrO observations
# ax.errorbar(Date, BrO_SO, yerr=StDev_SO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                                 # Error surf observations 

# # Plot the corrected BrO prediction
# #ax.plot(Date, SP_MMNB_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='Corrected BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)

# # Format x-axis
# plt.xlim(datetime(2018,11,7),datetime(2018,11,16)) # V1 18 (7-15 Nov 2018)
# xmajor_formatter = mdates.DateFormatter('%b %Y') # format how the date is displayed
# ax.xaxis.set_major_formatter(xmajor_formatter)
# xminor_formatter = mdates.DateFormatter('%d') # format how the date is displayed
# ax.xaxis.set_minor_formatter(xminor_formatter)
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # set the interval between dispalyed dates
# ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
# ax.tick_params(axis='x',pad=15)

# # Format y-axis 1
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
# ax.set_ylim(0,)

# # Plot the axis labels, legend and title
# plt.title('V1 (2018-19)', fontsize=15, y=1.05)
# #ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
# #ax.set_xlabel('Date', fontsize=15)
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)

# #-------------------------------------
# # Graph 5 (V2 2018-19)
# ax = plt.subplot(gs[1,2:4])

# # Plot BrO
# ax.plot(Date, BrO_SP_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)
# #ax.plot(Date, BrO_SP_Met,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (my coefficients)
# ax.plot(Date, BrO_SO,       marker='o', c='b',      markersize = 2.0, linestyle='none', label='BrO Surface Observations') # Surf BrO observations
# ax.errorbar(Date, BrO_SO, yerr=StDev_SO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                                 # Error surf observations 

# # Plot the corrected BrO prediction
# #ax.plot(Date, SP_MMNB_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='Corrected BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)

# # Format x-axis
# plt.xlim(datetime(2018,12,15),datetime(2018,12,31)) # V2 18 (15-30 Dec 2018)
# xmajor_formatter = mdates.DateFormatter('%b %Y') # format how the date is displayed
# ax.xaxis.set_major_formatter(xmajor_formatter)
# xminor_formatter = mdates.DateFormatter('%d') # format how the date is displayed
# ax.xaxis.set_minor_formatter(xminor_formatter)
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # set the interval between dispalyed dates
# ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
# ax.tick_params(axis='x',pad=15)

# # Format y-axis 1
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
# ax.set_ylim(0,)

# # Plot the axis labels, legend and title
# plt.title('V2 (2018-19)', fontsize=15, y=1.05)
# #ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
# #ax.set_xlabel('Date', fontsize=15)
# #plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

# #-------------------------------------
# # Graph 6 (V3 2018-19)
# ax = plt.subplot(gs[2,2:4])

# # Plot BrO
# ax.plot(Date, BrO_SP_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)
# #ax.plot(Date, BrO_SP_Met,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (my coefficients)
# ax.plot(Date, BrO_SO,       marker='o', c='b',      markersize = 2.0, linestyle='none', label='BrO Surface Observations') # Surf BrO observations
# ax.errorbar(Date, BrO_SO, yerr=StDev_SO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                                 # Error surf observations 

# # Plot the corrected BrO prediction
# #ax.plot(Date, SP_MMNB_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='Corrected BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)

# # Format x-axis
# plt.xlim(datetime(2019,1,26),datetime(2019,2,21)) # V3 18 (26 Jan 2019 - 20 Feb 2019)
# xmajor_formatter = mdates.DateFormatter('%b %Y') # format how the date is displayed
# ax.xaxis.set_major_formatter(xmajor_formatter)
# xminor_formatter = mdates.DateFormatter('%d') # format how the date is displayed
# ax.xaxis.set_minor_formatter(xminor_formatter)
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # set the interval between dispalyed dates
# ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
# ax.tick_params(axis='x',pad=15)

# # Format y-axis 1
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
# ax.set_ylim(0,)

# # Plot the axis labels, legend and title
# plt.title('V3 (2018-19)', fontsize=15, y=1.05)
# #ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
# ax.set_xlabel('Date', fontsize=15)
# #plt.legend(bbox_to_anchor=(0.85, 1.35), loc=2, borderaxespad=0.)

##-------------------------------------
## Graph 7 (SIPEXII 2012)
#ax = plt.subplot(gs[3,1:3])
#
## Plot BrO
#ax.plot(Date, BrO_SP_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, BrO_SP_Met,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (my coefficients)
#ax.plot(Date, BrO_SO,       marker='o', c='b',      markersize = 2.0, linestyle='none', label='BrO Surface Observations') # Surf BrO observations
#ax.errorbar(Date, BrO_SO, yerr=ErrorS, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                                 # Error surf observations 
#
## Format x-axis
#plt.xlim(datetime(2012,9,23),datetime(2012,11,12)) # SIPEXII (23 Sep - 11 Nov 2012)
#xmajor_formatter = mdates.DateFormatter('%b %Y') # format how the date is displayed
#ax.xaxis.set_major_formatter(xmajor_formatter)
#xminor_formatter = mdates.DateFormatter('%d') # format how the date is displayed
#ax.xaxis.set_minor_formatter(xminor_formatter)
#ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # set the interval between dispalyed dates
#ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
#ax.tick_params(axis='x',pad=15)
#
## Format y-axis 1
#ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
#ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
#ax.set_ylim(0,)

## Plot the axis labels, legend and title
##plt.title('Comparison of observations vs predictions', fontsize=25, y=1.05)
#ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(1.10, 0.5), loc=2, borderaxespad=0.)

#------------------------------------------------------------------------------
# PLOT THE GRAPH (LTcol BrO)

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

# Plot BrO
ax.plot(Date, BrO_LTP_Met,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, BrO_LTP_OLS,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (OLS coefficients)
ax.plot(Date, BrO_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations
ax.errorbar(Date, BrO_LTO, yerr=StDev_LTO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                           # Error LTcol observations

# Plot the corrected BrO prediction
#ax.plot(Date, LTP_MMNB_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='Corrected BrO LTcol Predictions')  # LTcol BrO prediction (Swanson coefficients)

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
ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 2 (V2 2017-18)
ax = plt.subplot(gs[1,0:2])

# Plot BrO
ax.plot(Date, BrO_LTP_Met,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, BrO_LTP_OLS,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (OLS coefficients)
ax.plot(Date, BrO_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations
ax.errorbar(Date, BrO_LTO, yerr=StDev_LTO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                           # Error LTcol observations

# Plot the corrected BrO prediction
#ax.plot(Date, LTP_MMNB_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='Corrected BrO LTcol Predictions')  # LTcol BrO prediction (Swanson coefficients)

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
ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 3 (V3 2017-18)
ax = plt.subplot(gs[2,0:2])

# Plot BrO
ax.plot(Date, BrO_LTP_Met,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, BrO_LTP_OLS,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (OLS coefficients)
ax.plot(Date, BrO_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations
ax.errorbar(Date, BrO_LTO, yerr=StDev_LTO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                           # Error LTcol observations

# Plot the corrected BrO prediction
#ax.plot(Date, LTP_MMNB_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='Corrected BrO LTcol Predictions')  # LTcol BrO prediction (Swanson coefficients)

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
ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

# #-------------------------------------
# # Graph 4 (V1 2018-19)
# ax = plt.subplot(gs[0,2:4])

# # Plot BrO
# ax.plot(Date, BrO_LTP_Met,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions\n (R: '+str("%6.4f"%(rval_LTMet))+')\n (R$^2$: '+str("%6.4f"%(R2_LTMet))+')\n (p: '+str("%6.4f"%(rho_LTMet))+')\n (MNB: '+str("%6.1f"%(MNB_LTP_Met))+'%)\n (MNE: '+str("%6.1f"%(MNE_LTP_Met))+'%)') # Surf BrO prediction (Swanson coefficients)
# #ax.plot(Date, BrO_LTP_OLS,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (OLS coefficients)
# ax.plot(Date, BrO_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations
# ax.errorbar(Date, BrO_LTO, yerr=StDev_LTO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                           # Error LTcol observations

# # Plot the corrected BrO prediction
# #ax.plot(Date, LTP_MMNB_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='Corrected BrO LTcol Predictions')  # LTcol BrO prediction (Swanson coefficients)

# # Format x-axis
# plt.xlim(datetime(2018,11,7),datetime(2018,11,16)) # V1 18 (7-15 Nov 2018)
# xmajor_formatter = mdates.DateFormatter('%b %Y') # format how the date is displayed
# ax.xaxis.set_major_formatter(xmajor_formatter)
# xminor_formatter = mdates.DateFormatter('%d') # format how the date is displayed
# ax.xaxis.set_minor_formatter(xminor_formatter)
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # set the interval between dispalyed dates
# ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
# ax.tick_params(axis='x',pad=15)

# # Format y-axis 1
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
# ax.set_ylim(0,)

# # Plot the axis labels, legend and title
# plt.title('V1 (2018-19)', fontsize=15, y=1.05)
# #ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
# #ax.set_xlabel('Date', fontsize=15)
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)

# #-------------------------------------
# # Graph 5 (V2 2018-19)
# ax = plt.subplot(gs[1,2:4])

# # Plot BrO
# ax.plot(Date, BrO_LTP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (Swanson coefficients)
# #ax.plot(Date, BrO_LTP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
# ax.plot(Date, BrO_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations
# ax.errorbar(Date, BrO_LTO, yerr=StDev_LTO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                           # Error LTcol observations

# # Plot the corrected BrO prediction
# #ax.plot(Date, LTP_MMNB_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='Corrected BrO LTcol Predictions')  # LTcol BrO prediction (Swanson coefficients)

# # Format x-axis
# plt.xlim(datetime(2018,12,15),datetime(2018,12,31)) # V2 18 (15-30 Dec 2018)
# xmajor_formatter = mdates.DateFormatter('%b %Y') # format how the date is displayed
# ax.xaxis.set_major_formatter(xmajor_formatter)
# xminor_formatter = mdates.DateFormatter('%d') # format how the date is displayed
# ax.xaxis.set_minor_formatter(xminor_formatter)
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # set the interval between dispalyed dates
# ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
# ax.tick_params(axis='x',pad=15)

# # Format y-axis 1
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
# ax.set_ylim(0,)

# # Plot the axis labels, legend and title
# plt.title('V2 (2018-19)', fontsize=15, y=1.05)
# #ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
# #ax.set_xlabel('Date', fontsize=15)
# #plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

# #-------------------------------------
# # Graph 6 (V3 2018-19)
# ax = plt.subplot(gs[2,2:4])

# # Plot BrO
# ax.plot(Date, BrO_LTP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (Swanson coefficients)
# #ax.plot(Date, BrO_LTP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
# ax.plot(Date, BrO_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations
# ax.errorbar(Date, BrO_LTO, yerr=StDev_LTO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                           # Error LTcol observations

# # Plot the corrected BrO prediction
# #ax.plot(Date, LTP_MMNB_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='Corrected BrO LTcol Predictions')  # LTcol BrO prediction (Swanson coefficients)

# # Format x-axis
# plt.xlim(datetime(2019,1,26),datetime(2019,2,21)) # V3 18 (26 Jan 2019 - 20 Feb 2019)
# xmajor_formatter = mdates.DateFormatter('%b %Y') # format how the date is displayed
# ax.xaxis.set_major_formatter(xmajor_formatter)
# xminor_formatter = mdates.DateFormatter('%d') # format how the date is displayed
# ax.xaxis.set_minor_formatter(xminor_formatter)
# ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # set the interval between dispalyed dates
# ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
# ax.tick_params(axis='x',pad=15)

# # Format y-axis 1
# ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
# ax.set_ylim(0,)

# # Plot the axis labels, legend and title
# plt.title('V3 (2018-19)', fontsize=15, y=1.05)
# #ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
# ax.set_xlabel('Date', fontsize=15)
# #plt.legend(bbox_to_anchor=(0.85, 1.35), loc=2, borderaxespad=0.)

##-------------------------------------
## Graph 7 (SIPEXII 2012)
#ax = plt.subplot(gs[3,1:3])
#
## Plot BrO
#ax.plot(Date, BrO_LTP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, BrO_LTP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
#ax.plot(Date, BrO_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations
#ax.errorbar(Date, BrO_LTO, yerr=ErrorS, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                           # Error LTcol observations
#
## Format x-axis
#plt.xlim(datetime(2012,9,23),datetime(2012,11,12)) # SIPEXII (23 Sep - 11 Nov 2012)
#xmajor_formatter = mdates.DateFormatter('%b %Y') # format how the date is displayed
#ax.xaxis.set_major_formatter(xmajor_formatter)
#xminor_formatter = mdates.DateFormatter('%d') # format how the date is displayed
#ax.xaxis.set_minor_formatter(xminor_formatter)
#ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # set the interval between dispalyed dates
#ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
#ax.tick_params(axis='x',pad=15)
#
## Format y-axis 1
#ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
#ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
#ax.set_ylim(0,)

## Plot the axis labels, legend and title
##plt.title('Comparison of observations vs predictions', fontsize=25, y=1.05)
#ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(1.10, 0.5), loc=2, borderaxespad=0.)
