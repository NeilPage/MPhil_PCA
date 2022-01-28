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
#PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN.csv', index_col=0) # CAMMPCAN (Swanson Loadings)
PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_SIPEXII.csv', index_col=0) # SIPEXII (Swanson Loadings)
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
## MERRA2 - SLP
#BrO_SP_SW  = PC_BrO['BrO_SurfPred_MERRA2_SLP']/1e13
#BrO_LTP_SW = PC_BrO['BrO_LtColPred_MERRA2_SLP']/1e13
#
## MERRA2 - SurfPres
#BrO_SP_M2_SurfPres  = PC_BrO['BrO_SurfPred_MERRA2_SurfPres']/1e13
#BrO_LTP_M2_SurfPres = PC_BrO['BrO_LtColPred_MERRA2_SurfPres']/1e13

# Met
BrO_SP_SW  = PC_BrO['BrO_SurfPred_Met']/1e13  # Surf BrO using Swanson coefficients
BrO_LTP_SW = PC_BrO['BrO_LtColPred_Met']/1e13 # LTcol BrO using Swanson coefficients

BrO_SP_Met  = PC_BrO['BrO_SurfPred_OLS']/1e13 # Surf BrO using My coefficients
BrO_LTP_Met = PC_BrO['BrO_LTPred_OLS']/1e13   # Ltcol BrO using My coefficients

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
SP_SW_Mavg,   date_avg = monthly(BrO_SP_SW,   Date) # BrO surface predictions Swanso coefficientsn
LTP_SW_Mavg,  date_avg = monthly(BrO_LTP_SW,  Date) # BrO LTcol predictions Swanson coefficients
SP_Met_Mavg,  date_avg = monthly(BrO_SP_Met,  Date) # BrO surface predictions My coefficients
LTP_Met_Mavg, date_avg = monthly(BrO_LTP_Met, Date) # BrO LTcol predictions My coefficients

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
# BrO surface predictions Swanson coefficient (SP_SW_Mavg)
#-----------------------------------
df = pd.DataFrame((SP_SW_Mavg), index=date_avg) 
df = pd.DataFrame({'X':BrO_SP_SW}, index=Date)
df = df.resample('MS').mean()
# Set the start and end dates
start_date = df.index.min() - pd.DateOffset(day=1)
end_date = df.index.max() + pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq='H')
dates.name = 'date'
df = df.reindex(dates, method='ffill')
df = df.reset_index()
df.index = (pd.to_datetime(df['date'], dayfirst=True))
df.rename(columns={ df.columns[1]: 'SP_SW_Mavg' }, inplace = True)
df = pd.concat([PC_BrO,df],axis=1,join='inner')
# Extract the monthly mean (hourly values)
SP_SW_MAvgH = np.array(df['SP_SW_Mavg'])

#-----------------------------------
# BrO LTcol predictions Swanson coefficient (LTP_SW_Mavg)
#-----------------------------------
df = pd.DataFrame((LTP_SW_Mavg), index=date_avg) 
df = pd.DataFrame({'X':BrO_LTP_SW}, index=Date)
df = df.resample('MS').mean()
# Set the start and end dates
start_date = df.index.min() - pd.DateOffset(day=1)
end_date = df.index.max() + pd.DateOffset(day=31)
dates = pd.date_range(start_date, end_date, freq='H')
dates.name = 'date'
df = df.reindex(dates, method='ffill')
df = df.reset_index()
df.index = (pd.to_datetime(df['date'], dayfirst=True))
df.rename(columns={ df.columns[1]: 'LTP_SW_Mavg' }, inplace = True)
df = pd.concat([PC_BrO,df],axis=1,join='inner')
# Extract the monthly mean (hourly values)
LTP_SW_MAvgH = np.array(df['LTP_SW_Mavg'])

#-----------------------------------
# BrO surface predictions New coefficient (SP_Met_Mavg)
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
# BrO LTcol predictions New coefficient (LTP_Met_Mavg)
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

#------------------------------------------------------------------------------
# Calculate the hourly variability (from Monthly Mean)
# (Hourly Values) - (Monthly Mean) 

HV_SO      = BrO_SO      - SO_MAvgH      # Surface observations
HV_LTO     = BrO_LTO     - LTO_MAvgH     # LTcol observations
HV_SP_SW   = BrO_SP_SW   - SP_SW_MAvgH   # Surface predictions Swanson coefficient
HV_LTP_SW  = BrO_LTP_SW  - LTP_SW_MAvgH  # LTcol predictions Swanson coefficient
HV_SP_Met  = BrO_SP_Met  - SP_Met_MAvgH  # Surface predictions New coefficient
HV_LTP_Met = BrO_LTP_Met - LTP_Met_MAvgH # LTcol predictions New coefficient

#------------------------------------------------------------------------------
# CALCULATE THE MEAN NORMALISED BIAS (between predictions and observation)
# MNB = Mean((predictions-observations)/observations)*100

MNB_SP_SW   = (np.mean((BrO_SP_SW   - BrO_SO)  / BrO_SO))  * 100 # Surface predictions Swanson coefficient
MNB_LTP_SW  = (np.mean((BrO_LTP_SW  - BrO_LTO) / BrO_LTO)) * 100 # LTcol predictions Swanson coefficient
MNB_SP_Met  = (np.mean((BrO_SP_Met  - BrO_SO)  / BrO_SO))  * 100 # Surface predictions New coefficient
MNB_LTP_Met = (np.mean((BrO_LTP_Met - BrO_LTO) / BrO_LTO)) * 100 # LTcol predictions New coefficient

#------------------------------------------------------------------------------
# CALCULATE THE MEAN NORMALISED ERROR (between predictions and observation)
# MNE = Mean((predictions-observations)/observations)*100

MNE_SP_SW   = (np.mean(abs(BrO_SP_SW   - BrO_SO)  / BrO_SO))  * 100 # Surface predictions Swanson coefficient
MNE_LTP_SW  = (np.mean(abs(BrO_LTP_SW  - BrO_LTO) / BrO_LTO)) * 100 # LTcol predictions Swanson coefficient
MNE_SP_Met  = (np.mean(abs(BrO_SP_Met  - BrO_SO)  / BrO_SO))  * 100 # Surface predictions New coefficient
MNE_LTP_Met = (np.mean(abs(BrO_LTP_Met - BrO_LTO) / BrO_LTO)) * 100 # LTcol predictions New coefficient

#------------------------------------------------------------------------------
# CORRECT THE PREDICTIONS FOR THE MEAN NORMALISED BIAS 

SP_MMNB_SW   = BrO_SP_SW   * (100/MNB_SP_SW)   # Surface predictions Swanson coefficient
LTP_MMNB_SW  = BrO_LTP_SW  * (100/MNB_LTP_SW)  # LTcol predictions Swanson coefficient
SP_MMNB_Met  = BrO_SP_Met  * (100/MNB_SP_Met)  # Surface predictions New coefficient
LTP_MMNB_Met = BrO_LTP_Met * (100/MNB_LTP_Met) # LTcol predictions New coefficient

#------------------------------------------------------------------------------
# CALCULATE THE CORRELATION COEFFICIENTS (R) AND (p)

#--------------------------------
# Linear correlation coefficient (R)
#--------------------------------
# Met
slope_SMet,  intercept_SMet,  rval_SMet,  pval_SMet,  stderr_SMet  = stats.linregress(BrO_SP_Met,  BrO_SO)
slope_LTMet, intercept_LTMet, rval_LTMet, pval_LTMet, stderr_LTMet = stats.linregress(BrO_LTP_Met, BrO_LTO)

slope_SSW,  intercept_SSW,  rval_SSW,  pval_SSW,  stderr_SSW  = stats.linregress(BrO_SP_SW,  BrO_SO)
slope_LTSW, intercept_LTSW, rval_LTSW, pval_LTSW, stderr_LTSW = stats.linregress(BrO_LTP_SW, BrO_LTO)

## MERRA-2 only (SLP)
#slope_SM2_SLP,  intercept_SM2_SLP,  rval_SM2_SLP,  pval_SM2_SLP,  stderr_SM2_SLP  = stats.linregress(BrO_SP_M2_SLP,  BrO_SO)
#slope_LTM2_SLP, intercept_LTM2_SLP, rval_LTM2_SLP, pval_LTM2_SLP, stderr_LTM2_SLP = stats.linregress(BrO_LTP_M2_SLP, BrO_LTO)
#
## MERRA-2 only (SurfPres)
#slope_SM2_SurfPres,  intercept_SM2_SurfPres,  rval_SM2_SurfPres,  pval_SM2_SurfPres,  stderr_SM2_SurfPres  = stats.linregress(BrO_SP_M2_SurfPres,  BrO_SO)
#slope_LTM2_SurfPres, intercept_LTM2_SurfPres, rval_LTM2_SurfPres, pval_LTM2_SurfPres, stderr_LTM2_SurfPres = stats.linregress(BrO_LTP_M2_SurfPres, BrO_LTO)

#--------------------------------
# Nonparametric Spearman rank correlation coefficient (p)
#--------------------------------
# Met
rho_SMet,  pval_SMet  = stats.spearmanr(BrO_SP_Met,  BrO_SO)
rho_LTMet, pval_LTMet = stats.spearmanr(BrO_LTP_Met, BrO_LTO)

rho_SSW,  pval_SSW  = stats.spearmanr(BrO_SP_SW,  BrO_SO)
rho_LTSW, pval_LTSW = stats.spearmanr(BrO_LTP_SW, BrO_LTO)

## MERRA-2 only (SLP)
#rho_SM2_SLP,  pval_SM2_SLP  = stats.spearmanr(BrO_SP_M2_SLP,  BrO_SO)
#rho_LTM2_SLP, pval_LTM2_SLP = stats.spearmanr(BrO_LTP_M2_SLP, BrO_LTO)
#
## MERRA-2 only (SurfPres)
#rho_SM2_SurfPres,  pval_SM2_SurfPres  = stats.spearmanr(BrO_SP_M2_SurfPres,  BrO_SO)
#rho_LTM2_SurfPres, pval_LTM2_SurfPres = stats.spearmanr(BrO_LTP_M2_SurfPres, BrO_LTO)

#--------------------------------
# Linear determination coefficient (R^2)
#--------------------------------

R2_SSW   = np.square(rval_SSW)   # Surface predictions Swanson coefficient
R2_LTSW  = np.square(rval_LTSW)  # Surface predictions Swanson coefficient
R2_SMet  = np.square(rval_SMet)  # Surface predictions New coefficient
R2_LTMet = np.square(rval_LTMet) # Surface predictions New coefficient

#------------------------------------------------------------------------------
#BUILD DATAFRAME FOR CORRELATION COEFFICIENTS

# Build a pandas dataframe
dfCorrCoef = {'Linear (R)':   [rval_SMet,rval_LTMet,rval_SSW,rval_LTSW],#,rval_SM2_SurfPres,rval_LTM2_SurfPres],
              'Linear (R2)':  [R2_SMet,R2_LTMet,R2_SSW,R2_LTSW],
              'Spearman (p)': [rho_SMet,rho_LTMet,rho_SSW,rho_LTSW]}#,rho_SM2_SurfPres,rho_LTM2_SurfPres]}
dfCorrCoef = pd.DataFrame(dfCorrCoef, columns = ['Linear (R)','Linear (R2)','Spearman (p)'],index = ['Surface Met','LTcol Met','Surface Swanson','LTcol SWanson'])#,'Surface M2 SurfPres','LTcol M2 SurfPres'])
dfCorrCoef.to_csv('/Users/ncp532/Documents/Data/MERRA2/CorrCoef.csv')

#------------------------------------------------------------------------------
# PLOT THE GRAPH (Surface BrO)

n = 4; m = 11; o = 3;
fig = plt.figure(figsize=(10,6))
plt.subplots_adjust(top=0.96,bottom=0.15,left=0.08,right=1.0,hspace=0.15, wspace=0.15)
gs = gridspec.GridSpec(ncols=3, nrows=2, width_ratios=[n, m, o])

#-------------------------------------
# Graph 1 (LTcol)
ax = plt.subplot(gs[0])

# HIDE THE SPINES TO THE RIGHT OF SUBPLOT 1
ax.spines['right'].set_visible(False)

# Plot the BrO Pred vs Obs
ax.plot(Date, BrO_LTP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions\n (R: '+str("%6.4f"%(rval_LTSW))+')\n (R$^2$: '+str("%6.4f"%(R2_LTSW))+')\n (p: '+str("%6.4f"%(rho_LTSW))+')\n (MNB: '+str("%6.1f"%(MNB_LTP_SW))+'%)\n (MNE: '+str("%6.1f"%(MNE_LTP_SW))+'%)') # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, BrO_LTP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
ax.plot(Date, BrO_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations
ax.errorbar(Date, BrO_LTO, yerr=StDev_LTO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                           # Error LTcol observations

# Format x-axis
#plt.xlim(datetime(2012,9,23),datetime(2012,11,12)) # SIPEXII (23 Sep - 11 Nov 2012)
plt.xlim(datetime(2012,9,22,0,1,0),datetime(2012,9,26,0,0,5)) # Period 1
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
#plt.title('V1 (2018-19)', fontsize=15, y=1.05)
ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 2 (LTcol)
ax = plt.subplot(gs[1])

# HIDE THE SPINES TO THE LEFT OF SUBPLOT 2
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

# Plot the BrO Pred vs Obs
ax.plot(Date, BrO_LTP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions\n (R: '+str("%6.4f"%(rval_LTSW))+')\n (R$^2$: '+str("%6.4f"%(R2_LTSW))+')\n (p: '+str("%6.4f"%(rho_LTSW))+')\n (MNB: '+str("%6.1f"%(MNB_LTP_SW))+'%)\n (MNE: '+str("%6.1f"%(MNE_LTP_SW))+'%)') # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, BrO_LTP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
ax.plot(Date, BrO_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations
ax.errorbar(Date, BrO_LTO, yerr=StDev_LTO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                           # Error LTcol observations

# Format x-axis
#plt.xlim(datetime(2012,9,23),datetime(2012,11,12)) # SIPEXII (23 Sep - 11 Nov 2012)
plt.xlim(datetime(2012,10,9,0,1,0),datetime(2012,10,20,0,0,5)) # Period 2
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
ax.axes.get_yaxis().set_visible(False)

# Plot the axis labels, legend and title
#plt.title('V1 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 3 (LTcol)
ax = plt.subplot(gs[2])

# HIDE THE SPINES TO THE LEFT OF SUBPLOT 2
ax.spines['left'].set_visible(False)
#ax.spines['right'].set_visible(False)

# Plot the BrO Pred vs Obs
ax.plot(Date, BrO_LTP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions\n (R: '+str("%6.4f"%(rval_LTSW))+')\n (R$^2$: '+str("%6.4f"%(R2_LTSW))+')\n (p: '+str("%6.4f"%(rho_LTSW))+')\n (MNB: '+str("%6.1f"%(MNB_LTP_SW))+'%)\n (MNE: '+str("%6.1f"%(MNE_LTP_SW))+'%)') # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, BrO_LTP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
ax.plot(Date, BrO_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations
ax.errorbar(Date, BrO_LTO, yerr=StDev_LTO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                           # Error LTcol observations

# Format x-axis
#plt.xlim(datetime(2012,9,23),datetime(2012,11,12)) # SIPEXII (23 Sep - 11 Nov 2012)
plt.xlim(datetime(2012,11,10,0,1,0),datetime(2012,11,13,0,0,5)) # Period 3
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
ax.axes.get_yaxis().set_visible(False)

# Plot the axis labels, legend and title
#plt.title('V1 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 4 (LTcol)
ax = plt.subplot(gs[3])

# HIDE THE SPINES TO THE RIGHT OF SUBPLOT 1
ax.spines['right'].set_visible(False)

# Plot the BrO Pred vs Obs
ax.plot(Date, BrO_SP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO surf Predictions\n (R: '+str("%6.4f"%(rval_SSW))+')\n (R$^2$: '+str("%6.4f"%(R2_SSW))+')\n (p: '+str("%6.4f"%(rho_SSW))+')\n (MNB: '+str("%6.1f"%(MNB_SP_SW))+'%)\n (MNE: '+str("%6.1f"%(MNE_SP_SW))+'%)') # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, BrO_SP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
ax.plot(Date, BrO_SO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO surf Observations')   # LTcol BrO observations
ax.errorbar(Date, BrO_SO, yerr=StDev_LTO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                           # Error LTcol observations

# Format x-axis
#plt.xlim(datetime(2012,9,23),datetime(2012,11,12)) # SIPEXII (23 Sep - 11 Nov 2012)
plt.xlim(datetime(2012,9,22,0,1,0),datetime(2012,9,26,0,0,5)) # Period 1
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
#plt.title('V1 (2018-19)', fontsize=15, y=1.05)
ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
ax.set_xlabel('Date', fontsize=15, x=2.75)
#plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 5 (LTcol)
ax = plt.subplot(gs[4])

# HIDE THE SPINES TO THE LEFT OF SUBPLOT 2
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

# Plot the BrO Pred vs Obs
ax.plot(Date, BrO_SP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO surf Predictions\n (R: '+str("%6.4f"%(rval_SSW))+')\n (R$^2$: '+str("%6.4f"%(R2_SSW))+')\n (p: '+str("%6.4f"%(rho_SSW))+')\n (MNB: '+str("%6.1f"%(MNB_SP_SW))+'%)\n (MNE: '+str("%6.1f"%(MNE_SP_SW))+'%)') # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, BrO_SP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
ax.plot(Date, BrO_SO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO surf Observations')   # LTcol BrO observations
ax.errorbar(Date, BrO_SO, yerr=StDev_LTO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                           # Error LTcol observations

# Format x-axis
#plt.xlim(datetime(2012,9,23),datetime(2012,11,12)) # SIPEXII (23 Sep - 11 Nov 2012)
plt.xlim(datetime(2012,10,9,0,1,0),datetime(2012,10,20,0,0,5)) # Period 2
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
ax.axes.get_yaxis().set_visible(False)

# Plot the axis labels, legend and title
#plt.title('V1 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 6 (LTcol)
ax = plt.subplot(gs[5])

# HIDE THE SPINES TO THE LEFT OF SUBPLOT 2
ax.spines['left'].set_visible(False)
#ax.spines['right'].set_visible(False)

# Plot the BrO Pred vs Obs
ax.plot(Date, BrO_SP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO surf Predictions\n (R: '+str("%6.4f"%(rval_SSW))+')\n (R$^2$: '+str("%6.4f"%(R2_SSW))+')\n (p: '+str("%6.4f"%(rho_SSW))+')\n (MNB: '+str("%6.1f"%(MNB_SP_SW))+'%)\n (MNE: '+str("%6.1f"%(MNE_SP_SW))+'%)') # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, BrO_SP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
ax.plot(Date, BrO_SO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO surf Observations')   # LTcol BrO observations
ax.errorbar(Date, BrO_SO, yerr=StDev_LTO, alpha=0.5, capsize=None, c='b', lw=1.0, ls='none')                           # Error LTcol observations

# Format x-axis
#plt.xlim(datetime(2012,9,23),datetime(2012,11,12)) # SIPEXII (23 Sep - 11 Nov 2012)
plt.xlim(datetime(2012,11,10,0,1,0),datetime(2012,11,13,0,0,5)) # Period 3
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
ax.axes.get_yaxis().set_visible(False)

# Plot the axis labels, legend and title
#plt.title('V1 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)
