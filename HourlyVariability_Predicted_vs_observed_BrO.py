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
#PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN.csv', index_col=0)
#PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_SIPEXII.csv', index_col=0) # SIPEXII (Swanson Loadings)
PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_Test.csv', index_col=0) # GridBox (BestChoice)

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
BrO_SP_SW  = PC_BrO['BrO_SurfPred_Met']/1e13  # Surf BrO using Swanson coefficients
BrO_LTP_SW = PC_BrO['BrO_LtColPred_Met']/1e13 # LTcol BrO using Swanson coefficients

BrO_SP_Met  = PC_BrO['BrO_SurfPred_OLS']/1e13 # Surf BrO using My coefficients
BrO_LTP_Met = PC_BrO['BrO_LTPred_OLS']/1e13   # Ltcol BrO using My coefficients

#-------------
# Observations
#-------------
BrO_SO  = PC_BrO['BrO_SurfObs']/1e13
BrO_LTO = PC_BrO['BrO_LtColObs']/1e13

#------------------------------------------------------------------------------    
# CALCULATE THE MEAN

Mean_SO      = np.mean(BrO_SO)      # BrO surface observations
Mean_LTO     = np.mean(BrO_LTO)     # BrO LTcol observations
Mean_SP_SW   = np.mean(BrO_SP_SW)   # BrO surface predictions Swanso coefficientsn
Mean_LTP_SW  = np.mean(BrO_LTP_SW)  # BrO LTcol predictions Swanson coefficients
Mean_SP_Met  = np.mean(BrO_SP_Met)  # BrO surface predictions My coefficients
Mean_LTP_Met = np.mean(BrO_LTP_Met) # BrO LTcol predictions My coefficients

#------------------------------------------------------------------------------    
# CALCULATE THE MONTHLY MEAN

# Define the formula
def monthly(x, date):
    df = pd.DataFrame({'X':x}, index=date) 
    df = df.resample('M').mean()
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

HV_MM_SO      = BrO_SO      - SO_MAvgH      # Surface observations
HV_MM_LTO     = BrO_LTO     - LTO_MAvgH     # LTcol observations
HV_MM_SP_SW   = BrO_SP_SW   - SP_SW_MAvgH   # Surface predictions Swanson coefficient
HV_MM_LTP_SW  = BrO_LTP_SW  - LTP_SW_MAvgH  # LTcol predictions Swanson coefficient
HV_MM_SP_Met  = BrO_SP_Met  - SP_Met_MAvgH  # Surface predictions New coefficient
HV_MM_LTP_Met = BrO_LTP_Met - LTP_Met_MAvgH # LTcol predictions New coefficient

#------------------------------------------------------------------------------
# Calculate the hourly variability (from Mean)
# (Hourly Values) - (Mean) 

HV_SO      = BrO_SO      - Mean_SO      # Surface observations
HV_LTO     = BrO_LTO     - Mean_LTO     # LTcol observations
HV_SP_SW   = BrO_SP_SW   - Mean_SP_SW   # Surface predictions Swanson coefficient
HV_LTP_SW  = BrO_LTP_SW  - Mean_LTP_SW  # LTcol predictions Swanson coefficient
HV_SP_Met  = BrO_SP_Met  - Mean_SP_Met  # Surface predictions New coefficient
HV_LTP_Met = BrO_LTP_Met - Mean_LTP_Met # LTcol predictions New coefficient

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
ax.plot(Date, HV_SP_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, HV_SP_Met,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (my coefficients)
ax.plot(Date, HV_SO,    marker='o', c='b',      markersize = 2.0, linestyle='none', label='BrO Surface Observations')  # Surf BrO observations

# Plot a line along the 0 axis
plt.axhline(0, linewidth=0.5, color='k')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V1 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 2 (V2 2017-18)
ax = plt.subplot(gs[1,0:2])

# Plot BrO
ax.plot(Date, HV_SP_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, HV_SP_Met,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (my coefficients)
ax.plot(Date, HV_SO,    marker='o', c='b',      markersize = 2.0, linestyle='none', label='BrO Surface Observations')  # Surf BrO observations

# Plot a line along the 0 axis
plt.axhline(0, linewidth=0.5, color='k')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V2 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 3 (V3 2017-18)
ax = plt.subplot(gs[2,0:2])

# Plot BrO
ax.plot(Date, HV_SP_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, HV_SP_Met,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (my coefficients)
ax.plot(Date, HV_SO,    marker='o', c='b',      markersize = 2.0, linestyle='none', label='BrO Surface Observations')  # Surf BrO observations

# Plot a line along the 0 axis
plt.axhline(0, linewidth=0.5, color='k')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V3 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 4 (V1 2018-19)
ax = plt.subplot(gs[0,2:4])

# Plot BrO
ax.plot(Date, HV_SP_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, HV_SP_Met,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (my coefficients)
ax.plot(Date, HV_SO,    marker='o', c='b',      markersize = 2.0, linestyle='none', label='BrO Surface Observations')  # Surf BrO observations

# Plot a line along the 0 axis
plt.axhline(0, linewidth=0.5, color='k')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V1 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 5 (V2 2018-19)
ax = plt.subplot(gs[1,2:4])

# Plot BrO
ax.plot(Date, HV_SP_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, HV_SP_Met,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (my coefficients)
ax.plot(Date, HV_SO,    marker='o', c='b',      markersize = 2.0, linestyle='none', label='BrO Surface Observations')  # Surf BrO observations

# Plot a line along the 0 axis
plt.axhline(0, linewidth=0.5, color='k')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V2 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 6 (V3 2018-19)
ax = plt.subplot(gs[2,2:4])

# Plot BrO
ax.plot(Date, HV_SP_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, HV_SP_Met,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (my coefficients)
ax.plot(Date, HV_SO,    marker='o', c='b',      markersize = 2.0, linestyle='none', label='BrO Surface Observations')  # Surf BrO observations

# Plot a line along the 0 axis
plt.axhline(0, linewidth=0.5, color='k')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V3 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 1.35), loc=2, borderaxespad=0.)

##-------------------------------------
## Graph 7 (SIPEXII 2012)
#ax = plt.subplot(gs[3,1:3])
#
## Plot BrO
#ax.plot(Date, HV_SP_SW,    marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, HV_SP_Met,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO Surface Predictions')  # Surf BrO prediction (my coefficients)
#ax.plot(Date, HV_SO,    marker='o', c='b',      markersize = 2.0, linestyle='none', label='BrO Surface Observations')  # Surf BrO observations

## Plot a line along the 0 axis
#plt.axhline(0, linewidth=0.5, color='k')

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
ax.plot(Date, HV_LTP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, HV_LTP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
ax.plot(Date, HV_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations

# Plot a line along the 0 axis
plt.axhline(0, linewidth=0.5, color='k')

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
#ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V1 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 2 (V2 2017-18)
ax = plt.subplot(gs[1,0:2])

# Plot BrO
ax.plot(Date, HV_LTP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, HV_LTP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
ax.plot(Date, HV_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations

# Plot a line along the 0 axis
plt.axhline(0, linewidth=0.5, color='k')

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
#ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V2 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 3 (V3 2017-18)
ax = plt.subplot(gs[2,0:2])

# Plot BrO
ax.plot(Date, HV_LTP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, HV_LTP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
ax.plot(Date, HV_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations

# Plot a line along the 0 axis
plt.axhline(0, linewidth=0.5, color='k')

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
#ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V3 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 4 (V1 2018-19)
ax = plt.subplot(gs[0,2:4])

# Plot BrO
ax.plot(Date, HV_LTP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, HV_LTP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
ax.plot(Date, HV_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations

# Plot a line along the 0 axis
plt.axhline(0, linewidth=0.5, color='k')

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
#ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V1 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 5 (V2 2018-19)
ax = plt.subplot(gs[1,2:4])

# Plot BrO
ax.plot(Date, HV_LTP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, HV_LTP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
ax.plot(Date, HV_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations

# Plot a line along the 0 axis
plt.axhline(0, linewidth=0.5, color='k')

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
#ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V2 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 0.95), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 6 (V3 2018-19)
ax = plt.subplot(gs[2,2:4])

# Plot BrO
ax.plot(Date, HV_LTP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, HV_LTP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
ax.plot(Date, HV_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations

# Plot a line along the 0 axis
plt.axhline(0, linewidth=0.5, color='k')

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
#ax.set_ylim(0,)

# Plot the axis labels, legend and title
plt.title('V3 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(0.85, 1.35), loc=2, borderaxespad=0.)

##-------------------------------------
## Graph 7 (SIPEXII 2012)
#ax = plt.subplot(gs[3,1:3])
#
## Plot BrO
#ax.plot(Date, HV_LTP_SW,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (Swanson coefficients)
#ax.plot(Date, HV_LTP_Met,  marker='o', c='r',      markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')    # Surf BrO prediction (my coefficients)
#ax.plot(Date, HV_LTO,      marker='o', c='b', markersize = 2.0, linestyle='none', label='BrO LTcol Observations')   # LTcol BrO observations

## Plot a line along the 0 axis
#plt.axhline(0, linewidth=0.5, color='k')

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
#ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
#ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#ax.set_ylim(0,)

## Plot the axis labels, legend and title
##plt.title('Comparison of observations vs predictions', fontsize=25, y=1.05)
#ax.set_ylabel('BrO$_L$$_T$$_c$$_o$$_l$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)
#plt.legend(bbox_to_anchor=(1.10, 0.5), loc=2, borderaxespad=0.)
