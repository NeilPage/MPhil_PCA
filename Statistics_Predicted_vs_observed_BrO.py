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
#PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_Test.csv', index_col=0)
PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_Test2.csv', index_col=0) # GridBox (BestChoice)

#------------------------------------------------------------------------------
# SET THE DATE

PC_BrO.index = (pd.to_datetime(PC_BrO.index, dayfirst=True))

#------------------------------------------------------------------------------
# SET THE VARIABLES

Date    = np.array(PC_BrO.index)

#-------------
# Predictions
#-------------
# Swanson
# BrO_SP_SW  = PC_BrO['BrO_SurfPred_Met']/1e13  # Surf BrO using Swanson coefficients
# BrO_LTP_SW = PC_BrO['BrO_LtColPred_Met']/1e13 # LTcol BrO using Swanson coefficients

# PCR (OLS)
BrO_SP_SW  = PC_BrO['OLS_BrOSurf']/1e13  # Surf BrO using OLS coefficients
BrO_LTP_SW = PC_BrO['OLS_BrOLTcol']/1e13 # Ltcol BrO using OLS coefficients

#-------------
# Observations
#-------------
BrO_SO  = PC_BrO['Obs_BrOSurf']/1e13
BrO_LTO = PC_BrO['Obs_BrOLTcol']/1e13

#-------------
# Observational error
#-------------
#ErrorS  = np.mean(PC_BrO['BrO_SurfObs_Err']/1e13)
#ErrorLT = np.mean(PC_BrO['BrO_SurfObs_Err']/1e13)

#-------------
# StDev error
#-------------
StDev_SO = np.std(BrO_SO)
StDev_LTO = np.std(BrO_LTO)

#------------------------------------------------------------------------------
# SEPERATE INTO SEPERATE SEASONS & VOYAGES

#-------------
# 2017-18 Season
#-------------
start_date = '2017-11-01'
end_date   = '2018-02-27'
# Surf Pred
Filter     = (BrO_SP_SW.index >= start_date) & (BrO_SP_SW.index < end_date)
BrO_SP_17  = BrO_SP_SW[Filter]
# LTcol Pred
Filter     = (BrO_LTP_SW.index >= start_date) & (BrO_LTP_SW.index < end_date)
BrO_LTP_17 = BrO_LTP_SW[Filter]
# Surf Obs
Filter     = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_17  = BrO_SO[Filter]
# LTcol Obs
Filter     = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_17 = BrO_LTO[Filter]

#-------------
# V1 (2017-18)
#-------------
start_date    = '2017-11-14'
end_date      = '2017-11-23'
# Surf Pred
Filter        = (BrO_SP_SW.index >= start_date) & (BrO_SP_SW.index < end_date)
BrO_SP_V1_17  = BrO_SP_SW[Filter]
# LTcol Pred
Filter        = (BrO_LTP_SW.index >= start_date) & (BrO_LTP_SW.index < end_date)
BrO_LTP_V1_17 = BrO_LTP_SW[Filter]
# Surf Obs
Filter        = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_V1_17  = BrO_SO[Filter]
# LTcol Obs
Filter        = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_V1_17 = BrO_LTO[Filter]

#-------------
# V2 (2017-18)
#-------------
start_date    = '2017-12-21'
end_date      = '2018-01-06'
# Surf Pred
Filter        = (BrO_SP_SW.index >= start_date) & (BrO_SP_SW.index < end_date)
BrO_SP_V2_17  = BrO_SP_SW[Filter]
# LTcol Pred
Filter        = (BrO_LTP_SW.index >= start_date) & (BrO_LTP_SW.index < end_date)
BrO_LTP_V2_17 = BrO_LTP_SW[Filter]
# Surf Obs
Filter        = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_V2_17  = BrO_SO[Filter]
# LTcol Obs
Filter        = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_V2_17 = BrO_LTO[Filter]

#-------------
# V3 (2017-18)
#-------------
start_date    = '2018-01-27'
end_date      = '2018-02-22'
# Surf Pred
Filter        = (BrO_SP_SW.index >= start_date) & (BrO_SP_SW.index < end_date)
BrO_SP_V3_17  = BrO_SP_SW[Filter]
# LTcol Pred
Filter        = (BrO_LTP_SW.index >= start_date) & (BrO_LTP_SW.index < end_date)
BrO_LTP_V3_17 = BrO_LTP_SW[Filter]
# Surf Obs
Filter        = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_V3_17  = BrO_SO[Filter]
# LTcol Obs
Filter        = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_V3_17 = BrO_LTO[Filter]

#-------------
# 2018-19 Season
#-------------
start_date = '2018-11-01'
end_date   = '2019-02-27'
# Surf Pred
Filter     = (BrO_SP_SW.index >= start_date) & (BrO_SP_SW.index < end_date)
BrO_SP_18  = BrO_SP_SW[Filter]
# LTcol Pred
Filter     = (BrO_LTP_SW.index >= start_date) & (BrO_LTP_SW.index < end_date)
BrO_LTP_18 = BrO_LTP_SW[Filter]
# Surf Obs
Filter     = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_18  = BrO_SO[Filter]
# LTcol Obs
Filter     = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_18 = BrO_LTO[Filter]

#-------------
# V1 (2018-19)
#-------------
start_date    = '2018-11-07'
end_date      = '2018-11-16'
# Surf Pred
Filter        = (BrO_SP_SW.index >= start_date) & (BrO_SP_SW.index < end_date)
BrO_SP_V1_18  = BrO_SP_SW[Filter]
# LTcol Pred
Filter        = (BrO_LTP_SW.index >= start_date) & (BrO_LTP_SW.index < end_date)
BrO_LTP_V1_18 = BrO_LTP_SW[Filter]
# Surf Obs
Filter        = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_V1_18  = BrO_SO[Filter]
# LTcol Obs
Filter        = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_V1_18 = BrO_LTO[Filter]

#-------------
# V2 (2018-19)
#-------------
start_date    = '2018-12-15'
end_date      = '2018-12-31'
# Surf Pred
Filter        = (BrO_SP_SW.index >= start_date) & (BrO_SP_SW.index < end_date)
BrO_SP_V2_18  = BrO_SP_SW[Filter]
# LTcol Pred
Filter        = (BrO_LTP_SW.index >= start_date) & (BrO_LTP_SW.index < end_date)
BrO_LTP_V2_18 = BrO_LTP_SW[Filter]
# Surf Obs
Filter        = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_V2_18  = BrO_SO[Filter]
# LTcol Obs
Filter        = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_V2_18 = BrO_LTO[Filter]

#-------------
# V3 (2018-19)
#-------------
start_date    = '2019-01-26'
end_date      = '2019-02-20'
# Surf Pred
Filter        = (BrO_SP_SW.index >= start_date) & (BrO_SP_SW.index < end_date)
BrO_SP_V3_18  = BrO_SP_SW[Filter]
# LTcol Pred
Filter        = (BrO_LTP_SW.index >= start_date) & (BrO_LTP_SW.index < end_date)
BrO_LTP_V3_18 = BrO_LTP_SW[Filter]
# Surf Obs
Filter        = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_V3_18  = BrO_SO[Filter]
# LTcol Obs
Filter        = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_V3_18 = BrO_LTO[Filter]

#------------------------------------------------------------------------------
# COUNT THE NUMBER OF OBSERVATIONS OVERALL, EACH SEASON & VOYAGE

#-------------
# All
#-------------
N_All   = len(BrO_SO)
#-------------
# 2017-18 Season
#-------------
N_17    = len(BrO_SO_17)
N_V1_17 = len(BrO_SO_V1_17)
N_V2_17 = len(BrO_SO_V2_17)
N_V3_17 = len(BrO_SO_V3_17)
#-------------
# 2018-19 Season
#-------------
N_18    = len(BrO_SO_18)
N_V1_18 = len(BrO_SO_V1_18)
N_V2_18 = len(BrO_SO_V2_18)
N_V3_18 = len(BrO_SO_V3_18)

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
SP_SW_Mavg,   date_avg = monthly(BrO_SP_SW,   Date) # BrO surface predictions Swanson coefficientsn
LTP_SW_Mavg,  date_avg = monthly(BrO_LTP_SW,  Date) # BrO LTcol predictions Swanson coefficients

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

#------------------------------------------------------------------------------
# Calculate the hourly variability (from Monthly Mean)
# (Hourly Values) - (Monthly Mean) 

HV_SO      = BrO_SO      - SO_MAvgH      # Surface observations
HV_LTO     = BrO_LTO     - LTO_MAvgH     # LTcol observations
HV_SP_SW   = BrO_SP_SW   - SP_SW_MAvgH   # Surface predictions Swanson coefficient
HV_LTP_SW  = BrO_LTP_SW  - LTP_SW_MAvgH  # LTcol predictions Swanson coefficient

#------------------------------------------------------------------------------
# CALCULATE THE MEAN

#-------------
# All
#-------------
Mean_SP  = np.mean(BrO_SP_SW)
Mean_LTP = np.mean(BrO_LTP_SW)
Mean_SO  = np.mean(BrO_SO)
Mean_LTO = np.mean(BrO_LTO)
    
#-------------
# 2017-18 Season
#-------------
Mean_SP_17  = np.mean(BrO_SP_17)
Mean_LTP_17 = np.mean(BrO_LTP_17)
Mean_SO_17  = np.mean(BrO_SO_17)
Mean_LTO_17 = np.mean(BrO_LTO_17)

# V1 (2017-18)
Mean_SP_V1_17  = np.mean(BrO_SP_V1_17)
Mean_LTP_V1_17 = np.mean(BrO_LTP_V1_17)
Mean_SO_V1_17  = np.mean(BrO_SO_V1_17)
Mean_LTO_V1_17 = np.mean(BrO_LTO_V1_17)

# V2 (2017-18)
Mean_SP_V2_17  = np.mean(BrO_SP_V2_17)
Mean_LTP_V2_17 = np.mean(BrO_LTP_V2_17)
Mean_SO_V2_17  = np.mean(BrO_SO_V2_17)
Mean_LTO_V2_17 = np.mean(BrO_LTO_V2_17)

# V3 (2017-18)
Mean_SP_V3_17  = np.mean(BrO_SP_V3_17)
Mean_LTP_V3_17 = np.mean(BrO_LTP_V3_17)
Mean_SO_V3_17  = np.mean(BrO_SO_V3_17)
Mean_LTO_V3_17 = np.mean(BrO_LTO_V3_17)

#-------------
# 2018-19 Season
#-------------
Mean_SP_18  = np.mean(BrO_SP_18)
Mean_LTP_18 = np.mean(BrO_LTP_18)
Mean_SO_18  = np.mean(BrO_SO_18)
Mean_LTO_18 = np.mean(BrO_LTO_18)

# V1 (2018-19)
Mean_SP_V1_18  = np.mean(BrO_SP_V1_18)
Mean_LTP_V1_18 = np.mean(BrO_LTP_V1_18)
Mean_SO_V1_18  = np.mean(BrO_SO_V1_18)
Mean_LTO_V1_18 = np.mean(BrO_LTO_V1_18)

# V2 (2018-19)
Mean_SP_V2_18  = np.mean(BrO_SP_V2_18)
Mean_LTP_V2_18 = np.mean(BrO_LTP_V2_18)
Mean_SO_V2_18  = np.mean(BrO_SO_V2_18)
Mean_LTO_V2_18 = np.mean(BrO_LTO_V2_18)

# V3 (2018-19)
Mean_SP_V3_18  = np.mean(BrO_SP_V3_18)
Mean_LTP_V3_18 = np.mean(BrO_LTP_V3_18)
Mean_SO_V3_18  = np.mean(BrO_SO_V3_18)
Mean_LTO_V3_18 = np.mean(BrO_LTO_V3_18)

#------------------------------------------------------------------------------
# CALCULATE THE STANDARD DEVIATION

#-------------
# All
#-------------
Std_SP  = np.std(BrO_SP_SW)
Std_LTP = np.std(BrO_LTP_SW)
Std_SO  = np.std(BrO_SO)
Std_LTO = np.std(BrO_LTO)
    
#-------------
# 2017-18 Season
#-------------
Std_SP_17  = np.std(BrO_SP_17)
Std_LTP_17 = np.std(BrO_LTP_17)
Std_SO_17  = np.std(BrO_SO_17)
Std_LTO_17 = np.std(BrO_LTO_17)

# V1 (2017-18)
Std_SP_V1_17  = np.std(BrO_SP_V1_17)
Std_LTP_V1_17 = np.std(BrO_LTP_V1_17)
Std_SO_V1_17  = np.std(BrO_SO_V1_17)
Std_LTO_V1_17 = np.std(BrO_LTO_V1_17)

# V2 (2017-18)
Std_SP_V2_17  = np.std(BrO_SP_V2_17)
Std_LTP_V2_17 = np.std(BrO_LTP_V2_17)
Std_SO_V2_17  = np.std(BrO_SO_V2_17)
Std_LTO_V2_17 = np.std(BrO_LTO_V2_17)

# V3 (2017-18)
Std_SP_V3_17  = np.std(BrO_SP_V3_17)
Std_LTP_V3_17 = np.std(BrO_LTP_V3_17)
Std_SO_V3_17  = np.std(BrO_SO_V3_17)
Std_LTO_V3_17 = np.std(BrO_LTO_V3_17)

#-------------
# 2018-19 Season
#-------------
Std_SP_18  = np.std(BrO_SP_18)
Std_LTP_18 = np.std(BrO_LTP_18)
Std_SO_18  = np.std(BrO_SO_18)
Std_LTO_18 = np.std(BrO_LTO_18)

# V1 (2018-19)
Std_SP_V1_18  = np.std(BrO_SP_V1_18)
Std_LTP_V1_18 = np.std(BrO_LTP_V1_18)
Std_SO_V1_18  = np.std(BrO_SO_V1_18)
Std_LTO_V1_18 = np.std(BrO_LTO_V1_18)

# V2 (2018-19)
Std_SP_V2_18  = np.std(BrO_SP_V2_18)
Std_LTP_V2_18 = np.std(BrO_LTP_V2_18)
Std_SO_V2_18  = np.std(BrO_SO_V2_18)
Std_LTO_V2_18 = np.std(BrO_LTO_V2_18)

# V3 (2018-19)
Std_SP_V3_18  = np.std(BrO_SP_V3_18)
Std_LTP_V3_18 = np.std(BrO_LTP_V3_18)
Std_SO_V3_18  = np.std(BrO_SO_V3_18)
Std_LTO_V3_18 = np.std(BrO_LTO_V3_18)

#------------------------------------------------------------------------------
# CALCULATE THE MEDIAN

#-------------
# All
#-------------
Median_SP  = np.median(BrO_SP_SW)
Median_LTP = np.median(BrO_LTP_SW)
Median_SO  = np.median(BrO_SO)
Median_LTO = np.median(BrO_LTO)
    
#-------------
# 2017-18 Season
#-------------
Median_SP_17  = np.median(BrO_SP_17)
Median_LTP_17 = np.median(BrO_LTP_17)
Median_SO_17  = np.median(BrO_SO_17)
Median_LTO_17 = np.median(BrO_LTO_17)

# V1 (2017-18)
Median_SP_V1_17  = np.median(BrO_SP_V1_17)
Median_LTP_V1_17 = np.median(BrO_LTP_V1_17)
Median_SO_V1_17  = np.median(BrO_SO_V1_17)
Median_LTO_V1_17 = np.median(BrO_LTO_V1_17)

# V2 (2017-18)
Median_SP_V2_17  = np.median(BrO_SP_V2_17)
Median_LTP_V2_17 = np.median(BrO_LTP_V2_17)
Median_SO_V2_17  = np.median(BrO_SO_V2_17)
Median_LTO_V2_17 = np.median(BrO_LTO_V2_17)

# V3 (2017-18)
Median_SP_V3_17  = np.median(BrO_SP_V3_17)
Median_LTP_V3_17 = np.median(BrO_LTP_V3_17)
Median_SO_V3_17  = np.median(BrO_SO_V3_17)
Median_LTO_V3_17 = np.median(BrO_LTO_V3_17)

#-------------
# 2018-19 Season
#-------------
Median_SP_18  = np.median(BrO_SP_18)
Median_LTP_18 = np.median(BrO_LTP_18)
Median_SO_18  = np.median(BrO_SO_18)
Median_LTO_18 = np.median(BrO_LTO_18)

# V1 (2018-19)
Median_SP_V1_18  = np.median(BrO_SP_V1_18)
Median_LTP_V1_18 = np.median(BrO_LTP_V1_18)
Median_SO_V1_18  = np.median(BrO_SO_V1_18)
Median_LTO_V1_18 = np.median(BrO_LTO_V1_18)

# V2 (2018-19)
Median_SP_V2_18  = np.median(BrO_SP_V2_18)
Median_LTP_V2_18 = np.median(BrO_LTP_V2_18)
Median_SO_V2_18  = np.median(BrO_SO_V2_18)
Median_LTO_V2_18 = np.median(BrO_LTO_V2_18)

# V3 (2018-19)
Median_SP_V3_18  = np.median(BrO_SP_V3_18)
Median_LTP_V3_18 = np.median(BrO_LTP_V3_18)
Median_SO_V3_18  = np.median(BrO_SO_V3_18)
Median_LTO_V3_18 = np.median(BrO_LTO_V3_18)

#------------------------------------------------------------------------------
# CALCULATE THE MEDIAN ABSOLUTE DEVIATION (MAD)

#-------------
# All
#-------------
Mad_SP  = stats.median_absolute_deviation(BrO_SP_SW)
Mad_LTP = stats.median_absolute_deviation(BrO_LTP_SW)
Mad_SO  = stats.median_absolute_deviation(BrO_SO)
Mad_LTO = stats.median_absolute_deviation(BrO_LTO)
    
#-------------
# 2017-18 Season
#-------------
Mad_SP_17  = stats.median_absolute_deviation(BrO_SP_17)
Mad_LTP_17 = stats.median_absolute_deviation(BrO_LTP_17)
Mad_SO_17  = stats.median_absolute_deviation(BrO_SO_17)
Mad_LTO_17 = stats.median_absolute_deviation(BrO_LTO_17)

# V1 (2017-18)
Mad_SP_V1_17  = stats.median_absolute_deviation(BrO_SP_V1_17)
Mad_LTP_V1_17 = stats.median_absolute_deviation(BrO_LTP_V1_17)
Mad_SO_V1_17  = stats.median_absolute_deviation(BrO_SO_V1_17)
Mad_LTO_V1_17 = stats.median_absolute_deviation(BrO_LTO_V1_17)

# V2 (2017-18)
Mad_SP_V2_17  = stats.median_absolute_deviation(BrO_SP_V2_17)
Mad_LTP_V2_17 = stats.median_absolute_deviation(BrO_LTP_V2_17)
Mad_SO_V2_17  = stats.median_absolute_deviation(BrO_SO_V2_17)
Mad_LTO_V2_17 = stats.median_absolute_deviation(BrO_LTO_V2_17)

# V3 (2017-18)
Mad_SP_V3_17  = stats.median_absolute_deviation(BrO_SP_V3_17)
Mad_LTP_V3_17 = stats.median_absolute_deviation(BrO_LTP_V3_17)
Mad_SO_V3_17  = stats.median_absolute_deviation(BrO_SO_V3_17)
Mad_LTO_V3_17 = stats.median_absolute_deviation(BrO_LTO_V3_17)

#-------------
# 2018-19 Season
#-------------
Mad_SP_18  = stats.median_absolute_deviation(BrO_SP_18)
Mad_LTP_18 = stats.median_absolute_deviation(BrO_LTP_18)
Mad_SO_18  = stats.median_absolute_deviation(BrO_SO_18)
Mad_LTO_18 = stats.median_absolute_deviation(BrO_LTO_18)

# V1 (2018-19)
Mad_SP_V1_18  = stats.median_absolute_deviation(BrO_SP_V1_18)
Mad_LTP_V1_18 = stats.median_absolute_deviation(BrO_LTP_V1_18)
Mad_SO_V1_18  = stats.median_absolute_deviation(BrO_SO_V1_18)
Mad_LTO_V1_18 = stats.median_absolute_deviation(BrO_LTO_V1_18)

# V2 (2018-19)
Mad_SP_V2_18  = stats.median_absolute_deviation(BrO_SP_V2_18)
Mad_LTP_V2_18 = stats.median_absolute_deviation(BrO_LTP_V2_18)
Mad_SO_V2_18  = stats.median_absolute_deviation(BrO_SO_V2_18)
Mad_LTO_V2_18 = stats.median_absolute_deviation(BrO_LTO_V2_18)

# V3 (2018-19)
Mad_SP_V3_18  = stats.median_absolute_deviation(BrO_SP_V3_18)
Mad_LTP_V3_18 = stats.median_absolute_deviation(BrO_LTP_V3_18)
Mad_SO_V3_18  = stats.median_absolute_deviation(BrO_SO_V3_18)
Mad_LTO_V3_18 = stats.median_absolute_deviation(BrO_LTO_V3_18)

#------------------------------------------------------------------------------
# CALCULATE THE MEAN NORMALISED BIAS (between predictions and observation)
# MNB = Mean((predictions-observations)/observations)*100

#-------------
# All
#-------------
MNB_SP_SW     = (np.mean((BrO_SP_SW  - BrO_SO)  / BrO_SO))  * 100 # Surface predictions Swanson coefficient
MNB_LTP_SW    = (np.mean((BrO_LTP_SW - BrO_LTO) / BrO_LTO)) * 100 # LTcol predictions Swanson coefficient

#-------------
# 2017-18 Season
#-------------
MNB_SP_17     = (np.mean((BrO_SP_17  - BrO_SO_17)  / BrO_SO_17))  * 100
MNB_LTP_17    = (np.mean((BrO_LTP_17 - BrO_LTO_17) / BrO_LTO_17)) * 100

# V1 (2017-18)
MNB_SP_V1_17  = (np.mean((BrO_SP_V1_17  - BrO_SO_V1_17)  / BrO_SO_V1_17))  * 100
MNB_LTP_V1_17 = (np.mean((BrO_LTP_V1_17 - BrO_LTO_V1_17) / BrO_LTO_V1_17)) * 100

# V2 (2017-18)
MNB_SP_V2_17  = (np.mean((BrO_SP_V2_17  - BrO_SO_V2_17)  / BrO_SO_V2_17))  * 100
MNB_LTP_V2_17 = (np.mean((BrO_LTP_V2_17 - BrO_LTO_V2_17) / BrO_LTO_V2_17)) * 100

# V3 (2017-18)
MNB_SP_V3_17  = (np.mean((BrO_SP_V3_17  - BrO_SO_V3_17)  / BrO_SO_V3_17))  * 100
MNB_LTP_V3_17 = (np.mean((BrO_LTP_V3_17 - BrO_LTO_V3_17) / BrO_LTO_V3_17)) * 100

#-------------
# 2018-19 Season
#-------------
MNB_SP_18     = (np.mean((BrO_SP_18   - BrO_SO_18)  / BrO_SO_18))  * 100
MNB_LTP_18    = (np.mean((BrO_LTP_18  - BrO_LTO_18) / BrO_LTO_18)) * 100

# V1 (2018-19)
MNB_SP_V1_18  = (np.mean((BrO_SP_V1_18  - BrO_SO_V1_18)  / BrO_SO_V1_18))  * 100
MNB_LTP_V1_18 = (np.mean((BrO_LTP_V1_18 - BrO_LTO_V1_18) / BrO_LTO_V1_18)) * 100

# V2 (2018-19)
MNB_SP_V2_18  = (np.mean((BrO_SP_V2_18  - BrO_SO_V2_18)  / BrO_SO_V2_18))  * 100
MNB_LTP_V2_18 = (np.mean((BrO_LTP_V2_18 - BrO_LTO_V2_18) / BrO_LTO_V2_18)) * 100

# V3 (2018-19)
MNB_SP_V3_18  = (np.mean((BrO_SP_V3_18  - BrO_SO_V3_18)  / BrO_SO_V3_18))  * 100
MNB_LTP_V3_18 = (np.mean((BrO_LTP_V3_18 - BrO_LTO_V3_18) / BrO_LTO_V3_18)) * 100

#------------------------------------------------------------------------------
# CALCULATE THE MEAN NORMALISED ERROR (between predictions and observation)
# MNE = Mean((predictions-observations)/observations)*100

#-------------
# All
#-------------
MNE_SP_SW   = (np.mean(abs(BrO_SP_SW   - BrO_SO)  / BrO_SO))  * 100 # Surface predictions Swanson coefficient
MNE_LTP_SW  = (np.mean(abs(BrO_LTP_SW  - BrO_LTO) / BrO_LTO)) * 100 # LTcol predictions Swanson coefficient

#-------------
# 2017-18 Season
#-------------
MNE_SP_17     = (np.mean(abs(BrO_SP_17  - BrO_SO_17)  / BrO_SO_17))  * 100
MNE_LTP_17    = (np.mean(abs(BrO_LTP_17 - BrO_LTO_17) / BrO_LTO_17)) * 100

# V1 (2017-18)
MNE_SP_V1_17  = (np.mean(abs(BrO_SP_V1_17  - BrO_SO_V1_17)  / BrO_SO_V1_17))  * 100
MNE_LTP_V1_17 = (np.mean(abs(BrO_LTP_V1_17 - BrO_LTO_V1_17) / BrO_LTO_V1_17)) * 100

# V2 (2017-18)
MNE_SP_V2_17  = (np.mean(abs(BrO_SP_V2_17  - BrO_SO_V2_17)  / BrO_SO_V2_17))  * 100
MNE_LTP_V2_17 = (np.mean(abs(BrO_LTP_V2_17 - BrO_LTO_V2_17) / BrO_LTO_V2_17)) * 100

# V3 (2017-18)
MNE_SP_V3_17  = (np.mean(abs(BrO_SP_V3_17  - BrO_SO_V3_17)  / BrO_SO_V3_17))  * 100
MNE_LTP_V3_17 = (np.mean(abs(BrO_LTP_V3_17 - BrO_LTO_V3_17) / BrO_LTO_V3_17)) * 100

#-------------
# 2018-19 Season
#-------------
MNE_SP_18     = (np.mean(abs(BrO_SP_18   - BrO_SO_18)  / BrO_SO_18))  * 100
MNE_LTP_18    = (np.mean(abs(BrO_LTP_18  - BrO_LTO_18) / BrO_LTO_18)) * 100

# V1 (2018-19)
MNE_SP_V1_18  = (np.mean(abs(BrO_SP_V1_18  - BrO_SO_V1_18)  / BrO_SO_V1_18))  * 100
MNE_LTP_V1_18 = (np.mean(abs(BrO_LTP_V1_18 - BrO_LTO_V1_18) / BrO_LTO_V1_18)) * 100

# V2 (2018-19)
MNE_SP_V2_18  = (np.mean(abs(BrO_SP_V2_18  - BrO_SO_V2_18)  / BrO_SO_V2_18))  * 100
MNE_LTP_V2_18 = (np.mean(abs(BrO_LTP_V2_18 - BrO_LTO_V2_18) / BrO_LTO_V2_18)) * 100

# V3 (2018-19)
MNE_SP_V3_18  = (np.mean(abs(BrO_SP_V3_18  - BrO_SO_V3_18)  / BrO_SO_V3_18))  * 100
MNE_LTP_V3_18 = (np.mean(abs(BrO_LTP_V3_18 - BrO_LTO_V3_18) / BrO_LTO_V3_18)) * 100

#------------------------------------------------------------------------------
# CORRECT THE PREDICTIONS FOR THE MEAN NORMALISED BIAS 

SP_MMNB_SW   = BrO_SP_SW   * (100/MNB_SP_SW)   # Surface predictions Swanson coefficient
LTP_MMNB_SW  = BrO_LTP_SW  * (100/MNB_LTP_SW)  # LTcol predictions Swanson coefficient

#------------------------------------------------------------------------------
# CALCULATE THE LINEAR CORRELATION COEFFICIENTS (R)

#-------------
# All
#-------------
slope_SSW,  intercept_SSW,  rval_SSW,  pval_SSW,  stderr_SSW  = stats.linregress(BrO_SP_SW,  BrO_SO)
slope_LTSW, intercept_LTSW, rval_LTSW, pval_LTSW, stderr_LTSW = stats.linregress(BrO_LTP_SW, BrO_LTO)

#-------------
# 2017-18 Season
#-------------
slope_SSW_17,  intercept_SSW_17,  rval_SSW_17,  pval_SSW_17,  stderr_SSW_17  = stats.linregress(BrO_SP_17,  BrO_SO_17)
slope_LTSW_17, intercept_LTSW_17, rval_LTSW_17, pval_LTSW_17, stderr_LTSW_17 = stats.linregress(BrO_LTP_17, BrO_LTO_17)

# V1 (2017-18)
slope_SSW_V1_17,  intercept_SSW_V1_17,  rval_SSW_V1_17,  pval_SSW_V1_17,  stderr_SSW_V1_17  = stats.linregress(BrO_SP_V1_17,  BrO_SO_V1_17)
slope_LTSW_V1_17, intercept_LTSW_V1_17, rval_LTSW_V1_17, pval_LTSW_V1_17, stderr_LTSW_V1_17 = stats.linregress(BrO_LTP_V1_17, BrO_LTO_V1_17)

# V2 (2017-18)
slope_SSW_V2_17,  intercept_SSW_V2_17,  rval_SSW_V2_17,  pval_SSW_V2_17,  stderr_SSW_V2_17  = stats.linregress(BrO_SP_V2_17,  BrO_SO_V2_17)
slope_LTSW_V2_17, intercept_LTSW_V2_17, rval_LTSW_V2_17, pval_LTSW_V2_17, stderr_LTSW_V2_17 = stats.linregress(BrO_LTP_V2_17, BrO_LTO_V2_17)

# V3 (2017-18)
slope_SSW_V3_17,  intercept_SSW_V3_17,  rval_SSW_V3_17,  pval_SSW_V3_17,  stderr_SSW_V3_17  = stats.linregress(BrO_SP_V3_17,  BrO_SO_V3_17)
slope_LTSW_V3_17, intercept_LTSW_V3_17, rval_LTSW_V3_17, pval_LTSW_V3_17, stderr_LTSW_V3_17 = stats.linregress(BrO_LTP_V3_17, BrO_LTO_V3_17)

#-------------
# 2018-19 Season
#-------------
slope_SSW_18,  intercept_SSW_18,  rval_SSW_18,  pval_SSW_18,  stderr_SSW_18  = stats.linregress(BrO_SP_18,  BrO_SO_18)
slope_LTSW_18, intercept_LTSW_18, rval_LTSW_18, pval_LTSW_18, stderr_LTSW_18 = stats.linregress(BrO_LTP_18, BrO_LTO_18)

# V1 (2018-19)
slope_SSW_V1_18,  intercept_SSW_V1_18,  rval_SSW_V1_18,  pval_SSW_V1_18,  stderr_SSW_V1_18  = stats.linregress(BrO_SP_V1_18,  BrO_SO_V1_18)
slope_LTSW_V1_18, intercept_LTSW_V1_18, rval_LTSW_V1_18, pval_LTSW_V1_18, stderr_LTSW_V1_18 = stats.linregress(BrO_LTP_V1_18, BrO_LTO_V1_18)

# V2 (2018-19)
slope_SSW_V2_18,  intercept_SSW_V2_18,  rval_SSW_V2_18,  pval_SSW_V2_18,  stderr_SSW_V2_18  = stats.linregress(BrO_SP_V2_18,  BrO_SO_V2_18)
slope_LTSW_V2_18, intercept_LTSW_V2_18, rval_LTSW_V2_18, pval_LTSW_V2_18, stderr_LTSW_V2_18 = stats.linregress(BrO_LTP_V2_18, BrO_LTO_V2_18)

# V3 (2018-19)
slope_SSW_V3_18,  intercept_SSW_V3_18,  rval_SSW_V3_18,  pval_SSW_V3_18,  stderr_SSW_V3_18  = stats.linregress(BrO_SP_V3_18,  BrO_SO_V3_18)
slope_LTSW_V3_18, intercept_LTSW_V3_18, rval_LTSW_V3_18, pval_LTSW_V3_18, stderr_LTSW_V3_18 = stats.linregress(BrO_LTP_V3_18, BrO_LTO_V3_18)

#------------------------------------------------------------------------------
# CALCULATE THE LINEAR DETERMINATION COEFFICIENTS (R2)

#-------------
# All
#-------------
R2_SSW  = np.square(rval_SSW)   # Surface predictions Swanson coefficient
R2_LTSW = np.square(rval_LTSW)  # Surface predictions Swanson coefficient

#-------------
# 2017-18 Season
#-------------
R2_SSW_17  = np.square(rval_SSW_17)
R2_LTSW_17 = np.square(rval_LTSW_17)

# V1 (2017-18)
R2_SSW_V1_17  = np.square(rval_SSW_V1_17)
R2_LTSW_V1_17 = np.square(rval_LTSW_V1_17)

# V2 (2017-18)
R2_SSW_V2_17  = np.square(rval_SSW_V2_17)
R2_LTSW_V2_17 = np.square(rval_LTSW_V2_17)

# V3 (2017-18)
R2_SSW_V3_17  = np.square(rval_SSW_V3_17)
R2_LTSW_V3_17 = np.square(rval_LTSW_V3_17)

#-------------
# 2018-19 Season
#-------------
R2_SSW_18  = np.square(rval_SSW_18)
R2_LTSW_18 = np.square(rval_LTSW_18)

# V1 (2018-19)
R2_SSW_V1_18  = np.square(rval_SSW_V1_18)
R2_LTSW_V1_18 = np.square(rval_LTSW_V1_18)

# V2 (2018-19)
R2_SSW_V2_18  = np.square(rval_SSW_V2_18)
R2_LTSW_V2_18 = np.square(rval_LTSW_V2_18)

# V3 (2018-19)
R2_SSW_V3_18  = np.square(rval_SSW_V3_18)
R2_LTSW_V3_18 = np.square(rval_LTSW_V3_18)

#------------------------------------------------------------------------------
# CALCULATE THE NONPARAMETRIC SPEARMAN RANK CORRELATION COEFFICIENTS (R)

#-------------
# All
#-------------
rho_SSW,  pval_SSW  = stats.spearmanr(BrO_SP_SW,  BrO_SO)
rho_LTSW, pval_LTSW = stats.spearmanr(BrO_LTP_SW, BrO_LTO)

#-------------
# 2017-18 Season
#-------------
rho_SSW_17,  pval_SSW_17  = stats.spearmanr(BrO_SP_17,  BrO_SO_17)
rho_LTSW_17, pval_LTSW_17 = stats.spearmanr(BrO_LTP_17, BrO_LTO_17)

# V1 (2017-18)
rho_SSW_V1_17,  pval_SSW_V1_17  = stats.spearmanr(BrO_SP_V1_17,  BrO_SO_V1_17)
rho_LTSW_V1_17, pval_LTSW_V1_17 = stats.spearmanr(BrO_LTP_V1_17, BrO_LTO_V1_17)

# V2 (2017-18)
rho_SSW_V2_17,  pval_SSW_V2_17  = stats.spearmanr(BrO_SP_V2_17,  BrO_SO_V2_17)
rho_LTSW_V2_17, pval_LTSW_V2_17 = stats.spearmanr(BrO_LTP_V2_17, BrO_LTO_V2_17)

# V3 (2017-18)
rho_SSW_V3_17,  pval_SSW_V3_17  = stats.spearmanr(BrO_SP_V3_17,  BrO_SO_V3_17)
rho_LTSW_V3_17, pval_LTSW_V3_17 = stats.spearmanr(BrO_LTP_V3_17, BrO_LTO_V3_17)

#-------------
# 2018-19 Season
#-------------
rho_SSW_18,  pval_SSW_18  = stats.spearmanr(BrO_SP_18,  BrO_SO_18)
rho_LTSW_18, pval_LTSW_18 = stats.spearmanr(BrO_LTP_18, BrO_LTO_18)

# V1 (2018-19)
rho_SSW_V1_18,  pval_SSW_V1_18  = stats.spearmanr(BrO_SP_V1_18,  BrO_SO_V1_18)
rho_LTSW_V1_18, pval_LTSW_V1_18 = stats.spearmanr(BrO_LTP_V1_18, BrO_LTO_V1_18)

# V2 (2018-19)
rho_SSW_V2_18,  pval_SSW_V2_18  = stats.spearmanr(BrO_SP_V2_18,  BrO_SO_V2_18)
rho_LTSW_V2_18, pval_LTSW_V2_18 = stats.spearmanr(BrO_LTP_V2_18, BrO_LTO_V2_18)

# V3 (2018-19)
rho_SSW_V3_18,  pval_SSW_V3_18  = stats.spearmanr(BrO_SP_V3_18,  BrO_SO_V3_18)
rho_LTSW_V3_18, pval_LTSW_V3_18 = stats.spearmanr(BrO_LTP_V3_18, BrO_LTO_V3_18)

#------------------------------------------------------------------------------
#BUILD DATAFRAME FOR THE STATISTICAL RESULTS

# Build a pandas dataframe
dfStatistics = {'No':[N_All,N_All,N_17,N_17,N_V1_17,N_V1_17,N_V2_17,N_V2_17,N_V3_17,N_V3_17,N_18,N_18,N_V1_18,N_V1_18,N_V2_18,N_V2_18,N_V3_18,N_V3_18],
                'Mean Obs':[Mean_LTO,Mean_SO,Mean_LTO_17,Mean_SO_17,Mean_LTO_V1_17,Mean_SO_V1_17,Mean_LTO_V2_17,Mean_SO_V2_17,Mean_LTO_V3_17,Mean_SO_V3_17,Mean_LTO_18,Mean_SO_18,Mean_LTO_V1_18,Mean_SO_V1_18,Mean_LTO_V2_18,Mean_SO_V2_18,Mean_LTO_V3_18,Mean_SO_V3_18],
                'StDev Obs':[Std_LTO,Std_SO,Std_LTO_17,Std_SO_17,Std_LTO_V1_17,Std_SO_V1_17,Std_LTO_V2_17,Std_SO_V2_17,Std_LTO_V3_17,Std_SO_V3_17,Std_LTO_18,Std_SO_18,Std_LTO_V1_18,Std_SO_V1_18,Std_LTO_V2_18,Std_SO_V2_18,Std_LTO_V3_18,Std_SO_V3_18],
                'Median Obs':[Median_LTO,Median_SO,Median_LTO_17,Median_SO_17,Median_LTO_V1_17,Median_SO_V1_17,Median_LTO_V2_17,Median_SO_V2_17,Median_LTO_V3_17,Median_SO_V3_17,Median_LTO_18,Median_SO_18,Median_LTO_V1_18,Median_SO_V1_18,Median_LTO_V2_18,Median_SO_V2_18,Median_LTO_V3_18,Median_SO_V3_18],
                'MAD Obs':[Mad_LTO,Mad_SO,Mad_LTO_17,Mad_SO_17,Mad_LTO_V1_17,Mad_SO_V1_17,Mad_LTO_V2_17,Mad_SO_V2_17,Mad_LTO_V3_17,Mad_SO_V3_17,Mad_LTO_18,Mad_SO_18,Mad_LTO_V1_18,Mad_SO_V1_18,Mad_LTO_V2_18,Mad_SO_V2_18,Mad_LTO_V3_18,Mad_SO_V3_18],
                'Mean Pred':[Mean_LTP,Mean_SP,Mean_LTP_17,Mean_SP_17,Mean_LTP_V1_17,Mean_SP_V1_17,Mean_LTP_V2_17,Mean_SP_V2_17,Mean_LTP_V3_17,Mean_SP_V3_17,Mean_LTP_18,Mean_SP_18,Mean_LTP_V1_18,Mean_SP_V1_18,Mean_LTP_V2_18,Mean_SP_V2_18,Mean_LTP_V3_18,Mean_SP_V3_18],
                'StDev Pred':[Std_LTP,Std_SP,Std_LTP_17,Std_SP_17,Std_LTP_V1_17,Std_SP_V1_17,Std_LTP_V2_17,Std_SP_V2_17,Std_LTP_V3_17,Std_SP_V3_17,Std_LTP_18,Std_SP_18,Std_LTP_V1_18,Std_SP_V1_18,Std_LTP_V2_18,Std_SP_V2_18,Std_LTP_V3_18,Std_SP_V3_18],
                'Median Pred':[Median_LTP,Median_SP,Median_LTP_17,Median_SP_17,Median_LTP_V1_17,Median_SP_V1_17,Median_LTP_V2_17,Median_SP_V2_17,Median_LTP_V3_17,Median_SP_V3_17,Median_LTP_18,Median_SP_18,Median_LTP_V1_18,Median_SP_V1_18,Median_LTP_V2_18,Median_SP_V2_18,Median_LTP_V3_18,Median_SP_V3_18],
                'MAD Pred':[Mad_LTP,Mad_SP,Mad_LTP_17,Mad_SP_17,Mad_LTP_V1_17,Mad_SP_V1_17,Mad_LTP_V2_17,Mad_SP_V2_17,Mad_LTP_V3_17,Mad_SP_V3_17,Mad_LTP_18,Mad_SP_18,Mad_LTP_V1_18,Mad_SP_V1_18,Mad_LTP_V2_18,Mad_SP_V2_18,Mad_LTP_V3_18,Mad_SP_V3_18],
                'Linear (R)':   [rval_LTSW,rval_SSW,rval_LTSW_17,rval_SSW_17,rval_LTSW_V1_17,rval_SSW_V1_17,rval_LTSW_V2_17,rval_SSW_V2_17,rval_LTSW_V3_17,rval_SSW_V3_17,rval_LTSW_18,rval_SSW_18,rval_LTSW_V1_18,rval_SSW_V1_18,rval_LTSW_V2_18,rval_SSW_V2_18,rval_LTSW_V3_18,rval_SSW_V3_18],
                'Linear (R2)':  [R2_LTSW,R2_SSW,R2_LTSW_17,R2_SSW_17,R2_LTSW_V1_17,R2_SSW_V1_17,R2_LTSW_V2_17,R2_SSW_V2_17,R2_LTSW_V3_17,R2_SSW_V3_17,R2_LTSW_18,R2_SSW_18,R2_LTSW_V1_18,R2_SSW_V1_18,R2_LTSW_V2_18,R2_SSW_V2_18,R2_LTSW_V3_18,R2_SSW_V3_18],
                'Spearman (p)': [rho_LTSW,rho_SSW,rho_LTSW_17,rho_SSW_17,rho_LTSW_V1_17,rho_SSW_V1_17,rho_LTSW_V2_17,rho_SSW_V2_17,rho_LTSW_V3_17,rho_SSW_V3_17,rho_LTSW_18,rho_SSW_18,rho_LTSW_V1_18,rho_SSW_V1_18,rho_LTSW_V2_18,rho_SSW_V2_18,rho_LTSW_V3_18,rho_SSW_V3_18],
                'MNB':[MNB_LTP_SW,MNB_SP_SW,MNB_LTP_17,MNB_SP_17,MNB_LTP_V1_17,MNB_SP_V1_17,MNB_LTP_V2_17,MNB_SP_V2_17,MNB_LTP_V3_17,MNB_SP_V3_17,MNB_LTP_18,MNB_SP_18,MNB_LTP_V1_18,MNB_SP_V1_18,MNB_LTP_V2_18,MNB_SP_V2_18,MNB_LTP_V3_18,MNB_SP_V3_18],
                'MNE':[MNE_LTP_SW,MNE_SP_SW,MNE_LTP_17,MNE_SP_17,MNE_LTP_V1_17,MNE_SP_V1_17,MNE_LTP_V2_17,MNE_SP_V2_17,MNE_LTP_V3_17,MNE_SP_V3_17,MNE_LTP_18,MNE_SP_18,MNE_LTP_V1_18,MNE_SP_V1_18,MNE_LTP_V2_18,MNE_SP_V2_18,MNE_LTP_V3_18,MNE_SP_V3_18]}
dfStatistics = pd.DataFrame(dfStatistics, columns = ['No','Mean Obs','StDev Obs','Median Obs','MAD Obs','Mean Pred','StDev Pred','Median Pred','MAD Pred','Linear (R)','Linear (R2)','Spearman (p)','MNB','MNE'],index = ['LTcol All','Surface All','LTcol 2017','Surface 2017','LTcol V1_17','Surface V1_17','LTcol V2_17','Surface V2_17','LTcol V3_17','Surface V3_17','LTcol 2018','Surface 2018','LTcol V1_18','Surface V1_18','LTcol V2_18','Surface V2_18','LTcol V3_18','Surface V3_18'])
dfStatistics.to_csv('/Users/ncp532/Documents/Data/MERRA2/Statistics.csv')
