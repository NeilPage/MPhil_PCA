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
#PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_PCA_PCR.csv', index_col=0) # PCA & PCR
#PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_GridBox_BestChoice.csv', index_col=0) # GridBox (BestChoice)
#PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_GridBox_Actual.csv', index_col=0) # GridBox (Actual)
PC_BrO = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_Test.csv', index_col=0) # GridBox (BestChoice)

#------------------------------------------------------------------------------
# SET THE DATE

PC_BrO.index = (pd.to_datetime(PC_BrO.index, dayfirst=True))

#------------------------------------------------------------------------------
# SET THE VARIABLES

Date    = np.array(PC_BrO.index)

#-------------
# Principle Components
#-------------
PC1 = PC_BrO['PC1'] # PC1
PC2 = PC_BrO['PC2'] # PC2
PC3 = PC_BrO['PC3'] # PC3

#-------------
# BrO Predictions
#-------------
BrO_SP  = PC_BrO['BrO_SurfPred_Met']/1e13  # Surf BrO using Swanson coefficients
BrO_LTP = PC_BrO['BrO_LtColPred_Met']/1e13 # LTcol BrO using Swanson coefficients

#BrO_SP  = PC_BrO['BrO_SurfPred_OLS']/1e13 # Surf BrO using OLS coefficients
#BrO_LTP = PC_BrO['BrO_LTPred_OLS']/1e13   # Ltcol BrO using OLS coefficients

#-------------
# BrO Observations
#-------------
BrO_SO  = PC_BrO['BrO_SurfObs']/1e13
BrO_LTO = PC_BrO['BrO_LtColObs']/1e13

#------------------------------------------------------------------------------
# SEPERATE INTO SEPERATE SEASONS & VOYAGES

#-------------
# 2017-18 Season
#-------------
start_date   = '2017-11-01'
end_date     = '2018-02-27'
# Surf Pred
Filter       = (BrO_SP.index >= start_date) & (BrO_SP.index < end_date)
BrO_SP_2017  = BrO_SP[Filter]
# LTcol Pred
Filter       = (BrO_LTP.index >= start_date) & (BrO_LTP.index < end_date)
BrO_LTP_2017 = BrO_LTP[Filter]

# Surf Obs
Filter       = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_2017  = BrO_SO[Filter]
# LTcol Obs
Filter       = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_2017 = BrO_LTO[Filter]

# PC1
Filter       = (PC1.index >= start_date) & (PC1.index < end_date)
PC1_2017     = PC1[Filter]
# PC2
Filter       = (PC2.index >= start_date) & (PC2.index < end_date)
PC2_2017     = PC2[Filter]
# PC3
Filter       = (PC3.index >= start_date) & (PC3.index < end_date)
PC3_2017     = PC3[Filter]


#-------------
# V1 (2017-18)
#-------------
start_date    = '2017-11-14'
end_date      = '2017-11-23'
# Surf Pred
Filter        = (BrO_SP.index >= start_date) & (BrO_SP.index < end_date)
BrO_SP_V1_17  = BrO_SP[Filter]
# LTcol Pred
Filter        = (BrO_LTP.index >= start_date) & (BrO_LTP.index < end_date)
BrO_LTP_V1_17 = BrO_LTP[Filter]

# Surf Obs
Filter        = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_V1_17  = BrO_SO[Filter]
# LTcol Obs
Filter        = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_V1_17 = BrO_LTO[Filter]

# PC1
Filter        = (PC1.index >= start_date) & (PC1.index < end_date)
PC1_V1_17     = PC1[Filter]
# PC2
Filter        = (PC2.index >= start_date) & (PC2.index < end_date)
PC2_V1_17     = PC2[Filter]
# PC3
Filter        = (PC3.index >= start_date) & (PC3.index < end_date)
PC3_V1_17     = PC3[Filter]

#-------------
# V2 (2017-18)
#-------------
start_date    = '2017-12-21'
end_date      = '2018-01-06'
# Surf Pred
Filter        = (BrO_SP.index >= start_date) & (BrO_SP.index < end_date)
BrO_SP_V2_17  = BrO_SP[Filter]
# LTcol Pred
Filter        = (BrO_LTP.index >= start_date) & (BrO_LTP.index < end_date)
BrO_LTP_V2_17 = BrO_LTP[Filter]

# Surf Obs
Filter        = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_V2_17  = BrO_SO[Filter]
# LTcol Obs
Filter        = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_V2_17 = BrO_LTO[Filter]

# PC1
Filter        = (PC1.index >= start_date) & (PC1.index < end_date)
PC1_V2_17     = PC1[Filter]
# PC2
Filter        = (PC2.index >= start_date) & (PC2.index < end_date)
PC2_V2_17     = PC2[Filter]
# PC3
Filter        = (PC3.index >= start_date) & (PC3.index < end_date)
PC3_V2_17     = PC3[Filter]

#-------------
# V3 (2017-18)
#-------------
start_date    = '2018-01-27'
end_date      = '2018-02-22'
# Surf Pred
Filter        = (BrO_SP.index >= start_date) & (BrO_SP.index < end_date)
BrO_SP_V3_17  = BrO_SP[Filter]
# LTcol Pred
Filter        = (BrO_LTP.index >= start_date) & (BrO_LTP.index < end_date)
BrO_LTP_V3_17 = BrO_LTP[Filter]

# Surf Obs
Filter        = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_V3_17  = BrO_SO[Filter]
# LTcol Obs
Filter        = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_V3_17 = BrO_LTO[Filter]

# PC1
Filter        = (PC1.index >= start_date) & (PC1.index < end_date)
PC1_V3_17     = PC1[Filter]
# PC2
Filter        = (PC2.index >= start_date) & (PC2.index < end_date)
PC2_V3_17     = PC2[Filter]
# PC3
Filter        = (PC3.index >= start_date) & (PC3.index < end_date)
PC3_V3_17     = PC3[Filter]

#-------------
# 2018-19 Season
#-------------
start_date = '2018-11-01'
end_date   = '2019-02-27'
# Surf Pred
Filter       = (BrO_SP.index >= start_date) & (BrO_SP.index < end_date)
BrO_SP_2018  = BrO_SP[Filter]
# LTcol Pred
Filter       = (BrO_LTP.index >= start_date) & (BrO_LTP.index < end_date)
BrO_LTP_2018 = BrO_LTP[Filter]

# Surf Obs
Filter       = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_2018  = BrO_SO[Filter]
# LTcol Obs
Filter       = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_2018 = BrO_LTO[Filter]

# PC1
Filter       = (PC1.index >= start_date) & (PC1.index < end_date)
PC1_2018     = PC1[Filter]
# PC2
Filter       = (PC2.index >= start_date) & (PC2.index < end_date)
PC2_2018     = PC2[Filter]
# PC3
Filter       = (PC3.index >= start_date) & (PC3.index < end_date)
PC3_2018     = PC3[Filter]

#-------------
# V1 (2018-19)
#-------------
start_date    = '2018-11-07'
end_date      = '2018-11-16'
# Surf Pred
Filter        = (BrO_SP.index >= start_date) & (BrO_SP.index < end_date)
BrO_SP_V1_18  = BrO_SP[Filter]
# LTcol Pred
Filter        = (BrO_LTP.index >= start_date) & (BrO_LTP.index < end_date)
BrO_LTP_V1_18 = BrO_LTP[Filter]

# Surf Obs
Filter        = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_V1_18  = BrO_SO[Filter]
# LTcol Obs
Filter        = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_V1_18 = BrO_LTO[Filter]

# PC1
Filter        = (PC1.index >= start_date) & (PC1.index < end_date)
PC1_V1_18     = PC1[Filter]
# PC2
Filter        = (PC2.index >= start_date) & (PC2.index < end_date)
PC2_V1_18     = PC2[Filter]
# PC3
Filter        = (PC3.index >= start_date) & (PC3.index < end_date)
PC3_V1_18     = PC3[Filter]

#-------------
# V2 (2018-19)
#-------------
start_date    = '2018-12-15'
end_date      = '2018-12-31'
# Surf Pred
Filter        = (BrO_SP.index >= start_date) & (BrO_SP.index < end_date)
BrO_SP_V2_18  = BrO_SP[Filter]
# LTcol Pred
Filter        = (BrO_LTP.index >= start_date) & (BrO_LTP.index < end_date)
BrO_LTP_V2_18 = BrO_LTP[Filter]

# Surf Obs
Filter        = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_V2_18  = BrO_SO[Filter]
# LTcol Obs
Filter        = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_V2_18 = BrO_LTO[Filter]

# PC1
Filter        = (PC1.index >= start_date) & (PC1.index < end_date)
PC1_V2_18     = PC1[Filter]
# PC2
Filter        = (PC2.index >= start_date) & (PC2.index < end_date)
PC2_V2_18     = PC2[Filter]
# PC3
Filter        = (PC3.index >= start_date) & (PC3.index < end_date)
PC3_V2_18     = PC3[Filter]

#-------------
# V3 (2018-19)
#-------------
start_date    = '2019-01-26'
end_date      = '2019-02-20'
# Surf Pred
Filter        = (BrO_SP.index >= start_date) & (BrO_SP.index < end_date)
BrO_SP_V3_18  = BrO_SP[Filter]
# LTcol Pred
Filter        = (BrO_LTP.index >= start_date) & (BrO_LTP.index < end_date)
BrO_LTP_V3_18 = BrO_LTP[Filter]

# Surf Obs
Filter        = (BrO_SO.index >= start_date) & (BrO_SO.index < end_date)
BrO_SO_V3_18  = BrO_SO[Filter]
# LTcol Obs
Filter        = (BrO_LTO.index >= start_date) & (BrO_LTO.index < end_date)
BrO_LTO_V3_18 = BrO_LTO[Filter]

# PC1
Filter        = (PC1.index >= start_date) & (PC1.index < end_date)
PC1_V3_18     = PC1[Filter]
# PC2
Filter        = (PC2.index >= start_date) & (PC2.index < end_date)
PC2_V3_18     = PC2[Filter]
# PC3
Filter        = (PC3.index >= start_date) & (PC3.index < end_date)
PC3_V3_18     = PC3[Filter]

#------------------------------------------------------------------------------
# COUNT THE NUMBER OF OBSERVATIONS OVERALL, EACH SEASON & VOYAGE

#-------------
# All
#-------------
N_All   = len(BrO_SO)
#-------------
# 2017-18 Season
#-------------
N_2017  = len(BrO_SO_2017)
N_V1_17 = len(BrO_SO_V1_17)
N_V2_17 = len(BrO_SO_V2_17)
N_V3_17 = len(BrO_SO_V3_17)
#-------------
# 2018-19 Season
#-------------
N_2018  = len(BrO_SO_2018)
N_V1_18 = len(BrO_SO_V1_18)
N_V2_18 = len(BrO_SO_V2_18)
N_V3_18 = len(BrO_SO_V3_18)

#------------------------------------------------------------------------------
# CALCULATE THE LINEAR CORRELATION COEFFICIENTS (R)
# Between each PC and BrO Obs

#-------------
# All
#-------------
slope_SPC1_All,  intercept_SPC1_All,  rval_SPC1_All,  pval_SPC1_All,  stderr_SPC1_All  = stats.linregress(PC1, BrO_SO)
slope_SPC2_All,  intercept_SPC2_All,  rval_SPC2_All,  pval_SPC2_All,  stderr_SPC2_All  = stats.linregress(PC2, BrO_SO)
slope_SPC3_All,  intercept_SPC3_All,  rval_SPC3_All,  pval_SPC3_All,  stderr_SPC3_All  = stats.linregress(PC3, BrO_SO)

slope_LTPC1_All, intercept_LTPC1_All, rval_LTPC1_All, pval_LTPC1_All, stderr_LTPC1_All = stats.linregress(PC1, BrO_LTO)
slope_LTPC2_All, intercept_LTPC2_All, rval_LTPC2_All, pval_LTPC2_All, stderr_LTPC2_All = stats.linregress(PC2, BrO_LTO)
slope_LTPC3_All, intercept_LTPC3_All, rval_LTPC3_All, pval_LTPC3_All, stderr_LTPC3_All = stats.linregress(PC3, BrO_LTO)

#-------------
# 2017-18 Season
#-------------
slope_SPC1_2017,  intercept_SPC1_2017,  rval_SPC1_2017,  pval_SPC1_2017,  stderr_SPC1_2017  = stats.linregress(PC1_2017,  BrO_SO_2017)
slope_SPC2_2017,  intercept_SPC2_2017,  rval_SPC2_2017,  pval_SPC2_2017,  stderr_SPC2_2017  = stats.linregress(PC2_2017,  BrO_SO_2017)
slope_SPC3_2017,  intercept_SPC3_2017,  rval_SPC3_2017,  pval_SPC3_2017,  stderr_SPC3_2017  = stats.linregress(PC3_2017,  BrO_SO_2017)

slope_LTPC1_2017, intercept_LTPC1_2017, rval_LTPC1_2017, pval_LTPC1_2017, stderr_LTPC1_2017 = stats.linregress(PC1_2017, BrO_LTO_2017)
slope_LTPC2_2017, intercept_LTPC2_2017, rval_LTPC2_2017, pval_LTPC2_2017, stderr_LTPC2_2017 = stats.linregress(PC2_2017, BrO_LTO_2017)
slope_LTPC3_2017, intercept_LTPC3_2017, rval_LTPC3_2017, pval_LTPC2_2017, stderr_LTPC3_2017 = stats.linregress(PC3_2017, BrO_LTO_2017)

# V1 (2017-18)
slope_SPC1_V1_17,  intercept_SPC1_V1_17,  rval_SPC1_V1_17,  pval_SPC1_V1_17,  stderr_SPC1_V1_17  = stats.linregress(PC1_V1_17,  BrO_SO_V1_17)
slope_SPC2_V1_17,  intercept_SPC2_V1_17,  rval_SPC2_V1_17,  pval_SPC2_V1_17,  stderr_SPC2_V1_17  = stats.linregress(PC2_V1_17,  BrO_SO_V1_17)
slope_SPC3_V1_17,  intercept_SPC3_V1_17,  rval_SPC3_V1_17,  pval_SPC3_V1_17,  stderr_SPC3_V1_17  = stats.linregress(PC3_V1_17,  BrO_SO_V1_17)

slope_LTPC1_V1_17, intercept_LTPC1_V1_17, rval_LTPC1_V1_17, pval_LTPC1_V1_17, stderr_LTPC1_V1_17 = stats.linregress(PC1_V1_17, BrO_LTO_V1_17)
slope_LTPC2_V1_17, intercept_LTPC2_V1_17, rval_LTPC2_V1_17, pval_LTPC2_V1_17, stderr_LTPC2_V1_17 = stats.linregress(PC2_V1_17, BrO_LTO_V1_17)
slope_LTPC3_V1_17, intercept_LTPC3_V1_17, rval_LTPC3_V1_17, pval_LTPC2_V1_17, stderr_LTPC3_V1_17 = stats.linregress(PC3_V1_17, BrO_LTO_V1_17)

# V2 (2017-18)
slope_SPC1_V2_17,  intercept_SPC1_V2_17,  rval_SPC1_V2_17,  pval_SPC1_V2_17,  stderr_SPC1_V2_17  = stats.linregress(PC1_V2_17,  BrO_SO_V2_17)
slope_SPC2_V2_17,  intercept_SPC2_V2_17,  rval_SPC2_V2_17,  pval_SPC2_V2_17,  stderr_SPC2_V2_17  = stats.linregress(PC2_V2_17,  BrO_SO_V2_17)
slope_SPC3_V2_17,  intercept_SPC3_V2_17,  rval_SPC3_V2_17,  pval_SPC3_V2_17,  stderr_SPC3_V2_17  = stats.linregress(PC3_V2_17,  BrO_SO_V2_17)

slope_LTPC1_V2_17, intercept_LTPC1_V2_17, rval_LTPC1_V2_17, pval_LTPC1_V2_17, stderr_LTPC1_V2_17 = stats.linregress(PC1_V2_17, BrO_LTO_V2_17)
slope_LTPC2_V2_17, intercept_LTPC2_V2_17, rval_LTPC2_V2_17, pval_LTPC2_V2_17, stderr_LTPC2_V2_17 = stats.linregress(PC2_V2_17, BrO_LTO_V2_17)
slope_LTPC3_V2_17, intercept_LTPC3_V2_17, rval_LTPC3_V2_17, pval_LTPC2_V2_17, stderr_LTPC3_V2_17 = stats.linregress(PC3_V2_17, BrO_LTO_V2_17)

# V3 (2017-18)
slope_SPC1_V3_17,  intercept_SPC1_V3_17,  rval_SPC1_V3_17,  pval_SPC1_V3_17,  stderr_SPC1_V3_17  = stats.linregress(PC1_V3_17,  BrO_SO_V3_17)
slope_SPC2_V3_17,  intercept_SPC2_V3_17,  rval_SPC2_V3_17,  pval_SPC2_V3_17,  stderr_SPC2_V3_17  = stats.linregress(PC2_V3_17,  BrO_SO_V3_17)
slope_SPC3_V3_17,  intercept_SPC3_V3_17,  rval_SPC3_V3_17,  pval_SPC3_V3_17,  stderr_SPC3_V3_17  = stats.linregress(PC3_V3_17,  BrO_SO_V3_17)

slope_LTPC1_V3_17, intercept_LTPC1_V3_17, rval_LTPC1_V3_17, pval_LTPC1_V3_17, stderr_LTPC1_V3_17 = stats.linregress(PC1_V3_17, BrO_LTO_V3_17)
slope_LTPC2_V3_17, intercept_LTPC2_V3_17, rval_LTPC2_V3_17, pval_LTPC2_V3_17, stderr_LTPC2_V3_17 = stats.linregress(PC2_V3_17, BrO_LTO_V3_17)
slope_LTPC3_V3_17, intercept_LTPC3_V3_17, rval_LTPC3_V3_17, pval_LTPC2_V3_17, stderr_LTPC3_V3_17 = stats.linregress(PC3_V3_17, BrO_LTO_V3_17)

#-------------
# 2018-19 Season
#-------------
slope_SPC1_2018,  intercept_SPC1_2018,  rval_SPC1_2018,  pval_SPC1_2018,  stderr_SPC1_2017  = stats.linregress(PC1_2018,  BrO_SO_2018)
slope_SPC2_2018,  intercept_SPC2_2018,  rval_SPC2_2018,  pval_SPC2_2018,  stderr_SPC2_2017  = stats.linregress(PC2_2018,  BrO_SO_2018)
slope_SPC3_2018,  intercept_SPC3_2018,  rval_SPC3_2018,  pval_SPC3_2018,  stderr_SPC3_2017  = stats.linregress(PC3_2018,  BrO_SO_2018)

slope_LTPC1_2018, intercept_LTPC1_2018, rval_LTPC1_2018, pval_LTPC1_2018, stderr_LTPC1_2017 = stats.linregress(PC1_2018, BrO_LTO_2018)
slope_LTPC2_2018, intercept_LTPC2_2018, rval_LTPC2_2018, pval_LTPC2_2018, stderr_LTPC2_2017 = stats.linregress(PC2_2018, BrO_LTO_2018)
slope_LTPC3_2018, intercept_LTPC3_2018, rval_LTPC3_2018, pval_LTPC3_2018, stderr_LTPC3_2017 = stats.linregress(PC3_2018, BrO_LTO_2018)

# V1 (2018-19)
slope_SPC1_V1_18,  intercept_SPC1_V1_18,  rval_SPC1_V1_18,  pval_SPC1_V1_18,  stderr_SPC1_V1_18  = stats.linregress(PC1_V1_18,  BrO_SO_V1_18)
slope_SPC2_V1_18,  intercept_SPC2_V1_18,  rval_SPC2_V1_18,  pval_SPC2_V1_18,  stderr_SPC2_V1_18  = stats.linregress(PC2_V1_18,  BrO_SO_V1_18)
slope_SPC3_V1_18,  intercept_SPC3_V1_18,  rval_SPC3_V1_18,  pval_SPC3_V1_18,  stderr_SPC3_V1_18  = stats.linregress(PC3_V1_18,  BrO_SO_V1_18)

slope_LTPC1_V1_18, intercept_LTPC1_V1_18, rval_LTPC1_V1_18, pval_LTPC1_V1_18, stderr_LTPC1_V1_18 = stats.linregress(PC1_V1_18, BrO_LTO_V1_18)
slope_LTPC2_V1_18, intercept_LTPC2_V1_18, rval_LTPC2_V1_18, pval_LTPC2_V1_18, stderr_LTPC2_V1_18 = stats.linregress(PC2_V1_18, BrO_LTO_V1_18)
slope_LTPC3_V1_18, intercept_LTPC3_V1_18, rval_LTPC3_V1_18, pval_LTPC2_V1_18, stderr_LTPC3_V1_18 = stats.linregress(PC3_V1_18, BrO_LTO_V1_18)

# V2 (2018-19)
slope_SPC1_V2_18,  intercept_SPC1_V2_18,  rval_SPC1_V2_18,  pval_SPC1_V2_18,  stderr_SPC1_V2_18  = stats.linregress(PC1_V2_18,  BrO_SO_V2_18)
slope_SPC2_V2_18,  intercept_SPC2_V2_18,  rval_SPC2_V2_18,  pval_SPC2_V2_18,  stderr_SPC2_V2_18  = stats.linregress(PC2_V2_18,  BrO_SO_V2_18)
slope_SPC3_V2_18,  intercept_SPC3_V2_18,  rval_SPC3_V2_18,  pval_SPC3_V2_18,  stderr_SPC3_V2_18  = stats.linregress(PC3_V2_18,  BrO_SO_V2_18)

slope_LTPC1_V2_18, intercept_LTPC1_V2_18, rval_LTPC1_V2_18, pval_LTPC1_V2_18, stderr_LTPC1_V2_18 = stats.linregress(PC1_V2_18, BrO_LTO_V2_18)
slope_LTPC2_V2_18, intercept_LTPC2_V2_18, rval_LTPC2_V2_18, pval_LTPC2_V2_18, stderr_LTPC2_V2_18 = stats.linregress(PC2_V2_18, BrO_LTO_V2_18)
slope_LTPC3_V2_18, intercept_LTPC3_V2_18, rval_LTPC3_V2_18, pval_LTPC2_V2_18, stderr_LTPC3_V2_18 = stats.linregress(PC3_V2_18, BrO_LTO_V2_18)

# V3 (2018-19)
slope_SPC1_V3_18,  intercept_SPC1_V3_18,  rval_SPC1_V3_18,  pval_SPC1_V3_18,  stderr_SPC1_V3_18  = stats.linregress(PC1_V3_18,  BrO_SO_V3_18)
slope_SPC2_V3_18,  intercept_SPC2_V3_18,  rval_SPC2_V3_18,  pval_SPC2_V3_18,  stderr_SPC2_V3_18  = stats.linregress(PC2_V3_18,  BrO_SO_V3_18)
slope_SPC3_V3_18,  intercept_SPC3_V3_18,  rval_SPC3_V3_18,  pval_SPC3_V3_18,  stderr_SPC3_V3_18  = stats.linregress(PC3_V3_18,  BrO_SO_V3_18)

slope_LTPC1_V3_18, intercept_LTPC1_V3_18, rval_LTPC1_V3_18, pval_LTPC1_V3_18, stderr_LTPC1_V3_18 = stats.linregress(PC1_V3_18, BrO_LTO_V3_18)
slope_LTPC2_V3_18, intercept_LTPC2_V3_18, rval_LTPC2_V3_18, pval_LTPC2_V3_18, stderr_LTPC2_V3_18 = stats.linregress(PC2_V3_18, BrO_LTO_V3_18)
slope_LTPC3_V3_18, intercept_LTPC3_V3_18, rval_LTPC3_V3_18, pval_LTPC2_V3_18, stderr_LTPC3_V3_18 = stats.linregress(PC3_V3_18, BrO_LTO_V3_18)

#------------------------------------------------------------------------------
# CALCULATE THE LINEAR DETERMINATION COEFFICIENTS (R2)

#-------------
# All
#-------------
R2_SPC1_All  = np.square(rval_SPC1_All)
R2_SPC2_All  = np.square(rval_SPC2_All)
R2_SPC3_All  = np.square(rval_SPC3_All)

R2_LTPC1_All = np.square(rval_LTPC1_All)
R2_LTPC2_All = np.square(rval_LTPC2_All)
R2_LTPC3_All = np.square(rval_LTPC3_All)

#-------------
# 2017-18 Season
#-------------
R2_SPC1_2017  = np.square(rval_SPC1_2017)
R2_SPC2_2017  = np.square(rval_SPC2_2017)
R2_SPC3_2017  = np.square(rval_SPC3_2017)

R2_LTPC1_2017 = np.square(rval_LTPC1_2017)
R2_LTPC2_2017 = np.square(rval_LTPC2_2017)
R2_LTPC3_2017 = np.square(rval_LTPC3_2017)

# V1 (2017-18)
R2_SPC1_V1_17  = np.square(rval_SPC1_V1_17)
R2_SPC2_V1_17  = np.square(rval_SPC2_V1_17)
R2_SPC3_V1_17  = np.square(rval_SPC3_V1_17)

R2_LTPC1_V1_17 = np.square(rval_LTPC1_V1_17)
R2_LTPC2_V1_17 = np.square(rval_LTPC2_V1_17)
R2_LTPC3_V1_17 = np.square(rval_LTPC3_V1_17)

# V2 (2017-18)
R2_SPC1_V2_17  = np.square(rval_SPC1_V2_17)
R2_SPC2_V2_17  = np.square(rval_SPC2_V2_17)
R2_SPC3_V2_17  = np.square(rval_SPC3_V2_17)

R2_LTPC1_V2_17 = np.square(rval_LTPC1_V2_17)
R2_LTPC2_V2_17 = np.square(rval_LTPC2_V2_17)
R2_LTPC3_V2_17 = np.square(rval_LTPC3_V2_17)

# V3 (2017-18)
R2_SPC1_V3_17  = np.square(rval_SPC1_V3_17)
R2_SPC2_V3_17  = np.square(rval_SPC2_V3_17)
R2_SPC3_V3_17  = np.square(rval_SPC3_V3_17)

R2_LTPC1_V3_17 = np.square(rval_LTPC1_V3_17)
R2_LTPC2_V3_17 = np.square(rval_LTPC2_V3_17)
R2_LTPC3_V3_17 = np.square(rval_LTPC3_V3_17)

#-------------
# 2018-19 Season
#-------------
R2_SPC1_2018  = np.square(rval_SPC1_2018)
R2_SPC2_2018  = np.square(rval_SPC2_2018)
R2_SPC3_2018  = np.square(rval_SPC3_2018)

R2_LTPC1_2018 = np.square(rval_LTPC1_2018)
R2_LTPC2_2018 = np.square(rval_LTPC2_2018)
R2_LTPC3_2018 = np.square(rval_LTPC3_2018)

# V1 (2018-19)
R2_SPC1_V1_18  = np.square(rval_SPC1_V1_18)
R2_SPC2_V1_18  = np.square(rval_SPC2_V1_18)
R2_SPC3_V1_18  = np.square(rval_SPC3_V1_18)

R2_LTPC1_V1_18 = np.square(rval_LTPC1_V1_18)
R2_LTPC2_V1_18 = np.square(rval_LTPC2_V1_18)
R2_LTPC3_V1_18 = np.square(rval_LTPC3_V1_18)

# V2 (2018-19)
R2_SPC1_V2_18  = np.square(rval_SPC1_V2_18)
R2_SPC2_V2_18  = np.square(rval_SPC2_V2_18)
R2_SPC3_V2_18  = np.square(rval_SPC3_V2_18)

R2_LTPC1_V2_18 = np.square(rval_LTPC1_V2_18)
R2_LTPC2_V2_18 = np.square(rval_LTPC2_V2_18)
R2_LTPC3_V2_18 = np.square(rval_LTPC3_V2_18)

# V3 (2018-19)
R2_SPC1_V3_18  = np.square(rval_SPC1_V3_18)
R2_SPC2_V3_18  = np.square(rval_SPC2_V3_18)
R2_SPC3_V3_18  = np.square(rval_SPC3_V3_18)

R2_LTPC1_V3_18 = np.square(rval_LTPC1_V3_18)
R2_LTPC2_V3_18 = np.square(rval_LTPC2_V3_18)
R2_LTPC3_V3_18 = np.square(rval_LTPC3_V3_18)

#------------------------------------------------------------------------------
# CALCULATE THE NONPARAMETRIC SPEARMAN RANK CORRELATION COEFFICIENTS (R)

#-------------
# All
#-------------
rho_SPC1_All,  pval_SPC1_All  = stats.spearmanr(PC1, BrO_SO)
rho_SPC2_All,  pval_SPC2_All  = stats.spearmanr(PC2, BrO_SO)
rho_SPC3_All,  pval_SPC3_All  = stats.spearmanr(PC3, BrO_SO)

rho_LTPC1_All, pval_LTPC1_All = stats.spearmanr(PC1, BrO_LTO)
rho_LTPC2_All, pval_LTPC2_All = stats.spearmanr(PC2, BrO_LTO)
rho_LTPC3_All, pval_LTPC3_All = stats.spearmanr(PC3, BrO_LTO)

#-------------
# 2017-18 Season
#-------------
rho_SPC1_2017,  pval_SPC1_2017  = stats.spearmanr(PC1_2017, BrO_SO_2017)
rho_SPC2_2017,  pval_SPC2_2017  = stats.spearmanr(PC2_2017, BrO_SO_2017)
rho_SPC3_2017,  pval_SPC3_2017  = stats.spearmanr(PC3_2017, BrO_SO_2017)

rho_LTPC1_2017, pval_LTPC1_2017 = stats.spearmanr(PC1_2017, BrO_LTO_2017)
rho_LTPC2_2017, pval_LTPC2_2017 = stats.spearmanr(PC2_2017, BrO_LTO_2017)
rho_LTPC3_2017, pval_LTPC3_2017 = stats.spearmanr(PC3_2017, BrO_LTO_2017)

# V1 (2017-18)
rho_SPC1_V1_17,  pval_SPC1_V1_17  = stats.spearmanr(PC1_V1_17, BrO_SO_V1_17)
rho_SPC2_V1_17,  pval_SPC2_V1_17  = stats.spearmanr(PC2_V1_17, BrO_SO_V1_17)
rho_SPC3_V1_17,  pval_SPC3_V1_17  = stats.spearmanr(PC3_V1_17, BrO_SO_V1_17)

rho_LTPC1_V1_17, pval_LTPC1_V1_17 = stats.spearmanr(PC1_V1_17, BrO_LTO_V1_17)
rho_LTPC2_V1_17, pval_LTPC2_V1_17 = stats.spearmanr(PC2_V1_17, BrO_LTO_V1_17)
rho_LTPC3_V1_17, pval_LTPC3_V1_17 = stats.spearmanr(PC3_V1_17, BrO_LTO_V1_17)

# V2 (2017-18)
rho_SPC1_V2_17,  pval_SPC1_V2_17  = stats.spearmanr(PC1_V2_17, BrO_SO_V2_17)
rho_SPC2_V2_17,  pval_SPC2_V2_17  = stats.spearmanr(PC2_V2_17, BrO_SO_V2_17)
rho_SPC3_V2_17,  pval_SPC3_V2_17  = stats.spearmanr(PC3_V2_17, BrO_SO_V2_17)

rho_LTPC1_V2_17, pval_LTPC1_V2_17 = stats.spearmanr(PC1_V2_17, BrO_LTO_V2_17)
rho_LTPC2_V2_17, pval_LTPC2_V2_17 = stats.spearmanr(PC2_V2_17, BrO_LTO_V2_17)
rho_LTPC3_V2_17, pval_LTPC3_V2_17 = stats.spearmanr(PC3_V2_17, BrO_LTO_V2_17)

# V3 (2017-18)
rho_SPC1_V3_17,  pval_SPC1_V3_17  = stats.spearmanr(PC1_V3_17, BrO_SO_V3_17)
rho_SPC2_V3_17,  pval_SPC2_V3_17  = stats.spearmanr(PC2_V3_17, BrO_SO_V3_17)
rho_SPC3_V3_17,  pval_SPC3_V3_17  = stats.spearmanr(PC3_V3_17, BrO_SO_V3_17)

rho_LTPC1_V3_17, pval_LTPC1_V3_17 = stats.spearmanr(PC1_V3_17, BrO_LTO_V3_17)
rho_LTPC2_V3_17, pval_LTPC2_V3_17 = stats.spearmanr(PC2_V3_17, BrO_LTO_V3_17)
rho_LTPC3_V3_17, pval_LTPC3_V3_17 = stats.spearmanr(PC3_V3_17, BrO_LTO_V3_17)

#-------------
# 2018-19 Season
#-------------
rho_SPC1_2018,  pval_SPC1_2018  = stats.spearmanr(PC1_2018, BrO_SO_2018)
rho_SPC2_2018,  pval_SPC2_2018  = stats.spearmanr(PC2_2018, BrO_SO_2018)
rho_SPC3_2018,  pval_SPC3_2018  = stats.spearmanr(PC3_2018, BrO_SO_2018)

rho_LTPC1_2018, pval_LTPC1_2018 = stats.spearmanr(PC1_2018, BrO_LTO_2018)
rho_LTPC2_2018, pval_LTPC2_2018 = stats.spearmanr(PC2_2018, BrO_LTO_2018)
rho_LTPC3_2018, pval_LTPC3_2018 = stats.spearmanr(PC3_2018, BrO_LTO_2018)

# V1 (2018-19)
rho_SPC1_V1_18,  pval_SPC1_V1_18  = stats.spearmanr(PC1_V1_18, BrO_SO_V1_18)
rho_SPC2_V1_18,  pval_SPC2_V1_18  = stats.spearmanr(PC2_V1_18, BrO_SO_V1_18)
rho_SPC3_V1_18,  pval_SPC3_V1_18  = stats.spearmanr(PC3_V1_18, BrO_SO_V1_18)

rho_LTPC1_V1_18, pval_LTPC1_V1_18 = stats.spearmanr(PC1_V1_18, BrO_LTO_V1_18)
rho_LTPC2_V1_18, pval_LTPC2_V1_18 = stats.spearmanr(PC2_V1_18, BrO_LTO_V1_18)
rho_LTPC3_V1_18, pval_LTPC3_V1_18 = stats.spearmanr(PC3_V1_18, BrO_LTO_V1_18)

# V2 (2018-19)
rho_SPC1_V2_18,  pval_SPC1_V2_18  = stats.spearmanr(PC1_V2_18, BrO_SO_V2_18)
rho_SPC2_V2_18,  pval_SPC2_V2_18  = stats.spearmanr(PC2_V2_18, BrO_SO_V2_18)
rho_SPC3_V2_18,  pval_SPC3_V2_18  = stats.spearmanr(PC3_V2_18, BrO_SO_V2_18)

rho_LTPC1_V2_18, pval_LTPC1_V2_18 = stats.spearmanr(PC1_V2_18, BrO_LTO_V2_18)
rho_LTPC2_V2_18, pval_LTPC2_V2_18 = stats.spearmanr(PC2_V2_18, BrO_LTO_V2_18)
rho_LTPC3_V2_18, pval_LTPC3_V2_18 = stats.spearmanr(PC3_V2_18, BrO_LTO_V2_18)

# V3 (2018-19)
rho_SPC1_V3_18,  pval_SPC1_V3_18  = stats.spearmanr(PC1_V3_18, BrO_SO_V3_18)
rho_SPC2_V3_18,  pval_SPC2_V3_18  = stats.spearmanr(PC2_V3_18, BrO_SO_V3_18)
rho_SPC3_V3_18,  pval_SPC3_V3_18  = stats.spearmanr(PC3_V3_18, BrO_SO_V3_18)

rho_LTPC1_V3_18, pval_LTPC1_V3_18 = stats.spearmanr(PC1_V3_18, BrO_LTO_V3_18)
rho_LTPC2_V3_18, pval_LTPC2_V3_18 = stats.spearmanr(PC2_V3_18, BrO_LTO_V3_18)
rho_LTPC3_V3_18, pval_LTPC3_V3_18 = stats.spearmanr(PC3_V3_18, BrO_LTO_V3_18)

#------------------------------------------------------------------------------
#BUILD DATAFRAME FOR THE STATISTICAL RESULTS

# Build a pandas dataframe
dfPC_CorrCoef = {'No':[N_All, N_All,
                       N_2017, N_2017,
                       N_V1_17, N_V1_17, N_V2_17, N_V2_17, N_V3_17, N_V3_17,
                       N_2018, N_2018,
                       N_V1_18, N_V1_18, N_V2_18, N_V2_18, N_V3_18, N_V3_18],
                 'PC1 (R)':  [rval_LTPC1_All,   rval_SPC1_All,
                              rval_LTPC1_2017,  rval_SPC1_2017,
                              rval_LTPC1_V1_17, rval_SPC1_V1_17, rval_LTPC1_V2_17, rval_SPC1_V2_17, rval_LTPC1_V3_17, rval_SPC1_V3_17,
                              rval_LTPC1_2018,  rval_SPC1_2018,
                              rval_LTPC1_V1_18, rval_SPC1_V1_18, rval_LTPC1_V2_18, rval_SPC1_V2_18, rval_LTPC1_V3_18, rval_SPC1_V3_18],
                 'PC2 (R)':  [rval_LTPC2_All,   rval_SPC2_All,
                              rval_LTPC2_2017,  rval_SPC2_2017,
                              rval_LTPC2_V1_17, rval_SPC2_V1_17, rval_LTPC2_V2_17, rval_SPC2_V2_17, rval_LTPC2_V3_17, rval_SPC2_V3_17,
                              rval_LTPC2_2018,  rval_SPC2_2018,
                              rval_LTPC2_V1_18, rval_SPC2_V1_18, rval_LTPC2_V2_18, rval_SPC2_V2_18, rval_LTPC2_V3_18, rval_SPC2_V3_18],
                 'PC3 (R)':  [rval_LTPC3_All,   rval_SPC3_All,
                              rval_LTPC3_2017,  rval_SPC3_2017,
                              rval_LTPC3_V1_17, rval_SPC3_V1_17, rval_LTPC3_V2_17, rval_SPC3_V2_17, rval_LTPC3_V3_17, rval_SPC3_V3_17,
                              rval_LTPC3_2018,  rval_SPC3_2018,
                              rval_LTPC3_V1_18, rval_SPC3_V1_18, rval_LTPC3_V2_18, rval_SPC3_V2_18, rval_LTPC3_V3_18, rval_SPC3_V3_18],
                 'PC1 (R2)': [R2_LTPC1_All,     R2_SPC1_All,
                              R2_LTPC1_2017,    R2_SPC1_2017,
                              R2_LTPC1_V1_17,   R2_SPC1_V1_17,   R2_LTPC1_V2_17,   R2_SPC1_V2_17,   R2_LTPC1_V3_17,   R2_SPC1_V3_17,
                              R2_LTPC1_2018,    R2_SPC1_2018,
                              R2_LTPC1_V1_18,   R2_SPC1_V1_18,   R2_LTPC1_V2_18,   R2_SPC1_V2_18,   R2_LTPC1_V3_18,   R2_SPC1_V3_18],
                 'PC2 (R2)': [R2_LTPC2_All,     R2_SPC2_All,
                              R2_LTPC2_2017,    R2_SPC2_2017,
                              R2_LTPC2_V1_17,   R2_SPC2_V1_17,   R2_LTPC2_V2_17,   R2_SPC2_V2_17,   R2_LTPC2_V3_17,   R2_SPC2_V3_17,
                              R2_LTPC2_2018,    R2_SPC2_2018,
                              R2_LTPC2_V1_18,   R2_SPC2_V1_18,   R2_LTPC2_V2_18,   R2_SPC2_V2_18,   R2_LTPC2_V3_18,   R2_SPC2_V3_18],
                 'PC3 (R2)': [R2_LTPC3_All,     R2_SPC3_All,
                              R2_LTPC3_2017,    R2_SPC3_2017,
                              R2_LTPC3_V1_17,   R2_SPC3_V1_17,   R2_LTPC3_V2_17,   R2_SPC3_V2_17,   R2_LTPC3_V3_17,   R2_SPC3_V3_17,
                              R2_LTPC3_2018,    R2_SPC3_2018,
                              R2_LTPC3_V1_18,   R2_SPC3_V1_18,   R2_LTPC3_V2_18,   R2_SPC3_V2_18,   R2_LTPC3_V3_18,   R2_SPC3_V3_18],
                 'PC1 (p)':  [rho_LTPC1_All,    rho_SPC1_All,
                              rho_LTPC1_2017,   rho_SPC1_2017,
                              rho_LTPC1_V1_17,  rho_SPC1_V1_17,  rho_LTPC1_V2_17,  rho_SPC1_V2_17,  rho_LTPC1_V3_17,  rho_SPC1_V3_17,
                              rho_LTPC1_2018,   rho_SPC1_2018,
                              rho_LTPC1_V1_18,  rho_SPC1_V1_18,  rho_LTPC1_V2_18,  rho_SPC1_V2_18,  rho_LTPC1_V3_18,  rho_SPC1_V3_18],
                 'PC2 (p)':  [rho_LTPC2_All,    rho_SPC2_All,
                              rho_LTPC2_2017,   rho_SPC2_2017,
                              rho_LTPC2_V1_17,  rho_SPC2_V1_17,  rho_LTPC2_V2_17,  rho_SPC2_V2_17,  rho_LTPC2_V3_17,  rho_SPC2_V3_17,
                              rho_LTPC2_2018,   rho_SPC2_2018,
                              rho_LTPC2_V1_18,  rho_SPC2_V1_18,  rho_LTPC2_V2_18,  rho_SPC2_V2_18,  rho_LTPC2_V3_18,  rho_SPC2_V3_18],
                 'PC3 (p)':  [rho_LTPC3_All,    rho_SPC3_All,
                              rho_LTPC3_2017,   rho_SPC3_2017,
                              rho_LTPC3_V1_17,  rho_SPC3_V1_17,  rho_LTPC3_V2_17,  rho_SPC3_V2_17,  rho_LTPC3_V3_17,  rho_SPC3_V3_17,
                              rho_LTPC3_2018,   rho_SPC3_2018,
                              rho_LTPC3_V1_18,  rho_SPC3_V1_18,  rho_LTPC3_V2_18,  rho_SPC3_V2_18,  rho_LTPC3_V3_18,  rho_SPC3_V3_18]}
dfPC_CorrCoef = pd.DataFrame(dfPC_CorrCoef, columns = ['No','PC1 (R)','PC2 (R)','PC3 (R)','PC1 (R2)','PC2 (R2)','PC3 (R2)','PC1 (p)','PC2 (p)','PC3 (p)'],index = ['LTcol All','Surface All','LTcol 2017','Surface 2017','LTcol V1_17','Surface V1_17','LTcol V2_17','Surface V2_17','LTcol V3_17','Surface V3_17','LTcol 2018','Surface 2018','LTcol V1_18','Surface V1_18','LTcol V2_18','Surface V2_18','LTcol V3_18','Surface V3_18'])
dfPC_CorrCoef.to_csv('/Users/ncp532/Documents/Data/MERRA2/PC_CorrCoef.csv')

#------------------------------------------------------------------------------
# PLOT THE GRAPH (BrOLTcol All PCs)

fig = plt.figure()

gs = gridspec.GridSpec(nrows=3,
                       ncols=4, 
                       figure=fig, 
                       width_ratios= [0.05,0.5,0.5,0.05],
                       height_ratios=[0.25, 0.25, 0.25],
                       hspace=0.3, wspace=0.35)

#-------------------------------------
# Graph 1 (V1 2017-18)
ax = plt.subplot(gs[0,1:2])
ax2 = ax.twinx()

# Plot PC1, PC2 & PC3
ln1 = ax.plot(Date, PC1,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='PC1 (R: '+str("%6.4f"%(rval_SPC1_V1_17))+')')  # PC1
ln2 = ax.plot(Date, PC2,   marker='o', c='b',      markersize = 2.0, linestyle='none', label='PC2 (R: '+str("%6.4f"%(rval_SPC2_V1_17))+')')  # PC2
ln3 = ax.plot(Date, PC3,   marker='o', c='g',      markersize = 2.0, linestyle='none', label='PC3 (R: '+str("%6.4f"%(rval_SPC3_V1_17))+')')  # PC3
# Plot BrO
ln4 = ax2.plot(Date, BrO_LTP,   marker='o', c='purple', markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')
ln5 = ax2.plot(Date, BrO_LTO,   marker='o', c='black',  markersize = 2.0, linestyle='none', label='BrO LTcol Observations')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5,5)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#ax2.set_ylim(-0.3,0.3)
ax2.set_ylim(-1.9,1.9)

# Plot the axis labels, legend and title
plt.title('V1 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('PCs', fontsize=10)
#ax2.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)

# Plot the legend
lns = ln1+ln2+ln3+ln4+ln5
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, bbox_to_anchor=(-0.5, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 2 (V2 2017-18)
ax = plt.subplot(gs[1,1:2])
ax2 = ax.twinx()

# Plot PC1, PC2 & PC3
ln1 = ax.plot(Date, PC1,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='PC1 (R: '+str("%6.4f"%(rval_SPC1_V2_17))+')')  # PC1
ln2 = ax.plot(Date, PC2,   marker='o', c='b',      markersize = 2.0, linestyle='none', label='PC2 (R: '+str("%6.4f"%(rval_SPC2_V2_17))+')')  # PC2
ln3 = ax.plot(Date, PC3,   marker='o', c='g',      markersize = 2.0, linestyle='none', label='PC3 (R: '+str("%6.4f"%(rval_SPC3_V2_17))+')')  # PC3
# Plot BrO
ln4 = ax2.plot(Date, BrO_LTP,   marker='o', c='purple', markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')
ln5 = ax2.plot(Date, BrO_LTO,   marker='o', c='black',  markersize = 2.0, linestyle='none', label='BrO LTcol Observations')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5,5)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#ax2.set_ylim(-0.3,0.3)
ax2.set_ylim(-1.9,1.9)

# Plot the axis labels, legend and title
plt.title('V2 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('PCs', fontsize=10)
#ax2.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)

# Plot the legend
lns = ln1+ln2+ln3+ln4+ln5
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, bbox_to_anchor=(-0.5, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 3 (V3 2017-18)
ax = plt.subplot(gs[2,1:2])
ax2 = ax.twinx()

# Plot PC1, PC2 & PC3
ln1 = ax.plot(Date, PC1,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='PC1 (R: '+str("%6.4f"%(rval_SPC1_V3_17))+')')  # PC1
ln2 = ax.plot(Date, PC2,   marker='o', c='b',      markersize = 2.0, linestyle='none', label='PC2 (R: '+str("%6.4f"%(rval_SPC2_V3_17))+')')  # PC2
ln3 = ax.plot(Date, PC3,   marker='o', c='g',      markersize = 2.0, linestyle='none', label='PC3 (R: '+str("%6.4f"%(rval_SPC3_V3_17))+')')  # PC3
# Plot BrO
ln4 = ax2.plot(Date, BrO_LTP,   marker='o', c='purple', markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')
ln3 = ax2.plot(Date, BrO_LTO,   marker='o', c='black',  markersize = 2.0, linestyle='none', label='BrO LTcol Observations')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5,5)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#ax2.set_ylim(-0.3,0.3)
ax2.set_ylim(-1.9,1.9)

# Plot the axis labels, legend and title
plt.title('V3 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('PCs', fontsize=10)
#ax2.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
ax.set_xlabel('Date', fontsize=15)

# Plot the legend
lns = ln1+ln2+ln3+ln4+ln5
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, bbox_to_anchor=(-0.5, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 4 (V1 2018-19)
ax = plt.subplot(gs[0,2:3])
ax2 = ax.twinx()

# Plot PC1, PC2 & PC3
ln1 = ax.plot(Date, PC1,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='PC1 (R: '+str("%6.4f"%(rval_SPC1_V1_18))+')')  # PC1
ln2 = ax.plot(Date, PC2,   marker='o', c='b',      markersize = 2.0, linestyle='none', label='PC2 (R: '+str("%6.4f"%(rval_SPC2_V1_18))+')')  # PC2
ln3 = ax.plot(Date, PC3,   marker='o', c='g',      markersize = 2.0, linestyle='none', label='PC3 (R: '+str("%6.4f"%(rval_SPC3_V1_18))+')')  # PC3
# Plot BrO
ln4 = ax2.plot(Date, BrO_LTP,   marker='o', c='purple', markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')
ln5 = ax2.plot(Date, BrO_LTO,   marker='o', c='black',  markersize = 2.0, linestyle='none', label='BrO LTcol Observations')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5,5)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#ax2.set_ylim(-0.3,0.3)
ax2.set_ylim(-1.9,1.9)

# Plot the axis labels and title
plt.title('V1 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('PCs', fontsize=10)
ax2.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)

# Plot the legend
lns = ln1+ln2+ln3+ln4+ln5
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, bbox_to_anchor=(1.15, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 5 (V2 2018-19)
ax = plt.subplot(gs[1,2:3])
ax2 = ax.twinx()

# Plot PC1, PC2 & PC3
ln1 = ax.plot(Date, PC1,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='PC1 (R: '+str("%6.4f"%(rval_SPC1_V2_18))+')')  # PC1
ln2 = ax.plot(Date, PC2,   marker='o', c='b',      markersize = 2.0, linestyle='none', label='PC2 (R: '+str("%6.4f"%(rval_SPC2_V2_18))+')')  # PC2
ln3 = ax.plot(Date, PC3,   marker='o', c='g',      markersize = 2.0, linestyle='none', label='PC3 (R: '+str("%6.4f"%(rval_SPC3_V2_18))+')')  # PC3
# Plot BrO
ln4 = ax2.plot(Date, BrO_LTP,   marker='o', c='purple', markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')
ln5 = ax2.plot(Date, BrO_LTO,   marker='o', c='black',  markersize = 2.0, linestyle='none', label='BrO LTcol Observations')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5,5)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#ax2.set_ylim(-0.3,0.3)
ax2.set_ylim(-1.9,1.9)

# Plot the axis labels, legend and title
plt.title('V2 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('PCs', fontsize=10)
ax2.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)

# Plot the legend
lns = ln1+ln2+ln3+ln4+ln5
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, bbox_to_anchor=(1.15, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 6 (V3 2018-19)
ax = plt.subplot(gs[2,2:3])
ax2 = ax.twinx()

# Plot PC1, PC2 & PC3
ln1 = ax.plot(Date, PC1,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='PC1 (R: '+str("%6.4f"%(rval_SPC1_V3_18))+')')  # PC1
ln2 = ax.plot(Date, PC2,   marker='o', c='b',      markersize = 2.0, linestyle='none', label='PC2 (R: '+str("%6.4f"%(rval_SPC2_V3_18))+')')  # PC2
ln3 = ax.plot(Date, PC3,   marker='o', c='g',      markersize = 2.0, linestyle='none', label='PC3 (R: '+str("%6.4f"%(rval_SPC3_V3_18))+')')  # PC3
# Plot BrO
ln4 = ax2.plot(Date, BrO_LTP,   marker='o', c='purple', markersize = 2.0, linestyle='none', label='BrO LTcol Predictions')
ln5 = ax2.plot(Date, BrO_LTO,   marker='o', c='black',  markersize = 2.0, linestyle='none', label='BrO LTcol Observations')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5,5)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#ax2.set_ylim(-0.3,0.3)
ax2.set_ylim(-1.9,1.9)

# Plot the axis labels, legend and title
plt.title('V3 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('PCs', fontsize=10)
ax2.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
ax.set_xlabel('Date', fontsize=15)

# Plot the legend
lns = ln1+ln2+ln3+ln4+ln5
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, bbox_to_anchor=(1.15, 1.0), loc=2, borderaxespad=0.)

#------------------------------------------------------------------------------
# PLOT THE GRAPH (BrOsurf All PCs)

fig = plt.figure()

gs = gridspec.GridSpec(nrows=3,
                       ncols=4, 
                       figure=fig, 
                       width_ratios= [0.05,0.5,0.5,0.05],
                       height_ratios=[0.25, 0.25, 0.25],
                       hspace=0.3, wspace=0.35)

#-------------------------------------
# Graph 1 (V1 2017-18)
ax = plt.subplot(gs[0,1:2])
ax2 = ax.twinx()

# Plot PC1, PC2 & PC3
ln1 = ax.plot(Date, PC1,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='PC1 (R: '+str("%6.4f"%(rval_SPC1_V1_17))+')')  # PC1
ln2 = ax.plot(Date, PC2,   marker='o', c='b',      markersize = 2.0, linestyle='none', label='PC2 (R: '+str("%6.4f"%(rval_SPC2_V1_17))+')')  # PC2
ln3 = ax.plot(Date, PC3,   marker='o', c='g',      markersize = 2.0, linestyle='none', label='PC3 (R: '+str("%6.4f"%(rval_SPC3_V1_17))+')')  # PC3
# Plot BrO
ln4 = ax2.plot(Date, BrO_SP,   marker='o', c='purple', markersize = 2.0, linestyle='none', label='BrO Surface Predictions')
ln5 = ax2.plot(Date, BrO_SO,   marker='o', c='black',  markersize = 2.0, linestyle='none', label='BrO Surface Observations')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5,5)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#ax2.set_ylim(-0.3,0.3)
ax2.set_ylim(-1.1,1.1)

# Plot the axis labels, legend and title
plt.title('V1 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('PCs', fontsize=10)
#ax2.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)

# Plot the legend
lns = ln1+ln2+ln3+ln4+ln5
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, bbox_to_anchor=(-0.55, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 2 (V2 2017-18)
ax = plt.subplot(gs[1,1:2])
ax2 = ax.twinx()

# Plot PC1, PC2 & PC3
ln1 = ax.plot(Date, PC1,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='PC1 (R: '+str("%6.4f"%(rval_SPC1_V2_17))+')')  # PC1
ln2 = ax.plot(Date, PC2,   marker='o', c='b',      markersize = 2.0, linestyle='none', label='PC2 (R: '+str("%6.4f"%(rval_SPC2_V2_17))+')')  # PC2
ln3 = ax.plot(Date, PC3,   marker='o', c='g',      markersize = 2.0, linestyle='none', label='PC3 (R: '+str("%6.4f"%(rval_SPC3_V2_17))+')')  # PC3
# Plot BrO
ln4 = ax2.plot(Date, BrO_SP,   marker='o', c='purple', markersize = 2.0, linestyle='none', label='BrO Surface Predictions')
ln5 = ax2.plot(Date, BrO_SO,   marker='o', c='black',  markersize = 2.0, linestyle='none', label='BrO Surface Observations')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5,5)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#ax2.set_ylim(-0.3,0.3)
ax2.set_ylim(-1.1,1.1)

# Plot the axis labels, legend and title
plt.title('V2 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('PCs', fontsize=10)
#ax2.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)

# Plot the legend
lns = ln1+ln2+ln3+ln4+ln5
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, bbox_to_anchor=(-0.55, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 3 (V3 2017-18)
ax = plt.subplot(gs[2,1:2])
ax2 = ax.twinx()

# Plot PC1, PC2 & PC3
ln1 = ax.plot(Date, PC1,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='PC1 (R: '+str("%6.4f"%(rval_SPC1_V3_17))+')')  # PC1
ln2 = ax.plot(Date, PC2,   marker='o', c='b',      markersize = 2.0, linestyle='none', label='PC2 (R: '+str("%6.4f"%(rval_SPC2_V3_17))+')')  # PC2
ln3 = ax.plot(Date, PC3,   marker='o', c='g',      markersize = 2.0, linestyle='none', label='PC3 (R: '+str("%6.4f"%(rval_SPC3_V3_17))+')')  # PC3
# Plot BrO
ln4 = ax2.plot(Date, BrO_SP,   marker='o', c='purple', markersize = 2.0, linestyle='none', label='BrO Surface Predictions')
ln3 = ax2.plot(Date, BrO_SO,   marker='o', c='black',  markersize = 2.0, linestyle='none', label='BrO Surface Observations')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5,5)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax2.set_ylim(-1.1,1.1)

# Plot the axis labels, legend and title
plt.title('V3 (2017-18)', fontsize=15, y=1.05)
ax.set_ylabel('PCs', fontsize=10)
#ax2.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
ax.set_xlabel('Date', fontsize=15)

# Plot the legend
lns = ln1+ln2+ln3+ln4+ln5
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, bbox_to_anchor=(-0.55, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 4 (V1 2018-19)
ax = plt.subplot(gs[0,2:3])
ax2 = ax.twinx()

# Plot PC1, PC2 & PC3
ln1 = ax.plot(Date, PC1,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='PC1 (R: '+str("%6.4f"%(rval_SPC1_V1_18))+')')  # PC1
ln2 = ax.plot(Date, PC2,   marker='o', c='b',      markersize = 2.0, linestyle='none', label='PC2 (R: '+str("%6.4f"%(rval_SPC2_V1_18))+')')  # PC2
ln3 = ax.plot(Date, PC3,   marker='o', c='g',      markersize = 2.0, linestyle='none', label='PC3 (R: '+str("%6.4f"%(rval_SPC3_V1_18))+')')  # PC3
# Plot BrO
ln4 = ax2.plot(Date, BrO_SP,   marker='o', c='purple', markersize = 2.0, linestyle='none', label='BrO Surface Predictions')
ln5 = ax2.plot(Date, BrO_SO,   marker='o', c='black',  markersize = 2.0, linestyle='none', label='BrO Surface Observations')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5,5)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax2.set_ylim(-1.1,1.1)

# Plot the axis labels and title
plt.title('V1 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('PCs', fontsize=10)
ax2.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)

# Plot the legend
lns = ln1+ln2+ln3+ln4+ln5
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, bbox_to_anchor=(1.15, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 5 (V2 2018-19)
ax = plt.subplot(gs[1,2:3])
ax2 = ax.twinx()

# Plot PC1, PC2 & PC3
ln1 = ax.plot(Date, PC1,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='PC1 (R: '+str("%6.4f"%(rval_SPC1_V2_18))+')')  # PC1
ln2 = ax.plot(Date, PC2,   marker='o', c='b',      markersize = 2.0, linestyle='none', label='PC2 (R: '+str("%6.4f"%(rval_SPC2_V2_18))+')')  # PC2
ln3 = ax.plot(Date, PC3,   marker='o', c='g',      markersize = 2.0, linestyle='none', label='PC3 (R: '+str("%6.4f"%(rval_SPC3_V2_18))+')')  # PC3
# Plot BrO
ln4 = ax2.plot(Date, BrO_SP,   marker='o', c='purple', markersize = 2.0, linestyle='none', label='BrO Surface Predictions')
ln5 = ax2.plot(Date, BrO_SO,   marker='o', c='black',  markersize = 2.0, linestyle='none', label='BrO Surface Observations')

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
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5,5)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax2.set_ylim(-1.1,1.1)

# Plot the axis labels, legend and title
plt.title('V2 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('PCs', fontsize=10)
ax2.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
#ax.set_xlabel('Date', fontsize=15)

# Plot the legend
lns = ln1+ln2+ln3+ln4+ln5
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, bbox_to_anchor=(1.15, 1.0), loc=2, borderaxespad=0.)

#-------------------------------------
# Graph 6 (V3 2018-19)
ax = plt.subplot(gs[2,2:3])
ax2 = ax.twinx()

# Plot PC1, PC2 & PC3
ln1 = ax.plot(Date, PC1,   marker='o', c='r',      markersize = 2.0, linestyle='none', label='PC1 (R: '+str("%6.4f"%(rval_SPC1_V3_18))+')')  # PC1
ln2 = ax.plot(Date, PC2,   marker='o', c='b',      markersize = 2.0, linestyle='none', label='PC2 (R: '+str("%6.4f"%(rval_SPC2_V3_18))+')')  # PC2
ln3 = ax.plot(Date, PC3,   marker='o', c='g',      markersize = 2.0, linestyle='none', label='PC3 (R: '+str("%6.4f"%(rval_SPC3_V3_18))+')')  # PC3
# Plot BrO
ln4 = ax2.plot(Date, BrO_SP,   marker='o', c='purple', markersize = 2.0, linestyle='none', label='BrO Surface Predictions')
ln5 = ax2.plot(Date, BrO_SO,   marker='o', c='black',  markersize = 2.0, linestyle='none', label='BrO Surface Observations')

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
ax.tick_params(axis='x', pad=15)

# Format y-axis 1
ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5,5)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax2.set_ylim(-1.1,1.1)

# Plot the axis labels, legend and title
plt.title('V3 (2018-19)', fontsize=15, y=1.05)
#ax.set_ylabel('PCs', fontsize=10)
ax2.set_ylabel('BrO$_s$$_u$$_r$$_f$ (10$^1$$^3$ molec/cm$^2$)', fontsize=10)
ax.set_xlabel('Date', fontsize=15)

# Plot the legend
lns = ln1+ln2+ln3+ln4+ln5
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, bbox_to_anchor=(1.15, 1.0), loc=2, borderaxespad=0.)

