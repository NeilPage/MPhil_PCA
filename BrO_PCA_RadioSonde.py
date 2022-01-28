#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:51:06 2020

@author: ncp532
"""
# File system packages
#from netCDF4 import Dataset				    # function used to open single netcdf file
#from netCDF4 import MFDataset				# function used to open multiple netcdf files
#import xarray as xr

# Drawing packages
import matplotlib.pyplot as plt             
import matplotlib.dates as mdates            
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# Data handing packages
import numpy as np                           
import pandas as pd
from scipy import signal, stats
from statsmodels.formula.api import ols      # For statistics. Requires statsmodels 5.0 or more
from statsmodels.stats.anova import anova_lm # Analysis of Variance (ANOVA) on linear models
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

# Date and Time handling package
from datetime import datetime,timedelta		# functions to handle date and time

#------------------------------------------------------------------------------
# DEFINE THE DATASETS

#--------------
# MAX-DOAS
#--------------

# BrO
BrO_V1_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V1_17/all_BrO/V1_17_BrO_retrieval.csv',index_col=0)       # BrO V1 (2017/18)
BrO_V2_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V2_17/all_BrO/V2_17_BrO_retrieval.csv',index_col=0)       # BrO V2 (2017/18)
BrO_V3_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V3_17/all_BrO/V3_17_BrO_retrieval.csv',index_col=0)       # BrO V3 (2017/18)

BrO_V1_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V1_18/all_BrO/V1_18_BrO_retrieval.csv',index_col=0)       # BrO V1 (2018/19)
BrO_V2_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V2_18/all_BrO/V2_18_BrO_retrieval.csv',index_col=0)       # BrO V2 (2018/19)
BrO_V3_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V3_18/all_BrO/V3_18_BrO_retrieval.csv',index_col=0)       # BrO V3 (2018/19)

BrO_SIPEXII = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/SIPEXII/all_BrO/SIPEXII_BrO_retrieval.csv',index_col=0) # BrO SIPEXII (2012)

# AEC
AEC_V1_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V1_17/all_Aerosol/V1_17_AeroExt_338.csv',index_col=0) # AEC at 338nm V1 (2017/18)
AEC_V2_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V2_17/all_Aerosol/V2_17_AeroExt_338.csv',index_col=0) # AEC at 338nm V2 (2017/18)
AEC_V3_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V3_17/all_Aerosol/V3_17_AeroExt_338.csv',index_col=0) # AEC at 338nm V3 (2017/18)

AEC_V1_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V1_18/all_Aerosol/V1_18_AeroExt_338.csv',index_col=0) # AEC at 338nm V1 (2018/19)
AEC_V2_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V2_18/all_Aerosol/V2_18_AeroExt_338.csv',index_col=0) # AEC at 338nm V2 (2018/19)
AEC_V3_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V3_18/all_Aerosol/V3_18_AeroExt_338.csv',index_col=0) # AEC at 338nm V3 (2018/19)

AEC_SIPEXII = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/SIPEXII/all_Aerosol/SIPEXII_AeroExt_338.csv',index_col=0) # AEC at 338nm SIPEXII (2012)

# NO2
# NO2_V1_17 = pd.read_csv('/Users/ncp532/Documents/data/.csv, index_col=0) # NO2 dSCD at 2 degrees
# NO2_V2_17 = pd.read_csv('/Users/ncp532/Documents/data/.csv, index_col=0) # NO2 dSCD at 2 degrees
# NO2_V3_17 = pd.read_csv('/Users/ncp532/Documents/data/.csv, index_col=0) # NO2 dSCD at 2 degrees

# NO2_V1_18 = pd.read_csv('/Users/ncp532/Documents/data/.csv, index_col=0) # NO2 dSCD at 2 degrees
# NO2_V2_18 = pd.read_csv('/Users/ncp532/Documents/data/.csv, index_col=0) # NO2 dSCD at 2 degrees
# NO2_V3_18 = pd.read_csv('/Users/ncp532/Documents/data/.csv, index_col=0) # NO2 dSCD at 2 degrees

# NO2_SIPEXII = pd.read_csv('/Users/ncp532/Documents/data/.csv, index_col=0) # NO2 dSCD at 2 degrees
                         
#--------------
# Met & O3
#--------------

# Met
Met_V1_17  = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2017_18/ShipTrack/V1_17_underway_60.csv', index_col=0)
Met_V2_17  = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2017_18/ShipTrack/V2_17_underway_60.csv', index_col=0)
Met_V3_17M = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2017_18/ShipTrack/V3_17_underway_60.csv', index_col=0)
Met_V3_17D = Met_V3_17M

Met_V1_18  = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2018_19/ShipTrack/V1_18_underway_60.csv', index_col=0) 
Met_V2_18  = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2018_19/ShipTrack/V2_18_underway_60.csv', index_col=0)
Met_V3_18M = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2018_19/ShipTrack/V3_18_underway_60.csv', index_col=0) 
Met_V3_18D = Met_V3_18M

Met_SIPEXII = pd.read_csv('/Users/ncp532/Documents/Data/SIPEXII_2012/ShipTrack/SIPEXII_underway_60.csv', index_col=0) 

# O3
O3_V1_17   = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2017_18/ARM/V1_O3_1min.csv', index_col=0)
O3_V2_17   = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2017_18/ARM/V2_O3_1min.csv', index_col=0)
O3_V3_17M  = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2017_18/ARM/V3_O3_1min.csv', index_col=0)
O3_V3_17D  = O3_V3_17M

O3_V1_18   = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2018_19/O3/V1_O3_1min.csv', index_col=0)
O3_V1_18.rename(columns={'O3':'O3_(ppb)'},inplace=True) # rename the column from 'O3' to 'O3_(ppb)'
O3_V2_18   = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2018_19/O3/V2_O3_1min.csv', index_col=0)
O3_V2_18.rename(columns={'O3':'O3_(ppb)'},inplace=True) # rename the column from 'O3' to 'O3_(ppb)'
O3_V2_18   = O3_V2_18.loc[~O3_V2_18.index.duplicated(keep='first')] # remove duplicate values from the .csv file
O3_V3_18   = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2018_19/O3/V3_O3_1min.csv', index_col=0)
O3_V3_18.rename(columns={'O3':'O3_(ppb)'},inplace=True) # rename the column from 'O3' to 'O3_(ppb)'
O3_V3_18M  = O3_V3_18.loc[~O3_V3_18.index.duplicated(keep='first')] # remove duplicate values from the .csv file
O3_V3_18D  = O3_V3_18M

O3_SIPEXII = pd.read_csv('/Users/ncp532/Documents/Data/SIPEXII_2012/SIPEXII_O3/SIPEXII_O3_QAQC.csv', index_col=0)
O3_SIPEXII = O3_SIPEXII.loc[~O3_SIPEXII.index.duplicated(keep='first')] # remove duplicate values from the .csv file

#--------------
# MERRA-2 (.CSV)
#--------------

MERRA2_V1_17   = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V1_17_MERRA2.csv',   index_col=0)
MERRA2_V2_17   = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V2_17_MERRA2.csv',   index_col=0)
MERRA2_V3_17M  = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V3_17M_MERRA2.csv',  index_col=0)
MERRA2_V3_17D  = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V3_17D_MERRA2.csv',  index_col=0)

MERRA2_V1_18   = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V1_18_MERRA2.csv',   index_col=0) 
MERRA2_V2_18   = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V2_18_MERRA2.csv',   index_col=0)
MERRA2_V3_18M  = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V3_18M_MERRA2.csv',  index_col=0) 
MERRA2_V3_18D  = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V3_18D_MERRA2.csv',  index_col=0)

MERRA2_SIPEXII = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/SIPEXII_MERRA2.csv', index_col=0) 

#--------------
# Radiosonde
#--------------
RS_V1_17      = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/CAMMPCAN_2017_Radiosonde.csv',           index_col=0)
RS_V2_17      = RS_V1_17
RS_V3_17M     = RS_V1_17 
RS_V3_17D     = RS_V1_17 

RS_V1_17_1hr  = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/CAMMPCAN_2017_Radiosonde_Interp1hr.csv', index_col=0)
RS_V2_17_1hr  = RS_V1_17_1hr
RS_V3_17M_1hr = RS_V1_17_1hr
RS_V3_17D_1hr = RS_V1_17_1hr

#------------------------------------------------------------------------------
# FILTER THE BrO DATA FOR RELATIVE ERROR 

# Calculate the Relative Error (>=0.6)
Filter1 = BrO_V1_17['err_surf_vmr'] / BrO_V1_17['surf_vmr(ppmv)']
Filter2 = BrO_V2_17['err_surf_vmr'] / BrO_V2_17['surf_vmr(ppmv)']
Filter3 = BrO_V3_17['err_surf_vmr'] / BrO_V3_17['surf_vmr(ppmv)']

Filter4 = BrO_V1_18['err_surf_vmr'] / BrO_V1_18['surf_vmr(ppmv)']
Filter5 = BrO_V2_18['err_surf_vmr'] / BrO_V2_18['surf_vmr(ppmv)']
Filter6 = BrO_V3_18['err_surf_vmr'] / BrO_V3_18['surf_vmr(ppmv)']

Filter7 = BrO_SIPEXII['err_surf_vmr'] / BrO_SIPEXII['surf_vmr(ppmv)']

# Apply the filter
V1_17F       = Filter1 < 0.6
BrO_V1_17T   = BrO_V1_17[V1_17F]

V2_17F       = Filter2 < 0.6
BrO_V2_17T   = BrO_V2_17[V2_17F]

V3_17F       = Filter3 < 0.6
BrO_V3_17T   = BrO_V3_17[V3_17F]

V1_18F       = Filter4 < 0.6
BrO_V1_18T   = BrO_V1_18[V1_18F]

V2_18F       = Filter5 < 0.6
BrO_V2_18T   = BrO_V2_18[V2_18F]

V3_18F       = Filter6 < 0.6
BrO_V3_18T   = BrO_V3_18[V3_18F]

SIPEXIIF     = Filter7 < 0.6
BrO_SIPEXIIT = BrO_SIPEXII[SIPEXIIF]

#------------------------------------------------------------------------------
# TRANSPOSE THE MAX-DOAS DATAFRAMES

# BrO
BrO_V1_17T   = BrO_V1_17T
BrO_V2_17T   = BrO_V2_17T
BrO_V3_17MT  = BrO_V3_17T
BrO_V3_17DT  = BrO_V3_17T

BrO_V1_18T   = BrO_V1_18T
BrO_V2_18T   = BrO_V2_18T
BrO_V3_18MT  = BrO_V3_18T
BrO_V3_18DT  = BrO_V3_18T

BrO_SIPEXIIT = BrO_SIPEXII

# AEC
AEC_V1_17   = AEC_V1_17.T
AEC_V2_17   = AEC_V2_17.T
AEC_V3_17M  = AEC_V3_17.T
AEC_V3_17D  = AEC_V3_17.T

AEC_V1_18   = AEC_V1_18.T
AEC_V2_18   = AEC_V2_18.T
AEC_V3_18M  = AEC_V3_18.T
AEC_V3_18D  = AEC_V3_18.T

AEC_SIPEXII = AEC_SIPEXII.T

# NO2
#NO2_V1_17   = NO2_V1_17.T
#NO2_V2_17   = NO2_V2_17.T
#NO2_V3_17M  = NO2_V3_17.T
#NO2_V3_17D  = NO2_V3_17.T
#
#NO2_V1_18   = NO2_V1_18.T
#NO2_V2_18   = NO2_V2_18.T
#NO2_V3_18M  = NO2_V3_18.T
#NO2_V3_18D  = NO2_V3_18.T
#
#NO2_SIPEXII = NO2_SIPEXII.T

#------------------------------------------------------------------------------
# Set the date

#--------------
# MAX-DOAS
#--------------

# BrO
BrO_V1_17T.index   = (pd.to_datetime(BrO_V1_17T.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
BrO_V2_17T.index   = (pd.to_datetime(BrO_V2_17T.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
BrO_V3_17MT.index  = (pd.to_datetime(BrO_V3_17MT.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
BrO_V3_17DT.index  = (pd.to_datetime(BrO_V3_17DT.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

BrO_V1_18T.index   = (pd.to_datetime(BrO_V1_18T.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
BrO_V2_18T.index   = (pd.to_datetime(BrO_V2_18T.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
BrO_V3_18MT.index  = (pd.to_datetime(BrO_V3_18MT.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
BrO_V3_18DT.index  = (pd.to_datetime(BrO_V3_18DT.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

BrO_SIPEXIIT.index = (pd.to_datetime(BrO_SIPEXIIT.index, dayfirst=True) + timedelta(hours=8)) # SIPEXII timezone is UT+8

# AEC
AEC_V1_17.index   = (pd.to_datetime(AEC_V1_17.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
AEC_V2_17.index   = (pd.to_datetime(AEC_V2_17.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
AEC_V3_17M.index  = (pd.to_datetime(AEC_V3_17M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
AEC_V3_17D.index  = (pd.to_datetime(AEC_V3_17D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

AEC_V1_18.index   = (pd.to_datetime(AEC_V1_18.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
AEC_V2_18.index   = (pd.to_datetime(AEC_V2_18.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
AEC_V3_18M.index  = (pd.to_datetime(AEC_V3_18M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
AEC_V3_18D.index  = (pd.to_datetime(AEC_V3_18D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

AEC_SIPEXII.index = (pd.to_datetime(AEC_SIPEXII.index, dayfirst=True) + timedelta(hours=8)) # SIPEXII timezone is UT+8

## NO2
#NO2_V1_17.index   = (pd.to_datetime(NO2_V1_17.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
#NO2_V2_17.index   = (pd.to_datetime(NO2_V2_17.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
#NO2_V3_17M.index  = (pd.to_datetime(NO2_V3_17M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
#NO2_V3_17D.index  = (pd.to_datetime(NO2_V3_17D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

#NO2_V1_18.index   = (pd.to_datetime(NO2_V1_18.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
#NO2_V2_18.index   = (pd.to_datetime(NO2_V2_18.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
#NO2_V3_18M.index  = (pd.to_datetime(NO2_V3_18M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
#NO2_V3_18D.index  = (pd.to_datetime(NO2_V3_18D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

#NO2_SIPEXII.index = (pd.to_datetime(NO2_SIPEXII.index, dayfirst=True) + timedelta(hours=8)) # SIPEXII timezone is UT+8

#--------------
# Met & O3
#--------------

# Met
Met_V1_17.index   = (pd.to_datetime(Met_V1_17.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
Met_V2_17.index   = (pd.to_datetime(Met_V2_17.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
Met_V3_17M.index  = (pd.to_datetime(Met_V3_17M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
Met_V3_17D.index  = (pd.to_datetime(Met_V3_17D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

Met_V1_18.index   = (pd.to_datetime(Met_V1_18.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
Met_V2_18.index   = (pd.to_datetime(Met_V2_18.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
Met_V3_18M.index  = (pd.to_datetime(Met_V3_18M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
Met_V3_18D.index  = (pd.to_datetime(Met_V3_18D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

Met_SIPEXII.index = (pd.to_datetime(Met_SIPEXII.index, dayfirst=True) + timedelta(hours=8)) # SIPEXII timezone is UT+8

# O3
O3_V1_17.index   = (pd.to_datetime(O3_V1_17.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
O3_V2_17.index   = (pd.to_datetime(O3_V2_17.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
O3_V3_17M.index  = (pd.to_datetime(O3_V3_17M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
O3_V3_17D.index  = (pd.to_datetime(O3_V3_17D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

O3_V1_18.index   = (pd.to_datetime(O3_V1_18.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
O3_V2_18.index   = (pd.to_datetime(O3_V2_18.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
O3_V3_18M.index  = (pd.to_datetime(O3_V3_18M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
O3_V3_18D.index  = (pd.to_datetime(O3_V3_18D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

O3_SIPEXII.index = (pd.to_datetime(O3_SIPEXII.index, dayfirst=True) + timedelta(hours=8)) # SIPEXII timezone is UT+8

#--------------
# MERRA-2
#--------------

MERRA2_V1_17.index   = (pd.to_datetime(MERRA2_V1_17.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
MERRA2_V2_17.index   = (pd.to_datetime(MERRA2_V2_17.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
MERRA2_V3_17M.index  = (pd.to_datetime(MERRA2_V3_17M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
MERRA2_V3_17D.index  = (pd.to_datetime(MERRA2_V3_17D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

MERRA2_V1_18.index   = (pd.to_datetime(MERRA2_V1_18.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
MERRA2_V2_18.index   = (pd.to_datetime(MERRA2_V2_18.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
MERRA2_V3_18M.index  = (pd.to_datetime(MERRA2_V3_18M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
MERRA2_V3_18D.index  = (pd.to_datetime(MERRA2_V3_18D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

MERRA2_SIPEXII.index = (pd.to_datetime(MERRA2_SIPEXII.index, dayfirst=True) + timedelta(hours=8)) # SIPEXII timezone is UT+8

#--------------
# Radiosonde
#--------------
RS_V1_17.index      = (pd.to_datetime(RS_V1_17.index,      dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
RS_V2_17.index      = (pd.to_datetime(RS_V2_17.index,      dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
RS_V3_17M.index     = (pd.to_datetime(RS_V3_17M.index,     dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
RS_V3_17D.index     = (pd.to_datetime(RS_V3_17D.index,     dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

RS_V1_17_1hr.index  = (pd.to_datetime(RS_V1_17_1hr.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
RS_V2_17_1hr.index  = (pd.to_datetime(RS_V2_17_1hr.index,  dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
RS_V3_17M_1hr.index = (pd.to_datetime(RS_V3_17M_1hr.index, dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
RS_V3_17D_1hr.index = (pd.to_datetime(RS_V3_17D_1hr.index, dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

#------------------------------------------------------------------------------
# REPLACE ERRONEOUS VALUES WITH NAN

#--------------
# MAX-DOAS
#--------------

# BrO
BrO_V1_17T   = BrO_V1_17T.replace(-9999.000000, np.nan)
BrO_V2_17T   = BrO_V2_17T.replace(-9999.000000, np.nan)
BrO_V3_17MT  = BrO_V3_17MT.replace(-9999.000000, np.nan)
BrO_V3_17DT  = BrO_V3_17DT.replace(-9999.000000, np.nan)

BrO_V1_18T   = BrO_V1_18T.replace(-9999.000000, np.nan)
BrO_V2_18T   = BrO_V2_18T.replace(-9999.000000, np.nan)
BrO_V3_18MT  = BrO_V3_18MT.replace(-9999.000000, np.nan)
BrO_V3_18DT  = BrO_V3_18DT.replace(-9999.000000, np.nan)

BrO_SIPEXIIT = BrO_SIPEXIIT.replace(9.67e-05,np.nan)
BrO_SIPEXIIT = BrO_SIPEXIIT.replace(7.67e-06,np.nan)
BrO_SIPEXIIT = BrO_SIPEXIIT.replace(7.67e-07,np.nan)
BrO_SIPEXIIT.loc[BrO_SIPEXIIT.isnull().any(axis=1), :] = np.nan # if any element in the row is nan, set the whole row to nan
BrO_SIPEXIIT = BrO_SIPEXIIT.replace(-9999.000000, np.nan)

# AEC
AEC_V1_17   = AEC_V1_17.replace(-9999.000000, np.nan)
AEC_V2_17   = AEC_V2_17.replace(-9999.000000, np.nan)
AEC_V3_17M  = AEC_V3_17M.replace(-9999.000000, np.nan)
AEC_V3_17D  = AEC_V3_17D.replace(-9999.000000, np.nan)

AEC_V1_18   = AEC_V1_18.replace(-9999.000000, np.nan)
AEC_V2_18   = AEC_V2_18.replace(-9999.000000, np.nan)
AEC_V3_18M  = AEC_V3_18M.replace(-9999.000000, np.nan)
AEC_V3_18D  = AEC_V3_18D.replace(-9999.000000, np.nan)

AEC_SIPEXII = AEC_SIPEXII.replace(-9999.000000, np.nan)

## NO2
#NO2_V1_17    = NO2_V1_17.replace(-9999.000000, np.nan)
#NO2_V2_17    = NO2_V2_17.replace(-9999.000000, np.nan)
#NO2_V3_17M   = NO2_V3_17M.replace(-9999.000000, np.nan)
#NO2_V3_17D   = NO2_V3_17D.replace(-9999.000000, np.nan)
#
#NO2_V1_18    = NO2_V1_18.replace(-9999.000000, np.nan)
#NO2_V2_18    = NO2_V2_18.replace(-9999.000000, np.nan)
#NO2_V3_18M   = NO2_V3_18M.replace(-9999.000000, np.nan)
#NO2_V3_18D   = NO2_V3_18D.replace(-9999.000000, np.nan)
#
#NO2_SIPEXII  = NO2_SIPEXII.replace(-9999.000000, np.nan)

#------------------------------------------------------------------------------
# CONVERT THE MAX-DOAS, MET & O3 DATASETS A 1-HOUR TIME RESOLUTION

#--------------
# MAX-DOAS
#--------------

# BrO
BrO_V1_17T   = BrO_V1_17T.resample('60T').mean()
BrO_V2_17T   = BrO_V2_17T.resample('60T').mean()
BrO_V3_17MT  = BrO_V3_17MT.resample('60T').mean()
BrO_V3_17DT  = BrO_V3_17DT.resample('60T').mean()

BrO_V1_18T   = BrO_V1_18T.resample('60T').mean()
BrO_V2_18T   = BrO_V2_18T.resample('60T').mean()
BrO_V3_18MT  = BrO_V3_18MT.resample('60T').mean()
BrO_V3_18DT  = BrO_V3_18DT.resample('60T').mean()

BrO_SIPEXIIT = BrO_SIPEXIIT.resample('60T').mean()

# AEC
AEC_V1_17   = AEC_V1_17.resample('60T').mean()
AEC_V2_17   = AEC_V2_17.resample('60T').mean()
AEC_V3_17M  = AEC_V3_17M.resample('60T').mean()
AEC_V3_17D  = AEC_V3_17D.resample('60T').mean()

AEC_V1_18   = AEC_V1_18.resample('60T').mean()
AEC_V2_18   = AEC_V2_18.resample('60T').mean()
AEC_V3_18M  = AEC_V3_18M.resample('60T').mean()
AEC_V3_18D  = AEC_V3_18D.resample('60T').mean()

AEC_SIPEXII = AEC_SIPEXII.resample('60T').mean()

# NO2
#NO2_V1_17   = NO2_V1_17.resample('60T').mean()
#NO2_V2_17   = NO2_V2_17.resample('60T').mean()
#NO2_V3_17M  = NO2_V3_17M.resample('60T').mean()
#NO2_V3_17D  = NO2_V3_17D.resample('60T').mean()

#NO2_V1_18   = NO2_V1_18.resample('60T').mean()
#NO2_V2_18   = NO2_V2_18.resample('60T').mean()
#NO2_V3_18M  = NO2_V3_18M.resample('60T').mean()
#NO2_V3_18D  = NO2_V3_18D.resample('60T').mean()

#NO2_SIPEXII = NO2_SIPEXII.resample('60T').mean()

#--------------
# Met
#--------------
Met_V1_17   = Met_V1_17.resample('60T').mean()
Met_V2_17   = Met_V2_17.resample('60T').mean()
Met_V3_17M  = Met_V3_17M.resample('60T').mean()
Met_V3_17D  = Met_V3_17D.resample('60T').mean()

Met_V1_18   = Met_V1_18.resample('60T').mean()
Met_V2_18   = Met_V2_18.resample('60T').mean()
Met_V3_18M  = Met_V3_18M.resample('60T').mean()
Met_V3_18D  = Met_V3_18D.resample('60T').mean()

Met_SIPEXII = Met_SIPEXII.resample('60T').mean()

#--------------
# O3
#--------------
O3_V1_17   = O3_V1_17.resample('60T').mean()
O3_V2_17   = O3_V2_17.resample('60T').mean()
O3_V3_17M  = O3_V3_17M.resample('60T').mean()
O3_V3_17D  = O3_V3_17D.resample('60T').mean()

O3_V1_18   = O3_V1_18.resample('60T').mean()
O3_V2_18   = O3_V2_18.resample('60T').mean()
O3_V3_18M  = O3_V3_18M.resample('60T').mean()
O3_V3_18D  = O3_V3_18D.resample('60T').mean()

O3_SIPEXII = O3_SIPEXII.resample('60T').mean()

#------------------------------------------------------------------------------
#  CALCULATE THE CHANGE IN SEA LEVEL PRESSURE FROM ONE HOUR TO THE NEXT (P1hr)

#-----------------------------------
# Sea level pressure at time t (hPa)
#-----------------------------------
# MERRA2 - SLP
PtM2a_V1_17   = np.array(MERRA2_V1_17['SLP'][1:1003])/100
PtM2a_V2_17   = np.array(MERRA2_V2_17['SLP'][1:812])/100
PtM2a_V3_17M  = np.array(MERRA2_V3_17M['SLP'][1:1052])/100
PtM2a_V3_17D  = np.array(MERRA2_V3_17D['SLP'][1:1052])/100

PtM2a_V1_18   = np.array(MERRA2_V1_18['SLP'][1:644])/100
PtM2a_V2_18   = np.array(MERRA2_V2_18['SLP'][1:980])/100
PtM2a_V3_18M  = np.array(MERRA2_V3_18D['SLP'][1:1244])/100
PtM2a_V3_18D  = np.array(MERRA2_V3_18M['SLP'][1:1244])/100

PtM2a_SIPEXII = np.array(MERRA2_SIPEXII['SLP'][1:2180])/100

# MERRA2 - SurfPres
PtM2b_V1_17   = np.array(MERRA2_V1_17['SurfPres'][1:1003])/100
PtM2b_V2_17   = np.array(MERRA2_V2_17['SurfPres'][1:812])/100
PtM2b_V3_17M  = np.array(MERRA2_V3_17M['SurfPres'][1:1052])/100
PtM2b_V3_17D  = np.array(MERRA2_V3_17D['SurfPres'][1:1052])/100

PtM2b_V1_18   = np.array(MERRA2_V1_18['SurfPres'][1:644])/100
PtM2b_V2_18   = np.array(MERRA2_V2_18['SurfPres'][1:980])/100
PtM2b_V3_18M  = np.array(MERRA2_V3_18D['SurfPres'][1:1244])/100
PtM2b_V3_18D  = np.array(MERRA2_V3_18M['SurfPres'][1:1244])/100

PtM2b_SIPEXII = np.array(MERRA2_SIPEXII['SurfPres'][1:2180])/100

# MET
PtMet_V1_17   = np.array(Met_V1_17['atm_press_hpa'][1:864])
PtMet_V2_17   = np.array(Met_V2_17['atm_press_hpa'][1:720])
PtMet_V3_17M  = np.array(Met_V3_17M['atm_press_hpa'][1:1156])
PtMet_V3_17D  = np.array(Met_V3_17D['atm_press_hpa'][1:1156])

PtMet_V1_18   = np.array(Met_V1_18['atm_press_hpa'][1:816])
PtMet_V2_18   = np.array(Met_V2_18['atm_press_hpa'][1:836])
PtMet_V3_18M  = np.array(Met_V3_18D['atm_press_hpa'][1:1153])
PtMet_V3_18D  = np.array(Met_V3_18M['atm_press_hpa'][1:1153])

PtMet_SIPEXII = np.array(Met_SIPEXII['atm_press_hpa'][1:1526])

#-----------------------------------
# Sea level pressure at time t-1 (hPa)
#-----------------------------------
# MERRA2 - SLP
Pt1M2a_V1_17   = np.array(MERRA2_V1_17['SLP'][0:1002])/100
Pt1M2a_V2_17   = np.array(MERRA2_V2_17['SLP'][0:811])/100
Pt1M2a_V3_17M  = np.array(MERRA2_V3_17M['SLP'][0:1051])/100
Pt1M2a_V3_17D  = np.array(MERRA2_V3_17D['SLP'][0:1051])/100

Pt1M2a_V1_18   = np.array(MERRA2_V1_18['SLP'][0:643])/100
Pt1M2a_V2_18   = np.array(MERRA2_V2_18['SLP'][0:979])/100
Pt1M2a_V3_18M  = np.array(MERRA2_V3_18M['SLP'][0:1243])/100
Pt1M2a_V3_18D  = np.array(MERRA2_V3_18D['SLP'][0:1243])/100

Pt1M2a_SIPEXII = np.array(MERRA2_SIPEXII['SLP'][0:2179])/100

# MERRA2 - SurfPres
Pt1M2b_V1_17   = np.array(MERRA2_V1_17['SurfPres'][0:1002])/100
Pt1M2b_V2_17   = np.array(MERRA2_V2_17['SurfPres'][0:811])/100
Pt1M2b_V3_17M  = np.array(MERRA2_V3_17M['SurfPres'][0:1051])/100
Pt1M2b_V3_17D  = np.array(MERRA2_V3_17D['SurfPres'][0:1051])/100

Pt1M2b_V1_18   = np.array(MERRA2_V1_18['SurfPres'][0:643])/100
Pt1M2b_V2_18   = np.array(MERRA2_V2_18['SurfPres'][0:979])/100
Pt1M2b_V3_18M  = np.array(MERRA2_V3_18M['SurfPres'][0:1243])/100
Pt1M2b_V3_18D  = np.array(MERRA2_V3_18D['SurfPres'][0:1243])/100

Pt1M2b_SIPEXII = np.array(MERRA2_SIPEXII['SurfPres'][0:2179])/100

# MET
Pt1Met_V1_17   = np.array(Met_V1_17['atm_press_hpa'][0:863])
Pt1Met_V2_17   = np.array(Met_V2_17['atm_press_hpa'][0:719])
Pt1Met_V3_17M  = np.array(Met_V3_17M['atm_press_hpa'][0:1155])
Pt1Met_V3_17D  = np.array(Met_V3_17D['atm_press_hpa'][0:1155])

Pt1Met_V1_18   = np.array(Met_V1_18['atm_press_hpa'][0:815])
Pt1Met_V2_18   = np.array(Met_V2_18['atm_press_hpa'][0:835])
Pt1Met_V3_18M  = np.array(Met_V3_18M['atm_press_hpa'][0:1152])
Pt1Met_V3_18D  = np.array(Met_V3_18D['atm_press_hpa'][0:1152])

Pt1Met_SIPEXII = np.array(Met_SIPEXII['atm_press_hpa'][0:1525])

#-----------------------------------
# Change in pressure from one hour to next (hPa)
#-----------------------------------
# MERRA2 - SLP
P1hrM2a_V1_17   = PtM2a_V1_17   - Pt1M2a_V1_17
P1hrM2a_V2_17   = PtM2a_V2_17   - Pt1M2a_V2_17
P1hrM2a_V3_17M  = PtM2a_V3_17M  - Pt1M2a_V3_17M
P1hrM2a_V3_17D  = PtM2a_V3_17D  - Pt1M2a_V3_17D

P1hrM2a_V1_18   = PtM2a_V1_18   - Pt1M2a_V1_18
P1hrM2a_V2_18   = PtM2a_V2_18   - Pt1M2a_V2_18
P1hrM2a_V3_18M  = PtM2a_V3_18M  - Pt1M2a_V3_18M
P1hrM2a_V3_18D  = PtM2a_V3_18D  - Pt1M2a_V3_18D

P1hrM2a_SIPEXII = PtM2a_SIPEXII - Pt1M2a_SIPEXII

# MERRA2 -SurfPres
P1hrM2b_V1_17   = PtM2b_V1_17   - Pt1M2b_V1_17
P1hrM2b_V2_17   = PtM2b_V2_17   - Pt1M2b_V2_17
P1hrM2b_V3_17M  = PtM2b_V3_17M  - Pt1M2b_V3_17M
P1hrM2b_V3_17D  = PtM2b_V3_17D  - Pt1M2b_V3_17D

P1hrM2b_V1_18   = PtM2b_V1_18   - Pt1M2b_V1_18
P1hrM2b_V2_18   = PtM2b_V2_18   - Pt1M2b_V2_18
P1hrM2b_V3_18M  = PtM2b_V3_18M  - Pt1M2b_V3_18M
P1hrM2b_V3_18D  = PtM2b_V3_18D  - Pt1M2b_V3_18D

P1hrM2b_SIPEXII = PtM2b_SIPEXII - Pt1M2b_SIPEXII

# MET
P1hrMet_V1_17   = PtMet_V1_17   - Pt1Met_V1_17
P1hrMet_V2_17   = PtMet_V2_17   - Pt1Met_V2_17
P1hrMet_V3_17M  = PtMet_V3_17M  - Pt1Met_V3_17M
P1hrMet_V3_17D  = PtMet_V3_17D  - Pt1Met_V3_17D

P1hrMet_V1_18   = PtMet_V1_18   - Pt1Met_V1_18
P1hrMet_V2_18   = PtMet_V2_18   - Pt1Met_V2_18
P1hrMet_V3_18M  = PtMet_V3_18M  - Pt1Met_V3_18M
P1hrMet_V3_18D  = PtMet_V3_18D  - Pt1Met_V3_18D

P1hrMet_SIPEXII = PtMet_SIPEXII - Pt1Met_SIPEXII

#-----------------------------------
# Append nan to begining of each array
#-----------------------------------
# MERRA2 - SLP
P1hrM2a_V1_17   = np.append(np.nan,P1hrM2a_V1_17)
P1hrM2a_V2_17   = np.append(np.nan,P1hrM2a_V2_17)
P1hrM2a_V3_17M  = np.append(np.nan,P1hrM2a_V3_17M)
P1hrM2a_V3_17D  = np.append(np.nan,P1hrM2a_V3_17D)

P1hrM2a_V1_18   = np.append(np.nan,P1hrM2a_V1_18)
P1hrM2a_V2_18   = np.append(np.nan,P1hrM2a_V2_18)
P1hrM2a_V3_18M  = np.append(np.nan,P1hrM2a_V3_18M)
P1hrM2a_V3_18D  = np.append(np.nan,P1hrM2a_V3_18D)

P1hrM2a_SIPEXII = np.append(np.nan,P1hrM2a_SIPEXII)

# MERRA2 - SurfPres
P1hrM2b_V1_17   = np.append(np.nan,P1hrM2b_V1_17)
P1hrM2b_V2_17   = np.append(np.nan,P1hrM2b_V2_17)
P1hrM2b_V3_17M  = np.append(np.nan,P1hrM2b_V3_17M)
P1hrM2b_V3_17D  = np.append(np.nan,P1hrM2b_V3_17D)

P1hrM2b_V1_18   = np.append(np.nan,P1hrM2b_V1_18)
P1hrM2b_V2_18   = np.append(np.nan,P1hrM2b_V2_18)
P1hrM2b_V3_18M  = np.append(np.nan,P1hrM2b_V3_18M)
P1hrM2b_V3_18D  = np.append(np.nan,P1hrM2b_V3_18D)

P1hrM2b_SIPEXII = np.append(np.nan,P1hrM2b_SIPEXII)

# MET
P1hrMet_V1_17   = np.append(np.nan,P1hrMet_V1_17)
P1hrMet_V2_17   = np.append(np.nan,P1hrMet_V2_17)
P1hrMet_V3_17M  = np.append(np.nan,P1hrMet_V3_17M)
P1hrMet_V3_17D  = np.append(np.nan,P1hrMet_V3_17D)

P1hrMet_V1_18   = np.append(np.nan,P1hrMet_V1_18)
P1hrMet_V2_18   = np.append(np.nan,P1hrMet_V2_18)
P1hrMet_V3_18M  = np.append(np.nan,P1hrMet_V3_18M)
P1hrMet_V3_18D  = np.append(np.nan,P1hrMet_V3_18D)

P1hrMet_SIPEXII = np.append(np.nan,P1hrMet_SIPEXII)

#-----------------------------------
# Add P1hr to the Pandas dataframes
#-----------------------------------
# MERRA2 - SLP
MERRA2_V1_17['P1hrM2a']   = P1hrM2a_V1_17
MERRA2_V2_17['P1hrM2a']   = P1hrM2a_V2_17
MERRA2_V3_17M['P1hrM2a']  = P1hrM2a_V3_17M
MERRA2_V3_17D['P1hrM2a']  = P1hrM2a_V3_17D

MERRA2_V1_18['P1hrM2a']   = P1hrM2a_V1_18
MERRA2_V2_18['P1hrM2a']   = P1hrM2a_V2_18
MERRA2_V3_18M['P1hrM2a']  = P1hrM2a_V3_18M
MERRA2_V3_18D['P1hrM2a']  = P1hrM2a_V3_18D

MERRA2_SIPEXII['P1hrM2a'] = P1hrM2a_SIPEXII

# MERRA2 - SurfPres
MERRA2_V1_17['P1hrM2b']   = P1hrM2b_V1_17
MERRA2_V2_17['P1hrM2b']   = P1hrM2b_V2_17
MERRA2_V3_17M['P1hrM2b']  = P1hrM2b_V3_17M
MERRA2_V3_17D['P1hrM2b']  = P1hrM2b_V3_17D

MERRA2_V1_18['P1hrM2b']   = P1hrM2b_V1_18
MERRA2_V2_18['P1hrM2b']   = P1hrM2b_V2_18
MERRA2_V3_18M['P1hrM2b']  = P1hrM2b_V3_18M
MERRA2_V3_18D['P1hrM2b']  = P1hrM2b_V3_18D

MERRA2_SIPEXII['P1hrM2b'] = P1hrM2b_SIPEXII

# MET
Met_V1_17['P1hrMet']   = P1hrMet_V1_17
Met_V2_17['P1hrMet']   = P1hrMet_V2_17
Met_V3_17M['P1hrMet']  = P1hrMet_V3_17M
Met_V3_17D['P1hrMet']  = P1hrMet_V3_17D

Met_V1_18['P1hrMet']   = P1hrMet_V1_18
Met_V2_18['P1hrMet']   = P1hrMet_V2_18
Met_V3_18M['P1hrMet']  = P1hrMet_V3_18M
Met_V3_18D['P1hrMet']  = P1hrMet_V3_18D

Met_SIPEXII['P1hrMet'] = P1hrMet_SIPEXII

#------------------------------------------------------------------------------
#  CALCULATE THE POTENTIAL TEMPERATURE DIFFERENTIAL IN LOWEST 100m (k)

# MERRA-2
MERRA2_V1_17['PTDif100m']   = MERRA2_V1_17['VPT100m']   - MERRA2_V1_17['VPT2m']
MERRA2_V2_17['PTDif100m']   = MERRA2_V2_17['VPT100m']   - MERRA2_V2_17['VPT2m']
MERRA2_V3_17M['PTDif100m']  = MERRA2_V3_17M['VPT100m']  - MERRA2_V3_17M['VPT2m']
MERRA2_V3_17D['PTDif100m']  = MERRA2_V3_17D['VPT100m']  - MERRA2_V3_17D['VPT2m']

MERRA2_V1_18['PTDif100m']   = MERRA2_V1_18['VPT100m']   - MERRA2_V1_18['VPT2m']
MERRA2_V2_18['PTDif100m']   = MERRA2_V2_18['VPT100m']   - MERRA2_V2_18['VPT2m']
MERRA2_V3_18M['PTDif100m']  = MERRA2_V3_18M['VPT100m']  - MERRA2_V3_18M['VPT2m']
MERRA2_V3_18D['PTDif100m']  = MERRA2_V3_18D['VPT100m']  - MERRA2_V3_18D['VPT2m']

MERRA2_SIPEXII['PTDif100m'] = MERRA2_SIPEXII['VPT100m'] - MERRA2_SIPEXII['VPT2m']

# Radiosonde
RS_V1_17['PTDif100m']   = RS_V1_17['VPT100m']   - RS_V1_17['VPT17m']
RS_V2_17['PTDif100m']   = RS_V2_17['VPT100m']   - RS_V2_17['VPT17m']
RS_V3_17M['PTDif100m']  = RS_V3_17M['VPT100m']  - RS_V3_17M['VPT17m']
RS_V3_17D['PTDif100m']  = RS_V3_17D['VPT100m']  - RS_V3_17D['VPT17m']

RS_V1_17_1hr['PTDif100m']   = RS_V1_17_1hr['VPT100m']   - RS_V1_17_1hr['VPT17m']
RS_V2_17_1hr['PTDif100m']   = RS_V2_17_1hr['VPT100m']   - RS_V2_17_1hr['VPT17m']
RS_V3_17M_1hr['PTDif100m']  = RS_V3_17M_1hr['VPT100m']  - RS_V3_17M_1hr['VPT17m']
RS_V3_17D_1hr['PTDif100m']  = RS_V3_17D_1hr['VPT100m']  - RS_V3_17D_1hr['VPT17m']

#------------------------------------------------------------------------------
#  CALCULATE THE POTENTIAL TEMPERATURE DIFFERENTIAL IN LOWEST 1000m (k)

# MERRA-2
MERRA2_V1_17['PTDif1000m']   = MERRA2_V1_17['VPT1000m']   - MERRA2_V1_17['VPT2m']
MERRA2_V2_17['PTDif1000m']   = MERRA2_V2_17['VPT1000m']   - MERRA2_V2_17['VPT2m']
MERRA2_V3_17M['PTDif1000m']  = MERRA2_V3_17M['VPT1000m']  - MERRA2_V3_17M['VPT2m']
MERRA2_V3_17D['PTDif1000m']  = MERRA2_V3_17D['VPT1000m']  - MERRA2_V3_17D['VPT2m']

MERRA2_V1_18['PTDif1000m']   = MERRA2_V1_18['VPT1000m']   - MERRA2_V1_18['VPT2m']
MERRA2_V2_18['PTDif1000m']   = MERRA2_V2_18['VPT1000m']   - MERRA2_V2_18['VPT2m']
MERRA2_V3_18M['PTDif1000m']  = MERRA2_V3_18M['VPT1000m']  - MERRA2_V3_18M['VPT2m']
MERRA2_V3_18D['PTDif1000m']  = MERRA2_V3_18D['VPT1000m']  - MERRA2_V3_18D['VPT2m']

MERRA2_SIPEXII['PTDif1000m'] = MERRA2_SIPEXII['VPT1000m'] - MERRA2_SIPEXII['VPT2m']

# Radiosonde
RS_V1_17['PTDif1000m']   = RS_V1_17['VPT1000m']   - RS_V1_17['VPT17m']
RS_V2_17['PTDif1000m']   = RS_V2_17['VPT1000m']   - RS_V2_17['VPT17m']
RS_V3_17M['PTDif1000m']  = RS_V3_17M['VPT1000m']  - RS_V3_17M['VPT17m']
RS_V3_17D['PTDif1000m']  = RS_V3_17D['VPT1000m']  - RS_V3_17D['VPT17m']

RS_V1_17_1hr['PTDif1000m']   = RS_V1_17_1hr['VPT1000m']   - RS_V1_17_1hr['VPT17m']
RS_V2_17_1hr['PTDif1000m']   = RS_V2_17_1hr['VPT1000m']   - RS_V2_17_1hr['VPT17m']
RS_V3_17M_1hr['PTDif1000m']  = RS_V3_17M_1hr['VPT1000m']  - RS_V3_17M_1hr['VPT17m']
RS_V3_17D_1hr['PTDif1000m']  = RS_V3_17D_1hr['VPT1000m']  - RS_V3_17D_1hr['VPT17m']

#------------------------------------------------------------------------------
# Filter the datasets based on the date

#-----------------------------
# V1_17 Davis (14-22 Nov 2017)
#-----------------------------
start_date   = '2017-11-14'
end_date     = '2017-11-23'
# BrO
Davis        = (BrO_V1_17T.index >= start_date) & (BrO_V1_17T.index < end_date)
V1_17_BrO    = BrO_V1_17T[Davis]
# AEC
Davis        = (AEC_V1_17.index >= start_date) & (AEC_V1_17.index < end_date)
V1_17_AEC    = AEC_V1_17[Davis]
# Met
Davis        = (Met_V1_17.index >= start_date) & (Met_V1_17.index < end_date)
V1_17_Met    = Met_V1_17[Davis]
# O3
Davis        = (O3_V1_17.index >= start_date) & (O3_V1_17.index < end_date)
V1_17_O3     = O3_V1_17[Davis]
# MERRA2
Davis        = (MERRA2_V1_17.index >= start_date) & (MERRA2_V1_17.index < end_date)
V1_17_MERRA2 = MERRA2_V1_17[Davis]
# Radiosonde
Davis        = (RS_V1_17.index >= start_date) & (RS_V1_17.index < end_date)
V1_17_RS     = RS_V1_17[Davis]
# Radiosonde (1hr)
Davis        = (RS_V1_17_1hr.index >= start_date) & (RS_V1_17_1hr.index < end_date)
V1_17_RS_1hr = RS_V1_17_1hr[Davis]

#-----------------------------
# V2_17 Casey (21-22 Dec 2017 and 26 Dec 2017 - 5 Jan 2018)
#-----------------------------
start_date1 = '2017-12-21'
end_date1 = '2017-12-23'
start_date2 = '2017-12-26'
end_date2 = '2018-01-6'
# BrO
Casey1       = (BrO_V2_17T.index >= start_date1) & (BrO_V2_17T.index < end_date1)
Casey2       = (BrO_V2_17T.index >= start_date2) & (BrO_V2_17T.index < end_date2)
V2_17_BrO1   = BrO_V2_17T[Casey1]
V2_17_BrO2   = BrO_V2_17T[Casey2]
V2_17_BrO    = pd.concat([V2_17_BrO1,V2_17_BrO2], axis =0)

# AEC
Casey1       = (AEC_V2_17.index >= start_date1) & (AEC_V2_17.index < end_date1)
Casey2       = (AEC_V2_17.index >= start_date2) & (AEC_V2_17.index < end_date2)
V2_17_AEC1   = AEC_V2_17[Casey1]
V2_17_AEC2   = AEC_V2_17[Casey2]
V2_17_AEC    = pd.concat([V2_17_AEC1,V2_17_AEC2], axis =0)
# Met
Casey1       = (Met_V2_17.index >= start_date1) & (Met_V2_17.index < end_date1)
Casey2       = (Met_V2_17.index >= start_date2) & (Met_V2_17.index < end_date2)
V2_17_Met1   = Met_V2_17[Casey1]
V2_17_Met2   = Met_V2_17[Casey2]
V2_17_Met    = pd.concat([V2_17_Met1,V2_17_Met2], axis =0)
# O3
Casey1       = (O3_V2_17.index >= start_date1) & (O3_V2_17.index < end_date1)
Casey2       = (O3_V2_17.index >= start_date2) & (O3_V2_17.index < end_date2)
V2_17_O31    = O3_V2_17[Casey1]
V2_17_O32    = O3_V2_17[Casey2]
V2_17_O3     = pd.concat([V2_17_O31,V2_17_O32], axis =0)
# MERRA2
Casey1       = (MERRA2_V2_17.index >= start_date1) & (MERRA2_V2_17.index < end_date1)
Casey2       = (MERRA2_V2_17.index >= start_date2) & (MERRA2_V2_17.index < end_date2)
V2_17_MERRA21= MERRA2_V2_17[Casey1]
V2_17_MERRA22= MERRA2_V2_17[Casey2]
V2_17_MERRA2 = pd.concat([V2_17_MERRA21,V2_17_MERRA22], axis =0)
# Radiosonde
Casey1       = (RS_V2_17.index >= start_date1) & (RS_V2_17.index < end_date1)
Casey2       = (RS_V2_17.index >= start_date2) & (RS_V2_17.index < end_date2)
V2_17_RS1    = RS_V2_17[Casey1]
V2_17_RS2    = RS_V2_17[Casey2]
V2_17_RS     = pd.concat([V2_17_RS1,V2_17_RS2], axis =0)
# Radiosonde (1hr)
Casey1       = (RS_V2_17_1hr.index >= start_date1) & (RS_V2_17_1hr.index < end_date1)
Casey2       = (RS_V2_17_1hr.index >= start_date2) & (RS_V2_17_1hr.index < end_date2)
V2_17_RS_1hr1= RS_V2_17_1hr[Casey1]
V2_17_RS_1hr2= RS_V2_17_1hr[Casey2]
V2_17_RS_1hr = pd.concat([V2_17_RS_1hr1,V2_17_RS_1hr2], axis =0)

#-----------------------------
# V3_17 Mawson (1-17 Feb 2018)
#-----------------------------
start_date    = '2018-02-01'
end_date      = '2018-02-18'
# BrO
Mawson        = (BrO_V3_17MT.index >= start_date) & (BrO_V3_17MT.index < end_date)
V3_17_BrOM    = BrO_V3_17MT[Mawson]
# AEC
Mawson        = (AEC_V3_17M.index >= start_date) & (AEC_V3_17M.index < end_date)
V3_17_AECM    = AEC_V3_17M[Mawson]
# Met
Mawson        = (Met_V3_17M.index >= start_date) & (Met_V3_17M.index < end_date)
V3_17_MetM    = Met_V3_17M[Mawson]
# O3
Mawson        = (O3_V3_17M.index >= start_date) & (O3_V3_17M.index < end_date)
V3_17_O3M     = O3_V3_17M[Mawson]
# MERRA2
Mawson        = (MERRA2_V3_17M.index >= start_date) & (MERRA2_V3_17M.index < end_date)
V3_17_MERRA2M = MERRA2_V3_17M[Mawson]
# Radiosonde
Mawson        = (RS_V3_17M.index >= start_date) & (RS_V3_17M.index < end_date)
V3_17_RSM     = RS_V3_17M[Mawson]
# Radiosonde (1hr)
Mawson        = (RS_V3_17M_1hr.index >= start_date) & (RS_V3_17M_1hr.index < end_date)
V3_17_RS_1hrM = RS_V3_17M_1hr[Mawson]

#-----------------------------
# V3_17 Davis (27-30 Jan 2018 and 19-21 Feb 2018)
#-----------------------------
start_date1   = '2018-01-27'
end_date1     = '2018-01-31'
start_date2   = '2018-02-19'
end_date2     = '2018-02-22'
# BrO
Davis1        = (BrO_V3_17DT.index >= start_date1) & (BrO_V3_17DT.index < end_date1)
Davis2        = (BrO_V3_17DT.index >= start_date2) & (BrO_V3_17DT.index < end_date2)
V3_17_BrO1    = BrO_V3_17DT[Davis1]
V3_17_BrO2    = BrO_V3_17DT[Davis2]
V3_17_BrOD    = pd.concat([V3_17_BrO1,V3_17_BrO2], axis =0)
# AEC
Davis1        = (AEC_V3_17D.index >= start_date1) & (AEC_V3_17D.index < end_date1)
Davis2        = (AEC_V3_17D.index >= start_date2) & (AEC_V3_17D.index < end_date2)
V3_17_AEC1    = AEC_V3_17D[Davis1]
V3_17_AEC2    = AEC_V3_17D[Davis2]
V3_17_AECD    = pd.concat([V3_17_AEC1,V3_17_AEC2], axis =0)
# Met
Davis1        = (Met_V3_17D.index >= start_date1) & (Met_V3_17D.index < end_date1)
Davis2        = (Met_V3_17D.index >= start_date2) & (Met_V3_17D.index < end_date2)
V3_17_Met1    = Met_V3_17D[Davis1]
V3_17_Met2    = Met_V3_17D[Davis2]
V3_17_MetD    = pd.concat([V3_17_Met1,V3_17_Met2], axis =0)
# O3
Davis1        = (O3_V3_17D.index >= start_date1) & (O3_V3_17D.index < end_date1)
Davis2        = (O3_V3_17D.index >= start_date2) & (O3_V3_17D.index < end_date2)
V3_17_O31     = O3_V3_17D[Davis1]
V3_17_O32     = O3_V3_17D[Davis2]
V3_17_O3D     = pd.concat([V3_17_O31,V3_17_O32], axis =0)
# MERRA2
Davis1        = (MERRA2_V3_17D.index >= start_date1) & (MERRA2_V3_17D.index < end_date1)
Davis2        = (MERRA2_V3_17D.index >= start_date2) & (MERRA2_V3_17D.index < end_date2)
V3_17_MERRA21 = MERRA2_V3_17D[Davis1]
V3_17_MERRA22 = MERRA2_V3_17D[Davis2]
V3_17_MERRA2D = pd.concat([V3_17_MERRA21,V3_17_MERRA22], axis =0)
# Radiosonde
Davis1        = (RS_V3_17D.index >= start_date1) & (RS_V3_17D.index < end_date1)
Davis2        = (RS_V3_17D.index >= start_date2) & (RS_V3_17D.index < end_date2)
V3_17_RS1     = RS_V3_17D[Davis1]
V3_17_RS2     = RS_V3_17D[Davis2]
V3_17_RSD     = pd.concat([V3_17_RS1,V3_17_RS2], axis =0)
# Radiosonde (1hr)
Davis1        = (RS_V3_17D_1hr.index >= start_date1) & (RS_V3_17D_1hr.index < end_date1)
Davis2        = (RS_V3_17D_1hr.index >= start_date2) & (RS_V3_17D_1hr.index < end_date2)
V3_17_RS_1hr1 = RS_V3_17D_1hr[Davis1]
V3_17_RS_1hr2 = RS_V3_17D_1hr[Davis2]
V3_17_RS_1hrD = pd.concat([V3_17_RS_1hr1,V3_17_RS_1hr2], axis =0)

#-----------------------------
# V1_18 Davis (7-15 Nov 2018)
#-----------------------------
start_date   = '2018-11-07'
end_date     = '2018-11-16'
# BrO
Davis        = (BrO_V1_18T.index >= start_date) & (BrO_V1_18T.index < end_date)
V1_18_BrO    = BrO_V1_18T[Davis]
# AEC
Davis        = (AEC_V1_18.index >= start_date) & (AEC_V1_18.index < end_date)
V1_18_AEC    = AEC_V1_18[Davis]
# Met
Davis        = (Met_V1_18.index >= start_date) & (Met_V1_18.index < end_date)
V1_18_Met    = Met_V1_18[Davis]
# O3
Davis        = (O3_V1_18.index >= start_date) & (O3_V1_18.index < end_date)
V1_18_O3     = O3_V1_18[Davis]
# MERRA2
Davis        = (MERRA2_V1_18.index >= start_date) & (MERRA2_V1_18.index < end_date)
V1_18_MERRA2 = MERRA2_V1_18[Davis]

#-----------------------------
# V2_18 Casey (15-30 Dec 2018)
#-----------------------------
start_date   = '2018-12-15'
end_date     = '2018-12-31'
# BrO
Casey        = (BrO_V2_18T.index >= start_date) & (BrO_V2_18T.index < end_date)
V2_18_BrO    = BrO_V2_18T[Casey]
# AEC
Casey        = (AEC_V2_18.index >= start_date) & (AEC_V2_18.index < end_date)
V2_18_AEC    = AEC_V2_18[Casey]
# Met
Casey        = (Met_V2_18.index >= start_date) & (Met_V2_18.index < end_date)
V2_18_Met    = Met_V2_18[Casey]
# O3
Casey        = (O3_V2_18.index >= start_date) & (O3_V2_18.index < end_date)
V2_18_O3     = O3_V2_18[Casey]
# MERRA2
Casey        = (MERRA2_V2_18.index >= start_date) & (MERRA2_V2_18.index < end_date)
V2_18_MERRA2 = MERRA2_V2_18[Casey]

#-----------------------------
# V3_18 Mawson (30 Jan - 9 Feb 2019)
#-----------------------------
start_date    = '2019-01-30'
end_date      = '2019-02-10'
# BrO
Mawson        = (BrO_V3_18MT.index >= start_date) & (BrO_V3_18MT.index < end_date)
V3_18_BrOM    = BrO_V3_18MT[Mawson]
# AEC
Mawson        = (AEC_V3_18M.index >= start_date) & (AEC_V3_18M.index < end_date)
V3_18_AECM    = AEC_V3_18M[Mawson]
# Met
Mawson        = (Met_V3_18M.index >= start_date) & (Met_V3_18M.index < end_date)
V3_18_MetM    = Met_V3_18M[Mawson]
# O3
Mawson        = (O3_V3_18M.index >= start_date) & (O3_V3_18M.index < end_date)
V3_18_O3M     = O3_V3_18M[Mawson]
# MERRA2
Mawson        = (MERRA2_V3_18M.index >= start_date) & (MERRA2_V3_18M.index < end_date)
V3_18_MERRA2M = MERRA2_V3_18M[Mawson]

#-----------------------------
# V3_18 Davis (26-28 Jan 2019 and 19-20 Feb 2019)
#-----------------------------
start_date1   = '2019-01-26'
end_date1     = '2019-01-29'
start_date2   = '2019-02-19'
end_date2     = '2019-02-21'
# BrO
Davis1        = (BrO_V3_18DT.index >= start_date1) & (BrO_V3_18DT.index < end_date1)
Davis2        = (BrO_V3_18DT.index >= start_date2) & (BrO_V3_18DT.index < end_date2)
V3_18_BrO1    = BrO_V3_18DT[Davis1]
V3_18_BrO2    = BrO_V3_18DT[Davis2]
V3_18_BrOD    = pd.concat([V3_18_BrO1,V3_18_BrO2], axis =0)
# AEC
Davis1        = (AEC_V3_18D.index >= start_date1) & (AEC_V3_18D.index < end_date1)
Davis2        = (AEC_V3_18D.index >= start_date2) & (AEC_V3_18D.index < end_date2)
V3_18_AEC1    = AEC_V3_18D[Davis1]
V3_18_AEC2    = AEC_V3_18D[Davis2]
V3_18_AECD    = pd.concat([V3_18_AEC1,V3_18_AEC2], axis =0)
# Met
Davis1        = (Met_V3_18D.index >= start_date1) & (Met_V3_18D.index < end_date1)
Davis2        = (Met_V3_18D.index >= start_date2) & (Met_V3_18D.index < end_date2)
V3_18_Met1    = Met_V3_18D[Davis1]
V3_18_Met2    = Met_V3_18D[Davis2]
V3_18_MetD    = pd.concat([V3_18_Met1,V3_18_Met2], axis =0)
# O3
Davis1        = (O3_V3_18D.index >= start_date1) & (O3_V3_18D.index < end_date1)
Davis2        = (O3_V3_18D.index >= start_date2) & (O3_V3_18D.index < end_date2)
V3_18_O31     = O3_V3_18D[Davis1]
V3_18_O32     = O3_V3_18D[Davis2]
V3_18_O3D     = pd.concat([V3_18_O31,V3_18_O32], axis =0)
# MERRA2
Davis1        = (MERRA2_V3_18D.index >= start_date1) & (MERRA2_V3_18D.index < end_date1)
Davis2        = (MERRA2_V3_18D.index >= start_date2) & (MERRA2_V3_18D.index < end_date2)
V3_18_MERRA21 = MERRA2_V3_18D[Davis1]
V3_18_MERRA22 = MERRA2_V3_18D[Davis2]
V3_18_MERRA2D = pd.concat([V3_18_MERRA21,V3_18_MERRA22], axis =0)

#-----------------------------
# SIPEXII (23 Sep to 11 Nov 2012)
#-----------------------------
start_date     = '2012-09-23'
end_date       = '2012-11-12'
# BrO
SIPEX          = (BrO_SIPEXIIT.index >= start_date) & (BrO_SIPEXIIT.index < end_date)
SIPEXII_BrO    = BrO_SIPEXIIT[SIPEX]
# AEC
SIPEX          = (AEC_SIPEXII.index >= start_date) & (AEC_SIPEXII.index < end_date)
SIPEXII_AEC    = AEC_SIPEXII[SIPEX]
# Met
SIPEX          = (Met_SIPEXII.index >= start_date) & (Met_SIPEXII.index < end_date)
SIPEXII_Met    = Met_SIPEXII[SIPEX]
# O3
SIPEX          = (O3_SIPEXII.index >= start_date) & (O3_SIPEXII.index < end_date)
SIPEXII_O3     = O3_SIPEXII[SIPEX]
# MERRA2
SIPEX          = (MERRA2_SIPEXII.index >= start_date) & (MERRA2_SIPEXII.index < end_date)
SIPEXII_MERRA2 = MERRA2_SIPEXII[SIPEX]

#------------------------------------------------------------------------------
# COMBINE THE DATAFRAMES FOR EACH VOYAGE INTO A SINGLE DATAFRAME

# BrO
BrO_All    = pd.concat([V1_17_BrO,V2_17_BrO,V3_17_BrOM,V3_17_BrOD],axis=0) # Radiosonde (2017 only)
#BrO_All    = pd.concat([V1_17_BrO,V2_17_BrO,V3_17_BrOM,V3_17_BrOD,V1_18_BrO,V2_18_BrO,V3_18_BrOM,V3_18_BrOD],axis=0) # Without SIPEXII
#BrO_All    = pd.concat([SIPEXII_BrO,V1_17_BrO,V2_17_BrO,V3_17_BrOM,V3_17_BrOD,V1_18_BrO,V2_18_BrO,V3_18_BrOM,V3_18_BrOD],axis=0) # With SIPEXII
#BrO_All    = SIPEXII_BrO # SIPEXII Only

# AEC
AEC_All    = pd.concat([V1_17_AEC,V2_17_AEC,V3_17_AECM,V3_17_AECD],axis=0) # # Radiosonde (2017 only)
#AEC_All    = pd.concat([V1_17_AEC,V2_17_AEC,V3_17_AECM,V3_17_AECD,V1_18_AEC,V2_18_AEC,V3_18_AECM,V3_18_AECD],axis=0) # Without SIPEXII
#AEC_All    = pd.concat([SIPEXII_AEC,V1_17_AEC,V2_17_AEC,V3_17_AECM,V3_17_AECD,V1_18_AEC,V2_18_AEC,V3_18_AECM,V3_18_AECD],axis=0) # With SIPEXII
#AEC_All    = SIPEXII_AEC # With SIPEXII

# NO2
#NO2_All    = pd.concat([V1_17_NO2,V2_17_NO2,V3_17_NO2M,V3_17_NO2D,V1_18_NO2C,V2_18_NO2,V3_18_NO2M,V3_18_NO2D],axis=0) # Without SIPEXII
#NO2_All    = pd.concat([SIPEXII_NO2,V1_17_NO2,V2_17_NO2,V3_17_NO2M,V3_17_NO2D,V1_18_NO2C,V2_18_NO2,V3_18_NO2M,V3_18_NO2D],axis=0) # With SIPEXII
#NO2_All    = SIPEXII_NO2 # With SIPEXII

# Met
Met_All    = pd.concat([V1_17_Met,V2_17_Met,V3_17_MetM,V3_17_MetD],axis=0) # # Radiosonde (2017 only)
#Met_All    = pd.concat([V1_17_Met,V2_17_Met,V3_17_MetM,V3_17_MetD,V1_18_Met,V2_18_Met,V3_18_MetM,V3_18_MetD],axis=0) # Without SIPEXII
#Met_All    = pd.concat([SIPEXII_Met,V1_17_Met,V2_17_Met,V3_17_MetM,V3_17_MetD,V1_18_Met,V2_18_Met,V3_18_MetM,V3_18_MetD],axis=0) # With SIPEXII
#Met_All    = SIPEXII_Met # With SIPEXII

# O3
O3_All     = pd.concat([V1_17_O3,V2_17_O3,V3_17_O3M,V3_17_O3D],axis=0) # # Radiosonde (2017 only)
#O3_All     = pd.concat([V1_17_O3,V2_17_O3,V3_17_O3M,V3_17_O3D,V1_18_O3,V2_18_O3,V3_18_O3M,V3_18_O3D],axis=0) # Without SIPEXII
#O3_All     = pd.concat([SIPEXII_O3,V1_17_O3,V2_17_O3,V3_17_O3M,V3_17_O3D,V1_18_O3,V2_18_O3,V3_18_O3M,V3_18_O3D],axis=0) # With SIPEXII
#O3_All     = SIPEXII_O3 # With SIPEXII

# MERRA2
MERRA2_All = pd.concat([V1_17_MERRA2,V2_17_MERRA2,V3_17_MERRA2M,V3_17_MERRA2D],axis=0) # # Radiosonde (2017 only)
#MERRA2_All = pd.concat([V1_17_MERRA2,V2_17_MERRA2,V3_17_MERRA2M,V3_17_MERRA2D,V1_18_MERRA2,V2_18_MERRA2,V3_18_MERRA2M,V3_18_MERRA2D],axis=0) # Without SIPEXII
#MERRA2_All = pd.concat([SIPEXII_MERRA2,V1_17_MERRA2,V2_17_MERRA2,V3_17_MERRA2M,V3_17_MERRA2D,V1_18_MERRA2,V2_18_MERRA2,V3_18_MERRA2M,V3_18_MERRA2D],axis=0) # With SIPEXII
#MERRA2_All = SIPEXII_MERRA2 # With SIPEXII

# Radiosonde
RS_All = pd.concat([V1_17_RS,V2_17_RS,V3_17_RSM,V3_17_RSD],axis=0)                 # Radiosonde (2017 only)
#RS_All = pd.concat([V1_17_RS_1hr,V2_17_RS_1hr,V3_17_RS_1hrM,V3_17_RS_1hrD],axis=0) # Radiosonde 1hr (2017 only)

#------------------------------------------------------------------------------
# DEFINE THE MAX-DOAS VARIABLES

#--------------
# MAX-DOAS
#--------------
# Surface BrO
BrO_Surf = BrO_All['surf_vmr(ppmv)'] # BrO surface mixing ratio (<200m)
BrO_Surf = BrO_Surf*1e6              # Convert to pptv

# Surface BrO Error
BrO_Surf_Err = BrO_All['err_surf_vmr'] # BrO surface mixing ratio error
BrO_Surf_Err = BrO_Surf_Err*1e6        # Convert to pptv

# LtCol BrO
BrO_LtCol   = BrO_All['BrO_VCD(molec/cm^2)'] # BrO VCD

# LtCol BrO Error
BrO_LtCol_Err   = BrO_All['err_BrO_VCD'] # BrO VCD error

# AEC
AEC        = AEC_All.iloc[:,0]       # Aerosol extinction coefficient (km-1)

# NO2
#NO2_2dSCD = # NO2 DCSD at 2 degrees

#------------------------------------------------------------------------------
# REMOVE NAN VALUES

#--------------
# MAX-DOAS
#--------------

BrO_Surf   = BrO_Surf.dropna()
BrO_LtCol  = BrO_LtCol.dropna()
AEC        = AEC.dropna()
#NO2_2dSCD = NO2_2dSCD.dropna()

#--------------
# O3
#--------------
O3_All = O3_All.dropna()

#------------------------------------------------------------------------------
# FILTER THE DATAFRAMES TO ONLY INCLUDE THE SAME DATES

# Find periods when BrO, O3 and Radiosonde are collocated
dfBrO_Surf  = pd.concat([BrO_Surf,O3_All],axis=1,join='inner')
dfBrO_LtCol = pd.concat([BrO_LtCol,O3_All],axis=1,join='inner')

dfBrO_Surf  = pd.concat([dfBrO_Surf,RS_All],axis=1,join='inner')
dfBrO_Surf.rename(columns={ dfBrO_Surf.columns[0]: 'BrO_(pptv)' }, inplace = True)

#dfBrO_LtCol.rename(columns={ dfBrO_LtCol.columns[0]: '0.1_(pptv)' }, inplace = True)
dfBrO_LtCol = pd.concat([dfBrO_LtCol,RS_All],axis=1,join='inner')

dfBrO_Surf  = pd.concat([dfBrO_Surf,AEC],axis=1,join='inner')
dfBrO_LtCol = pd.concat([dfBrO_LtCol,AEC],axis=1,join='inner')

dfBrO_Surf.rename(columns={ dfBrO_Surf.columns[48]: 'AEC' }, inplace = True)
dfBrO_LtCol.rename(columns={ dfBrO_LtCol.columns[48]: 'AEC' }, inplace = True)

dfBrO_Surf  = pd.concat([dfBrO_Surf,Met_All],axis=1,join='inner')
dfBrO_LtCol = pd.concat([dfBrO_LtCol,Met_All],axis=1,join='inner')

dfBrO_Surf  = pd.concat([dfBrO_Surf,BrO_Surf_Err],axis=1,join='inner')
dfBrO_LtCol = pd.concat([dfBrO_LtCol,BrO_LtCol_Err],axis=1,join='inner')

# Merge the Radiosonde, AEC and O3 dataframes
dfRS = pd.concat([RS_All,AEC],axis=1,join='inner')
dfRS = pd.concat([dfRS,O3_All],axis=1,join='inner')
dfRS.rename(columns={ dfRS.columns[46]: 'AEC' }, inplace = True)

#------------------------------------------------------------------------------
# CONVERT OBSERVED BrO FROM PPTV to MOLECULES/CM3

# variables
Avo     = 6.0221e23             # Avogadro number (molecules/mol)
MM      = 28.9644*1e-3          # Molar mass of dry air (kg/mol)
rho100m = dfBrO_Surf['rho100m'] # Mass density of dry air (kg/m3)

# Calculate the number density of air (nd)
nd100m = (Avo/MM)*rho100m # number density of air at 100m (molecules/m3)

#-----------
# BrO_Surf
#-----------

# Calculate BrO (molecules/m3)
dfBrO_Surf['BrO_(molecules/m3)'] = dfBrO_Surf['BrO_(pptv)']*nd100m/1e12
dfBrO_Surf['err_(molecules/m3)'] = dfBrO_Surf['err_surf_vmr']*nd100m/1e12

# Calculate BrO (molecules/m2)
dfBrO_Surf['BrO_(molecules/m2)'] = dfBrO_Surf['BrO_(molecules/m3)']*200 # multiply by height of level (200m)
dfBrO_Surf['err_(molecules/m2)'] = dfBrO_Surf['err_(molecules/m3)']*200 # multiply by height of level (200m)

# Calculate BrO (molecules/cm2)
dfBrO_Surf['BrO_(molecules/cm2)'] = dfBrO_Surf['BrO_(molecules/m2)']*1e-4 # convert from m2 to cm2
dfBrO_Surf['err_(molecules/cm2)'] = dfBrO_Surf['err_(molecules/m2)']*1e-4 # convert from m2 to cm2

#-----------
# BrO_LtCol
#-----------

dfBrO_LtCol['BrO_(molecules/cm2)'] = dfBrO_LtCol['BrO_VCD(molec/cm^2)']

#------------------------------------------------------------------------------
# RAINFALL SCREEN

# RainfallS  = dfBrO_Surf['Rainfall']  # Rainfall (kg/m2/s)
# RainfallLT = dfBrO_LtCol['Rainfall'] # Rainfall (kg/m2/s)

# # Apply the filter
# RainfallFS  = RainfallS <= 0  # Remove values when Rainfall present
# RainfallFLT = RainfallLT <= 0 # Remove values when Rainfall present
# dfBrO_Surf  = dfBrO_Surf[RainfallFS]
# dfBrO_LtCol = dfBrO_LtCol[RainfallFLT]

#------------------------------------------------------------------------------
# SEASONAL END DATE SCREEN

#BrO_1dSCD = dfBrO_Surf['Rainfall'] # BrO 1degree dSCD (molecules/cm2)
#
## Apply the filter
#SeasonF  = BrO_1dSCD > 5e13 # Remove values when BrO 1degree dSCD <5e13 molecules/cm2
#dfBrO_Surf = dfBrO_Surf[SeasonF]

#------------------------------------------------------------------------------
# OZONE SCREEN

O3S  = dfBrO_Surf['O3_(ppb)']  # Surface O3
O3LT = dfBrO_LtCol['O3_(ppb)'] # Surface O3

# Apply the filter
OzoneFS     = O3S > 2 # Remove values when O3 <2 ppb
OzoneFLT    = O3LT > 2 # Remove values when O3 <2 ppb
dfBrO_Surf  = dfBrO_Surf[OzoneFS]
dfBrO_LtCol = dfBrO_LtCol[OzoneFLT]

#------------------------------------------------------------------------------
# POLLUTION SCREEN

#NO2_2dSCD = dfBrO_Surf['Rainfall'] # NO2 2degree dSCD (molecules/cm2)
#
## Apply the filter
#PollutionF  = NO2_2dSCD <= 5e15 # Remove values when BrO 1degree dSCD <5e13 molecules/cm2
#dfBrO_Surf = dfBrO_Surf[PollutionF]

#------------------------------------------------------------------------------
# DEFINE THE VARIABLES

V1_17F       = Filter1 < 0.6
BrO_V1_17T   = BrO_V1_17[V1_17F]

SurfBrO      = dfBrO_Surf['BrO_(molecules/cm2)']      # Surface BrO (molecules/cm2)
LTBrO        = dfBrO_LtCol['BrO_(molecules/cm2)']     # Ltcol BrO (molecules/cm2)
SurfBrO_Err  = dfBrO_Surf['err_(molecules/cm2)']      # Surface BrO Error (molecules/cm2)
LTBrO_Err    = dfBrO_LtCol['err_BrO_VCD']             # Ltcol BrO Error (molecules/cm2)
AEC          = dfBrO_Surf['AEC']                      # Aerosol extinction coefficient (km-1)
O3           = dfBrO_Surf['O3_(ppb)']                 # Surface O3 (ppb)
Pres17m_RS   = dfBrO_Surf['Pres17m']                  # Radiosonde pressure 17m (hPa)
SurfPresMet  = dfBrO_Surf['atm_press_hpa']            # Met surface pressure (hPa)
Temp17m_RS   = dfBrO_Surf['Temp17m']-273.15           # Radiosonde temperature at 17m (C)
SurfTempMetP = dfBrO_Surf['temp_air_port_degc']       # Met temperature port side (C)
SurfTempMetS = dfBrO_Surf['temp_air_strbrd_degc']     # Met temperature strbrd side (C)
SurfTempMet  = (SurfTempMetP+SurfTempMetS)/2          # Met temperature average (port & strbrd side) (C)
P1hrMet      = dfBrO_Surf['P1hrMet']                  # Met change in pressure from one hour to next (hPa)
PTDif100m    = dfBrO_Surf['PTDif100m']                # Potential temperature differential lowest 100m (K)
PTDif1000m   = dfBrO_Surf['PTDif1000m']               # Potential temperature differential lowest 1000m (K)
WS17m_RS     = dfBrO_Surf['WS17m']                    # Radiosonde wind speed at 10m (Kg/m2/s)
WS10mMetP    = dfBrO_Surf['wnd_spd_port_corr_knot']* 0.514444444   # Convert wind speed port side from knots to m/s
WS10mMetS    = dfBrO_Surf['wnd_spd_strbrd_corr_knot']* 0.514444444 # Convert wind speed strbrd side from knots to m/s
WS10mMet     = (WS10mMetP+WS10mMetS)/2                # Met wind speed average (port & strbrd side) (m/s)
MLH          = dfBrO_Surf['MLH']*1000                 # Richardson MLH (m)

#------------------------------------------------------------------------------
# FIND WIND SPEED CERTAIN DATES

# 14 Nov 2017
start_date = '2017-11-14'
end_date   = '2017-11-15'
# BrO
filterWS1  = (WS10mMet.index >= start_date) & (WS10mMet.index < end_date)
WS_14Nov   = WS10mMet[filterWS1]

#------------------------------------------------------------------------------
# PERFORM A LOG TRANSFORMATION ON AEC & SQUARE-ROOT TRANSFORMATION ON BrO

AEC      = np.log(AEC)
SurfBrO  = np.sqrt(SurfBrO)
LTBrO    = np.sqrt(LTBrO)

#------------------------------------------------------------------------------
# CALCULATE THE MEAN

Mean_SurfBrO     = np.mean(SurfBrO)
Mean_LTBrO       = np.mean(LTBrO)
Mean_AEC         = np.mean(AEC)
Mean_O3          = np.mean(O3) 
Mean_Pres17m_RS  = np.mean(Pres17m_RS)
Mean_SurfPresMet = np.mean(SurfPresMet) 
Mean_Temp17m_RS  = np.mean(Temp17m_RS)
Mean_SurfTempMet = np.mean(SurfTempMet)
Mean_P1hrMet     = np.mean(P1hrMet) 
Mean_PTDif100m   = np.mean(PTDif100m) 
Mean_PTDif1000m  = np.mean(PTDif1000m) 
Mean_WS17m_RS    = np.mean(WS17m_RS)
Mean_WS10mMet    = np.mean(WS10mMet) 
Mean_MLH         = np.mean(MLH)

#------------------------------------------------------------------------------
# CALCULATE THE MEDIAN

Median_SurfBrO     = np.median(SurfBrO)
Median_LTBrO       = np.median(LTBrO)
Median_AEC         = np.median(AEC)
Median_O3          = np.median(O3) 
Median_Pres17m_RS  = np.median(Pres17m_RS)
Median_SurfPresMet = np.median(SurfPresMet) 
Median_Temp17m_RS  = np.median(Temp17m_RS)
Median_SurfTempMet = np.median(SurfTempMet)
Median_P1hrMet     = np.median(P1hrMet) 
Median_PTDif100m   = np.median(PTDif100m) 
Median_PTDif1000m  = np.median(PTDif1000m) 
Median_WS17m_RS    = np.median(WS17m_RS)
Median_WS10mMet    = np.median(WS10mMet) 
Median_MLH         = np.median(MLH)

#------------------------------------------------------------------------------
# CALCULATE THE MINIMUM

Min_SurfBrO     = np.min(SurfBrO)
Min_LTBrO       = np.min(LTBrO)
Min_AEC         = np.min(AEC)
Min_O3          = np.min(O3) 
Min_Pres17m_RS  = np.min(Pres17m_RS)
Min_SurfPresMet = np.min(SurfPresMet) 
Min_Temp17m_RS  = np.min(Temp17m_RS)
Min_SurfTempMet = np.min(SurfTempMet)
Min_P1hrMet     = np.min(P1hrMet) 
Min_PTDif100m   = np.min(PTDif100m) 
Min_PTDif1000m  = np.min(PTDif1000m) 
Min_WS17m_RS    = np.min(WS17m_RS)
Min_WS10mMet    = np.min(WS10mMet) 
Min_MLH         = np.min(MLH)

#------------------------------------------------------------------------------
# CALCULATE THE MAXIMUM

Max_SurfBrO     = np.max(SurfBrO)
Max_LTBrO       = np.max(LTBrO)
Max_AEC         = np.max(AEC)
Max_O3          = np.max(O3) 
Max_Pres17m_RS  = np.max(Pres17m_RS)
Max_SurfPresMet = np.max(SurfPresMet) 
Max_Temp17m_RS  = np.max(Temp17m_RS)
Max_SurfTempMet = np.max(SurfTempMet)
Max_P1hrMet     = np.max(P1hrMet) 
Max_PTDif100m   = np.max(PTDif100m) 
Max_PTDif1000m  = np.max(PTDif1000m) 
Max_WS17m_RS    = np.max(WS17m_RS)
Max_WS10mMet    = np.max(WS10mMet) 
Max_MLH         = np.max(MLH)

#------------------------------------------------------------------------------
# CALCULATE THE STANDARD DEVIATION

StDev_SurfBrO     = np.std(SurfBrO)
StDev_LTBrO       = np.std(LTBrO)
StDev_AEC         = np.std(AEC)
StDev_O3          = np.std(O3) 
StDev_Pres17m_RS  = np.std(Pres17m_RS)
StDev_SurfPresMet = np.std(SurfPresMet) 
StDev_Temp17m_RS  = np.std(Temp17m_RS)
StDev_SurfTempMet = np.std(SurfTempMet)
StDev_P1hrMet     = np.std(P1hrMet) 
StDev_PTDif100m   = np.std(PTDif100m) 
StDev_PTDif1000m  = np.std(PTDif1000m) 
StDev_WS17m_RS    = np.std(WS17m_RS)
StDev_WS10mMet    = np.std(WS10mMet) 
StDev_MLH         = np.std(MLH)

#------------------------------------------------------------------------------
# CALCULATE THE MEDIAN ABSOLUTE DEVIATION

Mad_SurfBrO     = stats.median_absolute_deviation(SurfBrO)
Mad_LTBrO       = stats.median_absolute_deviation(LTBrO)
Mad_AEC         = stats.median_absolute_deviation(AEC)
Mad_O3          = stats.median_absolute_deviation(O3) 
Mad_Pres17m_RS  = stats.median_absolute_deviation(Pres17m_RS)
Mad_SurfPresMet = stats.median_absolute_deviation(SurfPresMet) 
Mad_Temp17m_RS  = stats.median_absolute_deviation(Temp17m_RS)
Mad_SurfTempMet = stats.median_absolute_deviation(SurfTempMet)
Mad_P1hrMet     = stats.median_absolute_deviation(P1hrMet) 
Mad_PTDif100m   = stats.median_absolute_deviation(PTDif100m) 
Mad_PTDif1000m  = stats.median_absolute_deviation(PTDif1000m) 
Mad_WS17m_RS    = stats.median_absolute_deviation(WS17m_RS)
Mad_WS10mMet    = stats.median_absolute_deviation(WS10mMet)
Mad_MLH         = stats.median_absolute_deviation(MLH)

#------------------------------------------------------------------------------
# CALCULATE THE MEAN - ST DEV

MeanMStDev_SurfBrO     = Mean_SurfBrO     - StDev_SurfBrO
MeanMStDev_LTBrO       = Mean_LTBrO       - StDev_LTBrO
MeanMStDev_AEC         = Mean_AEC         - StDev_AEC
MeanMStDev_O3          = Mean_O3          - StDev_O3
MeanMStDev_Pres17m_RS  = Mean_Pres17m_RS  - StDev_Pres17m_RS
MeanMStDev_SurfPresMet = Mean_SurfPresMet - StDev_SurfPresMet
MeanMStDev_Temp17m_RS  = Mean_Temp17m_RS  - StDev_Temp17m_RS
MeanMStDev_SurfTempMet = Mean_SurfTempMet - StDev_SurfTempMet
MeanMStDev_P1hrMet     = Mean_P1hrMet     - StDev_P1hrMet
MeanMStDev_PTDif100m   = Mean_PTDif100m   - StDev_PTDif100m
MeanMStDev_PTDif1000m  = Mean_PTDif1000m  - StDev_PTDif1000m
MeanMStDev_WS17m_RS    = Mean_WS17m_RS    - StDev_WS17m_RS
MeanMStDev_WS10mMet    = Mean_WS10mMet    - StDev_WS10mMet
MeanMStDev_MLH         = Mean_MLH         - StDev_MLH

#------------------------------------------------------------------------------
# CALCULATE THE MEAN + ST DEV

MeanPStDev_SurfBrO     = Mean_SurfBrO     + StDev_SurfBrO
MeanPStDev_LTBrO       = Mean_LTBrO       + StDev_LTBrO
MeanPStDev_AEC         = Mean_AEC         + StDev_AEC
MeanPStDev_O3          = Mean_O3          + StDev_O3
MeanPStDev_Pres17m_RS  = Mean_Pres17m_RS  + StDev_Pres17m_RS
MeanPStDev_SurfPresMet = Mean_SurfPresMet + StDev_SurfPresMet
MeanPStDev_Temp17m_RS  = Mean_Temp17m_RS  + StDev_Temp17m_RS
MeanPStDev_SurfTempMet = Mean_SurfTempMet + StDev_SurfTempMet
MeanPStDev_P1hrMet     = Mean_P1hrMet     + StDev_P1hrMet
MeanPStDev_PTDif100m   = Mean_PTDif100m   + StDev_PTDif100m
MeanPStDev_PTDif1000m  = Mean_PTDif1000m  + StDev_PTDif1000m
MeanPStDev_WS17m_RS    = Mean_WS17m_RS    + StDev_WS17m_RS
MeanPStDev_WS10mMet    = Mean_WS10mMet    + StDev_WS10mMet
MeanPStDev_MLH         = Mean_MLH         + StDev_MLH

#------------------------------------------------------------------------------
# STANDARDISE THE VARIABLES (SUBTRACT MEAN & DIVIDE BY ST DEV)

Stand_SurfBrO     = (SurfBrO     - Mean_SurfBrO)     / StDev_SurfBrO
Stand_LTBrO       = (LTBrO       - Mean_LTBrO)       / StDev_LTBrO
Stand_AEC         = (AEC         - Mean_AEC)         / StDev_AEC
Stand_O3          = (O3          - Mean_O3)          / StDev_O3
Stand_Pres17m_RS  = (Pres17m_RS  - Mean_Pres17m_RS)  / StDev_Pres17m_RS
Stand_SurfPresMet = (SurfPresMet - Mean_SurfPresMet) / StDev_SurfPresMet
Stand_Temp17m_RS  = (Temp17m_RS  - Mean_Temp17m_RS)  / StDev_Temp17m_RS
Stand_SurfTempMet = (SurfTempMet - Mean_SurfTempMet) / StDev_SurfTempMet
Stand_P1hrMet     = (P1hrMet     - Mean_P1hrMet)     / StDev_P1hrMet
Stand_PTDif100m   = (PTDif100m   - Mean_PTDif100m)   / StDev_PTDif100m
Stand_PTDif1000m  = (PTDif1000m  - Mean_PTDif1000m)  / StDev_PTDif1000m
Stand_WS17m_RS    = (WS17m_RS    - Mean_WS17m_RS)    / StDev_WS17m_RS
Stand_WS10mMet    = (WS10mMet    - Mean_WS10mMet)    / StDev_WS10mMet
Stand_MLH         = (MLH         - Mean_MLH)         / StDev_MLH

# Build a pandas dataframe (need to exclude BrO)
StandVariables = np.column_stack((Stand_O3,Stand_AEC,Stand_SurfTempMet,Stand_SurfPresMet,Stand_WS10mMet,Stand_MLH,Stand_P1hrMet,Stand_PTDif1000m,Stand_PTDif100m))
StandVariables = pd.DataFrame(StandVariables, columns = ['O3','AEC','Surf_Temp','Surf_Pres','WS10m','MLH','P1hr','PTDif1000m','PTDif100m'])

#------------------------------------------------------------------------------
# SANITY CHECK ON STANDARDISED VARIABLES (MEAN = 0, StDev = 1, RANGE ~ 2-3 StDev MAX)

# Mean of standardised variables
Mean_Stand_SurfBrO     = np.mean(Stand_SurfBrO)
Mean_Stand_LTBrO       = np.mean(Stand_LTBrO)
Mean_Stand_AEC         = np.mean(Stand_AEC)
Mean_Stand_O3          = np.mean(Stand_O3)
Mean_Stand_Pres17m_RS  = np.mean(Stand_Pres17m_RS)
Mean_Stand_SurfPresMet = np.mean(Stand_SurfPresMet)
Mean_Stand_Temp17m_RS  = np.mean(Stand_Temp17m_RS)
Mean_Stand_SurfTempMet = np.mean(Stand_SurfTempMet)
Mean_Stand_P1hrMet     = np.mean(Stand_P1hrMet)
Mean_Stand_PTDif100m   = np.mean(Stand_PTDif100m)
Mean_Stand_PTDif1000m  = np.mean(Stand_PTDif1000m)
Mean_Stand_WS17m_RS    = np.mean(Stand_WS17m_RS)
Mean_Stand_WS10mMet    = np.mean(Stand_WS10mMet)
Mean_Stand_MLH         = np.mean(Stand_MLH)

# StDev of standardised variables
StDev_Stand_SurfBrO     = np.std(Stand_SurfBrO)
StDev_Stand_LTBrO       = np.std(Stand_LTBrO)
StDev_Stand_AEC         = np.std(Stand_AEC)
StDev_Stand_O3          = np.std(Stand_O3)
StDev_Stand_Pres17m_RS  = np.std(Stand_Pres17m_RS)
StDev_Stand_SurfPresMet = np.std(Stand_SurfPresMet)
StDev_Stand_Temp17m_RS  = np.std(Stand_Temp17m_RS)
StDev_Stand_SurfTempMet = np.std(Stand_SurfTempMet)
StDev_Stand_P1hrMet     = np.std(Stand_P1hrMet)
StDev_Stand_PTDif100m   = np.std(Stand_PTDif100m)
StDev_Stand_PTDif1000m  = np.std(Stand_PTDif1000m)
StDev_Stand_WS17m_RS    = np.std(Stand_WS17m_RS)
StDev_Stand_WS10mMet    = np.std(Stand_WS10mMet)
StDev_Stand_MLH         = np.std(Stand_MLH)

# Range of standardised variables
Range_Stand_SurfBrO     = np.ptp(Stand_SurfBrO)
Range_Stand_LTBrO       = np.ptp(Stand_LTBrO)
Range_Stand_AEC         = np.ptp(Stand_AEC)
Range_Stand_O3          = np.ptp(Stand_O3)
Range_Stand_Pres17m_RS  = np.ptp(Stand_Pres17m_RS)
Range_Stand_SurfPresMet = np.ptp(Stand_SurfPresMet)
Range_Stand_Temp17m_RS  = np.ptp(Stand_Temp17m_RS)
Range_Stand_SurfTempMet = np.ptp(Stand_SurfTempMet)
Range_Stand_P1hrMet     = np.ptp(Stand_P1hrMet)
Range_Stand_PTDif100m   = np.ptp(Stand_PTDif100m)
Range_Stand_PTDif1000m  = np.ptp(Stand_PTDif1000m)
Range_Stand_WS17m_RS    = np.ptp(Stand_WS17m_RS)
Range_Stand_WS10mMet    = np.ptp(Stand_WS10mMet)
Range_Stand_MLH         = np.ptp(Stand_MLH)

# Build a pandas dataframe
Sanity3 = {'Mean': [Mean_Stand_SurfBrO,Mean_Stand_LTBrO,Mean_Stand_AEC,Mean_Stand_O3,Mean_Stand_Pres17m_RS,Mean_Stand_SurfPresMet,Mean_Stand_Temp17m_RS,Mean_Stand_SurfTempMet,Mean_Stand_P1hrMet,Mean_Stand_PTDif100m,Mean_Stand_PTDif1000m,Mean_Stand_WS17m_RS,Mean_Stand_WS10mMet,Mean_Stand_MLH],
           'StDev': [Mean_Stand_SurfBrO,Mean_Stand_LTBrO,Mean_Stand_AEC,Mean_Stand_O3,Mean_Stand_Pres17m_RS,Mean_Stand_SurfPresMet,Mean_Stand_Temp17m_RS,Mean_Stand_SurfTempMet,Mean_Stand_P1hrMet,Mean_Stand_PTDif100m,Mean_Stand_PTDif1000m,Mean_Stand_WS17m_RS,Mean_Stand_WS10mMet,Mean_Stand_MLH],
           'Range': [Mean_Stand_SurfBrO,Mean_Stand_LTBrO,Mean_Stand_AEC,Mean_Stand_O3,Mean_Stand_Pres17m_RS,Mean_Stand_SurfPresMet,Mean_Stand_Temp17m_RS,Mean_Stand_SurfTempMet,Mean_Stand_P1hrMet,Mean_Stand_PTDif100m,Mean_Stand_PTDif1000m,Mean_Stand_WS17m_RS,Mean_Stand_WS10mMet,Mean_Stand_MLH]}
Sanity3 = pd.DataFrame(Sanity3, columns = ['Mean','StDev','Range'],index = ['SurfBrO','LTBrO','AEC','O3','Pres17m_RS','SurfPres_Met','Temp17m_RS','SurfTemp_Met','P1hr_Met','PTDif100m','PTDif1000m','WS17m_RS','WS10m_Met','MLH'])
Sanity3.to_csv('/Users/ncp532/Documents/Data/MERRA2/Sanity3.csv')

#------------------------------------------------------------------------------
# CONVERT AEC & BrO BACK

# BrO
SurfBrO            = np.square(SurfBrO)
Mean_SurfBrO       = np.square(Mean_SurfBrO)
StDev_SurfBrO      = np.square(StDev_SurfBrO)
MeanMStDev_SurfBrO = np.square(MeanMStDev_SurfBrO)
MeanPStDev_SurfBrO = np.square(MeanPStDev_SurfBrO)
Median_SurfBrO     = np.square(Median_SurfBrO)
Mad_SurfBrO        = np.square(Mad_SurfBrO)
Min_SurfBrO        = np.square(Min_SurfBrO)
Max_SurfBrO        = np.square(Max_SurfBrO)

LTBrO              = np.square(LTBrO)
Mean_LTBrO         = np.square(Mean_LTBrO)
StDev_LTBrO        = np.square(StDev_LTBrO)
MeanMStDev_LTBrO   = np.square(MeanMStDev_LTBrO)
MeanPStDev_LTBrO   = np.square(MeanPStDev_LTBrO)
Median_LTBrO       = np.square(Median_LTBrO)
Mad_LTBrO          = np.square(Mad_LTBrO)
Min_LTBrO          = np.square(Min_LTBrO)
Max_LTBrO          = np.square(Max_LTBrO)

# AEC
AEC                = np.exp(AEC)
Mean_AEC           = np.exp(Mean_AEC)
StDev_AEC          = np.exp(StDev_AEC)
MeanMStDev_AEC     = np.exp(MeanMStDev_AEC)
MeanPStDev_AEC     = np.exp(MeanPStDev_AEC)
Median_AEC         = np.square(Median_AEC)
Mad_AEC            = np.square(Mad_AEC)
Min_AEC            = np.square(Min_AEC)
Max_AEC            = np.square(Max_AEC)

#------------------------------------------------------------------------------
# BUILD DATAFRAME FOR MEAN & STDEV OF VARIABLES

# Build a pandas dataframe
dfMeanStDev = {'Mean': [Mean_SurfBrO/1e12,Mean_LTBrO/1e12,Mean_AEC,Mean_O3,Mean_Pres17m_RS,Mean_SurfPresMet,Mean_Temp17m_RS,Mean_SurfTempMet,Mean_P1hrMet,Mean_PTDif100m,Mean_PTDif1000m,Mean_WS17m_RS,Mean_WS10mMet,Mean_MLH],
               'StDev': [StDev_SurfBrO/1e12,StDev_LTBrO/1e12,StDev_AEC,StDev_O3,StDev_Pres17m_RS,StDev_SurfPresMet,StDev_Temp17m_RS,StDev_SurfTempMet,StDev_P1hrMet,StDev_PTDif100m,StDev_PTDif1000m,StDev_WS17m_RS,StDev_WS10mMet,StDev_MLH],
               'MeanMStDev': [MeanMStDev_SurfBrO/1e12,MeanMStDev_LTBrO/1e12,MeanMStDev_AEC,MeanMStDev_O3,MeanMStDev_Pres17m_RS,MeanMStDev_SurfPresMet,MeanMStDev_Temp17m_RS,MeanMStDev_SurfTempMet,MeanMStDev_P1hrMet,MeanMStDev_PTDif100m,MeanMStDev_PTDif1000m,MeanMStDev_WS17m_RS,MeanMStDev_WS10mMet,MeanMStDev_MLH],
               'MeanPStDev': [MeanPStDev_SurfBrO/1e12,MeanPStDev_LTBrO/1e12,MeanPStDev_AEC,MeanPStDev_O3,MeanPStDev_Pres17m_RS,MeanPStDev_SurfPresMet,MeanPStDev_Temp17m_RS,MeanPStDev_SurfTempMet,MeanPStDev_P1hrMet,MeanPStDev_PTDif100m,MeanPStDev_PTDif1000m,MeanPStDev_WS17m_RS,MeanPStDev_WS10mMet,MeanPStDev_MLH],
               'Median': [Median_SurfBrO/1e12,Median_LTBrO/1e12,Median_AEC,Median_O3,Median_Pres17m_RS,Median_SurfPresMet,Median_Temp17m_RS,Median_SurfTempMet,Median_P1hrMet,Median_PTDif100m,Median_PTDif1000m,Median_WS17m_RS,Median_WS10mMet,Median_MLH],
               'MAD': [Mad_SurfBrO/1e12,Mad_LTBrO/1e12,Mad_AEC,Mad_O3,Mad_Pres17m_RS,Mad_SurfPresMet,Mad_Temp17m_RS,Mad_SurfTempMet,Mad_P1hrMet,Mad_PTDif100m,Mad_PTDif1000m,Mad_WS17m_RS,Mad_WS10mMet,Mad_MLH],
               'Min': [Min_SurfBrO/1e12,Min_LTBrO/1e12,Min_AEC,Min_O3,Min_Pres17m_RS,Min_SurfPresMet,Min_Temp17m_RS,Min_SurfTempMet,Min_P1hrMet,Min_PTDif100m,Min_PTDif1000m,Min_WS17m_RS,Min_WS10mMet,Min_MLH],
               'Max': [Max_SurfBrO/1e12,Max_LTBrO/1e12,Max_AEC,Max_O3,Max_Pres17m_RS,Max_SurfPresMet,Max_Temp17m_RS,Max_SurfTempMet,Max_P1hrMet,Max_PTDif100m,Max_PTDif1000m,Max_WS17m_RS,Max_WS10mMet,Max_MLH]}
dfMeanStDev = pd.DataFrame(dfMeanStDev, columns = ['Mean','StDev','MeanMStDev','MeanPStDev','Median','MAD','Min','Max'],index = ['SurfBrO','LTBrO','AEC','O3','Pres17m_RS','SurfPres_Met','Temp17m_RS','SurfTemp_Met','P1hr_Met','PTDif100m','PTDif1000m','WS17m_RS','WS10m_Met','MLH'])
dfMeanStDev.to_csv('/Users/ncp532/Documents/Data/MERRA2/MeanStDev.csv')

#------------------------------------------------------------------------------
# COUNT THE NUMBER OF NEGATIVE AND POSITIVE PTDIF100M

list1 = PTDif100m

pos_count, neg_count = 0, 0
  
# iterating each number in list 
for num in list1: 
      
    # checking condition 
    if num >= 0: 
        pos_count += 1
  
    else: 
        neg_count += 1

print("Positive numbers in the list: ", pos_count) 
print("Negative numbers in the list: ", neg_count) 

#------------------------------------------------------------------------------
# PERFORM A PRINCIPAL COMPONENT ANALYSIS (PCA)

## Apply the PCA (Swanson et al has 9 PCs, but only 3 have variances greater 1)
#PCA_BrO = PCA() # All n components
#
## Retrieve the principal components (PCs)
#PrincipalComponents_Variables = PCA_BrO.fit_transform(StandVariables) # Variables
#
## Put the principle components into a DataFrame
#Principal_Variables_Df = pd.DataFrame(data = PrincipalComponents_Variables, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9']) # Variables
#    
## Explained variation per principal component
#print('Explained variation per principal component: {}'.format(PCA_BrO.explained_variance_ratio_))
#Explained_Variance = PCA_BrO.explained_variance_ratio_
#
## Get the loadings
#loadings = pd.DataFrame(PCA_BrO.components_.T, columns=Principal_Variables_Df.columns,index=StandVariables.columns)
#
## Calculate the normalised variance for each PC
#PV_PC1      = Principal_Variables_Df['PC1']
#PV_PC1_Mean = np.mean(Principal_Variables_Df['PC1'])
#NV_PC1      = np.mean(np.square(PV_PC1 - PV_PC1_Mean))
#
#PV_PC2      = Principal_Variables_Df['PC2']
#PV_PC2_Mean = np.mean(Principal_Variables_Df['PC2'])
#NV_PC2      = np.mean(np.square(PV_PC2 - PV_PC2_Mean))
#
#PV_PC3      = Principal_Variables_Df['PC3']
#PV_PC3_Mean = np.mean(Principal_Variables_Df['PC3'])
#NV_PC3      = np.mean(np.square(PV_PC3 - PV_PC3_Mean))
#
#PV_PC4      = Principal_Variables_Df['PC4']
#PV_PC4_Mean = np.mean(Principal_Variables_Df['PC4'])
#NV_PC4      = np.mean(np.square(PV_PC4 - PV_PC4_Mean))
#
#PV_PC5      = Principal_Variables_Df['PC5']
#PV_PC5_Mean = np.mean(Principal_Variables_Df['PC5'])
#NV_PC5      = np.mean(np.square(PV_PC5 - PV_PC5_Mean))
#
#PV_PC6      = Principal_Variables_Df['PC6']
#PV_PC6_Mean = np.mean(Principal_Variables_Df['PC6'])
#NV_PC6      = np.mean(np.square(PV_PC6 - PV_PC6_Mean))
#
#PV_PC7      = Principal_Variables_Df['PC7']
#PV_PC7_Mean = np.mean(Principal_Variables_Df['PC7'])
#NV_PC7      = np.mean(np.square(PV_PC7 - PV_PC7_Mean))
#
#PV_PC8      = Principal_Variables_Df['PC8']
#PV_PC8_Mean = np.mean(Principal_Variables_Df['PC8'])
#NV_PC8      = np.mean(np.square(PV_PC8 - PV_PC8_Mean))
#
#PV_PC9      = Principal_Variables_Df['PC9']
#PV_PC9_Mean = np.mean(Principal_Variables_Df['PC9'])
#NV_PC9      = np.mean(np.square(PV_PC9 - PV_PC9_Mean))
#
## Put the normalised variance into an array
#PCA_NV = np.array([NV_PC1,NV_PC2,NV_PC3,NV_PC4,NV_PC5,NV_PC6,NV_PC7,NV_PC8,NV_PC9])
#
## Export the PCA results
#dfPCA_BrO = np.row_stack((Explained_Variance,PCA_NV))
#dfPCA_BrO = pd.DataFrame(dfPCA_BrO, index = ['Explained_Variance','Normalised_Variance'],columns = Principal_Variables_Df.columns)
#dfPCA_BrO = pd.concat([loadings,dfPCA_BrO])
#dfPCA_BrO.to_csv('/Users/ncp532/Documents/Data/MERRA2/PCA_Loadings_&_Variance.csv')

#------------------------------------------------------------------------------
# LOADINGS FOR THE PRINCIPLE COMPONENTS

# Swanson (2020)
O3_PC1L,      O3_PC2L,      O3_PC3L      = -0.021, 0.347,  -0.569 # Ozone (ppb) # (nmol/mol)?
EXT_PC1L,     EXT_PC2L,     EXT_PC3L     = 0.246,  0.270,  0.216  # Aerosol extinction (km-1) # (m/km)?
STemp_PC1L,   STemp_PC2L,   STemp_PC3L   = 0.087,  -0.392, -0.582 # Surface Temp (K) # (C)?
SLP_PC1L,     SLP_PC2L,     SLP_PC3L     = -0.338, 0.160,  0.231  # Sea level pressure (hPa)
VW10m_PC1L,   VW10m_PC2L,   VW10m_PC3L   = 0.345,  0.459,  -0.263 # Windspeed at 10m (m/s)
MLH_PC1L,     MLH_PC2L,     MLH_PC3L     = 0.595,  0.041,  -0.008 # Richardson mixed layer height (m)
P1hr_PC1L,    P1hr_PC2L,    P1hr_PC3L    = -0.007, -0.271, 0.196  # Change in pressure from one hour to next (hPa)
PT1000m_PC1L, PT1000m_PC2L, PT1000m_PC3L = -0.326, 0.580,  0.041  # Potential temperature differential in lowest 1000m (m/K)
PT100m_PC1L,  PT100m_PC2L,  PT100m_PC3L  = -0.487, -0.069, -0.358 # Potential temperature differential in lowest 100m (m/K)

## My loadings
#O3_PC1L,      O3_PC2L,      O3_PC3L      = loadings['PC1'][0], loadings['PC2'][0], loadings['PC3'][0] # Ozone (ppb) # (nmol/mol)?
#EXT_PC1L,     EXT_PC2L,     EXT_PC3L     = loadings['PC1'][1], loadings['PC2'][1], loadings['PC3'][1] # Aerosol extinction (km-1) # (m/km)?
#STemp_PC1L,   STemp_PC2L,   STemp_PC3L   = loadings['PC1'][2], loadings['PC2'][2], loadings['PC3'][2] # Surface Temp (K) # (C)?
#SLP_PC1L,     SLP_PC2L,     SLP_PC3L     = loadings['PC1'][3], loadings['PC2'][3], loadings['PC3'][3] # Sea level pressure (hPa)
#VW10m_PC1L,   VW10m_PC2L,   VW10m_PC3L   = loadings['PC1'][4], loadings['PC2'][4], loadings['PC3'][4] # Windspeed at 10m (m/s)
#MLH_PC1L,     MLH_PC2L,     MLH_PC3L     = loadings['PC1'][5], loadings['PC2'][5], loadings['PC3'][5] # Richardson mixed layer height (m)
#P1hr_PC1L,    P1hr_PC2L,    P1hr_PC3L    = loadings['PC1'][6], loadings['PC2'][6], loadings['PC3'][6]  # Change in pressure from one hour to next (hPa)
#PT1000m_PC1L, PT1000m_PC2L, PT1000m_PC3L = loadings['PC1'][7], loadings['PC2'][7], loadings['PC3'][7]  # Potential temperature differential in lowest 1000m (m/K)
#PT100m_PC1L,  PT100m_PC2L,  PT100m_PC3L  = loadings['PC1'][8], loadings['PC2'][8], loadings['PC3'][8] # Potential temperature differential in lowest 100m (m/K)

#------------------------------------------------------------------------------
# CALCULATE THE PRINCIPLE COMPONENTS

# Met
PC1_Met = (Stand_O3*O3_PC1L) + (Stand_AEC*EXT_PC1L) + (Stand_SurfTempMet*STemp_PC1L) + (Stand_SurfPresMet*SLP_PC1L) + (Stand_WS10mMet*VW10m_PC1L) + (Stand_MLH*MLH_PC1L) + (Stand_P1hrMet*P1hr_PC1L) + (Stand_PTDif1000m*PT1000m_PC1L) + (Stand_PTDif100m*PT100m_PC1L)
PC2_Met = (Stand_O3*O3_PC2L) + (Stand_AEC*EXT_PC2L) + (Stand_SurfTempMet*STemp_PC2L) + (Stand_SurfPresMet*SLP_PC2L) + (Stand_WS10mMet*VW10m_PC2L) + (Stand_MLH*MLH_PC2L) + (Stand_P1hrMet*P1hr_PC2L) + (Stand_PTDif1000m*PT1000m_PC2L) + (Stand_PTDif100m*PT100m_PC2L) 
PC3_Met = (Stand_O3*O3_PC3L) + (Stand_AEC*EXT_PC3L) + (Stand_SurfTempMet*STemp_PC3L) + (Stand_SurfPresMet*SLP_PC3L) + (Stand_WS10mMet*VW10m_PC3L) + (Stand_MLH*MLH_PC3L) + (Stand_P1hrMet*P1hr_PC3L) + (Stand_PTDif1000m*PT1000m_PC3L) + (Stand_PTDif100m*PT100m_PC3L) 

#------------------------------------------------------------------------------
# PERFORM A PRINCIPLE COMPONENT REGRESSION (PCR)
# (if z = sqrt(BrO_obs)) “z ~ pc1 + pc2 + pc3

# Variables required
PC1a     = np.array(PC1_Met)
PC2a     = np.array(PC2_Met)
PC3a     = np.array(PC3_Met)
SurfBrOa = np.array(SurfBrO)
LTBrOa   = np.array(LTBrO)

# First we need to flatten the data: it's 2D layout is not relevent.
PC1a     = PC1a.flatten()
PC2a     = PC2a.flatten()
PC3a     = PC3a.flatten()
SurfBrOa = SurfBrOa.flatten()
LTBrOa   = LTBrOa.flatten()

# Build the DataFrame
dataS  = pd.DataFrame({'PC1': PC1a, 'PC2': PC2a, 'PC3': PC3a, 'z': np.sqrt(SurfBrOa)})
dataLT = pd.DataFrame({'PC1': PC1a, 'PC2': PC2a, 'PC3': PC3a, 'z': np.sqrt(LTBrOa)})
XS     = dataS[['PC1','PC2','PC3']]
YS     = dataS['z']
XLT    = dataLT[['PC1','PC2','PC3']]
YLT    = dataLT['z']

# Fit the model
modelS  = ols("z ~ PC1 + PC2 + PC3", dataS).fit()  # Surface StatsModel (ols)
modelLT = ols("z ~ PC1 + PC2 + PC3", dataLT).fit() # LTcol StatsModel (ols)
regS    = LinearRegression().fit(XS,  YS)          # Surface SkLearn (LinearRegresion)
regLT   = LinearRegression().fit(XLT, YLT)         # LTcol SkLearn (LinearRegresion)
 
# Retrieve the model results
Model_resultsS  = modelS._results.params                                  # Surface StatsModel (ols)
Model_resultsLT = modelLT._results.params                                 # LTcol StatsModel (ols)
Intercept_resultS, Coefficients_resultS   = regS.intercept_,  regS.coef_  # Surface SkLearn (LinearRegresion)
Intercept_resultLT, Coefficients_resultLT = regLT.intercept_, regLT.coef_ # Surface SkLearn (LinearRegresion)

# Peform analysis of variance on fitted linear model
anova_resultsS  = anova_lm(modelS)
anova_resultsLT = anova_lm(modelLT)

#------------------------------------------------------------------------------
# APPLY THE BrO REGRESSION MODEL

#--------------
# Intercept and Coefficients
#--------------

# Surface BrO
B0S = 2.06*1e6 # Intercept for the multiple linear regression
B1S = 1.46*1e5 # Slope PC1 (coefficient 1)
B2S = 2.24*1e5 # Slope PC2 (coefficient 2)
B3S = 3.94*1e5 # Slope PC3 (coefficient 3)

# Lower tropospheric BrO
B0LT = 3.67*1e6 # Intercept for the multiple linear regression
B1LT = 3.66*1e5 # Slope PC1 (coefficient 1)
B2LT = 9.88*1e4 # Slope PC2 (coefficient 2)
B3LT = 5.97*1e5 # Slope PC3 (coefficient 3)

# SkLearn (LinearRegression)
LR0S = Intercept_resultS
LR1S = Coefficients_resultS[0]
LR2S = Coefficients_resultS[1]
LR3S = Coefficients_resultS[2]

LR0LT = Intercept_resultLT
LR1LT = Coefficients_resultLT[0]
LR2LT = Coefficients_resultLT[1]
LR3LT = Coefficients_resultLT[2]

# StatsModels (ols)
OLS0S = Model_resultsS[0]
OLS1S = Model_resultsS[1]
OLS2S = Model_resultsS[2]
OLS3S = Model_resultsS[3]

OLS0LT = Model_resultsLT[0]
OLS1LT = Model_resultsLT[1]
OLS2LT = Model_resultsLT[2]
OLS3LT = Model_resultsLT[3]

#--------------
# Regression model
#--------------

# Surface BrO
BrO_SurfPred_Met = np.square(B0S + (B1S*PC1_Met) + (B2S*PC2_Met) + (B3S*PC3_Met))

# Lower tropospheric BrO
BrO_LTPred_Met = np.square(B0LT + (B1LT*PC1_Met) + (B2LT*PC2_Met) + (B3LT*PC3_Met))

# SkLearn (LinearRegression)
BrO_SurfPred_LR = np.square(LR0S  + (LR1S*PC1_Met)  + (LR2S*PC2_Met)  + (LR3S*PC3_Met))
BrO_LTPred_LR   = np.square(LR0LT + (LR1LT*PC1_Met) + (LR2LT*PC2_Met) + (LR3LT*PC3_Met))

# StatsModels (ols)
BrO_SurfPred_OLS = np.square(OLS0S  + (OLS1S*PC1_Met)  + (OLS2S*PC2_Met)  + (OLS3S*PC3_Met))
BrO_LTPred_OLS   = np.square(OLS0LT + (OLS1LT*PC1_Met) + (OLS2LT*PC2_Met) + (OLS3LT*PC3_Met))

#------------------------------------------------------------------------------
#BUILD DATAFRAME FOR INTERCEPTS & COEFFICIENTS

# Build a pandas dataframe
dfIntCoef = {'Intercept (B0)': [LR0S,LR0LT,OLS0S,OLS0LT],
             'Coefficient 1 (B1)': [LR1S,LR1LT,OLS1S,OLS1LT],
             'Coefficient 2 (B2)': [LR2S,LR2LT,OLS2S,OLS2LT],
             'Coefficient 3 (B3)': [LR3S,LR3LT,OLS3S,OLS3LT]}
dfIntCoef = pd.DataFrame(dfIntCoef, index = ['LR_Surf','LR_LTcol','ols_Surf','ols_LTcol'],columns = ['Intercept (B0)','Coefficient 1 (B1)','Coefficient 2 (B2)','Coefficient 3 (B3)'])
dfIntCoef.to_csv('/Users/ncp532/Documents/Data/MERRA2/IntCoef.csv')

# Export analysis of variance results
dfAnovaS  = pd.DataFrame(anova_resultsS)
dfAnovaLT = pd.DataFrame(anova_resultsLT)
dfAnovaS.to_csv('/Users/ncp532/Documents/Data/MERRA2/AnovaS.csv')
dfAnovaLT.to_csv('/Users/ncp532/Documents/Data/MERRA2/AnovaLT.csv')

#------------------------------------------------------------------------------
# BUILD A DATAFRAME FOR REGRESSION MODEL RESULTS

# Merge BrO observations & prediction
df1 = pd.concat([BrO_SurfPred_Met,BrO_LTPred_Met],axis=1,join='inner')
df1 = pd.concat([df1,dfBrO_Surf['BrO_(molecules/cm2)']],axis=1,join='outer')
df1 = pd.concat([df1,dfBrO_LtCol['BrO_(molecules/cm2)']],axis=1,join='outer')
df1 = pd.concat([df1,SurfBrO_Err],axis=1,join='outer')
df1 = pd.concat([df1,LTBrO_Err],axis=1,join='outer')
df1 = pd.concat([df1,BrO_SurfPred_LR],axis=1,join='outer')
df1 = pd.concat([df1,BrO_LTPred_LR],axis=1,join='outer')
df1 = pd.concat([df1,BrO_SurfPred_OLS],axis=1,join='outer')
df1 = pd.concat([df1,BrO_LTPred_OLS],axis=1,join='outer')

# Name the columns
df1.columns = ['BrO_SurfPred_Met','BrO_LtColPred_Met','BrO_SurfObs','BrO_LtColObs','BrO_SurfObs_Err','BrO_LtColObs_Err','BrO_SurfPred_LR','BrO_LTPred_LR','BrO_SurfPred_OLS','BrO_LTPred_OLS']

#------------------------------------------------------------------------------
# EXPORT THE DATAFRAMES AS .CSV

df1.to_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_Radiosonde.csv')
