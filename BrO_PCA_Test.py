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
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.decomposition import PCA

# Date and Time handling package
from datetime import datetime,timedelta		# functions to handle date and time

#------------------------------------------------------------------------------
# DEFINE THE DATASETS

#--------------
# MAX-DOAS
#--------------

# BrO (Retrieval)
BrO_V1_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V1_17/all_BrO/V1_17_BrO_retrieval.csv',index_col=0)       # BrO V1 (2017/18)
BrO_V2_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V2_17/all_BrO/V2_17_BrO_retrieval.csv',index_col=0)       # BrO V2 (2017/18)
BrO_V3_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V3_17/all_BrO/V3_17_BrO_retrieval.csv',index_col=0)       # BrO V3 (2017/18)

BrO_V1_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V1_18/all_BrO/V1_18_BrO_retrieval.csv',index_col=0)       # BrO V1 (2018/19)
BrO_V2_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V2_18/all_BrO/V2_18_BrO_retrieval.csv',index_col=0)       # BrO V2 (2018/19)
BrO_V3_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V3_18/all_BrO/V3_18_BrO_retrieval.csv',index_col=0)       # BrO V3 (2018/19)

BrO_SIPEXII = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/SIPEXII/all_BrO/SIPEXII_BrO_retrieval.csv',index_col=0) # BrO SIPEXII (2012)

# # BrO (VMR)
# BrO_VMR_V1_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V1_17/all_BrO/V1_17_BrO_VMR.csv', index_col=0)       # BrO V1 (2017/18)
# BrO_VMR_V2_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V2_17/all_BrO/V2_17_BrO_VMR.csv', index_col=0)       # BrO V2 (2017/18)
# BrO_VMR_V3_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V3_17/all_BrO/V3_17_BrO_VMR.csv', index_col=0)       # BrO V3 (2017/18)

# BrO_VMR_V1_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V1_18/all_BrO/V1_18_BrO_VMR.csv', index_col=0)       # BrO V1 (2018/19)
# BrO_VMR_V2_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V2_18/all_BrO/V2_18_BrO_VMR.csv', index_col=0)       # BrO V2 (2018/19)
# BrO_VMR_V3_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V3_18/all_BrO/V3_18_BrO_VMR.csv', index_col=0)       # BrO V3 (2018/19)

# BrO_VMR_SIPEXII = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/SIPEXII/all_BrO/SIPEXII_BrO_VMR.csv', index_col=0) # BrO SIPEXII (2012)

# # BrO (VMR Error)
# Err_VMR_V1_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V1_17/BrO_error/V1_17_BrO_error.csv', index_col=0) # BrO error V1 (2017/18)
# Err_VMR_V2_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V2_17/BrO_error/V2_17_BrO_error.csv', index_col=0) # BrO error V2 (2017/18)
# Err_VMR_V3_17 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V3_17/BrO_error/V3_17_BrO_error.csv', index_col=0) # BrO error V3 (2017/18)

# Err_VMR_V1_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V1_18/BrO_error/V1_18_BrO_error.csv', index_col=0) # BrO error V1 (2018/19)
# Err_VMR_V2_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V2_18/BrO_error/V2_18_BrO_error.csv', index_col=0) # BrO error V2 (2018/19)
# Err_VMR_V3_18 = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V3_18/BrO_error/V3_18_BrO_error.csv', index_col=0) # BrO error V3 (2018/19)

# Err_VMR_SIPEXII = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/SIPEXII/BrO_error/SIPEXII_BrO_error.csv', index_col=0) # BrO error SIPEXII (2012)

# AEC
AEC_V1_17   = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V1_17/all_Aerosol/V1_17_AeroExt_338.csv',index_col=0) # AEC at 338nm V1 (2017/18)
AEC_V2_17   = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V2_17/all_Aerosol/V2_17_AeroExt_338.csv',index_col=0) # AEC at 338nm V2 (2017/18)
AEC_V3_17M  = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V3_17/all_Aerosol/V3_17_AeroExt_338.csv',index_col=0) # AEC at 338nm V3 (2017/18)
AEC_V3_17D  = AEC_V3_17M

AEC_V1_18   = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V1_18/all_Aerosol/V1_18_AeroExt_338.csv',index_col=0) # AEC at 338nm V1 (2018/19)
AEC_V2_18   = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V2_18/all_Aerosol/V2_18_AeroExt_338.csv',index_col=0) # AEC at 338nm V2 (2018/19)
AEC_V3_18M  = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V3_18/all_Aerosol/V3_18_AeroExt_338.csv',index_col=0) # AEC at 338nm V3 (2018/19)
AEC_V3_18D  = AEC_V3_18M

AEC_SIPEXII = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/SIPEXII/all_Aerosol/SIPEXII_AeroExt_338.csv',index_col=0) # AEC at 338nm SIPEXII (2012)

# NO2
NO2_V1_17  = pd.read_csv('/Users/ncp532/Documents/data/CAMMPCAN_2017_18/MAX-DOAS/NO2_2dSCD_V1_17.csv', index_col=0) # NO2 dSCD at 2 degrees
NO2_V2_17  = pd.read_csv('/Users/ncp532/Documents/data/CAMMPCAN_2017_18/MAX-DOAS/NO2_2dSCD_V2_17.csv', index_col=0) # NO2 dSCD at 2 degrees
NO2_V3_17M = pd.read_csv('/Users/ncp532/Documents/data/CAMMPCAN_2017_18/MAX-DOAS/NO2_2dSCD_V3_17.csv', index_col=0) # NO2 dSCD at 2 degrees
NO2_V3_17D = NO2_V3_17M

NO2_V1_18  = pd.read_csv('/Users/ncp532/Documents/data/CAMMPCAN_2018_19/MAX-DOAS/NO2_2dSCD_V1_18.csv', index_col=0) # NO2 dSCD at 2 degrees
NO2_V2_18  = pd.read_csv('/Users/ncp532/Documents/data/CAMMPCAN_2018_19/MAX-DOAS/NO2_2dSCD_V2_18.csv', index_col=0) # NO2 dSCD at 2 degrees
NO2_V3_18M = pd.read_csv('/Users/ncp532/Documents/data/CAMMPCAN_2018_19/MAX-DOAS/NO2_2dSCD_V3_18.csv', index_col=0) # NO2 dSCD at 2 degrees
NO2_V3_18D = NO2_V3_18M

# NO2_SIPEXII = pd.read_csv('/Users/ncp532/Documents/data/.csv, index_col=0) # NO2 dSCD at 2 degrees

# BrOdSCD
BrOdSCD_V1_17  = pd.read_csv('/Users/ncp532/Documents/data/CAMMPCAN_2017_18/MAX-DOAS/BrO_1dSCD_V1_17.csv', index_col=0) # BrO dSCD at 1 degrees
BrOdSCD_V2_17  = pd.read_csv('/Users/ncp532/Documents/data/CAMMPCAN_2017_18/MAX-DOAS/BrO_1dSCD_V2_17.csv', index_col=0) # BrO dSCD at 1 degrees
BrOdSCD_V3_17M = pd.read_csv('/Users/ncp532/Documents/data/CAMMPCAN_2017_18/MAX-DOAS/BrO_1dSCD_V3_17.csv', index_col=0) # BrO dSCD at 1 degrees
BrOdSCD_V3_17D = BrOdSCD_V3_17M

BrOdSCD_V1_18  = pd.read_csv('/Users/ncp532/Documents/data/CAMMPCAN_2018_19/MAX-DOAS/BrO_1dSCD_V1_18.csv', index_col=0) # BrO dSCD at 1 degrees
BrOdSCD_V2_18  = pd.read_csv('/Users/ncp532/Documents/data/CAMMPCAN_2018_19/MAX-DOAS/BrO_1dSCD_V2_18.csv', index_col=0) # BrO dSCD at 1 degrees
BrOdSCD_V3_18M = pd.read_csv('/Users/ncp532/Documents/data/CAMMPCAN_2018_19/MAX-DOAS/BrO_1dSCD_V3_18.csv', index_col=0) # BrO dSCD at 1 degrees
BrOdSCD_V3_18D = BrOdSCD_V3_18M

# BrOdSCD_SIPEXII = pd.read_csv('/Users/ncp532/Documents/data/.csv, index_col=0) # BrO dSCD at 1 degrees

# SZA
SZA_V1_17  = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V1_17/all_SZA/V1_17_SZA.csv',index_col=0) # SZA V1 (2017/18)
SZA_V2_17  = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V2_17/all_SZA/V2_17_SZA.csv',index_col=0) # SZA V2 (2017/18)
SZA_V3_17M = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V3_17/all_SZA/V3_17_SZA.csv',index_col=0) # SZA V3 (2017/18)
SZA_V3_17D = SZA_V3_17M

SZA_V1_18  = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V1_18/all_SZA/V1_18_SZA.csv',index_col=0) # SZA V1 (2018/19)
SZA_V2_18  = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V2_18/all_SZA/V2_18_SZA.csv',index_col=0) # SZA V2 (2018/19)
SZA_V3_18M = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/V3_18/all_SZA/V3_18_SZA.csv',index_col=0) # SZA V3 (2018/19)
SZA_V3_18D = SZA_V3_18M

SZA_SIPEXII = pd.read_csv('/Users/ncp532/Documents/data/V1_17_APriori/SIPEXII/all_SZA/SIPEXII_SZA.csv',index_col=0) # SZA SIPEXII (2012)
                         
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
# Grid Box (Best Choice)
MERRA2_V1_17   = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V1_17_MERRA2.csv',   index_col=0)
MERRA2_V2_17   = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V2_17_MERRA2.csv',   index_col=0)
MERRA2_V3_17M  = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V3_17M_MERRA2.csv',  index_col=0)
MERRA2_V3_17D  = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V3_17D_MERRA2.csv',  index_col=0)

MERRA2_V1_18   = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V1_18_MERRA2.csv',   index_col=0) 
MERRA2_V2_18   = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V2_18_MERRA2.csv',   index_col=0)
MERRA2_V3_18M  = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V3_18M_MERRA2.csv',  index_col=0) 
MERRA2_V3_18D  = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/V3_18D_MERRA2.csv',  index_col=0)

MERRA2_SIPEXII = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/SIPEXII_MERRA2.csv', index_col=0) 

# # Grid Box (Actual Location)
# MERRA2_V1_17   = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/MERRA2_Actual_GridBox/V1_17_MERRA2.csv',   index_col=0)
# MERRA2_V2_17   = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/MERRA2_Actual_GridBox/V2_17_MERRA2.csv',   index_col=0)
# MERRA2_V3_17M  = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/MERRA2_Actual_GridBox/V3_17M_MERRA2.csv',  index_col=0)
# MERRA2_V3_17D  = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/MERRA2_Actual_GridBox/V3_17D_MERRA2.csv',  index_col=0)

# MERRA2_V1_18   = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/MERRA2_Actual_GridBox/V1_18_MERRA2.csv',   index_col=0) 
# MERRA2_V2_18   = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/MERRA2_Actual_GridBox/V2_18_MERRA2.csv',   index_col=0)
# MERRA2_V3_18M  = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/MERRA2_Actual_GridBox/V3_18M_MERRA2.csv',  index_col=0) 
# MERRA2_V3_18D  = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/MERRA2_Actual_GridBox/V3_18D_MERRA2.csv',  index_col=0)

# MERRA2_SIPEXII = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/MERRA2_Actual_GridBox/SIPEXII_MERRA2.csv', index_col=0) 

#------------------------------------------------------------------------------
# RENAME THE AEC DATAFRAME COLUMNS

AEC_V1_17.index   = ['0.1km','0.3km','0.5km','0.7km','0.9km',
                     '1.1km','1.3km','1.5km','1.7km','1.9km',
                     '2.1km','2.3km','2.5km','2.7km','2.9km',
                     '3.1km','3.3km','3.5km','3.7km','3.9km']

AEC_V2_17.index   = ['0.1km','0.3km','0.5km','0.7km','0.9km',
                     '1.1km','1.3km','1.5km','1.7km','1.9km',
                     '2.1km','2.3km','2.5km','2.7km','2.9km',
                     '3.1km','3.3km','3.5km','3.7km','3.9km']

AEC_V3_17M.index  = ['0.1km','0.3km','0.5km','0.7km','0.9km',
                     '1.1km','1.3km','1.5km','1.7km','1.9km',
                     '2.1km','2.3km','2.5km','2.7km','2.9km',
                     '3.1km','3.3km','3.5km','3.7km','3.9km']

AEC_V3_17D.index  = ['0.1km','0.3km','0.5km','0.7km','0.9km',
                     '1.1km','1.3km','1.5km','1.7km','1.9km',
                     '2.1km','2.3km','2.5km','2.7km','2.9km',
                     '3.1km','3.3km','3.5km','3.7km','3.9km']

AEC_V1_18.index   = ['0.1km','0.3km','0.5km','0.7km','0.9km',
                     '1.1km','1.3km','1.5km','1.7km','1.9km',
                     '2.1km','2.3km','2.5km','2.7km','2.9km',
                     '3.1km','3.3km','3.5km','3.7km','3.9km']

AEC_V2_18.index   = ['0.1km','0.3km','0.5km','0.7km','0.9km',
                     '1.1km','1.3km','1.5km','1.7km','1.9km',
                     '2.1km','2.3km','2.5km','2.7km','2.9km',
                     '3.1km','3.3km','3.5km','3.7km','3.9km']

AEC_V3_18M.index  = ['0.1km','0.3km','0.5km','0.7km','0.9km',
                     '1.1km','1.3km','1.5km','1.7km','1.9km',
                     '2.1km','2.3km','2.5km','2.7km','2.9km',
                     '3.1km','3.3km','3.5km','3.7km','3.9km']

AEC_V3_18D.index  = ['0.1km','0.3km','0.5km','0.7km','0.9km',
                     '1.1km','1.3km','1.5km','1.7km','1.9km',
                     '2.1km','2.3km','2.5km','2.7km','2.9km',
                     '3.1km','3.3km','3.5km','3.7km','3.9km']

AEC_SIPEXII.index = ['0.1km','0.3km','0.5km','0.7km','0.9km',
                     '1.1km','1.3km','1.5km','1.7km','1.9km',
                     '2.1km','2.3km','2.5km','2.7km','2.9km',
                     '3.1km','3.3km','3.5km','3.7km','3.9km']

#------------------------------------------------------------------------------
# FILTER THE BrO DATA FOR RELATIVE ERROR 

#----------------
# BrO (Retrieval)
#----------------
# Calculate the Relative Error (>=0.6)
Filter1_BrO = BrO_V1_17['err_surf_vmr'] / BrO_V1_17['surf_vmr(ppmv)']
Filter2_BrO = BrO_V2_17['err_surf_vmr'] / BrO_V2_17['surf_vmr(ppmv)']
Filter3_BrO = BrO_V3_17['err_surf_vmr'] / BrO_V3_17['surf_vmr(ppmv)']

Filter4_BrO = BrO_V1_18['err_surf_vmr'] / BrO_V1_18['surf_vmr(ppmv)']
Filter5_BrO = BrO_V2_18['err_surf_vmr'] / BrO_V2_18['surf_vmr(ppmv)']
Filter6_BrO = BrO_V3_18['err_surf_vmr'] / BrO_V3_18['surf_vmr(ppmv)']

Filter7_BrO = BrO_SIPEXII['err_surf_vmr'] / BrO_SIPEXII['surf_vmr(ppmv)']

# Apply the filter
V1_17F       = Filter1_BrO < 0.6
BrO_V1_17T   = BrO_V1_17[V1_17F]

V2_17F       = Filter2_BrO < 0.6
BrO_V2_17T   = BrO_V2_17[V2_17F]

V3_17F       = Filter3_BrO < 0.6
BrO_V3_17T   = BrO_V3_17[V3_17F]

V1_18F       = Filter4_BrO < 0.6
BrO_V1_18T   = BrO_V1_18[V1_18F]

V2_18F       = Filter5_BrO < 0.6
BrO_V2_18T   = BrO_V2_18[V2_18F]

V3_18F       = Filter6_BrO < 0.6
BrO_V3_18T   = BrO_V3_18[V3_18F]

SIPEXIIF     = Filter7_BrO < 0.6
BrO_SIPEXIIT = BrO_SIPEXII[SIPEXIIF]

# #----------------
# # BrO (VMR)
# #----------------
# # Calculate the Relative Error (>=0.6)
# Filter1_VMR = Err_VMR_V1_17 / BrO_VMR_V1_17
# Filter2_VMR = Err_VMR_V2_17 / BrO_VMR_V2_17
# Filter3_VMR = Err_VMR_V3_17 / BrO_VMR_V3_17

# Filter4_VMR = Err_VMR_V1_18 / BrO_VMR_V1_18
# Filter5_VMR = Err_VMR_V2_18 / BrO_VMR_V2_18
# Filter6_VMR = Err_VMR_V3_18 / BrO_VMR_V3_18

# Filter7_VMR = Err_VMR_SIPEXII / BrO_VMR_SIPEXII

# # Apply the filter
# V1_17F_VMR     = Filter1_VMR < 0.6
# BrO_VMR_V1_17T = BrO_VMR_V1_17[V1_17F_VMR]

# V2_17F_VMR     = Filter2_VMR < 0.6
# BrO_VMR_V2_17T = BrO_VMR_V2_17[V2_17F_VMR]

# V3_17F_VMR     = Filter3_VMR < 0.6
# BrO_VMR_V3_17T = BrO_VMR_V3_17[V3_17F_VMR]

# V1_18F_VMR     = Filter4_VMR < 0.6
# BrO_VMR_V1_18T = BrO_VMR_V1_18[V1_18F_VMR]

# V2_18F_VMR     = Filter5_VMR < 0.6
# BrO_VMR_V2_18T = BrO_VMR_V2_18[V2_18F_VMR]

# V3_18F_VMR     = Filter6_VMR < 0.6
# BrO_VMR_V3_18T = BrO_VMR_V3_18[V3_18F_VMR]

# SIPEXIIF_VMR     = Filter7_VMR < 0.6
# BrO_VMR_SIPEXIIT = BrO_VMR_SIPEXII[SIPEXIIF_VMR]

#------------------------------------------------------------------------------
# TRANSPOSE THE MAX-DOAS DATAFRAMES

# BrO (Retrieval)
BrO_V1_17T   = BrO_V1_17T
BrO_V2_17T   = BrO_V2_17T
BrO_V3_17MT  = BrO_V3_17T
BrO_V3_17DT  = BrO_V3_17T

BrO_V1_18T   = BrO_V1_18T
BrO_V2_18T   = BrO_V2_18T
BrO_V3_18MT  = BrO_V3_18T
BrO_V3_18DT  = BrO_V3_18T

BrO_SIPEXIIT = BrO_SIPEXII

# # BrO (VMR)
# BrO_VMR_V1_17T   = BrO_VMR_V1_17T.T
# BrO_VMR_V2_17T   = BrO_VMR_V2_17T.T
# BrO_VMR_V3_17MT  = BrO_VMR_V3_17T.T
# BrO_VMR_V3_17DT  = BrO_VMR_V3_17T.T

# BrO_VMR_V1_18T   = BrO_VMR_V1_18T.T
# BrO_VMR_V2_18T   = BrO_VMR_V2_18T.T
# BrO_VMR_V3_18MT  = BrO_VMR_V3_18T.T
# BrO_VMR_V3_18DT  = BrO_VMR_V3_18T.T

# BrO_VMR_SIPEXIIT = BrO_VMR_SIPEXII.T

# AEC
AEC_V1_17   = AEC_V1_17.T
AEC_V2_17   = AEC_V2_17.T
AEC_V3_17M  = AEC_V3_17M.T
AEC_V3_17D  = AEC_V3_17D.T

AEC_V1_18   = AEC_V1_18.T
AEC_V2_18   = AEC_V2_18.T
AEC_V3_18M  = AEC_V3_18M.T
AEC_V3_18D  = AEC_V3_18D.T

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

# BrO (Retrieval)
BrO_V1_17T.index   = (pd.to_datetime(BrO_V1_17T.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
BrO_V2_17T.index   = (pd.to_datetime(BrO_V2_17T.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
BrO_V3_17MT.index  = (pd.to_datetime(BrO_V3_17MT.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
BrO_V3_17DT.index  = (pd.to_datetime(BrO_V3_17DT.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

BrO_V1_18T.index   = (pd.to_datetime(BrO_V1_18T.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
BrO_V2_18T.index   = (pd.to_datetime(BrO_V2_18T.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
BrO_V3_18MT.index  = (pd.to_datetime(BrO_V3_18MT.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
BrO_V3_18DT.index  = (pd.to_datetime(BrO_V3_18DT.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

BrO_SIPEXIIT.index = (pd.to_datetime(BrO_SIPEXIIT.index, dayfirst=True) + timedelta(hours=8)) # SIPEXII timezone is UT+8

# # BrO (VMR)
# BrO_VMR_V1_17T.index   = (pd.to_datetime(BrO_VMR_V1_17T.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
# BrO_VMR_V2_17T.index   = (pd.to_datetime(BrO_VMR_V2_17T.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
# BrO_VMR_V3_17MT.index  = (pd.to_datetime(BrO_VMR_V3_17MT.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
# BrO_VMR_V3_17DT.index  = (pd.to_datetime(BrO_VMR_V3_17DT.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

# BrO_VMR_V1_18T.index   = (pd.to_datetime(BrO_VMR_V1_18T.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
# BrO_VMR_V2_18T.index   = (pd.to_datetime(BrO_VMR_V2_18T.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
# BrO_VMR_V3_18MT.index  = (pd.to_datetime(BrO_VMR_V3_18MT.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
# BrO_VMR_V3_18DT.index  = (pd.to_datetime(BrO_VMR_V3_18DT.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

# BrO_VMR_SIPEXIIT.index = (pd.to_datetime(BrO_VMR_SIPEXIIT.index, dayfirst=True) + timedelta(hours=8)) # SIPEXII timezone is UT+8

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

# NO2
NO2_V1_17.index   = (pd.to_datetime(NO2_V1_17.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
NO2_V2_17.index   = (pd.to_datetime(NO2_V2_17.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
NO2_V3_17M.index  = (pd.to_datetime(NO2_V3_17M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
NO2_V3_17D.index  = (pd.to_datetime(NO2_V3_17D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

NO2_V1_18.index   = (pd.to_datetime(NO2_V1_18.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
NO2_V2_18.index   = (pd.to_datetime(NO2_V2_18.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
NO2_V3_18M.index  = (pd.to_datetime(NO2_V3_18M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
NO2_V3_18D.index  = (pd.to_datetime(NO2_V3_18D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

#NO2_SIPEXII.index = (pd.to_datetime(NO2_SIPEXII.index, dayfirst=True) + timedelta(hours=8)) # SIPEXII timezone is UT+8

# BrO 1dSCD
BrOdSCD_V1_17.index   = (pd.to_datetime(BrOdSCD_V1_17.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
BrOdSCD_V2_17.index   = (pd.to_datetime(BrOdSCD_V2_17.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
BrOdSCD_V3_17M.index  = (pd.to_datetime(BrOdSCD_V3_17M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
BrOdSCD_V3_17D.index  = (pd.to_datetime(BrOdSCD_V3_17D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

BrOdSCD_V1_18.index   = (pd.to_datetime(BrOdSCD_V1_18.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
BrOdSCD_V2_18.index   = (pd.to_datetime(BrOdSCD_V2_18.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
BrOdSCD_V3_18M.index  = (pd.to_datetime(BrOdSCD_V3_18M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
BrOdSCD_V3_18D.index  = (pd.to_datetime(BrOdSCD_V3_18D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

#BrOdSCD_SIPEXII.index = (pd.to_datetime(BrOdSCD_SIPEXII.index, dayfirst=True) + timedelta(hours=8)) # SIPEXII timezone is UT+8

# SZA
SZA_V1_17.index   = (pd.to_datetime(SZA_V1_17.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
SZA_V2_17.index   = (pd.to_datetime(SZA_V2_17.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
SZA_V3_17M.index  = (pd.to_datetime(SZA_V3_17M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
SZA_V3_17D.index  = (pd.to_datetime(SZA_V3_17D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

SZA_V1_18.index   = (pd.to_datetime(SZA_V1_18.index,   dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7
SZA_V2_18.index   = (pd.to_datetime(SZA_V2_18.index,   dayfirst=True) + timedelta(hours=8)) # Casey timezone is UT+8
SZA_V3_18M.index  = (pd.to_datetime(SZA_V3_18M.index,  dayfirst=True) + timedelta(hours=5)) # Mawson timezone is UT+5
SZA_V3_18D.index  = (pd.to_datetime(SZA_V3_18D.index,  dayfirst=True) + timedelta(hours=7)) # Davis timezone is UT+7

SZA_SIPEXII.index = (pd.to_datetime(SZA_SIPEXII.index, dayfirst=True) + timedelta(hours=8)) # SIPEXII timezone is UT+8

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

#------------------------------------------------------------------------------
# Filter the SZA for outliers

# Define the filter
def hampel(vals_orig, k=11, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    #Make copy so original not edited
    vals=vals_orig.copy()    
    #Hampel Filter
    L= 1.4826
    rolling_median=vals.rolling(k).median()
    difference=np.abs(rolling_median-vals)
    median_abs_deviation=difference.rolling(k).median()
    threshold= t0 *L * median_abs_deviation
    outlier_idx=difference>threshold
    vals[outlier_idx]=np.nan
    return(vals)

# Apply the filter
SZA_1 = hampel(SZA_V1_17['SZA'])
SZA_2 = hampel(SZA_V2_17['SZA'])
SZA_3 = hampel(SZA_V3_17M['SZA'])
SZA_4 = hampel(SZA_V3_17D['SZA'])

SZA_5 = hampel(SZA_V1_18['SZA'])
SZA_6 = hampel(SZA_V2_18['SZA'])
SZA_7 = hampel(SZA_V3_18M['SZA'])
SZA_8 = hampel(SZA_V3_18D['SZA'])

SZA_9 = hampel(SZA_SIPEXII['SZA'])

#------------------------------------------------------------------------------
# FILTER THE BrO DATA FOR SZA (less than 75 degrees)

# Apply the filter
SZA_V1_17F   = SZA_1 < 75
SZA_V1_17T   = SZA_1[SZA_V1_17F]

SZA_V2_17F   = SZA_2 < 75
SZA_V2_17T   = SZA_2[SZA_V2_17F]

SZA_V3_17MF  = SZA_3 < 75
SZA_V3_17MT  = SZA_3[SZA_V3_17MF]

SZA_V3_17DF  = SZA_4 < 75
SZA_V3_17DT  = SZA_4[SZA_V3_17DF]


SZA_V1_18F   = SZA_5 < 75
SZA_V1_18T   = SZA_5[SZA_V1_18F]

SZA_V2_18F   = SZA_6 < 75
SZA_V2_18T   = SZA_6[SZA_V2_18F]

SZA_V3_18MF  = SZA_7 < 75
SZA_V3_18MT  = SZA_7[SZA_V3_18MF]

SZA_V3_18DF  = SZA_8 < 75
SZA_V3_18DT  = SZA_8[SZA_V3_18DF]


SZA_SIPEXIIF = SZA_9 < 75
SZA_SIPEXIIT = SZA_9[SZA_SIPEXIIF]

#------------------------------------------------------------------------------
# RESAMPLE THE SZA DATASETS TO 20-MINUTE TIME RESOLUTION

SZA_V1_17T   = SZA_V1_17T.resample('20T',   offset='10T').mean()
SZA_V2_17T   = SZA_V2_17T.resample('20T',   offset='10T').mean()
SZA_V3_17MT  = SZA_V3_17MT.resample('20T',  offset='10T').mean()
SZA_V3_17DT  = SZA_V3_17DT.resample('20T',  offset='10T').mean()

SZA_V1_18T   = SZA_V1_18T.resample('20T',   offset='10T').mean()
SZA_V2_18T   = SZA_V2_18T.resample('20T',   offset='10T').mean()
SZA_V3_18MT  = SZA_V3_18MT.resample('20T',  offset='10T').mean()
SZA_V3_18DT  = SZA_V3_18DT.resample('20T',  offset='10T').mean()

SZA_SIPEXIIT = SZA_SIPEXIIT.resample('20T', offset='10T').mean()

#------------------------------------------------------------------------------
# RESAMPLE THE NO2 DATASETS TO 20-MINUTE TIME RESOLUTION

NO2_V1_17   = NO2_V1_17.resample('20T',   offset='10T').mean()
NO2_V2_17   = NO2_V2_17.resample('20T',   offset='10T').mean()
NO2_V3_17M  = NO2_V3_17M.resample('20T',  offset='10T').mean()
NO2_V3_17D  = NO2_V3_17D.resample('20T',  offset='10T').mean()

NO2_V1_18   = NO2_V1_18.resample('20T',   offset='10T').mean()
NO2_V2_18   = NO2_V2_18.resample('20T',   offset='10T').mean()
NO2_V3_18M  = NO2_V3_18M.resample('20T',  offset='10T').mean()
NO2_V3_18D  = NO2_V3_18D.resample('20T',  offset='10T').mean()

#NO2_SIPEXII = NO2_SIPEXII.resample('20T', offset='10T').mean()

#------------------------------------------------------------------------------
# RESAMPLE THE BrOdSCD DATASETS TO 20-MINUTE TIME RESOLUTION

BrOdSCD_V1_17   = BrOdSCD_V1_17.resample('20T',   offset='10T').mean()
BrOdSCD_V2_17   = BrOdSCD_V2_17.resample('20T',   offset='10T').mean()
BrOdSCD_V3_17M  = BrOdSCD_V3_17M.resample('20T',  offset='10T').mean()
BrOdSCD_V3_17D  = BrOdSCD_V3_17D.resample('20T',  offset='10T').mean()

BrOdSCD_V1_18   = BrOdSCD_V1_18.resample('20T',   offset='10T').mean()
BrOdSCD_V2_18   = BrOdSCD_V2_18.resample('20T',   offset='10T').mean()
BrOdSCD_V3_18M  = BrOdSCD_V3_18M.resample('20T',  offset='10T').mean()
BrOdSCD_V3_18D  = BrOdSCD_V3_18D.resample('20T',  offset='10T').mean()

#BrOdSCD_SIPEXII = BrOdSCD_SIPEXII.resample('20T', offset='10T').mean()

#------------------------------------------------------------------------------
# RESAMPLE THE O3 DATASETS TO 20-MINUTE TIME RESOLUTION

O3_V1_17   = O3_V1_17.resample('20T',   offset='10T').mean()
O3_V2_17   = O3_V2_17.resample('20T',   offset='10T').mean()
O3_V3_17M  = O3_V3_17M.resample('20T',  offset='10T').mean()
O3_V3_17D  = O3_V3_17D.resample('20T',  offset='10T').mean()

O3_V1_18   = O3_V1_18.resample('20T',   offset='10T').mean()
O3_V2_18   = O3_V2_18.resample('20T',   offset='10T').mean()
O3_V3_18M  = O3_V3_18M.resample('20T',  offset='10T').mean()
O3_V3_18D  = O3_V3_18D.resample('20T',  offset='10T').mean()

O3_SIPEXII = O3_SIPEXII.resample('20T', offset='10T').mean()

#------------------------------------------------------------------------------
# COMBINE THE SZA DATAFRAMES

# BrO (Retrieval)
BrO_V1_17T   = pd.concat([BrO_V1_17T,   SZA_V1_17T],   axis=1, join='inner')
BrO_V2_17T   = pd.concat([BrO_V2_17T,   SZA_V2_17T],   axis=1, join='inner')
BrO_V3_17MT  = pd.concat([BrO_V3_17MT,  SZA_V3_17MT],  axis=1, join='inner')
BrO_V3_17DT  = pd.concat([BrO_V3_17DT,  SZA_V3_17DT],  axis=1, join='inner')

BrO_V1_18T   = pd.concat([BrO_V1_18T,   SZA_V1_18T],   axis=1, join='inner')
BrO_V2_18T   = pd.concat([BrO_V2_18T,   SZA_V2_18T],   axis=1, join='inner')
BrO_V3_18MT  = pd.concat([BrO_V3_18MT,  SZA_V3_18MT],  axis=1, join='inner')
BrO_V3_18DT  = pd.concat([BrO_V3_18DT,  SZA_V3_18DT],  axis=1, join='inner')

BrO_SIPEXIIT = pd.concat([BrO_SIPEXIIT, SZA_SIPEXIIT], axis=1, join='inner')

# Drop nan values
BrO_V1_17T   = BrO_V1_17T.dropna()
BrO_V2_17T   = BrO_V2_17T.dropna()
BrO_V3_17MT  = BrO_V3_17MT.dropna()
BrO_V3_17DT  = BrO_V3_17DT.dropna()

BrO_V1_18T   = BrO_V1_18T.dropna()
BrO_V2_18T   = BrO_V2_18T.dropna()
BrO_V3_18MT  = BrO_V3_18MT.dropna()
BrO_V3_18DT  = BrO_V3_18DT.dropna()

BrO_SIPEXIIT = BrO_SIPEXIIT.dropna()

#------------------------------------------------------------------------------
# COMBINE BrO WITH THE OTHER DATAFRAMES

# BrO & NO2
BrO_V1_17T   = pd.concat([BrO_V1_17T,   NO2_V1_17],   axis=1, join='inner')
BrO_V2_17T   = pd.concat([BrO_V2_17T,   NO2_V2_17],   axis=1, join='inner')
BrO_V3_17MT  = pd.concat([BrO_V3_17MT,  NO2_V3_17M],  axis=1, join='inner')
BrO_V3_17DT  = pd.concat([BrO_V3_17DT,  NO2_V3_17D],  axis=1, join='inner')

BrO_V1_18T   = pd.concat([BrO_V1_18T,   NO2_V1_18],   axis=1, join='inner')
BrO_V2_18T   = pd.concat([BrO_V2_18T,   NO2_V2_18],   axis=1, join='inner')
BrO_V3_18MT  = pd.concat([BrO_V3_18MT,  NO2_V3_18M],  axis=1, join='inner')
BrO_V3_18DT  = pd.concat([BrO_V3_18DT,  NO2_V3_18D],  axis=1, join='inner')

#BrO_SIPEXIIT = pd.concat([BrO_SIPEXIIT, NO2_SIPEXII], axis=1, join='inner')

# BrO & BrOdSCD
BrO_V1_17T   = pd.concat([BrO_V1_17T,   BrOdSCD_V1_17],   axis=1, join='inner')
BrO_V2_17T   = pd.concat([BrO_V2_17T,   BrOdSCD_V2_17],   axis=1, join='inner')
BrO_V3_17MT  = pd.concat([BrO_V3_17MT,  BrOdSCD_V3_17M],  axis=1, join='inner')
BrO_V3_17DT  = pd.concat([BrO_V3_17DT,  BrOdSCD_V3_17D],  axis=1, join='inner')

BrO_V1_18T   = pd.concat([BrO_V1_18T,   BrOdSCD_V1_18],   axis=1, join='inner')
BrO_V2_18T   = pd.concat([BrO_V2_18T,   BrOdSCD_V2_18],   axis=1, join='inner')
BrO_V3_18MT  = pd.concat([BrO_V3_18MT,  BrOdSCD_V3_18M],  axis=1, join='inner')
BrO_V3_18DT  = pd.concat([BrO_V3_18DT,  BrOdSCD_V3_18D],  axis=1, join='inner')

#BrO_SIPEXIIT = pd.concat([BrO_SIPEXIIT, BrOdSCD_SIPEXII], axis=1, join='inner')

# BrO & O3
BrO_V1_17T   = pd.concat([BrO_V1_17T,   O3_V1_17],   axis=1, join='inner')
BrO_V2_17T   = pd.concat([BrO_V2_17T,   O3_V2_17],   axis=1, join='inner')
BrO_V3_17MT  = pd.concat([BrO_V3_17MT,  O3_V3_17M],  axis=1, join='inner')
BrO_V3_17DT  = pd.concat([BrO_V3_17DT,  O3_V3_17D],  axis=1, join='inner')

BrO_V1_18T   = pd.concat([BrO_V1_18T,   O3_V1_18],   axis=1, join='inner')
BrO_V2_18T   = pd.concat([BrO_V2_18T,   O3_V2_18],   axis=1, join='inner')
BrO_V3_18MT  = pd.concat([BrO_V3_18MT,  O3_V3_18M],  axis=1, join='inner')
BrO_V3_18DT  = pd.concat([BrO_V3_18DT,  O3_V3_18D],  axis=1, join='inner')

BrO_SIPEXIIT = pd.concat([BrO_SIPEXIIT, O3_SIPEXII], axis=1, join='inner')

# BrO & AEC
BrO_V1_17T   = pd.concat([BrO_V1_17T,   AEC_V1_17],   axis=1, join='inner')
BrO_V2_17T   = pd.concat([BrO_V2_17T,   AEC_V2_17],   axis=1, join='inner')
BrO_V3_17MT  = pd.concat([BrO_V3_17MT,  AEC_V3_17M],  axis=1, join='inner')
BrO_V3_17DT  = pd.concat([BrO_V3_17DT,  AEC_V3_17D],  axis=1, join='inner')

BrO_V1_18T   = pd.concat([BrO_V1_18T,   AEC_V1_18],   axis=1, join='inner')
BrO_V2_18T   = pd.concat([BrO_V2_18T,   AEC_V2_18],   axis=1, join='inner')
BrO_V3_18MT  = pd.concat([BrO_V3_18MT,  AEC_V3_18M],  axis=1, join='inner')
BrO_V3_18DT  = pd.concat([BrO_V3_18DT,  AEC_V3_18D],  axis=1, join='inner')

BrO_SIPEXIIT = pd.concat([BrO_SIPEXIIT, AEC_SIPEXII], axis=1, join='inner')

# #------------------------------------------------------------------------------
# # SEASONAL END DATE SCREEN

# # Set the filter
# F_BrOdSCD_V1_17   = BrO_V1_17T['336-357.SlCol(BrO)']
# F_BrOdSCD_V2_17   = BrO_V2_17T['336-357.SlCol(BrO)']
# F_BrOdSCD_V3_17M  = BrO_V3_17MT['336-357.SlCol(BrO)']
# F_BrOdSCD_V3_17D  = BrO_V3_17DT['336-357.SlCol(BrO)']

# F_BrOdSCD_V1_18   = BrO_V1_18T['336-357.SlCol(BrO)']
# F_BrOdSCD_V2_18   = BrO_V2_18T['336-357.SlCol(BrO)']
# F_BrOdSCD_V3_18M  = BrO_V3_18MT['336-357.SlCol(BrO)']
# F_BrOdSCD_V3_18D  = BrO_V3_18DT['336-357.SlCol(BrO)']

# #F_BrO_SIPEXII = BrO_SIPEXIIT['336-357.SlCol(BrO)']

# # Apply the filter (Remove values when BrO 1degree dSCD <5e13 molecules/cm2)
# SeasonalF_V1_17   = F_BrOdSCD_V1_17    > 5e13 
# BrO_V1_17T        = BrO_V1_17T[SeasonalF_V1_17]

# SeasonalF_V2_17   = F_BrOdSCD_V2_17    > 5e13 
# BrO_V2_17T        = BrO_V2_17T[SeasonalF_V2_17]

# SeasonalF_V3_17M  = F_BrOdSCD_V3_17M   > 5e13 
# BrO_V3_17MT       = BrO_V3_17MT[SeasonalF_V3_17M]

# SeasonalF_V3_17D  = F_BrOdSCD_V3_17D   > 5e13 
# BrO_V3_17DT       = BrO_V3_17DT[SeasonalF_V3_17D]

# SeasonalF_V1_18   = F_BrOdSCD_V1_18    > 5e13 
# BrO_V1_18T        = BrO_V1_18T[SeasonalF_V1_18]

# SeasonalF_V2_18   = F_BrOdSCD_V2_18    > 5e13 
# BrO_V2_18T        = BrO_V2_18T[SeasonalF_V2_18]

# SeasonalF_V3_18M  = F_BrOdSCD_V3_18M   > 5e13 
# BrO_V3_18MT       = BrO_V3_18MT[SeasonalF_V3_18M]

# SeasonalF_V3_18D  = F_BrOdSCD_V3_18D   > 5e13 
# BrO_V3_18DT       = BrO_V3_18DT[SeasonalF_V3_18D]

# #SeasonalF_SIPEXII = F_BrOdSCD_SIPEXII > 5e13 
# #BrO_SIPEXIIT      = BrO_SIPEXIIT[SeasonalF_SIPEXII]

#------------------------------------------------------------------------------
# OZONE SCREEN

# Set the filter
F_O3_V1_17   = BrO_V1_17T['O3_(ppb)']
F_O3_V2_17   = BrO_V2_17T['O3_(ppb)']
F_O3_V3_17M  = BrO_V3_17MT['O3_(ppb)']
F_O3_V3_17D  = BrO_V3_17DT['O3_(ppb)']

F_O3_V1_18   = BrO_V1_18T['O3_(ppb)']
F_O3_V2_18   = BrO_V2_18T['O3_(ppb)']
F_O3_V3_18M  = BrO_V3_18MT['O3_(ppb)']
F_O3_V3_18D  = BrO_V3_18DT['O3_(ppb)']

F_O3_SIPEXII = BrO_SIPEXIIT['O3_(ppb)']

# Apply the filter (Remove values when O3 <2 ppb)
OzoneF_V1_17   = F_O3_V1_17   > 2
BrO_V1_17T     = BrO_V1_17T[OzoneF_V1_17]

OzoneF_V2_17   = F_O3_V2_17   > 2
BrO_V2_17T     = BrO_V2_17T[OzoneF_V2_17]

OzoneF_V3_17M  = F_O3_V3_17M  > 2
BrO_V3_17MT    = BrO_V3_17MT[OzoneF_V3_17M]

OzoneF_V3_17D  = F_O3_V3_17D  > 2
BrO_V3_17DT    = BrO_V3_17DT[OzoneF_V3_17D]

OzoneF_V1_18   = F_O3_V1_18   > 2
BrO_V1_18T     = BrO_V1_18T[OzoneF_V1_18]

OzoneF_V2_18   = F_O3_V2_18   > 2
BrO_V2_18T     = BrO_V2_18T[OzoneF_V2_18]

OzoneF_V3_18M  = F_O3_V3_18M  > 2
BrO_V3_18MT    = BrO_V3_18MT[OzoneF_V3_18M]

OzoneF_V3_18D  = F_O3_V3_18D  > 2
BrO_V3_18DT    = BrO_V3_18DT[OzoneF_V3_18D]

OzoneF_SIPEXII = F_O3_SIPEXII > 2
BrO_SIPEXIIT   = BrO_SIPEXIIT[OzoneF_SIPEXII]

#------------------------------------------------------------------------------
# POLLUTION SCREEN

# Set the filter
F_NO2_V1_17   = BrO_V1_17T['338-370.SlCol(NO2t)']
F_NO2_V2_17   = BrO_V2_17T['338-370.SlCol(NO2t)']
F_NO2_V3_17M  = BrO_V3_17MT['338-370.SlCol(NO2t)']
F_NO2_V3_17D  = BrO_V3_17DT['338-370.SlCol(NO2t)']

F_NO2_V1_18   = BrO_V1_18T['338-370.SlCol(NO2t)']
F_NO2_V2_18   = BrO_V2_18T['338-370.SlCol(NO2t)']
F_NO2_V3_18M  = BrO_V3_18MT['338-370.SlCol(NO2t)']
F_NO2_V3_18D  = BrO_V3_18DT['338-370.SlCol(NO2t)']

#F_NO2_SIPEXII = BrO_SIPEXIIT['338-370.SlCol(NO2t)']

# Apply the filter (Remove values when NO2 2degree dSCD >5e15 molecules/cm2)
PollutionF_V1_17  = F_NO2_V1_17    <= 5e15 
BrO_V1_17T        = BrO_V1_17T[PollutionF_V1_17]

PollutionF_V2_17  = F_NO2_V2_17    <= 5e15 
BrO_V2_17T        = BrO_V2_17T[PollutionF_V2_17]

PollutionF_V3_17M = F_NO2_V3_17M   <= 5e15 
BrO_V3_17MT       = BrO_V3_17MT[PollutionF_V3_17M]

PollutionF_V3_17D = F_NO2_V3_17D   <= 5e15 
BrO_V3_17DT       = BrO_V3_17DT[PollutionF_V3_17D]

PollutionF_V1_18  = F_NO2_V1_18    <= 5e15 
BrO_V1_18T        = BrO_V1_18T[PollutionF_V1_18]

PollutionF_V2_18  = F_NO2_V2_18    <= 5e15 
BrO_V2_18T        = BrO_V2_18T[PollutionF_V2_18]

PollutionF_V3_18M = F_NO2_V3_18M   <= 5e15 
BrO_V3_18MT       = BrO_V3_18MT[PollutionF_V3_18M]

PollutionF_V3_18D = F_NO2_V3_18D   <= 5e15 
BrO_V3_18DT       = BrO_V3_18DT[PollutionF_V3_18D]

#PollutionF_SIPEXII = F_NO2_SIPEXII <= 5e15 
#BrO_SIPEXIIT       = BrO_SIPEXIIT[PollutionF_SIPEXII]

#------------------------------------------------------------------------------
# REPLACE ERRONEOUS VALUES WITH NAN

#--------------
# MAX-DOAS
#--------------

# BrO (Retrieval)
BrO_V1_17T   = BrO_V1_17T.replace(-9999.000000,  np.nan)
BrO_V2_17T   = BrO_V2_17T.replace(-9999.000000,  np.nan)
BrO_V3_17MT  = BrO_V3_17MT.replace(-9999.000000, np.nan)
BrO_V3_17DT  = BrO_V3_17DT.replace(-9999.000000, np.nan)

BrO_V1_18T   = BrO_V1_18T.replace(-9999.000000,  np.nan)
BrO_V2_18T   = BrO_V2_18T.replace(-9999.000000,  np.nan)
BrO_V3_18MT  = BrO_V3_18MT.replace(-9999.000000, np.nan)
BrO_V3_18DT  = BrO_V3_18DT.replace(-9999.000000, np.nan)

BrO_SIPEXIIT = BrO_SIPEXIIT.replace(9.67e-05, np.nan)
BrO_SIPEXIIT = BrO_SIPEXIIT.replace(7.67e-06, np.nan)
BrO_SIPEXIIT = BrO_SIPEXIIT.replace(7.67e-07, np.nan)
BrO_SIPEXIIT.loc[BrO_SIPEXIIT.isnull().any(axis=1), :] = np.nan # if any element in the row is nan, set the whole row to nan
BrO_SIPEXIIT = BrO_SIPEXIIT.replace(-9999.000000, np.nan)

# # BrO (VMR)
# BrO_VMR_V1_17T   = BrO_VMR_V1_17T.replace(-9999.000000,  np.nan)
# BrO_VMR_V2_17T   = BrO_VMR_V2_17T.replace(-9999.000000,  np.nan)
# BrO_VMR_V3_17MT  = BrO_VMR_V3_17MT.replace(-9999.000000, np.nan)
# BrO_VMR_V3_17DT  = BrO_VMR_V3_17DT.replace(-9999.000000, np.nan)

# BrO_VMR_V1_18T   = BrO_VMR_V1_18T.replace(-9999.000000,  np.nan)
# BrO_VMR_V2_18T   = BrO_VMR_V2_18T.replace(-9999.000000,  np.nan)
# BrO_VMR_V3_18MT  = BrO_VMR_V3_18MT.replace(-9999.000000, np.nan)
# BrO_VMR_V3_18DT  = BrO_VMR_V3_18DT.replace(-9999.000000, np.nan)

# BrO_VMR_SIPEXIIT = BrO_VMR_SIPEXIIT.replace(9.67e-05,np.nan)
# BrO_VMR_SIPEXIIT = BrO_VMR_SIPEXIIT.replace(7.67e-06,np.nan)
# BrO_VMR_SIPEXIIT = BrO_VMR_SIPEXIIT.replace(7.67e-07,np.nan)
# BrO_VMR_SIPEXIIT.loc[BrO_VMR_SIPEXIIT.isnull().any(axis=1), :] = np.nan # if any element in the row is nan, set the whole row to nan
# BrO_VMR_SIPEXIIT = BrO_VMR_SIPEXIIT.replace(-9999.000000, np.nan)

#------------------------------------------------------------------------------
# CONVERT THE MAX-DOAS & MET DATASETS A 1-HOUR TIME RESOLUTION

#--------------
# MAX-DOAS
#--------------

# BrO (Retrieval)
BrO_V1_17T   = BrO_V1_17T.resample('60T').mean()
BrO_V2_17T   = BrO_V2_17T.resample('60T').mean()
BrO_V3_17MT  = BrO_V3_17MT.resample('60T').mean()
BrO_V3_17DT  = BrO_V3_17DT.resample('60T').mean()

BrO_V1_18T   = BrO_V1_18T.resample('60T').mean()
BrO_V2_18T   = BrO_V2_18T.resample('60T').mean()
BrO_V3_18MT  = BrO_V3_18MT.resample('60T').mean()
BrO_V3_18DT  = BrO_V3_18DT.resample('60T').mean()

BrO_SIPEXIIT = BrO_SIPEXIIT.resample('60T').mean()

# # BrO (VMR)
# BrO_VMR_V1_17T   = BrO_VMR_V1_17T.resample('60T').mean()
# BrO_VMR_V2_17T   = BrO_VMR_V2_17T.resample('60T').mean()
# BrO_VMR_V3_17MT  = BrO_VMR_V3_17MT.resample('60T').mean()
# BrO_VMR_V3_17DT  = BrO_VMR_V3_17DT.resample('60T').mean()

# BrO_VMR_V1_18T   = BrO_VMR_V1_18T.resample('60T').mean()
# BrO_VMR_V2_18T   = BrO_VMR_V2_18T.resample('60T').mean()
# BrO_VMR_V3_18MT  = BrO_VMR_V3_18MT.resample('60T').mean()
# BrO_VMR_V3_18DT  = BrO_VMR_V3_18DT.resample('60T').mean()

# BrO_VMR_SIPEXIIT = BrO_VMR_SIPEXIIT.resample('60T').mean()

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

# Potential temperature differential in lowest 100m (K)
MERRA2_V1_17['PTDif100m']   = MERRA2_V1_17['VPT100m']   - MERRA2_V1_17['VPT2m']
MERRA2_V2_17['PTDif100m']   = MERRA2_V2_17['VPT100m']   - MERRA2_V2_17['VPT2m']
MERRA2_V3_17M['PTDif100m']  = MERRA2_V3_17M['VPT100m']  - MERRA2_V3_17M['VPT2m']
MERRA2_V3_17D['PTDif100m']  = MERRA2_V3_17D['VPT100m']  - MERRA2_V3_17D['VPT2m']

MERRA2_V1_18['PTDif100m']   = MERRA2_V1_18['VPT100m']   - MERRA2_V1_18['VPT2m']
MERRA2_V2_18['PTDif100m']   = MERRA2_V2_18['VPT100m']   - MERRA2_V2_18['VPT2m']
MERRA2_V3_18M['PTDif100m']  = MERRA2_V3_18M['VPT100m']  - MERRA2_V3_18M['VPT2m']
MERRA2_V3_18D['PTDif100m']  = MERRA2_V3_18D['VPT100m']  - MERRA2_V3_18D['VPT2m']

MERRA2_SIPEXII['PTDif100m'] = MERRA2_SIPEXII['VPT100m'] - MERRA2_SIPEXII['VPT2m']

#------------------------------------------------------------------------------
#  CALCULATE THE POTENTIAL TEMPERATURE DIFFERENTIAL IN LOWEST 1000m (k)

# Potential temperature differential in lowest 1000m (K)
MERRA2_V1_17['PTDif1000m']   = MERRA2_V1_17['VPT1000m']   - MERRA2_V1_17['VPT2m']
MERRA2_V2_17['PTDif1000m']   = MERRA2_V2_17['VPT1000m']   - MERRA2_V2_17['VPT2m']
MERRA2_V3_17M['PTDif1000m']  = MERRA2_V3_17M['VPT1000m']  - MERRA2_V3_17M['VPT2m']
MERRA2_V3_17D['PTDif1000m']  = MERRA2_V3_17D['VPT1000m']  - MERRA2_V3_17D['VPT2m']

MERRA2_V1_18['PTDif1000m']   = MERRA2_V1_18['VPT1000m']   - MERRA2_V1_18['VPT2m']
MERRA2_V2_18['PTDif1000m']   = MERRA2_V2_18['VPT1000m']   - MERRA2_V2_18['VPT2m']
MERRA2_V3_18M['PTDif1000m']  = MERRA2_V3_18M['VPT1000m']  - MERRA2_V3_18M['VPT2m']
MERRA2_V3_18D['PTDif1000m']  = MERRA2_V3_18D['VPT1000m']  - MERRA2_V3_18D['VPT2m']

MERRA2_SIPEXII['PTDif1000m'] = MERRA2_SIPEXII['VPT1000m'] - MERRA2_SIPEXII['VPT2m']

#------------------------------------------------------------------------------
# Filter the datasets based on the date

#-----------------------------
# V1_17 Davis (14-22 Nov 2017)
#-----------------------------
start_date   = '2017-11-14'
end_date     = '2017-11-23'
# BrO (Retrieval)
Davis        = (BrO_V1_17T.index >= start_date) & (BrO_V1_17T.index < end_date)
V1_17_BrO    = BrO_V1_17T[Davis]
# # BrO (VMR)
# Davis        = (BrO_VMR_V1_17T.index >= start_date) & (BrO_VMR_V1_17T.index < end_date)
# V1_17_BrO_VMR = BrO_VMR_V1_17T[Davis]
# Met
Davis        = (Met_V1_17.index >= start_date) & (Met_V1_17.index < end_date)
V1_17_Met    = Met_V1_17[Davis]
# MERRA2
Davis        = (MERRA2_V1_17.index >= start_date) & (MERRA2_V1_17.index < end_date)
V1_17_MERRA2 = MERRA2_V1_17[Davis]

#-----------------------------
# V2_17 Casey (21-22 Dec 2017 and 26 Dec 2017 - 5 Jan 2018)
#-----------------------------
start_date1 = '2017-12-21'
end_date1 = '2017-12-23'
start_date2 = '2017-12-26'
end_date2 = '2018-01-6'
# BrO (Retrieval)
Casey1       = (BrO_V2_17T.index >= start_date1) & (BrO_V2_17T.index < end_date1)
Casey2       = (BrO_V2_17T.index >= start_date2) & (BrO_V2_17T.index < end_date2)
V2_17_BrO1   = BrO_V2_17T[Casey1]
V2_17_BrO2   = BrO_V2_17T[Casey2]
V2_17_BrO    = pd.concat([V2_17_BrO1,V2_17_BrO2], axis =0)
# # BrO (VMR)
# Casey1       = (BrO_VMR_V2_17T.index >= start_date1) & (BrO_VMR_V2_17T.index < end_date1)
# Casey2       = (BrO_VMR_V2_17T.index >= start_date2) & (BrO_VMR_V2_17T.index < end_date2)
# V2_17_BrO_VMR1 = BrO_VMR_V2_17T[Casey1]
# V2_17_BrO_VMR2 = BrO_VMR_V2_17T[Casey2]
# V2_17_BrO_VMR  = pd.concat([V2_17_BrO_VMR1,V2_17_BrO_VMR2], axis =0)
# Met
Casey1       = (Met_V2_17.index >= start_date1) & (Met_V2_17.index < end_date1)
Casey2       = (Met_V2_17.index >= start_date2) & (Met_V2_17.index < end_date2)
V2_17_Met1   = Met_V2_17[Casey1]
V2_17_Met2   = Met_V2_17[Casey2]
V2_17_Met    = pd.concat([V2_17_Met1,V2_17_Met2], axis =0)
# MERRA2
Casey1       = (MERRA2_V2_17.index >= start_date1) & (MERRA2_V2_17.index < end_date1)
Casey2       = (MERRA2_V2_17.index >= start_date2) & (MERRA2_V2_17.index < end_date2)
V2_17_MERRA21= MERRA2_V2_17[Casey1]
V2_17_MERRA22= MERRA2_V2_17[Casey2]
V2_17_MERRA2 = pd.concat([V2_17_MERRA21,V2_17_MERRA22], axis =0)

#-----------------------------
# V3_17 Mawson (1-17 Feb 2018)
#-----------------------------
start_date    = '2018-02-01'
end_date      = '2018-02-18'
# BrO (Retrieval)
Mawson        = (BrO_V3_17MT.index >= start_date) & (BrO_V3_17MT.index < end_date)
V3_17_BrOM    = BrO_V3_17MT[Mawson]
# # BrO (VMR)
# Mawson         = (BrO_VMR_V3_17MT.index >= start_date) & (BrO_VMR_V3_17MT.index < end_date)
# V3_17_BrO_VMRM = BrO_VMR_V3_17MT[Mawson]
# Met
Mawson        = (Met_V3_17M.index >= start_date) & (Met_V3_17M.index < end_date)
V3_17_MetM    = Met_V3_17M[Mawson]
# MERRA2
Mawson        = (MERRA2_V3_17M.index >= start_date) & (MERRA2_V3_17M.index < end_date)
V3_17_MERRA2M = MERRA2_V3_17M[Mawson]

#-----------------------------
# V3_17 Davis (27-30 Jan 2018 and 19-21 Feb 2018)
#-----------------------------
start_date1   = '2018-01-27'
end_date1     = '2018-01-31'
start_date2   = '2018-02-19'
end_date2     = '2018-02-22'
# BrO (Retrieval)
Davis1        = (BrO_V3_17DT.index >= start_date1) & (BrO_V3_17DT.index < end_date1)
Davis2        = (BrO_V3_17DT.index >= start_date2) & (BrO_V3_17DT.index < end_date2)
V3_17_BrO1    = BrO_V3_17DT[Davis1]
V3_17_BrO2    = BrO_V3_17DT[Davis2]
V3_17_BrOD    = pd.concat([V3_17_BrO1,V3_17_BrO2], axis =0)
# # BrO (VMR)
# Davis1        = (BrO_VMR_V3_17DT.index >= start_date1) & (BrO_VMR_V3_17DT.index < end_date1)
# Davis2        = (BrO_VMR_V3_17DT.index >= start_date2) & (BrO_VMR_V3_17DT.index < end_date2)
# V3_17_BrO_VMR1 = BrO_VMR_V3_17DT[Davis1]
# V3_17_BrO_VMR2 = BrO_VMR_V3_17DT[Davis2]
# V3_17_BrO_VMRD = pd.concat([V3_17_BrO_VMR1,V3_17_BrO_VMR2], axis =0)
# Met
Davis1        = (Met_V3_17D.index >= start_date1) & (Met_V3_17D.index < end_date1)
Davis2        = (Met_V3_17D.index >= start_date2) & (Met_V3_17D.index < end_date2)
V3_17_Met1    = Met_V3_17D[Davis1]
V3_17_Met2    = Met_V3_17D[Davis2]
V3_17_MetD    = pd.concat([V3_17_Met1,V3_17_Met2], axis =0)
# MERRA2
Davis1        = (MERRA2_V3_17D.index >= start_date1) & (MERRA2_V3_17D.index < end_date1)
Davis2        = (MERRA2_V3_17D.index >= start_date2) & (MERRA2_V3_17D.index < end_date2)
V3_17_MERRA21 = MERRA2_V3_17D[Davis1]
V3_17_MERRA22 = MERRA2_V3_17D[Davis2]
V3_17_MERRA2D = pd.concat([V3_17_MERRA21,V3_17_MERRA22], axis =0)

#-----------------------------
# V1_18 Davis (7-15 Nov 2018)
#-----------------------------
start_date   = '2018-11-07'
end_date     = '2018-11-16'
# BrO (Retrieval)
Davis        = (BrO_V1_18T.index >= start_date) & (BrO_V1_18T.index < end_date)
V1_18_BrO    = BrO_V1_18T[Davis]
# # BrO (VMR)
# Davis         = (BrO_VMR_V1_18T.index >= start_date) & (BrO_VMR_V1_18T.index < end_date)
# V1_18_BrO_VMR = BrO_VMR_V1_18T[Davis]
# Met
Davis        = (Met_V1_18.index >= start_date) & (Met_V1_18.index < end_date)
V1_18_Met    = Met_V1_18[Davis]
# MERRA2
Davis        = (MERRA2_V1_18.index >= start_date) & (MERRA2_V1_18.index < end_date)
V1_18_MERRA2 = MERRA2_V1_18[Davis]

#-----------------------------
# V2_18 Casey (15-30 Dec 2018)
#-----------------------------
start_date   = '2018-12-15'
end_date     = '2018-12-31'
# BrO (Retrieval)
Casey        = (BrO_V2_18T.index >= start_date) & (BrO_V2_18T.index < end_date)
V2_18_BrO    = BrO_V2_18T[Casey]
# # BrO (VMR)
# Casey         = (BrO_VMR_V2_18T.index >= start_date) & (BrO_VMR_V2_18T.index < end_date)
# V2_18_BrO_VMR = BrO_VMR_V2_18T[Casey]
# Met
Casey        = (Met_V2_18.index >= start_date) & (Met_V2_18.index < end_date)
V2_18_Met    = Met_V2_18[Casey]
# MERRA2
Casey        = (MERRA2_V2_18.index >= start_date) & (MERRA2_V2_18.index < end_date)
V2_18_MERRA2 = MERRA2_V2_18[Casey]

#-----------------------------
# V3_18 Mawson (30 Jan - 9 Feb 2019)
#-----------------------------
start_date    = '2019-01-30'
end_date      = '2019-02-10'
# BrO (Retrieval)
Mawson        = (BrO_V3_18MT.index >= start_date) & (BrO_V3_18MT.index < end_date)
V3_18_BrOM    = BrO_V3_18MT[Mawson]
# # BrO (VMR)
# Mawson         = (BrO_VMR_V3_18MT.index >= start_date) & (BrO_VMR_V3_18MT.index < end_date)
# V3_18_BrO_VMRM = BrO_VMR_V3_18MT[Mawson]
# Met
Mawson        = (Met_V3_18M.index >= start_date) & (Met_V3_18M.index < end_date)
V3_18_MetM    = Met_V3_18M[Mawson]
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
# BrO (Retrieval)
Davis1        = (BrO_V3_18DT.index >= start_date1) & (BrO_V3_18DT.index < end_date1)
Davis2        = (BrO_V3_18DT.index >= start_date2) & (BrO_V3_18DT.index < end_date2)
V3_18_BrO1    = BrO_V3_18DT[Davis1]
V3_18_BrO2    = BrO_V3_18DT[Davis2]
V3_18_BrOD    = pd.concat([V3_18_BrO1,V3_18_BrO2], axis =0)
# # BrO (VMR)
# Davis1         = (BrO_VMR_V3_18DT.index >= start_date1) & (BrO_VMR_V3_18DT.index < end_date1)
# Davis2         = (BrO_VMR_V3_18DT.index >= start_date2) & (BrO_VMR_V3_18DT.index < end_date2)
# V3_18_BrO_VMR1 = BrO_VMR_V3_18DT[Davis1]
# V3_18_BrO_VMR2 = BrO_VMR_V3_18DT[Davis2]
# V3_18_BrO_VMRD = pd.concat([V3_18_BrO_VMR1,V3_18_BrO_VMR2], axis =0)
# Met
Davis1        = (Met_V3_18D.index >= start_date1) & (Met_V3_18D.index < end_date1)
Davis2        = (Met_V3_18D.index >= start_date2) & (Met_V3_18D.index < end_date2)
V3_18_Met1    = Met_V3_18D[Davis1]
V3_18_Met2    = Met_V3_18D[Davis2]
V3_18_MetD    = pd.concat([V3_18_Met1,V3_18_Met2], axis =0)
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
# BrO (Retrieval)
SIPEX          = (BrO_SIPEXIIT.index >= start_date) & (BrO_SIPEXIIT.index < end_date)
SIPEXII_BrO    = BrO_SIPEXIIT[SIPEX]
# # BrO (VMR)
# SIPEX           = (BrO_VMR_SIPEXIIT.index >= start_date) & (BrO_VMR_SIPEXIIT.index < end_date)
# SIPEXII_BrO_VMR = BrO_VMR_SIPEXIIT[SIPEX]
# Met
SIPEX          = (Met_SIPEXII.index >= start_date) & (Met_SIPEXII.index < end_date)
SIPEXII_Met    = Met_SIPEXII[SIPEX]
# MERRA2
SIPEX          = (MERRA2_SIPEXII.index >= start_date) & (MERRA2_SIPEXII.index < end_date)
SIPEXII_MERRA2 = MERRA2_SIPEXII[SIPEX]

#------------------------------------------------------------------------------
# COMBINE THE DATAFRAMES FOR EACH VOYAGE INTO A SINGLE DATAFRAME

# BrO (Retrieval)
BrO_All    = pd.concat([V1_17_BrO,V2_17_BrO,V3_17_BrOM,V3_17_BrOD,V1_18_BrO,V2_18_BrO,V3_18_BrOM,V3_18_BrOD],axis=0) # Without SIPEXII
#BrO_All    = pd.concat([V1_17_BrO,V2_17_BrO,V1_18_BrO,V2_18_BrO],axis=0) # Without V3 & SIPEXII
#BrO_All    = pd.concat([SIPEXII_BrO,V1_17_BrO,V2_17_BrO,V3_17_BrOM,V3_17_BrOD,V1_18_BrO,V2_18_BrO,V3_18_BrOM,V3_18_BrOD],axis=0) # With SIPEXII
#BrO_All    = SIPEXII_BrO # SIPEXII Only

# # BrO (VMR)
# BrO_VMR_All    = pd.concat([V1_17_BrO_VMR,V2_17_BrO_VMR,V3_17_BrO_VMRM,V3_17_BrO_VMRD,V1_18_BrO_VMR,V2_18_BrO_VMR,V3_18_BrO_VMRM,V3_18_BrO_VMRD],axis=0) # Without SIPEXII

# Met
Met_All    = pd.concat([V1_17_Met,V2_17_Met,V3_17_MetM,V3_17_MetD,V1_18_Met,V2_18_Met,V3_18_MetM,V3_18_MetD],axis=0) # Without SIPEXII
#Met_All    = pd.concat([V1_17_Met,V2_17_Met,V1_18_Met,V2_18_Met],axis=0) # Without V3 & SIPEXII
#Met_All    = pd.concat([SIPEXII_Met,V1_17_Met,V2_17_Met,V3_17_MetM,V3_17_MetD,V1_18_Met,V2_18_Met,V3_18_MetM,V3_18_MetD],axis=0) # With SIPEXII
#Met_All    = SIPEXII_Met # With SIPEXII

# MERRA2
MERRA2_All = pd.concat([V1_17_MERRA2,V2_17_MERRA2,V3_17_MERRA2M,V3_17_MERRA2D,V1_18_MERRA2,V2_18_MERRA2,V3_18_MERRA2M,V3_18_MERRA2D],axis=0) # Without SIPEXII
#MERRA2_All = pd.concat([V1_17_MERRA2,V2_17_MERRA2,V1_18_MERRA2,V2_18_MERRA2],axis=0) # Without V3 & SIPEXII
#MERRA2_All = pd.concat([SIPEXII_MERRA2,V1_17_MERRA2,V2_17_MERRA2,V3_17_MERRA2M,V3_17_MERRA2D,V1_18_MERRA2,V2_18_MERRA2,V3_18_MERRA2M,V3_18_MERRA2D],axis=0) # With SIPEXII
#MERRA2_All = SIPEXII_MERRA2 # With SIPEXII

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
BrO_LtCol = BrO_All['BrO_VCD(molec/cm^2)'] # BrO VCD

# LtCol BrO Error
BrO_LtCol_Err = BrO_All['err_BrO_VCD'] # BrO VCD error

# AEC
AEC = BrO_All['0.1km'] # Aerosol extinction coefficient (km-1)

# O3
O3 = BrO_All['O3_(ppb)'] # O3 (ppb)

#------------------------------------------------------------------------------
# REMOVE NAN VALUES

#--------------
# MAX-DOAS
#--------------

BrO_Surf   = BrO_Surf.dropna()
BrO_LtCol  = BrO_LtCol.dropna()
AEC        = AEC.dropna()
O3         = O3.dropna()

#------------------------------------------------------------------------------
# FILTER THE DATAFRAMES TO ONLY INCLUDE THE SAME DATES

# BrO & O3
dfBrO_Surf  = pd.concat([BrO_Surf,  O3], axis=1, join='inner')
dfBrO_LtCol = pd.concat([BrO_LtCol, O3], axis=1, join='inner')

# dfBrO & MERRA2
dfBrO_Surf  = pd.concat([dfBrO_Surf,  MERRA2_All], axis=1, join='inner')
dfBrO_LtCol = pd.concat([dfBrO_LtCol, MERRA2_All], axis=1, join='inner')

# rename first column of dfBrO_Surf 
dfBrO_Surf.rename(columns={ dfBrO_Surf.columns[0]: 'BrO_(pptv)' }, inplace = True)

# dfBrO & AEC
dfBrO_Surf  = pd.concat([dfBrO_Surf,  AEC], axis=1, join='inner')
dfBrO_LtCol = pd.concat([dfBrO_LtCol, AEC], axis=1, join='inner')

# rename 0.1km to AEC
dfBrO_Surf.rename(columns  ={ dfBrO_Surf.columns[62]:  'AEC' }, inplace = True)
dfBrO_LtCol.rename(columns ={ dfBrO_LtCol.columns[62]: 'AEC' }, inplace = True)

# dfBrO & Met
dfBrO_Surf  = pd.concat([dfBrO_Surf,  Met_All], axis=1, join='inner')
dfBrO_LtCol = pd.concat([dfBrO_LtCol, Met_All], axis=1, join='inner')

# dfBrO & BrO_Err
dfBrO_Surf  = pd.concat([dfBrO_Surf,  BrO_Surf_Err],  axis=1, join='inner')
dfBrO_LtCol = pd.concat([dfBrO_LtCol, BrO_LtCol_Err], axis=1, join='inner')

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
# DEFINE THE VARIABLES

# BrO
SurfBrO      = dfBrO_Surf['BrO_(molecules/cm2)']      # Surface BrO (molecules/cm2)
LTBrO        = dfBrO_LtCol['BrO_(molecules/cm2)']     # Ltcol BrO (molecules/cm2)
SurfBrO_Err  = dfBrO_Surf['err_(molecules/cm2)']      # Surface BrO Error (molecules/cm2)
LTBrO_Err    = dfBrO_LtCol['err_BrO_VCD']             # Ltcol BrO Error (molecules/cm2)

# AEC
AEC          = dfBrO_Surf['AEC']                      # Aerosol extinction coefficient (km-1)

# O3
O3           = dfBrO_Surf['O3_(ppb)']                 # Surface O3 (ppb)

# Sea level pressure
SLPM2        = dfBrO_Surf['SLP']/100                  # MERRA2 sea level pressure (hPa)
SurfPresM2   = dfBrO_Surf['SurfPres']/100             # MERRA2 surface pressure (hPa)
SurfPresMet  = dfBrO_Surf['atm_press_hpa']            # Met surface pressure (hPa)

# Surface temperature
SurfTempM2   = dfBrO_Surf['Temp2m']-273.15            # MERRA2 temperature at 2m (C)
SurfTempMetP = dfBrO_Surf['temp_air_port_degc']       # Met temperature port side (C)
SurfTempMetS = dfBrO_Surf['temp_air_strbrd_degc']     # Met temperature strbrd side (C)
SurfTempMet  = (SurfTempMetP+SurfTempMetS)/2          # Met temperature average (port & strbrd side) (C)

# Change in pressure over 1 hour
P1hrM2a      = dfBrO_Surf['P1hrM2a']                  # MERRA2 change in sea level pressure from one hour to next (hPa)
P1hrM2b      = dfBrO_Surf['P1hrM2b']                  # MERRA2 change in sea level pressure from one hour to next (hPa)
P1hrMet      = dfBrO_Surf['P1hrMet']                  # Met change in pressure from one hour to next (hPa)

# Potential temperature differential
PTDif100m    = dfBrO_Surf['PTDif100m']                # Potential temperature differential lowest 100m (K)
PTDif1000m   = dfBrO_Surf['PTDif1000m']               # Potential temperature differential lowest 1000m (K)

# Wind Speed
WS10mM2      = dfBrO_Surf['WS10m']                    # MERRA2 wind speed at 10m (Kg/m2/s)
WS10mMetP    = dfBrO_Surf['wnd_spd_port_corr_knot']   * 0.514444444   # Convert wind speed port side from knots to m/s
WS10mMetS    = dfBrO_Surf['wnd_spd_strbrd_corr_knot'] * 0.514444444 # Convert wind speed strbrd side from knots to m/s
WS10mMet     = (WS10mMetP+WS10mMetS)/2                # Met wind speed average (port & strbrd side) (m/s)

# Wind Direction
WindDirP     = dfBrO_Surf['wnd_dir_port_corr_deg'] 
WindDirS     = dfBrO_Surf['wnd_dir_strbrd_corr_deg'] 
WindDir      = (WindDirP+WindDirS/2)

# Mixing layer height
MLH          = dfBrO_Surf['MLH']*1000                 # Richardson MLH (m)

#------------------------------------------------------------------------------
# FIND WIND SPEED CERTAIN DATES

# 14 Nov 2017
start_date = '2017-11-14'
end_date   = '2017-11-15'
# BrO
filterWS1  = (WS10mMet.index >= start_date) & (WS10mMet.index < end_date)
WS_14Nov   = WS10mMet[filterWS1]

# 20 Dec 2018
start_date = '2018-12-20'
end_date   = '2018-12-21'
# BrO
filterWS2  = (WS10mMet.index >= start_date) & (WS10mMet.index < end_date)
WS_20Dec   = WS10mMet[filterWS2]

# 26 Dec 2018
start_date = '2018-12-26'
end_date   = '2018-12-27'
# BrO
filterWS3  = (WS10mMet.index >= start_date) & (WS10mMet.index < end_date)
WS_26Dec   = WS10mMet[filterWS3]

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
Mean_SLPM2       = np.mean(SLPM2)
Mean_SurfPresM2  = np.mean(SurfPresM2)
Mean_SurfPresMet = np.mean(SurfPresMet) 
Mean_SurfTempM2  = np.mean(SurfTempM2)
Mean_SurfTempMet = np.mean(SurfTempMet)
Mean_P1hrM2a     = np.mean(P1hrM2a)
Mean_P1hrM2b     = np.mean(P1hrM2b)
Mean_P1hrMet     = np.mean(P1hrMet) 
Mean_PTDif100m   = np.mean(PTDif100m) 
Mean_PTDif1000m  = np.mean(PTDif1000m) 
Mean_WS10mM2     = np.mean(WS10mM2)
Mean_WS10mMet    = np.mean(WS10mMet) 
Mean_MLH         = np.mean(MLH)
Mean_WindDir     = np.mean(WindDir)

#------------------------------------------------------------------------------
# CALCULATE THE MEDIAN

Median_SurfBrO     = np.median(SurfBrO)
Median_LTBrO       = np.median(LTBrO)
Median_AEC         = np.median(AEC)
Median_O3          = np.median(O3) 
Median_SLPM2       = np.median(SLPM2)
Median_SurfPresM2  = np.median(SurfPresM2)
Median_SurfPresMet = np.median(SurfPresMet) 
Median_SurfTempM2  = np.median(SurfTempM2)
Median_SurfTempMet = np.median(SurfTempMet)
Median_P1hrM2a     = np.median(P1hrM2a)
Median_P1hrM2b     = np.median(P1hrM2b)
Median_P1hrMet     = np.median(P1hrMet) 
Median_PTDif100m   = np.median(PTDif100m) 
Median_PTDif1000m  = np.median(PTDif1000m) 
Median_WS10mM2     = np.median(WS10mM2)
Median_WS10mMet    = np.median(WS10mMet) 
Median_MLH         = np.median(MLH)
Median_WindDir     = np.median(WindDir)

#------------------------------------------------------------------------------
# CALCULATE THE MINIMUM

Min_SurfBrO     = np.min(SurfBrO)
Min_LTBrO       = np.min(LTBrO)
Min_AEC         = np.min(AEC)
Min_O3          = np.min(O3) 
Min_SLPM2       = np.min(SLPM2)
Min_SurfPresM2  = np.min(SurfPresM2)
Min_SurfPresMet = np.min(SurfPresMet) 
Min_SurfTempM2  = np.min(SurfTempM2)
Min_SurfTempMet = np.min(SurfTempMet)
Min_P1hrM2a     = np.min(P1hrM2a)
Min_P1hrM2b     = np.min(P1hrM2b)
Min_P1hrMet     = np.min(P1hrMet) 
Min_PTDif100m   = np.min(PTDif100m) 
Min_PTDif1000m  = np.min(PTDif1000m) 
Min_WS10mM2     = np.min(WS10mM2)
Min_WS10mMet    = np.min(WS10mMet) 
Min_MLH         = np.min(MLH)
Min_WindDir     = np.min(WindDir)

#------------------------------------------------------------------------------
# CALCULATE THE MAXIMUM

Max_SurfBrO     = np.max(SurfBrO)
Max_LTBrO       = np.max(LTBrO)
Max_AEC         = np.max(AEC)
Max_O3          = np.max(O3) 
Max_SLPM2       = np.max(SLPM2)
Max_SurfPresM2  = np.max(SurfPresM2)
Max_SurfPresMet = np.max(SurfPresMet) 
Max_SurfTempM2  = np.max(SurfTempM2)
Max_SurfTempMet = np.max(SurfTempMet)
Max_P1hrM2a     = np.max(P1hrM2a)
Max_P1hrM2b     = np.max(P1hrM2b)
Max_P1hrMet     = np.max(P1hrMet) 
Max_PTDif100m   = np.max(PTDif100m) 
Max_PTDif1000m  = np.max(PTDif1000m) 
Max_WS10mM2     = np.max(WS10mM2)
Max_WS10mMet    = np.max(WS10mMet) 
Max_MLH         = np.max(MLH)
Max_WindDir     = np.max(WindDir)

#------------------------------------------------------------------------------
# CALCULATE THE STANDARD DEVIATION

StDev_SurfBrO     = np.std(SurfBrO)
StDev_LTBrO       = np.std(LTBrO)
StDev_AEC         = np.std(AEC)
StDev_O3          = np.std(O3) 
StDev_SLPM2       = np.std(SLPM2)
StDev_SurfPresM2  = np.std(SurfPresM2) 
StDev_SurfPresMet = np.std(SurfPresMet)
StDev_SurfTempM2  = np.std(SurfTempM2)
StDev_SurfTempMet = np.std(SurfTempMet) 
StDev_P1hrM2a     = np.std(P1hrM2a)
StDev_P1hrM2b     = np.std(P1hrM2b)
StDev_P1hrMet     = np.std(P1hrMet)
StDev_PTDif100m   = np.std(PTDif100m) 
StDev_PTDif1000m  = np.std(PTDif1000m) 
StDev_WS10mM2     = np.std(WS10mM2)
StDev_WS10mMet    = np.std(WS10mMet)
StDev_MLH         = np.std(MLH)
StDev_WindDir     = np.std(WindDir)

#------------------------------------------------------------------------------
# CALCULATE THE MEDIAN ABSOLUTE DEVIATION

Mad_SurfBrO     = stats.median_absolute_deviation(SurfBrO)
Mad_LTBrO       = stats.median_absolute_deviation(LTBrO)
Mad_AEC         = stats.median_absolute_deviation(AEC)
Mad_O3          = stats.median_absolute_deviation(O3) 
Mad_SLPM2       = stats.median_absolute_deviation(SLPM2)
Mad_SurfPresM2  = stats.median_absolute_deviation(SurfPresM2) 
Mad_SurfPresMet = stats.median_absolute_deviation(SurfPresMet)
Mad_SurfTempM2  = stats.median_absolute_deviation(SurfTempM2)
Mad_SurfTempMet = stats.median_absolute_deviation(SurfTempMet) 
Mad_P1hrM2a     = stats.median_absolute_deviation(P1hrM2a)
Mad_P1hrM2b     = stats.median_absolute_deviation(P1hrM2b)
Mad_P1hrMet     = stats.median_absolute_deviation(P1hrMet)
Mad_PTDif100m   = stats.median_absolute_deviation(PTDif100m) 
Mad_PTDif1000m  = stats.median_absolute_deviation(PTDif1000m) 
Mad_WS10mM2     = stats.median_absolute_deviation(WS10mM2)
Mad_WS10mMet    = stats.median_absolute_deviation(WS10mMet)
Mad_MLH         = stats.median_absolute_deviation(MLH)
Mad_WindDir     = stats.median_absolute_deviation(WindDir)

#------------------------------------------------------------------------------
# CALCULATE THE MEAN - ST DEV

MeanMStDev_SurfBrO     = Mean_SurfBrO     - StDev_SurfBrO
MeanMStDev_LTBrO       = Mean_LTBrO       - StDev_LTBrO
MeanMStDev_AEC         = Mean_AEC         - StDev_AEC
MeanMStDev_O3          = Mean_O3          - StDev_O3
MeanMStDev_SLPM2       = Mean_SLPM2       - StDev_SLPM2
MeanMStDev_SurfPresM2  = Mean_SurfPresM2  - StDev_SurfPresM2
MeanMStDev_SurfPresMet = Mean_SurfPresMet - StDev_SurfPresMet
MeanMStDev_SurfTempM2  = Mean_SurfTempM2  - StDev_SurfTempM2
MeanMStDev_SurfTempMet = Mean_SurfTempMet - StDev_SurfTempMet
MeanMStDev_P1hrM2a     = Mean_P1hrM2a     - StDev_P1hrM2a
MeanMStDev_P1hrM2b     = Mean_P1hrM2b     - StDev_P1hrM2b
MeanMStDev_P1hrMet     = Mean_P1hrMet     - StDev_P1hrMet
MeanMStDev_PTDif100m   = Mean_PTDif100m   - StDev_PTDif100m
MeanMStDev_PTDif1000m  = Mean_PTDif1000m  - StDev_PTDif1000m
MeanMStDev_WS10mM2     = Mean_WS10mM2     - StDev_WS10mM2
MeanMStDev_WS10mMet    = Mean_WS10mMet    - StDev_WS10mMet
MeanMStDev_MLH         = Mean_MLH         - StDev_MLH
MeanMStDev_WindDir     = Mean_WindDir     - StDev_WindDir

#------------------------------------------------------------------------------
# CALCULATE THE MEAN + ST DEV

MeanPStDev_SurfBrO     = Mean_SurfBrO     + StDev_SurfBrO
MeanPStDev_LTBrO       = Mean_LTBrO       + StDev_LTBrO
MeanPStDev_AEC         = Mean_AEC         + StDev_AEC
MeanPStDev_O3          = Mean_O3          + StDev_O3
MeanPStDev_SLPM2       = Mean_SLPM2       + StDev_SLPM2
MeanPStDev_SurfPresM2  = Mean_SurfPresM2  + StDev_SurfPresM2
MeanPStDev_SurfPresMet = Mean_SurfPresMet + StDev_SurfPresMet
MeanPStDev_SurfTempM2  = Mean_SurfTempM2  + StDev_SurfTempM2
MeanPStDev_SurfTempMet = Mean_SurfTempMet + StDev_SurfTempMet
MeanPStDev_P1hrM2a     = Mean_P1hrM2a     + StDev_P1hrM2a
MeanPStDev_P1hrM2b     = Mean_P1hrM2b     + StDev_P1hrM2b
MeanPStDev_P1hrMet     = Mean_P1hrMet     + StDev_P1hrMet
MeanPStDev_PTDif100m   = Mean_PTDif100m   + StDev_PTDif100m
MeanPStDev_PTDif1000m  = Mean_PTDif1000m  + StDev_PTDif1000m
MeanPStDev_WS10mM2     = Mean_WS10mM2     + StDev_WS10mM2
MeanPStDev_WS10mMet    = Mean_WS10mMet    + StDev_WS10mMet
MeanPStDev_MLH         = Mean_MLH         + StDev_MLH
MeanPStDev_WindDir     = Mean_WindDir     + StDev_WindDir

#------------------------------------------------------------------------------
# STANDARDISE THE VARIABLES (SUBTRACT MEAN & DIVIDE BY ST DEV)

Stand_SurfBrO     = (SurfBrO     - Mean_SurfBrO)     / StDev_SurfBrO
Stand_LTBrO       = (LTBrO       - Mean_LTBrO)       / StDev_LTBrO
Stand_AEC         = (AEC         - Mean_AEC)         / StDev_AEC
Stand_O3          = (O3          - Mean_O3)          / StDev_O3
Stand_SLPM2       = (SLPM2       - Mean_SLPM2)       / StDev_SLPM2
Stand_SurfPresM2  = (SurfPresM2  - Mean_SurfPresM2)  / StDev_SurfPresM2
Stand_SurfPresMet = (SurfPresMet - Mean_SurfPresMet) / StDev_SurfPresMet
Stand_SurfTempM2  = (SurfTempM2  - Mean_SurfTempM2)  / StDev_SurfTempM2
Stand_SurfTempMet = (SurfTempMet - Mean_SurfTempMet) / StDev_SurfTempMet
Stand_P1hrM2a     = (P1hrM2a     - Mean_P1hrM2a)     / StDev_P1hrM2a
Stand_P1hrM2b     = (P1hrM2b     - Mean_P1hrM2b)     / StDev_P1hrM2b
Stand_P1hrMet     = (P1hrMet     - Mean_P1hrMet)     / StDev_P1hrMet
Stand_PTDif100m   = (PTDif100m   - Mean_PTDif100m)   / StDev_PTDif100m
Stand_PTDif1000m  = (PTDif1000m  - Mean_PTDif1000m)  / StDev_PTDif1000m
Stand_WS10mM2     = (WS10mM2     - Mean_WS10mM2)     / StDev_WS10mM2
Stand_WS10mMet    = (WS10mMet    - Mean_WS10mMet)    / StDev_WS10mMet
Stand_MLH         = (MLH         - Mean_MLH)         / StDev_MLH
Stand_WindDir     = (WindDir     - Mean_WindDir)     / StDev_WindDir

# Build a pandas dataframe (need to exclude BrO)
StandVariables = np.column_stack((Stand_O3,Stand_AEC,Stand_SurfTempMet,Stand_SurfPresMet,Stand_WS10mMet,Stand_MLH,Stand_P1hrMet,Stand_PTDif1000m,Stand_PTDif100m,Stand_WindDir))
StandVariables = pd.DataFrame(StandVariables, columns = ['O3','AEC','Surf_Temp','Surf_Pres','WS10m','MLH','P1hr','PTDif1000m','PTDif100m','Wind_Dir'])

#------------------------------------------------------------------------------
# SANITY CHECK ON STANDARDISED VARIABLES (MEAN = 0, StDev = 1, RANGE ~ 2-3 StDev MAX)

# Mean of standardised variables
Mean_Stand_SurfBrO     = np.mean(Stand_SurfBrO)
Mean_Stand_LTBrO       = np.mean(Stand_LTBrO)
Mean_Stand_AEC         = np.mean(Stand_AEC)
Mean_Stand_O3          = np.mean(Stand_O3)
Mean_Stand_SLPM2       = np.mean(Stand_SLPM2)
Mean_Stand_SurfPresM2  = np.mean(Stand_SurfPresM2)
Mean_Stand_SurfPresMet = np.mean(Stand_SurfPresMet)
Mean_Stand_SurfTempM2  = np.mean(Stand_SurfTempM2)
Mean_Stand_SurfTempMet = np.mean(Stand_SurfTempMet)
Mean_Stand_P1hrM2a     = np.mean(Stand_P1hrM2a)
Mean_Stand_P1hrM2b     = np.mean(Stand_P1hrM2b)
Mean_Stand_P1hrMet     = np.mean(Stand_P1hrMet)
Mean_Stand_PTDif100m   = np.mean(Stand_PTDif100m)
Mean_Stand_PTDif1000m  = np.mean(Stand_PTDif1000m)
Mean_Stand_WS10mM2     = np.mean(Stand_WS10mM2)
Mean_Stand_WS10mMet    = np.mean(Stand_WS10mMet)
Mean_Stand_MLH         = np.mean(Stand_MLH)
Mean_Stand_WindDir     = np.mean(Stand_WindDir)

# StDev of standardised variables
StDev_Stand_SurfBrO     = np.std(Stand_SurfBrO)
StDev_Stand_LTBrO       = np.std(Stand_LTBrO)
StDev_Stand_AEC         = np.std(Stand_AEC)
StDev_Stand_O3          = np.std(Stand_O3)
StDev_Stand_SLPM2       = np.std(Stand_SLPM2)
StDev_Stand_SurfPresM2  = np.std(Stand_SurfPresM2)
StDev_Stand_SurfPresMet = np.std(Stand_SurfPresMet)
StDev_Stand_SurfTempM2  = np.std(Stand_SurfTempM2)
StDev_Stand_SurfTempMet = np.std(Stand_SurfTempMet)
StDev_Stand_P1hrM2a     = np.std(Stand_P1hrM2a)
StDev_Stand_P1hrM2b     = np.std(Stand_P1hrM2b)
StDev_Stand_P1hrMet     = np.std(Stand_P1hrMet)
StDev_Stand_PTDif100m   = np.std(Stand_PTDif100m)
StDev_Stand_PTDif1000m  = np.std(Stand_PTDif1000m)
StDev_Stand_WS10mM2     = np.std(Stand_WS10mM2)
StDev_Stand_WS10mMet    = np.std(Stand_WS10mMet)
StDev_Stand_MLH         = np.std(Stand_MLH)
StDev_Stand_WindDir     = np.std(Stand_WindDir)

# Range of standardised variables
Range_Stand_SurfBrO     = np.ptp(Stand_SurfBrO)
Range_Stand_LTBrO       = np.ptp(Stand_LTBrO)
Range_Stand_AEC         = np.ptp(Stand_AEC)
Range_Stand_O3          = np.ptp(Stand_O3)
Range_Stand_SLPM2       = np.ptp(Stand_SLPM2)
Range_Stand_SurfPresM2  = np.ptp(Stand_SurfPresM2)
Range_Stand_SurfPresMet = np.ptp(Stand_SurfPresMet)
Range_Stand_SurfTempM2  = np.ptp(Stand_SurfTempM2)
Range_Stand_SurfTempMet = np.ptp(Stand_SurfTempMet)
Range_Stand_P1hrM2a     = np.ptp(Stand_P1hrM2a)
Range_Stand_P1hrM2b     = np.ptp(Stand_P1hrM2b)
Range_Stand_P1hrMet     = np.ptp(Stand_P1hrMet)
Range_Stand_PTDif100m   = np.ptp(Stand_PTDif100m)
Range_Stand_PTDif1000m  = np.ptp(Stand_PTDif1000m)
Range_Stand_WS10mM2     = np.ptp(Stand_WS10mM2)
Range_Stand_WS10mMet    = np.ptp(Stand_WS10mMet)
Range_Stand_MLH         = np.ptp(Stand_MLH)
Range_Stand_WindDir     = np.ptp(Stand_WindDir)

# Build a pandas dataframe
Sanity3 = {'Mean': [Mean_Stand_SurfBrO,Mean_Stand_LTBrO,Mean_Stand_AEC,Mean_Stand_O3,Mean_Stand_SLPM2,Mean_Stand_SurfPresM2,Mean_Stand_SurfPresMet,Mean_Stand_SurfTempM2,Mean_Stand_SurfTempMet,Mean_Stand_P1hrM2a,Mean_Stand_P1hrM2b,Mean_Stand_P1hrMet,Mean_Stand_PTDif100m,Mean_Stand_PTDif1000m,Mean_Stand_WS10mM2,Mean_Stand_WS10mMet,Mean_Stand_MLH,Mean_Stand_WindDir],
           'StDev': [StDev_Stand_SurfBrO,StDev_Stand_LTBrO,StDev_Stand_AEC,StDev_Stand_O3,StDev_Stand_SLPM2,StDev_Stand_SurfPresM2,StDev_Stand_SurfPresMet,StDev_Stand_SurfTempM2,StDev_Stand_SurfTempMet,StDev_Stand_P1hrM2a,StDev_Stand_P1hrM2b,StDev_Stand_P1hrMet,StDev_Stand_PTDif100m,StDev_Stand_PTDif1000m,StDev_Stand_WS10mM2,StDev_Stand_WS10mMet,StDev_Stand_MLH,StDev_Stand_WindDir],
           'Range': [Range_Stand_SurfBrO,Range_Stand_LTBrO,Range_Stand_AEC,Range_Stand_O3,Range_Stand_SLPM2,Range_Stand_SurfPresM2,Range_Stand_SurfPresMet,Range_Stand_SurfTempM2,Range_Stand_SurfTempMet,Range_Stand_P1hrM2a,Range_Stand_P1hrM2b,Range_Stand_P1hrMet,Range_Stand_PTDif100m,Range_Stand_PTDif1000m,Range_Stand_WS10mM2,Range_Stand_WS10mMet,Range_Stand_MLH,Range_Stand_WindDir]}
Sanity3 = pd.DataFrame(Sanity3, columns = ['Mean','StDev','Range'],index = ['SurfBrO','LTBrO','AEC','O3','SLP_MERRA2','SurfPres_MERRA2','SurfPres_Met','SurfTemp_MERRA2','SurfTemp_Met','P1hr_MERRA2_SLP','P1hr_MERRA2_SurfPres','P1hr_Met','PTDif100m','PTDif1000m','WS10m_MERRA2','WS10m_Met','MLH','Wind_Dir'])
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
dfMeanStDev = {'Mean': [Mean_SurfBrO/1e12,Mean_LTBrO/1e12,Mean_AEC,Mean_O3,Mean_SLPM2,Mean_SurfPresM2,Mean_SurfPresMet,Mean_SurfTempM2,Mean_SurfTempMet,Mean_P1hrM2a,Mean_P1hrM2b,Mean_P1hrMet,Mean_PTDif100m,Mean_PTDif1000m,Mean_WS10mM2,Mean_WS10mMet,Mean_MLH],
               'StDev': [StDev_SurfBrO/1e12,StDev_LTBrO/1e12,StDev_AEC,StDev_O3,StDev_SLPM2,StDev_SurfPresM2,StDev_SurfPresMet,StDev_SurfTempM2,StDev_SurfTempMet,StDev_P1hrM2a,StDev_P1hrM2b,StDev_P1hrMet,StDev_PTDif100m,StDev_PTDif1000m,StDev_WS10mM2,StDev_WS10mMet,StDev_MLH],
               'MeanMStDev': [MeanMStDev_SurfBrO/1e12,MeanMStDev_LTBrO/1e12,MeanMStDev_AEC,MeanMStDev_O3,MeanMStDev_SLPM2,MeanMStDev_SurfPresM2,MeanMStDev_SurfPresMet,MeanMStDev_SurfTempM2,MeanMStDev_SurfTempMet,MeanMStDev_P1hrM2a,MeanMStDev_P1hrM2b,MeanMStDev_P1hrMet,MeanMStDev_PTDif100m,MeanMStDev_PTDif1000m,MeanMStDev_WS10mM2,MeanMStDev_WS10mMet,MeanMStDev_MLH],
               'MeanPStDev': [MeanPStDev_SurfBrO/1e12,MeanPStDev_LTBrO/1e12,MeanPStDev_AEC,MeanPStDev_O3,MeanPStDev_SLPM2,MeanPStDev_SurfPresM2,MeanPStDev_SurfPresMet,MeanPStDev_SurfTempM2,MeanPStDev_SurfTempMet,MeanPStDev_P1hrM2a,MeanPStDev_P1hrM2b,MeanPStDev_P1hrMet,MeanPStDev_PTDif100m,MeanPStDev_PTDif1000m,MeanPStDev_WS10mM2,MeanPStDev_WS10mMet,MeanPStDev_MLH],
               'Median': [Median_SurfBrO/1e12,Median_LTBrO/1e12,Median_AEC,Median_O3,Median_SLPM2,Median_SurfPresM2,Median_SurfPresMet,Median_SurfTempM2,Median_SurfTempMet,Median_P1hrM2a,Median_P1hrM2b,Median_P1hrMet,Median_PTDif100m,Median_PTDif1000m,Median_WS10mM2,Median_WS10mMet,Median_MLH],
               'MAD': [Mad_SurfBrO/1e12,Mad_LTBrO/1e12,Mad_AEC,Mad_O3,Mad_SLPM2,Mad_SurfPresM2,Mad_SurfPresMet,Mad_SurfTempM2,Mad_SurfTempMet,Mad_P1hrM2a,Mad_P1hrM2b,Mad_P1hrMet,Mad_PTDif100m,Mad_PTDif1000m,Mad_WS10mM2,Mad_WS10mMet,Mad_MLH],
               'Min': [Min_SurfBrO/1e12,Min_LTBrO/1e12,Min_AEC,Min_O3,Min_SLPM2,Min_SurfPresM2,Min_SurfPresMet,Min_SurfTempM2,Min_SurfTempMet,Min_P1hrM2a,Min_P1hrM2b,Min_P1hrMet,Min_PTDif100m,Min_PTDif1000m,Min_WS10mM2,Min_WS10mMet,Min_MLH],
               'Max': [Max_SurfBrO/1e12,Max_LTBrO/1e12,Max_AEC,Max_O3,Max_SLPM2,Max_SurfPresM2,Max_SurfPresMet,Max_SurfTempM2,Max_SurfTempMet,Max_P1hrM2a,Max_P1hrM2b,Max_P1hrMet,Max_PTDif100m,Max_PTDif1000m,Max_WS10mM2,Max_WS10mMet,Max_MLH]}
dfMeanStDev = pd.DataFrame(dfMeanStDev, columns = ['Mean','StDev','MeanMStDev','MeanPStDev','Median','MAD','Min','Max'],index = ['SurfBrO','LTBrO','AEC','O3','SLP_MERRA2','SurfPres_MERRA2','SurfPres_Met','SurfTemp_MERRA2','SurfTemp_Met','P1hr_MERRA2_SLP','P1hr_MERRA2_SurfPres','P1hr_Met','PTDif100m','PTDif1000m','WS10m_MERRA2','WS10m_Met','MLH'])
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

# Apply the PCA (Swanson et al has 9 PCs, but only 3 have variances greater 1)
PCA_BrO = PCA() # All n components

# Retrieve the principal components (PCs)
PrincipalComponents_Variables = PCA_BrO.fit_transform(StandVariables) # Variables
#PrincipalComponents_Variables = PCA_BrO.inverse_transform(StandVariables) # Variables
#PrincipalComponents_Variables = PCA_BrO.transform(StandVariables) # Variables
#PrincipalComponents_Variables = PCA_BrO.fit(StandVariables) # Variables

# Put the principle components into a DataFrame
Principal_Variables_Df = pd.DataFrame(data = PrincipalComponents_Variables, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9']) # Variables
    
# Explained variation per principal component
print('Explained variation per principal component: {}'.format(PCA_BrO.explained_variance_ratio_))
Explained_Variance = PCA_BrO.explained_variance_ratio_

# Get the loadings
loadings = pd.DataFrame(PCA_BrO.components_.T, columns=Principal_Variables_Df.columns,index=StandVariables.columns)

# Calculate the normalised variance for each PC
PV_PC1      = Principal_Variables_Df['PC1']
PV_PC1_Mean = np.mean(Principal_Variables_Df['PC1'])
NV_PC1      = np.mean(np.square(PV_PC1 - PV_PC1_Mean))

PV_PC2      = Principal_Variables_Df['PC2']
PV_PC2_Mean = np.mean(Principal_Variables_Df['PC2'])
NV_PC2      = np.mean(np.square(PV_PC2 - PV_PC2_Mean))

PV_PC3      = Principal_Variables_Df['PC3']
PV_PC3_Mean = np.mean(Principal_Variables_Df['PC3'])
NV_PC3      = np.mean(np.square(PV_PC3 - PV_PC3_Mean))

PV_PC4      = Principal_Variables_Df['PC4']
PV_PC4_Mean = np.mean(Principal_Variables_Df['PC4'])
NV_PC4      = np.mean(np.square(PV_PC4 - PV_PC4_Mean))

PV_PC5      = Principal_Variables_Df['PC5']
PV_PC5_Mean = np.mean(Principal_Variables_Df['PC5'])
NV_PC5      = np.mean(np.square(PV_PC5 - PV_PC5_Mean))

PV_PC6      = Principal_Variables_Df['PC6']
PV_PC6_Mean = np.mean(Principal_Variables_Df['PC6'])
NV_PC6      = np.mean(np.square(PV_PC6 - PV_PC6_Mean))

PV_PC7      = Principal_Variables_Df['PC7']
PV_PC7_Mean = np.mean(Principal_Variables_Df['PC7'])
NV_PC7      = np.mean(np.square(PV_PC7 - PV_PC7_Mean))

PV_PC8      = Principal_Variables_Df['PC8']
PV_PC8_Mean = np.mean(Principal_Variables_Df['PC8'])
NV_PC8      = np.mean(np.square(PV_PC8 - PV_PC8_Mean))

PV_PC9      = Principal_Variables_Df['PC9']
PV_PC9_Mean = np.mean(Principal_Variables_Df['PC9'])
NV_PC9      = np.mean(np.square(PV_PC9 - PV_PC9_Mean))

# Put the normalised variance into an array
PCA_NV = np.array([NV_PC1,NV_PC2,NV_PC3,NV_PC4,NV_PC5,NV_PC6,NV_PC7,NV_PC8,NV_PC9])

# Export the PCA results
dfPCA_BrO = np.row_stack((Explained_Variance,PCA_NV))
dfPCA_BrO = pd.DataFrame(dfPCA_BrO, index = ['Explained_Variance','Normalised_Variance'],columns = Principal_Variables_Df.columns)
dfPCA_BrO = pd.concat([loadings,dfPCA_BrO])
dfPCA_BrO.to_csv('/Users/ncp532/Documents/Data/MERRA2/PCA_Loadings_&_Variance.csv')

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

# # My loadings
# O3_PC1L,      O3_PC2L,      O3_PC3L      = loadings['PC1'][0], loadings['PC2'][0], loadings['PC3'][0] # Ozone (ppb) # (nmol/mol)?
# EXT_PC1L,     EXT_PC2L,     EXT_PC3L     = loadings['PC1'][1], loadings['PC2'][1], loadings['PC3'][1] # Aerosol extinction (km-1) # (m/km)?
# STemp_PC1L,   STemp_PC2L,   STemp_PC3L   = loadings['PC1'][2], loadings['PC2'][2], loadings['PC3'][2] # Surface Temp (K) # (C)?
# SLP_PC1L,     SLP_PC2L,     SLP_PC3L     = loadings['PC1'][3], loadings['PC2'][3], loadings['PC3'][3] # Sea level pressure (hPa)
# VW10m_PC1L,   VW10m_PC2L,   VW10m_PC3L   = loadings['PC1'][4], loadings['PC2'][4], loadings['PC3'][4] # Windspeed at 10m (m/s)
# MLH_PC1L,     MLH_PC2L,     MLH_PC3L     = loadings['PC1'][5], loadings['PC2'][5], loadings['PC3'][5] # Richardson mixed layer height (m)
# P1hr_PC1L,    P1hr_PC2L,    P1hr_PC3L    = loadings['PC1'][6], loadings['PC2'][6], loadings['PC3'][6] # Change in pressure from one hour to next (hPa)
# PT1000m_PC1L, PT1000m_PC2L, PT1000m_PC3L = loadings['PC1'][7], loadings['PC2'][7], loadings['PC3'][7] # Potential temperature differential in lowest 1000m (m/K)
# PT100m_PC1L,  PT100m_PC2L,  PT100m_PC3L  = loadings['PC1'][8], loadings['PC2'][8], loadings['PC3'][8] # Potential temperature differential in lowest 100m (m/K)

#------------------------------------------------------------------------------
# CALCULATE THE PRINCIPLE COMPONENTS

# # MERRA2 - SLP
# PC1_M2a = (Stand_O3*O3_PC1L) + (Stand_AEC*EXT_PC1L) + (Stand_SurfTempM2*STemp_PC1L) + (Stand_SLPM2*SLP_PC1L) + (Stand_WS10mM2*VW10m_PC1L) + (Stand_MLH*MLH_PC1L) + (Stand_P1hrM2a*P1hr_PC1L) + (Stand_PTDif1000m*PT1000m_PC1L) + (Stand_PTDif100m*PT100m_PC1L)
# PC2_M2a = (Stand_O3*O3_PC2L) + (Stand_AEC*EXT_PC2L) + (Stand_SurfTempM2*STemp_PC2L) + (Stand_SLPM2*SLP_PC2L) + (Stand_WS10mM2*VW10m_PC2L) + (Stand_MLH*MLH_PC2L) + (Stand_P1hrM2a*P1hr_PC2L) + (Stand_PTDif1000m*PT1000m_PC2L) + (Stand_PTDif100m*PT100m_PC2L) 
# PC3_M2a = (Stand_O3*O3_PC3L) + (Stand_AEC*EXT_PC3L) + (Stand_SurfTempM2*STemp_PC3L) + (Stand_SLPM2*SLP_PC3L) + (Stand_WS10mM2*VW10m_PC3L) + (Stand_MLH*MLH_PC3L) + (Stand_P1hrM2a*P1hr_PC3L) + (Stand_PTDif1000m*PT1000m_PC3L) + (Stand_PTDif100m*PT100m_PC3L) 

# # MERRA2 - SurfPres
# PC1_M2b = (Stand_O3*O3_PC1L) + (Stand_AEC*EXT_PC1L) + (Stand_SurfTempM2*STemp_PC1L) + (Stand_SurfPresM2*SLP_PC1L) + (Stand_WS10mM2*VW10m_PC1L) + (Stand_MLH*MLH_PC1L) + (Stand_P1hrM2b*P1hr_PC1L) + (Stand_PTDif1000m*PT1000m_PC1L) + (Stand_PTDif100m*PT100m_PC1L)
# PC2_M2b = (Stand_O3*O3_PC2L) + (Stand_AEC*EXT_PC2L) + (Stand_SurfTempM2*STemp_PC2L) + (Stand_SurfPresM2*SLP_PC2L) + (Stand_WS10mM2*VW10m_PC2L) + (Stand_MLH*MLH_PC2L) + (Stand_P1hrM2b*P1hr_PC2L) + (Stand_PTDif1000m*PT1000m_PC2L) + (Stand_PTDif100m*PT100m_PC2L) 
# PC3_M2b = (Stand_O3*O3_PC3L) + (Stand_AEC*EXT_PC3L) + (Stand_SurfTempM2*STemp_PC3L) + (Stand_SurfPresM2*SLP_PC3L) + (Stand_WS10mM2*VW10m_PC3L) + (Stand_MLH*MLH_PC3L) + (Stand_P1hrM2b*P1hr_PC3L) + (Stand_PTDif1000m*PT1000m_PC3L) + (Stand_PTDif100m*PT100m_PC3L) 

# Met
PC1_Met = (Stand_O3*O3_PC1L) + (Stand_AEC*EXT_PC1L) + (Stand_SurfTempMet*STemp_PC1L) + (Stand_SurfPresMet*SLP_PC1L) + (Stand_WS10mMet*VW10m_PC1L) + (Stand_MLH*MLH_PC1L) + (Stand_P1hrMet*P1hr_PC1L) + (Stand_PTDif1000m*PT1000m_PC1L) + (Stand_PTDif100m*PT100m_PC1L)
PC2_Met = (Stand_O3*O3_PC2L) + (Stand_AEC*EXT_PC2L) + (Stand_SurfTempMet*STemp_PC2L) + (Stand_SurfPresMet*SLP_PC2L) + (Stand_WS10mMet*VW10m_PC2L) + (Stand_MLH*MLH_PC2L) + (Stand_P1hrMet*P1hr_PC2L) + (Stand_PTDif1000m*PT1000m_PC2L) + (Stand_PTDif100m*PT100m_PC2L) 
PC3_Met = (Stand_O3*O3_PC3L) + (Stand_AEC*EXT_PC3L) + (Stand_SurfTempMet*STemp_PC3L) + (Stand_SurfPresMet*SLP_PC3L) + (Stand_WS10mMet*VW10m_PC3L) + (Stand_MLH*MLH_PC3L) + (Stand_P1hrMet*P1hr_PC3L) + (Stand_PTDif1000m*PT1000m_PC3L) + (Stand_PTDif100m*PT100m_PC3L) 

#------------------------------------------------------------------------------
# PERFORM MULTIPLE LINEAR REGRESSION (MLR)
# (if z = sqrt(BrO_obs)) “z ~ O3a + AECa + SurfTempa + SLPa +WS10ma + MLHa + P1hra + PTD1000ma + PTD100ma

# Variables required
O3a       = np.array(Stand_O3)
AECa      = np.array(Stand_AEC)
SurfTempa = np.array(Stand_SurfTempMet)
SLPa      = np.array(Stand_SurfPresMet)
WS10ma    = np.array(Stand_WS10mMet)
MLHa      = np.array(Stand_MLH)
P1hra     = np.array(Stand_P1hrMet)
PTD1000ma = np.array(Stand_PTDif1000m)
PTD100ma  = np.array(Stand_PTDif100m)
WindDira  = np.array(Stand_WindDir)
SurfBrOa  = np.array(SurfBrO)
LTBrOa    = np.array(LTBrO)

# First we need to flatten the data: it's 2D layout is not relevent.
O3a       = O3a.flatten()
AECa      = AECa.flatten()
SurfTempa = SurfTempa.flatten()
SLPa      = SLPa.flatten()
WS10ma    = WS10ma.flatten()
MLHa      = MLHa.flatten()
P1hra     = P1hra.flatten()
PTD1000ma = PTD1000ma.flatten()
PTD100ma  = PTD100ma.flatten()
WindDira  = WindDira.flatten()
SurfBrOa  = SurfBrOa.flatten()
LTBrOa    = LTBrOa.flatten()

# Build the DataFrame
dataS  = pd.DataFrame({'O3': O3a, 'AEC': AECa, 'SurfTemp': SurfTempa, 'SLP': SLPa, 'WS10m': WS10ma, 'MLH': MLHa, 'P1hr': P1hra, 'PTD1000m': PTD1000ma, 'PTD100m': PTD100ma, 'Wind_Dir': WindDira, 'z': np.sqrt(SurfBrOa)})
dataLT = pd.DataFrame({'O3': O3a, 'AEC': AECa, 'SurfTemp': SurfTempa, 'SLP': SLPa, 'WS10m': WS10ma, 'MLH': MLHa, 'P1hr': P1hra, 'PTD1000m': PTD1000ma, 'PTD100m': PTD100ma, 'Wind_Dir': WindDira, 'z': np.sqrt(LTBrOa)})
XS     = dataS[['O3','AEC','SurfTemp','SLP','WS10m','MLH','P1hr','PTD1000m','PTD100m', 'Wind_Dir']]
YS     = dataS['z']
XLT    = dataLT[['O3','AEC','SurfTemp','SLP','WS10m','MLH','P1hr','PTD1000m','PTD100m', 'Wind_Dir']]
YLT    = dataLT['z']

#-------------------------------
# Individual variable regression
#-------------------------------
# Variables
BrO_S      = dataS['z']
O3_S       = dataS['O3']
AEC_S      = dataS['AEC']
SurfTemp_S = dataS['SurfTemp']
SLP_S      = dataS['SLP']
WS10m_S    = dataS['WS10m']
MLH_S      = dataS['MLH']
P1hr_S     = dataS['P1hr']
PTD1000m_S = dataS['PTD1000m']
PTD100m_S  = dataS['PTD100m']
WindDir_S  = dataS['Wind_Dir']

BrO_LT      = dataLT['z']
O3_LT       = dataLT['O3']
AEC_LT      = dataLT['AEC']
SurfTemp_LT = dataLT['SurfTemp']
SLP_LT      = dataLT['SLP']
WS10m_LT    = dataLT['WS10m']
MLH_LT      = dataLT['MLH']
P1hr_LT     = dataLT['P1hr']
PTD1000m_LT = dataLT['PTD1000m']
PTD100m_LT  = dataLT['PTD100m']
WindDir_LT  = dataLT['Wind_Dir']

# Linear Regression
slope_O3_S,       intercept_O3_S,       r_O3_S,       p_O3_S,       std_err_O3_S       = stats.linregress(BrO_S, O3_S)
slope_AEC_S,      intercept_AEC_S,      r_AEC_S,      p_AEC_S,      std_err_AEC_S      = stats.linregress(BrO_S, AEC_S)
slope_SurfTemp_S, intercept_SurfTemp_S, r_SurfTemp_S, p_SurfTemp_S, std_err_SurfTemp_S = stats.linregress(BrO_S, SurfTemp_S)
slope_SLP_S,      intercept_SLP_S,      r_SLP_S,      p_SLP_S,      std_err_SLP_S      = stats.linregress(BrO_S, SLP_S)
slope_WS10m_S,    intercept_WS10m_S,    r_WS10m_S,    p_WS10m_S,    std_err_WS10m_S    = stats.linregress(BrO_S, WS10m_S)
slope_MLH_S,      intercept_MLH_S,      r_MLH_S,      p_MLH_S,      std_err_MLH_S      = stats.linregress(BrO_S, MLH_S)
slope_P1hr_S,     intercept_P1hr_S,     r_P1hr_S,     p_P1hr_S,     std_err_P1hr_S     = stats.linregress(BrO_S, P1hr_S)
slope_PTD1000m_S, intercept_PTD1000m_S, r_PTD1000m_S, p_PTD1000m_S, std_err_PTD1000m_S = stats.linregress(BrO_S, PTD1000m_S)
slope_PTD100m_S,  intercept_PTD100m_S,  r_PTD100m_S,  p_PTD100m_S,  std_err_PTD100m_S  = stats.linregress(BrO_S, PTD100m_S)
slope_WindDir_S,  intercept_WindDir_S,  r_WindDir_S,  p_WindDir_S,  std_err_WindDir_S  = stats.linregress(BrO_S, WindDir_S)

slope_O3_LT,       intercept_O3_LT,       r_O3_LT,       p_O3_LT,       std_err_O3_LT       = stats.linregress(BrO_LT, O3_LT)
slope_AEC_LT,      intercept_AEC_LT,      r_AEC_LT,      p_AEC_LT,      std_err_AEC_LT      = stats.linregress(BrO_LT, AEC_LT)
slope_SurfTemp_LT, intercept_SurfTemp_LT, r_SurfTemp_LT, p_SurfTemp_LT, std_err_SurfTemp_LT = stats.linregress(BrO_LT, SurfTemp_LT)
slope_SLP_LT,      intercept_SLP_LT,      r_SLP_LT,      p_SLP_LT,      std_err_SLP_LT      = stats.linregress(BrO_LT, SLP_LT)
slope_WS10m_LT,    intercept_WS10m_LT,    r_WS10m_LT,    p_WS10m_LT,    std_err_WS10m_LT    = stats.linregress(BrO_LT, WS10m_LT)
slope_MLH_LT,      intercept_MLH_LT,      r_MLH_LT,      p_MLH_LT,      std_err_MLH_LT      = stats.linregress(BrO_LT, MLH_LT)
slope_P1hr_LT,     intercept_P1hr_LT,     r_P1hr_LT,     p_P1hr_LT,     std_err_P1hr_LT     = stats.linregress(BrO_LT, P1hr_LT)
slope_PTD1000m_LT, intercept_PTD1000m_LT, r_PTD1000m_LT, p_PTD1000m_LT, std_err_PTD1000m_LT = stats.linregress(BrO_LT, PTD1000m_LT)
slope_PTD100m_LT,  intercept_PTD100m_LT,  r_PTD100m_LT,  p_PTD100m_LT,  std_err_PTD100m_LT  = stats.linregress(BrO_LT, PTD100m_LT)
slope_WindDir_LT,  intercept_WindDir_LT,  r_WindDir_LT,  p_WindDir_LT,  std_err_WindDir_LT  = stats.linregress(BrO_LT, WindDir_LT)

# R-squared
r2_O3_S       = r_O3_S**2
r2_AEC_S      = r_AEC_S**2
r2_SurfTemp_S = r_SurfTemp_S**2
r2_SLP_S      = r_SLP_S**2
r2_WS10m_S    = r_WS10m_S**2
r2_MLH_S      = r_MLH_S**2
r2_P1hr_S     = r_P1hr_S**2
r2_PTD1000m_S = r_PTD1000m_S**2
r2_PTD100m_S  = r_PTD100m_S**2
r2_WindDir_S  = r_WindDir_S**2

r2_O3_LT       = r_O3_LT**2
r2_AEC_LT      = r_AEC_LT**2
r2_SurfTemp_LT = r_SurfTemp_LT**2
r2_SLP_LT      = r_SLP_LT**2
r2_WS10m_LT    = r_WS10m_LT**2
r2_MLH_LT      = r_MLH_LT**2
r2_P1hr_LT     = r_P1hr_LT**2
r2_PTD1000m_LT = r_PTD1000m_LT**2
r2_PTD100m_LT  = r_PTD100m_LT**2
r2_WindDir_LT  = r_WindDir_LT**2

# Put values in a DataFrame
dfLinearRegression = pd.DataFrame({'r2 (BrO_Surf)':       [r2_O3_S,  r2_AEC_S,  r2_SurfTemp_S,  r2_SLP_S,  r2_WS10m_S,  r2_MLH_S,  r2_P1hr_S,  r2_PTD1000m_S,  r2_PTD100m_S,  r2_WindDir_S],
                                   'r2 (BrO_LTcol)':      [r2_O3_LT, r2_AEC_LT, r2_SurfTemp_LT, r2_SLP_LT, r2_WS10m_LT, r2_MLH_LT, r2_P1hr_LT, r2_PTD1000m_LT, r2_PTD100m_LT, r2_WindDir_LT],
                                   'p-value (BrO_Surf)':  [p_O3_S,   p_AEC_S,   p_SurfTemp_S,   p_SLP_S,   p_WS10m_S,   p_MLH_S,   p_P1hr_S,   p_PTD1000m_S,   p_PTD100m_S,   p_WindDir_S],
                                   'p-value (BrO_LTcol)': [p_O3_LT,  p_AEC_LT,  p_SurfTemp_LT,  p_SLP_LT,  p_WS10m_LT,  p_MLH_LT,  p_P1hr_LT,  p_PTD1000m_LT,  p_PTD100m_LT,  p_WindDir_LT]})
dfLinearRegression.index = ['O3','AEC','SurfTemp','SLP','WS10m','MLH','P1hr','PTD1000m','PTD100m','WindDir']
dfLinearRegression.to_csv('/Users/ncp532/Documents/Data/MERRA2/LinearRegression.csv')

#-------------------------------
# Fit the model
modelS  = ols("z ~ O3a + AECa + SurfTempa + SLPa +WS10ma + MLHa + P1hra + PTD1000ma + PTD100ma", dataS).fit()  # Surface StatsModel (ols)
modelLT = ols("z ~ O3a + AECa + SurfTempa + SLPa +WS10ma + MLHa + P1hra + PTD1000ma + PTD100ma", dataLT).fit() # LTcol StatsModel (ols)
regS    = LinearRegression().fit(XS,  YS)  # Surface SkLearn (LinearRegresion)
regLT   = LinearRegression().fit(XLT, YLT) # LTcol SkLearn   (LinearRegresion)
 
# Retrieve the model results
Model_resultsS  = modelS._results.params                                  # Surface StatsModel (ols)
Model_resultsLT = modelLT._results.params                                 # LTcol StatsModel   (ols)
Intercept_resultS,  Coefficients_resultS  = regS.intercept_,  regS.coef_  # Surface SkLearn (LinearRegresion)
Intercept_resultLT, Coefficients_resultLT = regLT.intercept_, regLT.coef_ # LTcol SkLearn   (LinearRegresion)

# Peform analysis of variance on fitted linear model
anova_resultsS  = anova_lm(modelS)
anova_resultsLT = anova_lm(modelLT)

# #------------------------------------------------------------------------------
# # PERFORM A PRINCIPLE COMPONENT REGRESSION (PCR)
# # (if z = sqrt(BrO_obs)) “z ~ pc1 + pc2 + pc3

# # Variables required
# PC1a     = np.array(PC1_Met)
# PC2a     = np.array(PC2_Met)
# PC3a     = np.array(PC3_Met)
# SurfBrOa = np.array(SurfBrO)
# LTBrOa   = np.array(LTBrO)
# #SurfBrOa = np.array(Stand_SurfBrO)
# #LTBrOa   = np.array(Stand_LTBrO)

# # First we need to flatten the data: it's 2D layout is not relevent.
# PC1a     = PC1a.flatten()
# PC2a     = PC2a.flatten()
# PC3a     = PC3a.flatten()
# SurfBrOa = SurfBrOa.flatten()
# LTBrOa   = LTBrOa.flatten()

# # Build the DataFrame
# dataS  = pd.DataFrame({'PC1': PC1a, 'PC2': PC2a, 'PC3': PC3a, 'z': np.sqrt(SurfBrOa)})
# dataLT = pd.DataFrame({'PC1': PC1a, 'PC2': PC2a, 'PC3': PC3a, 'z': np.sqrt(LTBrOa)})
# XS     = dataS[['PC1','PC2','PC3']]
# YS     = dataS['z']
# XLT    = dataLT[['PC1','PC2','PC3']]
# YLT    = dataLT['z']

# # Fit the model
# modelS  = ols("z ~ PC1 + PC2 + PC3", dataS).fit()  # Surface StatsModel (ols)
# modelLT = ols("z ~ PC1 + PC2 + PC3", dataLT).fit() # LTcol StatsModel (ols)
# regS    = LinearRegression().fit(XS,  YS)          # Surface SkLearn (LinearRegresion)
# regLT   = LinearRegression().fit(XLT, YLT)         # LTcol SkLearn   (LinearRegresion)
 
# # Retrieve the model results
# Model_resultsS  = modelS._results.params                                  # Surface StatsModel (ols)
# Model_resultsLT = modelLT._results.params                                 # LTcol StatsModel (ols)
# Intercept_resultS, Coefficients_resultS   = regS.intercept_,  regS.coef_  # Surface SkLearn (LinearRegresion)
# Intercept_resultLT, Coefficients_resultLT = regLT.intercept_, regLT.coef_ # Surface SkLearn (LinearRegresion)

# # Peform analysis of variance on fitted linear model
# anova_resultsS  = anova_lm(modelS)
# anova_resultsLT = anova_lm(modelLT)

#------------------------------------------------------------------------------
# APPLY THE BrO MLR MODEL

#--------------
# Intercept and Coefficients
#--------------

# Surface BrO
B0S = 2.06*1e6 # Intercept for the multiple linear regression
#B0S = 1.37*1e6 # Intercept for the multiple linear regression
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
LR4S = Coefficients_resultS[3]
LR5S = Coefficients_resultS[4]
LR6S = Coefficients_resultS[5]
LR7S = Coefficients_resultS[6]
LR8S = Coefficients_resultS[7]
LR9S = Coefficients_resultS[8]

LR0LT = Intercept_resultLT
LR1LT = Coefficients_resultLT[0]
LR2LT = Coefficients_resultLT[1]
LR3LT = Coefficients_resultLT[2]
LR4LT = Coefficients_resultLT[3]
LR5LT = Coefficients_resultLT[4]
LR6LT = Coefficients_resultLT[5]
LR7LT = Coefficients_resultLT[6]
LR8LT = Coefficients_resultLT[7]
LR9LT = Coefficients_resultLT[8]

# StatsModels (ols)
OLS0S = Model_resultsS[0]
OLS1S = Model_resultsS[1]
OLS2S = Model_resultsS[2]
OLS3S = Model_resultsS[3]
OLS4S = Model_resultsS[4]
OLS5S = Model_resultsS[5]
OLS6S = Model_resultsS[6]
OLS7S = Model_resultsS[7]
OLS8S = Model_resultsS[8]
OLS9S = Model_resultsS[9]

OLS0LT = Model_resultsLT[0]
OLS1LT = Model_resultsLT[1]
OLS2LT = Model_resultsLT[2]
OLS3LT = Model_resultsLT[3]
OLS4LT = Model_resultsLT[4]
OLS5LT = Model_resultsLT[5]
OLS6LT = Model_resultsLT[6]
OLS7LT = Model_resultsLT[7]
OLS8LT = Model_resultsLT[8]
OLS9LT = Model_resultsLT[9]

#--------------
# PCR model
#--------------

# Surface BrO
# BrO_SurfPred_M2a = np.square(B0S + (B1S*PC1_M2a) + (B2S*PC2_M2a) + (B3S*PC3_M2a)) # MERRA-2 (SLP)
# BrO_SurfPred_M2b = np.square(B0S + (B1S*PC1_M2b) + (B2S*PC2_M2b) + (B3S*PC3_M2b)) # MERRA-2 (SurfPres)
BrO_SurfPred_Met = np.square(B0S + (B1S*PC1_Met) + (B2S*PC2_Met) + (B3S*PC3_Met)) # Met

# Lower tropospheric BrO
# BrO_LTPred_M2a = np.square(B0LT + (B1LT*PC1_M2a) + (B2LT*PC2_M2a) + (B3LT*PC3_M2a)) # MERRA-2 (SLP)
# BrO_LTPred_M2b = np.square(B0LT + (B1LT*PC1_M2b) + (B2LT*PC2_M2b) + (B3LT*PC3_M2b)) # MERRA-2 (SurfPres)
BrO_LTPred_Met = np.square(B0LT + (B1LT*PC1_Met) + (B2LT*PC2_Met) + (B3LT*PC3_Met)) # Met

# SkLearn (LinearRegression)
BrO_SurfPred_LR = np.square(LR0S  + (LR1S*Stand_O3)  + (LR2S*Stand_AEC)  + (LR3S*Stand_SurfTempMet)  + (LR4S*Stand_SurfPresMet)  + (LR5S*Stand_WS10mMet)  + (LR6S*Stand_MLH)  + (LR7S*Stand_P1hrMet)  + (LR8S*Stand_PTDif1000m)  + (LR9S*Stand_PTDif100m))
BrO_LTPred_LR   = np.square(LR0LT + (LR1LT*Stand_O3) + (LR2LT*Stand_AEC) + (LR3LT*Stand_SurfTempMet) + (LR4LT*Stand_SurfPresMet) + (LR5LT*Stand_WS10mMet) + (LR6LT*Stand_MLH) + (LR7LT*Stand_P1hrMet) + (LR8LT*Stand_PTDif1000m) + (LR9LT*Stand_PTDif100m))

# StatsModels (ols)
BrO_SurfPred_OLS = np.square(OLS0S  + (OLS1S*Stand_O3)  + (OLS2S*Stand_AEC)  + (OLS3S*Stand_SurfTempMet)  + (OLS4S*Stand_SurfPresMet)  + (OLS5S*Stand_WS10mMet)  + (OLS6S*Stand_MLH)  + (OLS7S*Stand_P1hrMet)  + (OLS8S*Stand_PTDif1000m)  + (OLS9S*Stand_PTDif100m))
BrO_LTPred_OLS   = np.square(OLS0LT + (OLS1LT*Stand_O3) + (OLS2LT*Stand_AEC) + (OLS3LT*Stand_SurfTempMet) + (OLS4LT*Stand_SurfPresMet) + (OLS5LT*Stand_WS10mMet) + (OLS6LT*Stand_MLH) + (OLS7LT*Stand_P1hrMet) + (OLS8LT*Stand_PTDif1000m) + (OLS9LT*Stand_PTDif100m))

# #------------------------------------------------------------------------------
# # APPLY THE BrO PCR MODEL

# #--------------
# # Intercept and Coefficients
# #--------------

# # Surface BrO
# B0S = 2.06*1e6 # Intercept for the multiple linear regression
# #B0S = 1.37*1e6 # Intercept for the multiple linear regression
# B1S = 1.46*1e5 # Slope PC1 (coefficient 1)
# B2S = 2.24*1e5 # Slope PC2 (coefficient 2)
# B3S = 3.94*1e5 # Slope PC3 (coefficient 3)

# # Lower tropospheric BrO
# B0LT = 3.67*1e6 # Intercept for the multiple linear regression
# B1LT = 3.66*1e5 # Slope PC1 (coefficient 1)
# B2LT = 9.88*1e4 # Slope PC2 (coefficient 2)
# B3LT = 5.97*1e5 # Slope PC3 (coefficient 3)

# # SkLearn (LinearRegression)
# LR0S = Intercept_resultS
# LR1S = Coefficients_resultS[0]
# LR2S = Coefficients_resultS[1]
# LR3S = Coefficients_resultS[2]

# LR0LT = Intercept_resultLT
# LR1LT = Coefficients_resultLT[0]
# LR2LT = Coefficients_resultLT[1]
# LR3LT = Coefficients_resultLT[2]

# # StatsModels (ols)
# OLS0S = Model_resultsS[0]
# OLS1S = Model_resultsS[1]
# OLS2S = Model_resultsS[2]
# OLS3S = Model_resultsS[3]

# OLS0LT = Model_resultsLT[0]
# OLS1LT = Model_resultsLT[1]
# OLS2LT = Model_resultsLT[2]
# OLS3LT = Model_resultsLT[3]

# #--------------
# # PCR model
# #--------------

# # Surface BrO
# # BrO_SurfPred_M2a = np.square(B0S + (B1S*PC1_M2a) + (B2S*PC2_M2a) + (B3S*PC3_M2a)) # MERRA-2 (SLP)
# # BrO_SurfPred_M2b = np.square(B0S + (B1S*PC1_M2b) + (B2S*PC2_M2b) + (B3S*PC3_M2b)) # MERRA-2 (SurfPres)
# BrO_SurfPred_Met = np.square(B0S + (B1S*PC1_Met) + (B2S*PC2_Met) + (B3S*PC3_Met)) # Met

# # Lower tropospheric BrO
# # BrO_LTPred_M2a = np.square(B0LT + (B1LT*PC1_M2a) + (B2LT*PC2_M2a) + (B3LT*PC3_M2a)) # MERRA-2 (SLP)
# # BrO_LTPred_M2b = np.square(B0LT + (B1LT*PC1_M2b) + (B2LT*PC2_M2b) + (B3LT*PC3_M2b)) # MERRA-2 (SurfPres)
# BrO_LTPred_Met = np.square(B0LT + (B1LT*PC1_Met) + (B2LT*PC2_Met) + (B3LT*PC3_Met)) # Met

# # SkLearn (LinearRegression)
# BrO_SurfPred_LR = np.square(LR0S  + (LR1S*PC1_Met)  + (LR2S*PC2_Met)  + (LR3S*PC3_Met))
# BrO_LTPred_LR   = np.square(LR0LT + (LR1LT*PC1_Met) + (LR2LT*PC2_Met) + (LR3LT*PC3_Met))

# # StatsModels (ols)
# BrO_SurfPred_OLS = np.square(OLS0S  + (OLS1S*PC1_Met)  + (OLS2S*PC2_Met)  + (OLS3S*PC3_Met))
# BrO_LTPred_OLS   = np.square(OLS0LT + (OLS1LT*PC1_Met) + (OLS2LT*PC2_Met) + (OLS3LT*PC3_Met))

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
#-------------------------------------
# Merge BrO observations & prediction
#-------------------------------------
# Predictions Swanson Coefficients
df1 = pd.concat([BrO_SurfPred_Met,BrO_LTPred_Met],axis=1,join='inner')        # Met BrOsurf & # Met BrOLTcol
# Observations
df1 = pd.concat([df1,dfBrO_Surf['BrO_(molecules/cm2)']],axis=1,join='outer')  # Add Obs BrOsurf
df1 = pd.concat([df1,dfBrO_LtCol['BrO_(molecules/cm2)']],axis=1,join='outer') # Add Obs BrOLTcol
# Predictions PCA Coefficients
df1 = pd.concat([df1,BrO_SurfPred_OLS],axis=1,join='outer')                   # Add BrOsurf  (OLS intercepts)
df1 = pd.concat([df1,BrO_LTPred_OLS],axis=1,join='outer')                     # Add BrOLTcol (OLS intercepts)
# Principle components
df1 = pd.concat([df1,PC1_Met],axis=1,join='outer')                            # Add PC1
df1 = pd.concat([df1,PC2_Met],axis=1,join='outer')                            # Add PC2
df1 = pd.concat([df1,PC3_Met],axis=1,join='outer')                            # Add PC3

# df1 = pd.concat([df1,BrO_SurfPred_M2b],axis=1,join='outer')            # Add MERRA-2 BrOsurf  (SurfPres)
# df1 = pd.concat([df1,BrO_LTPred_M2b],axis=1,join='outer')              # Add MERRA-2 BrOLTcol (SurfPres)
# df1 = pd.concat([df1,BrO_SurfPred_M2a],axis=1,join='outer')            # Add MERRA-2 BrOsurf (SLP)
# df1 = pd.concat([df1,BrO_LTPred_M2a],axis=1,join='outer')              # Add MERRA-2 BrOLTcol (SLP)
# df1 = pd.concat([df1,SurfBrO_Err],axis=1,join='outer')                 # Add Obs Error BrOsurf
# df1 = pd.concat([df1,LTBrO_Err],axis=1,join='outer')                   # Add Obs Error BrOLTcol
# df1 = pd.concat([df1,BrO_SurfPred_LR],axis=1,join='outer')             # Add BrOsurf  (LR intercepts)
# df1 = pd.concat([df1,BrO_LTPred_LR],axis=1,join='outer')               # Add BrOLTcol (LR intercepts)


# Name the columns
df1.columns = ['BrO_SurfPred_Met','BrO_LtColPred_Met','BrO_SurfObs','BrO_LtColObs','BrO_SurfPred_OLS','BrO_LTPred_OLS','PC1','PC2','PC3']

#------------------------------------------------------------------------------
# EXPORT THE DATAFRAMES AS .CSV

df1.to_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_Test.csv')
#df1.to_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_GridBox_BestChoice.csv')
