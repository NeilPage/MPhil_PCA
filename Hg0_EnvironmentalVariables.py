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
from statsmodels.stats.diagnostic import lilliefors
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.decomposition import PCA

# Date and Time handling package
from datetime import datetime,timedelta		# functions to handle date and time

#------------------------------------------------------------------------------
# DEFINE THE DATASETS

                         
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

#--------------
# SEA ICE CONTACT TIME
#--------------
Traj_V1_17  = pd.read_csv('/Users/ncp532/Documents/Data/SeaIce_Trajectories/Traj_V1_17_BrO.csv',  index_col=0)
Traj_V2_17  = pd.read_csv('/Users/ncp532/Documents/Data/SeaIce_Trajectories/Traj_V2_17_BrO.csv',  index_col=0)
Traj_V3_17M = pd.read_csv('/Users/ncp532/Documents/Data/SeaIce_Trajectories/Traj_V3_17M_BrO.csv', index_col=0)
Traj_V3_17D = pd.read_csv('/Users/ncp532/Documents/Data/SeaIce_Trajectories/Traj_V3_17D_BrO.csv', index_col=0)

Traj_V1_18  = pd.read_csv('/Users/ncp532/Documents/Data/SeaIce_Trajectories/Traj_V1_18_BrO.csv',  index_col=0)
Traj_V2_18  = pd.read_csv('/Users/ncp532/Documents/Data/SeaIce_Trajectories/Traj_V2_18_BrO.csv',  index_col=0)
Traj_V3_18M = pd.read_csv('/Users/ncp532/Documents/Data/SeaIce_Trajectories/Traj_V3_18M_BrO.csv', index_col=0)
Traj_V3_18D = pd.read_csv('/Users/ncp532/Documents/Data/SeaIce_Trajectories/Traj_V3_18D_BrO.csv', index_col=0)

#------------------------------------------------------------------------------
# Set the date

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
# SEA ICE CONTACT TIME
#--------------
Traj_V1_17.index   = (pd.to_datetime(Traj_V1_17.index,   dayfirst=True)) # Davis timezone is UT+7
Traj_V2_17.index   = (pd.to_datetime(Traj_V2_17.index,   dayfirst=True)) # Casey timezone is UT+8
Traj_V3_17M.index  = (pd.to_datetime(Traj_V3_17M.index,  dayfirst=True)) # Mawson timezone is UT+5
Traj_V3_17D.index  = (pd.to_datetime(Traj_V3_17D.index,  dayfirst=True)) # Davis timezone is UT+7

Traj_V1_18.index   = (pd.to_datetime(Traj_V1_18.index,   dayfirst=True)) # Davis timezone is UT+7
Traj_V2_18.index   = (pd.to_datetime(Traj_V2_18.index,   dayfirst=True)) # Casey timezone is UT+8
Traj_V3_18M.index  = (pd.to_datetime(Traj_V3_18M.index,  dayfirst=True)) # Mawson timezone is UT+5
Traj_V3_18D.index  = (pd.to_datetime(Traj_V3_18D.index,  dayfirst=True)) # Davis timezone is UT+7

#------------------------------------------------------------------------------
# CLEAN UP THE O3 DATA (REMOVE ERRONEOUS DATA)

# O3 (positive values only)
filter1   = O3_V1_17['O3_(ppb)']  >= 0
O3_V1_17  = O3_V1_17[filter1]

filter1   = O3_V2_17['O3_(ppb)']  >= 0
O3_V2_17  = O3_V2_17[filter1]

filter1   = O3_V3_17M['O3_(ppb)'] >= 0
O3_V3_17M = O3_V3_17M[filter1]

filter1   = O3_V3_17D['O3_(ppb)'] >= 0
O3_V3_17D = O3_V3_17D[filter1]

# O3 (get rid of stupidly high values)
filter2   = O3_V1_17['O3_(ppb)']  <= 50
O3_V1_17  = O3_V1_17[filter2]

filter2   = O3_V2_17['O3_(ppb)']  <= 50
O3_V2_17  = O3_V2_17[filter2]

filter2   = O3_V3_17M['O3_(ppb)'] <= 50
O3_V3_17M = O3_V3_17M[filter2]

filter2   = O3_V3_17D['O3_(ppb)'] <= 50
O3_V3_17D = O3_V3_17D[filter2]

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
# OZONE SCREEN

# Set the filter
F_O3_V1_17   = O3_V1_17['O3_(ppb)']
F_O3_V2_17   = O3_V2_17['O3_(ppb)']
F_O3_V3_17M  = O3_V3_17M['O3_(ppb)']
F_O3_V3_17D  = O3_V3_17D['O3_(ppb)']

F_O3_V1_18   = O3_V1_18['O3_(ppb)']
F_O3_V2_18   = O3_V2_18['O3_(ppb)']
F_O3_V3_18M  = O3_V3_18M['O3_(ppb)']
F_O3_V3_18D  = O3_V3_18D['O3_(ppb)']

F_O3_SIPEXII = O3_SIPEXII['O3_(ppb)']

# Apply the filter (Remove values when O3 <2 ppb)
OzoneF_V1_17   = F_O3_V1_17   > 2
O3_V1_17       = O3_V1_17[OzoneF_V1_17]

OzoneF_V2_17   = F_O3_V2_17   > 2
O3_V2_17       = O3_V2_17[OzoneF_V2_17]

OzoneF_V3_17M  = F_O3_V3_17M  > 2
O3_V3_17M      = O3_V3_17M[OzoneF_V3_17M]

OzoneF_V3_17D  = F_O3_V3_17D  > 2
O3_V3_17D      = O3_V3_17D[OzoneF_V3_17D]

OzoneF_V1_18   = F_O3_V1_18   > 2
O3_V1_18       = O3_V1_18[OzoneF_V1_18]

OzoneF_V2_18   = F_O3_V2_18   > 2
O3_V2_18       = O3_V2_18[OzoneF_V2_18]

OzoneF_V3_18M  = F_O3_V3_18M  > 2
O3_V3_18M      = O3_V3_18M[OzoneF_V3_18M]

OzoneF_V3_18D  = F_O3_V3_18D  > 2
O3_V3_18D      = O3_V3_18D[OzoneF_V3_18D]

OzoneF_SIPEXII = F_O3_SIPEXII > 2
O3_SIPEXII     = O3_SIPEXII[OzoneF_SIPEXII]

#------------------------------------------------------------------------------
# CONVERT THE DATASETS TO 1-HOUR TIME RESOLUTION

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
# O3
Davis        = (O3_V1_17.index >= start_date) & (O3_V1_17.index < end_date)
V1_17_O3     = O3_V1_17[Davis]
# Met
Davis        = (Met_V1_17.index >= start_date) & (Met_V1_17.index < end_date)
V1_17_Met    = Met_V1_17[Davis]
# MERRA2
Davis        = (MERRA2_V1_17.index >= start_date) & (MERRA2_V1_17.index < end_date)
V1_17_MERRA2 = MERRA2_V1_17[Davis]
# Traj
Davis        = (Traj_V1_17.index >= start_date) & (Traj_V1_17.index < end_date)
V1_17_Traj   = Traj_V1_17[Davis]

#-----------------------------
# V2_17 Casey (21-22 Dec 2017 and 26 Dec 2017 - 5 Jan 2018)
#-----------------------------
start_date1 = '2017-12-21'
end_date1 = '2017-12-23'
start_date2 = '2017-12-26'
end_date2 = '2018-01-06'
# O3
Casey1       = (O3_V2_17.index >= start_date1) & (O3_V2_17.index < end_date1)
Casey2       = (O3_V2_17.index >= start_date2) & (O3_V2_17.index < end_date2)
V2_17_O31    = O3_V2_17[Casey1]
V2_17_O32    = O3_V2_17[Casey2]
V2_17_O3     = pd.concat([V2_17_O31,V2_17_O32], axis =0)
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
# Traj
Casey1       = (Traj_V2_17.index >= start_date1) & (Traj_V2_17.index < end_date1)
Casey2       = (Traj_V2_17.index >= start_date2) & (Traj_V2_17.index < end_date2)
V2_17_Traj1= Traj_V2_17[Casey1]
V2_17_Traj2= Traj_V2_17[Casey2]
V2_17_Traj = pd.concat([V2_17_Traj1,V2_17_Traj2], axis =0)

#-----------------------------
# V3_17 Mawson (1-17 Feb 2018)
#-----------------------------
start_date    = '2018-02-01'
end_date      = '2018-02-18'
# O3
Mawson        = (O3_V3_17M.index >= start_date) & (O3_V3_17M.index < end_date)
V3_17_O3M     = O3_V3_17M[Mawson]
# Met
Mawson        = (Met_V3_17M.index >= start_date) & (Met_V3_17M.index < end_date)
V3_17_MetM    = Met_V3_17M[Mawson]
# MERRA2
Mawson        = (MERRA2_V3_17M.index >= start_date) & (MERRA2_V3_17M.index < end_date)
V3_17_MERRA2M = MERRA2_V3_17M[Mawson]
# Traj
Mawson        = (Traj_V3_17M.index >= start_date) & (Traj_V3_17M.index < end_date)
V3_17M_Traj   = Traj_V3_17M[Mawson]

#-----------------------------
# V3_17 Davis (27-30 Jan 2018 and 19-21 Feb 2018)
#-----------------------------
start_date1   = '2018-01-27'
end_date1     = '2018-01-31'
start_date2   = '2018-02-19'
end_date2     = '2018-02-22'
# O3
Davis1        = (O3_V3_17D.index >= start_date1) & (O3_V3_17D.index < end_date1)
Davis2        = (O3_V3_17D.index >= start_date2) & (O3_V3_17D.index < end_date2)
V3_17_O31     = O3_V3_17D[Davis1]
V3_17_O32     = O3_V3_17D[Davis2]
V3_17_O3D     = pd.concat([V3_17_O31,V3_17_O32], axis =0)
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
# Traj
Davis1        = (Traj_V3_17D.index >= start_date1) & (Traj_V3_17D.index < end_date1)
Davis2        = (Traj_V3_17D.index >= start_date2) & (Traj_V3_17D.index < end_date2)
V3_17D_Traj1  = Traj_V3_17D[Davis1]
V3_17D_Traj2  = Traj_V3_17D[Davis2]
V3_17D_Traj   = pd.concat([V3_17D_Traj1,V3_17D_Traj2], axis =0)

#-----------------------------
# V1_18 Davis (7-15 Nov 2018)
#-----------------------------
start_date   = '2018-11-07'
end_date     = '2018-11-16'
# O3
Davis        = (O3_V1_18.index >= start_date) & (O3_V1_18.index < end_date)
V1_18_O3     = O3_V1_18[Davis]
# Met
Davis        = (Met_V1_18.index >= start_date) & (Met_V1_18.index < end_date)
V1_18_Met    = Met_V1_18[Davis]
# MERRA2
Davis        = (MERRA2_V1_18.index >= start_date) & (MERRA2_V1_18.index < end_date)
V1_18_MERRA2 = MERRA2_V1_18[Davis]
# Traj
Davis        = (Traj_V1_18.index >= start_date) & (Traj_V1_18.index < end_date)
V1_18_Traj   = Traj_V1_18[Davis]

#-----------------------------
# V2_18 Casey (15-30 Dec 2018)
#-----------------------------
start_date   = '2018-12-15'
end_date     = '2018-12-31'
# O3
Casey        = (O3_V2_18.index >= start_date) & (O3_V2_18.index < end_date)
V2_18_O3     = O3_V2_18[Casey]
# Met
Casey        = (Met_V2_18.index >= start_date) & (Met_V2_18.index < end_date)
V2_18_Met    = Met_V2_18[Casey]
# MERRA2
Casey        = (MERRA2_V2_18.index >= start_date) & (MERRA2_V2_18.index < end_date)
V2_18_MERRA2 = MERRA2_V2_18[Casey]
# Traj
Casey        = (Traj_V2_18.index >= start_date) & (Traj_V2_18.index < end_date)
V2_18_Traj   = Traj_V2_18[Casey]

#-----------------------------
# V3_18 Mawson (30 Jan - 9 Feb 2019)
#-----------------------------
start_date    = '2019-01-30'
end_date      = '2019-02-10'
# O3
Mawson        = (O3_V3_18M.index >= start_date) & (O3_V3_18M.index < end_date)
V3_18_O3M     = O3_V3_18M[Mawson]
# Met
Mawson        = (Met_V3_18M.index >= start_date) & (Met_V3_18M.index < end_date)
V3_18_MetM    = Met_V3_18M[Mawson]
# MERRA2
Mawson        = (MERRA2_V3_18M.index >= start_date) & (MERRA2_V3_18M.index < end_date)
V3_18_MERRA2M = MERRA2_V3_18M[Mawson]
# Traj
Mawson        = (Traj_V3_18M.index >= start_date) & (Traj_V3_18M.index < end_date)
V3_18M_Traj   = Traj_V3_18M[Mawson]

#-----------------------------
# V3_18 Davis (26-28 Jan 2019 and 19-20 Feb 2019)
#-----------------------------
start_date1   = '2019-01-26'
end_date1     = '2019-01-29'
start_date2   = '2019-02-19'
end_date2     = '2019-02-21'
# O3
Davis1        = (O3_V3_18D.index >= start_date1) & (O3_V3_18D.index < end_date1)
Davis2        = (O3_V3_18D.index >= start_date2) & (O3_V3_18D.index < end_date2)
V3_18_O31    = O3_V3_18D[Davis1]
V3_18_O32    = O3_V3_18D[Davis2]
V3_18_O3D    = pd.concat([V3_18_O31,V3_18_O32], axis =0)
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
# Traj
Davis1        = (Traj_V3_18D.index >= start_date1) & (Traj_V3_18D.index < end_date1)
Davis2        = (Traj_V3_18D.index >= start_date2) & (Traj_V3_18D.index < end_date2)
V3_18D_Traj1  = Traj_V3_18D[Davis1]
V3_18D_Traj2  = Traj_V3_18D[Davis2]
V3_18D_Traj   = pd.concat([V3_18D_Traj1,V3_18D_Traj2], axis =0)

#-----------------------------
# SIPEXII (23 Sep to 11 Nov 2012)
#-----------------------------
start_date     = '2012-09-23'
end_date       = '2012-11-12'
# O3
SIPEX          = (O3_SIPEXII.index >= start_date) & (O3_SIPEXII.index < end_date)
SIPEXII_O3     = O3_SIPEXII[SIPEX]
# Met
SIPEX          = (Met_SIPEXII.index >= start_date) & (Met_SIPEXII.index < end_date)
SIPEXII_Met    = Met_SIPEXII[SIPEX]
# MERRA2
SIPEX          = (MERRA2_SIPEXII.index >= start_date) & (MERRA2_SIPEXII.index < end_date)
SIPEXII_MERRA2 = MERRA2_SIPEXII[SIPEX]

#------------------------------------------------------------------------------
# COMBINE THE DATAFRAMES FOR EACH VOYAGE INTO A SINGLE DATAFRAME

# O3 (Retrieval)
O3_All     = pd.concat([V1_17_O3,V2_17_O3,V3_17_O3M,V3_17_O3D,V1_18_O3,V2_18_O3,V3_18_O3M,V3_18_O3D],axis=0) # All

# Met
Met_All    = pd.concat([V1_17_Met,V2_17_Met,V3_17_MetM,V3_17_MetD,V1_18_Met,V2_18_Met,V3_18_MetM,V3_18_MetD],axis=0) # All

# MERRA2
MERRA2_All = pd.concat([V1_17_MERRA2,V2_17_MERRA2,V3_17_MERRA2M,V3_17_MERRA2D,V1_18_MERRA2,V2_18_MERRA2,V3_18_MERRA2M,V3_18_MERRA2D],axis=0) # All

# Sea Ice Contact All
Traj_All   = pd.concat([V1_17_Traj,V2_17_Traj,V3_17M_Traj,V3_17D_Traj,V1_18_Traj,V2_18_Traj,V3_18M_Traj,V3_18D_Traj],axis=0) # All

#------------------------------------------------------------------------------
# REMOVE NAN VALUES

O3_All = O3_All.dropna()

#------------------------------------------------------------------------------
# FILTER THE DATAFRAMES TO ONLY INCLUDE THE SAME DATES

# O3 & Met
dfEnv_Var = pd.concat([O3_All,    Met_All],    axis=1, join='inner')

# dfEnv_Var & MERRA2
dfEnv_Var = pd.concat([dfEnv_Var, MERRA2_All], axis=1, join='inner')

# dfEnv_Var &  Traj
dfEnv_Var = pd.concat([dfEnv_Var, Traj_All],   axis=1, join='inner')

#------------------------------------------------------------------------------
# DEFINE THE VARIABLES

# O3
O3             = dfEnv_Var['O3_(ppb)']                 # Surface O3 (ppb)

# Sea level pressure
SLPM2          = dfEnv_Var['SLP']/100                  # MERRA2 sea level pressure (hPa)
SurfPresM2     = dfEnv_Var['SurfPres']/100             # MERRA2 surface pressure (hPa)
SurfPresMet    = dfEnv_Var['atm_press_hpa']            # Met surface pressure (hPa)

# Surface temperature
SurfTempM2     = dfEnv_Var['Temp2m']-273.15            # MERRA2 temperature at 2m (C)
SurfTempMet    = dfEnv_Var[['temp_air_port_degc', 'temp_air_strbrd_degc']].mean(axis=1) # Met temperature average (port & strbrd side) (C)

# Change in pressure over 1 hour
P1hrM2a        = dfEnv_Var['P1hrM2a']                  # MERRA2 change in sea level pressure from one hour to next (hPa)
P1hrM2b        = dfEnv_Var['P1hrM2b']                  # MERRA2 change in sea level pressure from one hour to next (hPa)
P1hrMet        = dfEnv_Var['P1hrMet']                  # Met change in pressure from one hour to next (hPa)

# Potential temperature differential
PTDif100m      = dfEnv_Var['PTDif100m']                # Potential temperature differential lowest 100m (K)
PTDif1000m     = dfEnv_Var['PTDif1000m']               # Potential temperature differential lowest 1000m (K)

# Wind Speed
WS10mM2        = dfEnv_Var['WS10m']                    # MERRA2 wind speed at 10m (Kg/m2/s)
WS10mMet       = dfEnv_Var[['wnd_spd_port_corr_knot', 'wnd_spd_strbrd_corr_knot']].mean(axis=1) * 0.514444444 # Met wind speed average (port & strbrd side) (m/s)

# Wind Direction
WindDir        = dfEnv_Var[['wnd_dir_port_corr_deg', 'wnd_dir_strbrd_corr_deg']].mean(axis=1) # Wind direction (degrees)

# Mixing layer height
MLH            = dfEnv_Var['MLH']*1000                 # Richardson MLH (m)

# Solar Radiation
SolRad         = dfEnv_Var[['rad_slr_port_wperm2', 'rad_slr_strbrd_wperm2']].mean(axis=1) # Solar radiation (W/m2)

# Ice Contact
IceContact     = dfEnv_Var['Over_SeaIce']     # Time above sea ice (hours)
IceContact100m = dfEnv_Var['IceContact_100m'] # Time above sea ice & < 100m (hours)
IceContactMLH  = dfEnv_Var['IceContact_MLH']  # Time above sea ice & < MLH  (hours)

# Sea Ice Concentration 
SeaIceConc     = dfEnv_Var['Sea Ice Conc (0-1)'].replace(np.nan,0) # Sea ice concentration (%)
IceContactPerc = dfEnv_Var['ContactIcePerc']                       # Sum of Sea ice concentration (%) * Time above sea ice (hours)

# Land Contact
LandContact    = dfEnv_Var['Over_Land']    # Time above land (hours)
LandContactMLH = dfEnv_Var['Land_MLH']     # Time above land & < MLH (hours)

# Ocean Contact
OceanContact   = dfEnv_Var['Over_Ocean']   # Time above ocean (hours)

# Chlorophyll
Chlorophyll    = dfEnv_Var['chlorophyll_ugperl'] # Chlorophyll in seawater (ug/l)

# Water temperature (C)
Water_Temp     = dfEnv_Var['temp_sea_wtr_degc']

# Water sainity (C)
Water_Sal      = dfEnv_Var['salinity_optode_psu']

# Relative humidity (%)
RelHum         = dfEnv_Var[['rel_humidity_port_percent', 'rel_humidity_strbrd_percent']].mean(axis=1)

#------------------------------------------------------------------------------
# BUILD DATAFRAME FOR PCA VARIABLES

PCA_Variables = pd.concat([O3,SurfPresMet,SurfTempMet,P1hrMet,PTDif100m,PTDif1000m,WS10mMet,WindDir,MLH,SolRad,IceContact,IceContact100m,IceContactMLH,SeaIceConc,IceContactPerc,LandContact,LandContactMLH,OceanContact,Chlorophyll,Water_Temp,Water_Sal,RelHum],  axis=1, join='inner')
PCA_Variables.columns = ['O3','SurfPres','SurfTemp','P1hr','PTDif100m','PTDif1000m','WS10m','WindDir','MLH','SolRad','IceContact','IceContact100m','IceContactMLH','SeaIceConc','IceContactPerc','LandContact','LandContactMLH','OceanContact','Chlorophyll','Water_Temp','Water_Sal','RelHum']
PCA_Variables.to_csv('/Users/ncp532/Documents/Data/PCA_Hg0/PCA_EnvVar.csv')
