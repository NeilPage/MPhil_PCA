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
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.decomposition import PCA
from statsmodels.multivariate.pca import PCA as PCA2
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

# Date and Time handling package
from datetime import datetime,timedelta		# functions to handle date and time

#------------------------------------------------------------------------------
# DEFINE THE DATASETS

dfPCA = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/PCA_Variables3.csv', index_col=0)
dfPCA = dfPCA.dropna()

#------------------------------------------------------------------------------
# SET THE DATE

dfPCA.index = pd.to_datetime(dfPCA.index)
dfPCA.sort_index()

#------------------------------------------------------------------------------
# PERFORM A LOG TRANSFORMATION ON AEC & SQUARE-ROOT TRANSFORMATION ON BrO

dfPCA['log_AEC']       = np.log(dfPCA['AEC'])
dfPCA['sqrt_SurfBrO']  = np.sqrt(dfPCA['SurfBrO'])
dfPCA['sqrt_LTBrO']    = np.sqrt(dfPCA['LTBrO'])
dfPCA['sqrt_SurfBrO2'] = np.sqrt(dfPCA['SurfBrO2'])
dfPCA['sqrt_SurfBrO3'] = np.sqrt(dfPCA['SurfBrO3'])

#------------------------------------------------------------------------------
# CALCULATE THE STATISTICS

# Mean
dfPCA_Mean = dfPCA.mean()

# Median
dfPCA_Median = dfPCA.median()

# Min
dfPCA_Min = dfPCA.min()

# Max
dfPCA_Max = dfPCA.max()

# Std
dfPCA_Std = dfPCA.std()

# Mean - Std
dfPCA_MeanMStd = dfPCA_Mean - dfPCA_Std

# Mean + Std
dfPCA_MeanPStd = dfPCA_Mean + dfPCA_Std

#----------------------
# Standardised (Manual Method)
dfPCA_Standard  = (dfPCA - dfPCA_Mean) / dfPCA_Std

#----------------------
# Standardised (preprocessing.scale() function)
# NOTE: THIS METHOD GENERATES A USERWARNING 
dfPCA_Standard2 = preprocessing.scale(dfPCA)
dfPCA_Standard2 = pd.DataFrame(dfPCA_Standard2, index = dfPCA.index, columns = dfPCA.columns)

#----------------------
# Standardised (StandardScaler() function)
scale2 = StandardScaler()
dfPCA_Standard3 = scale2.fit_transform(dfPCA)
dfPCA_Standard3 = pd.DataFrame(dfPCA_Standard3, index = dfPCA.index, columns = dfPCA.columns)

#------------------------------------------------------------------------------
# SELECT STANDARD VARIABLES FOR THE PCA

# Swanson Variables
SwansonVariables    = dfPCA_Standard3.drop(['SurfBrO','LTBrO','sqrt_SurfBrO','sqrt_LTBrO','AEC','WindDir','SolRad','IceContact','IceContact100m',
                                            'IceContactMLH','SeaIceConc','IceContactPerc','LandContact','LandContactMLH','OceanContact','Weighted_Ice',
                                            'Weighted_Land','Weighted_Ocean','Percentage_Ice','Percentage_Land','Percentage_Ocean','Chlorophyll',
                                            'Water_Temp','Water_Sal','RelHum','InfraRed','Fluro','nd100m','SurfBrO2','SurfBrO3','sqrt_SurfBrO2','sqrt_SurfBrO3'], 1)

# All Additional Variables
#AdditionalVariables = dfPCA_Standard3.drop(['SurfBrO','LTBrO','sqrt_SurfBrO','sqrt_LTBrO','AEC'], 1)
AdditionalVariables = dfPCA_Standard3.drop(['SurfBrO','LTBrO','sqrt_SurfBrO','sqrt_LTBrO','AEC','IceContactPerc','LandContactMLH','Weighted_Ice','Weighted_Land','Weighted_Ocean',
                                            'Percentage_Ice','Percentage_Land','Percentage_Ocean','InfraRed','Fluro','nd100m','SurfBrO2','SurfBrO3','sqrt_SurfBrO2','sqrt_SurfBrO3'], 1)

# BrO_Surf Varaiables
BrOSurfVariables    = dfPCA_Standard3.drop(['SurfBrO','LTBrO','sqrt_SurfBrO','sqrt_LTBrO','AEC','IceContact100m','IceContactMLH','IceContactPerc',
                                            'LandContact','LandContactMLH','OceanContact','Weighted_Ice','Weighted_Land','Weighted_Ocean','Percentage_Ice',
                                            'Percentage_Land','Percentage_Ocean','P1hr','nd100m','InfraRed','Fluro','SurfPres','log_AEC','MLH','PTDif1000m',
                                            'WindDir','Chlorophyll','SurfBrO2','SurfBrO3','sqrt_SurfBrO2','sqrt_SurfBrO3'], 1)

# BrO_LTcol Varaiables
BrOLTcolVariables   = dfPCA_Standard3.drop(['SurfBrO','LTBrO','sqrt_SurfBrO','sqrt_LTBrO','AEC','IceContact100m','IceContactMLH','IceContactPerc',
                                            'LandContact','LandContactMLH','OceanContact','Weighted_Ice','Weighted_Land','Weighted_Ocean','Percentage_Ice',
                                            'Percentage_Land','Percentage_Ocean','P1hr','nd100m','InfraRed','Fluro','Water_Sal',
                                            'SurfBrO2','SurfBrO3','sqrt_SurfBrO2','sqrt_SurfBrO3'], 1)

#------------------------------------------------------------------------------
# PERFORM A PRINCIPAL COMPONENT ANALYSIS (PCA)

#--------------------
# BrO_Surf
#--------------------
# Apply the PCA (Swanson et al has 9 PCs, but only 3 have variances greater 1)
PCA_BrOS  = PCA() # All n components

# Retrieve the principal components (PCs)
PrincipalComponents_VariablesS  = PCA_BrOS.fit_transform(BrOSurfVariables) # Additional Variables

# Put the principle components into a DataFrame
Principal_Variables_DfS = pd.DataFrame(data = PrincipalComponents_VariablesS, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9',
                                                                                       'PC10'])#,'PC11','PC12','PC13','PC14','PC15','PC16','PC17',
                                                                                       #'PC18','PC19','PC20','PC21','PC22','PC23','PC24','PC25','PC26']) # Variables

                                                                                      
# Explained variation per principal component
print('Explained variation per principal component: {}'.format(PCA_BrOS.explained_variance_ratio_))
Explained_VarianceS = PCA_BrOS.explained_variance_ratio_

# Get the loadings
loadingsS  = pd.DataFrame(PCA_BrOS.components_.T,  columns=Principal_Variables_DfS.columns,index=BrOSurfVariables.columns)
loadingsS.to_csv('/Users/ncp532/Documents/Data/MERRA2/PC_Loadings_BrOSurf.csv')

# Calculate the normalised variance for each PC
NormVarS = np.mean(np.square(Principal_Variables_DfS - Principal_Variables_DfS.mean()))

# Calculate the sample error
SampleErrS = stats.sem(Principal_Variables_DfS)

#--------------------
# BrO_LTcol
#--------------------
# Apply the PCA (Swanson et al has 9 PCs, but only 3 have variances greater 1)
PCA_BrOLT = PCA() # All n components

# Retrieve the principal components (PCs)
PrincipalComponents_VariablesLT = PCA_BrOLT.fit_transform(BrOLTcolVariables) # Additional Variables

# Put the principle components into a DataFrame
Principal_Variables_DfLT = pd.DataFrame(data = PrincipalComponents_VariablesLT, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9',
                                                                                       'PC10','PC11','PC12','PC13','PC14','PC15'])#,'PC16','PC17',
                                                                                       #'PC18','PC19','PC20','PC21','PC22','PC23','PC24','PC25','PC26']) # Variables
    
# Explained variation per principal component
print('Explained variation per principal component: {}'.format(PCA_BrOLT.explained_variance_ratio_))
Explained_VarianceLT = PCA_BrOLT.explained_variance_ratio_

# Get the loadings
loadingsLT = pd.DataFrame(PCA_BrOLT.components_.T, columns=Principal_Variables_DfLT.columns,index=BrOLTcolVariables.columns)
loadingsLT.to_csv('/Users/ncp532/Documents/Data/MERRA2/PC_Loadings_BrOLTcol.csv')

# Calculate the normalised variance for each PC
NormVarLT = np.mean(np.square(Principal_Variables_DfLT - Principal_Variables_DfLT.mean()))

# Calculate the sample error
SampleErrLT = stats.sem(Principal_Variables_DfLT)

#--------------------
# How many variables are needed to explain variance in the data?
#--------------------
plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
x1 = np.arange(1, 16, step=1)
y1 = np.cumsum(PCA_BrOLT.explained_variance_ratio_)
x2 = np.arange(1, 11, step=1)
y2 = np.cumsum(PCA_BrOS.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(x1, y1, marker='o', linestyle='--', color='b', label="BrO_LTcol (cumulative variance)")
plt.plot(x1, PCA_BrOLT.explained_variance_ratio_, marker='o', linestyle='--', color='black', label="BrO_LTcol (variance)")
plt.plot(x2, y2, marker='o', linestyle='--', color='g', label="BrO_Surf (cumulative variance)")
plt.plot(x2, PCA_BrOS.explained_variance_ratio_, marker='o', linestyle='--', color='grey', label="BrO_Surf (variance)")

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 16, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(11, 0.90, '95% cut-off threshold', color = 'red', fontsize=16)

plt.axhline(y=0.6, color='r', linestyle='-')
plt.text(11, 0.55, 'Swanson 60% threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.legend(loc=2)
plt.show()

#------------------------------------------------------------------------------
# LOADINGS FOR THE PRINCIPLE COMPONENTS

#--------------------
# Swanson (2020)
#--------------------
O3_PC1_Swanson,      O3_PC2_Swanson,      O3_PC3_Swanson      = -0.021, 0.347,  -0.569 # Ozone (ppb) # (nmol/mol)?
EXT_PC1_Swanson,     EXT_PC2_Swanson,     EXT_PC3_Swanson     = 0.246,  0.270,  0.216  # Aerosol extinction (km-1) # (m/km)?
STemp_PC1_Swanson,   STemp_PC2_Swanson,   STemp_PC3_Swanson   = 0.087,  -0.392, -0.582 # Surface Temp (K) # (C)?
SLP_PC1_Swanson,     SLP_PC2_Swanson,     SLP_PC3_Swanson     = -0.338, 0.160,  0.231  # Sea level pressure (hPa)
VW10m_PC1_Swanson,   VW10m_PC2_Swanson,   VW10m_PC3_Swanson   = 0.345,  0.459,  -0.263 # Windspeed at 10m (m/s)
MLH_PC1_Swanson,     MLH_PC2_Swanson,     MLH_PC3_Swanson     = 0.595,  0.041,  -0.008 # Richardson mixed layer height (m)
P1hr_PC1_Swanson,    P1hr_PC2_Swanson,    P1hr_PC3_Swanson    = -0.007, -0.271, 0.196  # Change in pressure from one hour to next (hPa)
PT1000m_PC1_Swanson, PT1000m_PC2_Swanson, PT1000m_PC3_Swanson = -0.326, 0.580,  0.041  # Potential temperature differential in lowest 1000m (m/K)
PT100m_PC1_Swanson,  PT100m_PC2_Swanson,  PT100m_PC3_Swanson  = -0.487, -0.069, -0.358 # Potential temperature differential in lowest 100m (m/K)

#------------------------------------------------------------------------------
# CALCULATE THE STANDARD VARIABLE LOADINGS

#--------------------
# Swanson (2020)
#--------------------
Stand_O3_PC1,       Stand_O3_PC2,       Stand_O3_PC3       = (SwansonVariables['O3']*O3_PC1_Swanson),              (SwansonVariables['O3']*O3_PC2_Swanson),              (SwansonVariables['O3']*O3_PC3_Swanson)
Stand_AEC_PC1,      Stand_AEC_PC2,      Stand_AEC_PC3      = (SwansonVariables['log_AEC']*EXT_PC1_Swanson),        (SwansonVariables['log_AEC']*EXT_PC2_Swanson),        (SwansonVariables['log_AEC']*EXT_PC3_Swanson)
Stand_SurfTemp_PC1, Stand_SurfTemp_PC2, Stand_SurfTemp_PC3 = (SwansonVariables['SurfTemp']*STemp_PC1_Swanson),     (SwansonVariables['SurfTemp']*STemp_PC2_Swanson),     (SwansonVariables['SurfTemp']*STemp_PC3_Swanson)
Stand_SLP_PC1,      Stand_SLP_PC2,      Stand_SLP_PC3      = (SwansonVariables['SurfPres']*SLP_PC1_Swanson),       (SwansonVariables['SurfPres']*SLP_PC2_Swanson),       (SwansonVariables['SurfPres']*SLP_PC3_Swanson)
Stand_VW10m_PC1,    Stand_VW10m_PC2,    Stand_VW10m_PC3    = (SwansonVariables['WS10m']*VW10m_PC1_Swanson),        (SwansonVariables['WS10m']*VW10m_PC2_Swanson),        (SwansonVariables['WS10m']*VW10m_PC3_Swanson)
Stand_MLH_PC1,      Stand_MLH_PC2,      Stand_MLH_PC3      = (SwansonVariables['MLH']*MLH_PC1_Swanson),            (SwansonVariables['MLH']*MLH_PC2_Swanson),            (SwansonVariables['MLH']*MLH_PC3_Swanson)
Stand_P1hr_PC1,     Stand_P1hr_PC2,     Stand_P1hr_PC3     = (SwansonVariables['P1hr']*P1hr_PC1_Swanson),          (SwansonVariables['P1hr']*P1hr_PC2_Swanson),          (SwansonVariables['P1hr']*P1hr_PC3_Swanson)
Stand_PT1000m_PC1,  Stand_PT1000m_PC2,  Stand_PT1000m_PC3  = (SwansonVariables['PTDif1000m']*PT1000m_PC1_Swanson), (SwansonVariables['PTDif1000m']*PT1000m_PC2_Swanson), (SwansonVariables['PTDif1000m']*PT1000m_PC3_Swanson)
Stand_PT100m_PC1,   Stand_PT100m_PC2,   Stand_PT100m_PC3   = (SwansonVariables['PTDif100m']*PT100m_PC1_Swanson),   (SwansonVariables['PTDif100m']*PT100m_PC2_Swanson),   (SwansonVariables['PTDif100m']*PT100m_PC3_Swanson)

# Put values in a DataFrame
PC1_Loading_Swanson = pd.DataFrame([Stand_O3_PC1, Stand_AEC_PC1, Stand_SurfTemp_PC1, Stand_SLP_PC1, Stand_VW10m_PC1, Stand_MLH_PC1, Stand_P1hr_PC1, Stand_PT1000m_PC1, Stand_PT100m_PC1]).T
PC1_Loading_Swanson.columns = ['O3','AEC','SurfTemp','SLP','WS10m','MLH','P1hr','PTD1000m','PTD100m']
#PC1_Loading_Swanson.to_csv('/Users/ncp532/Documents/Data/MERRA2/PC1_Loading_Swanson.csv')

PC2_Loading_Swanson = pd.DataFrame([Stand_O3_PC2, Stand_AEC_PC2, Stand_SurfTemp_PC2, Stand_SLP_PC2, Stand_VW10m_PC2, Stand_MLH_PC2, Stand_P1hr_PC2, Stand_PT1000m_PC2, Stand_PT100m_PC2]).T
PC2_Loading_Swanson.columns = ['O3','AEC','SurfTemp','SLP','WS10m','MLH','P1hr','PTD1000m','PTD100m']
#PC2_Loading_Swanson.to_csv('/Users/ncp532/Documents/Data/MERRA2/PC2_Loading_Swanson.csv')

PC3_Loading_Swanson = pd.DataFrame([Stand_O3_PC3, Stand_AEC_PC3, Stand_SurfTemp_PC3, Stand_SLP_PC3, Stand_VW10m_PC3, Stand_MLH_PC3, Stand_P1hr_PC3, Stand_PT1000m_PC3, Stand_PT100m_PC3]).T
PC3_Loading_Swanson.columns = ['O3','AEC','SurfTemp','SLP','WS10m','MLH','P1hr','PTD1000m','PTD100m']
#PC3_Loading_Swanson.to_csv('/Users/ncp532/Documents/Data/MERRA2/PC3_Loading_Swanson.csv')

#--------------------
# BrO_Surf
#--------------------

# Transorm the loadings dataframe
loadingsST = loadingsS.T

SVL_PC1_BrOSurf  = BrOSurfVariables.mul(loadingsST.iloc[0],  axis='columns') # PC1
SVL_PC2_BrOSurf  = BrOSurfVariables.mul(loadingsST.iloc[1],  axis='columns') # PC2
SVL_PC3_BrOSurf  = BrOSurfVariables.mul(loadingsST.iloc[2],  axis='columns') # PC3
SVL_PC4_BrOSurf  = BrOSurfVariables.mul(loadingsST.iloc[3],  axis='columns') # PC4
SVL_PC5_BrOSurf  = BrOSurfVariables.mul(loadingsST.iloc[4],  axis='columns') # PC5
SVL_PC6_BrOSurf  = BrOSurfVariables.mul(loadingsST.iloc[5],  axis='columns') # PC6
SVL_PC7_BrOSurf  = BrOSurfVariables.mul(loadingsST.iloc[6],  axis='columns') # PC7
SVL_PC8_BrOSurf  = BrOSurfVariables.mul(loadingsST.iloc[7],  axis='columns') # PC8
SVL_PC9_BrOSurf  = BrOSurfVariables.mul(loadingsST.iloc[8],  axis='columns') # PC9
SVL_PC10_BrOSurf = BrOSurfVariables.mul(loadingsST.iloc[9],  axis='columns') # PC10
# SVL_PC11_BrOSurf = BrOSurfVariables.mul(loadingsST.iloc[10], axis='columns') # PC11
# SVL_PC12_BrOSurf = BrOSurfVariables.mul(loadingsST.iloc[11], axis='columns') # PC12
# SVL_PC13_BrOSurf = BrOSurfVariables.mul(loadingsST.iloc[12], axis='columns') # PC13
# SVL_PC14_BrOSurf = BrOSurfVariables.mul(loadingsST.iloc[13], axis='columns') # PC14
# SVL_PC15_BrOSurf = BrOSurfVariables.mul(loadingsST.iloc[14], axis='columns') # PC15
# SVL_PC16_BrOSurf = BrOSurfVariables.mul(loadingsST.iloc[15], axis='columns') # PC16
# SVL_PC17_BrOSurf = BrOSurfVariables.mul(loadingsST.iloc[16], axis='columns') # PC17
# SVL_PC18_BrOSurf = BrOSurfVariables.mul(loadingsST.iloc[17], axis='columns') # PC18

#--------------------
# BrO_LTcol
#--------------------

# Transorm the loadings dataframe
loadingsLTT = loadingsLT.T

SVL_PC1_BrOLTcol  = BrOLTcolVariables.mul(loadingsLTT.iloc[0],  axis='columns') # PC1
SVL_PC2_BrOLTcol  = BrOLTcolVariables.mul(loadingsLTT.iloc[1],  axis='columns') # PC2
SVL_PC3_BrOLTcol  = BrOLTcolVariables.mul(loadingsLTT.iloc[2],  axis='columns') # PC3
SVL_PC4_BrOLTcol  = BrOLTcolVariables.mul(loadingsLTT.iloc[3],  axis='columns') # PC4
SVL_PC5_BrOLTcol  = BrOLTcolVariables.mul(loadingsLTT.iloc[4],  axis='columns') # PC5
SVL_PC6_BrOLTcol  = BrOLTcolVariables.mul(loadingsLTT.iloc[5],  axis='columns') # PC6
SVL_PC7_BrOLTcol  = BrOLTcolVariables.mul(loadingsLTT.iloc[6],  axis='columns') # PC7
SVL_PC8_BrOLTcol  = BrOLTcolVariables.mul(loadingsLTT.iloc[7],  axis='columns') # PC8
SVL_PC9_BrOLTcol  = BrOLTcolVariables.mul(loadingsLTT.iloc[8],  axis='columns') # PC9
SVL_PC10_BrOLTcol = BrOLTcolVariables.mul(loadingsLTT.iloc[9],  axis='columns') # PC10
SVL_PC11_BrOLTcol = BrOLTcolVariables.mul(loadingsLTT.iloc[10], axis='columns') # PC11
SVL_PC12_BrOLTcol = BrOLTcolVariables.mul(loadingsLTT.iloc[11], axis='columns') # PC12
SVL_PC13_BrOLTcol = BrOLTcolVariables.mul(loadingsLTT.iloc[12], axis='columns') # PC13
SVL_PC14_BrOLTcol = BrOLTcolVariables.mul(loadingsLTT.iloc[13], axis='columns') # PC14
SVL_PC15_BrOLTcol = BrOLTcolVariables.mul(loadingsLTT.iloc[14], axis='columns') # PC15
#SVL_PC16_BrOLTcol = BrOLTcolVariables.mul(loadingsLTT.iloc[15], axis='columns') # PC16
# SVL_PC17_BrOLTcol = BrOLTcolVariables.mul(loadingsLTT.iloc[16], axis='columns') # PC17
# SVL_PC18_BrOLTcol = BrOLTcolVariables.mul(loadingsLTT.iloc[17], axis='columns') # PC18

#------------------------------------------------------------------------------
# CALCULATE THE PRINCIPLE COMPONENTS

#--------------------
# Swanson (2020)
#--------------------
PC1_Swanson = PC1_Loading_Swanson.sum(axis=1)
PC2_Swanson = PC2_Loading_Swanson.sum(axis=1)
PC3_Swanson = PC3_Loading_Swanson.sum(axis=1)

#--------------------
# BrO_Surf
#--------------------
PC1_S  = SVL_PC1_BrOSurf.sum(axis=1)
PC2_S  = SVL_PC2_BrOSurf.sum(axis=1)
PC3_S  = SVL_PC3_BrOSurf.sum(axis=1)
PC4_S  = SVL_PC4_BrOSurf.sum(axis=1)
PC5_S  = SVL_PC5_BrOSurf.sum(axis=1)
PC6_S  = SVL_PC6_BrOSurf.sum(axis=1)
PC7_S  = SVL_PC7_BrOSurf.sum(axis=1)
PC8_S  = SVL_PC8_BrOSurf.sum(axis=1)
PC9_S  = SVL_PC9_BrOSurf.sum(axis=1)
PC10_S = SVL_PC10_BrOSurf.sum(axis=1)
# PC11_S = SVL_PC11_BrOSurf.sum(axis=1)
# PC12_S = SVL_PC12_BrOSurf.sum(axis=1)
# PC13_S = SVL_PC13_BrOSurf.sum(axis=1)
# PC14_S = SVL_PC14_BrOSurf.sum(axis=1)
# PC15_S = SVL_PC15_BrOSurf.sum(axis=1)
# PC16_S = SVL_PC16_BrOSurf.sum(axis=1)
# PC17_S = SVL_PC17_BrOSurf.sum(axis=1)
# PC18_S = SVL_PC18_BrOSurf.sum(axis=1)

df_PC_BrOSurf = pd.concat([PC1_S,PC2_S,PC3_S,PC4_S,PC5_S,PC6_S,PC7_S,PC8_S,PC9_S,PC10_S], axis=1, join='inner')
df_PC_BrOSurf.columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']
df_PC_BrOSurf.to_csv('/Users/ncp532/Documents/Data/MERRA2/PC_BrOSurf.csv')

#--------------------
# BrO_LTcol
#--------------------
PC1_LT  = SVL_PC1_BrOLTcol.sum(axis=1)
PC2_LT  = SVL_PC2_BrOLTcol.sum(axis=1)
PC3_LT  = SVL_PC3_BrOLTcol.sum(axis=1)
PC4_LT  = SVL_PC4_BrOLTcol.sum(axis=1)
PC5_LT  = SVL_PC5_BrOLTcol.sum(axis=1)
PC6_LT  = SVL_PC6_BrOLTcol.sum(axis=1)
PC7_LT  = SVL_PC7_BrOLTcol.sum(axis=1)
PC8_LT  = SVL_PC8_BrOLTcol.sum(axis=1)
PC9_LT  = SVL_PC9_BrOLTcol.sum(axis=1)
PC10_LT = SVL_PC10_BrOLTcol.sum(axis=1)
PC11_LT = SVL_PC11_BrOLTcol.sum(axis=1)
PC12_LT = SVL_PC12_BrOLTcol.sum(axis=1)
PC13_LT = SVL_PC13_BrOLTcol.sum(axis=1)
PC14_LT = SVL_PC14_BrOLTcol.sum(axis=1)
PC15_LT = SVL_PC15_BrOLTcol.sum(axis=1)
#PC16_LT = SVL_PC16_BrOLTcol.sum(axis=1)
# PC17_LT = SVL_PC17_BrOLTcol.sum(axis=1)
# PC18_LT = SVL_PC18_BrOLTcol.sum(axis=1)

df_PC_BrOLTcol = pd.concat([PC1_LT,PC2_LT,PC3_LT,PC4_LT,PC5_LT,PC6_LT,PC7_LT,PC8_LT,PC9_LT,PC10_LT,PC11_LT,PC12_LT,PC13_LT,PC14_LT,PC15_LT], axis=1, join='inner')
df_PC_BrOLTcol.columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15']
df_PC_BrOLTcol.to_csv('/Users/ncp532/Documents/Data/MERRA2/PC_BrOLTcol.csv')

#------------------------------------------------------------------------------
# PERFORM A PRINCIPLE COMPONENT REGRESSION (PCR)
# (if z = sqrt(BrO_obs)) “z ~ pc1 + pc2 + pc3

#--------------------
# My loadings
#--------------------

# BrO Surf
dataPCA_S  = pd.DataFrame({'PC1': PC1_S, 'PC2': PC2_S, 'PC3': PC3_S, 'PC4': PC4_S, 'PC5': PC5_S, 'PC6': PC6_S, 'PC7': PC7_S, 'PC8': PC8_S, 'PC9': PC9_S, 'PC10': PC10_S, 'z': dfPCA['sqrt_SurfBrO']})

# BrO LTcol
dataPCA_LT = pd.DataFrame({'PC1': PC1_LT, 'PC2': PC2_LT, 'PC15': PC15_LT, 'z': dfPCA['sqrt_LTBrO']})

# Fit the model
model_PC_Surf  = ols("z ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10", dataPCA_S).fit()  # Surface StatsModel (ols)
model_PC_LTcol = ols("z ~ PC1 + PC2 + PC15", dataPCA_LT).fit() # LTcol StatsModel (ols)
 
# Retrieve the model results
Model_results_PC_Surf  = model_PC_Surf._results.params  # Surface StatsModel (ols)
Model_results_PC_LTcol = model_PC_LTcol._results.params # LTcol StatsModel (ols)

# Peform analysis of variance on fitted linear model
anova_results_PC_Surf  = anova_lm(model_PC_Surf)
anova_results_PC_LTcol = anova_lm(model_PC_LTcol)

#------------------------------------------------------------------------------
# PCR RETRIEVE THE INTERCEPT & COEFFICIENTS

#--------------
# Swanson (2020)
#--------------

# BrO Surf
B0S = 2.06*1e6 # Intercept
B1S = 1.46*1e5 # Coefficient PC1
B2S = 2.24*1e5 # Coefficient PC2
B3S = 3.94*1e5 # Coefficient PC3

# BrO LTcol
B0LT = 3.67*1e6 # Intercept
B1LT = 3.66*1e5 # Coefficient PC1
B2LT = 9.88*1e4 # Coefficient PC2
B3LT = 5.97*1e5 # Coefficient PC3

#--------------
# My loadings
#--------------

# BrO Surf
OLS_PC_0S  = Model_results_PC_Surf[0]  # Intercept
OLS_PC_1S  = Model_results_PC_Surf[1]  # Coefficient PC1
OLS_PC_2S  = Model_results_PC_Surf[2]  # Coefficient PC2
OLS_PC_3S  = Model_results_PC_Surf[3]  # Coefficient PC3
OLS_PC_4S  = Model_results_PC_Surf[4]  # Coefficient PC4
OLS_PC_5S  = Model_results_PC_Surf[5]  # Coefficient PC5
OLS_PC_6S  = Model_results_PC_Surf[6]  # Coefficient PC6
OLS_PC_7S  = Model_results_PC_Surf[7]  # Coefficient PC7
OLS_PC_8S  = Model_results_PC_Surf[8]  # Coefficient PC8
OLS_PC_9S  = Model_results_PC_Surf[9]  # Coefficient PC9
OLS_PC_10S = Model_results_PC_Surf[10] # Coefficient PC10
# OLS_PC_11S = Model_results_PC_Surf[11] # Coefficient PC11
# OLS_PC_12S = Model_results_PC_Surf[12] # Coefficient PC12
# OLS_PC_13S = Model_results_PC_Surf[13] # Coefficient PC13
# OLS_PC_14S = Model_results_PC_Surf[14] # Coefficient PC14
# OLS_PC_15S = Model_results_PC_Surf[15] # Coefficient PC15
# OLS_PC_16S = Model_results_PC_Surf[16] # Coefficient PC16
# OLS_PC_17S = Model_results_PC_Surf[17] # Coefficient PC17
# OLS_PC_18S = Model_results_PC_Surf[18] # Coefficient PC18

# BrO LTcol
OLS_PC_0LT  = Model_results_PC_LTcol[0]  # Intercept
OLS_PC_1LT  = Model_results_PC_LTcol[1]  # Coefficient PC1
OLS_PC_2LT  = Model_results_PC_LTcol[2]  # Coefficient PC2
#OLS_PC_3LT  = Model_results_PC_LTcol[3]  # Coefficient PC3
#OLS_PC_4LT  = Model_results_PC_LTcol[4]  # Coefficient PC4
#OLS_PC_5LT  = Model_results_PC_LTcol[4]  # Coefficient PC5
#OLS_PC_6LT  = Model_results_PC_LTcol[6]  # Coefficient PC6
#OLS_PC_7LT  = Model_results_PC_LTcol[5]  # Coefficient PC7
#OLS_PC_8LT  = Model_results_PC_LTcol[3]  # Coefficient PC8
#OLS_PC_9LT  = Model_results_PC_LTcol[7]  # Coefficient PC9
#OLS_PC_10LT = Model_results_PC_LTcol[5] # Coefficient PC10
#OLS_PC_11LT = Model_results_PC_LTcol[8] # Coefficient PC11
#OLS_PC_12LT = Model_results_PC_LTcol[9] # Coefficient PC12
#OLS_PC_13LT = Model_results_PC_LTcol[8] # Coefficient PC13
#OLS_PC_14LT = Model_results_PC_LTcol[3] # Coefficient PC14
OLS_PC_15LT = Model_results_PC_LTcol[3] # Coefficient PC15
#OLS_PC_16LT = Model_results_PC_LTcol[16] # Coefficient PC16
# OLS_PC_17LT = Model_results_PC_LTcol[17] # Coefficient PC17
# OLS_PC_18LT = Model_results_PC_LTcol[18] # Coefficient PC18

#------------------------------------------------------------------------------
# APPLY THE BrO PCR MODEL

#--------------
# Swanson (2020)
#--------------

Pred_BrOSurf_Swanson  = np.square(B0S  + (B1S*PC1_Swanson)  + (B2S*PC2_Swanson)  + (B3S*PC3_Swanson))
Pred_BrOLTcol_Swanson = np.square(B0LT + (B1LT*PC1_Swanson) + (B2LT*PC2_Swanson) + (B3LT*PC3_Swanson))

#--------------
# My loadings
#--------------
Pred_BrOSurf_OLS  = np.square(OLS_PC_0S  + (OLS_PC_1S*PC1_S)   + (OLS_PC_2S*PC2_S)   + (OLS_PC_3S*PC3_S)   + (OLS_PC_4S*PC4_S)   + (OLS_PC_5S*PC5_S)   + (OLS_PC_6S*PC6_S)   + (OLS_PC_7S*PC7_S)   + (OLS_PC_8S*PC8_S)   + (OLS_PC_9S*PC9_S)   + (OLS_PC_10S*PC10_S))
#Pred_BrOLTcol_OLS = np.square(OLS_PC_0LT + (OLS_PC_1LT*PC1_LT) + (OLS_PC_2LT*PC2_LT) + (OLS_PC_3LT*PC3_LT) + (OLS_PC_4LT*PC4_LT) + (OLS_PC_5LT*PC5_LT) + (OLS_PC_6LT*PC6_LT) + (OLS_PC_7LT*PC7_LT) + (OLS_PC_8LT*PC8_LT) + (OLS_PC_9LT*PC9_LT) + (OLS_PC_10LT*PC10_LT) + (OLS_PC_11LT*PC11_LT) + (OLS_PC_12LT*PC12_LT) + (OLS_PC_13LT*PC13_LT) + (OLS_PC_14LT*PC14_LT) + (OLS_PC_15LT*PC15_LT))
Pred_BrOLTcol_OLS = np.square(OLS_PC_0LT + (OLS_PC_1LT*PC1_LT) + (OLS_PC_2LT*PC2_LT) + (OLS_PC_15LT*PC15_LT))

SVL_PC1_BrOLTcol  = BrOLTcolVariables.mul(loadingsLTT.iloc[0],  axis='columns') # PC1

#------------------------------------------------------------------------------
# PERFORM A PARTIAL LEAST SQUARES (PLS) REGRESSION
# (if z = sqrt(BrO_obs)) “z ~ pc1 + pc2 + pc3

#--------------------
# My loadings
#--------------------

# Apply the PLS
PLS_Surf  = PLSRegression(n_components=3) # Surface PLS Regression
PLS_LTcol = PLSRegression(n_components=3) # LTcol PLS Regression

# Fit the model
PLS_Surf.fit(BrOSurfVariables,dfPCA['sqrt_SurfBrO'])
PLS_LTcol.fit(BrOLTcolVariables,dfPCA['sqrt_LTBrO'])

# Retrieve the model results
Pred_BrOSurf_PLS  = np.square(PLS_Surf.predict(BrOSurfVariables))  # Surface StatsModel (ols)
Pred_BrOLTcol_PLS = np.square(PLS_LTcol.predict(BrOLTcolVariables)) # LTcol StatsModel (ols)

# First we need to flatten the data: it's 2D layout is not relevent.
Pred_BrOSurf_PLS  = Pred_BrOSurf_PLS.flatten()
Pred_BrOLTcol_PLS = Pred_BrOLTcol_PLS.flatten()

#------------------------------------------------------------------------------
# INDIVIDUAL VARIABLE REGRESSION

#--------------
# BrO Surf
#--------------
dataS      = BrOSurfVariables
dataS['z'] = dfPCA['sqrt_SurfBrO']

# Swanson variables
slope_O3_S,             intercept_O3_S,             r_O3_S,             p_O3_S,             std_err_O3_S             = stats.linregress(dataS['z'], dataS['O3'])
#slope_AEC_S,            intercept_AEC_S,            r_AEC_S,            p_AEC_S,            std_err_AEC_S            = stats.linregress(dataS['z'], dataS['log_AEC'])
slope_SurfTemp_S,       intercept_SurfTemp_S,       r_SurfTemp_S,       p_SurfTemp_S,       std_err_SurfTemp_S       = stats.linregress(dataS['z'], dataS['SurfTemp'])
#slope_SLP_S,            intercept_SLP_S,            r_SLP_S,            p_SLP_S,            std_err_SLP_S            = stats.linregress(dataS['z'], dataS['SurfPres'])
slope_WS10m_S,          intercept_WS10m_S,          r_WS10m_S,          p_WS10m_S,          std_err_WS10m_S          = stats.linregress(dataS['z'], dataS['WS10m'])
#slope_MLH_S,            intercept_MLH_S,            r_MLH_S,            p_MLH_S,            std_err_MLH_S            = stats.linregress(dataS['z'], dataS['MLH'])
#slope_P1hr_S,           intercept_P1hr_S,           r_P1hr_S,           p_P1hr_S,           std_err_P1hr_S           = stats.linregress(dataS['z'], dataS['P1hr'])
#slope_PTD1000m_S,       intercept_PTD1000m_S,       r_PTD1000m_S,       p_PTD1000m_S,       std_err_PTD1000m_S       = stats.linregress(dataS['z'], dataS['PTDif1000m'])
slope_PTD100m_S,        intercept_PTD100m_S,        r_PTD100m_S,        p_PTD100m_S,        std_err_PTD100m_S        = stats.linregress(dataS['z'], dataS['PTDif100m'])

# Additional variables
#slope_WindDir_S,        intercept_WindDir_S,        r_WindDir_S,        p_WindDir_S,        std_err_WindDir_S        = stats.linregress(dataS['z'], dataS['WindDir'])
slope_SolRad_S,         intercept_SolRad_S,         r_SolRad_S,         p_SolRad_S,         std_err_SolRad_S         = stats.linregress(dataS['z'], dataS['SolRad'])
slope_IceContact_S,     intercept_IceContact_S,     r_IceContact_S,     p_IceContact_S,     std_err_IceContact_S     = stats.linregress(dataS['z'], dataS['IceContact'])
# slope_IceContact100m_S, intercept_IceContact100m_S, r_IceContact100m_S, p_IceContact100m_S, std_err_IceContact100m_S = stats.linregress(dataS['z'], dataS['IceContact100m'])
# slope_IceContactMLH_S,  intercept_IceContactMLH_S,  r_IceContactMLH_S,  p_IceContactMLH_S,  std_err_IceContactMLH_S  = stats.linregress(dataS['z'], dataS['IceContactMLH'])
slope_SeaIceConc_S,     intercept_SeaIceConc_S,     r_SeaIceConc_S,     p_SeaIceConc_S,     std_err_SeaIceConc_S     = stats.linregress(dataS['z'], dataS['SeaIceConc'])
# slope_IceContactPerc_S, intercept_IceContactPerc_S, r_IceContactPerc_S, p_IceContactPerc_S, std_err_IceContactPerc_S = stats.linregress(dataS['z'], dataS['IceContactPerc'])
# slope_LandContact_S,    intercept_LandContact_S,    r_LandContact_S,    p_LandContact_S,    std_err_LandContact_S    = stats.linregress(dataS['z'], dataS['LandContact'])
# slope_LandContactMLH_S, intercept_LandContactMLH_S, r_LandContactMLH_S, p_LandContactMLH_S, std_err_LandContactMLH_S = stats.linregress(dataS['z'], dataS['LandContactMLH'])
# slope_OceanContact_S,   intercept_OceanContact_S,   r_OceanContact_S,   p_OceanContact_S,   std_err_OceanContact_S   = stats.linregress(dataS['z'], dataS['OceanContact'])
# slope_WeightedIce_S,    intercept_WeightedIce_S,    r_WeightedIce_S,    p_WeightedIce_S,    std_err_WeightedIce_S    = stats.linregress(dataS['z'], dataS['Weighted_Ice'])
# slope_WeightedLand_S,   intercept_WeightedLand_S,   r_WeightedLand_S,   p_WeightedLand_S,   std_err_WeightedLand_S   = stats.linregress(dataS['z'], dataS['Weighted_Land'])
# slope_WeightedOcean_S,  intercept_WeightedOcean_S,  r_WeightedOcean_S,  p_WeightedOcean_S,  std_err_WeightedOcean_S  = stats.linregress(dataS['z'], dataS['Weighted_Ocean'])
# slope_PercentageIce_S,  intercept_PercentageIce_S,  r_PercentageIce_S,  p_PercentageIce_S,  std_err_PercentageIce_S  = stats.linregress(dataS['z'], dataS['Percentage_Ice'])
# slope_PercentageLand_S, intercept_PercentageLand_S, r_PercentageLand_S, p_PercentageLand_S, std_err_PercentageLand_S = stats.linregress(dataS['z'], dataS['Percentage_Land'])
# slope_PercentageOcean_S,intercept_PercentageOcean_S,r_PercentageOcean_S,p_PercentageOcean_S,std_err_PercentageOcean_S= stats.linregress(dataS['z'], dataS['Percentage_Ocean'])
#slope_Chloro_S,         intercept_Chloro_S,         r_Chloro_S,         p_Chloro_S,         std_err_Chloro_S         = stats.linregress(dataS['z'], dataS['Chlorophyll'])
slope_WaterTemp_S,      intercept_WaterTemp_S,      r_WaterTemp_S,      p_WaterTemp_S,      std_err_WaterTemp_S      = stats.linregress(dataS['z'], dataS['Water_Temp'])
slope_WaterSal_S,       intercept_WaterSal_S,       r_WaterSal_S,       p_WaterSal_S,       std_err_WaterSal_S       = stats.linregress(dataS['z'], dataS['Water_Sal'])
slope_RelHum_S,         intercept_RelHum_S,         r_RelHum_S,         p_RelHum_S,         std_err_RelHum_S         = stats.linregress(dataS['z'], dataS['RelHum'])
#slope_InfraRed_S,       intercept_InfraRed_S,       r_InfraRed_S,       p_InfraRed_S,       std_err_InfraRed_S       = stats.linregress(dataS['z'], dataS['InfraRed'])
#slope_Fluro_S,          intercept_Fluro_S,          r_Fluro_S,          p_Fluro_S,          std_err_Fluro_S          = stats.linregress(dataS['z'], dataS['Fluro'])

# R-squared
r2_O3_S             = r_O3_S**2
#r2_AEC_S            = r_AEC_S**2
r2_SurfTemp_S       = r_SurfTemp_S**2
#r2_SLP_S            = r_SLP_S**2
r2_WS10m_S          = r_WS10m_S**2
#r2_MLH_S            = r_MLH_S**2
#r2_P1hr_S           = r_P1hr_S**2
#r2_PTD1000m_S       = r_PTD1000m_S**2
r2_PTD100m_S        = r_PTD100m_S**2

# Additional variables
#r2_WindDir_S        = r_WindDir_S**2
r2_SolRad_S         = r_SolRad_S**2
r2_IceContact_S     = r_IceContact_S**2
# r2_IceContact100m_S = r_IceContact100m_S**2
# r2_IceContactMLH_S  = r_IceContactMLH_S**2
r2_SeaIceConc_S     = r_SeaIceConc_S**2
# r2_IceContactPerc_S = r_IceContactPerc_S**2
# r2_LandContact_S    = r_LandContact_S**2
# r2_LandContactMLH_S = r_LandContactMLH_S**2
# r2_OceanContact_S   = r_OceanContact_S**2
# r2_WeightedIce_S    = r_WeightedIce_S**2
# r2_WeightedLand_S   = r_WeightedLand_S**2
# r2_WeightedOcean_S  = r_WeightedOcean_S**2
# r2_PercentageIce_S  = r_PercentageIce_S**2
# r2_PercentageLand_S = r_PercentageLand_S**2
# r2_PercentageOcean_S= r_PercentageOcean_S**2
#r2_Chloro_S         = r_Chloro_S**2
r2_WaterTemp_S      = r_WaterTemp_S**2
r2_WaterSal_S       = r_WaterSal_S**2
r2_RelHum_S         = r_RelHum_S**2
#r2_InfraRed_S       = r_InfraRed_S**2
#r2_Fluro_S          = r_Fluro_S**2

#--------------
# BrO LTcol
#--------------
dataLT      = BrOLTcolVariables
dataLT['z'] = dfPCA['sqrt_LTBrO']

# Swanson variables
slope_O3_LT,             intercept_O3_LT,             r_O3_LT,             p_O3_LT,             std_err_O3_LT             = stats.linregress(dataLT['z'], dataLT['O3'])
slope_AEC_LT,            intercept_AEC_LT,            r_AEC_LT,            p_AEC_LT,            std_err_AEC_LT            = stats.linregress(dataLT['z'], dataLT['log_AEC'])
slope_SurfTemp_LT,       intercept_SurfTemp_LT,       r_SurfTemp_LT,       p_SurfTemp_LT,       std_err_SurfTemp_LT       = stats.linregress(dataLT['z'], dataLT['SurfTemp'])
slope_SLP_LT,            intercept_SLP_LT,            r_SLP_LT,            p_SLP_LT,            std_err_SLP_LT            = stats.linregress(dataLT['z'], dataLT['SurfPres'])
slope_WS10m_LT,          intercept_WS10m_LT,          r_WS10m_LT,          p_WS10m_LT,          std_err_WS10m_LT          = stats.linregress(dataLT['z'], dataLT['WS10m'])
slope_MLH_LT,            intercept_MLH_LT,            r_MLH_LT,            p_MLH_LT,            std_err_MLH_LT            = stats.linregress(dataLT['z'], dataLT['MLH'])
#slope_P1hr_LT,           intercept_P1hr_LT,           r_P1hr_LT,           p_P1hr_LT,           std_err_P1hr_LT           = stats.linregress(dataLT['z'], dataLT['P1hr'])
slope_PTD1000m_LT,       intercept_PTD1000m_LT,       r_PTD1000m_LT,       p_PTD1000m_LT,       std_err_PTD1000m_LT       = stats.linregress(dataLT['z'], dataLT['PTDif1000m'])
slope_PTD100m_LT,        intercept_PTD100m_LT,        r_PTD100m_LT,        p_PTD100m_LT,        std_err_PTD100m_LT        = stats.linregress(dataLT['z'], dataLT['PTDif100m'])

# Additional variables
slope_WindDir_LT,        intercept_WindDir_LT,        r_WindDir_LT,        p_WindDir_LT,        std_err_WindDir_LT        = stats.linregress(dataLT['z'], dataLT['WindDir'])
slope_SolRad_LT,         intercept_SolRad_LT,         r_SolRad_LT,         p_SolRad_LT,         std_err_SolRad_LT         = stats.linregress(dataLT['z'], dataLT['SolRad'])
slope_IceContact_LT,     intercept_IceContact_LT,     r_IceContact_LT,     p_IceContact_LT,     std_err_IceContact_LT     = stats.linregress(dataLT['z'], dataLT['IceContact'])
# slope_IceContact100m_LT, intercept_IceContact100m_LT, r_IceContact100m_LT, p_IceContact100m_LT, std_err_IceContact100m_LT = stats.linregress(dataLT['z'], dataLT['IceContact100m'])
# slope_IceContactMLH_LT,  intercept_IceContactMLH_LT,  r_IceContactMLH_LT,  p_IceContactMLH_LT,  std_err_IceContactMLH_LT  = stats.linregress(dataLT['z'], dataLT['IceContactMLH'])
slope_SeaIceConc_LT,     intercept_SeaIceConc_LT,     r_SeaIceConc_LT,     p_SeaIceConc_LT,     std_err_SeaIceConc_LT     = stats.linregress(dataLT['z'], dataLT['SeaIceConc'])
# slope_IceContactPerc_LT, intercept_IceContactPerc_LT, r_IceContactPerc_LT, p_IceContactPerc_LT, std_err_IceContactPerc_LT = stats.linregress(dataLT['z'], dataLT['IceContactPerc'])
# slope_LandContact_LT,    intercept_LandContact_LT,    r_LandContact_LT,    p_LandContact_LT,    std_err_LandContact_LT    = stats.linregress(dataLT['z'], dataLT['LandContact'])
# slope_LandContactMLH_LT, intercept_LandContactMLH_LT, r_LandContactMLH_LT, p_LandContactMLH_LT, std_err_LandContactMLH_LT = stats.linregress(dataLT['z'], dataLT['LandContactMLH'])
# slope_OceanContact_LT,   intercept_OceanContact_LT,   r_OceanContact_LT,   p_OceanContact_LT,   std_err_OceanContact_LT   = stats.linregress(dataLT['z'], dataLT['OceanContact'])
# slope_WeightedIce_LT,    intercept_WeightedIce_LT,    r_WeightedIce_LT,    p_WeightedIce_LT,    std_err_WeightedIce_LT    = stats.linregress(dataLT['z'], dataLT['Weighted_Ice'])
# slope_WeightedLand_LT,   intercept_WeightedLand_LT,   r_WeightedLand_LT,   p_WeightedLand_LT,   std_err_WeightedLand_LT   = stats.linregress(dataLT['z'], dataLT['Weighted_Land'])
# slope_WeightedOcean_LT,  intercept_WeightedOcean_LT,  r_WeightedOcean_LT,  p_WeightedOcean_LT,  std_err_WeightedOcean_LT  = stats.linregress(dataLT['z'], dataLT['Weighted_Ocean'])
# slope_PercentageIce_LT,  intercept_PercentageIce_LT,  r_PercentageIce_LT,  p_PercentageIce_LT,  std_err_PercentageIce_LT  = stats.linregress(dataLT['z'], dataLT['Percentage_Ice'])
# slope_PercentageLand_LT, intercept_PercentageLand_LT, r_PercentageLand_LT, p_PercentageLand_LT, std_err_PercentageLand_LT = stats.linregress(dataLT['z'], dataLT['Percentage_Land'])
# slope_PercentageOcean_LT,intercept_PercentageOcean_LT,r_PercentageOcean_LT,p_PercentageOcean_LT,std_err_PercentageOcean_LT= stats.linregress(dataLT['z'], dataLT['Percentage_Ocean'])
slope_Chloro_LT,         intercept_Chloro_LT,         r_Chloro_LT,         p_Chloro_LT,         std_err_Chloro_LT         = stats.linregress(dataLT['z'], dataLT['Chlorophyll'])
slope_WaterTemp_LT,      intercept_WaterTemp_LT,      r_WaterTemp_LT,      p_WaterTemp_LT,      std_err_WaterTemp_LT      = stats.linregress(dataLT['z'], dataLT['Water_Temp'])
#slope_WaterSal_LT,       intercept_WaterSal_LT,       r_WaterSal_LT,       p_WaterSal_LT,       std_err_WaterSal_LT       = stats.linregress(dataLT['z'], dataLT['Water_Sal'])
slope_RelHum_LT,         intercept_RelHum_LT,         r_RelHum_LT,         p_RelHum_LT,         std_err_RelHum_LT         = stats.linregress(dataLT['z'], dataLT['RelHum'])
#slope_InfraRed_LT,       intercept_InfraRed_LT,       r_InfraRed_LT,       p_InfraRed_LT,       std_err_InfraRed_LT       = stats.linregress(dataLT['z'], dataLT['InfraRed'])
#slope_Fluro_LT,          intercept_Fluro_LT,          r_Fluro_LT,          p_Fluro_LT,          std_err_Fluro_LT          = stats.linregress(dataLT['z'], dataLT['Fluro'])

# R-squared
r2_O3_LT             = r_O3_LT**2
r2_AEC_LT            = r_AEC_LT**2
r2_SurfTemp_LT       = r_SurfTemp_LT**2
r2_SLP_LT            = r_SLP_LT**2
r2_WS10m_LT          = r_WS10m_LT**2
r2_MLH_LT            = r_MLH_LT**2
#r2_P1hr_LT           = r_P1hr_LT**2
r2_PTD1000m_LT       = r_PTD1000m_LT**2
r2_PTD100m_LT        = r_PTD100m_LT**2

# Additional variables
r2_WindDir_LT        = r_WindDir_LT**2
r2_SolRad_LT         = r_SolRad_LT**2
r2_IceContact_LT     = r_IceContact_LT**2
# r2_IceContact100m_LT = r_IceContact100m_LT**2
# r2_IceContactMLH_LT  = r_IceContactMLH_LT**2
r2_SeaIceConc_LT     = r_SeaIceConc_LT**2
# r2_IceContactPerc_LT = r_IceContactPerc_LT**2
# r2_LandContact_LT    = r_LandContact_LT**2
# r2_LandContactMLH_LT = r_LandContactMLH_LT**2
# r2_OceanContact_LT   = r_OceanContact_LT**2
# r2_WeightedIce_LT    = r_WeightedIce_LT**2
# r2_WeightedLand_LT   = r_WeightedLand_LT**2
# r2_WeightedOcean_LT  = r_WeightedOcean_LT**2
# r2_PercentageIce_LT  = r_PercentageIce_LT**2
# r2_PercentageLand_LT = r_PercentageLand_LT**2
# r2_PercentageOcean_LT= r_PercentageOcean_LT**2
r2_Chloro_LT         = r_Chloro_LT**2
r2_WaterTemp_LT      = r_WaterTemp_LT**2
#r2_WaterSal_LT       = r_WaterSal_LT**2
r2_RelHum_LT         = r_RelHum_LT**2
#r2_InfraRed_LT       = r_InfraRed_LT**2
#r2_Fluro_LT          = r_Fluro_LT**2

# Correlation of SeaIceCover vs Chlorophyll
slope_SI_Chl_LT, intercept_SI_Chl_LT, r_SI_Chl_LT, p_SI_Chl_LT, std_err_SI_Chl_LT = stats.linregress(dataLT['SeaIceConc'], dataLT['Chlorophyll'])
r2_SI_Chl_LT = r_SI_Chl_LT**2

#--------------
# Additional variables
#--------------
# dfLinearRegression = pd.DataFrame({'R (BrO_Surf)':        [r_O3_S,   r_AEC_S,   r_SurfTemp_S,   r_SLP_S,   r_WS10m_S,   r_MLH_S,   r_P1hr_S,   r_PTD1000m_S,   r_PTD100m_S,   r_WindDir_S,   r_SolRad_S,   r_IceContact_S,   r_IceContact100m_S,   r_IceContactMLH_S,   r_SeaIceConc_S,   r_IceContactPerc_S,   r_LandContact_S,   r_LandContactMLH_S,   r_OceanContact_S,   r_WeightedIce_S,   r_WeightedLand_S,   r_WeightedOcean_S,   r_PercentageIce_S,   r_PercentageLand_S,   r_PercentageOcean_S,   r_Chloro_S],
#                                    'R^2 (BrO_Surf)':      [r2_O3_S,  r2_AEC_S,  r2_SurfTemp_S,  r2_SLP_S,  r2_WS10m_S,  r2_MLH_S,  r2_P1hr_S,  r2_PTD1000m_S,  r2_PTD100m_S,  r2_WindDir_S,  r2_SolRad_S,  r2_IceContact_S,  r2_IceContact100m_S,  r2_IceContactMLH_S,  r2_SeaIceConc_S,  r2_IceContactPerc_S,  r2_LandContact_S,  r2_LandContactMLH_S,  r2_OceanContact_S,  r2_WeightedIce_S,  r2_WeightedLand_S,  r2_WeightedOcean_S,  r2_PercentageIce_S,  r2_PercentageLand_S,  r2_PercentageOcean_S,  r2_Chloro_S],
#                                    'p-value (BrO_Surf)':  [p_O3_S,   p_AEC_S,   p_SurfTemp_S,   p_SLP_S,   p_WS10m_S,   p_MLH_S,   p_P1hr_S,   p_PTD1000m_S,   p_PTD100m_S,   p_WindDir_S,   p_SolRad_S,   p_IceContact_S,   p_IceContact100m_S,   p_IceContactMLH_S,   p_SeaIceConc_S,   p_IceContactPerc_S,   p_LandContact_S,   p_LandContactMLH_S,   p_OceanContact_S,   p_WeightedIce_S,   p_WeightedLand_S,   p_WeightedOcean_S,   p_PercentageIce_S,   p_PercentageLand_S,   p_PercentageOcean_S,   p_Chloro_S],
#                                    'R (BrO_LTcol)':       [r_O3_LT,  r_AEC_LT,  r_SurfTemp_LT,  r_SLP_LT,  r_WS10m_LT,  r_MLH_LT,  r_P1hr_LT,  r_PTD1000m_LT,  r_PTD100m_LT,  r_WindDir_LT,  r_SolRad_LT,  r_IceContact_LT,  r_IceContact100m_LT,  r_IceContactMLH_LT,  r_SeaIceConc_LT,  r_IceContactPerc_LT,  r_LandContact_LT,  r_LandContactMLH_LT,  r_OceanContact_LT,  r_WeightedIce_LT,  r_WeightedLand_LT,  r_WeightedOcean_LT,  r_PercentageIce_LT,  r_PercentageLand_LT,  r_PercentageOcean_LT,  r_Chloro_LT],
#                                    'R^2 (BrO_LTcol)':     [r2_O3_LT, r2_AEC_LT, r2_SurfTemp_LT, r2_SLP_LT, r2_WS10m_LT, r2_MLH_LT, r2_P1hr_LT, r2_PTD1000m_LT, r2_PTD100m_LT, r2_WindDir_LT, r2_SolRad_LT, r2_IceContact_LT, r2_IceContact100m_LT, r2_IceContactMLH_LT, r2_SeaIceConc_LT, r2_IceContactPerc_LT, r2_LandContact_LT, r2_LandContactMLH_LT, r2_OceanContact_LT, r2_WeightedIce_LT, r2_WeightedLand_LT, r2_WeightedOcean_LT, r2_PercentageIce_LT, r2_PercentageLand_LT, r2_PercentageOcean_LT, r2_Chloro_LT],
#                                    'p-value (BrO_LTcol)': [p_O3_LT,  p_AEC_LT,  p_SurfTemp_LT,  p_SLP_LT,  p_WS10m_LT,  p_MLH_LT,  p_P1hr_LT,  p_PTD1000m_LT,  p_PTD100m_LT,  p_WindDir_LT,  p_SolRad_LT,  p_IceContact_LT,  p_IceContact100m_LT,  p_IceContactMLH_LT,  p_SeaIceConc_LT,  p_IceContactPerc_LT,  p_LandContact_LT,  p_LandContactMLH_LT,  p_OceanContact_LT,  p_WeightedIce_LT,  p_WeightedLand_LT,  p_WeightedOcean_LT,  p_PercentageIce_LT,  p_PercentageLand_LT,  p_PercentageOcean_LT,  p_Chloro_LT]})
# dfLinearRegression.index = ['O3','AEC','SurfTemp','SLP','WS10m','MLH','P1hr','PTD1000m','PTD100m','WindDir','SolRad','IceContact','IceContact100m','IceContactMLH','SeaIceConc','IceContactPerc','LandContact','LandContactMLH','OceanContact','WeightedIce','WeightedLand','WeightedOcean','PercentageIce','PercentageLand','PercentageOcean','Chlorophyll']

#--------------
# BrO Surf
#--------------
dfLinearRegressionS = pd.DataFrame({'R (BrO_Surf)':       [r_O3_S,   r_SurfTemp_S,   r_WS10m_S,   r_PTD100m_S,   r_SolRad_S,   r_IceContact_S,   r_SeaIceConc_S,   r_WaterTemp_S,   r_WaterSal_S,   r_RelHum_S],
                                   'R^2 (BrO_Surf)':      [r2_O3_S,  r2_SurfTemp_S,  r2_WS10m_S,  r2_PTD100m_S,  r2_SolRad_S,  r2_IceContact_S,  r2_SeaIceConc_S,  r2_WaterTemp_S,  r2_WaterSal_S,  r2_RelHum_S],
                                   'p-value (BrO_Surf)':  [p_O3_S,   p_SurfTemp_S,   p_WS10m_S,   p_PTD100m_S,   p_SolRad_S,   p_IceContact_S,   p_SeaIceConc_S,   p_WaterTemp_S,   p_WaterSal_S,   p_RelHum_S]})
dfLinearRegressionS.index = ['O3','SurfTemp','WS10m','PTD100m','SolRad','IceContact','SeaIceConc','WaterTemp','WaterSal','RelHum']
dfLinearRegressionS.to_csv('/Users/ncp532/Documents/Data/MERRA2/LinearRegressionS.csv')

#--------------
# BrO LTcol
#--------------
dfLinearRegressionLT = pd.DataFrame({'R (BrO_LTcol)':     [r_O3_LT,  r_AEC_LT,  r_SurfTemp_LT,  r_SLP_LT,  r_WS10m_LT,  r_MLH_LT,  r_PTD1000m_LT,  r_PTD100m_LT,  r_WindDir_LT,  r_SolRad_LT,  r_IceContact_LT,  r_SeaIceConc_LT,  r_Chloro_LT,  r_WaterTemp_LT,  r_RelHum_LT],
                                   'R^2 (BrO_LTcol)':     [r2_O3_LT, r2_AEC_LT, r2_SurfTemp_LT, r2_SLP_LT, r2_WS10m_LT, r2_MLH_LT, r2_PTD1000m_LT, r2_PTD100m_LT, r2_WindDir_LT, r2_SolRad_LT, r2_IceContact_LT, r2_SeaIceConc_LT, r2_Chloro_LT, r2_WaterTemp_LT, r2_RelHum_LT],
                                   'p-value (BrO_LTcol)': [p_O3_LT,  p_AEC_LT,  p_SurfTemp_LT,  p_SLP_LT,  p_WS10m_LT,  p_MLH_LT,  p_PTD1000m_LT,  p_PTD100m_LT,  p_WindDir_LT,  p_SolRad_LT,  p_IceContact_LT,  p_SeaIceConc_LT,  p_Chloro_LT,  p_WaterTemp_LT,  p_RelHum_LT]})
dfLinearRegressionLT.index = ['O3','AEC','SurfTemp','SLP','WS10m','MLH','PTD1000m','PTD100m','WindDir','SolRad','IceContact','SeaIceConc','Chlorophyll','WaterTemp','RelHum']
dfLinearRegressionLT.to_csv('/Users/ncp532/Documents/Data/MERRA2/LinearRegressionLT.csv')


#------------------------------------------------------------------------------
# PERFORM MULTIPLE LINEAR REGRESSION (MLR)
# (if z = sqrt(BrO_obs)) “z ~ O3a + AECa + SurfTempa + SLPa +WS10ma + MLHa + P1hra + PTD1000ma + PTD100ma

# Fit the MLR model
# model_MLR_Surf  = ols("z ~ O3 + log_AEC + SurfTemp + SurfPres + WS10m + MLH + P1hr + PTDif1000m + PTDif100m + WindDir + SolRad + IceContact + IceContact100m + IceContactMLH + SeaIceConc + IceContactPerc + LandContact + LandContactMLH + OceanContact + Weighted_Ice + Weighted_Land + Weighted_Ocean + Percentage_Ice + Percentage_Land + Percentage_Ocean + Chlorophyll", dataS).fit()  # Surface StatsModel (ols)
# model_MLR_LTcol = ols("z ~ O3 + log_AEC + SurfTemp + SurfPres + WS10m + MLH + P1hr + PTDif1000m + PTDif100m + WindDir + SolRad + IceContact + IceContact100m + IceContactMLH + SeaIceConc + IceContactPerc + LandContact + LandContactMLH + OceanContact + Weighted_Ice + Weighted_Land + Weighted_Ocean + Percentage_Ice + Percentage_Land + Percentage_Ocean + Chlorophyll", dataLT).fit() # LTcol StatsModel (ols)

dataS      = BrOSurfVariables
dataS['z'] = dfPCA['sqrt_SurfBrO']
model_MLR_Surf  = ols("z ~ O3 + SurfTemp + WS10m + PTDif100m + SolRad + IceContact + SeaIceConc + Water_Temp + Water_Sal + RelHum", dataS).fit()  # Surface StatsModel (ols)

dataLTcol      = BrOLTcolVariables
dataLTcol['z'] = dfPCA['sqrt_LTBrO']
model_MLR_LTcol = ols("z ~ O3 + log_AEC + SurfTemp + SurfPres + WS10m + MLH + PTDif1000m + PTDif100m + WindDir + SolRad + IceContact + SeaIceConc + Chlorophyll + Water_Temp + RelHum", dataLT).fit() # LTcol StatsModel (ols)
 
# Retrieve the model results
Model_results_MLR_Surf  = model_MLR_Surf._results.params  # Surface StatsModel (ols)
Model_results_MLR_LTcol = model_MLR_LTcol._results.params # LTcol StatsModel   (ols)

# Model summary
Summary_MLR_Surf  = model_MLR_Surf.summary()
Summary_MLR_LTcol = model_MLR_LTcol.summary()

# Peform analysis of variance on fitted linear model
anova_results_MLR_Surf  = anova_lm(model_MLR_Surf)
anova_results_MLR_LTcol = anova_lm(model_MLR_LTcol)

#------------------------------------------------------------------------------
# Code for VIF Calculation

# Function to calculate the VIF values
def vif_cal(input_data, dependent_col):
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)
        #print (xvar_names[i], " RSQ = " , rsq)

# # Calculate the VIF values
# Vif_O3              = vif_cal(input_data=AdditionalVariables, dependent_col="O3")
# Vif_SurfPres        = vif_cal(input_data=AdditionalVariables, dependent_col="SurfPres")
# Vif_SurfTemp        = vif_cal(input_data=AdditionalVariables, dependent_col="SurfTemp")
# Vif_P1hr            = vif_cal(input_data=AdditionalVariables, dependent_col="P1hr")
# Vif_PTDif100m       = vif_cal(input_data=AdditionalVariables, dependent_col="PTDif100m")
# Vif_PTDif1000m      = vif_cal(input_data=AdditionalVariables, dependent_col="PTDif1000m")
# Vif_WS10m           = vif_cal(input_data=AdditionalVariables, dependent_col="WS10m")
# Vif_WindDir         = vif_cal(input_data=AdditionalVariables, dependent_col="WindDir")
# Vif_MLH             = vif_cal(input_data=AdditionalVariables, dependent_col="MLH")
# Vif_SolRad          = vif_cal(input_data=AdditionalVariables, dependent_col="SolRad")
# Vif_IceContact      = vif_cal(input_data=AdditionalVariables, dependent_col="IceContact")
# Vif_IceContact100m  = vif_cal(input_data=AdditionalVariables, dependent_col="IceContact100m")
# Vif_IceContactMLH   = vif_cal(input_data=AdditionalVariables, dependent_col="IceContactMLH")
# Vif_SeaIceConc      = vif_cal(input_data=AdditionalVariables, dependent_col="SeaIceConc")
# Vif_LandContact     = vif_cal(input_data=AdditionalVariables, dependent_col="LandContact")
# Vif_OceanContact    = vif_cal(input_data=AdditionalVariables, dependent_col="OceanContact")
# Vif_Chlorophyll     = vif_cal(input_data=AdditionalVariables, dependent_col="Chlorophyll")
# Vif_WaterTemp       = vif_cal(input_data=AdditionalVariables, dependent_col="Water_Temp")
# Vif_WaterSal        = vif_cal(input_data=AdditionalVariables, dependent_col="Water_Sal")
# Vif_RelHum          = vif_cal(input_data=AdditionalVariables, dependent_col="RelHum")
# Vif_AEC             = vif_cal(input_data=AdditionalVariables, dependent_col="log_AEC")

#------------------------------------------------------------------------------
# MLR RETRIEVE THE INTERCEPT & COEFFICIENTS

#--------------------
# BrO Surf
#--------------------
# Swanson variables
OLS_MLR_0S  = Model_results_MLR_Surf[0]  # Intercept
OLS_MLR_1S  = Model_results_MLR_Surf[1]  # Coefficient O3
#OLS_MLR_2S  = Model_results_MLR_Surf[2]  # Coefficient log_AEC
OLS_MLR_2S  = Model_results_MLR_Surf[2]  # Coefficient SurfTemp
#OLS_MLR_4S  = Model_results_MLR_Surf[4]  # Coefficient SurfPres
OLS_MLR_3S  = Model_results_MLR_Surf[3]  # Coefficient PTD100m
OLS_MLR_4S  = Model_results_MLR_Surf[4]  # Coefficient WS10m
#OLS_MLR_6S  = Model_results_MLR_Surf[6]  # Coefficient MLH
#OLS_MLR_7S  = Model_results_MLR_Surf[7]  # Coefficient P1hr
#OLS_MLR_8S  = Model_results_MLR_Surf[7]  # Coefficient PTD1000m

# Additional variables
# OLS_MLR_10S = Model_results_MLR_Surf[10] # Coefficient WindDir
OLS_MLR_5S  = Model_results_MLR_Surf[5] # Coefficient SolRad
OLS_MLR_6S  = Model_results_MLR_Surf[6] # Coefficient IceContact
# OLS_MLR_13S = Model_results_MLR_Surf[13] # Coefficient IceContact100m
# OLS_MLR_14S = Model_results_MLR_Surf[14] # Coefficient IceContactMLH
OLS_MLR_7S  = Model_results_MLR_Surf[7] # Coefficient SeaIceConc
# OLS_MLR_16S = Model_results_MLR_Surf[16] # Coefficient IceContactPerc
# OLS_MLR_17S = Model_results_MLR_Surf[17] # Coefficient LandContact
# OLS_MLR_18S = Model_results_MLR_Surf[18] # Coefficient LandContactMLH
# OLS_MLR_19S = Model_results_MLR_Surf[19] # Coefficient OceanContact
# OLS_MLR_20S = Model_results_MLR_Surf[20] # Coefficient WeightedIce
# OLS_MLR_21S = Model_results_MLR_Surf[21] # Coefficient WeightedLand
# OLS_MLR_22S = Model_results_MLR_Surf[22] # Coefficient WeightedOcean
# OLS_MLR_23S = Model_results_MLR_Surf[23] # Coefficient PercentageIce
# OLS_MLR_24S = Model_results_MLR_Surf[24] # Coefficient PercentageLand
# OLS_MLR_25S = Model_results_MLR_Surf[25] # Coefficient PercentageOcean
# OLS_MLR_26S = Model_results_MLR_Surf[26] # Coefficient Chlorophyll
OLS_MLR_8S  = Model_results_MLR_Surf[8] # Coefficient WaterTemp
OLS_MLR_9S  = Model_results_MLR_Surf[9] # Coefficient WaterSal
OLS_MLR_10S = Model_results_MLR_Surf[10] # Coefficient RelHum
#OLS_MLR_17S = Model_results_MLR_Surf[16] # Coefficient InfraRed
#OLS_MLR_18S = Model_results_MLR_Surf[17] # Coefficient Fluroescence

#--------------------
# BrO LTcol
#--------------------
# Swanson variables
OLS_MLR_0LT  = Model_results_MLR_LTcol[0]  # Intercept
OLS_MLR_1LT  = Model_results_MLR_LTcol[1]  # Coefficient O3
OLS_MLR_2LT  = Model_results_MLR_LTcol[2]  # Coefficient SurfPres
OLS_MLR_3LT  = Model_results_MLR_LTcol[3]  # Coefficient SurfTemp
OLS_MLR_4LT  = Model_results_MLR_LTcol[4]  # Coefficient PTD100m
OLS_MLR_5LT  = Model_results_MLR_LTcol[5]  # Coefficient PTD1000m
OLS_MLR_6LT  = Model_results_MLR_LTcol[6]  # Coefficient WS10m
OLS_MLR_7LT  = Model_results_MLR_LTcol[7]  # Coefficient WindDir
OLS_MLR_8LT  = Model_results_MLR_LTcol[8]  # Coefficient MLH
#OLS_MLR_7LT  = Model_results_MLR_LTcol[7]  # Coefficient P1hr

# Additional variables
# OLS_MLR_10LT = Model_results_MLR_LTcol[10] # Coefficient WindDir
# OLS_MLR_11LT = Model_results_MLR_LTcol[11] # Coefficient SolRad
# OLS_MLR_12LT = Model_results_MLR_LTcol[12] # Coefficient IceContact
# OLS_MLR_13LT = Model_results_MLR_LTcol[13] # Coefficient IceContact100m
# OLS_MLR_14LT = Model_results_MLR_LTcol[14] # Coefficient IceContactMLH
# OLS_MLR_15LT = Model_results_MLR_LTcol[15] # Coefficient SeaIceConc
# OLS_MLR_16LT = Model_results_MLR_LTcol[16] # Coefficient IceContactPerc
# OLS_MLR_17LT = Model_results_MLR_LTcol[17] # Coefficient LandContact
# OLS_MLR_18LT = Model_results_MLR_LTcol[18] # Coefficient LandContactMLH
# OLS_MLR_19LT = Model_results_MLR_LTcol[19] # Coefficient OceanContact
# OLS_MLR_20LT = Model_results_MLR_LTcol[20] # Coefficient WeightedIce
#OLS_MLR_7LT = Model_results_MLR_LTcol[21] # Coefficient WeightedLand
# OLS_MLR_22LT = Model_results_MLR_LTcol[22] # Coefficient WeightedOcean
# OLS_MLR_23LT = Model_results_MLR_LTcol[23] # Coefficient PercentageIce
# OLS_MLR_24LT = Model_results_MLR_LTcol[24] # Coefficient PercentageLand
# OLS_MLR_25LT = Model_results_MLR_LTcol[25] # Coefficient PercentageOcean
# OLS_MLR_26LT = Model_results_MLR_LTcol[26] # Coefficient Chlorophyll


OLS_MLR_9LT  = Model_results_MLR_LTcol[9]  # Coefficient SolRad
OLS_MLR_10LT = Model_results_MLR_LTcol[10] # Coefficient IceContact
OLS_MLR_11LT = Model_results_MLR_LTcol[11] # Coefficient SeaIceConc
#OLS_MLR_7LT  = Model_results_MLR_LTcol[13] # Coefficient WeightedLand
OLS_MLR_12LT = Model_results_MLR_LTcol[12] # Coefficient Chlorophyll
OLS_MLR_13LT = Model_results_MLR_LTcol[13] # Coefficient WaterTemp
#OLS_MLR_15LT = Model_results_MLR_LTcol[15] # Coefficient WaterSal
OLS_MLR_14LT = Model_results_MLR_LTcol[14] # Coefficient RelHum
#OLS_MLR_17LT = Model_results_MLR_LTcol[17] # Coefficient InfraRed
#OLS_MLR_15LT = Model_results_MLR_LTcol[15] # Coefficient Fluroescence
OLS_MLR_15LT = Model_results_MLR_LTcol[15] # Coefficient log_AEC

#------------------------------------------------------------------------------
# APPLY THE BrO MLR MODEL

# Pred_BrOSurf_MLR  = np.square(OLS_MLR_0S  + (OLS_MLR_1S*AdditionalVariables['O3'])  + (OLS_MLR_2S*AdditionalVariables['log_AEC'])  + (OLS_MLR_3S*AdditionalVariables['SurfTemp'])  + (OLS_MLR_4S*AdditionalVariables['SurfPres'])  + (OLS_MLR_5S*AdditionalVariables['WS10m'])  + (OLS_MLR_6S*AdditionalVariables['MLH'])  + (OLS_MLR_7S*AdditionalVariables['P1hr'])  + (OLS_MLR_8S*AdditionalVariables['PTDif1000m'])  + (OLS_MLR_9S*AdditionalVariables['PTDif100m'])  + (OLS_MLR_10S*AdditionalVariables['WindDir'])  + (OLS_MLR_11S*AdditionalVariables['SolRad'])  + (OLS_MLR_12S*AdditionalVariables['IceContact'])  + (OLS_MLR_13S*AdditionalVariables['IceContact100m'])  + (OLS_MLR_14S*AdditionalVariables['IceContactMLH'])  + (OLS_MLR_15S*AdditionalVariables['SeaIceConc'])  + (OLS_MLR_16S*AdditionalVariables['IceContactPerc']) + (OLS_MLR_17S*AdditionalVariables['LandContact'])  + (OLS_MLR_18S*AdditionalVariables['LandContactMLH'])  + (OLS_MLR_19S*AdditionalVariables['OceanContact'])  + (OLS_MLR_20S*AdditionalVariables['Weighted_Ice'])  + (OLS_MLR_21S*AdditionalVariables['Weighted_Land'])  + (OLS_MLR_22S*AdditionalVariables['Weighted_Ocean'])  + (OLS_MLR_23S*AdditionalVariables['Percentage_Ice'])  + (OLS_MLR_24S*AdditionalVariables['Percentage_Land'])  + (OLS_MLR_25S*AdditionalVariables['Percentage_Ocean'])  + (OLS_MLR_26S*AdditionalVariables['Chlorophyll']))
# Pred_BrOLTcol_MLR = np.square(OLS_MLR_0LT + (OLS_MLR_1LT*AdditionalVariables['O3']) + (OLS_MLR_2LT*AdditionalVariables['log_AEC']) + (OLS_MLR_3LT*AdditionalVariables['SurfTemp']) + (OLS_MLR_4LT*AdditionalVariables['SurfPres']) + (OLS_MLR_5LT*AdditionalVariables['WS10m']) + (OLS_MLR_6LT*AdditionalVariables['MLH']) + (OLS_MLR_7LT*AdditionalVariables['P1hr']) + (OLS_MLR_8LT*AdditionalVariables['PTDif1000m']) + (OLS_MLR_9LT*AdditionalVariables['PTDif100m']) + (OLS_MLR_10LT*AdditionalVariables['WindDir']) + (OLS_MLR_11LT*AdditionalVariables['SolRad']) + (OLS_MLR_12LT*AdditionalVariables['IceContact']) + (OLS_MLR_13LT*AdditionalVariables['IceContact100m']) + (OLS_MLR_14LT*AdditionalVariables['IceContactMLH']) + (OLS_MLR_15LT*AdditionalVariables['SeaIceConc']) + (OLS_MLR_16S*AdditionalVariables['IceContactPerc']) + (OLS_MLR_17LT*AdditionalVariables['LandContact']) + (OLS_MLR_18LT*AdditionalVariables['LandContactMLH']) + (OLS_MLR_19LT*AdditionalVariables['OceanContact']) + (OLS_MLR_20LT*AdditionalVariables['Weighted_Ice']) + (OLS_MLR_21LT*AdditionalVariables['Weighted_Land']) + (OLS_MLR_22LT*AdditionalVariables['Weighted_Ocean']) + (OLS_MLR_23LT*AdditionalVariables['Percentage_Ice']) + (OLS_MLR_24LT*AdditionalVariables['Percentage_Land']) + (OLS_MLR_25LT*AdditionalVariables['Percentage_Ocean']) + (OLS_MLR_26LT*AdditionalVariables['Chlorophyll']))
Pred_BrOSurf_MLR  = np.square(OLS_MLR_0S  + (OLS_MLR_1S*BrOSurfVariables['O3'])   + (OLS_MLR_2S*BrOSurfVariables['SurfTemp'])   + (OLS_MLR_3S*BrOSurfVariables['PTDif100m'])  + (OLS_MLR_4S*BrOSurfVariables['WS10m'])       + (OLS_MLR_5S*BrOSurfVariables['SolRad'])       + (OLS_MLR_6S*BrOSurfVariables['IceContact']) + (OLS_MLR_7S*BrOSurfVariables['SeaIceConc']) + (OLS_MLR_8S*BrOSurfVariables['Water_Temp']) + (OLS_MLR_9S*BrOSurfVariables['Water_Sal']) + (OLS_MLR_10S*BrOSurfVariables['RelHum']))
Pred_BrOLTcol_MLR = np.square(OLS_MLR_0LT + (OLS_MLR_1LT*BrOLTcolVariables['O3']) + (OLS_MLR_2LT*BrOLTcolVariables['SurfPres']) + (OLS_MLR_3LT*BrOLTcolVariables['SurfTemp']) + (OLS_MLR_4LT*BrOLTcolVariables['PTDif100m']) + (OLS_MLR_5LT*BrOLTcolVariables['PTDif1000m']) + (OLS_MLR_6LT*BrOLTcolVariables['WS10m'])    + (OLS_MLR_7LT*BrOLTcolVariables['WindDir'])  + (OLS_MLR_8LT*BrOLTcolVariables['MLH'])      + (OLS_MLR_9LT*BrOLTcolVariables['SolRad'])  + (OLS_MLR_10LT*BrOLTcolVariables['IceContact']) + (OLS_MLR_11LT*BrOLTcolVariables['SeaIceConc']) + (OLS_MLR_12LT*BrOLTcolVariables['Chlorophyll']) + (OLS_MLR_13LT*BrOLTcolVariables['Water_Temp']) + (OLS_MLR_14LT*BrOLTcolVariables['RelHum']) + (OLS_MLR_15LT*BrOLTcolVariables['log_AEC']))

#------------------------------------------------------------------------------
#BUILD DATAFRAME FOR THE PCR MODEL INTERCEPTS & COEFFICIENTS

# Build a pandas dataframe
dfPC_IntCoef = {'Intercept (B0)':       [OLS_PC_0S, OLS_PC_0LT],
                'Coefficient 1 (B1)':   [OLS_PC_1S, OLS_PC_1LT],
                'Coefficient 2 (B2)':   [OLS_PC_2S, OLS_PC_2LT],
                'Coefficient 3 (B3)':   [OLS_PC_3S, np.nan],
                'Coefficient 4 (B4)':   [OLS_PC_4S, np.nan],
                'Coefficient 5 (B5)':   [OLS_PC_5S, np.nan],
                'Coefficient 6 (B6)':   [OLS_PC_6S, np.nan],
                'Coefficient 7 (B7)':   [OLS_PC_7S, np.nan],
                'Coefficient 8 (B8)':   [OLS_PC_8S, np.nan],
                'Coefficient 9 (B9)':   [OLS_PC_9S, np.nan],
                'Coefficient 10 (B10)': [OLS_PC_10S,np.nan],
                'Coefficient 11 (B11)': [np.nan,    np.nan],
                'Coefficient 12 (B12)': [np.nan,    np.nan],
                'Coefficient 13 (B13)': [np.nan,    np.nan],
                'Coefficient 14 (B14)': [np.nan,    np.nan],
                'Coefficient 15 (B15)': [np.nan,    OLS_PC_15LT]}
dfPC_IntCoef = pd.DataFrame(dfPC_IntCoef, index = ['ols_Surf','ols_LTcol'],columns = ['Intercept (B0)','Coefficient 1 (B1)','Coefficient 2 (B2)','Coefficient 3 (B3)','Coefficient 4 (B4)',
                                                                                      'Coefficient 5 (B5)','Coefficient 6 (B6)','Coefficient 7 (B7)','Coefficient 8 (B8)','Coefficient 9 (B9)',
                                                                                      'Coefficient 10 (B10)','Coefficient 11 (B11)','Coefficient 12 (B12)','Coefficient 13 (B13)','Coefficient 14 (B14)',
                                                                                      'Coefficient 15 (B15)'])
dfPC_IntCoef.to_csv('/Users/ncp532/Documents/Data/MERRA2/IntCoef.csv')

# Export analysis of variance results
dfAnova_PC_Surf  = pd.DataFrame(anova_results_PC_Surf)
dfAnova_PC_LTcol = pd.DataFrame(anova_results_PC_LTcol)
dfAnova_PC_Surf.to_csv('/Users/ncp532/Documents/Data/MERRA2/Anova_PC_Surf.csv')
dfAnova_PC_LTcol.to_csv('/Users/ncp532/Documents/Data/MERRA2/Anova_PC_LTcol.csv')

#------------------------------------------------------------------------------
# BUILD A DATAFRAME FOR BrO OBSERVATIONS & PREDICTIONS

df1 = pd.concat([dfPCA['SurfBrO'], Pred_BrOSurf_Swanson, Pred_BrOSurf_OLS, Pred_BrOSurf_MLR, dfPCA['LTBrO'], Pred_BrOLTcol_Swanson, Pred_BrOLTcol_OLS, Pred_BrOLTcol_MLR], axis=1, join='inner')
df1.columns = ['Obs_BrOSurf', 'Swanson_BrOSurf', 'OLS_BrOSurf', 'MLR_BrOSurf', 'Obs_BrOLTcol', 'Swanson_BrOLTcol', 'OLS_BrOLTcol', 'MLR_BrOLTcol']
df1['PLS_BrOSurf']  = Pred_BrOSurf_PLS
df1['PLS_BrOLTcol'] = Pred_BrOLTcol_PLS

#------------------------------------------------------------------------------
# EXPORT THE DATAFRAMES AS .CSV

df1.to_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_Test2.csv')
#df1.to_csv('/Users/ncp532/Documents/Data/MERRA2/BrO_Pred_CAMMPCAN_GridBox_BestChoice.csv')
