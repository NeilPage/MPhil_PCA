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

#--------------
# Environmental variables
#--------------
dfPCA = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/PCA_Variables3.csv', index_col=0)
dfPCA = dfPCA.dropna()

#--------------
# Hg0
#--------------
Hg0_V1_17 = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2017_18/CAMMPCAN_V1_Hg0_QAQC_17-18.csv', index_col=0)
Hg0_V2_17 = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2017_18/CAMMPCAN_V2_Hg0_QAQC_17-18.csv', index_col=0)
Hg0_V3_17 = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2017_18/CAMMPCAN_V3_Hg0_QAQC_17-18.csv', index_col=0)

Hg0_V1_18 = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2018_19/CAMMPCAN_V1_Hg0_QAQC_18-19.csv', index_col=0)
Hg0_V2_18 = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2018_19/CAMMPCAN_V2_Hg0_QAQC_18-19.csv', index_col=0)
Hg0_V3_18 = pd.read_csv('/Users/ncp532/Documents/Data/CAMMPCAN_2018_19/CAMMPCAN_V3_Hg0_QAQC_18-19.csv', index_col=0)

#------------------------------------------------------------------------------
# SET THE DATE

#--------------
# Environmnetal variables
#--------------
dfPCA.index = pd.to_datetime(dfPCA.index)
dfPCA.sort_index()

#--------------
# Hg0
#--------------
Hg0_V1_17.index = pd.to_datetime(Hg0_V1_17.index, dayfirst=True)
Hg0_V2_17.index = pd.to_datetime(Hg0_V2_17.index, dayfirst=True)
Hg0_V3_17.index = pd.to_datetime(Hg0_V3_17.index, dayfirst=True)

Hg0_V1_18.index = pd.to_datetime(Hg0_V1_18.index, dayfirst=True)
Hg0_V2_18.index = pd.to_datetime(Hg0_V2_18.index, dayfirst=True)
Hg0_V3_18.index = pd.to_datetime(Hg0_V3_18.index, dayfirst=True)

#------------------------------------------------------------------------------
# PASSIVATION ISSUE WITH CELL A ON VOYAGES V3_18M (FILTER DATA)

Filter1   = Hg0_V3_18['Cart'] == "B"
Hg0_V3_18 = Hg0_V3_18[Filter1]

#------------------------------------------------------------------------------
# RESAMPLE THE Hg0 DATASETS TO 1-HOUR TIME RESOLUTION

Hg0_V1_17 = Hg0_V1_17.resample('60T').mean()
Hg0_V2_17 = Hg0_V2_17.resample('60T').mean()
Hg0_V3_17 = Hg0_V3_17.resample('60T').mean()

Hg0_V1_18 = Hg0_V1_18.resample('60T').mean()
Hg0_V2_18 = Hg0_V2_18.resample('60T').mean()
Hg0_V3_18 = Hg0_V3_18.resample('60T').mean()

#------------------------------------------------------------------------------
# COMBINE THE Hg0 DATASETS FOR EACH VOYAGE INTO A SINGLE DATAFRAME

Hg0_All = pd.concat([Hg0_V1_17,Hg0_V2_17,Hg0_V3_17,Hg0_V1_18,Hg0_V2_18,Hg0_V3_18],axis=0) # All
Hg0_All = Hg0_All['ng/m3']
Hg0_All = Hg0_All.dropna()

#------------------------------------------------------------------------------
# FILTER THE DATAFRAMES TO ONLY INCLUDE THE SAME DATES

# PCA & Hg0
dfPCA_Hg0 = pd.concat([dfPCA, Hg0_All], axis=1, join='inner')

#------------------------------------------------------------------------------
# PERFORM A LOG TRANSFORMATION ON AEC & SQUARE-ROOT TRANSFORMATION ON Hg0

Hg0      = dfPCA_Hg0['ng/m3']
log_Hg0  = np.log(dfPCA_Hg0['ng/m3'])
sqrt_Hg0 = np.sqrt(dfPCA_Hg0['ng/m3'])

#------------------------------------------------------------------------------
# CHECK IF DISTRIBUTION IS GAUSSIAN

# Shapiro-Wilk test for normality
stat_Hg0,      pval_Hg0      = stats.shapiro(Hg0)
stat_log_Hg0,  pval_log_Hg0  = stats.shapiro(log_Hg0)
stat_sqrt_Hg0, pval_sqrt_Hg0 = stats.shapiro(sqrt_Hg0)

#------------------------------------------------------------------------------
# PLOT HISTOGRAMS OF BrO AND AEC (CHECK IF DISTRIBUTION IS GAUSSIAN)

# Setup the Figure
fig = plt.figure(figsize=(10,15))
plt.subplots_adjust(hspace=0.4)

#------------------
# Subplot 1
ax=plt.subplot(311) # options graph 1 (vertical no, horizontal no, graph no)

# Plot the variables
plt.hist(Hg0, density=False, bins=10, edgecolor = 'black', linewidth = 0.5)

# Label the axis
plt.ylabel('Frequency')
plt.xlabel('Hg$^0$ (ng/m$^3$)');

# Figure title
plt.title('Before transformation', fontsize=25, y=1.2)

#------------------
# Subplot 2
ax=plt.subplot(312) # options graph 1 (vertical no, horizontal no, graph no)

# Plot the variables
plt.hist(log_Hg0, density=False, bins=10, color = 'red', edgecolor = 'black', linewidth = 0.5)

# Label the axis
plt.ylabel('Frequency')
plt.xlabel('log(Hg$^0$) (ng/m$^3$)');

# Figure title
plt.title('After transformation', fontsize=25, y=1.2)

#------------------
# Subplot 3
ax=plt.subplot(313) # options graph 1 (vertical no, horizontal no, graph no)

# Plot the variables
plt.hist(sqrt_Hg0, density=False, bins=10, color = 'green', edgecolor = 'black', linewidth = 0.5)

# Label the axis
plt.ylabel('Frequency')
plt.xlabel('sqrt(Hg$^0$) (ng/m$^3$)')

#------------------------------------------------------------------------------
# PERFORM A LOG TRANSFORMATION ON AEC & SQUARE-ROOT TRANSFORMATION ON BrO

dfPCA_Hg0 ['log_AEC']      = np.log(dfPCA_Hg0 ['AEC'])
dfPCA_Hg0 ['sqrt_SurfBrO'] = np.sqrt(dfPCA_Hg0 ['SurfBrO'])
dfPCA_Hg0 ['sqrt_LTBrO']   = np.sqrt(dfPCA_Hg0 ['LTBrO'])
dfPCA_Hg0 ['sqrt_Hg0']     = np.sqrt(dfPCA_Hg0 ['ng/m3'])

#------------------------------------------------------------------------------
# CALCULATE THE STATISTICS

# Mean
dfPCA_Mean = dfPCA_Hg0 .mean()

# Median
dfPCA_Median = dfPCA_Hg0 .median()

# Min
dfPCA_Min = dfPCA_Hg0 .min()

# Max
dfPCA_Max = dfPCA_Hg0 .max()

# Std
dfPCA_Std = dfPCA_Hg0 .std()

# Mean - Std
dfPCA_MeanMStd = dfPCA_Mean - dfPCA_Std

# Mean + Std
dfPCA_MeanPStd = dfPCA_Mean + dfPCA_Std

#----------------------
# Standardised (Manual Method)
dfPCA_Standard = (dfPCA_Hg0 - dfPCA_Mean) / dfPCA_Std

#----------------------
# Standardised (preprocessing.scale() function)
# NOTE: THIS METHOD GENERATES A USERWARNING 
dfPCA_Standard2 = preprocessing.scale(dfPCA_Hg0)
dfPCA_Standard2 = pd.DataFrame(dfPCA_Standard2, index = dfPCA_Hg0.index, columns = dfPCA_Hg0 .columns)

#----------------------
# Standardised (StandardScaler() function)
scale2 = StandardScaler()
dfPCA_Standard3 = scale2.fit_transform(dfPCA_Hg0 )
dfPCA_Standard3 = pd.DataFrame(dfPCA_Standard3, index = dfPCA_Hg0 .index, columns = dfPCA_Hg0 .columns)

#------------------------------------------------------------------------------
# SELECT STANDARD VARIABLES FOR THE PCA

# All Variables
All_Variables = dfPCA_Standard3.drop(['SurfBrO','LTBrO','AEC','IceContactPerc','LandContactMLH','Weighted_Ice','Weighted_Land','Weighted_Ocean','ng/m3',
                                      'Percentage_Ice','Percentage_Land','Percentage_Ocean','InfraRed','Fluro','nd100m','SurfBrO2','SurfBrO3','P1hr',
                                      'WindDir','IceContact100m','IceContactMLH','LandContact','OceanContact','Chlorophyll','Water_Sal','sqrt_SurfBrO','sqrt_Hg0'], 1)

# Hg0 Varaiables
Hg0_Variables = dfPCA_Standard3.drop(['SurfBrO','LTBrO','AEC','IceContactPerc','LandContactMLH','Weighted_Ice','Weighted_Land','Weighted_Ocean','ng/m3',
                                      'Percentage_Ice','Percentage_Land','Percentage_Ocean','InfraRed','Fluro','nd100m','SurfBrO2','SurfBrO3','P1hr',
                                      'WindDir','IceContact100m','IceContactMLH','LandContact','OceanContact','Water_Sal','sqrt_SurfBrO','sqrt_Hg0','sqrt_LTBrO'], 1)

#------------------------------------------------------------------------------
# PERFORM A PRINCIPAL COMPONENT ANALYSIS (PCA)

#--------------------
# Hg0
#--------------------
# Apply the PCA (Swanson et al has 9 PCs, but only 3 have variances greater 1)
PCA_Hg0 = PCA() # All n components

# Retrieve the principal components (PCs)
PrincipalComponents_Variables  = PCA_Hg0.fit_transform(Hg0_Variables) # Additional Variables

# Put the principle components into a DataFrame
Principal_Variables_Hg0 = pd.DataFrame(data = PrincipalComponents_Variables, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9',
                                                                                       'PC10','PC11','PC12','PC13','PC14'])#,'PC15','PC16','PC17',
                                                                                       #'PC18','PC19','PC20','PC21','PC22','PC23','PC24','PC25','PC26']) # Variables

                                                                                      
# Explained variation per principal component
print('Explained variation per principal component: {}'.format(PCA_Hg0.explained_variance_ratio_))
Explained_VarianceHg0 = PCA_Hg0.explained_variance_ratio_

# Get the loadings
loadingsHg0  = pd.DataFrame(PCA_Hg0.components_.T,  columns=Principal_Variables_Hg0.columns,index=Hg0_Variables.columns)
loadingsHg0.to_csv('/Users/ncp532/Documents/Data/PCA_Hg0/PC_Loadings_Hg0.csv')

# Calculate the normalised variance for each PC
NormVarHg0 = np.mean(np.square(Principal_Variables_Hg0 - Principal_Variables_Hg0.mean()))

# Calculate the sample error
SampleErrHg0 = stats.sem(Principal_Variables_Hg0)

#--------------------
# How many variables are needed to explain variance in the data?
#--------------------
plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
x1 = np.arange(1, 15, step=1)
y1 = np.cumsum(PCA_Hg0.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(x1, y1, marker='o', linestyle='--', color='b', label="Hg$^0$ (cumulative variance)")
plt.plot(x1, PCA_Hg0.explained_variance_ratio_, marker='o', linestyle='--', color='black', label="Hg$^0$ (variance)")

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 15, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(11, 0.90, '95% cut-off threshold', color = 'red', fontsize=16)

plt.axhline(y=0.6, color='r', linestyle='-')
plt.text(11, 0.55, '60% threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.legend(loc=2)
plt.show()

#------------------------------------------------------------------------------
# CALCULATE THE STANDARD VARIABLE LOADINGS

# Transorm the loadings dataframe
loadingsHg0T = loadingsHg0.T

SVL_PC1_Hg0  = Hg0_Variables.mul(loadingsHg0T.iloc[0],  axis='columns') # PC1
SVL_PC2_Hg0  = Hg0_Variables.mul(loadingsHg0T.iloc[1],  axis='columns') # PC2
SVL_PC3_Hg0  = Hg0_Variables.mul(loadingsHg0T.iloc[2],  axis='columns') # PC3
SVL_PC4_Hg0  = Hg0_Variables.mul(loadingsHg0T.iloc[3],  axis='columns') # PC4
SVL_PC5_Hg0  = Hg0_Variables.mul(loadingsHg0T.iloc[4],  axis='columns') # PC5
SVL_PC6_Hg0  = Hg0_Variables.mul(loadingsHg0T.iloc[5],  axis='columns') # PC6
SVL_PC7_Hg0  = Hg0_Variables.mul(loadingsHg0T.iloc[6],  axis='columns') # PC7
SVL_PC8_Hg0  = Hg0_Variables.mul(loadingsHg0T.iloc[7],  axis='columns') # PC8
SVL_PC9_Hg0  = Hg0_Variables.mul(loadingsHg0T.iloc[8],  axis='columns') # PC9
SVL_PC10_Hg0 = Hg0_Variables.mul(loadingsHg0T.iloc[9],  axis='columns') # PC10
SVL_PC11_Hg0 = Hg0_Variables.mul(loadingsHg0T.iloc[10], axis='columns') # PC11
SVL_PC12_Hg0 = Hg0_Variables.mul(loadingsHg0T.iloc[11], axis='columns') # PC12
SVL_PC13_Hg0 = Hg0_Variables.mul(loadingsHg0T.iloc[12], axis='columns') # PC13
SVL_PC14_Hg0 = Hg0_Variables.mul(loadingsHg0T.iloc[13], axis='columns') # PC14
# SVL_PC15_Hg0 = Hg0_Variables.mul(loadingsHg0T.iloc[14], axis='columns') # PC15
# SVL_PC16_Hg0 = Hg0_Variables.mul(loadingsHg0T.iloc[15], axis='columns') # PC16
# SVL_PC17_Hg0 = Hg0_Variables.mul(loadingsHg0T.iloc[16], axis='columns') # PC17
# SVL_PC18_Hg0 = Hg0_Variables.mul(loadingsHg0T.iloc[17], axis='columns') # PC18
# SVL_PC19_Hg0 = Hg0_Variables.mul(loadingsHg0T.iloc[18], axis='columns') # PC19
# SVL_PC20_Hg0 = Hg0_Variables.mul(loadingsHg0T.iloc[19], axis='columns') # PC20
# SVL_PC21_Hg0 = Hg0_Variables.mul(loadingsHg0T.iloc[20], axis='columns') # PC21
# SVL_PC22_Hg0 = Hg0_Variables.mul(loadingsHg0T.iloc[21], axis='columns') # PC22
# SVL_PC23_Hg0 = Hg0_Variables.mul(loadingsHg0T.iloc[22], axis='columns') # PC23

#------------------------------------------------------------------------------
# CALCULATE THE PRINCIPLE COMPONENTS

PC1  = SVL_PC1_Hg0.sum(axis=1)
PC2  = SVL_PC2_Hg0.sum(axis=1)
PC3  = SVL_PC3_Hg0.sum(axis=1)
PC4  = SVL_PC4_Hg0.sum(axis=1)
PC5  = SVL_PC5_Hg0.sum(axis=1)
PC6  = SVL_PC6_Hg0.sum(axis=1)
PC7  = SVL_PC7_Hg0.sum(axis=1)
PC8  = SVL_PC8_Hg0.sum(axis=1)
PC9  = SVL_PC9_Hg0.sum(axis=1)
PC10 = SVL_PC10_Hg0.sum(axis=1)
PC11 = SVL_PC11_Hg0.sum(axis=1)
PC12 = SVL_PC12_Hg0.sum(axis=1)
PC13 = SVL_PC13_Hg0.sum(axis=1)
PC14 = SVL_PC14_Hg0.sum(axis=1)
# PC15 = SVL_PC15_Hg0.sum(axis=1)
# PC16 = SVL_PC16_Hg0.sum(axis=1)
# PC17 = SVL_PC17_Hg0.sum(axis=1)
# PC18 = SVL_PC18_Hg0.sum(axis=1)
# PC19 = SVL_PC19_Hg0.sum(axis=1)
# PC20 = SVL_PC20_Hg0.sum(axis=1)
# PC21 = SVL_PC21_Hg0.sum(axis=1)
# PC22 = SVL_PC22_Hg0.sum(axis=1)
# PC23 = SVL_PC23_Hg0.sum(axis=1)

df_PC_Hg0 = pd.concat([PC1, PC2, PC3, PC4, PC5, PC6, PC7, PC8, PC9, PC10,
                       PC11,PC12,PC13,PC14], axis=1, join='inner')#,PC15,PC16,PC17,PC18,PC19,PC20,
                       #PC21,PC22,PC23], axis=1, join='inner')
df_PC_Hg0.columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10',
                     'PC11','PC12','PC13','PC14']#,'PC15','PC16','PC17','PC18','PC19','PC20',
                     #'PC21','PC22','PC23']
df_PC_Hg0.to_csv('/Users/ncp532/Documents/Data/PCA_Hg0/PC_Hg0.csv')

#------------------------------------------------------------------------------
# PERFORM A PRINCIPLE COMPONENT REGRESSION (PCR)
# (if z = sqrt(BrO_obs)) “z ~ pc1 + pc2 + pc3

# Hg0 dataframe for the PCA
dataPCA_Hg0  = pd.DataFrame({'PC1': PC1, 'PC2': PC2, 'PC3': PC3, 'PC4': PC4, 'PC5': PC5, 'PC6': PC6, 'PC7': PC7, 'PC8': PC8, 'PC9': PC9, 'PC10': PC10,
                             'PC11':PC11,'PC12':PC12,'PC13':PC13,'PC14':PC14, 'z': dfPCA_Hg0['sqrt_Hg0']})#,'PC15':PC15,'PC16':PC16,'PC17':PC17,'PC18':PC18,'PC19':PC19,'PC20': PC20,
                             #'PC21':PC21,'PC22':PC22,'PC23':PC23, 'z': dfPCA_Hg0['ng/m3']})

# Fit the model
#model_PC_Hg0  = ols("z ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14 + PC15 + PC16 + PC17 + PC18 + PC19 + PC20 + PC21 + PC22 + PC23", dataPCA_Hg0).fit()  # Surface StatsModel (ols)
model_PC_Hg0  = ols("z ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10 + PC11 + PC12 + PC13 + PC14", dataPCA_Hg0).fit()  # Surface StatsModel (ols)
 
# Retrieve the model results
Model_results_PC_Hg0  = model_PC_Hg0._results.params

# Peform analysis of variance on fitted linear model
anova_results_PC_Hg0  = anova_lm(model_PC_Hg0)

#------------------------------------------------------------------------------
# PCR RETRIEVE THE INTERCEPT & COEFFICIENTS

OLS_PC_0  = Model_results_PC_Hg0[0]  # Intercept
OLS_PC_1  = Model_results_PC_Hg0[1]  # Coefficient PC1
OLS_PC_2  = Model_results_PC_Hg0[2]  # Coefficient PC2
OLS_PC_3  = Model_results_PC_Hg0[3]  # Coefficient PC3
OLS_PC_4  = Model_results_PC_Hg0[4]  # Coefficient PC4
OLS_PC_5  = Model_results_PC_Hg0[5]  # Coefficient PC5
OLS_PC_6  = Model_results_PC_Hg0[6]  # Coefficient PC6
OLS_PC_7  = Model_results_PC_Hg0[7]  # Coefficient PC7
OLS_PC_8  = Model_results_PC_Hg0[8]  # Coefficient PC8
OLS_PC_9  = Model_results_PC_Hg0[9]  # Coefficient PC9
OLS_PC_10 = Model_results_PC_Hg0[10] # Coefficient PC10
OLS_PC_11 = Model_results_PC_Hg0[11] # Coefficient PC11
OLS_PC_12 = Model_results_PC_Hg0[12] # Coefficient PC12
OLS_PC_13 = Model_results_PC_Hg0[13] # Coefficient PC13
OLS_PC_14 = Model_results_PC_Hg0[14] # Coefficient PC14
# OLS_PC_15 = Model_results_PC_Hg0[15] # Coefficient PC15
# OLS_PC_16 = Model_results_PC_Hg0[16] # Coefficient PC16
# OLS_PC_17 = Model_results_PC_Hg0[17] # Coefficient PC17
# OLS_PC_18 = Model_results_PC_Hg0[18] # Coefficient PC18
# OLS_PC_19 = Model_results_PC_Hg0[19] # Coefficient PC19
# OLS_PC_20 = Model_results_PC_Hg0[20] # Coefficient PC20
# OLS_PC_21 = Model_results_PC_Hg0[21] # Coefficient PC21
# OLS_PC_22 = Model_results_PC_Hg0[22] # Coefficient PC22
# OLS_PC_23 = Model_results_PC_Hg0[23] # Coefficient PC23

#------------------------------------------------------------------------------
# APPLY THE BrO PCR MODEL


Pred_Hg0_OLS  = np.square(OLS_PC_0 + (OLS_PC_1*PC1)   + (OLS_PC_2*PC2)   + (OLS_PC_3*PC3)   + (OLS_PC_4*PC4)   + (OLS_PC_5*PC5)   + (OLS_PC_6*PC6)   + (OLS_PC_7*PC7)   + (OLS_PC_8*PC8)   + (OLS_PC_9*PC9)   + (OLS_PC_10*PC10)
                                   + (OLS_PC_11*PC11) + (OLS_PC_12*PC12) + (OLS_PC_13*PC13) + (OLS_PC_14*PC14))# + (OLS_PC_15*PC15) + (OLS_PC_16*PC16) + (OLS_PC_17*PC17) + (OLS_PC_18*PC18) + (OLS_PC_19*PC19) + (OLS_PC_20*PC20)
                                   #+ (OLS_PC_21*PC21) + (OLS_PC_22*PC22) + (OLS_PC_23*PC23))

#------------------------------------------------------------------------------
# PERFORM A PARTIAL LEAST SQUARES (PLS) REGRESSION
# (if z = Hg0) “z ~ pc1 + pc2 + pc3

# Apply the PLS
PLS_Hg0  = PLSRegression(n_components=3)

# Fit the model
PLS_Hg0.fit(Hg0_Variables,dfPCA_Hg0['sqrt_Hg0'])

# Retrieve the model results
Pred_Hg0_PLS = PLS_Hg0.predict(Hg0_Variables)  # Surface StatsModel (ols)

# First we need to flatten the data: it's 2D layout is not relevent.
Pred_Hg0_PLS = Pred_Hg0_PLS.flatten()
Pred_Hg0_PLS = pd.DataFrame(Pred_Hg0_PLS)
Pred_Hg0_PLS = np.square(Pred_Hg0_PLS)

#------------------------------------------------------------------------------
# INDIVIDUAL VARIABLE REGRESSION

data      = Hg0_Variables
data['z'] = dfPCA_Hg0['ng/m3']

# Environmental variables
slope_O3_S,             intercept_O3_S,             r_O3,             p_O3,             std_err_O3_S             = stats.linregress(data['z'], data['O3'])
slope_AEC_S,            intercept_AEC_S,            r_AEC,            p_AEC,            std_err_AEC_S            = stats.linregress(data['z'], data['log_AEC'])
slope_SurfTemp_S,       intercept_SurfTemp_S,       r_SurfTemp,       p_SurfTemp,       std_err_SurfTemp_S       = stats.linregress(data['z'], data['SurfTemp'])
slope_SLP_S,            intercept_SLP_S,            r_SLP,            p_SLP,            std_err_SLP_S            = stats.linregress(data['z'], data['SurfPres'])
slope_WS10m_S,          intercept_WS10m_S,          r_WS10m,          p_WS10m,          std_err_WS10m_S          = stats.linregress(data['z'], data['WS10m'])
slope_MLH_S,            intercept_MLH_S,            r_MLH,            p_MLH,            std_err_MLH_S            = stats.linregress(data['z'], data['MLH'])
#slope_P1hr_S,           intercept_P1hr_S,           r_P1hr,           p_P1hr,           std_err_P1hr_S           = stats.linregress(data['z'], data['P1hr'])
slope_PTD1000m_S,       intercept_PTD1000m_S,       r_PTD1000m,       p_PTD1000m,       std_err_PTD1000m_S       = stats.linregress(data['z'], data['PTDif1000m'])
slope_PTD100m_S,        intercept_PTD100m_S,        r_PTD100m,        p_PTD100m,        std_err_PTD100m_S        = stats.linregress(data['z'], data['PTDif100m'])
#slope_WindDir_S,        intercept_WindDir_S,        r_WindDir,        p_WindDir,        std_err_WindDir_S        = stats.linregress(data['z'], data['WindDir'])
slope_SolRad_S,         intercept_SolRad_S,         r_SolRad,         p_SolRad,         std_err_SolRad_S         = stats.linregress(data['z'], data['SolRad'])
slope_IceContact_S,     intercept_IceContact_S,     r_IceContact,     p_IceContact,     std_err_IceContact_S     = stats.linregress(data['z'], data['IceContact'])
#slope_IceContact100m_S, intercept_IceContact100m_S, r_IceContact100m, p_IceContact100m, std_err_IceContact100m_S = stats.linregress(data['z'], data['IceContact100m'])
#slope_IceContactMLH_S,  intercept_IceContactMLH_S,  r_IceContactMLH,  p_IceContactMLH,  std_err_IceContactMLH_S  = stats.linregress(data['z'], data['IceContactMLH'])
slope_SeaIceConc_S,     intercept_SeaIceConc_S,     r_SeaIceConc,     p_SeaIceConc,     std_err_SeaIceConc_S     = stats.linregress(data['z'], data['SeaIceConc'])
#slope_LandContact_S,    intercept_LandContact_S,    r_LandContact,    p_LandContact,    std_err_LandContact_S    = stats.linregress(data['z'], data['LandContact'])
#slope_OceanContact_S,   intercept_OceanContact_S,   r_OceanContact,   p_OceanContact,   std_err_OceanContact_S   = stats.linregress(data['z'], data['OceanContact'])
#slope_Chloro_S,         intercept_Chloro_S,         r_Chloro,         p_Chloro,         std_err_Chloro_S         = stats.linregress(data['z'], data['Chlorophyll'])
slope_WaterTemp_S,      intercept_WaterTemp_S,      r_WaterTemp,      p_WaterTemp,      std_err_WaterTemp_S      = stats.linregress(data['z'], data['Water_Temp'])
#slope_WaterSal_S,       intercept_WaterSal_S,       r_WaterSal,       p_WaterSal,       std_err_WaterSal_S       = stats.linregress(data['z'], data['Water_Sal'])
slope_RelHum_S,         intercept_RelHum_S,         r_RelHum,         p_RelHum,         std_err_RelHum_S         = stats.linregress(data['z'], data['RelHum'])
#slope_BrOSurf_S,        intercept_BrOSurf_S,        r_BrOSurf,        p_BrOSurf,        std_err_BrOSurf_S        = stats.linregress(data['z'], data['sqrt_SurfBrO'])
slope_BrOLTcol_S,       intercept_BrOLTcol_S,       r_BrOLTcol,       p_BrOLTcol,       std_err_BrOLTcol_S       = stats.linregress(data['z'], data['sqrt_LTBrO'])

# R-squared
r2_O3             = r_O3**2
r2_AEC            = r_AEC**2
r2_SurfTemp       = r_SurfTemp**2
r2_SLP            = r_SLP**2
r2_WS10m          = r_WS10m**2
r2_MLH            = r_MLH**2
#r2_P1hr           = r_P1hr**2
r2_PTD1000m       = r_PTD1000m**2
r2_PTD100m        = r_PTD100m**2
#r2_WindDir        = r_WindDir**2
r2_SolRad         = r_SolRad**2
r2_IceContact     = r_IceContact**2
#r2_IceContact100m = r_IceContact100m**2
#r2_IceContactMLH  = r_IceContactMLH**2
r2_SeaIceConc     = r_SeaIceConc**2
#r2_LandContact    = r_LandContact**2
#r2_OceanContact   = r_OceanContact**2
#r2_Chloro         = r_Chloro**2
r2_WaterTemp      = r_WaterTemp**2
#r2_WaterSal       = r_WaterSal**2
r2_RelHum         = r_RelHum**2
#r2_BrOSurf        = r_BrOSurf**2
r2_BrOLTcol       = r_BrOLTcol**2

# dfLinearRegressionHg0 = pd.DataFrame({'R (Hg0)':       [r_O3,  r_AEC,  r_SurfTemp,  r_SLP,  r_WS10m,  r_MLH,  r_P1hr,  r_PTD1000m,  r_PTD100m,  r_WindDir,  r_SolRad,  r_IceContact,  r_IceContact100m,  r_IceContactMLH,  r_SeaIceConc,  r_LandContact,  r_OceanContact,  r_Chloro,  r_WaterTemp,  r_WaterSal,  r_RelHum,  r_BrOSurf,  r_BrOLTcol],
#                                       'R^2 (Hg0)':     [r2_O3, r2_AEC, r2_SurfTemp, r2_SLP, r2_WS10m, r2_MLH, r2_P1hr, r2_PTD1000m, r2_PTD100m, r2_WindDir, r2_SolRad, r2_IceContact, r2_IceContact100m, r2_IceContactMLH, r2_SeaIceConc, r2_LandContact, r2_OceanContact, r2_Chloro, r2_WaterTemp, r2_WaterSal, r2_RelHum, r2_BrOSurf, r2_BrOLTcol],
#                                       'p-value (Hg0)': [p_O3,  p_AEC,  p_SurfTemp,  p_SLP,  p_WS10m,  p_MLH,  p_P1hr,  p_PTD1000m,  p_PTD100m,  p_WindDir,  p_SolRad,  p_IceContact,  p_IceContact100m,  p_IceContactMLH,  p_SeaIceConc,  p_LandContact,  p_OceanContact,  p_Chloro,  p_WaterTemp,  p_WaterSal,  p_RelHum,  p_BrOSurf,  p_BrOLTcol]})
dfLinearRegressionHg0 = pd.DataFrame({'R (Hg0)':       [r_O3,  r_AEC,  r_SurfTemp,  r_SLP,  r_WS10m,  r_MLH,  r_PTD1000m,  r_PTD100m,  r_SolRad,  r_IceContact,  r_SeaIceConc,  r_WaterTemp,  r_RelHum,  r_BrOLTcol],
                                      'R^2 (Hg0)':     [r2_O3, r2_AEC, r2_SurfTemp, r2_SLP, r2_WS10m, r2_MLH, r2_PTD1000m, r2_PTD100m, r2_SolRad, r2_IceContact, r2_SeaIceConc, r2_WaterTemp, r2_RelHum, r2_BrOLTcol],
                                      'p-value (Hg0)': [p_O3,  p_AEC,  p_SurfTemp,  p_SLP,  p_WS10m,  p_MLH,  p_PTD1000m,  p_PTD100m,  p_SolRad,  p_IceContact,  p_SeaIceConc,  p_WaterTemp,  p_RelHum,  p_BrOLTcol]})

dfLinearRegressionHg0.index = ['O3','AEC','SurfTemp','SurfPres','WS10m','MLH','PTD1000m','PTD100m','SolRad','IceContact','SeaIceConc','WaterTemp','RelHum','BrOLTcol']
dfLinearRegressionHg0.to_csv('/Users/ncp532/Documents/Data/PCA_Hg0/LinearRegressionHg0.csv')

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
# Vif_O3              = vif_cal(input_data=All_Variables, dependent_col="O3")
# Vif_SurfPres        = vif_cal(input_data=All_Variables, dependent_col="SurfPres")
# Vif_SurfTemp        = vif_cal(input_data=All_Variables, dependent_col="SurfTemp")
# Vif_P1hr            = vif_cal(input_data=All_Variables, dependent_col="P1hr")
# Vif_PTDif100m       = vif_cal(input_data=All_Variables, dependent_col="PTDif100m")
# Vif_PTDif1000m      = vif_cal(input_data=All_Variables, dependent_col="PTDif1000m")
# Vif_WS10m           = vif_cal(input_data=All_Variables, dependent_col="WS10m")
# Vif_WindDir         = vif_cal(input_data=All_Variables, dependent_col="WindDir")
# Vif_MLH             = vif_cal(input_data=All_Variables, dependent_col="MLH")
# Vif_SolRad          = vif_cal(input_data=All_Variables, dependent_col="SolRad")
# Vif_IceContact      = vif_cal(input_data=All_Variables, dependent_col="IceContact")
# Vif_IceContact100m  = vif_cal(input_data=All_Variables, dependent_col="IceContact100m")
# Vif_IceContactMLH   = vif_cal(input_data=All_Variables, dependent_col="IceContactMLH")
# Vif_SeaIceConc      = vif_cal(input_data=All_Variables, dependent_col="SeaIceConc")
# Vif_LandContact     = vif_cal(input_data=All_Variables, dependent_col="LandContact")
# Vif_OceanContact    = vif_cal(input_data=All_Variables, dependent_col="OceanContact")
# Vif_Chlorophyll     = vif_cal(input_data=All_Variables, dependent_col="Chlorophyll")
# Vif_WaterTemp       = vif_cal(input_data=All_Variables, dependent_col="Water_Temp")
# Vif_WaterSal        = vif_cal(input_data=All_Variables, dependent_col="Water_Sal")
# Vif_RelHum          = vif_cal(input_data=All_Variables, dependent_col="RelHum")
# Vif_AEC             = vif_cal(input_data=All_Variables, dependent_col="log_AEC")

#------------------------------------------------------------------------------
#BUILD DATAFRAME FOR THE PCR MODEL INTERCEPTS & COEFFICIENTS

# Build a pandas dataframe
dfPC_IntCoef = {'Intercept (B0)':       [OLS_PC_0],
                'Coefficient 1 (B1)':   [OLS_PC_1],
                'Coefficient 2 (B2)':   [OLS_PC_2],
                'Coefficient 3 (B3)':   [OLS_PC_3],
                'Coefficient 4 (B4)':   [OLS_PC_4],
                'Coefficient 5 (B5)':   [OLS_PC_5],
                'Coefficient 6 (B6)':   [OLS_PC_6],
                'Coefficient 7 (B7)':   [OLS_PC_7],
                'Coefficient 8 (B8)':   [OLS_PC_8],
                'Coefficient 9 (B9)':   [OLS_PC_9],
                'Coefficient 10 (B10)': [OLS_PC_10],
                'Coefficient 11 (B11)': [OLS_PC_11],
                'Coefficient 12 (B12)': [OLS_PC_12],
                'Coefficient 13 (B13)': [OLS_PC_13],
                'Coefficient 14 (B14)': [OLS_PC_14]}
                # 'Coefficient 15 (B15)': [OLS_PC_15],
                # 'Coefficient 16 (B16)': [OLS_PC_16],
                # 'Coefficient 17 (B17)': [OLS_PC_17],
                # 'Coefficient 18 (B18)': [OLS_PC_18],
                # 'Coefficient 19 (B19)': [OLS_PC_19],
                # 'Coefficient 20 (B20)': [OLS_PC_20],
                # 'Coefficient 21 (B21)': [OLS_PC_21],
                # 'Coefficient 22 (B22)': [OLS_PC_22],
                # 'Coefficient 23 (B23)': [OLS_PC_23]}
dfPC_IntCoef = pd.DataFrame(dfPC_IntCoef, index = ['ols_Hg0'],columns = ['Intercept (B0)','Coefficient 1 (B1)','Coefficient 2 (B2)','Coefficient 3 (B3)','Coefficient 4 (B4)','Coefficient 5 (B5)','Coefficient 6 (B6)',
                                                                         'Coefficient 7 (B7)','Coefficient 8 (B8)','Coefficient 9 (B9)','Coefficient 10 (B10)','Coefficient 11 (B11)','Coefficient 12 (B12)',
                                                                         'Coefficient 13 (B13)','Coefficient 14 (B14)'])#,'Coefficient 15 (B15)','Coefficient 16 (B16)','Coefficient 17 (B17)','Coefficient 18 (B18)',
                                                                         #'Coefficient 19 (B19)','Coefficient 20 (B20)','Coefficient 21 (B21)','Coefficient 22 (B22)','Coefficient 23 (B23)'])
dfPC_IntCoef.to_csv('/Users/ncp532/Documents/Data/PCA_Hg0/IntCoef.csv')

# Export analysis of variance results
dfAnova_PC_Hg0  = pd.DataFrame(anova_results_PC_Hg0)
dfAnova_PC_Hg0.to_csv('/Users/ncp532/Documents/Data/PCA_Hg0/Anova_PC_Hg0.csv')

#------------------------------------------------------------------------------
# BUILD A DATAFRAME FOR BrO OBSERVATIONS & PREDICTIONS

df1 = pd.concat([dfPCA_Hg0['ng/m3'], Pred_Hg0_OLS], axis=1, join='inner')
df1.columns = ['Obs_Hg0', 'OLS_Hg0']

#------------------------------------------------------------------------------
# EXPORT THE DATAFRAMES AS .CSV

df1.to_csv('/Users/ncp532/Documents/Data/PCA_Hg0/Hg0_Pred_CAMMPCAN_Test.csv')

