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
from matplotlib.lines import Line2D

#------------------------------------------------------------------------------
# DEFINE THE DATASET

# Hg0 principal components
PrinCon_Hg0 = pd.read_csv('/Users/ncp532/Documents/Data/PCA_Hg0/PC_Hg0.csv', index_col=0) # BrO
#PrinCon_Hg0 = pd.read_csv('/Users/ncp532/Documents/Data/PCA_Hg0/Without_BrO/PC_Hg0_noBrO.csv', index_col=0) # BrO

# Hg0 predictions & observations
PC_Hg0 = pd.read_csv('/Users/ncp532/Documents/Data/PCA_Hg0/Hg0_Pred_CAMMPCAN_Test.csv', index_col=0) # BrO
#PC_Hg0 = pd.read_csv('/Users/ncp532/Documents/Data/PCA_Hg0/Without_BrO/Hg0_Pred_CAMMPCAN_noBrO.csv', index_col=0) # Without BrO

#------------------------------------------------------------------------------
# CALCULATE THE CORRELATION COEFFICIENTS (R) AND (p)

#--------------------------------
# Linear correlation coefficient (R)
#--------------------------------

# PC1 & Obs
slope_PC1, intercept_PC1, rval_PC1, pval_PC1, stderr_PC1 = stats.linregress(PrinCon_Hg0['PC1'], PC_Hg0['Obs_Hg0'])

# PC2 & Obs
slope_PC2, intercept_PC2, rval_PC2, pval_PC2, stderr_PC2 = stats.linregress(PrinCon_Hg0['PC2'], PC_Hg0['Obs_Hg0'])

# PC3 & Obs
slope_PC3, intercept_PC3, rval_PC3, pval_PC3, stderr_PC3 = stats.linregress(PrinCon_Hg0['PC3'], PC_Hg0['Obs_Hg0'])

#--------------------------------
# Nonparametric Spearman rank correlation coefficient (p)
#--------------------------------

# PC1 & Obs
rho_Spear_PC1, pval_Spear_PC1 = stats.spearmanr(PrinCon_Hg0['PC1'], PC_Hg0['Obs_Hg0'])

# PC2 & Obs
rho_Spear_PC2, pval_Spear_PC2 = stats.spearmanr(PrinCon_Hg0['PC2'], PC_Hg0['Obs_Hg0'])

# PC3 & Obs
rho_Spear_PC3, pval_Spear_PC3 = stats.spearmanr(PrinCon_Hg0['PC3'], PC_Hg0['Obs_Hg0'])

#--------------------------------
# Linear determination coefficient (R^2)
#--------------------------------

# PC1 & Obs
R2_PC1 = np.square(rval_PC1)

# PC2 & Obs
R2_PC2 = np.square(rval_PC2)

# PC3 & Obs
R2_PC3 = np.square(rval_PC3)

#------------------------------------------------------------------------------
# SET THE DATE

PrinCon_Hg0.index = (pd.to_datetime(PrinCon_Hg0.index, dayfirst=True))
PC_Hg0.index      = (pd.to_datetime(PC_Hg0.index,      dayfirst=True))

#------------------------------------------------------------------------------
# PLOT THE GRAPH (PC_BrOLTcol)

fig1 = plt.figure()

#-------------------------------------
ax = plt.subplot(3,2,1)

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot PC1
ax.plot(PrinCon_Hg0.index, PrinCon_Hg0['PC1'], marker='o', c='r', markersize = 2.0, linestyle='none', label='PC1')

# Plot vertical lines to seperate the voyages
V1V2_17 = ax.axvline(datetime(2017,12,7),  linewidth=0.5, color='grey')
V2V3_17 = ax.axvline(datetime(2018,1,18),  linewidth=0.5, color='grey')

# Text boxes for the voyages
props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
a = datetime(2017,11,10)
b = datetime(2017,12,7)
c = datetime(2018,1,18)
d = datetime(2018,2,28)
ax.text(a+(b-a)/2, 4.7, "V1",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(b+(c-b)/2, 4.7, "V2",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(c+(d-c)/2, 4.7, "V3",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)

# Format the x-axis
plt.xlim(datetime(2017,11,10),datetime(2018,2,28))
ax.axes.get_xaxis().set_visible(False)

# Format y-axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5.5,5.5)
ax.tick_params(axis='y', which='both', colors='r', labelsize=10)
ax.spines["right"].set_color('r')
ax.spines["left"].set_color('r')
# Plot the axis labels
#ax.set_xlabel('Date', fontsize=16, labelpad= 10)
ax.set_ylabel('PC1 Loading', fontsize=16, labelpad= 5, color='r')

# Plot title
plt.title('CAMMPCAN (2017-18)', fontsize=20, y=1.05)

#-------------------------------------
ax = plt.subplot(3,2,3)

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot PC2
ax.plot(PrinCon_Hg0.index, PrinCon_Hg0['PC2'], marker='o', c='b', markersize = 2.0, linestyle='none', label='PC2')

# Plot vertical lines to seperate the voyages
V1V2_17 = ax.axvline(datetime(2017,12,7,0,0,0),  linewidth=0.5, color='grey')
V2V3_17 = ax.axvline(datetime(2018,1,18,0,0,0),  linewidth=0.5, color='grey')

# Text boxes for the voyages
props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
a = datetime(2017,11,10)
b = datetime(2017,12,7)
c = datetime(2018,1,18)
d = datetime(2018,2,28)
ax.text(a+(b-a)/2, 4.7, "V1",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(b+(c-b)/2, 4.7, "V2",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(c+(d-c)/2, 4.7, "V3",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)

# Format the x-axis
plt.xlim(datetime(2017,11,10),datetime(2018,2,28))
xmajor_formatter = mdates.DateFormatter('%d %b') # format how the date is displayed
ax.xaxis.set_major_formatter(xmajor_formatter)
ax.axes.get_xaxis().set_visible(False)

# Format y-axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5.5,5.5)
ax.tick_params(axis='y', which='both', colors='b', labelsize=10)
ax.spines["right"].set_color('b')
ax.spines["left"].set_color('b')

# Plot the axis labels
ax.set_xlabel('Date', fontsize=16, labelpad= 10)
ax.set_ylabel('PC2 Loading', fontsize=16, labelpad= 5, color='b')

#-------------------------------------
ax = plt.subplot(3,2,5)

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot PC3
ax.plot(PrinCon_Hg0.index, PrinCon_Hg0['PC3'], marker='o', c='g', markersize = 2.0, linestyle='none', label='PC3')

# Plot vertical lines to seperate the voyages
V1V2_17 = ax.axvline(datetime(2017,12,7,0,0,0),  linewidth=0.5, color='grey')
V2V3_17 = ax.axvline(datetime(2018,1,18,0,0,0),  linewidth=0.5, color='grey')

# Text boxes for the voyages
props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
a = datetime(2017,11,10)
b = datetime(2017,12,7)
c = datetime(2018,1,18)
d = datetime(2018,2,28)
ax.text(a+(b-a)/2, 4.7, "V1",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(b+(c-b)/2, 4.7, "V2",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(c+(d-c)/2, 4.7, "V3",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)

# Format the x-axis
plt.xlim(datetime(2017,11,10),datetime(2018,2,28))
xmajor_formatter = mdates.DateFormatter('%d %b') # format how the date is displayed
ax.xaxis.set_major_formatter(xmajor_formatter)

# Format y-axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5.5,5.5)
ax.tick_params(axis='y', which='both', colors='g', labelsize=10)
ax.spines["right"].set_color('g')
ax.spines["left"].set_color('g')

# Plot the axis labels
ax.set_xlabel('Date', fontsize=16, labelpad= 10)
ax.set_ylabel('PC3 Loading', fontsize=16, labelpad= 5, color='g')

#-------------------------------------
ax = plt.subplot(3,2,2)

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot PC1
ax.plot(PrinCon_Hg0.index, PrinCon_Hg0['PC1'], marker='o', c='r', markersize = 2.0, linestyle='none', label='PC1')

# Plot vertical lines to seperate the voyages
V1V2_18 = ax.axvline(datetime(2018,12,7,0,0,0),  linewidth=0.5, color='grey')
V2V3_18 = ax.axvline(datetime(2019,1,18,0,0,0),  linewidth=0.5, color='grey')

# Text boxes for the voyages
props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
a = datetime(2018,11,1)
b = datetime(2018,12,7)
c = datetime(2019,1,18)
d = datetime(2019,2,21)
ax.text(a+(b-a)/2, 4.7, "V1",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(b+(c-b)/2, 4.7, "V2",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(c+(d-c)/2, 4.7, "V3",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)

# Format the x-axis
plt.xlim(datetime(2018,11,1),datetime(2019,2,21))
ax.axes.get_xaxis().set_visible(False)

# Format y-axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5.5,5.5)
ax.axes.get_yaxis().set_visible(False)
ax.tick_params(axis='y', which='both', colors='r', labelsize=10)
ax.spines["right"].set_color('r')
ax.spines["left"].set_color('r')

# Plot title
plt.title('CAMMPCAN (2018-19)', fontsize=20, y=1.05)

#-------------------------------------
ax = plt.subplot(3,2,4)

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot PC2
ax.plot(PrinCon_Hg0.index, PrinCon_Hg0['PC2'], marker='o', c='b', markersize = 2.0, linestyle='none', label='PC2')

# Plot vertical lines to seperate the voyages
V1V2_18 = ax.axvline(datetime(2018,12,7,0,0,0),  linewidth=0.5, color='grey')
V2V3_18 = ax.axvline(datetime(2019,1,18,0,0,0),  linewidth=0.5, color='grey')

# Text boxes for the voyages
props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
a = datetime(2018,11,1)
b = datetime(2018,12,7)
c = datetime(2019,1,18)
d = datetime(2019,2,21)
ax.text(a+(b-a)/2, 4.7, "V1",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(b+(c-b)/2, 4.7, "V2",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(c+(d-c)/2, 4.7, "V3",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)

# Format the x-axis
plt.xlim(datetime(2018,11,1),datetime(2019,2,21))
xmajor_formatter = mdates.DateFormatter('%d %b') # format how the date is displayed
ax.xaxis.set_major_formatter(xmajor_formatter)
ax.axes.get_xaxis().set_visible(False)

# Format y-axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5.5,5.5)
ax.axes.get_yaxis().set_visible(False)
ax.tick_params(axis='y', which='both', colors='b', labelsize=10)
ax.spines["right"].set_color('b')
ax.spines["left"].set_color('b')

# Plot the axis labels
ax.set_xlabel('Date', fontsize=16, labelpad= 10)

#-------------------------------------
ax = plt.subplot(3,2,6)

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot PC3
ax.plot(PrinCon_Hg0.index, PrinCon_Hg0['PC3'], marker='o', c='g', markersize = 2.0, linestyle='none', label='PC3')

# Plot vertical lines to seperate the voyages
V1V2_18 = ax.axvline(datetime(2018,12,7,0,0,0),  linewidth=0.5, color='grey')
V2V3_18 = ax.axvline(datetime(2019,1,18,0,0,0),  linewidth=0.5, color='grey')

# Text boxes for the voyages
props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
a = datetime(2018,11,1)
b = datetime(2018,12,7)
c = datetime(2019,1,18)
d = datetime(2019,2,21)
ax.text(a+(b-a)/2, 4.7, "V1",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(b+(c-b)/2, 4.7, "V2",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(c+(d-c)/2, 4.7, "V3",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)

# Format the x-axis
plt.xlim(datetime(2018,11,1),datetime(2019,2,21))
xmajor_formatter = mdates.DateFormatter('%d %b') # format how the date is displayed
ax.xaxis.set_major_formatter(xmajor_formatter)

# Format y-axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5.5,5.5)
ax.axes.get_yaxis().set_visible(False)
ax.tick_params(axis='y', which='both', colors='g', labelsize=10)
ax.spines["right"].set_color('g')
ax.spines["left"].set_color('g')

# Plot the axis labels
ax.set_xlabel('Date', fontsize=16, labelpad= 10)

# #-------------------------------------
# # Custom Legend
# custom_lines = [Line2D([0], [0], color='blue',  lw=4),
#                 Line2D([0], [0], color='red',   lw=4)]
# fig1.legend(custom_lines, ['PC1','PC2'], loc='upper left', bbox_to_anchor=(0.915, 0.245), fontsize=12, title='BrO_LTcol')

#------------------------------------------------------------------------------
# PLOT THE GRAPH (PC1, PC2 & PC3 vs predictions BrOLTcol)

fig1 = plt.figure()

#-------------------------------------
ax = plt.subplot(3,2,1)
ax2 = ax.twinx()

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot PC1
ax.plot(PrinCon_Hg0.index, PrinCon_Hg0['PC1'], marker='o', c='r', markersize = 2.0, linestyle='none', label='PC1')

# Plot Hg0 predictions
ax2.plot(PC_Hg0.index, PC_Hg0['OLS_Hg0'], marker='o', c='black', markersize = 2.0, linestyle='none', label='Hg$^0$ Predictions')

# Plot vertical lines to seperate the voyages
V1V2_17 = ax.axvline(datetime(2017,12,7),  linewidth=0.5, color='grey')
V2V3_17 = ax.axvline(datetime(2018,1,18),  linewidth=0.5, color='grey')

# Text boxes for the voyages
props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
a = datetime(2017,11,10)
b = datetime(2017,12,7)
c = datetime(2018,1,18)
d = datetime(2018,2,28)
ax.text(a+(b-a)/2, 7.0, "V1",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(b+(c-b)/2, 7.0, "V2",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(c+(d-c)/2, 7.0, "V3",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)

# Format the x-axis
plt.xlim(datetime(2017,11,10),datetime(2018,2,28))
ax.axes.get_xaxis().set_visible(False)

# Format y-axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5.5,5.5)
ax.tick_params(axis='y', which='both', colors='r', labelsize=10)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax2.set_ylim(0.0,1.0)
ax2.spines["left"].set_color('r')
ax2.axes.get_yaxis().set_visible(False)

# Plot the axis labels
#ax.set_xlabel('Date', fontsize=16, labelpad= 10)
ax.set_ylabel('PC1 Loading', fontsize=16, labelpad= 5, color='r')

# Plot title
plt.title('CAMMPCAN (2017-18)', fontsize=20, y=1.2)

#-------------------------------------
ax = plt.subplot(3,2,3)
ax2 = ax.twinx()

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot PC2
ax.plot(PrinCon_Hg0.index, PrinCon_Hg0['PC2'], marker='o', c='b', markersize = 2.0, linestyle='none', label='PC2')

# Plot Hg0 predictions
ax2.plot(PC_Hg0.index, PC_Hg0['OLS_Hg0'], marker='o', c='black', markersize = 2.0, linestyle='none', label='Hg$^0$ Predictions')

# Plot vertical lines to seperate the voyages
V1V2_17 = ax.axvline(datetime(2017,12,7,0,0,0),  linewidth=0.5, color='grey')
V2V3_17 = ax.axvline(datetime(2018,1,18,0,0,0),  linewidth=0.5, color='grey')

# # Text boxes for the voyages
# props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
# a = datetime(2017,11,10)
# b = datetime(2017,12,7)
# c = datetime(2018,1,18)
# d = datetime(2018,2,28)
# ax.text(a+(b-a)/2, 4.7, "V1",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
# ax.text(b+(c-b)/2, 4.7, "V2",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
# ax.text(c+(d-c)/2, 4.7, "V3",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)

# Format the x-axis
plt.xlim(datetime(2017,11,10),datetime(2018,2,28))
xmajor_formatter = mdates.DateFormatter('%d %b') # format how the date is displayed
ax.xaxis.set_major_formatter(xmajor_formatter)
ax.axes.get_xaxis().set_visible(False)

# Format y-axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5.5,5.5)
ax.tick_params(axis='y', which='both', colors='b', labelsize=10)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax2.set_ylim(0.0,1.0)
ax2.spines["left"].set_color('b')
ax2.axes.get_yaxis().set_visible(False)

# Plot the axis labels
ax.set_xlabel('Date', fontsize=16, labelpad= 10)
ax.set_ylabel('PC2 Loading', fontsize=16, labelpad= 5, color='b')

#-------------------------------------
ax = plt.subplot(3,2,5)
ax2 = ax.twinx()

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot PC2
ax.plot(PrinCon_Hg0.index, PrinCon_Hg0['PC3'], marker='o', c='g', markersize = 2.0, linestyle='none', label='PC3')

# Plot Hg0 predictions
ax2.plot(PC_Hg0.index, PC_Hg0['OLS_Hg0'], marker='o', c='black', markersize = 2.0, linestyle='none', label='Hg$^0$ Predictions')

# Plot vertical lines to seperate the voyages
V1V2_17 = ax.axvline(datetime(2017,12,7,0,0,0),  linewidth=0.5, color='grey')
V2V3_17 = ax.axvline(datetime(2018,1,18,0,0,0),  linewidth=0.5, color='grey')

# # Text boxes for the voyages
# props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
# a = datetime(2017,11,10)
# b = datetime(2017,12,7)
# c = datetime(2018,1,18)
# d = datetime(2018,2,28)
# ax.text(a+(b-a)/2, 4.7, "V1",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
# ax.text(b+(c-b)/2, 4.7, "V2",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
# ax.text(c+(d-c)/2, 4.7, "V3",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)

# Format the x-axis
plt.xlim(datetime(2017,11,10),datetime(2018,2,28))
xmajor_formatter = mdates.DateFormatter('%d %b') # format how the date is displayed
ax.xaxis.set_major_formatter(xmajor_formatter)

# Format y-axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5.5,5.5)
ax.tick_params(axis='y', which='both', colors='g', labelsize=10)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax2.set_ylim(0.0,1.0)
ax2.spines["left"].set_color('g')
ax2.axes.get_yaxis().set_visible(False)

# Plot the axis labels
ax.set_xlabel('Date', fontsize=16, labelpad= 10)
ax.set_ylabel('PC3 Loading', fontsize=16, labelpad= 5, color='g')

#-------------------------------------
ax = plt.subplot(3,2,2)
ax2 = ax.twinx()

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot PC1
ax.plot(PrinCon_Hg0.index, PrinCon_Hg0['PC1'], marker='o', c='r', markersize = 2.0, linestyle='none', label='PC1')

# Plot Hg0 predictions
ax2.plot(PC_Hg0.index, PC_Hg0['OLS_Hg0'], marker='o', c='black', markersize = 2.0, linestyle='none', label='Hg$^0$ Predictions')

# Plot vertical lines to seperate the voyages
V1V2_18 = ax.axvline(datetime(2018,12,7,0,0,0),  linewidth=0.5, color='grey')
V2V3_18 = ax.axvline(datetime(2019,1,18,0,0,0),  linewidth=0.5, color='grey')

# Text boxes for the voyages
props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
a = datetime(2018,11,1)
b = datetime(2018,12,7)
c = datetime(2019,1,18)
d = datetime(2019,2,21)
ax.text(a+(b-a)/2, 7.0, "V1",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(b+(c-b)/2, 7.0, "V2",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
ax.text(c+(d-c)/2, 7.0, "V3",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)

# Format the x-axis
plt.xlim(datetime(2018,11,1),datetime(2019,2,21))
ax.axes.get_xaxis().set_visible(False)

# Format y-axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5.5,5.5)
ax.axes.get_yaxis().set_visible(False)
ax.tick_params(axis='y', which='both', colors='r', labelsize=10)

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax2.set_ylim(0.0,1.0)
ax2.spines["left"].set_color('r')

# Plot title
plt.title('CAMMPCAN (2018-19)', fontsize=20, y=1.2)

# Plot the axis labels
ax2.set_ylabel('Hg$^0$ (ng/m$^3$)', labelpad= 5, fontsize=16)

#-------------------------------------
ax = plt.subplot(3,2,4)
ax2 = ax.twinx()

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot PC2
ax.plot(PrinCon_Hg0.index, PrinCon_Hg0['PC2'], marker='o', c='b', markersize = 2.0, linestyle='none', label='PC2')

# Plot Hg0 predictions
ax2.plot(PC_Hg0.index, PC_Hg0['OLS_Hg0'], marker='o', c='black', markersize = 2.0, linestyle='none', label='Hg$^0$ Predictions')

# Plot vertical lines to seperate the voyages
V1V2_18 = ax.axvline(datetime(2018,12,7,0,0,0),  linewidth=0.5, color='grey')
V2V3_18 = ax.axvline(datetime(2019,1,18,0,0,0),  linewidth=0.5, color='grey')

# # Text boxes for the voyages
# props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
# a = datetime(2018,11,1)
# b = datetime(2018,12,7)
# c = datetime(2019,1,18)
# d = datetime(2019,2,21)
# ax.text(a+(b-a)/2, 4.7, "V1",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
# ax.text(b+(c-b)/2, 4.7, "V2",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
# ax.text(c+(d-c)/2, 4.7, "V3",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)

# Format the x-axis
plt.xlim(datetime(2018,11,1),datetime(2019,2,21))
xmajor_formatter = mdates.DateFormatter('%d %b') # format how the date is displayed
ax.xaxis.set_major_formatter(xmajor_formatter)
ax.axes.get_xaxis().set_visible(False)

# Format y-axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5.5,5.5)
ax.axes.get_yaxis().set_visible(False)
ax.tick_params(axis='y', which='both', colors='b', labelsize=10)
ax.spines["right"].set_color('b')
ax.spines["left"].set_color('b')

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax2.set_ylim(0.0,1.0)
ax2.spines["left"].set_color('b')

# Plot the axis labels
ax.set_xlabel('Date', fontsize=16, labelpad= 10)
ax2.set_ylabel('Hg$^0$ (ng/m$^3$)', labelpad= 5, fontsize=16)

#-------------------------------------
ax = plt.subplot(3,2,6)
ax2 = ax.twinx()

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot PC2
ax.plot(PrinCon_Hg0.index, PrinCon_Hg0['PC3'], marker='o', c='g', markersize = 2.0, linestyle='none', label='PC3')

# Plot Hg0 predictions
ax2.plot(PC_Hg0.index, PC_Hg0['OLS_Hg0'], marker='o', c='black', markersize = 2.0, linestyle='none', label='Hg$^0$ Predictions')

# Plot vertical lines to seperate the voyages
V1V2_18 = ax.axvline(datetime(2018,12,7,0,0,0),  linewidth=0.5, color='grey')
V2V3_18 = ax.axvline(datetime(2019,1,18,0,0,0),  linewidth=0.5, color='grey')

# # Text boxes for the voyages
# props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
# a = datetime(2018,11,1)
# b = datetime(2018,12,7)
# c = datetime(2019,1,18)
# d = datetime(2019,2,21)
# ax.text(a+(b-a)/2, 4.7, "V1",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
# ax.text(b+(c-b)/2, 4.7, "V2",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
# ax.text(c+(d-c)/2, 4.7, "V3",  color='black', fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)

# Format the x-axis
plt.xlim(datetime(2018,11,1),datetime(2019,2,21))
xmajor_formatter = mdates.DateFormatter('%d %b') # format how the date is displayed
ax.xaxis.set_major_formatter(xmajor_formatter)

# Format y-axis
ax.yaxis.set_major_locator(ticker.MultipleLocator(2.5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.set_ylim(-5.5,5.5)
ax.axes.get_yaxis().set_visible(False)
ax.tick_params(axis='y', which='both', colors='g', labelsize=10)
ax.spines["right"].set_color('g')
ax.spines["left"].set_color('g')

# Format y-axis 2
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax2.set_ylim(0.0,1.0)
ax2.spines["left"].set_color('g')

# Plot the axis labels
ax.set_xlabel('Date', fontsize=16, labelpad= 10)
ax2.set_ylabel('Hg$^0$ (ng/m$^3$)', labelpad= 5, fontsize=16)

# #-------------------------------------
# # Custom Legend
# custom_lines = [Line2D([0], [0], color='blue',  lw=4),
#                 Line2D([0], [0], color='red',   lw=4)]
# fig1.legend(custom_lines, ['PC1','PC2'], loc='upper left', bbox_to_anchor=(0.915, 0.245), fontsize=12, title='BrO_LTcol')
