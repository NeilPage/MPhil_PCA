#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:46:38 2021

@author: ncp532
"""
# Drawing packages
import matplotlib.pyplot as plt             
import matplotlib.dates as mdates            
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

# Data handing packages
import numpy as np                           
import pandas as pd

#------------------------------------------------------------------------------
# DEFINE THE DATASETS
#dfPC_Loadings = pd.read_csv('/Users/ncp532/Documents/Data/PCA_Hg0/PC_Loadings_Hg0.csv', index_col=0) # BrO
dfPC_Loadings = pd.read_csv('/Users/ncp532/Documents/Data/PCA_Hg0/Without_BrO/PC_Loadings_Hg0_noBrO.csv', index_col=0) # No BrO

#------------------------------------------------------------------------------
# PLOT THE GRAPH (Individuial PCs)

fig1 = plt.figure()

#-------------------------------------
# PC1
ax = plt.subplot(3,1,1)

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot horizontal gridlines
plt.grid(True, which='major', axis='y')

# Set labels for x-axis 

my_xticks = ['O$_3$', 'Surf Pres', 'Surf Temp', 'PTD100m', 'PTD1000m', 'WS10m', 'MLH', 'Sol Rad',
             'Ice Contact', 'Sea Ice Conc', 'Water Temp', 'Rel Hum']

# Plot loadings for PC1
ax.bar(my_xticks, dfPC_Loadings['PC1'], width=0.8, color='r', zorder=2)

# Format x-axis
plt.setp(ax.get_xticklabels(), Rotation=45)
ax.axes.get_xaxis().set_visible(False)

# Format y-axis 
ax.set_ylim(-0.6, 0.6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.tick_params(axis='both', labelsize=16)

# Plot axis labels and titles
plt.ylabel('PC1 Loading', fontsize=25, labelpad= 5)
#plt.xlabel('Environmental Variables', fontsize=25, labelpad= 10)
plt.title('PCA Loadings', fontsize=25, y=1.05)

#-------------------------------------
# PC2
ax = plt.subplot(3,1,2)

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot horizontal gridlines
plt.grid(True, which='major',axis='y')

# Plot loadings for PC1
ax.bar(my_xticks, dfPC_Loadings['PC2'], width=0.8, color='b', zorder=2)

# Format x-axis
plt.setp(ax.get_xticklabels(), Rotation=45)
ax.axes.get_xaxis().set_visible(False)

# Format y-axis 
ax.set_ylim(-0.6, 0.6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.tick_params(axis='both', labelsize=16)

# Plot axis labels and titles
plt.ylabel('PC2 Loading', fontsize=25, labelpad= 5)
plt.xlabel('Environmental Variables', fontsize=25, labelpad= 10)
#plt.title('PCA Loadings', fontsize=25, y=1.05)

#-------------------------------------
# PC3
ax = plt.subplot(3,1,3)

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot horizontal gridlines
plt.grid(True, which='major',axis='y')

# Plot loadings for PC1
ax.bar(my_xticks, dfPC_Loadings['PC3'], width=0.8, color='g', zorder=2)

# Format x-axis
plt.setp(ax.get_xticklabels(), Rotation=45)

# Format y-axis 
ax.set_ylim(-0.6, 0.6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.tick_params(axis='both', labelsize=16)

# Plot axis labels and titles
plt.ylabel('PC3 Loading', fontsize=25, labelpad= 5)
plt.xlabel('Environmental Variables', fontsize=25, labelpad= 10)
#plt.title('PCA Loadings', fontsize=25, y=1.05)

plt.show()

#------------------------------------------------------------------------------
# PLOT THE GRAPH (Adding PCs)

fig1 = plt.figure()

#-------------------------------------
# PC1
ax = plt.subplot(3,1,1)

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot horizontal gridlines
plt.grid(True, which='major', axis='y')

# Set labels for x-axis 

my_xticks = ['O$_3$', 'Surf Pres', 'Surf Temp', 'PTD100m', 'PTD1000m', 'WS10m', 'MLH', 'Sol Rad',
             'Ice Contact', 'Sea Ice Conc', 'Water Temp', 'Rel Hum']

# Plot loadings for PC1
ax.bar(my_xticks, dfPC_Loadings['PC1'] + dfPC_Loadings['PC2'], width=0.8, color='r', zorder=2)

# Format x-axis
plt.setp(ax.get_xticklabels(), Rotation=45)
ax.axes.get_xaxis().set_visible(False)

# Format y-axis 
ax.set_ylim(-0.6, 0.6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.tick_params(axis='both', labelsize=16)

# Plot axis labels and titles
plt.ylabel('PC1 + PC2\nLoading', fontsize=25, labelpad= 5)
#plt.xlabel('Environmental Variables', fontsize=25, labelpad= 10)
plt.title('PCA Loadings', fontsize=25, y=1.05)

#-------------------------------------
# PC2
ax = plt.subplot(3,1,2)

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot horizontal gridlines
plt.grid(True, which='major',axis='y')

# Plot loadings for PC1
ax.bar(my_xticks, dfPC_Loadings['PC1']  + dfPC_Loadings['PC3'], width=0.8, color='b', zorder=2)

# Format x-axis
plt.setp(ax.get_xticklabels(), Rotation=45)
ax.axes.get_xaxis().set_visible(False)

# Format y-axis 
ax.set_ylim(-0.6, 0.6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.tick_params(axis='both', labelsize=16)

# Plot axis labels and titles
plt.ylabel('PC1 + PC3\nLoading', fontsize=25, labelpad= 5)
plt.xlabel('Environmental Variables', fontsize=25, labelpad= 10)
#plt.title('PCA Loadings', fontsize=25, y=1.05)

#-------------------------------------
# PC3
ax = plt.subplot(3,1,3)

# Plot horizontal axis line at 0
ax.axhline(y=0, c='black', linewidth=1)

# Plot horizontal gridlines
plt.grid(True, which='major',axis='y')

# Plot loadings for PC1
ax.bar(my_xticks, dfPC_Loadings['PC2'] + dfPC_Loadings['PC3'], width=0.8, color='g', zorder=2)

# Format x-axis
plt.setp(ax.get_xticklabels(), Rotation=45)

# Format y-axis 
ax.set_ylim(-0.6, 0.6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.tick_params(axis='both', labelsize=16)

# Plot axis labels and titles
plt.ylabel('PC2 + PC3\nLoading', fontsize=25, labelpad= 5)
plt.xlabel('Environmental Variables', fontsize=25, labelpad= 10)
#plt.title('PCA Loadings', fontsize=25, y=1.05)

plt.show()