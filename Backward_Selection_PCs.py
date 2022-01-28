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
dfPC_Select = pd.read_csv('/Users/ncp532/Documents/Data/MERRA2/PC_BackwardSelection_NoWD.csv', index_col=1)

#------------------------------------------------------------------------------
# PLOT THE GRAPH (PC_BrOLTcol)

#--------------------
# How many variables are needed to explain variance in the data?
#--------------------
fig, ax = plt.subplots()

# Plot R^2
ax.plot(dfPC_Select.index, dfPC_Select['R^2'], marker='o', c='b', markersize = 6.0, linestyle='--', label='R$^2$')
plt.grid(True)

# Format x-axis 
ax.set_xlim(14.5, 0.5)
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.tick_params(axis='both', labelsize=16)

# Format y-axis 
ax.set_ylim(0, 0.35)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

# Plot axis labels and titles
plt.ylabel('R$^2$', fontsize=25, labelpad= 5)
plt.xlabel('Number of PC used', fontsize=25, labelpad= 10)
plt.title('How the number of components affects R$^2$', fontsize=25, y=1.05)

plt.show()
