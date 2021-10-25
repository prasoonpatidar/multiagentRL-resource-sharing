#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:18:29 2020

@author: xuhuiying
"""

import numpy as np
import matplotlib.pyplot as plt

def plotHistory(history,times,xLabelText,yLabelText,legendText):#画出每个history plot each history
    history = np.array(history) #history是二维数组 history is a 2D array
    history = history.T
    iteration = range(0,times)
#    plt.figure()
    for j in range(0,history.shape[0]):
        plt.plot(iteration,history[j],label = "%s %d"%(legendText,j + 1))
#    plt.legend(loc='upper left',prop = {'size': 10},handlelength = 1)
    plt.xlabel(xLabelText,fontsize = 8)
    plt.ylabel(yLabelText,fontsize = 8)
    plt.tick_params(labelsize=8)
#    plt.savefig('%sWith%dSellersAnd%dBuyer.jpg'%(fileNamePre,N,M), dpi=300,bbox_inches = 'tight')
#    plt.show()
