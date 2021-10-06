#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#“实验二 WoLF-PHC算法的性能 ———— 1.多条N曲线，横轴改变M”的主体代码
# experiment 2 Performance evaluation  of WoLF-PHC algorithm---
# Overall performance when the number of IoT devices 𝑀 varies with different numbers of providers 𝑁.

import numpy as np
import wolfphc_MultiState
import gambitFunc
import saveToCsv
import plotFunc

#参数的默认值
# default parameters
N = 3              #卖家数 Providers
M = 100             #买家数 IoT devices
c_max = 20         #卖家的成本系数的上限值 Upper bound of c_j (Unit cost for computing service of j)
V_max = 500        #买家的任务完成奖励的上限值 Upper bound of V_i (task completion utility of i)
a_max = 2          #买家的完成任务所需的CPU工作时的上限值 upper bound of a_i (required CPU occupied time of i)
y_min = 0.020      #动作空间的下限值 Min Auxiliary price profile for all providers
y_max = 0.060      #动作空间的上限值 Max Auxiliary price profile for all providers
actionNumber = 4   #动作空间的大小 Action space of j
repeatTimes = 50    #单次实验的重复次数 iteration times each experiment

curveVariableStart = 1
curveVariableEnd = 4
curveVariableInterval = 1
mainVariableStart = 1
mainVariableEnd = 11
mainVariableInterval = 4

for N in np.arange(curveVariableStart,curveVariableEnd,curveVariableInterval):
    gambitResultMeanRecord = []
    wolfphcResultMeanRecord = []
    for M in np.arange(mainVariableStart,mainVariableEnd,mainVariableInterval):
        gambitResultHistory = []
        wolfphcResultHistory = []
        for time in range(0,repeatTimes):
            print("N = %r, M = %r, time = %r"%(N,M,time))
            while 1:
                #每次随机生成不同的卖家参数、买家参数
                # random generation of parameters
                c = np.random.uniform(c_max - 10, c_max, size = N)
                V = np.random.uniform(V_max - 50, V_max, size = M)
                a = np.random.uniform(a_max - 0.5, a_max, size = M)
                
                gambitResult = gambitFunc.gambitFunc(N,M,c,V,a,y_min,y_max,actionNumber)
                if type(gambitResult) == type(False):
                    continue
                wolfphcResult = wolfphc_MultiState.wolfphc_MultiState(N,M,c,V,a,y_min,y_max,actionNumber)
                if type(wolfphcResult) == type(False):
                    continue
                break
            gambitResultHistory.append(gambitResult)
            wolfphcResultHistory.append(wolfphcResult)
        gambitResultHistory = np.array(gambitResultHistory)
        wolfphcResultHistory = np.array(wolfphcResultHistory)
        gambitResultMean = np.mean(gambitResultHistory,axis = 0)
        wolfphcResultMean = np.mean(wolfphcResultHistory,axis = 0)
        gambitResultMeanRecord.append(gambitResultMean)
        wolfphcResultMeanRecord.append(wolfphcResultMean)
    gambitResultMeanRecord = np.array(gambitResultMeanRecord)
    wolfphcResultMeanRecord = np.array(wolfphcResultMeanRecord)
    gambitResultMeanRecord = gambitResultMeanRecord.T
    wolfphcResultMeanRecord = wolfphcResultMeanRecord.T
    
    #保存数据
    # save the data
    saveToCsv.saveToCsv("M",np.arange(mainVariableStart,mainVariableEnd,mainVariableInterval),
                        gambitResultMeanRecord,"N=%r_varyingM_gambit.csv"%N)
    saveToCsv.saveToCsv("M",np.arange(mainVariableStart,mainVariableEnd,mainVariableInterval),
                        wolfphcResultMeanRecord,"N=%r_varyingM_wolfphc.csv"%N)
    
#在实验结束后，画出七个性能指标图
# plot the performance evaluation results, compare the seven indicators
plotFunc.plotAllSevenPerformance("N",curveVariableStart,curveVariableEnd,
                                 curveVariableInterval,"$N$","M","Number of IoT devices")

    
    
    
    