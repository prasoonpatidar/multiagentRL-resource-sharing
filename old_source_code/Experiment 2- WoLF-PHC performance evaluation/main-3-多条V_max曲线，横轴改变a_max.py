#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#“实验二 WoLF-PHC算法的性能 ———— 2.多条V_max曲线，横轴改变a_max”的主体代码
# experiment 2 Performance evaluation  of WoLF-PHC algorithm---
# Overall performance when 𝑎max varies under different values of 𝑉max.

"""
在测试实验曲线OK的情况下，改:N = 3, repeatTimes = 50，开始跑
If the experiment results looks good, change the repeatTimes = 50, rerun the experiment
"""

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
repeatTimes = 50   #单次实验的重复次数  iteration times for each experiment

curveVariableStart = 350
curveVariableEnd = 501
curveVariableInterval = 50
mainVariableStart = 0.7
mainVariableEnd = 2.71
mainVariableInterval = 0.4

for V_max in np.arange(curveVariableStart,curveVariableEnd,curveVariableInterval):
    gambitResultMeanRecord = []
    wolfphcResultMeanRecord = []
    for a_max in np.arange(mainVariableStart,mainVariableEnd,mainVariableInterval):
        gambitResultHistory = []
        wolfphcResultHistory = []
        for time in range(0,repeatTimes):
            print("V_max = %r, a_max = %r, time = %r"%(V_max,a_max,time))
            while 1:
                #每次随机生成不同的卖家参数、买家参数
                # random generation of parameters
                c = np.random.uniform(c_max - 3, c_max, size = N)
                V = np.random.uniform(V_max - 50, V_max, size = M)
                a = np.random.uniform(a_max - 0.2, a_max, size = M)
                
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
    saveToCsv.saveToCsv("a_max",np.arange(mainVariableStart,mainVariableEnd,mainVariableInterval),
                        gambitResultMeanRecord,"V_max=%r_varyinga_max_gambit.csv"%V_max)
    saveToCsv.saveToCsv("a_max",np.arange(mainVariableStart,mainVariableEnd,mainVariableInterval),
                        wolfphcResultMeanRecord,"V_max=%r_varyinga_max_wolfphc.csv"%V_max)
    
#在实验结束后，画出七个性能指标图
# plot the performance evaluation results, compare the seven indicators
plotFunc.plotAllSevenPerformance("V_max",curveVariableStart,curveVariableEnd,
                                 curveVariableInterval,"$V_{max}$",
                                 "a_max","$a_{max}$")

    
    
    
    