#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:59:57 2020

@author: xuhuiying
"""

#“实验二、对比性能 ———— 6.调节第1个卖家的成本系数，看每个卖家的效益的变化”的主体代码

import gambitFunc
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas

#参数的默认值
N = 3              #卖家数
M = 10             #买家数
c_max = 20         #卖家的成本系数的上限值
V_max = 500        #买家的任务完成奖励的上限值 
a_max = 2          #买家的完成任务所需的CPU工作时的上限值
y_min =  0.020     #动作空间的下限值
y_max =  0.060     #动作空间的上限值
actionNumber = 4   #动作空间的大小
repeatTimes = 10   #单次实验的重复次数
changeSeq = range(2,c_max,2) #第0个卖家的成本系数的变化序列
#changeSeq = range(7,16,1) #第0个卖家的成本系数的变化序列


c = np.array([c_max / 2] * N)   #三个卖家的成本系数
c[1] = c_max / 2 - 2            #第1个卖家
c[2] = c_max / 2 + 2            #第2个卖家

sellerUtilitiesMeanRecord = [] 
for c[0] in changeSeq:               #改变第0个卖家的成本系数
    sellerUtilitiesHistory = []
    for time in range(0,repeatTimes):
        print("\nc[0] =",c[0],"time =",time)
        while 1:
            #每次随机生成不同的买家参数
            V = np.random.uniform(V_max - 10, V_max, size = M)
            a = np.random.uniform(a_max - 0.1, a_max, size = M)
        
            #调用gambitFunc函数，返回由【每个卖家的效益】组成的数组
            sellerUtilities = gambitFunc.gambitFunc(N,M,c,V,a,y_min,y_max,actionNumber)
            if type(sellerUtilities) == type(c):#c是由每个卖家的成本系数组成的数组，是np.array类型
                break
        sellerUtilitiesHistory.append(sellerUtilities)
    #在重复实验结束后，处理sellerUtilitiesHistory[]
    sellerUtilitiesHistory = np.array(sellerUtilitiesHistory)
    sellerUtilitiesMean = np.mean(sellerUtilitiesHistory,axis = 0)        #平均值
    sellerUtilitiesMeanRecord.append(sellerUtilitiesMean)
sellerUtilitiesMeanRecord = np.array(sellerUtilitiesMeanRecord)
sellerUtilitiesMeanRecord = sellerUtilitiesMeanRecord.T

#将画图数据保存一份.将changeSeq、sellerUtilitiesMeanRecord等数据保存到同名.csv文件
dic = {"c_0":changeSeq,
        "seller_1_utility":sellerUtilitiesMeanRecord[0],
        "seller_2_utility":sellerUtilitiesMeanRecord[1],
        "seller_3_utility":sellerUtilitiesMeanRecord[2]}
data = pandas.DataFrame(dic,
                        columns = ["c_0","seller_1_utility",
                                   "seller_2_utility","seller_3_utility"])
data.to_csv('实验二、对比性能 ———— 6.调节第1个卖家的成本系数，看每个卖家的效益的变化.csv',
            index = False)

#根据sellerUtilitiesMeanRecord[]画图
markerList = ['o-','s-','x-']
plt.figure()
for j in range(0,N):
    plt.plot(changeSeq,sellerUtilitiesMeanRecord[j],markerList[j],label = "seller %d"%(j + 1))
plt.legend(prop = {'size': 14},handlelength = 1)
plt.grid()
plt.xlabel('Cost factor of the first seller $c_1$',fontsize = 14)  
plt.ylabel("Seller's utility",fontsize = 14)  
plt.tick_params(labelsize=13)
plt.savefig('实验二、对比性能 ———— 6.调节第1个卖家的成本系数，看每个卖家的效益的变化.jpg', 
            bbox_inches = 'tight',dpi = 300)
plt.show()

    
    
    
    