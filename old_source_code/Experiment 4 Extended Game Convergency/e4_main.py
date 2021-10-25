#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:59:57 2020

@author: xuhuiying
"""

#“实验一、对比收敛速度”的主体代码
# experiment 1--main--convergence comparasion

import wolfphc_MultiState
import qlearning
import numpy as np
import matplotlib.pyplot as plt
import plotHistory as ph
import pickle

experiment_name='baseline'
# experiment_config = {
#     'producer': {
#         'count':3,
#         'max_unit_cost':20,
#
#     },
#     'device':{
#         'count':5,
#         'device':
#     }
#     'producer_count':3,
#     'device_count':5,
#     'max_unit_cost':20,
#     'max_utility':500,
#     'upper_bound_cpu_time':2,
#     ''
# }

#参数的默认值 parameters
N = 3              #卖家数 Number of providers
M = 5              #买家数  Number of IoT devices
c_max = 20         #卖家的成本系数的上限值 Upper bound of c_j (Unit cost for computing service of j)
V_max = 500        #买家的任务完成奖励的上限值  Upper bound of V_i (task completion utility of i)
a_max = 2          #买家的完成任务所需的CPU工作时的上限值 upper bound of a_i (required CPU occupied time of i)
y_min =  0.020     #动作空间的下限值 Min Auxiliary price profile for all providers
y_max =  0.060     #动作空间的上限值 Max Auxiliary price profile for all providers
actionNumber = 10   #动作空间的大小 Action space of j
times = 10000      #迭代次数 iteration times

results_dir = 'results'

#生成买家参数、卖家参数
# generate provider's unit cost, task completion utility of device j, and required resource occupied time of i
c = np.random.uniform(c_max - 10, c_max, size = N)
V = np.random.uniform(V_max - 300, V_max, size = M)
a = np.random.uniform(a_max - 0.5, a_max, size = M)
max_resources_per_seller = np.random.randint(5,100,N)
#使用相同的输入参数。分别用以下20算法跑一次，对比收敛速度。
#Use the same set of parameters (c, V,a) to run Q-learning and WoLF-PHC for convergence comparasion
# QpricesHistory,QpurchasesHistory,Qtimes = qlearning.qlearning(N,M,c,V,a,y_min,y_max,actionNumber,times)#Q-learning算法
WpricesHistory,WpurchasesHistory, WprovidedResourcesHistory,Wtimes = wolfphc_MultiState.wolfphc_MultiState(N,M,c,V,a,y_min,y_max,actionNumber,times, max_resources_per_seller)#WoLF-PHC算法

'''
Write data to be able to generate folloring plots
1. Plots A: change in resources provided to consumers over time(per device level)
2. Plots B: change in resources asked by consumers over time(per device level)
3. Plots C: change in producer pricing over time(per producer level)
'''

experiment_summary_filename = 'baseline'

with open(f'{results_dir}/{experiment_summary_filename}.pb','w') as f:
    pickle.dump(
        {
            'experiment':experiment_summary_filename,
            ''
        }
    )




# #画定价图
# # plot auxiliary price
# plt.figure()
# #画出Q-leanring每个卖家的price的变化
# # plot Q-learning Auxiliary price changes
# plt.subplot(2,2,1)
# # ph.plotHistory(QpricesHistory[::100],Qtimes//100,"Iterations during Q-learning","Price","provider")
# #画出WoLF-PHC每个卖家的price的变化
# # plot WoLF-PHC Auxiliary price changes
# plt.subplot(2,2,2)
# ph.plotHistory(WpricesHistory[::100],Wtimes//100,"Iterations during RLPM",None,"provider")
# plt.legend(bbox_to_anchor=(0.35, 1.3), ncol=2, prop = {'size': 8},handlelength = 0.8)
# #plt.legend(loc='upper left',prop = {'size': 10},handlelength = 1)
# plt.savefig('Convergence-price.png', dpi=300,bbox_inches = 'tight')
# # plt.show()
#
# #画购买量图
# # plot the purchased resources
# plt.figure()
# #画出Q-leanring每个买家的总购买数量的变化
# # plot the each device's total resource purchase  -->using Q-learning to generate price strategy
# plt.subplot(2,2,3)
# # ph.plotHistory(QpurchasesHistory[::100],Qtimes//100,"Iterations during Q-learning","Demand","device")
# #画出WoLF-PHC每个买家的总购买数量的变化
# # plot the each device's total resource purchase  -->using WoLF-PHC to generate price strategy
# plt.subplot(2,2,4)
# ph.plotHistory(WpurchasesHistory[::100],Wtimes//100,"Iterations during RLPM",None,"device")
# plt.legend(bbox_to_anchor=(0.85, 1.3), ncol=5, prop = {'size': 8},handlelength = 0.7)
# plt.savefig('Convergence-purchase.pdf', dpi=300,bbox_inches = 'tight')
# # plt.show()
#
# print("experiment 4 completes。")
# # print experiment 1 completes
#
# #while 1:
# #    #生成买家参数、卖家参数
# #    c = np.random.uniform(c_max - 10, c_max, size = N)
# #    V = np.random.uniform(V_max - 300, V_max, size = M)
# #    a = np.random.uniform(a_max - 0.5, a_max, size = M)
# #
# #    #使用相同的输入参数。分别用以下2个算法跑一次，对比收敛速度。
# #    returnValue = wolfphc_MultiState.wolfphc_MultiState(N,M,c,V,a,y_min,y_max,actionNumber)#WoLF-PHC算法
# #    if returnValue == False:
# #        continue
# #    qlearning.qlearning(N,M,c,V,a,y_min,y_max,actionNumber)#Q-learning算法
# #    break
# #
