#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###################################################
#              根据已有的画图数据，画出曲线            #  
###################################################


import pandas
import matplotlib.pyplot as plt


#读入数据
data = pandas.read_csv('实验二、对比性能 ———— 6.调节第1个卖家的成本系数，看每个卖家的效益的变化.csv',
                       names = ["c_0","seller_1_utility","seller_2_utility",
                                "seller_3_utility"],header = 0)
# # experiment 2--tuning provider's cost to investigate each provider's utility changes

#画图 plot
markerList = ['o-','s-','x-']
plt.figure()
plt.plot(data.c_0,data.seller_1_utility,markerList[0],label = "provider 1")#卖家1 provider 1
plt.plot(data.c_0,data.seller_2_utility,markerList[1],label = "provider 2")#卖家2 provider 2
plt.plot(data.c_0,data.seller_3_utility,markerList[2],label = "provider 3")#卖家3 provider 3
plt.legend(prop = {'size': 14},handlelength = 1)
plt.grid()
plt.xlabel('$c_1$',fontsize = 14)  
plt.ylabel("Provider's utility",fontsize = 14)  
plt.tick_params(labelsize=13)

ax = plt.gca()  # 获取当前图像的坐标轴信息
ax.yaxis.get_major_formatter().set_powerlimits((0,3)) # 将坐标轴的base number设置为一位。

plt.savefig('provider_utility_vs_cost_factor_of_the_first_provider.pdf',
            bbox_inches = 'tight',dpi = 300)
plt.show()