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

#画图
markerList = ['o-','s-','x-']
plt.figure()
plt.plot(data.c_0,data.seller_1_utility,markerList[0],label = "seller 1")#卖家1
plt.plot(data.c_0,data.seller_2_utility,markerList[1],label = "seller 2")#卖家2
plt.plot(data.c_0,data.seller_3_utility,markerList[2],label = "seller 3")#卖家3
plt.legend(prop = {'size': 14},handlelength = 1)
plt.grid()
plt.xlabel('Cost factor of the first seller $c_1$',fontsize = 14)  
plt.ylabel("Seller's utility",fontsize = 14)  
plt.tick_params(labelsize=13)
plt.savefig('实验二、对比性能 ———— 6.调节第1个卖家的成本系数，看每个卖家的效益的变化.jpg',
            bbox_inches = 'tight',dpi = 300)
plt.show()