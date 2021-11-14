#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:54:22 2020

@author: xuhuiying
"""

import numpy as np
import math

def performance(X,ys,c,V,a,N,M):
#在给定所有卖家的定价和所有买家的购买数量的情况下，计算七个性能指标值并返回。
# return performance matrix for the given providers prices and given amount of resource purchases
#    输入参数：input variables
#    X是在最后一次迭代时，由数组x_j组成的数组。x_j是由【所有买家向第j个卖家购买的产品数量】组成的数组。
# X is a list for all the x_j at the last iteration. x_j is a list of [number of resources provider i provided to all the devices]
#    ys是在最后一次迭代时，由动作值y组成的数组。
# ys is  a list [all the actions y] at the last iteration
#    c是由N个卖家的【成本系数】组成的数组。
# c is a list [N providers cost]
#    V是由M个买家的【任务完成奖励】组成的数组。
#V is a list [M devices' reward for complete the task]
#    a是由M个买家的【完成任务所需的CPU工作时】组成的数组。
# a is a list for M devices ' [CPU needs for complete the task]
#    N是卖家数 N --> number of providers
#    M是买家数 M--> number of devices
    
    #前期准备：
    # before the experiment
    #计算每个卖家的定价
    # get each provider's price
    prices = 1 / ys
    #prices是由【每个卖家的报价】组成的数组。
    # prices is a list describes each providers price
    
    #卖家-- providers
    sellerRevenues = sellerRevenuesCalculator(X,ys,N)
    #sellerRevenues是由【每个卖家的收入】组成的数组。
    # sellerRevenues--a list for all each providers revenue
    sellerExpenses = sellerExpensesCalculator(X,ys,c,N)
    #sellerExpenses是由【每个卖家的成本】组成的数组。
    #sellerExpenses a list for each providers cost
    sellerUtilities = sellerRevenues - sellerExpenses
    #sellerUtilities是由【每个卖家的效益】组成的数组。
    # a list for each provider's utility
    
    #买家--device
    buyerRevenues = buyerRevenuesCalculator(X,ys,V,a,N,M)
    #buyerRevenues是由【每个买家的收入】组成的数组。
    # a list for each devices revenue
    buyerExpenses = buyerExpensesCalculator(X,ys,N,M)
    #buyerExpenses是由【每个买家的付费】组成的数组。
    # a list for each devices' cost
    buyerUtilities = buyerRevenues - buyerExpenses
    #buyerUtilities是由【每个买家的效益】组成的数组。
    # a list for each devices' utility
    
    
    #计算七个返回值：
    # calculate the performance matrix
    socialWelfare = sum(sellerUtilities) + sum (buyerUtilities)#1.社会福利 social warfare
    meanSellerUtility = np.mean(sellerUtilities)               #2.平均卖家效益 average utilitiy for providers
    meanBuyerUtility = np.mean(buyerUtilities)                 #3.平均买家效益 average utility for devices
    meanPrice = np.mean(prices)                                #4.卖家的平均定价 providers average price
    meanPurchases = np.mean(X)                                 #5.买家的平均购买数量 device's average resource purachase
    meanSellerRevenue = np.mean(sellerRevenues)                #6.卖家的平均收入 providers average revenue
    meanBuyerRevenue = np.mean(buyerRevenues)                  #7.买家的平均收入 devices average revenue
    
    #返回由七个返回值组成的数组，类型为np.array
    # Return performance matrix, datatype: np array
    return np.array([socialWelfare,meanSellerUtility,meanBuyerUtility,\
meanPrice,meanPurchases,meanSellerRevenue,meanBuyerRevenue])

def sellerAndBuyerUtilities(X,ys,c,V,a,N,M):
#在给定所有卖家的定价和所有买家的购买数量的情况下，计算所有卖家的的效益和所有买家的效益并返回
# for a given provider price and device purchase, calculate provider utility and device utility
#    输入参数：input
#    X是在最后一次迭代时，由数组x_j组成的数组。x_j是由【所有买家向第j个卖家购买的产品数量】组成的数组。
# X is a list for all the x_j at the last iteration. x_j is a list of [number of resources provider i provided to all the devices]
#    ys是在最后一次迭代时，由动作值y组成的数组。
# ys is  a list [all the actions y] at the last iteration
#    c是由N个卖家的【成本系数】组成的数组。
# c is a list [N providers cost]
#    V是由M个买家的【任务完成奖励】组成的数组。
# V is a list [M devices' reward for complete the task]
#    a是由M个买家的【完成任务所需的CPU工作时】组成的数组。
# a is a list for M devices ' [CPU needs for complete the task]
#    N是卖家数 N --> number of providers
#    M是买家数 M--> number of devices

#前期准备： before experiment
    
    #卖家--provider
    sellerRevenues = sellerRevenuesCalculator(X,ys,N) #sellerRevenues是由【每个卖家的收入】组成的数组。
    # sellerRevenues--a list for all each providers revenue
    sellerExpenses = sellerExpensesCalculator(X,ys,c,N) #sellerExpenses是由【每个卖家的成本】组成的数组。
    #sellerExpenses a list for each providers cost
    sellerUtilities = sellerRevenues - sellerExpenses  #sellerUtilities是由【每个卖家的效益】组成的数组。
    # a list for each provider's utility
    
    #买家--device
    buyerRevenues = buyerRevenuesCalculator(X,ys,V,a,N,M) #buyerRevenues是由【每个买家的收入】组成的数组。
    # a list for each devices revenue
    buyerExpenses = buyerExpensesCalculator(X,ys,N,M)     #buyerExpenses是由【每个买家的付费】组成的数组。
    # a list for each devices' cost
    buyerUtilities = buyerRevenues - buyerExpenses        #buyerUtilities是由【每个买家的效益】组成的数组。
    # a list for each devices' utility

    
    #返回两个返回值：return sellerUtilities and buyerUtilities
    #1.sellerUtilities，由【每个卖家的效益】组成的数组；
    #2.buyerUtilities，由【每个买家的效益】组成的数组。
    return sellerUtilities,buyerUtilities

    
def buyerRevenuesCalculator(X,ys,V,a,N,M):#计算所有买家的收入 calculate all the devices revenue
    buyerRevenues = []
    for i in range(0,M):#计算第i个买家的效益 calculate device i's revenue
        buyerRevenue = 0
        for j in range(0,N):
            buyerRevenue += (V[i] * math.log(X[j][i] - a[i] + np.e)) \
                              * (ys[j] / sum(ys))
        buyerRevenues.append(buyerRevenue)
    buyerRevenues = np.array(buyerRevenues)
    return buyerRevenues

def buyerExpensesCalculator(X,ys,N,M):#计算所有买家的付费 calculate all the devices cost/payment to providers
    buyerExpenses= []
    for i in range(0,M):#计算第i个买家的效益 device i's payment/cost
        buyerExpense = 0
        for j in range(0,N):
            buyerExpense += (X[j][i] / ys[j]) * (ys[j] / sum(ys))
        buyerExpenses.append(buyerExpense)
    buyerExpenses= np.array(buyerExpenses)
    return buyerExpenses
    
def sellerRevenuesCalculator(X,ys,N): #计算所有卖家的收入 all the providers' revenue
    sellerRevenues = []
    for j in range(0,N):
        sellerRevenue = sum(X[j]) / sum(ys)
        sellerRevenues.append(sellerRevenue)
    sellerRevenues = np.array(sellerRevenues)
    return sellerRevenues
        
def sellerExpensesCalculator(X,ys,c,N): #计算所有卖家的成本 all the providers cost
    sellerExpenses = []
    for j in range(0,N):
        sellerExpense = sum(X[j]) * c[j] * ys[j] / sum(ys)
        sellerExpenses.append(sellerExpense)
    sellerExpenses = np.array(sellerExpenses)
    return sellerExpenses