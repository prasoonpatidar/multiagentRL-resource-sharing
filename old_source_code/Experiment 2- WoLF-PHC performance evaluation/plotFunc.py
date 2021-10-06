#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei']

def readAndPlot(filename,variableName,performanceName,
                marker,colorName,labelName,
                lineWidth,fillStyleName,markerSize,markerEdgeWidth):#作用：读取一个csv文件，画出一条曲线
    data = pandas.read_csv(filename)
    x = data[variableName].values
    y = data[performanceName].values
    plt.plot(x,y,marker,label = labelName,color = colorName,
             linewidth = lineWidth,fillstyle = fillStyleName,
             markersize = markerSize,markeredgewidth = markerEdgeWidth)
#    ax = plt.gca()  # 获取当前图像的坐标轴信息 get current figure's  axes information
#    ax.yaxis.get_major_formatter().set_powerlimits((0,3)) # 将坐标轴的base number设置为一位。
# ax.yaxis.get_major_formatter().set_powerlimits((0,3)) set the base number for axes to one digit

def plotASinglePerformance(curveVariableName,curveVariableStart,
                           curveVariableEnd,curveVariableInterval,
                           curveVariableLabel,
                           mainVariableName,performanceName,
                           ylabelName,xlabelName):#作用：画出这个实验的某一个性能指标图
                         # plot one performance index of that experiment
    
#    plt.figure()
    
    print("huiying")
    for curveVariable in np.arange(curveVariableStart,
                                   curveVariableEnd,curveVariableInterval):
        index = int((curveVariable - curveVariableStart) / curveVariableInterval)
        if index == 0:
            labelName = "NE"
        else:
            labelName = None
        readAndPlot("%s=%r_varying%s_gambit.csv"%(curveVariableName,
                                                   curveVariable,
                                                   mainVariableName),
                    mainVariableName,performanceName,"o","lightslategray",
                    labelName,7,"none",15,3)
        
    #画WoLF-PHC算法的曲线 plot WoLF-PHC algorithm curve
#    colorNames = ["cornflowerblue","peru","darkorchid","firebrick",
#                  "darkred","hotpink","orange","darkseagreen"]
#    markers = ["s-","o-","v-","d-","^-"]
#    for curveVariable in np.arange(curveVariableStart,
#                                   curveVariableEnd,curveVariableInterval):
#        index = int((curveVariable - curveVariableStart) / curveVariableInterval)
#        readAndPlot("%s=%r_varying%s_wolfphc.csv"%(curveVariableName,
#                                                   curveVariable,
#                                                   mainVariableName),
#                    mainVariableName,performanceName,markers[index],
#                    colorNames[index],
#                    "RLPM,%s=%r"%(curveVariableLabel,curveVariable),
#                    2,"full",8,0)
    
    #显示图片
#    plt.legend(prop = {'size': 14},handlelength = 1)
#    plt.grid()
#    plt.xlabel(xlabelName,fontsize = 14)
#    plt.ylabel(ylabelName,fontsize = 14)
#    plt.tick_params(labelsize=13)
#    plt.show()

def plotAllSevenPerformance(curveVariableName,curveVariableStart,
                            curveVariableEnd,curveVariableInterval,
                            curveVariableLabel,
                            mainVariableName,xlabelName):
    #作用：画出这个实验所对应的七个性能指标图 plot the performance matrix
    print("huiying----")
    
    #画七个性能指标图 plot the performance matrix
    performanceNames = ["socialWelfare","meanSellerUtility","meanBuyerUtility",
                        "meanPrice","meanPurchases",
                        "meanSellerRevenue","meanBuyerRevenue"]
    ylabelNames = ["socialWelfare",
                   "Average utility of providers",
                   "Average utility of IoT devices",
                   "Average price","Average service demand",
                   "Average revenue of sellers","Average revenue of buyers"]
    for i in range(0,7):
        plotASinglePerformance(curveVariableName,curveVariableStart,
                            curveVariableEnd,curveVariableInterval,
                            curveVariableLabel,
                            mainVariableName,performanceNames[i],
                            ylabelNames[i],xlabelName)
        