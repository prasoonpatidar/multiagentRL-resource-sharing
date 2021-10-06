#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 18:19:47 2021

@author: xuhuiying
"""

def readAndPlot(filename,variableName,performanceName,
                marker,colorName,labelName,
                lineWidth,fillStyleName,markerSize,markerEdgeWidth):#作用：读取一个csv文件，画出一条曲线
    # read csv file and plot
    data = pandas.read_csv(filename)
    x = data[variableName].values
    y = data[performanceName].values
    plt.plot(x,y,marker,label = labelName,color = colorName,
             linewidth = lineWidth,fillstyle = fillStyleName,
             markersize = markerSize,markeredgewidth = markerEdgeWidth)
    ax = plt.gca()  # 获取当前图像的坐标轴信息 get current figure's  axes information
    ax.yaxis.get_major_formatter().set_powerlimits((0,3)) # 将坐标轴的base number设置为一位。
    # set the base number for axes to one digit

def plotASinglePerformance(curveVariableName,curveVariableStart,
                           curveVariableEnd,curveVariableInterval,
                           curveVariableLabel,
                           mainVariableName,performanceName,
                           ylabelName,xlabelName):#作用：画出这个实验的某一个性能指标图
    # plot one performance index of that experiment
        
    
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
                    labelName,7,"none",10,2)
        
    #画WoLF-PHC算法的曲线 plot WoLF-PHC algorithm curve
    colorNames = ["cornflowerblue","peru","darkorchid","firebrick",
                  "darkred","hotpink","orange","darkseagreen"]
    markers = ["s-","o-","v-","d-","^-"]
    for curveVariable in np.arange(curveVariableStart,
                                   curveVariableEnd,curveVariableInterval):
        index = int((curveVariable - curveVariableStart) / curveVariableInterval)
        readAndPlot("%s=%r_varying%s_wolfphc.csv"%(curveVariableName,
                                                   curveVariable,
                                                   mainVariableName),
                    mainVariableName,performanceName,markers[index],
                    colorNames[index],
                    "RLPM,%s=%r"%(curveVariableLabel,curveVariable),
                    2,"full",5,0)
    
    #显示图片 show plot
    plt.grid()
    print("xlabelName = %s, ylabelName = %s"%(xlabelName,ylabelName))
    plt.xlabel(xlabelName,fontsize = 9)
    plt.ylabel(ylabelName,fontsize = 9)
    plt.tick_params(labelsize = 9)
    
def plotThreePerformance(curveVariableName,curveVariableStart,
                         curveVariableEnd,curveVariableInterval,
                         curveVariableLabel,
                         mainVariableName,xlabelName):
    #画三个性能指标图 plot the following three performance indicators
    plt.figure()
    performanceNames = ["meanBuyerUtility","meanSellerUtility","socialWelfare"]
    ylabelNames = ["Average utility of IoT devices",
                   "Average utility of providers",
                   "social welfare"]
    left   = [0.2,0.92,0.2,0.92]
    bottom = [1.05,1.05,0.2,0.2]
    width  = [0.6,0.6,0.6,0.6]
    height = [0.7,0.7,0.7,0.7]
    for i in range(1,4):
        plt.axes([left[i-1], bottom[i-1], width[i-1],height[i-1]])
        plotASinglePerformance(curveVariableName,curveVariableStart,
                            curveVariableEnd,curveVariableInterval,
                            curveVariableLabel,
                            mainVariableName,performanceNames[i-1],
                            ylabelNames[i-1],xlabelName)
    plt.legend(bbox_to_anchor=(1.5, 0.7), ncol=1, prop = {'size': 8},handlelength = 2)
    plt.savefig("performance_Multi_%s_varying%s.pdf"%(curveVariableName,
                                                         mainVariableName),
                bbox_inches = 'tight',dpi = 300)
    plt.show()
    
def plotFourPerformance(curveVariableName,curveVariableStart,
                         curveVariableEnd,curveVariableInterval,
                         curveVariableLabel,
                         mainVariableName,xlabelName):
    #画四个性能指标图 plot the following four performance indicators
    plt.figure()
    performanceNames = ["meanPrice","meanPurchases","meanBuyerUtility","socialWelfare"]
    ylabelNames = ["Average price","Average service demand",
                   "Average utility of IoT devices",
                   "social welfare"]
    left   = [0.2,0.92,0.2,0.92]
    bottom = [1.05,1.05,0.2,0.2]
    width  = [0.6,0.6,0.6,0.6]
    height = [0.7,0.7,0.7,0.7]
    for i in range(1,5):
        plt.axes([left[i-1], bottom[i-1], width[i-1],height[i-1]])
        plotASinglePerformance(curveVariableName,curveVariableStart,
                            curveVariableEnd,curveVariableInterval,
                            curveVariableLabel,
                            mainVariableName,performanceNames[i-1],
                            ylabelNames[i-1],xlabelName)
    plt.legend(bbox_to_anchor=(0.5, 2.4), ncol=3, prop = {'size': 8},handlelength = 2)
    plt.savefig("performance_Multi_%s_varying%s.pdf"%(curveVariableName,
                                                         mainVariableName),
                bbox_inches = 'tight',dpi = 300)
    plt.show()
        
    
def plotMN():#画出M_N的三个性能指标图 plot following indicators  for M-N
    curveVariableStart = 1
    curveVariableEnd = 4
    curveVariableInterval = 1
    plotThreePerformance("N",curveVariableStart,curveVariableEnd,
                                 curveVariableInterval,"$N$","M","Number of IoT devices")
    
def plotaV():#画出a_V的三个性能指标图 plot the following indicators for a_V
    curveVariableStart = 350
    curveVariableEnd = 501
    curveVariableInterval = 50
    plotThreePerformance("V_max",curveVariableStart,curveVariableEnd,
                                 curveVariableInterval,"$V_{max}$",
                                 "a_max","$a_{max}$")
    
def plotcV():#画出c_V的四个性能指标图 plot following indicators  for c_V
    curveVariableStart = 350
    curveVariableEnd = 501
    curveVariableInterval = 50
    plotFourPerformance("V_max",curveVariableStart,curveVariableEnd,
                                 curveVariableInterval,"$V_{max}$",
                                 "c_max","$c_{max}$")

#plotMN()
#plotaV()
plotcV()