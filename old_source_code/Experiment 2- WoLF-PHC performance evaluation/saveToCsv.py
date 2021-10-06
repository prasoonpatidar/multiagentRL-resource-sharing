#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas

def saveToCsv(variableName,changeSeq,resultMeanRecord,filename):
    dic = {variableName:changeSeq,
           "socialWelfare":resultMeanRecord[0],
           "meanSellerUtility":resultMeanRecord[1],
           "meanBuyerUtility":resultMeanRecord[2],
           "meanPrice":resultMeanRecord[3],
           "meanPurchases":resultMeanRecord[4],
           "meanSellerRevenue":resultMeanRecord[5],
           "meanBuyerRevenue":resultMeanRecord[6]}
    data = pandas.DataFrame(dic,
                            columns = [variableName,"socialWelfare",
                                       "meanSellerUtility","meanBuyerUtility",
                                       "meanPrice","meanPurchases",
                                       "meanSellerRevenue","meanBuyerRevenue"])
    data.to_csv(filename,index = False)
    