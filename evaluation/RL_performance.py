'''
Input:
--priceHistory = {}, each value of the dic is a M len 1D list
--purchaseHistroy = {}, each value of the dic is a M len 1D list
--providedResourceHistory ={}, each value of the dic is 2D array, size-M*N
--sellersUtilityHistory = {}, each value of the dic is a N len 1D list
--buyersUtilityHistory = {}, each value of the dic is a M len 1D list
-- time--training steps
'''

# import python libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt

# import custom libraries
from configs.train_configs import get_train_config

train_dir = '../results/training'
eval_dir = '../results/evaluation'
plot_dir = '../results/plots'
name = 'test3'
market_config = "test_market"
train_config = "q_r1"

results = pickle.load(open(f'{eval_dir}/{name}_{market_config}_{train_config}.pb','rb'))

pricesHistory = results['price_history'] # P_ij
purchasesHistory = results['demand_history'] # X_ij
providedResourcesHistory = results['supply_history'] # Z_ij
sellerUtilitiesHistory = results['seller_utilties'] # fi_j
buyerUtilitiesHistory = results['buyer_utilties'] # fi_i


class performance:
    def __init__(self, results):
        self.__pricesHistory = results['price_history']
        self.__purchasesHistory = results['demand_history']
        self.__providedResourcesHistory = results['supply_history']
        self.__sellerUtilitiesHistory =  results['seller_utilties']
        self.__buyerUtilitiesHistory = results['buyer_utilties']
        self.__times = len(buyerUtilitiesHistory)

    def trading(self):
        # axis = 0, get each buyer/devices total provided resource, sum z_ij over j
        # axis = 1, get each seller/providers total provided resource, sum z_ij over i
        provided_buyer = []
        provided_seller = []
        for values in self.__providedResourcesHistory.values():
            tem_buyer = values.sum(axis=0)
            temp_seller = values.sum(axis=1)
            provided_buyer.append(tem_buyer)
            provided_seller.append(temp_seller)
        return provided_seller, provided_buyer

    def get_mean_min_max(self, data, varName):
        # get market mean, min, and max of the variable at each step
        mean = []
        min = []
        max = []
        mean_min_max = {}
        if isinstance(data, dict):
            for values in data.values():
                mean.append(values.mean())
                min.append(values.min())
                max.append(values.max())
        else:
            for values in data:
                mean.append(values.mean())
                min.append(values.min())
                max.append(values.max())
        mean_min_max[varName] = varName
        mean_min_max['mean'] = mean
        mean_min_max['max'] = max
        mean_min_max['min'] = min
        return mean_min_max

    def plot_mean_min_max(self, data, title, num=None):
        # num --is the latest number of points to be plotted
        fontsize = 12
        if num == None:
            mean = data['mean']
            max = data['max']
            min = data['min']
        else:
            mean = data['mean'][-num:]
            max = data['max'][-num:]
            min = data['min'][-num:]
        x = [*range(len(mean))]
        plt.plot(x, mean, c='r')
        plt.fill_between(x, min, max, alpha=0.3)
        plt.xlabel('iterations', fontsize=fontsize)
        plt.ylabel('Resources', fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
        plt.savefig(f'{plot_dir}/{name}_{market_config}_{train_config}_{title}.png', dpi=150)

    def provided_buyer_plot(self, num=None):
        provided_seller, provided_buyer = self.trading()
        data = self.get_mean_min_max(provided_buyer, 'provided_buyers')
        self.plot_mean_min_max(data, 'provided_buyers', num=num)

    def provided_seller_plot(self, num=None):
        provided_seller, provided_buyer = self.trading()
        data = self.get_mean_min_max(provided_seller, 'provided_seller')
        self.plot_mean_min_max(data, 'provided_seller', num=num)

    def priceHistroy_plot(self, num=None):
        name = 'prices History'
        data = self.get_mean_min_max(self.__pricesHistory, name)
        self.plot_mean_min_max(data, name, num=num)

    def purchaseHistory_plot(self, num=None):
        name = ' purchase History'
        data = self.get_mean_min_max(self.__purchasesHistory, name)
        self.plot_mean_min_max(data, name, num=num)

    def sellerUtilitiesHistory_plot(self, num=None):
        name = ' seller Utilities History '
        data = self.get_mean_min_max(self.__purchasesHistory, name)
        self.plot_mean_min_max(data, name, num=num)

    def buyerUtilitiesHistory_plot(self, num=None):
        name = ' buyer Utilities History '
        data = self.get_mean_min_max(self.__purchasesHistory, name)
        self.plot_mean_min_max(data, name, num=num)

    def socialWelfare_plot(self, num=None):
        socialWelfare = []
        for t in range(self.__times):
            buyerUtility = self.__buyerUtilitiesHistory[t]
            sellerUtility = self.__sellerUtilitiesHistory[t]
            socialWelfare.append(sum(buyerUtility) + sum(sellerUtility))
        fontsize = 12
        title = 'socialWelfare'
        x = [*range(self.__times)]
        plt.plot(x, socialWelfare)
        plt.xlabel('iterations', fontsize=fontsize)
        plt.ylabel('Social welfare', fontsize=fontsize)
        plt.title(title, fontsize=fontsize)
        plt.savefig(f'{plot_dir}/{name}_{market_config}_{train_config}_{title}.png', dpi=150)


evaluation = performance(results)
evaluation.buyerUtilitiesHistory_plot(num=200)
