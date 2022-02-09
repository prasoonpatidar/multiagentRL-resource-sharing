'''
Input:
--priceHistory = {}, each value of the dic is a N len 1D list
--purchaseHistroy = {}, each value of the dic is a M len 1D list, X_ij
--providedResourceHistory ={}, each value of the dic is 2D array, size-M*N, Z_ij
--sellersUtilityHistory = {}, each value of the dic is a N len 1D list
--buyersUtilityHistory = {}, each value of the dic is a M len 1D list
-- time--training steps
'''

# import python libraries
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools

# import custom libraries
# from training_plot import  , getX, sliceList

train_dir = '../results/training'
eval_dir = '../results/evaluation'
sup_plot_dir = '../results/plots/10k_run3am'

market_configs = ['tightMarket','looseMarket','distMarket','monoMarket']
train_configs = ['q_r1','wolf_r1','dqn_r2','ddqn_r2','dqn_duel_r2']
env_ids = [0,1]
sliceCompare = 100
# slice = 200

label_dict = {
    'q_r1':'Q-Learning',
    'wolf_r1': 'WoLF-PHC',
    'dqn_r2':'Deep Q-Learning',
    'ddqn_r2':'Double Q-Learning',
    'dqn_duel_r2':'Dueling Networks'
}

color_label = {
    'q_r1':'r',
    'wolf_r1':'b',
    'dqn_r2':'c',
    'ddqn_r2':'m',
    'dqn_duel_r2':'g'
}

for market_config in market_configs:
    for seller_id in range(5):
        for env_id in env_ids:
            # load compare results
            compare_dir = '../results/compare'
            compare_name = f'compare_{market_config}_seller_{seller_id}_env_{env_id}'
            print(f"Plotting {compare_name}")
            compare_results = pickle.load(open(f'{compare_dir}/{compare_name}.pb','rb'))
            results_plot_dir  =f"{compare_dir}/plots_10k_5pm/{compare_name}"
            if not os.path.exists(results_plot_dir):
                os.makedirs((results_plot_dir))
            def get_performance(results):
                pricesHistory = results['price_history'] # P_ij
                purchasesHistory = results['demand_history'] # X_ij
                providedResourcesHistory = results['supply_history'] # Z_ij
                sellerUtilitiesHistory = results['seller_utilties'] # fi_j
                buyerUtilitiesHistory = results['buyer_utilties'] # fi_i
                return pricesHistory, purchasesHistory,  providedResourcesHistory, sellerUtilitiesHistory, buyerUtilitiesHistory

            class performance():
                def __init__(self, pricesHistory, purchasesHistory,  providedResourcesHistory, sellerUtilitiesHistory, buyerUtilitiesHistory):
                    self.__pricesHistory = pricesHistory
                    self.__purchasesHistory = purchasesHistory
                    self.__providedResourcesHistory = providedResourcesHistory
                    self.__sellerUtilitiesHistory = sellerUtilitiesHistory
                    self.__buyerUtilitiesHistory = buyerUtilitiesHistory
                    self.__times = len(self.buyerUtilitiesHistory)

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
                    plt.figure()
                    plt.plot(x, mean, c='r')
                    plt.fill_between(x, min, max, alpha=0.3)
                    plt.xlabel('iterations', fontsize=fontsize)
                    plt.ylabel('Resources', fontsize=fontsize)
                    plt.title(title, fontsize=fontsize)
                    plt.savefig(f'{results_plot_dir}/{title}.png', dpi=150)
                    plt.close()

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
                    plt.figure()
                    plt.plot(x, socialWelfare)
                    plt.xlabel('iterations', fontsize=fontsize)
                    plt.ylabel('Social welfare', fontsize=fontsize)
                    plt.title(title, fontsize=fontsize)
                    plt.savefig(f'{results_plot_dir}/{title}.png', dpi=150)
                    plt.close()


            def slice_params(compare_seller_id, params):
                env_params = [np.concatenate((val[:compare_seller_id],val[compare_seller_id+1:])) for val in params]
                mean = []
                for values in env_params:
                    mean.append(sum(values) / len(values))
                # list(itertools.chain.from_iterable(mean))
                t_params = [val[compare_seller_id] for val in params]
                return mean, t_params

            def get_socialWelfare(iterates, buyerUtilitiesHistory, sellerUtilitiesHistory):
                socialWelfare = []
                for t in range(iterates):
                    buyU = list(buyerUtilitiesHistory[t])
                    # list(itertools.chain.from_iterable(buyU))
                    sellerU = sellerUtilitiesHistory[t]
                    socialWelfare.append(np.sum(buyU) + np.sum(sellerU))
                return socialWelfare


            def data_slice(compare_results):
                # container for separated data
                socialWelfares = []
                prices = []
                seller_utilities = []
                labels = ['market average']
                i = -1
                for results in compare_results:
                    i += 1
                    compare_seller_id = results['compare_seller_id']
                    agent_name = results['compared_agents'][compare_seller_id]
                    labels.append(agent_name)

                    pricesHistory, purchasesHistory, \
                    providedResourcesHistory, \
                    sellerUtilitiesHistory, buyerUtilitiesHistory = get_performance(results)

                    # get the average market price, and target agent's price

                    env_price, t_price = slice_params(compare_seller_id, pricesHistory)
                    # get the average market seller utility and target agent's utility
                    env_utility, t_utility = slice_params(compare_seller_id, sellerUtilitiesHistory)

                    if i==0:
                        # only append the env parameters at the first time
                        prices.append(env_price)
                        seller_utilities.append(env_utility)
                    prices.append(t_price)
                    seller_utilities.append(t_utility)

                    iterates = len(purchasesHistory)
                    socialWelfare = get_socialWelfare(iterates, buyerUtilitiesHistory, sellerUtilitiesHistory)
                    socialWelfares.append(socialWelfare)
                return prices, seller_utilities, labels, socialWelfares

            def data_slice(compare_results):
                target_utilities = []
                target_prices = []
                env_utilities = None
                env_prices = None
                social_welfares = []
                labels = []
                for results in compare_results:
                    compare_seller_id = results['compare_seller_id']
                    agent_name = results['compared_agents'][compare_seller_id]
                    labels.append(agent_name)

                    pricesHistory, purchasesHistory, \
                    providedResourcesHistory, \
                    sellerUtilitiesHistory, buyerUtilitiesHistory = get_performance(results)

                    # get the average market price, and target agent's price

                    env_price, t_price = slice_params(compare_seller_id, pricesHistory)
                    # get the average market seller utility and target agent's utility
                    env_utility, t_utility = slice_params(compare_seller_id, sellerUtilitiesHistory)

                    target_utilities.append(t_utility)
                    target_prices.append(t_price)
                    if env_utilities is None:
                        env_utilities = np.array(env_utility)
                        env_prices = np.array(env_price)
                    else:
                        env_utilities += np.array(env_utility)
                        env_prices += np.array(env_price)

                    iterates = len(purchasesHistory)
                    socialWelfare = get_socialWelfare(iterates, buyerUtilitiesHistory, sellerUtilitiesHistory)
                    social_welfares.append(socialWelfare)

                # get average over env utilities and prices:
                env_utilities = list(env_utilities/len(compare_results))
                env_prices = list(env_prices/len(compare_results))
                labels.insert(0,'market average')
                target_utilities.insert(0,env_utilities)
                target_prices.insert(0,env_prices)
                social_welfares.insert(0,None)
                return target_prices, target_utilities, labels, social_welfares

            def get_average(segment):
                average = sum(segment)/len(segment)
                return average


            def slice_data(data, slice):
                container = []
                for iter in range(1, len(data)):
                    if iter % slice == 0:
                        ndata = data[:iter]
                        segment = ndata[-slice:]
                        container.append(get_average(segment))
                return container

            def sliceList(dataList, slice):
                container = []
                for data in dataList:
                    if data is not None:
                        dataS = slice_data(data, slice)
                        container.append(dataS)
                    else:
                        container.append(None)
                return container

            def getX(sl, l):
                x = []
                for iter in range(1, len(l)):
                    if iter % sl == 0:
                        x.append(iter)
                return x

            def plot_sellers(x, ys, labels, title, n_fig=None):
                # x = [*range(len(ys[0]))]
                # x = getX(sliceCompare, ys[0])
                plt.figure(n_fig)
                for id in range(len(ys)):
                    if ys[id] is None:
                        continue
                    if labels[id]=='market average':
                        plt.plot(x, ys[id], label='Market Average',linewidth=5,alpha=0.5,color='k')
                    else:
                        plt.plot(x, ys[id], label=label_dict[labels[id]].split("_")[0],color=color_label[labels[id]])
                plt.legend(loc="upper left")
                plt.title(title)
                if 'Prices' in title:
                    plt.ylim(20,50)
                plt.savefig(f'{results_plot_dir}/{title}.png', dpi=150)
                plt.close()


            prices, seller_utilities, labels, socialWelfares = data_slice(compare_results)
            prices =  [list(p) for p in prices]
            x = getX(sliceCompare, prices[0])
            plot_sellers(x, sliceList(prices, sliceCompare), labels, 'Seller Prices', n_fig=0)
            plot_sellers(x,sliceList(seller_utilities, sliceCompare), labels, 'Seller Utilities', n_fig=1)
            plot_sellers(x,sliceList(socialWelfares, sliceCompare), labels, 'Social Welfares', n_fig=2)