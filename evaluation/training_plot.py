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
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools

# import custom libraries

train_dir = '../results/training'
eval_dir = '../results/evaluation'
plot_dir = '../results/plots'
name = 'test3'
market_config = "test_market"
train_config = 'wolfPHC_r2'


results = pickle.load(open(f'{eval_dir}/{market_config}_{train_config}.pb','rb'))

pricesHistory = results['price_history'] # P_ij
purchasesHistory = results['demand_history'] # X_ij
providedResourcesHistory = results['supply_history'] # Z_ij
sellerUtilitiesHistory = results['seller_utilties'] # fi_j
buyerUtilitiesHistory = results['buyer_utilties']
seller_penalty_history = results['seller_penalties']
buyer_penalty_history = results['buyer_penalties']

def get_average(segment):
    average = sum(segment)/len(segment)
    return average

def slice_data(data, slice):
    container = []
    for iter in range(1, len(data)):
        if iter % 100 == 0:
            segment = data[-slice:]
            container.append(get_average(segment))
    return container

def get_welfare(sellerUtilitiesHistory, buyerUtilitiesHistory):
    seller = [sum(val) for val in sellerUtilitiesHistory]
    buyer = [sum(val) for val in buyerUtilitiesHistory]
    sw = zip(seller, buyer)
    socialWelfare = [s+b for (s, b) in sw]
    socialLoss = [abs(s) + abs(b) for (s, b) in sw]
    return socialWelfare

def get_loss(seller_penalty_history, buyer_penalty_history):
    seller = [sum(val) for val in sellerUtilitiesHistory]
    buyer = [sum(val) for val in buyerUtilitiesHistory]
    seller = [abs(ele) for ele in seller]
    buyer = [abs(ele) for ele in buyer]
    sl = zip(seller, buyer)
    socialLoss = [s+b for (s, b) in sl]
    return socialLoss

def trading(providedResourcesHistory):
    # axis = 0, get each buyer/devices total provided resource, sum z_ij over j
    # axis = 1, get each seller/providers total provided resource, sum z_ij over i
    provided_buyer = []
    provided_seller = []
    for values in providedResourcesHistory:
        tem_buyer = values.sum(axis=0)
        temp_seller = values.sum(axis=1)
        provided_buyer.append(tem_buyer)
        provided_seller.append(temp_seller)
    return provided_seller, provided_buyer

def get_mean_min_max(data):
    # get market mean, min, and max of the variable at each step
    mean = []
    min = []
    max = []
    mean_min_max = {}
    for values in data:
        mean.append(values.mean())
        min.append(values.min())
        max.append(values.max())
    mean_min_max['mean'] = mean
    mean_min_max['max'] = max
    mean_min_max['min'] = min
    return mean_min_max

def sliceList(dataList, slice):
    container = []
    for data in dataList:
        dataS = slice_data(data, slice)
        container.append(dataS)
    return container
def getX(slice):
    x = []
    for iter in range(1, len(pricesHistory)):
        if iter % 100 == 0:
            x.append(iter)
    return x
def train_performance(slice):
    # get all the data we want to visualize
    # sellers: seller pirce-min, max, mean; seller utility; seller penalty; provided resources
    # buyers: purchase -min, max, mean; buyer utility; buyer penalty; provided resources
    # overall: social welfare, social loss

    sPrice_3m = get_mean_min_max(pricesHistory)
    bPurchase_3m = get_mean_min_max(purchasesHistory)
    provided_seller, provided_buyer = trading(providedResourcesHistory)
    provided_seller = [sum(ele) for ele in provided_seller]
    provided_buyer = [sum(ele) for ele in provided_buyer]

    socialWelfare = get_welfare(sellerUtilitiesHistory, buyerUtilitiesHistory)
    socialLoss = get_loss(seller_penalty_history, buyer_penalty_history)
    buyerU = [sum(val) for val in buyerUtilitiesHistory]
    sellerU = [sum(val) for val in sellerUtilitiesHistory]
    buyer_penalty = [sum(val) for val in buyer_penalty_history]
    seller_penalty = [sum(val) for val in sellerUtilitiesHistory]

    prices = [sPrice_3m['mean'], sPrice_3m['max'], sPrice_3m['min']]
    purchases = [bPurchase_3m['mean'], bPurchase_3m['max'], bPurchase_3m['min']]
    utilities = [buyerU, sellerU]
    penalties = [buyer_penalty, seller_penalty]
    providedResources = [provided_buyer, provided_seller]
    overall = [socialLoss, socialWelfare]

    # slice the data into given length
    pricesS = sliceList(prices, slice)
    purchasesS = sliceList(purchases, slice)
    utilitiesS = sliceList(utilities, slice)
    penaltiesS = sliceList(penalties, slice)
    providedResourcesS = sliceList(providedResources, slice)
    overallS = sliceList(overall, slice)

    # ploting
    x = getX(slice)
    plot_dir = '../results/plots'
    title = f'Average in every {slice} iterations'
    plt.subplots(figsize=(14,8))
    plt.tight_layout()
    labelsAll = ['Seller prices', 'Buyer purchases']
    labelsSB = [ 'Buyers', 'Sellers']
    labelS = ['Social welfare', 'Social loss']

    plot(x, pricesS, labelsAll[0], 'Seller prices', i=1, j=1,  fill=True)
    plot(x, purchasesS, labelsAll[1], 'Buyer purchases', i=1, j=2, fill=True)
    plot(x, utilitiesS, labelsSB, 'Utilities', i=1, j=3, fill=False)
    plt.suptitle(title)
    plt.show()
    plt.savefig(f'Figure1_{plot_dir}/{slice}_{title}.png', dpi=150)

    plt.subplots(figsize=(14, 8))
    plt.tight_layout()
    plot(x, penaltiesS, labelsSB, 'Penalties', i=1, j=1, fill=False)
    plot(x,  providedResourcesS, labelsSB, 'Provided resources', i=1, j=2, fill=False)
    plot(x, overallS, labelS, 'Social benefits', i=1, j=3, fill=False)

    plt.suptitle(title)

    plt.show()
    plt.savefig(f'Figure2_{plot_dir}/{slice}_{title}.png', dpi=150)

def plot(x, data, labels, subtitle, i=None, j=None, fill=False):
    plt.subplot(i, 3, j)
    if fill:
        plt.plot(x, data[0], c='r', label=subtitle)
        plt.fill_between(x, data[2], data[1], alpha=0.3)
    else:
        for n in range(0, len(data)):
            plt.plot(x, data[n], label = labels[n])
    plt.legend(loc='center right')
    plt.title(subtitle)
    # plt.show()
train_performance(1)