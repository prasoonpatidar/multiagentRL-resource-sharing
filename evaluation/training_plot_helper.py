import pickle
import numpy as np
import os
import matplotlib.pyplot as plt


def get_average(segment):
    average = sum(segment) / len(segment)
    return average


def slice_data(data, slice):
    container = []
    for iter in range(1, len(data)):
        if iter % slice == 0:
            ndata = data[:iter]
            segment = ndata[-slice:]
            container.append(get_average(segment))
    return container


def get_welfare(sellerUtilitiesHistory, buyerUtilitiesHistory):
    seller = [sum(val) for val in sellerUtilitiesHistory]
    buyer = [sum(val) for val in buyerUtilitiesHistory]
    sw = zip(seller, buyer)
    socialWelfare = [s + b for (s, b) in sw]
    return socialWelfare


def get_loss(seller_penalty_history, buyer_penalty_history):
    seller = [sum(val) for val in seller_penalty_history]
    buyer = [sum(val) for val in buyer_penalty_history]
    seller = [abs(ele) for ele in seller]
    buyer = [abs(ele) for ele in buyer]
    sl = zip(seller, buyer)
    socialLoss = [s + b for (s, b) in sl]
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


def getX(slice, pricesHistory):
    x = []
    for iter in range(1, len(pricesHistory)):
        if iter % slice == 0:
            x.append(iter)
    return x


def mplot_data(history, slice):
    # get the mean, max, and mean
    m = get_mean_min_max(history)
    # get all the data into a list
    mList = [m['mean'], m['max'], m['min']]
    # slice the data into multiple segments
    mList_s = sliceList(mList, slice)
    return mList_s


def mplot(data, x, labels, suptitle, slice, plot_dir):
    plt.subplots(figsize=(14, 8))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    for indx in range(len(data)):
        plt.subplot(1, len(data), indx + 1)
        plt.plot(x, data[indx][0], c='r', label=labels[indx])
        plt.fill_between(x, data[indx][1], data[indx][2], alpha=0.3)
        plt.legend(loc='upper left')
        plt.title(labels[indx])
        plt.xlabel('Iterations')
    plt.suptitle(suptitle, y=0.98, fontsize=16)
    plt.savefig(f'{plot_dir}/min_max_{slice}_{suptitle}.png', dpi=150)


def average_performance(slice, pricesHistory, sellerUtilitiesHistory, seller_penalty_history, purchasesHistory,
                        buyerUtilitiesHistory, buyer_penalty_history, providedResourcesHistory, plot_dir):
    # get all the data we want to visualize
    # sellers: seller pirce-min, max, mean; seller utility; seller penalty; provided resources
    # buyers: purchase -min, max, mean; buyer utility; buyer penalty; provided resources
    # overall: social welfare, social loss

    # some plotting parameters
    x = getX(slice)

    # seller plotting
    # min, max, mean plotting
    sPrice_mList = mplot_data(pricesHistory, slice)
    sUtility = mplot_data(sellerUtilitiesHistory, slice)
    spenalty = mplot_data(seller_penalty_history, slice)

    # put all the data into a list for plotting
    seller_3m_data = [sPrice_mList, sUtility, spenalty]
    ms_title = 'Seller mean, min and max purchases'
    ms_labels = ['Seller prices', 'Seller utilities', 'Seller penalties']
    mplot(seller_3m_data, x, ms_labels, ms_title, slice, plot_dir)

    # buyer plotting
    # min, max, mean plotting
    bPrice_mList = mplot_data(purchasesHistory, slice)
    bUtility = mplot_data(buyerUtilitiesHistory, slice)
    bpenalty = mplot_data(buyer_penalty_history, slice)
    buyer_3m_data = [bPrice_mList, bUtility, bpenalty]
    mb_title = 'Buyer mean, min and max prices'
    mb_labels = ['Buyer purchases', 'Buyer utilities', 'Buyer penalties']
    mplot(buyer_3m_data, x, mb_labels, mb_title, slice, plot_dir)

    # social welfare, social loss, and provided resource plotting
    socialWelfare = get_welfare(sellerUtilitiesHistory, buyerUtilitiesHistory)
    socialLoss = get_loss(seller_penalty_history, buyer_penalty_history)
    provided_seller, provided_buyer = trading(providedResourcesHistory)
    sw_mList = mplot_data(socialWelfare, slice)
    sl_mList = mplot_data(socialLoss, slice)
    pr_mList = mplot_data(provided_buyer, slice)
    all_3m_data = [sw_mList, sl_mList, pr_mList]
    all_title = 'Overall performance measurements-mean, min and max'
    all_labels = ['Social welfare', 'Losses', 'Provided resources']
    mplot(all_3m_data, x, all_labels, all_title, slice, plot_dir)


def one_agent_plot(single_agent, slice, x, title, name, plot_dir):
    individual = sliceList(single_agent, slice)
    fig, axn = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    for ele_i in range(len(individual)):
        if ele_i < 1:
            nameT = 'Seller'
        else:
            nameT = 'Buyer'
        for indx in range(len(individual[ele_i][0])):
            axn[ele_i].plot(x, [e[indx] for e in individual[ele_i]], label=f'{nameT}_{indx + 1}')
        axn[ele_i].legend(loc='upper left')
        axn[ele_i].set_title(title[ele_i])
        axn[ele_i].set_xlabel('Iterations')
        axn[ele_i].set_ylabel('Utilities')
    plt.suptitle(name, y=0.98, fontsize=16)
    plt.savefig(f'{plot_dir}/Buyer VS Seller_{name}_{slice}.png', dpi=150)


def total_social_welfare(single_agent, slice, x, title, name, plot_dir):
    individual = sliceList(single_agent, slice)
    fig, axn = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    for ele_i in range(len(individual)):
        if ele_i < 1:
            nameT = 'Seller'
        else:
            nameT = 'Buyer'
        for indx in range(len(individual[ele_i][0])):
            axn[ele_i].plot(x, [e[indx] for e in individual[ele_i]], label=f'{nameT}_{indx + 1}')
        axn[ele_i].legend(loc='upper left')
        axn[ele_i].set_title(title[ele_i])
        axn[ele_i].set_xlabel('Iterations')
        axn[ele_i].set_ylabel('Utilities')
    plt.suptitle(name, y=0.98, fontsize=16)
    plt.savefig(f'{plot_dir}/Buyer VS Seller_{name}_{slice}.png', dpi=150)


def individual_plot(slice, x, pricesHistory, sellerUtilitiesHistory, purchasesHistory,
                        buyerUtilitiesHistory, providedResourcesHistory):
    provided_seller, provided_buyer = trading(providedResourcesHistory)
    # seller plot
    one_seller = [pricesHistory, sellerUtilitiesHistory, purchasesHistory, buyerUtilitiesHistory]
    title = ['Seller prices', 'Seller utilities', 'Buyer demand', 'Buyer utilities']
    names = 'Seller'
    one_agent_plot(one_seller, slice, x, title, names)

    # # buyer plot
    # one_buyer = [purchasesHistory, buyerUtilitiesHistory, buyer_penalty_history, provided_buyer]
    # titleb = ['Buyer demand', 'Buyer utilities', 'Buyer penalties', 'Buyer provided resources']
    # nameb = 'Buyer'
    # one_agent_plot(one_buyer, slice, x, titleb, nameb)
    #
    # # the buyer purchase and provided plot
    # purchase = slice_data(purchasesHistory, slice)
    # provide = slice_data(provided_buyer, slice)
    #
    #
    #
    # fig, axn = plt.subplots(1, 3, figsize=(14, 8))
    # def comp_plot(fig, y1, y2, title, ny1, ny2):
    #     for i in range(len(y1[0])):
    #         py = [val[i] for val in y1]
    #         axn[fig].plot(x, py, label = f'{ny1} {i}')
    #         if i < len(y2[0]):
    #             pr = [val[i] for val in y2]
    #             axn[fig].plot(x, pr, label =f' {ny2} {i}', linestyle = '--')
    #         axn[fig].legend()
    #     axn[fig].set_xlabel('iterations', fontsize=12)
    #     axn[fig].set_ylabel('Computation resources', fontsize=12)
    #     axn[fig].set_title(title, fontsize=12)
    #
    # comp_plot(0, purchase, provide, 'Buyer purchases VS buyer provided', nameb, nameb)
    # comp_plot(1, bU, sU, 'Buyer utilities VS seller utilities', nameb, names)
    #
    # plt.savefig(f'{plot_dir}/seller VS buyer_{slice}_{name}.png', dpi=150)
