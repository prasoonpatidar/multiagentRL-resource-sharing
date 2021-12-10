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
import os
import matplotlib.pyplot as plt
import itertools

# import custom libraries

train_dir = '../results/training'
eval_dir = '../results/evaluation'
plot_dir = '../results/plots'
name = 'test4'
market_config = "looseMarket"
train_config = 'ddqn_r2'
slice = 500

results = pickle.load(open(f'{train_dir}/{market_config}_{train_config}.pb','rb'))

plot_dir = f"{plot_dir}/{market_config}_{train_config}_{slice}"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

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
        if iter % slice == 0:
            ndata = data[:iter]
            segment = ndata[-slice:]
            container.append(get_average(segment))
    return container

def get_welfare(sellerUtilitiesHistory, buyerUtilitiesHistory):
    seller = [sum(val) for val in sellerUtilitiesHistory]
    buyer = [sum(val) for val in buyerUtilitiesHistory]
    sw = zip(seller, buyer)
    socialWelfare = [s+b for (s, b) in sw]
    return socialWelfare

def get_loss(seller_penalty_history, buyer_penalty_history):
    seller = [sum(val) for val in seller_penalty_history]
    buyer = [sum(val) for val in buyer_penalty_history]
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

def mplot(data, x, labels, suptitle, slice):
    plt.subplots(figsize=(14, 8))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    for indx in range(len(data)):
        plt.subplot(1, len(data), indx+1)
        plt.plot(x, data[indx][0], c='r', label=labels[indx])
        plt.fill_between(x, data[indx][1], data[indx][2], alpha=0.3)
        plt.legend(loc='upper left')
        plt.title(labels[indx])
        plt.xlabel('Iterations')
    plt.suptitle(suptitle, y=0.98, fontsize=16)
    plt.savefig(f'{plot_dir}/min_max_{slice}_{suptitle}.png', dpi=150)

def average_performance(slice):
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
    seller_3m_data = [sPrice_mList, sUtility , spenalty]
    ms_title = 'Seller mean, min and max purchases'
    ms_labels = ['Seller prices', 'Seller utilities', 'Seller penalties']
    mplot(seller_3m_data, x, ms_labels, ms_title , slice)

    # buyer plotting
    # min, max, mean plotting
    bPrice_mList = mplot_data(purchasesHistory, slice)
    bUtility = mplot_data(buyerUtilitiesHistory, slice)
    bpenalty = mplot_data(buyer_penalty_history, slice)
    buyer_3m_data = [bPrice_mList, bUtility,  bpenalty]
    mb_title = 'Buyer mean, min and max prices'
    mb_labels = ['Buyer purchases', 'Buyer utilities', 'Buyer penalties']
    mplot(buyer_3m_data, x, mb_labels, mb_title, slice)

    # social welfare, social loss, and provided resource plotting
    socialWelfare = get_welfare(sellerUtilitiesHistory, buyerUtilitiesHistory)
    socialLoss = get_loss(seller_penalty_history, buyer_penalty_history)
    provided_seller, provided_buyer = trading(providedResourcesHistory)
    sw_mList = mplot_data(socialWelfare, slice)
    sl_mList = mplot_data(socialLoss,slice)
    pr_mList = mplot_data(provided_buyer, slice)
    all_3m_data = [sw_mList, sl_mList, pr_mList ]
    all_title = 'Overall performance measurements-mean, min and max'
    all_labels = ['Social welfare', 'Losses', 'Provided resources']
    mplot(all_3m_data, x,  all_labels, all_title, slice)

def one_agent_plot(single_agent, slice, x, title, name):
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
            axn[ele_i].plot(x, [e[indx] for e in individual[ele_i]], label = f'{nameT}_{indx+1}')
        axn[ele_i].legend(loc='upper left')
        axn[ele_i].set_title(title[ele_i])
        axn[ele_i].set_xlabel('Iterations')
        axn[ele_i].set_ylabel('Utilities')
    plt.suptitle(name, y=0.98, fontsize=16)
    plt.savefig(f'{plot_dir}/Buyer VS Seller_{name}_{slice}.png', dpi=150)

def total_social_welfare(single_agent, slice, x, title, name):
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
            axn[ele_i].plot(x, [e[indx] for e in individual[ele_i]], label = f'{nameT}_{indx+1}')
        axn[ele_i].legend(loc='upper left')
        axn[ele_i].set_title(title[ele_i])
        axn[ele_i].set_xlabel('Iterations')
        axn[ele_i].set_ylabel('Utilities')
    plt.suptitle(name, y=0.98, fontsize=16)
    plt.savefig(f'{plot_dir}/Buyer VS Seller_{name}_{slice}.png', dpi=150)



def individual_plot(slice):


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


x = getX(slice)
provided_seller, provided_buyer = trading(providedResourcesHistory)

# average_performance(slice)
# individual_plot(slice)

plt.subplots()
purchase = [sum(ele) for ele in slice_data(purchasesHistory, slice)]
provide = [sum(ele) for ele in slice_data(provided_buyer, slice)]
plt.plot(x, provide,label = 'provided resource', c='g')
plt.plot(x, purchase,label = 'demanded resource', c='b', linestyle = '--')
plt.legend(loc='upper left')
plt.xlabel('Iterations')
plt.ylabel('Resources')
plt.title("Buyers' demanded resources VS provided resources")
plt.savefig(f'{plot_dir}/seller VS buyer_{slice}_provided VS demand.png', dpi=150)

plt.subplots()
bU = [sum(e) for e in slice_data(buyerUtilitiesHistory, slice)]
b_std = [np.std(e) for e in slice_data(buyerUtilitiesHistory, slice)]
sU = [sum(e) for e in slice_data(sellerUtilitiesHistory, slice)]
s_std = [np.std(e) for e in slice_data(sellerUtilitiesHistory, slice)]

bp = [sum(e) for e in slice_data(buyer_penalty_history, slice)]
bp_std = [np.std(e) for e in slice_data(buyer_penalty_history, slice)]
sp = [sum(e) for e in slice_data(seller_penalty_history, slice)]
sp_std = [np.std(e) for e in slice_data(seller_penalty_history, slice)]
plt.errorbar(x, bU,b_std, label='Buyer utilities', c='g')
plt.errorbar(x, bp, bp_std, label='Buyer penalties', c='g', linestyle='--')
plt.errorbar(x, sU, s_std, label='Seller utilities', c='b')
plt.errorbar(x, sp, sp_std, label='Seller penalties', c='b', linestyle='--')
plt.legend(loc='upper left')
plt.xlabel('Iterations')
plt.ylabel('Utilities')
plt.title("Utilities VS penalties")
plt.savefig(f'{plot_dir}/seller VS buyer_{slice}_Utilities VS penalties', dpi=150)

plt.subplots()
bU = np.array([sum(e) for e in slice_data(buyerUtilitiesHistory, slice)])
# b_std = [np.std(e) for e in slice_data(buyerUtilitiesHistory, slice)]
sU = np.array([sum(e) for e in slice_data(sellerUtilitiesHistory, slice)])
# s_std = [np.std(e) for e in slice_data(sellerUtilitiesHistory, slice)]
plt.plot(x, sU+bU, label='Social Welfare', c='g', marker='.')
# plt.errorbar(x, bp, bp_std, label='Buyer penalties', c='g', linestyle='--', marker='^')
# plt.errorbar(x, sU, s_std, label='Seller utilities', c='b', marker='^')
# plt.errorbar(x, sp, sp_std, label='Seller penalties', c='b', linestyle='--', marker='^')
plt.legend(loc='upper left')
plt.xlabel('Iterations')
plt.ylabel('Total Utility')
plt.title("Social Welfare")
plt.savefig(f'{plot_dir}/social_welfare_{slice}', dpi=150)


plt.subplots()
socialLoss = np.array(slice_data(get_loss(seller_penalty_history, buyer_penalty_history), slice))
# b_std = [np.std(e) for e in slice_data(buyerUtilitiesHistory, slice)]
# sU = np.array([sum(e) for e in slice_data(sellerUtilitiesHistory, slice)])
# s_std = [np.std(e) for e in slice_data(sellerUtilitiesHistory, slice)]
plt.plot(x, socialLoss, label='Total Penalties', c='r', marker='.')
# plt.errorbar(x, bp, bp_std, label='Buyer penalties', c='g', linestyle='--', marker='^')
# plt.errorbar(x, sU, s_std, label='Seller utilities', c='b', marker='^')
# plt.errorbar(x, sp, sp_std, label='Seller penalties', c='b', linestyle='--', marker='^')
plt.legend(loc='upper left')
plt.xlabel('Iterations')
plt.ylabel('Total Penalty')
plt.title("Total penalties")
plt.savefig(f'{plot_dir}/social_loss_{slice}', dpi=150)


# plot the buyer utilities and seller utilities
plt.subplots()
bsU = [sellerUtilitiesHistory, buyerUtilitiesHistory]
sub_titles = ['Seller utilities', 'Buyer utilities']
sup_title = 'Utilities'
one_agent_plot(bsU , slice, x, sub_titles, sup_title)

#plot seller prices
plt.subplots()
seller_count = len(sellerUtilitiesHistory[0])
s_price = [np.mean(e) for e in slice_data(pricesHistory, slice)]
# price_std = [np.std(e) for e in slice_data(pricesHistory, slice)]
# plt.errorbar(x,s_price, label='Seller Prices', c='g', marker='.' )
for i in range(seller_count):
    s_price_i = [e[i] for e in slice_data(pricesHistory, slice)]
    # price_std = [np.std(e) for e in slice_data(pricesHistory, slice)]
    plt.errorbar(x,s_price_i, label=f'seller {i}', marker='.' )
plt.legend(loc='upper left')
plt.xlabel('Iterations')
plt.ylabel('Prices')
plt.title("Seller prices")
plt.savefig(f'{plot_dir}/seller_{slice}_Seller prices', dpi=150)

#plot buyer demand
plt.subplots()
s_price = [np.mean(e) for e in slice_data(purchasesHistory, slice)]
# price_std = [np.std(e) for e in sli ce_data(purchasesHistory, slice)]
# plt.errorbar(x,s_price, label='Buyer demand', c='g', marker='.' )
buyer_count = len(purchasesHistory[0])
for i in range(buyer_count):
    s_purchase_i = [e[i] for e in slice_data(purchasesHistory, slice)]
    plt.errorbar(x,s_purchase_i, label=f'Buyer {i}', marker='.' )
# plt.legend()
plt.legend(loc='center left', fontsize='xx-small', ncol=2,bbox_to_anchor=(1, 0.5))

# plt.ylim(bottom=-2)
plt.xlabel('Iterations')
plt.ylabel('Buyer demand')
plt.title('Buyer demand')
# plt.tight_layout()
plt.savefig(f'{plot_dir}/buyer_{slice}_buyer demands', dpi=150, bbox_inches='tight')