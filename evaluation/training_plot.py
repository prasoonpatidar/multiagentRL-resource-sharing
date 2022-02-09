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
# import seaborn as sns

# import custom libraries
from evaluation.training_plot_helper import *

train_dir = '../results/training'
eval_dir = '../results/evaluation'
sup_plot_dir = '../results/plots/10k_5pm/trading'
name = 'test4'
market_configs = ['tightMarket','looseMarket','distMarket','monoMarket']
train_configs = ['dqn_duel_r2']
slice = 1999


for market_config in market_configs:
    for train_config in train_configs:
# market_config = "tightMarket"
# train_config = 'dqn_r2'
        print(f"Plotting for market {market_config}, trainer {train_config}")
        results = pickle.load(open(f'{train_dir}/{market_config}_{train_config}.pb','rb'))

        plot_dir = f"{sup_plot_dir}/{market_config}_{train_config}_{slice}"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        pricesHistory = results['price_history'] # P_ij
        purchasesHistory = results['demand_history'] # X_ij
        providedResourcesHistory = results['supply_history'] # Z_ij
        sellerUtilitiesHistory = results['seller_utilties'] # fi_j
        buyerUtilitiesHistory = results['buyer_utilties']
        seller_penalty_history = results['seller_penalties']
        buyer_penalty_history = results['buyer_penalties']


        x = getX(slice, pricesHistory)
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
        plt.ylim(100,400)
        plt.title("Buyers' demanded resources VS provided resources")
        plt.savefig(f'{plot_dir}/seller VS buyer_{slice}_provided VS demand.png', dpi=150)
        plt.close()

        # plt.subplots()
        # bU = [sum(e) for e in slice_data(buyerUtilitiesHistory, slice)]
        # b_std = [np.std(e) for e in slice_data(buyerUtilitiesHistory, slice)]
        # sU = [sum(e) for e in slice_data(sellerUtilitiesHistory, slice)]
        # s_std = [np.std(e) for e in slice_data(sellerUtilitiesHistory, slice)]
        #
        # bp = [sum(e) for e in slice_data(buyer_penalty_history, slice)]
        # bp_std = [np.std(e) for e in slice_data(buyer_penalty_history, slice)]
        # sp = [sum(e) for e in slice_data(seller_penalty_history, slice)]
        # sp_std = [np.std(e) for e in slice_data(seller_penalty_history, slice)]
        # plt.errorbar(x, bU,b_std, label='Buyer utilities', c='g')
        # plt.errorbar(x, bp, bp_std, label='Buyer penalties', c='g', linestyle='--')
        # plt.errorbar(x, sU, s_std, label='Seller utilities', c='b')
        # plt.errorbar(x, sp, sp_std, label='Seller penalties', c='b', linestyle='--')
        # plt.legend(loc='upper left')
        # plt.xlabel('Iterations')
        # plt.ylabel('Utilities')
        # plt.title("Utilities VS penalties")
        # plt.savefig(f'{plot_dir}/seller VS buyer_{slice}_Utilities VS penalties', dpi=150)
        # plt.close()

        # plt.subplots()
        # bU = np.array([sum(e) for e in slice_data(buyerUtilitiesHistory, slice)])
        # # b_std = [np.std(e) for e in slice_data(buyerUtilitiesHistory, slice)]
        # sU = np.array([sum(e) for e in slice_data(sellerUtilitiesHistory, slice)])
        # # s_std = [np.std(e) for e in slice_data(sellerUtilitiesHistory, slice)]
        # plt.plot(x, sU+bU, label='Social Welfare', c='g', marker='.')
        # # plt.errorbar(x, bp, bp_std, label='Buyer penalties', c='g', linestyle='--', marker='^')
        # # plt.errorbar(x, sU, s_std, label='Seller utilities', c='b', marker='^')
        # # plt.errorbar(x, sp, sp_std, label='Seller penalties', c='b', linestyle='--', marker='^')
        # plt.legend(loc='upper left')
        # plt.xlabel('Iterations')
        # plt.ylabel('Total Utility')
        # plt.title("Social Welfare")
        # plt.savefig(f'{plot_dir}/social_welfare_{slice}', dpi=150)
        # plt.close()


        # plt.subplots()
        # socialLoss = np.array(slice_data(get_loss(seller_penalty_history, buyer_penalty_history), slice))
        # # b_std = [np.std(e) for e in slice_data(buyerUtilitiesHistory, slice)]
        # # sU = np.array([sum(e) for e in slice_data(sellerUtilitiesHistory, slice)])
        # # s_std = [np.std(e) for e in slice_data(sellerUtilitiesHistory, slice)]
        # plt.plot(x, socialLoss, label='Total Penalties', c='r', marker='.')
        # # plt.errorbar(x, bp, bp_std, label='Buyer penalties', c='g', linestyle='--', marker='^')
        # # plt.errorbar(x, sU, s_std, label='Seller utilities', c='b', marker='^')
        # # plt.errorbar(x, sp, sp_std, label='Seller penalties', c='b', linestyle='--', marker='^')
        # plt.legend(loc='upper left')
        # plt.xlabel('Iterations')
        # plt.ylabel('Total Penalty')
        # plt.title("Total penalties")
        # plt.savefig(f'{plot_dir}/social_loss_{slice}', dpi=150)
        # plt.close()


        # plot the buyer utilities and seller utilities
        # plt.subplots()
        # bsU = [sellerUtilitiesHistory, buyerUtilitiesHistory]
        # sub_titles = ['Seller utilities', 'Buyer utilities']
        # sup_title = 'Utilities'
        # one_agent_plot(bsU , slice, x, sub_titles, sup_title, plot_dir)

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
        plt.ylim(20,50)
        plt.title("Seller prices")
        plt.savefig(f'{plot_dir}/seller_{slice}_Seller prices', dpi=150)
        plt.close()

        #plot buyer demand
        plt.subplots()
        plt.boxplot(slice_data(purchasesHistory, slice),whis=0.5)
        s_price = [np.mean(e) for e in slice_data(purchasesHistory, slice)]
        # price_std = [np.std(e) for e in slice_data(purchasesHistory, slice)]
        #
        # plt.errorbar(x,s_price,price_std, label='Buyer demand', c='g', marker='.' )
        # buyer_count = len(purchasesHistory[0])
        # for i in range(buyer_count):
        #     s_purchase_i = [e[i] for e in slice_data(purchasesHistory, slice)]
        #     plt.errorbar(x,s_purchase_i, label=f'Buyer {i}', marker='.' )
        # plt.legend()
        # plt.legend(loc='center left', fontsize='xx-small', ncol=2,bbox_to_anchor=(1, 0.5))

        # plt.ylim(bottom=-2)
        plt.xticks(range(6),range(0,10001,2000))
        plt.ylim(0,15)
        plt.xlabel('Iterations')
        plt.ylabel('Market demand distribution')
        # plt.title('Buyer Demand Distribution')
        # plt.tight_layout()
        plt.savefig(f'{plot_dir}/market_demand_distribution', dpi=150, bbox_inches='tight')
        plt.close()