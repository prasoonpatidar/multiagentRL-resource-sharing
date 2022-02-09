# import python libraries
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools

# import custom libraries
from evaluation.training_plot_helper import *

train_dir = '../results/training'
eval_dir = '../results/evaluation'
sup_plot_dir = '../results/plots/10k_5pm'
name = 'test4'
market_configs = ['tightMarket','looseMarket','distMarket','monoMarket']
train_configs = ['q_r1','wolf_r1','dqn_r2','ddqn_r2','dqn_duel_r2']
slice = 1999

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
    print(f"Running for market {market_config}")
    market_welfares = []
    for train_config in train_configs:
        print(f"Plotting for market {market_config}, trainer {train_config}")
        results = pickle.load(open(f'{train_dir}/{market_config}_{train_config}.pb','rb'))

        plot_dir = f"{sup_plot_dir}/penalty_plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        sellerPenaltiesHistory = results['seller_penalties'] # fi_j
        buyerPenaltiesHistory = results['buyer_penalties']

        x = getX(slice, sellerPenaltiesHistory)

        bP = np.array([sum(e) for e in slice_data(buyerPenaltiesHistory, slice)])
        sP = np.array([sum(e) for e in slice_data(sellerPenaltiesHistory, slice)])

        total_penalty = sP + bP
        market_welfares.append((train_config,x,total_penalty))

    plt.figure()
    for i in range(len(market_welfares)):
        label, x,y = market_welfares[i]
        plt.plot(x[:5],y[:5],'--',label=label_dict[label], color=color_label[label])
        # plt.xlim(right = 10000)
    plt.legend(loc='upper left')
    plt.xlabel('Iterations')
    plt.ylabel('Total Penalty')
    plt.savefig(f'{plot_dir}/{market_config}_{slice}.png', dpi=150)
    plt.close()
