'''
Create plots for experiments
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
sns.set('paper')

results_dir = 'results'
plot_dir = 'plots'
experiment_name='lam5_sig0.5'
iterations=5000
max_plot_points = 50
colors = ['b','r','g']
results_dict = pickle.load(open(f'{results_dir}/{experiment_name}_it{iterations}.pb','rb'))


fig, axn = plt.subplots(1,3,figsize=(20,6))

#### Price history vs time
priceHistory = results_dict['priceHistory']
N = len(priceHistory[0])
for j in range(N):
    producerPrices = [xr[j] for xr in priceHistory[::iterations//max_plot_points]]
    axn[0].plot(range(0,iterations,iterations//max_plot_points), producerPrices,label=f'producer {j}', color=colors[j])
    axn[0].legend()

axn[0].set_xlabel('iterations',fontsize=24)
axn[0].set_ylabel('Prices',fontsize=24)
axn[0].set_title('Change in producer prices',fontsize=24)


### Demand and Purchases(Buyer)
demandHistory = results_dict['demandHistory']
purchaseHistory = results_dict['purchaseHistory']
totalBuyerPurchases = [xr.sum(axis=0) for xr in purchaseHistory]

M = len(demandHistory[0])
# for i in range(M):
#     buyerDemands = [xr[i] for xr in demandHistory[::iterations//max_plot_points]]
#     buyerPurchases = [xr[i] for xr in totalBuyerPurchases[::iterations//max_plot_points]]
#     axn[1].plot(range(0,iterations,iterations//max_plot_points), buyerDemands,label=f'buyer {i}-Demand')
#     axn[1].plot(range(0, iterations, iterations // max_plot_points), buyerPurchases, label=f'buyer {i}-Purchase',
#                 linestyle='--')
#     axn[1].legend()

buyerDemands = [sum(xr) for xr in demandHistory[::iterations//max_plot_points]]
buyerPurchases = [sum(xr) for xr in totalBuyerPurchases[::iterations//max_plot_points]]
axn[1].plot(range(0,iterations,iterations//max_plot_points), buyerDemands,label=f'total buyer demands',color='k')
axn[1].plot(range(0, iterations, iterations // max_plot_points), buyerPurchases, label=f'total producer supplies',
            color='k',linestyle='--')
axn[1].legend()
axn[1].set_xlabel('iterations',fontsize=24)
axn[1].set_ylabel('Resources',fontsize=24)
axn[1].set_title('Total Demand Vs Supply',fontsize=24)
# producer limits and supplies
totalProducerSupplies = [xr.sum(axis=1) for xr in purchaseHistory]
producer_limits = [12,40,80]
for j in range(N):
    producerSupplies = [xr[j]/producer_limits[j] for xr in totalProducerSupplies[::iterations//max_plot_points]]
    axn[2].plot(range(0,iterations,iterations//max_plot_points), producerSupplies,label=f'producer {j}(MaxResources:{producer_limits[j]})', color=colors[j])
    axn[2].legend()

axn[2].set_xlabel('iterations',fontsize=24)
axn[2].set_ylabel('Utilization Fraction',fontsize=24)
axn[2].set_title('Producer Resource Utilization Fraction',fontsize=24)

plt.savefig(f'{plot_dir}/{experiment_name}_convergence.png',dpi=150)

