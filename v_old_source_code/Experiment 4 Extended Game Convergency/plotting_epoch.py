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
experiment_name='baseline_3p5c'
iterations=50000
max_plot_points = 5000
colors = ['b','r','g']
results_dict = pickle.load(open(f'{results_dir}/{experiment_name}_it{iterations}.pb','rb'))


fig, axn = plt.subplots(1,3,figsize=(20,6))

#### Price history vs time
priceHistory_dict = results_dict['priceHistory']
priceHistory = []
for i in range(iterations):
    priceHistory.append(priceHistory_dict[i])
N = len(priceHistory[0])
epoch_length = iterations//max_plot_points
num_epochs = max_plot_points
epochProducerPricesMean = []
for epoch in range(num_epochs):
    epochProducerPricesMean.append(np.array(priceHistory[epoch_length*(epoch):epoch_length*(epoch+1)]).mean(axis=0))

for j in range(N):
    producerPrices = [xr[j] for xr in epochProducerPricesMean]
    axn[0].plot(range(0,iterations,iterations//max_plot_points), producerPrices,label=f'producer {j}', color=colors[j])
    axn[0].legend()

axn[0].set_xlabel('iterations',fontsize=24)
axn[0].set_ylabel('Prices',fontsize=24)
axn[0].set_title('Change in producer prices',fontsize=24)


### Demand and Purchases(Buyer)
demandHistory_dict = results_dict['demandHistory']
demandHistory = []
for i in range(iterations):
    demandHistory.append(demandHistory_dict[i])

epochDemandMean = []
for epoch in range(num_epochs):
    epochDemandMean.append(np.array(demandHistory[epoch_length*(epoch):epoch_length*(epoch+1)]).mean(axis=0))

purchaseHistory_dict = results_dict['purchaseHistory']
purchaseHistory = []
for i in range(iterations):
    purchaseHistory.append(purchaseHistory_dict[i])


totalBuyerPurchases = [xr.sum(axis=0) for xr in purchaseHistory]
epochBuyerPurchasesMean = []
for epoch in range(num_epochs):
    epochBuyerPurchasesMean.append(np.array(totalBuyerPurchases[epoch_length*(epoch):epoch_length*(epoch+1)]).mean(axis=0))

M = len(demandHistory[0])
# for i in range(M):
#     buyerDemands = [xr[i] for xr in demandHistory[::iterations//max_plot_points]]
#     buyerPurchases = [xr[i] for xr in totalBuyerPurchases[::iterations//max_plot_points]]
#     axn[1].plot(range(0,iterations,iterations//max_plot_points), buyerDemands,label=f'buyer {i}-Demand')
#     axn[1].plot(range(0, iterations, iterations // max_plot_points), buyerPurchases, label=f'buyer {i}-Purchase',
#                 linestyle='--')
#     axn[1].legend()

buyerDemands = [sum(xr) for xr in epochDemandMean]
buyerPurchases = [sum(xr) for xr in epochBuyerPurchasesMean]
axn[1].plot(range(0,iterations,iterations//max_plot_points), buyerDemands,label=f'total buyer demands',color='k')
axn[1].plot(range(0, iterations, iterations // max_plot_points), buyerPurchases, label=f'total producer supplies',
            color='k',linestyle='--')
axn[1].legend()
axn[1].set_xlabel('iterations',fontsize=24)
axn[1].set_ylabel('Resources',fontsize=24)
axn[1].set_title('Total Demand Vs Supply',fontsize=24)
# producer limits and supplies
totalProducerSupplies = [xr.sum(axis=1) for xr in purchaseHistory]
epochProducerSuppliesMean = []
for epoch in range(num_epochs):
    epochProducerSuppliesMean.append(np.array(totalProducerSupplies[epoch_length*(epoch):epoch_length*(epoch+1)]).mean(axis=0))



producer_limits = [12,40,80]
for j in range(N):
    producerSupplies = [xr[j]/producer_limits[j] for xr in epochProducerSuppliesMean]
    axn[2].plot(range(0,iterations,iterations//max_plot_points), producerSupplies,label=f'producer {j}(MaxResources:{producer_limits[j]})', color=colors[j])
    axn[2].legend()

axn[2].set_xlabel('iterations',fontsize=24)
axn[2].set_ylabel('Utilization Fraction',fontsize=24)
axn[2].set_title('Producer Resource Utilization Fraction',fontsize=24)

plt.savefig(f'{plot_dir}/{experiment_name}_it{iterations}_convergence.png',dpi=150)

