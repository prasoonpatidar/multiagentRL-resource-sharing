# import python libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt

compare_dir = '../results/compare'
plot_dir = '../results/plots'
compare_name = 'compare1'
market_name = 'test_market'

compare_results = pickle.load(open(f'{compare_dir}/{compare_name}_{market_name}.pb','rb'))

