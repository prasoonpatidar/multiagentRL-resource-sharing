'''
This is the main python wrapper to compare the learned policies from different reinforcement learning algorithms
in multi seller, multi buyer resource sharing environment
'''

# import python libraries
import json
import os, sys
import logging
from logging.handlers import WatchedFileHandler
from types import SimpleNamespace
import pickle

# import custom libraries
from evaluation.head2Head_comparasion import run_comparison


if __name__ == '__main__':
    # set a comparison config
    compare_config = {
        'name': 'compare1',
        'market_config': "test_market",
        'log_dir': 'logs/',
        'run_config_name':
    }

    # get config named tuples
    compare_config = SimpleNamespace(**compare_config)
    market_config = json.load(open(f"configs/market/{compare_config.market_config}.json"))
    seller_info = SimpleNamespace(**market_config["seller"])
    buyer_info = SimpleNamespace(**market_config["buyer"])

    # Initialize Logger
    logger_master = logging.getLogger(compare_config.name)
    logger_master.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    core_logging_handler = WatchedFileHandler(compare_config.log_dir + '/' + compare_config.name + '.log')
    core_logging_handler.setFormatter(formatter)
    logger_master.addHandler(core_logging_handler)
    console_log = logging.StreamHandler()
    console_log.setLevel(logging.DEBUG)
    console_log.setFormatter(formatter)
    logger_master.addHandler(console_log)
    logger = logging.LoggerAdapter(logger_master, {})
    logger_pass = {'logger_base': logger_master}

    # Get results file name and appropriate training suite
    results_file = f'results/training/{run_config.name}_{run_config.market_config}_{run_config.train_config}.pb'

    compare_results = run_comparison(seller_info, buyer_info, results_dir, logger_pass)

