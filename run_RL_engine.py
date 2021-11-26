'''
This is the main python wrapper to run reinforcement learning algorithms for multi seller, multi buyer resource sharing environment
'''

# import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os, sys
import logging
from logging.handlers import WatchedFileHandler
from types import SimpleNamespace
import pickle

# import custom libraries
from configs.train_configs import get_train_config
from training.get_trainer import get_trainer

if __name__ == '__main__':

    # set run config
    run_config = {
        'name': 'test4',
        'market_config': "test_market",
        'train_config':  'wolfPHC_r2',
        'results_dir': 'results/',
        'log_dir': 'logs/',
    }

    # get config named tuples
    run_config = SimpleNamespace(**run_config)
    train_config = get_train_config(run_config.train_config)
    market_config = json.load(open(f"configs/market/{run_config.market_config}.json"))
    seller_info = SimpleNamespace(**market_config["seller"])
    buyer_info = SimpleNamespace(**market_config["buyer"])

    # Initialize Logger
    logger_master = logging.getLogger(run_config.name)
    logger_master.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    core_logging_handler = WatchedFileHandler(run_config.log_dir + '/' + run_config.name + '.log')
    core_logging_handler.setFormatter(formatter)
    logger_master.addHandler(core_logging_handler)
    console_log = logging.StreamHandler()
    console_log.setLevel(logging.DEBUG)
    console_log.setFormatter(formatter)
    logger_master.addHandler(console_log)
    logger = logging.LoggerAdapter(logger_master, {})
    logger_pass = {'logger_base':logger_master}

    # Get results file name and appropriate training suite
    results_file = f'results/training/{run_config.market_config}_{run_config.train_config}.pb'
    trainer = get_trainer(train_config)

    # Training the policy
    if train_config.train:

        # learn a new policy
        results_dict = trainer.learn_policy(run_config, seller_info, buyer_info, train_config, logger_pass)

        # save results
        if train_config.store_results:
            pickle.dump(results_dict, open(results_file, 'wb'))
            # Add code for plotting results as well

        # show results
        if train_config.show_results:  # add code to present results in console
            pass

        logger.info("Training Finished...")

    # Evaluating the trained policy
    if train_config.evaluate:

        # Load a policy
        if os.path.exists(results_file):
            results_dir = pickle.load(open(results_file, 'rb'))
        else:
            logger.error("policy file not present, exiting")
            exit(1)

        # Evaluate the policy
        eval_dict = trainer.eval_policy(seller_info, buyer_info, train_config, results_dir, logger_pass)

        # save evaluations
        if train_config.store_results:
            eval_file = f'results/evaluation/{run_config.market_config}_{run_config.train_config}.pb'
            pickle.dump(eval_dict, open(eval_file, 'wb'))
            # Add code for plotting evaluation results

        # show evaluation results
        if train_config.show_results:  # add code to present evaluation results in console
            pass

        logger.info("Evaluation Finished...")
