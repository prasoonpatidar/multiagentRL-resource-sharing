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
from evaluation.head2Head_comparision import run_comparison


if __name__ == '__main__':
    # set a comparison config
    market_configs = ['tightMarket', 'looseMarket', 'distMarket', 'monoMarket']
    random_env_count = 2
    for comp_market_name in market_configs:
        for seller_id in range(5):
            for env_id in range(random_env_count):
                compare_config = {
                    'name': f'compare_{comp_market_name}_seller_{seller_id}_env_{env_id}',
                    'market_config': comp_market_name,
                    'log_dir': 'logs/',
                    'iterations': 1000,
                    'seller_id':seller_id,
                    'env_id':env_id
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
                logger.info(f"Running comparison for {comp_market_name}, for seller {seller_id}, random env: {env_id}")
                market_name = compare_config.market_config
                compare_results = run_comparison(compare_config.seller_id, seller_info, buyer_info, logger_pass, market_name, compare_config.iterations)

                compare_file = f'results/compare/{compare_config.name}.pb'
                pickle.dump( compare_results, open(compare_file, 'wb'))

                logger.info("Evaluation Finished...")

