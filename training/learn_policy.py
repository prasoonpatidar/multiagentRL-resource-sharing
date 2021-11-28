'''
Wrapper function to learn policies for all types of agents
'''

import numpy as np
import time
import logging

from training.get_trainer import get_trainer
import training.buyer_utils as buyer_utils
import training.seller_utils as seller_utils

def learn_policy(run_config, seller_info, buyer_info, train_config, logger_pass, evaluate=False):
    # Get RL trainer
    rl_trainer_name = train_config.rl_trainer
    trainer = get_trainer(rl_trainer_name)

    # initialize logger
    logger_name = f'{run_config.market_config}__{run_config.train_config}'
    logger_name = f'{logger_name}__training' if not evaluate else f'{logger_name}__evaluation'
    logger_pass = dict(logger_pass)
    logger_base = logger_pass.get('logger_base').getChild(logger_name)
    logger = logging.LoggerAdapter(logger_base, logger_pass.get('logging_dict'))
    logger_pass['logger_base'] = logger_base

    # initialize agents
    sellers = trainer.initialize_agents(seller_info, buyer_info, train_config, logger, evaluate=evaluate)
    logger.info(f"Initialized {len(sellers)} seller agents...")

    # Get containers to record history
    price_history = []
    purchase_history = []
    provided_resource_history = []
    seller_utility_history = []
    seller_penalty_history = []
    buyer_utility_history = []
    buyer_penalty_history = []
    logger.info(f"Initialized containers to record history")

    # start loop for training
    env_state = trainer.get_initial_state(seller_info, buyer_info, train_config, logger)
    next_state = None

    logger.info(f"Starting training iterations")
    start_time = time.time()
    for train_iter in range(0, train_config.iterations):

        # debug info
        if train_iter % train_config.print_freq == 0:
            logger.info("Finished %d %s iterations in %.3f secs..." % (train_iter, 'evaluation' if evaluate else 'training', time.time() - start_time))

        # get next set of actions from all sellers from trainer
        actions, ys = trainer.get_actions(sellers,env_state)
        probAll, yAll = seller_utils.choose_prob(ys, compare=False, yAll=None)

        # Save prices in history
        prices = 1 / ys
        price_history.append(prices)

        # get buyer actions based on their experiences
        cumulativeBuyerExperience = buyer_utils.getBuyerExperience(sellers, buyer_info)
        X = buyer_utils.getPurchases(buyer_info, cumulativeBuyerExperience, ys, probAll)

        # Save purchased history
        purchases = X.sum(axis=0)
        purchase_history.append(purchases)

        # get buyer utilities and penalties
        buyer_utilities, buyer_penalties = buyer_utils.get_buyer_rewards(X, ys, probAll, cumulativeBuyerExperience, buyer_info)
        buyer_utility_history.append(buyer_utilities)
        buyer_penalty_history.append(buyer_penalties)

        # get next state based on current state and actions
        next_state = trainer.get_next_state(sellers, env_state, actions)

        # get seller utilities based on current state and actions
        seller_utilities, seller_penalties, distributed_resources = seller_utils.get_rewards(sellers, X, yAll)
        seller_utility_history.append(seller_utilities)
        seller_penalty_history.append(seller_penalties)
        provided_resource_history.append(distributed_resources)

        # train step for all agents
        trainer.step(sellers, train_iter, env_state, actions, next_state, X, seller_utilities, seller_penalties, distributed_resources,
                     train_config, evaluate=evaluate)

        #set env state to next state
        env_state = next_state

    # save trained policies in policy store
    if not evaluate:
        trainer.save_policies(sellers)
    logger.info(f"{'Evaluation' if evaluate else 'Training'} completed")

    results_dict = {
        'seller_policies': train_config.policy_store,
        'buyer_info': buyer_info,
        'seller_info': seller_info,
        'price_history': price_history,
        'seller_utilties': seller_utility_history,
        'seller_penalties': seller_penalty_history,
        'buyer_utilties': buyer_utility_history,
        'buyer_penalties': buyer_penalty_history,
        'demand_history': purchase_history,
        'supply_history': provided_resource_history
    }

    return results_dict
