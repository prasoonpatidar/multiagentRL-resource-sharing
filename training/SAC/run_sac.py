'''
Main wrapper function to train and evaluate SAC algorithm
'''

import numpy as np
import time

# custom libraries
from training.SAC.run_helper import buyerPenaltiesCalculator, buyerUtilitiesCalculator, action2y, ydiff2action
from training.SAC.run_helper import logger_handle, initialize_agent, get_ys, choose_prob, cumlativeBuyerExp, \
    getPurchases


def learn_policy(run_config, seller_info, buyer_info, train_config, logger_pass):
    # Initialize the logger
    logger = logger_handle(logger_pass)
    # get required parameters for WolFPHC algorithm
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    logger.info("Fetched raw market information..")

    # initialize seller agents
    sellers, logger = initialize_agent(seller_info, buyer_info, train_config, logger)

    # Get Containers to record history(Interesting insight: append in python list is O(1))
    price_history = []
    purchase_history = []
    provided_resource_history = []
    seller_utility_history = []
    seller_penalty_history = []
    buyer_utility_history = []
    buyer_penalty_history = []

    # Start Loop for training
    logger.info("Starting training iterations...")
    start_time = time.time()
    env_state = np.random.randint(0, train_config.action_count, seller_info.count)
    next_state = np.random.randint(0, train_config.action_count, seller_info.count)

    for train_iter in range(0, train_config.iterations):

        if train_iter % 10 == 0:
            logger.info("Finished %d training iterations in %.3f secs..." % (train_iter, time.time() - start_time))

        # get the prices for all seller agents
        ydiffActions = []
        for tmpSeller in sellers:
            ydiffActions.append(tmpSeller.policy_net.get_action(env_state, deterministic=train_config.deterministic))
        ydiffActions = np.array(ydiffActions).flatten()
        ys = aux_price_min + ydiffActions
        probAll, yAll = choose_prob(ys, compare=False, yAll=None)

        # Take step in environment: update env state by getting demands from consumers.

        # Save prices in history
        prices = 1 / ys
        price_history.append(prices)

        cumulativeBuyerExperience = cumlativeBuyerExp(buyer_info, sellers)
        X = getPurchases(buyer_info, cumulativeBuyerExperience, ys, probAll)

        # Save purchased history
        purchases = X.sum(axis=0)
        purchase_history.append(purchases)

        # Get Buyer utilities and penalties in history
        buyerUtilities = buyerUtilitiesCalculator(X, ys, buyer_info.V, buyer_info.a_val, probAll,
                                                  buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)
        buyer_utility_history.append(buyerUtilities)

        buyerPenalties = buyerPenaltiesCalculator(X, ys, buyer_info.V, buyer_info.a_val, buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)
        buyer_penalty_history.append(buyerPenalties)

        # get next state based on actions taken in this round
        next_state = ydiff2action(ydiffActions,train_config.action_count, aux_price_min,aux_price_max)  # actions taken in this round is next state

        # Based on demands, calculate reward for all agents, and add observation to agents
        seller_utilities = []
        seller_penalties = []
        seller_provided_resources = []

        for j in range(0, seller_info.count):
            x_j = X[j]
            tmpSellerUtility, tmpSellerPenalty, z_j = sellers[j].reward(x_j, yAll)
            reward = tmpSellerUtility + tmpSellerPenalty

            # Update seller values
            sellers[j].add_purchase_history(x_j, z_j)
            seller_utilities.append(tmpSellerUtility)
            seller_penalties.append(tmpSellerPenalty)
            seller_provided_resources.append(z_j)

            # train agent

            sellers[j].replay_buffer.push(env_state, [ydiffActions[sellers[j].id]], reward, next_state, False)
            if len(sellers[j].replay_buffer) > train_config.batch_size:
                for i in range(train_config.update_itr):
                    _ = sellers[j].update(train_config.batch_size, reward_scale=10.,
                                          auto_entropy=train_config.auto_entropy,
                                          target_entropy=-1. * sellers[j].action_size)

            if train_iter % (train_config.update_step_size) == 0:
                sellers[j].save_model()

        # set current state to next state
        env_state = next_state

        # Get seller utilties and penalties in history
        seller_utilities = np.array(seller_utilities)
        seller_penalties = np.array(seller_penalties)
        seller_utility_history.append(seller_utilities)
        seller_penalty_history.append(seller_penalties)

        # update provided resources history
        seller_provided_resources = np.array(seller_provided_resources)
        provided_resource_history.append(seller_provided_resources)


    results_dict = {
        'policy_store': train_config.agents_store_dir,
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

def eval_policy(seller_info, buyer_info, train_config, results_dir, logger_pass):
    # Initialize the logger
    logger = logger_handle(logger_pass)

    # get required parameters for WolFPHC algorithm
    aux_price_min = 1 / seller_info.max_price
    aux_price_max = 1 / seller_info.min_price
    logger.info("Fetched raw market information..")

    # set mode to testing
    train_config.test = True

    # initialize seller agents
    sellers, logger = initialize_agent(seller_info, buyer_info, train_config, logger, is_trainer=False)

    # Get Containers to record history(Interesting insight: append in python list is O(1))
    price_history = []
    purchase_history = []
    provided_resource_history = []
    seller_utility_history = []
    seller_penalty_history = []
    buyer_utility_history = []
    buyer_penalty_history = []

    # Start Loop for training
    logger.info("Starting training iterations...")
    start_time = time.time()
    env_state = np.random.randint(0, train_config.action_count, seller_info.count)
    next_state = np.random.randint(0, train_config.action_count, seller_info.count)


    for eval_iter in range(0, train_config.iterations):

        if eval_iter % 10 == 0:
            logger.info("Finished %d evaluation iterations in %.3f secs..." % (eval_iter, time.time() - start_time))

        # get the prices for all seller agents
        ydiffActions = []
        for tmpSeller in sellers:
            ydiffActions.append(tmpSeller.policy_net.get_action(env_state, deterministic=train_config.deterministic))
        ydiffActions = np.array(ydiffActions).flatten()
        ys = aux_price_min + ydiffActions
        probAll, yAll = choose_prob(ys, compare=False, yAll=None)

        # Save prices in history
        prices = 1 / ys
        price_history.append(prices)

        cumulativeBuyerExperience = cumlativeBuyerExp(buyer_info, sellers)
        X = getPurchases(buyer_info, cumulativeBuyerExperience, ys, probAll)

        # Save purchased history
        purchases = X.sum(axis=0)
        purchase_history.append(purchases)

        # Get Buyer utilities and penalties in history
        buyerUtilities = buyerUtilitiesCalculator(X, ys, buyer_info.V, buyer_info.a_val, probAll,
                                                  buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)
        buyer_utility_history.append(buyerUtilities)

        buyerPenalties = buyerPenaltiesCalculator(X, ys, buyer_info.V, buyer_info.a_val, buyer_info.count,
                                                  cumulativeBuyerExperience, buyer_info.unfinished_task_penalty)
        buyer_penalty_history.append(buyerPenalties)

        # get next state based on actions taken in this round
        next_state = ydiff2action(ydiffActions,train_config.action_count, aux_price_min,aux_price_max)  # actions taken in this round is next state

        # Based on demands, calculate reward for all agents, and add observation to agents
        seller_utilities = []
        seller_penalties = []
        seller_provided_resources = []

        for j in range(0, seller_info.count):
            x_j = X[j]
            tmpSellerUtility, tmpSellerPenalty, z_j = sellers[j].reward(x_j,yAll)
            reward = tmpSellerUtility+tmpSellerPenalty

            # Update seller values
            sellers[j].add_purchase_history(x_j, z_j)
            seller_utilities.append(tmpSellerUtility)
            seller_penalties.append(tmpSellerPenalty)
            seller_provided_resources.append(z_j)

        # set current state to next state
        env_state=next_state

        # Get seller utilties and penalties in history
        seller_utilities = np.array(seller_utilities)
        seller_penalties = np.array(seller_penalties)
        seller_utility_history.append(seller_utilities)
        seller_penalty_history.append(seller_penalties)

        # update provided resources history
        seller_provided_resources = np.array(seller_provided_resources)
        provided_resource_history.append(seller_provided_resources)



    eval_dict = {
        'policy_store': train_config.agents_store_dir,
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

    return eval_dict

