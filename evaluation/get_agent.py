'''
This is a wrapper to fetch different RL agents based on training config
'''

# custom project libraries
from training.wolfPHC import run_helper as wolfPHC_helper
from training.QLearning import run_helper as QLearning_helper

# 'wolfPHC': wolfPHC_helper,
agents = {
    'QLearning': QLearning_helper,
    'wolfPHC': wolfPHC_helper,
}

def get_agent(train_config):
    return agents.get(train_config.rl_trainer)