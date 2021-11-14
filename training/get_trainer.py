'''
This is a wrapper to fetch different trainers based on training config
'''

from training.wolfPHC import run_wolfPHC as WolfPHCTrainer
from training.QLearning import run_qlearning as QLearningTrainer

trainers = {
    'wolfPHC':WolfPHCTrainer,
    'QLearning': QLearningTrainer
}

def get_trainer(train_config):
    return trainers.get(train_config.rl_trainer)
