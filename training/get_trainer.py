'''
This is a wrapper to fetch different trainers based on training config
'''

from training.WoLF import wolfTrainer
from training.QLearning import qTrainer
from training.DQN import dqnTrainer
from training.SAC import sacTrainer

trainers = {
    'wolfPHC': wolfTrainer,
    'QLearning': qTrainer,
    'DQN': dqnTrainer,
    'SAC':sacTrainer
}


def get_trainer(rl_trainer_name):
    return trainers.get(rl_trainer_name)
