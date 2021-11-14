'''
This file contains various training algorithm configurations
'''

from types import SimpleNamespace

train_config = {
    # Wolf-PHC Configs
    'wolfPHC_r1': {
        "rl_trainer": "wolfPHC",
        "action_count": 8,
        "discount_factor": 0.3,
        "learning_rate": 0.33,
        "iterations": 10000,
        "train":True,
        "evaluate":True,
        "show_results":True,
        "store_results":True
    }

}

def get_train_config(config_name):
    return SimpleNamespace(**train_config[config_name])