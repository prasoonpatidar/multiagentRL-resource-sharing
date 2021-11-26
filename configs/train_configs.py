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
        "iterations": 10,
        "train":True,
        "evaluate":True,
        "show_results":True,
        "store_results":True
    },
    'q_r1':{
        "rl_trainer":"QLearning",
        "action_count":8,
        "discount_factor":0.99,
        "explore_prob":0.04,
        "iterations": 200,
        "train": True,
        "evaluate": True,
        "show_results": True,
        "store_results": True
    },
    'dqn_r1':{
        "rl_trainer": "DQN",
        "agents_store_dir":"results/DQNrun/agents_info",
        "action_count": 4,
        "iterations": 200,
        "train": True,
        "evaluate": True,
        "show_results": True,
        "store_results": True,
        # Particular to DQN Brain in agents
        "learning_rate":0.05,
        "optimizer": "RMSProp",
        "memory_capacity": 100000,
        "batch_size": 64,
        "target_frequency": 10000,
        "maximum_exploration": 100000,
        "first_step_memory": 0,
        "replay_steps": 4,
        "number_nodes": 256,
        "target_type": "DQN",  # options DQN, DDQN
        "memory": "UER",  # options UER, PER
        "prioritization_scale": 0.5,
        "dueling": False,
        "gpu_num": 3,
        "test": False,
        "reward_mode": 2,
    },
    'wolfPHC_r2': {
        "rl_trainer": "wolfPHC",
        "action_count": 8,
        "discount_factor": 0.1,
        "learning_rate": 0.33,
        "iterations": 10,
        "train":True,
        "evaluate":True,
        "show_results":True,
        "store_results":True
    }
}

def get_train_config(config_name):
    return SimpleNamespace(**train_config[config_name])
