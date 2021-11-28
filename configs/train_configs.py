'''
This file contains various training algorithm configurations
'''

from types import SimpleNamespace

train_config = {
    # Wolf-PHC Configs
    'wolf_r1': {
        "rl_trainer": "wolfPHC",
        "policy_store":"policy_store/wolf_r1/agents_info",
        "print_freq":50,
        # "action_count": 8,
        "discount_factor": 0.3,
        "learning_rate": 0.33,
        "iterations": 1000,
        "train":True,
        "evaluate":True,
        "show_results":True,
        "store_results":True
    },
    'q_r1':{
        "rl_trainer":"QLearning",
        "policy_store":"policy_store/q_r1/agents_info",
        "print_freq":50,
        # "action_count":8,
        "discount_factor":0.99,
        "learning_rate":0.33,
        "explore_prob":0.04,
        "iterations": 1000,
        "train": False,
        "evaluate": True,
        "show_results": True,
        "store_results": True
    },
    'dqn_r1':{
        "rl_trainer": "DQN",
        "policy_store":"policy_store/dqn_r1/agents_info",
        "print_freq":50,
        # "action_count": 8,
        "iterations": 1000,
        "train": True,
        "evaluate": True,
        "show_results": True,
        "store_results": True,
        # Particular to DQN Brain in agents
        "learning_rate": 0.05,
        "optimizer": "RMSProp",
        "memory_capacity": 10000,
        "reward_buffer_size": 100,
        "batch_size": 64,
        "target_frequency": 100,
        "maximum_exploration": 100,
        "first_step_memory": 10,
        "replay_steps": 10,
        "number_nodes": 256,
        "target_type": "DQN",  # options DQN, DDQN
        "memory": "UER",  # options UER, PER
        "prioritization_scale": 0.5,
        "dueling": False,
        "gpu_num": 3,
        "reward_mode": 2,
    },
    'wolf_r2': {
        "rl_trainer": "wolfPHC",
        "policy_store":"policy_store/wolf_r2/agents_info",
        "print_freq":50,
        # "action_count": 8,
        "discount_factor": 0.1,
        "learning_rate": 0.33,
        "iterations": 1000,
        "train":True,
        "evaluate":True,
        "show_results":True,
        "store_results":True
    },
    'dqn_r2': {
        "rl_trainer": "DQN",
        "policy_store": "policy_store/dqn_r1/agents_info",
        "print_freq": 50,
        # "action_count": 8,
        "iterations": 1000,
        "train": True,
        "evaluate": True,
        "show_results": True,
        "store_results": True,
        # Particular to DQN Brain in agents
        "learning_rate": 0.05,
        "optimizer": "RMSProp",
        "memory_capacity": 10000,
        "reward_buffer_size": 100,
        "batch_size": 64,
        "target_frequency": 100,
        "maximum_exploration": 100,
        "first_step_memory": 10,
        "replay_steps": 10,
        "number_nodes": 256,
        "target_type": "DQN",  # options DQN, DDQN
        "memory": "PER",  # options UER, PER
        "prioritization_scale": 0.5,
        "dueling": False,
        "gpu_num": 3,
        "reward_mode": 2,
    },
    'ddqn_r2': {
        "rl_trainer": "DQN",
        "policy_store": "policy_store/dqn_r1/agents_info",
        "print_freq": 50,
        # "action_count": 8,
        "iterations": 1000,
        "train": True,
        "evaluate": True,
        "show_results": True,
        "store_results": True,
        # Particular to DQN Brain in agents
        "learning_rate": 0.05,
        "optimizer": "RMSProp",
        "memory_capacity": 10000,
        "reward_buffer_size": 100,
        "batch_size": 64,
        "target_frequency": 100,
        "maximum_exploration": 100,
        "first_step_memory": 10,
        "replay_steps": 10,
        "number_nodes": 256,
        "target_type": "DDQN",  # options DQN, DDQN
        "memory": "PER",  # options UER, PER
        "prioritization_scale": 0.5,
        "dueling": False,
        "gpu_num": 3,
        "reward_mode": 2,
    },
    'dqn_duel_r2': {
        "rl_trainer": "DQN",
        "policy_store": "policy_store/dqn_r1/agents_info",
        "print_freq": 50,
        # "action_count": 8,
        "iterations": 1000,
        "train": True,
        "evaluate": True,
        "show_results": True,
        "store_results": True,
        # Particular to DQN Brain in agents
        "learning_rate": 0.05,
        "optimizer": "RMSProp",
        "memory_capacity": 10000,
        "reward_buffer_size": 100,
        "batch_size": 64,
        "target_frequency": 100,
        "maximum_exploration": 100,
        "first_step_memory": 10,
        "replay_steps": 10,
        "number_nodes": 256,
        "target_type": "DQN",  # options DQN, DDQN
        "memory": "PER",  # options UER, PER
        "prioritization_scale": 0.5,
        "dueling": True,
        "gpu_num": 3,
        "reward_mode": 2,
    },
    'sac_r1':{
        "rl_trainer": "SAC",
        "policy_store":"policy_store/sac_r1/agents_info",
        "print_freq":50,
        # "action_count": 8,
        "iterations": 1000,
        "train": True,
        "evaluate": True,
        "show_results": True,
        "store_results": True,

        # SAC specific parameters
        'batch_size': 10,
        'explore_steps': 10,  # for random action sampling in the beginning of training
        'update_itr': 1,
        'auto_entropy':True,
        'deterministic':False,
        'hidden_layer_size': 64,
        'replay_buffer_size':1e3,
        'update_step_size':20,

    },
    'sac_r2':{
        "rl_trainer": "SAC",
        "policy_store":"policy_store/sac_r1/agents_info",
        "print_freq":50,
        # "action_count": 8,
        "iterations": 1000,
        "train": True,
        "evaluate": True,
        "show_results": True,
        "store_results": True,

        # SAC specific parameters
        'batch_size': 10,
        'explore_steps': 10,  # for random action sampling in the beginning of training
        'update_itr': 1,
        'auto_entropy':True,
        'deterministic':True,
        'hidden_layer_size': 64,
        'replay_buffer_size':1e3,
        'update_step_size':20,

    }
}

def get_train_config(config_name):
    t_config = SimpleNamespace(**train_config[config_name])
    t_config.config_name = config_name
    return t_config
