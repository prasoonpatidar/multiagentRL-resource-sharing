'''
Soft Actor Critic Agent
'''

import math
import random
import argparse
import gym
import numpy as np
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from training.SAC.sacBrain import PolicyNetwork, SoftQNetwork, ValueNetwork, device
from training.SAC.replayBuffer import ReplayBuffer


class sacAgent:

    def __init__(self, seller_id, max_resources, cost_per_unit, action_number,
                 aux_price_min, aux_price_max, seller_idle_penalty, seller_count,
                 buyer_count,train_config,evaluate=False):
        # get basic information
        self.cost_per_unit = cost_per_unit
        self.id = seller_id
        self.max_resources = max_resources
        self.action_number = action_number
        self.penalty_coeff = seller_idle_penalty
        self.seller_count = seller_count
        self.buyer_count = buyer_count
        self.y_min = aux_price_min
        self.y_max = aux_price_max
        self.deterministic = train_config.deterministic

        # get derived information
        self.state_size = seller_count
        self.action_size = 1
        self.hidden_layer_size = train_config.hidden_layer_size
        self.replay_buffer_size = train_config.replay_buffer_size
        self.action_range = self.y_max - self.y_min
        self.weights_dir = f'{train_config.policy_store}/{seller_id}'
        # self.weights_file = f'{self.weights_dir}/weights.hd5'
        # self.__y = -1

        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        # Initialize agent networks
        self.soft_q_net1 = SoftQNetwork(self.state_size, self.action_size, self.hidden_layer_size).to(device)
        self.soft_q_net2 = SoftQNetwork(self.state_size, self.action_size, self.hidden_layer_size).to(device)
        self.target_soft_q_net1 = SoftQNetwork(self.state_size, self.action_size, self.hidden_layer_size).to(device)
        self.target_soft_q_net2 = SoftQNetwork(self.state_size, self.action_size, self.hidden_layer_size).to(device)
        self.policy_net = PolicyNetwork(self.state_size, self.action_size, self.hidden_layer_size, self.action_range).to(
            device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=device)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)


        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 3e-4
        alpha_lr  = 3e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.__providedResources = [np.zeros(self.buyer_count)]
        self.__demandedResources = [np.zeros(self.buyer_count)]



    def reward(self, x_j, yAll):

        deficit = np.maximum(0, np.sum(x_j) - self.max_resources)
        z_j = x_j * (1 - deficit / np.sum(x_j))

        # Get reward value based on everything
        R = 0
        self.__y = yAll[self.id]
        for i in range(0, self.buyer_count):
            R += (self.__y / (np.sum(yAll))) * (x_j[i] * (1 / self.__y) - z_j[i] * self.cost_per_unit)

        utility = R
        penalty = self.penalty_coeff * (np.sum(z_j) - self.max_resources)
        # R += penalty
        return utility, penalty, z_j

    def add_purchase_history(self, x_j, z_j):
        self.__demandedResources.append(x_j)
        self.__providedResources.append(z_j)

    def getBuyerExperience(self, i):
        return np.mean([xr[i] for xr in self.__providedResources])

    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(
            device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.policy_net.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(
            dim=0) + 1e-6)  # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q)
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

        # Training Q Function
        target_q_min = torch.min(self.target_soft_q_net1(next_state, new_next_action),
                                 self.target_soft_q_net2(next_state, new_next_action)) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min  # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1,
                                               target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        # Training Policy Function
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # print('q loss: ', q_value_loss1, q_value_loss2)
        # print('policy loss: ', policy_loss )

        # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()

    def save_model(self):
        path = f'{self.weights_dir}/'
        torch.save(self.soft_q_net1.state_dict(), path+'_q1')
        torch.save(self.soft_q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self):
        path = f'{self.weights_dir}/'
        self.soft_q_net1.load_state_dict(torch.load(path+'_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path+'_q2'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()
