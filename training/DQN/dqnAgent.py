'''
Wrapper class for dqn agent around DQN brain
'''

import numpy as np
import random
import os

# custom project libraries
from training.DQN.brain import Brain
from training.DQN.uniform_experience_replay import Memory as UER
from training.DQN.prioritized_experience_replay import Memory as PER

MAX_EPSILON = 1.0
MIN_EPSILON = 0.05

MIN_BETA = 0.4
MAX_BETA = 1.0


class dqnAgent:
    epsilon = MAX_EPSILON
    beta = MIN_BETA
    # constructor for dqnAgent
    def __init__(self, seller_id, max_resources, cost_per_unit, action_number,
                 aux_price_min, aux_price_max, seller_idle_penalty, seller_count,
                 buyer_count, train_config, evaluate=False):

        # get basic information
        self.cost_per_unit = cost_per_unit
        self.id = seller_id
        self.max_resources = max_resources
        self.action_number = action_number
        self.penalty_coeff = seller_idle_penalty
        self.first_step_memory = train_config.first_step_memory
        self.replay_steps = train_config.replay_steps
        self.seller_count = seller_count
        self.buyer_count = buyer_count
        self.y_min = aux_price_min
        self.y_max = aux_price_max

        # get derived information
        self.state_size = seller_count
        self.action_size = action_number
        self.weights_dir = f'{train_config.policy_store}/{seller_id}'
        self.weights_file = f'{self.weights_dir}/weights.hd5'
        # self.__y = -1

        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)

        self.bee_index = seller_id
        self.learning_rate = train_config.learning_rate
        self.gamma = 0.95
        self.brain = Brain(self.state_size, self.action_size, self.weights_file, train_config, evaluate)
        self.memory_model = train_config.memory


        if self.memory_model == 'UER':
            self.memory = UER(train_config.memory_capacity)

        elif self.memory_model == 'PER':
            self.memory = PER(train_config.memory_capacity, train_config.prioritization_scale)

        else:
            print('Invalid memory model!')

        self.target_type = train_config.target_type
        self.update_target_frequency = train_config.target_frequency
        self.max_exploration_step = train_config.maximum_exploration
        self.batch_size = train_config.batch_size
        self.step = 0
        self.test = evaluate
        if self.test:
            self.epsilon = MIN_EPSILON

        self.__providedResources = [np.full(self.buyer_count, 0.)]
        self.__demandedResources = [np.full(self.buyer_count, 0.)]

        # circular buffer to remember past rewards to select best model
        self.max_average_rewards = -np.inf
        self.reward_buffer_size = train_config.reward_buffer_size
        self.reward_buffer = np.zeros(train_config.reward_buffer_size)
        self.reward_buffer_pos = 0


    def greedy_actor(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.brain.predict_one_sample(state))

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
        return np.mean([xr[i] for xr in self.__providedResources[-100:]])

    def find_targets_per(self, batch):
        batch_len = len(batch)

        states = np.array([o[1][0] for o in batch])
        states_ = np.array([o[1][3] for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batch_len, self.state_size))
        y = np.zeros((batch_len, self.action_size))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i][1]
            s = o[0]
            a = o[1][self.bee_index]
            r = o[2]
            s_ = o[3]
            done = o[4]

            t = p[i]
            old_value = t[a]
            if done:
                t[a] = r
            else:
                if self.target_type == 'DDQN':
                    t[a] = r + self.gamma * pTarget_[i][np.argmax(p_[i])]
                elif self.target_type == 'DQN':
                    t[a] = r + self.gamma * np.amax(pTarget_[i])
                else:
                    print('Invalid type for target network!')

            x[i] = s
            y[i] = t
            errors[i] = np.abs(t[a] - old_value)

        return [x, y, errors]

    def find_targets_uer(self, batch):
        batch_len = len(batch)

        states = np.array([o[0] for o in batch])
        states_ = np.array([o[3] for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batch_len, self.state_size))
        y = np.zeros((batch_len, self.action_size))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i]
            s = o[0]
            a = o[1][self.bee_index]
            r = o[2]
            s_ = o[3]
            done = o[4]

            t = p[i]
            old_value = t[a]
            if done:
                t[a] = r
            else:
                if self.target_type == 'DDQN':
                    t[a] = r + self.gamma * pTarget_[i][np.argmax(p_[i])]
                elif self.target_type == 'DQN':
                    t[a] = r + self.gamma * np.amax(pTarget_[i])
                else:
                    print('Invalid type for target network!')

            x[i] = s
            y[i] = t
            errors[i] = np.abs(t[a] - old_value)

        return [x, y]

    def observe(self, sample):

        if self.memory_model == 'UER':
            self.memory.remember(sample)

        elif self.memory_model == 'PER':
            _, _, errors = self.find_targets_per([[0, sample]])
            self.memory.remember(sample, errors[0])

        else:
            print('Invalid memory model!')

    def decay_epsilon(self):
        # slowly decrease Epsilon based on our experience
        self.step += 1

        if self.test:
            self.epsilon = MIN_EPSILON
            self.beta = MAX_BETA
        else:
            if self.step < self.max_exploration_step:
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (self.max_exploration_step - self.step)/self.max_exploration_step
                self.beta = MAX_BETA + (MIN_BETA - MAX_BETA) * (self.max_exploration_step - self.step)/self.max_exploration_step
            else:
                self.epsilon = MIN_EPSILON

    def replay(self):

        if self.memory_model == 'UER':
            batch = self.memory.sample(self.batch_size)
            x, y = self.find_targets_uer(batch)
            self.brain.train(x, y)

        elif self.memory_model == 'PER':
            [batch, batch_indices, batch_priorities] = self.memory.sample(self.batch_size)
            x, y, errors = self.find_targets_per(batch)

            normalized_batch_priorities = [float(i) / sum(batch_priorities) for i in batch_priorities]
            importance_sampling_weights = [(self.batch_size * i) ** (-1 * self.beta)
                                           for i in normalized_batch_priorities]
            normalized_importance_sampling_weights = [float(i) / max(importance_sampling_weights)
                                                      for i in importance_sampling_weights]
            sample_weights = [errors[i] * normalized_importance_sampling_weights[i] for i in range(len(errors))]

            self.brain.train(x, y, np.array(sample_weights))

            self.memory.update(batch_indices, errors)

        else:
            print('Invalid memory model!')

    def update_target_model(self):
        if self.step % self.update_target_frequency == 0:
            self.brain.update_target_model()