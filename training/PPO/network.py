'''
Networks used for policy and value function evaluations
'''

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class DefaultNN(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(DefaultNN,self).__init__()
        self.l1 = nn.Linear(input_dim, 64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,output_dim)

    def forward(self,obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs.astype(np.float32))
        out = F.relu(self.l1(obs))
        out = F.relu(self.l2(out))
        out = self.l3(out)
        return out
