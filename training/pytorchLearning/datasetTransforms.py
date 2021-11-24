import math
import torch
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset as torchDataset, DataLoader as torchDataLoader
import numpy as np

# Data process
from sklearn import datasets


class BCDataset(torchDataset):
    def __init__(self, X, y, transform=None):
        self.x = X
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.x.shape[0]


# Transform class
class ToTensor:
    def __call__(self, sample):
        x, y = sample
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(np.array(y).astype(np.float32))


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

bcdataset = BCDataset(X, y, transform=ToTensor())
print(bcdataset[1])
bcloader = torchDataLoader(bcdataset, batch_size=4, shuffle=True)

num_epochs = 3
num_iters = math.ceil(len(bcdataset) / 4)

for epoch in range(num_epochs):
    for step, (X_batch, y_batch) in enumerate(bcloader):
        if (step) % 30 == 0:
            print(f'epoch {epoch}/{num_epochs}, step {step}/{num_iters}, sum: {y_batch.mean()}')

# bciter = iter(bcloader)
# data = bciter.next()
# features, labels = data
# print(features, labels)
