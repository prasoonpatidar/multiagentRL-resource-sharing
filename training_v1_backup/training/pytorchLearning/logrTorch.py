'''
0) Setup Data
1) Design Model (input_size, output_size, forward_pass)
2) Construct Loss, Optimizer
3) Training Loop
    - forward pass
    - gradients
    - update weights
'''
import torch
import torch as th
import torch.nn as nn
import numpy as np

# Data process
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# 0) prepare data
bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target

n_samples, n_features = X.shape
print(n_samples,n_features)

X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=123)

# scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) # use mean and std from training data

# to torch
X_train = th.from_numpy(X_train.astype(np.float32))
X_test = th.from_numpy(X_test.astype(np.float32))
y_train = th.from_numpy(y_train.astype(np.float32))
y_test = th.from_numpy(y_test.astype(np.float32))

# reshape y
y_train, y_test = y_train.view(y_train.shape[0], 1), y_test.view(y_test.shape[0],1)

# 1) model

class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.lin = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = th.sigmoid(self.lin(x))
        return y_pred


model = LogisticRegression(n_features)

#2) Loss
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# 3) training epoch

num_epochs = 100

for epoch in range(num_epochs):
    # forward
    y_pred = model(X_train)

    # loss
    loss = criterion(y_pred, y_train)

    # gradients
    loss.backward()

    # updates and zero gradients
    optimizer.step()
    optimizer.zero_grad()

    # epoch info
    if (epoch + 1)%10==0:
        print(f'epoch:{epoch+1},loss={loss.item():.8f}')


with torch.no_grad():
    y_predicted = model(X_test) # sigmoid return
    y_predicted_class = y_predicted.round() # >0.5 1 else 0

    acc = y_predicted_class.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy: {acc:.3f}')







