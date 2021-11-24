import torch as th
import torch.nn as nn
import numpy as np
'''
prediction: 
gradient computation:
loss computation:
parameter updates:
'''

'''Complete training pipeline with torch'''

# 1. Design model(input_size, output_size, forward pass)
# 2. Construct Loss and optimizer
# 3. Training loop
#   - forward pass:
#   - backward pass:
#   - update weights

# f = w*x
# f = 2.x

X = th.tensor([[1],[2],[3],[4]],dtype=th.float32)
Y = th.tensor([[2],[4],[6],[8]],dtype=th.float32)

X_test = th.tensor([5], dtype=th.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features
# model = nn.Linear(input_size, output_size)

#custom model

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression,self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)
# w = th.tensor(0.0,dtype=th.float32,requires_grad=True)

# def forward(x):
#     return w*x

# def loss(y,yp):
#     return ((y-yp)**2).mean()

# def gradient(x,y,yp):
#     #MSE = 1/N*(wx-y)**2
#     #dJ/dw = 1/N*2x*(w*x - y)
#     return np.dot(2*x, yp-y).mean()

print(f'prediction before training f(5)= {model(X_test).item():.3f}')

# training
lr = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = th.optim.SGD(model.parameters(),lr=lr)

for epoch in range(n_iters):
    # prediction
    y_pred = model(X)

    # loss
    l = loss(Y,y_pred)

    #calculate gradients
    l.backward() # dl/dw

    # update weights
    # before optim step, gradients are already there wrt loss function in w.grad aha!
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch%10==0:
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w: {w[0][0]}, b: {b[0]}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')


