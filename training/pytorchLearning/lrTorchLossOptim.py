import torch as th
import torch.nn as nn
import numpy as np
'''
prediction: 
gradient computation:
loss computation:
parameter updates:
'''

'''Use loss and optimizer with torch'''

# 1. Design model(input_size, output_size, forward pass)
# 2. Construct Loss and optimizer
# 3. Training loop
#   - forward pass:
#   - backward pass:
#   - update weights





# f = w*x
# f = 2.x

X = th.tensor([1,2,3,4],dtype=th.float32)
Y = th.tensor([2,4,6,8],dtype=th.float32)

w = th.tensor(0.0,dtype=th.float32,requires_grad=True)

def forward(x):
    return w*x

# def loss(y,yp):
#     return ((y-yp)**2).mean()

# def gradient(x,y,yp):
#     #MSE = 1/N*(wx-y)**2
#     #dJ/dw = 1/N*2x*(w*x - y)
#     return np.dot(2*x, yp-y).mean()

print(f'prediction before training f(5)= {forward(5):.3f}')

# training
lr = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = th.optim.SGD([w],lr=lr)

for epoch in range(n_iters):
    # prediction
    y_pred = forward(X)

    # loss
    l = loss(Y,y_pred)

    #gradient
    l.backward() # dl/dw

    # dw = gradient(X,Y,y_pred)
    # before optim step, gradients are already there wrt loss function in w.grad aha!
    optimizer.step()

    # Update weights without gradient tracking
    with th.no_grad():
        w -= lr * w.grad

    # zero gradients
    w.grad.zero_()

    if epoch%10==0:
        print(f'epoch {epoch+1}: w: {w:.8f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')


