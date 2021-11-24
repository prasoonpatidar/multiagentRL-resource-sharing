import torch as th
import numpy as np
'''
prediction: 
gradient computation:
loss computation:
parameter updates:
'''

'''All Manual with numpy'''
# f = w*x
# f = 2.x

X = np.array([1,2,3,4],dtype=np.float32)
Y = np.array([2,4,6,8],dtype=np.float32)

w = 0.0

def forward(x):
    return w*x
def loss(y,yp):
    return np.mean((y-yp)**2)
def gradient(x,y,yp):
    #MSE = 1/N*(wx-y)**2
    #dJ/dw = 1/N*2x*(w*x - y)
    return np.dot(2*x, yp-y).mean()

print(f'prediction before training f(5)= {forward(5):.3f}')

# training
lr = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction
    y_pred = forward(X)

    # loss
    l = loss(Y,y_pred)

    #gradient
    dw = gradient(X,Y,y_pred)

    # Update weights
    w-=lr*dw
    if epoch%10==0:
        print(f'epoch {epoch+1}: w: {w:.8f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')


