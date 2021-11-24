'''
Learning pytorch on the go
'''
import torch as th
import numpy as np

x = th.empty(3,4, dtype=th.float32)
y= th.ones(2,dtype=th.int)
z = th.tensor([3.4,5.2])

# print(y+z)
# Add multiply div
print(y.add_(th.ones(2, dtype=th.int)))
# slicing

w = th.empty(4,4)

print(w[1:4,0:2])

z = w.view(-1,2)
# print(z)

# th to np
a = th.ones(5)
# print(a)
b = a.numpy()
# print(b)

# np to th
c = th.from_numpy(b)
c += 1
# if modified in place, .numpy and .from_numpy points to same memory location
# print(a,b,c)

# cuda enabling .to functionality
x = th.ones(5, requires_grad=True)
# print(x)

# autograd package pytorch

x = th.randn(3, requires_grad=True)
print(x)
y = x+2
print(y)
z = y*y*2
# z = z.mean()
print(z)
v = th.tensor([0.1,1.0,0.001], dtype=th.float32) # need if final dz/dx is not scaler (why?)
# z.backward(v) # dz/dx
print(x.grad)

# prevent from tracking

# x.requires_grad(False)
# x.detach(), new tensor
with th.no_grad():
    x = x+2
print(x)
z.backward(v)

weights = th.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    # Important to empty gradients after every loop
    weights.grad.zero_()

# Same for custom optimizers
# optimizer = th.optim.SGD(weights, lr = 0.01)
# optimizer.step()
# optimizer.zero_grad()

## backpropogation

# Chain Rule and computational graph

'''
Three steps:
- Forward pass
- local gradients
- backpropogation
'''

x = th.tensor(1.0)
y=th.tensor(2.0)
w = th.tensor(1.0, requires_grad=True)

# Forward pass
y_h = w*x
loss = (y_h-y)**2

print(loss)
#backward pass
loss.backward()
print(w.grad)

#update weights










