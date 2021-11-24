import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import sys

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter("runs/mnist2")

# hyper parameters
input_size=784
hidden_size=50
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.01

# load dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

#get dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


examples = iter(train_loader)
samples, labels =examples.next()
print(samples.shape, labels.shape)
#
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmap='gray')
# plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist images',img_grid)


class NeuralNet(nn.Module):

    def __init__(self, input_size,hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, samples.reshape(-1,28*28))
# training loop
n_total_steps = len(train_loader)
running_loss = 0.
running_correct = 0.
for epoch in range(num_epochs):
    for step, (images, labels) in enumerate(train_loader):
        # reshape the images
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        #forward
        outs = model(images)
        loss = criterion(outs, labels)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss +=loss.item()
        _, predicted = torch.max(outs.data, 1)
        running_correct += (predicted==labels).sum().item()
        if (step+1) % 100==0:
            print(f'epoch {epoch+1}/{num_epochs}, step: {step}/{n_total_steps}, loss={loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch*n_total_steps + step)
            writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + step)
            running_loss=0.
            running_correct=0.

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)

        # get predictions
        outputs = model(images)
        _,predictions = torch.max(outputs, 1)
        n_samples = labels.shape[0]
        n_correct = (predictions==labels).sum().item()

    acc = 100. * (n_correct/n_samples)
    print(f'accuracy:{acc}%')

writer.close()