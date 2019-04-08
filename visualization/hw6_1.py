import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
import time

#Device configuration
torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Hyper parameters
num_epochs = 100
num_classes = 10
batch_size = 128
learning_rate =0.0001
DIM = 32

#load data
print('load data')
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

# Define CNN
class discriminator(nn.Module):
   
        
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=196, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=2, padding=1)

        
        self.layernorm1 = nn.LayerNorm((196, 32, 32))
        self.layernorm2 = nn.LayerNorm((196, 16, 16))
        self.layernorm3 = nn.LayerNorm((196, 16, 16))
        self.layernorm4 = nn.LayerNorm((196, 8, 8))
        self.layernorm5 = nn.LayerNorm((196, 8, 8))
        self.layernorm6 = nn.LayerNorm((196, 8, 8))
        self.layernorm7 = nn.LayerNorm((196, 8, 8))
        self.layernorm8 = nn.LayerNorm((196, 4, 4))

        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)

    def forward(self, x):
        x = self.conv1( x )
        x = F.leaky_relu(self.layernorm1( x ))
        x = self.conv2( x )
        x = F.leaky_relu(self.layernorm2( x ))
        x = self.conv3( x )
        x = F.leaky_relu(self.layernorm3( x ))
        x = self.conv4( x )
        x = F.leaky_relu(self.layernorm4( x ))
        x = self.conv5( x )
        x = F.leaky_relu(self.layernorm5( x ))
        x = self.conv6( x )
        x = F.leaky_relu(self.layernorm6( x ))
        x = self.conv7( x )
        x = F.leaky_relu(self.layernorm7( x ))
        x = self.conv8( x )
        x = F.leaky_relu(self.layernorm8( x ))
    
        x = F.max_pool2d(x, kernel_size=4, stride=4)
        
        x = x.view(x.size(0), -1)
        fc1_x = self.fc1(x)
        fc10_x = self.fc10(x)
        return fc1_x, fc10_x

model =  discriminator()
model.cuda()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


print(model)
train_accu = []
test_accu = []
# Train the model
for epoch in range(0,num_epochs):
    model.train()

    if epoch > 10:
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if ('step' in state) and (state['step'] >= 1024):
                    state['step'] = 1000    

    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
         for param_group in optimizer.param_groups:
             param_group['lr'] = learning_rate/100.0

    epoch_acc = 0.0
    epoch_counter = 0

    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        if(Y_train_batch.shape[0] < batch_size):
            continue
        labels =Y_train_batch.cuda()
        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        #h = model(X_train_batch)
        _, output = model(X_train_batch)

        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        _,prediction = torch.max(output,1)   # first column has actual prob.
        epoch_acc += float((prediction==labels).sum())
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    train_accu.append(epoch_acc)

    #print("Train:  ", "%.2f" % (epoch_acc*100.0))
    torch.save(model,'cifar10.model')
## test
    model.eval()

    epoch_acc = 0.0
    epoch_counter = 0

    time1 = time.time()

    for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):

        if(Y_test_batch.shape[0]<batch_size):
            continue
        labels =Y_test_batch.cuda()
        X_test_batch = Variable(X_test_batch).cuda()
        Y_test_batch = Variable(Y_test_batch).cuda()

        _, output = model(X_test_batch)
        loss = criterion(output, Y_test_batch)       
        #h = model(X_test_batch)
        #loss = criterion(h,Y_test_batch)
        #pred = F.softmax(h,dim=1)
        
        _,prediction = torch.max(output,1)   # first column has actual prob.
        epoch_acc += float((prediction==labels).sum())
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    test_accu.append(epoch_acc)

    time2 = time.time()
    time_elapsed = time2 - time1

    print("TEST:  ", "%.2f" % (epoch_acc*100.0))
