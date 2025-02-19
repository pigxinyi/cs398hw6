import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
batch_size=128

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
    def set(self,layer=0):
        self.extract_features =layer
    def forward(self, x):
      if self.extract_features == 8:
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
    
            x = F.max_pool2d(x, kenel_size=4, stride=4)
            x = x.view(-1, 196)
            return x
     elif self.extract_features == 4:
            x = self.conv1( x )
            x = F.leaky_relu(self.layernorm1( x ))
            x = self.conv2( x )
            x = F.leaky_relu(self.layernorm2( x ))
            x = self.conv3( x )
            x = F.leaky_relu(self.layernorm3( x ))
            x = self.conv4( x )
            x = F.leaky_relu(self.layernorm4( x ))

            x = F.max_pool2d(x, kernel_size=8, stride=8)
            x = x.view(-1, 196)
            return x
     else:
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

        
def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

model = torch.load('cifar10.model')
model.cuda()
model.eval()


batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

###################################part 1##########################################
## save real images
samples = X_batch.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/real_images.png', bbox_inches='tight')
plt.close(fig)

_, output = model(X_batch)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)


## slightly jitter all input images
criterion = nn.CrossEntropyLoss(reduce=False)
loss = criterion(output, Y_batch_alternate)

gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                          grad_outputs=torch.ones(loss.size()).cuda(),
                          create_graph=True, retain_graph=False, only_inputs=True)[0]

# save gradient jitter
gradient_image = gradients.data.cpu().numpy()
gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
gradient_image = gradient_image.transpose(0,2,3,1)
fig = plot(gradient_image[0:100])
plt.savefig('visualization/gradient_image.png', bbox_inches='tight')
plt.close(fig)


# jitter input image
gradients[gradients>0.0] = 1.0
gradients[gradients<0.0] = -1.0

gain = 8.0
X_batch_modified = X_batch - gain*0.007843137*gradients
X_batch_modified[X_batch_modified>1.0] = 1.0
X_batch_modified[X_batch_modified<-1.0] = -1.0

## evaluate new fake images
_, output = model(X_batch_modified)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)

## save fake images
samples = X_batch_modified.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/jittered_images.png', bbox_inches='tight')
plt.close(fig)



##########################part 2 A###################################################

X = X_batch.mean(dim=0)
X = X.repeat(10,1,1,1)

Y = torch.arange(10).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in xrange(200):
    _, output = model(X)

    loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples)
plt.savefig('visualization/max_class_discriminator_without_generator.png', bbox_inches='tight')
plt.close(fig)






#############################################part 3 A#########
X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in xrange(200):
    model.set(layer=4)
    output = model(X)
   
    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/max_features_discriminator_without_generator_4.png', bbox_inches='tight')
plt.close(fig)


for i in xrange(200):
    model.set(layer=8)
    output = model(X)
   
    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).cuda(),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/max_features_discriminator_without_generator_8.png', bbox_inches='tight')
plt.close(fig)




