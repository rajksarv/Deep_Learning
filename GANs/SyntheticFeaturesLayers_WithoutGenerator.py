import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

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

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3,128,kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm([128,32,32]).cuda()
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(128,128,kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm([128,16,16]).cuda()
        self.relu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1)
        self.ln3 = nn.LayerNorm([128,16,16]).cuda()
        self.relu3 = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv2d(128,128,kernel_size=3, stride=2, padding=1)
        self.ln4 = nn.LayerNorm([128,8,8]).cuda()
        self.relu4 = nn.LeakyReLU(0.2)
        self.conv5 = nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1)
        self.ln5 = nn.LayerNorm([128,8,8]).cuda()
        self.relu5 = nn.LeakyReLU(0.2)
        self.conv6 = nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1)
        self.ln6 = nn.LayerNorm([128,8,8]).cuda()
        self.relu6 = nn.LeakyReLU(0.2)
        self.conv7 = nn.Conv2d(128,128,kernel_size=3, stride=1, padding=1)
        self.ln7 = nn.LayerNorm([128,8,8]).cuda()
        self.relu7 = nn.LeakyReLU(0.2)
        self.conv8 = nn.Conv2d(128,128,kernel_size=3, stride=2, padding=1)
        self.ln8 = nn.LayerNorm([128,4,4]).cuda()
        self.relu8 = nn.LeakyReLU(0.2)
        self.pool1 = nn.MaxPool2d(4, 4)
        #self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(128, 10)

    def forward(self, x, extract_features=0):
        x = (self.relu1((self.conv1(x))))
        x = self.ln1(x)
        #print(x.shape)
        x = (self.relu2((self.conv2(x))))
        x = self.ln2(x)
        #print(x.shape)
        x = (self.relu3((self.conv3(x))))
        x = self.ln3(x)
        #print(x.shape)
        x = (self.relu4((self.conv4(x))))
        x = self.ln4(x)
        #if(extract_features==4):
            #x = F.max_pool2d(x,8,8)
            #x = x.view(x.size(0),-1)
            #return x
        #print(x.shape)
        x = (self.relu5((self.conv5(x))))
        x = self.ln5(x)
        #print(x.shape)
        x = (self.relu6((self.conv6(x))))
        x = self.ln6(x)
        #print(x.shape)
        x = (self.relu7((self.conv7(x))))
        x = self.ln7(x)
        #print(x.shape)
        x = (self.relu8((self.conv8(x))))
        x = self.ln8(x)
        if(extract_features==8):
            x = F.max_pool2d(x,4,4)
            x = x.view(x.size(0),-1)
            return x
        x = self.pool1(x)
        #print(x.shape)
        x = x.view(x.size(0),-1)
        #x = (self.fc1(x))
        #print(x.shape)
        x = (self.fc10(x))
        #print(x.shape)
        #x = self.fc3(x)
        return x


model = discriminator()
model.load_state_dict(torch.load('parameters.ckpt'))
#model = torch.load('discriminator.model')
model.cuda()

batch_size = 64
batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()
X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).cuda()

lr = 0.1
weight_decay = 0.001
for i in range(200):
    output = model(X, extract_features=8)

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
plt.savefig('visualization/max_features_8_d.png', bbox_inches='tight')
plt.close(fig)