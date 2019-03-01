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
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)
testloader = enumerate(testloader)
model = discriminator()
model.load_state_dict(torch.load('parameters.ckpt'))
#model = torch.load('cifar10.model')
model.cuda()
model.eval()

batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()

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

    def forward(self, x):
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
        x = self.pool1(x)
        #print(x.shape)
        x = x.view(x.size(0),-1)
        #x = (self.fc1(x))
        #print(x.shape)
        x = (self.fc10(x))
        #print(x.shape)
        #x = self.fc3(x)
        return x

samples = X_batch.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/real_images.png', bbox_inches='tight')
plt.close(fig)

batch_size = 64
output = model(X_batch)
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
output = model(X_batch_modified)
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