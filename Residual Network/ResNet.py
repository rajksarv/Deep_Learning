import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

#Data Augmentation

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, kernel_size=3, stride=1):
    #print(in_planes, out_planes)
    return nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=1)

class BasicBlock1(nn.Module):
    #expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, 3, 1, padding=1)
        #print(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32,3,1,1)
        self.bn2 = nn.BatchNorm2d(32)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        #print(residual.shape)
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)

        x = self.conv2(x)
        #print(x.shape)
        x = self.bn2(x)
        #print(x.shape)

        if self.downsample is not None:
            residual = self.downsample(x)
            #print(residual.shape)
        
       
        x += residual
        #print(x.shape)
        #x = self.relu(x)
        #print(x.shape)

        return x

class BasicBlock2(nn.Module):
    #expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1)
        #print(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64,3,1,1)
        self.bn2 = nn.BatchNorm2d(64)
        self.upsample = upsample
        self.upsample1 = nn.Upsample(32)
        self.stride = stride
        self.conv3 = nn.Conv2d(32, 64,1,1,0)
        self.inplanes = inplanes

    def forward(self, x):
        if(self.inplanes ==32):
            residual = self.conv3(x)
        else:
            residual = x
        #print(residual.shape)
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)

        x = self.conv2(x)
        #print(x.shape)
        x = self.bn2(x)
        #print(x.shape)

        if self.upsample is not None:
            x = self.upsample1(x)
            #print(x.shape)
        
       
        x += residual
        #print(x.shape)
        #x = self.relu(x)
        #print(x.shape)

        return x

class BasicBlock3(nn.Module):
    #expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(BasicBlock3, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1)
        #print(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 128,3,1,1)
        self.bn2 = nn.BatchNorm2d(128)
        self.upsample = upsample
        self.upsample1 = nn.Upsample(32)
        self.stride = stride
        self.conv3 = nn.Conv2d(64, 128,1,1,0)
        self.inplanes = inplanes

    def forward(self, x):
        if(self.inplanes ==64):
            residual = self.conv3(x)
        else:
            residual = x
        #print(residual.shape)
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)

        x = self.conv2(x)
        #print(x.shape)
        x = self.bn2(x)
        #print(x.shape)

        if self.upsample is not None:
            x = self.upsample1(x)
            #print(x.shape)
        
       
        x += residual
        #print(x.shape)
        #x = self.relu(x)
        #print(x.shape)

        return x

class BasicBlock4(nn.Module):
    #expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(BasicBlock4, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1)
        #print(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 256,3,1,1)
        self.bn2 = nn.BatchNorm2d(256)
        self.upsample = upsample
        self.upsample1 = nn.Upsample(32)
        self.stride = stride
        self.conv3 = nn.Conv2d(128, 256,1,1,0)
        self.inplanes = inplanes

    def forward(self, x):
        if(self.inplanes ==128):
            residual = self.conv3(x)
        else:
            residual = x
        #print(residual.shape)
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.bn1(x)
        #print(x.shape)
        x = self.relu(x)
        #print(x.shape)

        x = self.conv2(x)
        #print(x.shape)
        x = self.bn2(x)
        #print(x.shape)

        if self.upsample is not None:
            x = self.upsample1(x)
            #print(x.shape)
        
       
        x += residual
        #print(x.shape)
        #x = self.relu(x)
        #print(x.shape)

        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3,32, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.3)
        self.layers1 = self._make_layer1(2, 32, 32, 1)
        self.layers2 = self._make_layer2(4, 32, 64, 2)
        self.layers3 = self._make_layer3(4, 64, 128, 2)
        self.layers4 = self._make_layer4(2, 128, 256, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(65536, 100)
        
        
    def _make_layer1(self, layer_count, channels_in, channels, stride):
        basicblock = BasicBlock1(channels_in, channels, stride)
        basicblock1 = BasicBlock1(channels, channels)
        downsample = None
        if ((stride != 1) or (channels_in != channels)):
            downsample = nn.Sequential(conv3x3(channels_in, channels, kernel_size=1, stride=stride),nn.BatchNorm2d(channels))

        layers = []
        #print(channels_in, channels)
        layers.append(basicblock)
        #print(channels_in, channels,stride)
        for i in range(1,layer_count):
            layers.append(basicblock1)
            #print(channels_in, channels)
        return nn.Sequential(*layers)
    
    def _make_layer2(self, layer_count, channels_in, channels, stride):
        upsample = None

        if (stride != 1):
            upsample = 1 
        
        layers = []
     
        layers.append(BasicBlock2(channels_in, channels, stride, upsample))
        #print(channels_in, channels,stride)
        for i in range(1,layer_count):
            layers.append(BasicBlock2(channels, channels))
            #print(channels_in, channels)
        return nn.Sequential(*layers)
    
    def _make_layer3(self, layer_count, channels_in, channels, stride):
        upsample = None

        if (stride != 1):
            upsample = 1 
        
        layers = []
     
        layers.append(BasicBlock3(channels_in, channels, stride, upsample))
        #print(channels_in, channels,stride)
        for i in range(1,layer_count):
            layers.append(BasicBlock3(channels, channels))
            #print(channels_in, channels)
        return nn.Sequential(*layers)
    
    def _make_layer4(self, layer_count, channels_in, channels, stride):
        upsample = None

        if (stride != 1):
            upsample = 1 
        
        layers = []
     
        layers.append(BasicBlock4(channels_in, channels, stride, upsample))
        #print(channels_in, channels,stride)
        for i in range(1,layer_count):
            layers.append(BasicBlock4(channels, channels))
            #print(channels_in, channels)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        #print(x.shape)
        x = (F.relu(x))
        #print(x.shape)
        x = self.dropout1(x)
        #print(x.shape)
        x = self.layers1(x)
        #print(x.shape)
        x = self.layers2(x)
        #print(x.shape)
        x = self.layers3(x)
        #print(x.shape)
        x = self.layers4(x)
        #print(x.shape)
        x = self.pool1(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        #print('shapesfixed')
        x = self.linear(x)
        #print(x.shape)
        #print('shapesfixed')
        return x
        
net = ResNet()
net.cuda()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
#optimizer = optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(100):  

    running_loss = 0.0
    train_accu = []
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        batch_size = len(inputs)
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        #print('input_shape',inputs.shape)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prediction = outputs.data.max(1)[1]
        accuracy = ( float( prediction.eq(labels.data).sum() ) /float(batch_size))*100.0
        train_accu.append(accuracy)

    accuracy_epoch = np.mean(train_accu)
    print(epoch, accuracy_epoch)

    if (epoch%5==4):
        correct = 0
        total = 0
        for data in testloader:
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #correct += (predicted == labels).sum().item()
            correct+=predicted.eq(labels.data).sum()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100.0 * float(correct) / float(total)))



print('Finished Training')





