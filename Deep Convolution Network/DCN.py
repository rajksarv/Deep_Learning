import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=4, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,64,kernel_size=4, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2) 
        self.conv3 = nn.Conv2d(64,64,kernel_size=4, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,64,kernel_size=4, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.2)
        self.conv5 = nn.Conv2d(64,64,kernel_size=4, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=0)
        self.dropout3 = nn.Dropout(0.2)
        self.conv7 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64,64,kernel_size=3, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(64)
        self.dropout4 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 10)


    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.pool2(F.relu(self.conv4(x)))
        x = self.dropout2(x)
        x = self.bn3(F.relu(self.conv5(x)))
        x = (F.relu(self.conv6(x)))
        x = self.dropout3(x)
        x = self.bn4(F.relu(self.conv7(x)))
        x = self.bn5(F.relu(self.conv8(x)))
        x = self.dropout4(x)
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


net = Net()
net.cuda()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

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
        if(i>6):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if(state['step']>=1024):
                        state['step'] = 1000
        optimizer.step()
        prediction = outputs.data.max(1)[1]
        accuracy = ( float( prediction.eq(labels.data).sum() ) /float(batch_size))*100.0
        train_accu.append(accuracy)

    accuracy_epoch = np.mean(train_accu)
    print(epoch, accuracy_epoch)

print('Finished Training')

#Test Accuracy

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

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100.0 * float(correct) / float(total)))

