import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch.nn as nn

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)



#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def resnet18(pretrained = True) :
    model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained :
        model.load_state_dict(torch.utils.model_zoo.load_url(model_urls['resnet18'], model_dir ='./'))
    return model

model = resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)

model = model.cuda()


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


for epoch in range(100):  

    running_loss = 0.0
    train_accu = []
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
 
        US = nn.Upsample(scale_factor=7)
        inputs = US(inputs)
       
        batch_size = len(inputs)
        inputs = inputs.cuda()
        labels = Variable(labels).cuda()
        #print('input_shape',inputs.shape)
        # zero the parameter gradients
        optimizer_ft.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_ft.step()
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

            US = nn.Upsample(scale_factor=7)
            inputs = US(inputs)
       
            inputs, labels = inputs.cuda(), Variable(labels).cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #correct += (predicted == labels).sum().item()
            correct+=predicted.eq(labels.data).sum()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100.0 * float(correct) / float(total)))



print('Finished Training')





