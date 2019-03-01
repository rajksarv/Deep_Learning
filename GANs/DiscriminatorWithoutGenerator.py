import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)

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

model =  discriminator()
#model.load_state_dict(torch.load('parameters.ckpt'))
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

batch_size = 64
learning_rate = 0.0001
for epoch in range(100):  

    running_loss = 0.0
    train_accu = []
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue
        # get the inputs
        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        output = model(X_train_batch)
        optimizer.zero_grad()
        loss = criterion(output, Y_train_batch)
        loss.backward()
        if(epoch>6):
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if(state['step']>=1024):
                        state['step'] = 1000
        optimizer.step()
        prediction = output.data.max(1)[1]
        accuracy = ( float( prediction.eq(Y_train_batch.data).sum() ) /float(batch_size))*100.0
        train_accu.append(accuracy)

    accuracy_epoch = np.mean(train_accu)
    print(epoch, accuracy_epoch)
    
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0

    if (epoch%10==9):
        correct = 0
        total = 0
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch = Variable(X_test_batch).cuda()
            Y_test_batch = Variable(Y_test_batch).cuda()
            output = model(X_test_batch)
            _, predicted = torch.max(output.data, 1)
            total += Y_test_batch.size(0)
            #correct += (predicted == labels).sum().item()
            correct+=predicted.eq(Y_test_batch.data).sum()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100.0 * float(correct) / float(total)))
        torch.save(model.state_dict(),'parameters.ckpt')

