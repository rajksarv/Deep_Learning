import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms                                                                                                                                                               
from torch.autograd import Variable
import torch.distributed as dist
import torchvision

from helperFunctions import getUCF101
from helperFunctions import loadSequence

import h5py
import cv2

from multiprocessing import Pool

IMAGE_SIZE = 224
#sample_size = IMAGE_SIZE                                                                                                                                                                                   
NUM_CLASSES = 101
batch_size = 100
lr = 0.0001
num_of_epochs = 10


data_directory = '/projects/training/bauh/AR/'
class_list, train, test = getUCF101(base_directory = data_directory)
model_sing = torch.load('single_frame.model')
model = torch.load('3d_resnet.model')
model_sing.cuda()
model.cuda()

##### save predictions directory                                                                                                                                                                            
prediction_directory = 'UCF-101-predictions/'
if not os.path.exists(prediction_directory):
    os.makedirs(prediction_directory)
for label in class_list:
    if not os.path.exists(prediction_directory+label+'/'):
        os.makedirs(prediction_directory+label+'/')

acc_top1 = 0.0
acc_top5 = 0.0
acc_top10 = 0.0
confusion_matrix = np.zeros((NUM_CLASSES,NUM_CLASSES),dtype=np.float32)
random_indices = np.random.permutation(len(test[0]))
mean = np.asarray([0.485, 0.456, 0.406],np.float32)
std = np.asarray([0.229, 0.224, 0.225],np.float32)
model.eval()

for i in range(len(test[0])):

    t1 = time.time()

    index = random_indices[i]
    filename = test[0][index]
    filename = filename.replace('.avi','.hdf5')
    filename = filename.replace('UCF-101','UCF-101-hdf5')

    h = h5py.File(filename,'r')
    nFrames = len(h['video'])
    num_batches = int(len(h['video'])/16)
    sing_data = np.zeros((nFrames,3,IMAGE_SIZE,IMAGE_SIZE),dtype=np.float32)
    data = np.zeros((1,3,16,IMAGE_SIZE,IMAGE_SIZE),dtype=np.float32)
    prediction = np.zeros((num_batches,NUM_CLASSES),dtype=np.float32)
    prediction_sing = np.zeros((nFrames,NUM_CLASSES),dtype=np.float32)


    for j in range(nFrames):
        frame = h['video'][j]
        frame = frame.astype(np.float32)
        frame = cv2.resize(frame,(IMAGE_SIZE,IMAGE_SIZE))
        frame = frame/255.0
        frame = (frame - mean)/std
        frame = frame.transpose(2,0,1)
        sing_data[j,:,:,:] = frame
    loop_i = list(range(0,nFrames,200))
    loop_i.append(nFrames)

    for j in range(len(loop_i)-1):
        data_batch = sing_data[loop_i[j]:loop_i[j+1]]

    with torch.no_grad():
            x_sing = np.asarray(data_batch,dtype=np.float32)
            x_sing = Variable(torch.FloatTensor(x_sing)).cuda().contiguous()

            output_sing = model_sing(x_sing)

    prediction_sing[loop_i[j]:loop_i[j+1]] = output_sing.cpu().numpy()

    # softmax                                                                                                                                                                                           
    for j in range(prediction_sing.shape[0]):
        prediction_sing[j] = np.exp(prediction_sing[j])/np.sum(np.exp(prediction_sing[j]))

    prediction_sing = np.mean(np.log(prediction_sing),axis=0)                                                                                                                                                        
    y = 0
    for j in range(num_batches):
        for k in range(16):
            frame = h['video'][y]
            frame = frame.astype(np.float32)
            frame = cv2.resize(frame,(IMAGE_SIZE,IMAGE_SIZE))
            frame = frame/255.0
            frame = (frame - mean)/std
            frame = frame.transpose(2,0,1)
            data[0,:,k,:,:] = frame
            #sing_data[j,:,:,:] = frame                                                                                                                                                                     
            y = y+1
        with torch.no_grad():
            x = np.asarray(data,dtype=np.float32)
            x = Variable(torch.FloatTensor(x)).cuda().contiguous()

            output = model(x)

    prediction[j,:] = output.cpu().numpy()
            #h.close()                                                                                                                                                                                      


    # softmax                                                                                                                                                                                               
    for j in range(prediction.shape[0]):
        prediction[j] = np.exp(prediction[j])/np.sum(np.exp(prediction[j]))

    prediction = np.mean(np.log(prediction),axis=0)
#argsort_pred = np.argsort(-prediction)[0:10]                                                                                                                                                               

    fin_predict = np.mean(np.array([prediction_sing,prediction]),axis=0)
    argsort_pred = np.argsort(-fin_predict)[0:10]

    label = test[1][index]
    confusion_matrix[label,argsort_pred[0]] += 1
    if(label==argsort_pred[0]):
        acc_top1 += 1.0
    if(np.any(argsort_pred[0:5]==label)):
        acc_top5 += 1.0
    if(np.any(argsort_pred[:]==label)):
        acc_top10 += 1.0

    print('i:%d nFrames:%d t:%f (%f,%f,%f)'
          % (i,nFrames,time.time()-t1,acc_top1/(i+1),acc_top5/(i+1), acc_top10/(i+1)))

number_of_examples = np.sum(confusion_matrix,axis=1)
for i in range(NUM_CLASSES):
    confusion_matrix[i,:] = confusion_matrix[i,:]/np.sum(confusion_matrix[i,:])

results = np.diag(confusion_matrix)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

for i in range(NUM_CLASSES):
    print(sorted_list[i],sorted_results[i],number_of_examples[indices[i]])

np.save('combined_confusion_matrix.npy',confusion_matrix)
pool_threads.close()
pool_threads.terminate()

