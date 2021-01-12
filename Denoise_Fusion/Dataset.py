import os
import os.path
import numpy as np
import random
import torch
import cv2
import glob
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
from scipy.fftpack import dct,idct
from torchsummary import summary
from torchvision import transforms, datasets
import torch
import torchvision
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from skimage import metrics
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import pickle
from Network import *
from utils import *

####prepare for the dataset

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True


def normalization(x):
    y=np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[i][j]=np.float32(x[i][j])/255
    return y



### dataset and dataloader######
def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]

    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):

            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win,TotalPatNum])


def dataset_patch_prepare(data_set,win,stride=1):
    total_patch=[]
    for i in range(data_set.shape[0]):
        patches=Im2Patch(data_set[i],win=win,stride=stride)
        patches=np.transpose(patches,(3,0,1,2))
        total_patch.append(patches)
    patch_array=np.array(total_patch)
    patch_out=np.reshape(patch_array,(patch_array.shape[0]*patch_array.shape[1],patch_array.shape[2],patch_array.shape[3],patch_array.shape[4]))
    return patch_out   

    
def ensemble_dataset_process(inputs,targets,mode='multi_channel'):
    if mode=='multi_channel':
        inputs=np.transpose(inputs,(0,1,4,2,3))
        inputs=np.reshape(inputs,(inputs.shape[0],inputs.shape[1]*inputs.shape[2],inputs.shape[3],inputs.shape[4]))
        targets=np.transpose(targets,(0,3,1,2))
                
    elif mode=='single_channel':
        targets=np.transpose(targets,(0,3,1,2))
        targets=np.repeat(targets,inputs.shape[1],axis=0)
        inputs=np.transpose(inputs,(0,1,4,2,3))
        inputs=np.reshape(inputs,(inputs.shape[0]*inputs.shape[1],inputs.shape[2],inputs.shape[3],inputs.shape[4]))
        
    return inputs,targets


class mydataset(Dataset):
    def __init__(self,data_x,data_y):
        self.inputs=data_x
        self.targets=data_y
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self,idx):
        inputs=self.inputs[idx]
        targets=self.targets[idx]
        return torch.from_numpy(inputs).type(torch.FloatTensor),torch.from_numpy(targets).type(torch.FloatTensor)



def prepare_dataloader(noise_idx,input_data,target_data,batch_size,shuffle,patch_mode,win,stride):
    input_datas=input_data[noise_idx,:,:,:,:]
    input_datas=normalization(input_datas)
    target_datas=normalization(target_data)
    
    train_in,train_ta=ensemble_dataset_process(input_datas,target_datas,mode='multi_channel')
    if patch_mode==True:
        train_in= dataset_patch_prepare(train_in,win=win,stride=stride)
        train_ta= dataset_patch_prepare(train_ta,win=win,stride=stride)
    else:
        train_in=train_in
        train_ta=train_ta
    
    train_dataset=mydataset(train_in,train_ta)
    train_loader=torch.utils.data.DataLoader(dataset=list(train_dataset), num_workers=0, batch_size=batch_size, shuffle=shuffle)
    return train_loader
