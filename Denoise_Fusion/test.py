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
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
from skimage import metrics
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import pickle
from utils import *
from Network import *
import seaborn as sns
from tqdm import tqdm
from scipy import ndimage
import argparse

######## cude setting######
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("Using device {}".format(device))

###### random_seed ##########
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

######parser default##########
parser = argparse.ArgumentParser(description="ensemble_train_and_inference")
parser.add_argument("--color_mode", type=str, default='gray', help='Grayscale (gray) or color (color) model')
parser.add_argument("--noise_std_values",nargs='+', type=int, default=[50,40,30,20,10], help='the noise level list')
parser.add_argument("--manipulation_mode", type=str, default="Joint", help='manipulation_mode,choose SM(manipulation in Spatial domain) or FM(manipulation in Frequency domain) or Joint')
parser.add_argument("--test_path", type=str, default='./data/images/test', help='dataset')
parser.add_argument("--ensemble_method", type=str, default='F', help='choose S(spatial position attention) or C(channel attention) or F(Fusion),S and C are just used for gray models')
parser.add_argument("--denoise_net", type=str, default='DnCNN', help='choose the denoised pre-trained model')
parser.add_argument("--noise_mode", type=str, default='normal_noise', help='choose normal_noise or varying noise')
parser.add_argument("--img_size", type=int, default=300, help='resize image size')
opt = parser.parse_args()

def main():
    test_img=read_clean_img(opt.test_path,color_mode=opt.color_mode,img_size=opt.img_size)
    if opt.color_mode=='gray':
        in_channels=1
    elif opt.color_mode=='color':
        in_channels=3
        
    if opt.manipulation_mode=='SM':
        mode_list=[0,1,2,3,4,5,6,7]
    elif opt.manipulation_mode=='FM':
        mode_list=[0,8,9,10,11,12]
    elif opt.manipulation_mode=='Joint':
        mode_list=[0,1,2,3,4,5,6,7,8,9,10,11,12]
        
    print("the denoise net is %s, "%str(opt.denoise_net),"the ensemble method is %s, "%str(opt.ensemble_method),"the manipulation mode is %s."%str(opt.manipulation_mode))
    print("Now,show the average PSNR of test datasets for different modes:")
    _,baseline_test_psnr,baseline_test_ssim=data_aug_denoise(test_img,opt.noise_std_values,[0],opt.denoise_net,opt.noise_mode)
    test_data,test_psnr,test_ssim=data_aug_denoise(test_img,opt.noise_std_values,mode_list,opt.denoise_net,opt.noise_mode)
    print(np.mean(test_psnr,axis=0))
    
    print("the PSNR of simple ensemble method for test set:",simple_ensemble(test_data,test_img,opt.noise_std_values))
    for i in range(len(opt.noise_std_values)):
        test_loader=prepare_dataloader(i,test_data,test_img,100,shuffle=False,patch_mode=False,win=50,stride=50)
        if opt.ensemble_method =='F':
            model=Ensemble_fusion(len(mode_list),in_channels).cuda()
        elif opt.ensemble_method =='S':
            model=Spatial_attention(len(mode_list)).cuda()
        elif opt.ensemble_method =='C':
            model=Channel_attention(len(mode_list)).cuda()
        model.load_state_dict(torch.load(os.path.join("./saved_models",str(opt.denoise_net),str(opt.ensemble_method),str(opt.manipulation_mode),'net_%d.pth'%(opt.noise_std_values[i]))))
        model.eval()
        criterion=nn.MSELoss()
        criterion.cuda()

        
        with torch.no_grad():
            test_loss,test_psnr,test_ssim,test_out=ensemble_evaluate(model,test_loader,criterion)
            
            print("the noise level is:",opt.noise_std_values[i])
            print("the PSNR of test_data_set at baseline model:",np.mean(baseline_test_psnr,axis=0)[i])
            print("the PSNR of test_data_set after network",test_psnr)
            
if __name__ == "__main__":
    main()

            
    
 
    

    
    
