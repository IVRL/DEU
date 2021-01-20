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
from Dataset import *
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
#parser.add_argument("--mode_list", nargs="+", type=int,default=[0,1,2,3,4,5,6,7], help='augmentation_mode,0-7 mean the filp and rotation,8-12 mean the DCT masking')
parser.add_argument("--manipulation_mode", type=str, default="Joint", help='manipulation_mode,choose SM(manipulation in Spatial domain) or FM(manipulation in Frequency domain) or Joint')
parser.add_argument("--train_path", type=str, default='./data/images/train', help='ensemble network train_path')
parser.add_argument("--test_path", type=str, default='./data/images/test', help='ensemble network test_path')
parser.add_argument("--ensemble_method", type=str, default='F', help='choose S(spatial position attention) or C(channel attention) or F(Fusion),S and C are just used for gray models ')
parser.add_argument("--denoise_net", type=str, default='DnCNN', help='choose the denoised pre-trained model')
parser.add_argument("--noise_mode", type=str, default='normal_noise', help='choose normal_noise or varying noise')
parser.add_argument("--img_size", type=int, default=300, help='resize image size')
parser.add_argument("--lr", type=float, default=0.01, help='learning rate')
opt = parser.parse_args()

def train_spa_data(model_dir,train_data,test_data,train_img,test_img,noise_std_values,mode_list,baseline_train_psnr,baseline_test_psnr,in_channels):
    test_out_results=[]
    for i in range(len(noise_std_values)):
        train_loader=prepare_dataloader(i,train_data,train_img,100,shuffle=True,patch_mode=True,win=50,stride=50)
        test_loader=prepare_dataloader(i,test_data,test_img,100,shuffle=False,patch_mode=False,win=50,stride=50)
        if opt.ensemble_method =='F':
            model_net=Ensemble_fusion(len(mode_list),in_channels).cuda()
        elif opt.ensemble_method =='S':
            model_net=Spatial_attention(len(mode_list)).cuda()
        elif opt.ensemble_method =='C':
            model_net=Channel_attention(len(mode_list)).cuda()
        
        model_net.apply(weights_init_kaiming)
        criterion=nn.MSELoss()
        criterion.cuda()
        optimizer=optim.Adam(model_net.parameters(),lr=opt.lr)
           
        print("the train process of noise level %d:"%noise_std_values[i])
        train_loss,train_psnr,train_ssim,test_loss,test_psnr,test_ssim,test_out=\
        train_ensemble(model_dir,noise_std_values[i],train_loader,test_loader,model_net,optimizer,criterion,False)
        
        test_out_results.append(test_out)
        print("the PSNR of train_data_set at baseline model:",np.mean(baseline_train_psnr,axis=0)[i])
        print("the PSNR of train_data_set after network:",train_psnr)
        print("the SSIM of train_data_set after network:",train_ssim)
        print("the PSNR of test_data_set at baseline model:",np.mean(baseline_test_psnr,axis=0)[i])
        print("the PSNR of test_data_set after network",test_psnr)
        print("the SSIM of test_data_set after network:",test_ssim)
        print("exam",psnr_ini(test_out,test_img))
    test_out_results=np.array(test_out_results)
    return test_out_results

def main():
    #get the train images and test images
    model_dir = os.path.join('saved_models', str(opt.denoise_net), str(opt.ensemble_method), str(opt.manipulation_mode))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    #torch.save(model.state_dict(), os.path.join(model_dir, 'net_%d.pth' % (epoch)) )
    
    if opt.color_mode=='gray':
        in_channels=1
    elif opt.color_mode=='color':
        in_channels=3
    
    train_img=read_clean_img(opt.train_path,color_mode=opt.color_mode,img_size=opt.img_size)
    test_img=read_clean_img(opt.test_path,color_mode=opt.color_mode,img_size=opt.img_size)
    
    if opt.manipulation_mode=='SM':
        mode_list=[0,1,2,3,4,5,6,7]
    elif opt.manipulation_mode=='FM':
        mode_list=[0,8,9,10,11,12]
    elif opt.manipulation_mode=='Joint':
        mode_list=[0,1,2,3,4,5,6,7,8,9,10,11,12]
    
    print("the denoise net is %s, "%str(opt.denoise_net),"the ensemble method is %s, "%str(opt.ensemble_method),"the manipulation mode is %s."%str(opt.manipulation_mode))
    #get the psnr results for denoised images for different mode lists
    print("Now,show the average PSNR of train datasets for different modes:")
    _,baseline_train_psnr,baseline_train_ssim=data_aug_denoise(train_img,opt.noise_std_values,[0],opt.denoise_net,opt.noise_mode)
    train_data,train_psnr,train_ssim=data_aug_denoise(train_img,opt.noise_std_values,mode_list,opt.denoise_net,opt.noise_mode)
    print(np.mean(train_psnr,axis=0))
    print("Now,show the average SSIM of train datasets for different modes:")
    print(np.mean(train_ssim,axis=0))
    
    print("Now,show the average PSNR of test datasets for different modes:")
    _,baseline_test_psnr,baseline_test_ssim=data_aug_denoise(test_img,opt.noise_std_values,[0],opt.denoise_net,opt.noise_mode)
    test_data,test_psnr,test_ssim=data_aug_denoise(test_img,opt.noise_std_values,mode_list,opt.denoise_net,opt.noise_mode)
    print(np.mean(test_psnr,axis=0))
    print("Now,show the average SSIM of test datasets for different modes:")
    print(np.mean(test_ssim,axis=0))
    
    print("the PSNR of simple ensemble method for train set:",simple_ensemble(train_data,train_img,opt.noise_std_values))
    print("the PSNR of simple ensemble method for test set:",simple_ensemble(test_data,test_img,opt.noise_std_values))

    # prepare the data_loader
    test_out_results=train_spa_data(model_dir,train_data,test_data,train_img,test_img,\
                                                opt.noise_std_values,mode_list,baseline_train_psnr,baseline_test_psnr,in_channels)
       
        
    test_results_dir= os.path.join('saved_test_results',str(opt.denoise_net),'net_%s'%str(opt.ensemble_method),str(opt.manipulation_mode))
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)
        
    test_results_pk=open(os.path.join(test_results_dir,'test_data.pickle'),'wb')
    pickle.dump(test_out_results,test_results_pk)
    test_results_pk.close()
        
if __name__ == "__main__":
    main()
    
    


