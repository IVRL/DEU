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
from Dataset import *


np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']

########read image########

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def read_clean_img(file_path,color_mode='gray',img_size=300):
    '''
    read the clean ing from fil_path,resize the image;
    if the color_mode='gray',it will get size H*W*1
    if the color_mode='color',it will get H*W*C
    '''
    
    files_input = glob.glob(file_path+'/*')
    files_input.sort()
    
    clean_img=[]
    
    for file_idx in range(len(files_input)):
        if is_image_file(files_input[file_idx]):
            if color_mode=='gray':
                img = cv2.imread(files_input[file_idx], cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img,(img_size,img_size))
                img = np.expand_dims(img, axis=2)
                clean_img.append(img)
                
    
            elif color_mode=='color':
                img = cv2.imread(files_input[file_idx], cv2.IMREAD_COLOR)
                img = cv2.resize(img,(img_size,img_size))
                clean_img.append(img)
        else:
            continue
    clean_img=np.array(clean_img)
    return clean_img

   

###### image type transformation#########

def uint2single(img):
    # do the normalization of img 
    return np.float32(img/255.)

def single2tensor4(img):
    #make the image become the 1*C*H*W
    img=np.transpose(img,(2,0,1))
    return torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0)

def tensor2uint(img):
    # make the iamge from tensor to uint8
    img = img.data.squeeze(0).float().clamp_(0, 1).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return np.uint8((img*255.0).round())

###### augmentation ########
def augment_mode_choose(img, mode=0,radius_perc=0.1):
    #choose the mode
    if mode==0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    elif mode == 8:
        return DCT_mask(img,radius_perc=0.1,branch=1)
    elif mode == 9:
        return DCT_mask(img,radius_perc=0.3,branch=1)
    elif mode ==10:
        return DCT_mask(img,radius_perc=0.5,branch=1)
    elif mode ==11:
         return DCT_mask(img,radius_perc=0.4,branch=0)
    elif mode ==12:
         return DCT_mask(img,radius_perc=0.8,branch=0)
        
###### noise type############
        
def add_white_gaussian_noise(mean,sigma,img_size,noise_mode='normal_noise'):
    # generate gaussian noise with mean and var.
    # need add the varing the noise after
    if noise_mode=='normal_noise':
        gauss_noise=np.random.normal(mean,sigma,img_size)
    
    #need varing noise later
    elif noise_mode=='varying_noise':
        gauss_noise=create_varying_noise(mean,img_size)
    
    return gauss_noise

def create_varying_noise(mean,img_size):
    noise_std_min=5
    noise_std_max=55
    
    noise=np.zeros(img_size)
    for i in range(img_size[0]):
        std=noise_std_min+(noise_std_max-noise_std_min)*(i/(img_size[0]-1))
        noise[:,:,:][i]=np.random.normal(0,std,(img_size[0],1))
        
    return noise


def inverse_aug(img,mode=0):
    # in order to make the rotation(flip) imgs back
    if mode==0:
        return img
    if mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=1)
    elif mode == 4:
        return np.flipud(np.rot90(img,k=2))
    elif mode == 5:
        return np.rot90(img,k=3)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    else:
        return img
    
    
def DCT_mask(img_s,radius_perc,branch):
    #Do the DCT_mask
    img=np.copy(img_s)
    (w,h,c)=np.shape(img)
    mask= get_dct_mask(w,h,radius_perc,branch)
    if c==1:
        img_dct=dct(dct(img[:,:,0], axis=0, norm='ortho'), axis=1, norm='ortho')
        img_dct=img_dct*mask
        img[:,:,0]=idct(idct(img_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
    elif c==3:
        img0_dct = dct(dct(img[:,:,0], axis=0, norm='ortho'), axis=1, norm='ortho')
        img1_dct = dct(dct(img[:,:,1], axis=0, norm='ortho'), axis=1, norm='ortho')
        img2_dct = dct(dct(img[:,:,2], axis=0, norm='ortho'), axis=1, norm='ortho')
        
        img0_dct = img0_dct*mask
        img1_dct = img1_dct*mask
        img2_dct = img2_dct*mask
        
        img[:,:,0]= idct(idct(img0_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        img[:,:,1]= idct(idct(img1_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        img[:,:,2]= idct(idct(img2_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        
    return img

def get_dct_mask(w,h,radius_perc=-1,branch=-1):
    '''
    branch 0 is the area after p will be masked
    branch 1 is the area between p and p+0.1
    '''
    if radius_perc < 0:
        raise Exception('radius_perc must be positive.')
    radius = np.sqrt(w*w+h*h)
    center_radius = radius_perc * radius
    
    X, Y = np.meshgrid(np.linspace(0,h-1,h), np.linspace(0,w-1,w))
    D = np.sqrt(X*X+Y*Y)
    
   
    a1 = center_radius
    a2 = radius
    a3=radius*(radius_perc+0.1)
    mask = np.ones((w,h))
    
    if branch ==0:
        mask[(D>=a1)&(D<=a2)] = 0
    elif branch==1:
        mask[(D>=a1)&(D<=a3)] = 0
    else:
        raise Exception('branch should be in 1 or 0.')
        
        
    return mask


def data_aug_denoise(img,std_values,mode_list,denoise_net,noise_mode):
    
    #the noise_mode:varying according to the row of image or special noise std.
    #denoise_net:the pre_trained denoise model
    #mode_list: type number of augmentation
    #std_values: for special noise, one special std value.
    
    #output the psnr of denoised images and augmented images themselves
    
    np.random.seed(0)
    img_size=img[0].shape
    (w,h,c)=img[0].shape
    noise_mean=0
    
    pic=[]
    psnr_results=np.zeros((len(img),len(std_values),len(mode_list)))
    ssim_results=np.zeros((len(img),len(std_values),len(mode_list)))
    if denoise_net=='DnCNN':
        net=DnCNN_RL(channels=1, num_of_layers=17)
    elif denoise_net=='MemNet':
        net=MemNet(in_channels=1, channels=20, num_memblock=6, num_resblock=4)
    elif denoise_net=='RIDNet':
        net=RIDNET(in_channels=1)
    elif denoise_net=='DnCNN_color':
        net=DnCNN_RL(channels=3, num_of_layers=17)
        
    
    model = nn.DataParallel(net).cuda()
    model.load_state_dict(torch.load(os.path.join("./model",denoise_net,'net.pth' )))
    model.eval()
    
    for noise_idx,noise_std in enumerate(std_values):
        np.random.seed(0)
        for idx in range(img.shape[0]):
            noise=add_white_gaussian_noise(noise_mean,noise_std,img_size,noise_mode)
            noisy_img=img[idx]+noise
            for mode_idx in range(len(mode_list)):
                img_aug=augment_mode_choose(noisy_img,mode_list[mode_idx])
                img_aug=uint2single(img_aug)
                img_aug=single2tensor4(img_aug)
                
                INoisy = Variable(img_aug.cuda())
                INoisy = torch.clamp(INoisy, 0., 1.)
                
                with torch.no_grad():
                   
                    NoiseNetwork=model(INoisy)
                    NoiseNetwork=NoiseNetwork
                    INetwork = tensor2uint(NoiseNetwork)

                    INetwork = inverse_aug(INetwork,mode_list[mode_idx])

                    pic.append(INetwork)
                    psnr_results[idx][noise_idx][mode_idx]=metrics.peak_signal_noise_ratio(INetwork,img[idx],data_range=255.)
                    ssim_results[idx][noise_idx][mode_idx]=metrics.structural_similarity(INetwork,img[idx],data_range=255.,multichannel=True)
                    
    pic=np.array(pic)
    pic=pic.reshape((len(std_values),img.shape[0],len(mode_list),w,h,c),order='C')           
    return pic,psnr_results,ssim_results



def psnr_ini(a,b):
    c=0
    for i in range(a.shape[0]):
        c+=metrics.peak_signal_noise_ratio(a[i],b[i]/255,data_range=1.)
    return c/a.shape[0]

def ssim_ini(a,b):
    c=0
    for i in range(a.shape[0]):
        c+=metrics.structural_similarity(a[i],b[i]/255,data_range=1.)
    return c/a.shape[0]
    


def batch_PSNR(img, imclean, data_range):
    PSNR = 0
    for i in range(img.shape[0]):
        PSNR += metrics.peak_signal_noise_ratio(imclean[i,:,:,:], img[i,:,:,:], data_range=data_range)
    if math.isnan(PSNR):
        import pdb; pdb.set_trace()
    return (PSNR/img.shape[0])

def batch_SSIM(img, imclean, data_range):
    SSIM=0
    for i in range(img.shape[0]):
        SSIM += metrics.structural_similarity(imclean[i,:,:,:], img[i,:,:,:], data_range=data_range,multichannel=True)
    if math.isnan(SSIM):
        import pdb; pdb.set_trace()
    return (SSIM/img.shape[0])



def DCT_transform(imgs):
    #do the DCT transform
    img=imgs.copy()
    dct_img=np.zeros(img.shape)

    if img.ndim==4:
        for i in range(img.shape[0]):
            for k in range(img.shape[1]):
                dct_img[i][k,:,:]=dct(dct(img[i][k,:,:], axis=0, norm='ortho'), axis=1, norm='ortho')
    elif img.ndim==5:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[2]):
                    dct_img[i][j][k,:,:]=dct(dct(img[i][j][k,:,:], axis=0, norm='ortho'), axis=1, norm='ortho')
    
    return dct_img


def IDCT_transform(imgs):
    #do the inverse DCT transform
    img=imgs.copy()
    idct_img=np.zeros(img.shape)
    if img.ndim==4:
        for i in range(img.shape[0]):
            for k in range(img.shape[3]):
                idct_img[i][:,:,k]=idct(idct(img[i][:,:,k], axis=0, norm='ortho'), axis=1, norm='ortho')
    elif img.ndim==5:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(img.shape[4]):
                    idct_img[i][j][:,:,k]=idct(idct(img[i][j][:,:,k], axis=0, norm='ortho'), axis=1, norm='ortho')
    return idct_img


def simple_ensemble(aug_img,img,noise_std):
    sim_ensemble=[]
    for i in range(len(noise_std)):
        aug_imgs=aug_img[i,:,:,:,:]
        aug_imgs=normalization(aug_imgs)
        imgs=normalization(img)
        psnr=0
        ensembled_img=np.mean(aug_imgs,axis=1)
        for i in range (imgs.shape[0]):
            psnr+=metrics.peak_signal_noise_ratio(ensembled_img[i,:,:,:],imgs[i,:,:,:],data_range=1.)
        psnr=psnr/img.shape[0]
        sim_ensemble.append(psnr)
    return sim_ensemble
        
            
##### Network usage###########     
        
def ensemble_evaluate(model,data_loader,criterion):

    #### evaluate the performance of network 
    loss=0
    psnr=0
    ssim=0
    count=0
    for ensemble_data,target in data_loader:
        count=count+1
        ensemble_data=Variable(ensemble_data).cuda()
        target=Variable(target).cuda()
        output=model(ensemble_data)
        loss+=criterion(output,target)
        output=output.data.cpu().numpy().astype(np.float32).clip(0.,1.)
        target=target.data.cpu().numpy().astype(np.float32).clip(0.,1.)
        output=np.transpose(output,(0,2,3,1))
        target=np.transpose(target,(0,2,3,1))
        psnr+=batch_PSNR(output,target,data_range=1.)
        ssim+=batch_SSIM(output,target,data_range=1.)
    psnr=psnr/count
    ssim=ssim/count
    return loss,psnr,ssim,output


def train_ensemble(model_dir,noise_std,train_loader,test_loader,model,optimizer,criterion,pbar=True, epochs=100,gamma=0.5):
    ####### train the ensemble network
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)
    for epoch in range(epochs):
        for i,data in enumerate(train_loader):
            ensemble_input,ensemble_target=data
            
            ensemble_input=Variable(ensemble_input).cuda()
            ensemble_target=Variable(ensemble_target).cuda()
    
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
        
            output=model(ensemble_input)
            loss=criterion(output,ensemble_target)
        
            loss.backward()
            optimizer.step()
            
            if gamma != 0 and epoch > 50:
                scheduler.step()
            
        if (epoch+1)%5==0:
            print('Epoch[{}/{}],loss:{:.6f}'.format(epoch+1,epochs,loss.item()))
        
            if pbar:
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(ensemble_target.shape[0])
    torch.save(model.state_dict(), os.path.join(model_dir, 'net_%d.pth' % (noise_std)) )

    model.eval()
    with torch.no_grad():
        train_loss,train_psnr,train_ssim,_=ensemble_evaluate(model,train_loader,criterion)
        test_loss,test_psnr,test_ssim,test_out=ensemble_evaluate(model,test_loader,criterion)
    
    return train_loss,train_psnr,train_ssim,test_loss,test_psnr,test_ssim,test_out