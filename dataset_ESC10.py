import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
#import torchvision.transforms as transforms
import transforms
import torchvision

import os
import numpy as np
import imageio
import random
import collections
import csv
import librosa

import config


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



class MyDataset(data.Dataset):
    
    def __init__(self, train=True):
        self.root = '/work/anasiri/CPC_audio_spectrograms/data/waves_ESC50/'
        self.train = train
        
        #getting name of all files inside the all of the train_folds
        temp = open('data/ESC10_file_names.txt', 'r').read().split('\n')
        temp.sort()
        self.file_names = []
        if train:
            for i in range(len(temp)):
                if int(temp[i].split('-')[0]) in config.train_folds:
                    self.file_names.append(temp[i])
        else:
            for i in range(len(temp)):
                if int(temp[i].split('-')[0]) in config.test_fold:
                    self.file_names.append(temp[i])
        
        if self.train:
            self.wave_transforms = torchvision.transforms.Compose([ transforms.ToTensor1D(), 
                                                              transforms.RandomScale(max_scale = 1.25), 
                                                              transforms.RandomPadding(out_len = 220500),
                                                              transforms.RandomCrop(out_len = 220500)])
            
            
            self.spec_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                  transforms.FrequencyMask(),
                                                                   transforms.TimeMask()])
            
            
        else: #for test
            self.wave_transforms = torchvision.transforms.Compose([ transforms.ToTensor1D(),
                                                              transforms.RandomPadding(out_len = 220500),
                                                             transforms.RandomCrop(out_len = 220500)])
        
            self.spec_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor() ])

    
    def __len__(self):
        return len(self.file_names) # * 2
    
    

    def __getitem__(self, index):
        file_name = self.file_names[index ]  #  % len(self.file_names)
        path = self.root + file_name
        wave, rate = librosa.load(path, sr=44100)
        
        
        temp = file_name.split('.')[0]
        class_id = int(temp.split('-')[-1])
        class_id = config.ESC10_classIds.index(class_id)
        
        if wave.ndim == 1:
            wave = wave[:, np.newaxis]

        if np.abs(wave.max()) > 1.0:
            wave = transforms.scale(wave, wave.min(), wave.max(), -1.0, 1.0)
        wave = wave.T * 32768.0
        
        # Remove silent sections
        start = wave.nonzero()[1].min()
        end = wave.nonzero()[1].max()
        wave = wave[:, start: end + 1]  
        
        wave_copy = np.copy(wave)
        wave_copy = self.wave_transforms(wave_copy)
        wave_copy.squeeze_(0)
        
        s = librosa.feature.melspectrogram(wave_copy.numpy(), sr=44100, n_mels=128, n_fft=1024, hop_length=512) 
        log_s = librosa.power_to_db(s, ref=np.max)
        
        log_s = self.spec_transforms(log_s)
        
        '''
        spec = torch.empty(0, 128, 431)
        for i in range(3):
            
            log_s_augmented = self.spec_transforms(log_s)
            
            spec = torch.cat((spec, log_s_augmented), dim=0)
        '''
        
        #creating 3 channels by copying log_s1 3 times 
        spec = torch.cat((log_s, log_s, log_s), dim=0)
        
        
        
        #if not self.train: # for the test loader
        return file_name, spec, class_id
        
    
    
    
    
    
    
    
    
    

def create_generators():
    
    train_dataset = MyDataset(train=True)
    test_dataset = MyDataset(train=False)
    
    train_loader = data.DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True, num_workers=1,drop_last=False)
    
    test_loader = data.DataLoader(test_dataset, batch_size = config.batch_size, shuffle=True, num_workers = 1,drop_last=False)
    
    return train_loader, test_loader
    
    
    '''
    overall_dataset = MyDataset(train=True)
    
    
    
    train_dataset, val_dataset = data.random_split(overall_dataset, [int(len(overall_dataset)*0.8), 
                                                                         len(overall_dataset)-int(len(overall_dataset)*0.8)])
    
    train_loader = data.DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True, num_workers=1,drop_last=True)
    
    val_loader = data.DataLoader(val_dataset, batch_size = config.batch_size, shuffle=True, num_workers = 1,drop_last=True)
    
    return train_loader, val_loader
    
    '''

    '''
    # Creating data indices for training and validation splits:
    dataset_size = len(overall_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.15 * dataset_size))
    
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    return train_loader, validation_loader
    '''






