import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from utils import transforms
import torchvision

import os
import numpy as np
import imageio
import random
import collections
import csv
import librosa
import math

import config


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



class MyDataset(data.Dataset):
    
	def __init__(self, train=True):
		self.root = './data/US8K/audio/'
		self.train = train
        
		self.file_paths = [] #only includes the name of the fold and name of the file, like: 'fold2/4201-3-0-0.wav'
        
		if train:
			for f in config.train_folds:
				file_names = os.listdir(self.root + 'fold' + str(f) + '/' )
                
				for name in file_names:
					if name.split('.')[-1] == 'wav':
						self.file_paths.append('fold' + str(f) + '/' + name)
		else:
			file_names = os.listdir(self.root + 'fold' + str(config.test_fold[0]) + '/' )
			for name in file_names:
				if name.split('.')[-1] == 'wav':
					self.file_paths.append('fold' + str(config.test_fold[0]) + '/' + name)
        

		if self.train:
			self.wave_transforms = torchvision.transforms.Compose([ transforms.ToTensor1D(),
                                                                   transforms.RandomScale(max_scale = 1.25), 
                                                                  transforms.RandomPadding(out_len = 176400),
                                                                  transforms.RandomCrop(out_len = 176400)])
             
			self.spec_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                   transforms.FrequencyMask(max_width = config.freq_masks_width, numbers = config.freq_masks),
                                                                   transforms.TimeMask(max_width = config.time_masks_width, numbers = config.time_masks)])
            
            
		else: #for test
			self.wave_transforms = torchvision.transforms.Compose([ transforms.ToTensor1D(),
                                                                    transforms.RandomPadding(out_len = 176400),
                                                                    transforms.RandomCrop(out_len = 176400)])
            
			self.spec_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor() ])

    
	def __len__(self):
		return len(self.file_paths) 
    
    

	def __getitem__(self, index):
		file_path = self.file_paths[index ]  
		path = self.root + file_path
        
		wave, rate = librosa.load(path, sr=44100)

		if wave.ndim > 1:
			wave = (wave[0, : ] + wave[1, : ]) / 2        
        
		class_id = int(file_path.split('-')[1])
        
        
		if wave.ndim == 1:
			wave = wave[:, np.newaxis]
		
		# normalizing waves to [-1, 1]
		if np.abs(wave.max()) > 1.0:
			wave = transforms.scale(wave, wave.min(), wave.max(), -1.0, 1.0)
		wave = wave.T * 32768.0
        
		#Remove silent sections
		start = wave.nonzero()[1].min()
		end = wave.nonzero()[1].max()
		wave = wave[:, start: end + 1]  
        
		wave = self.wave_transforms(wave)
		wave.squeeze_(0)
        
		s = librosa.feature.melspectrogram(wave.numpy(), sr=rate, n_mels=128, n_fft=1024, hop_length=512) 

		log_s = librosa.power_to_db(s, ref=np.max)
        
		log_s = self.spec_transforms(log_s)
        
		#creating 3 channels by copying log_s1 3 times 
		spec = torch.cat((log_s, log_s, log_s), dim=0)
        	
		return file_path, spec, class_id 
    
 
  

def create_generators():
    
	train_dataset = MyDataset(train=True)
	test_dataset = MyDataset(train=False)
    
	train_loader = data.DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True, num_workers=10, drop_last=False)
    
	test_loader = data.DataLoader(test_dataset, batch_size = config.batch_size, shuffle=True, num_workers =10, drop_last=False)
    
	return train_loader, test_loader
    






