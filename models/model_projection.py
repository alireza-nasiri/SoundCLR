import torch
import torch.nn as nn
import torch.nn.functional as F

import config



# CUDA for PyTorch
#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:2" if use_cuda else "cpu")


class ProjectionModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc = nn.Linear(2048, 64)
		#self.fc = nn.Linear(1920, 64)
        
        
	def forward(self, x):
		x = self.fc(x)
		return x
    
    
    
