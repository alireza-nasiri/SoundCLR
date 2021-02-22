import torch
import torch.nn as nn
import torch.nn.functional as F

import config



class ProjectionModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc = nn.Linear(2048, 64)
        
        
	def forward(self, x):
		x = self.fc(x)
		
		return x
    
    
    
