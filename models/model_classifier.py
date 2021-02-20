import torch
import torch.nn as nn
import torch.nn.functional as F

import config



# CUDA for PyTorch
#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:2" if use_cuda else "cpu")


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, config.class_numbers)
        
        
        
        
    def forward(self, x):
        x = self.fc1(x)
        
        
        return x
    
    
    
