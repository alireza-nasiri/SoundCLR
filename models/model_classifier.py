import torch
import torch.nn as nn
import torch.nn.functional as F

import config



class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2048, config.class_numbers)
        
        
        
        
    def forward(self, x):
        x = self.fc(x)
        
        return x
    
    
    
