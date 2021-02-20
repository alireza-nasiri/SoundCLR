import torch
import torch.nn as nn

import loss as contrastive_loss
import config

class HybridLoss(nn.Module):
	def __init__(self, weights):
		super(HybridLoss, self).__init__()
		self.contrastive_loss = contrastive_loss.SupConLoss(weights=weights)
		self.class_weights = weights
    
	def cross_entropy_one_hot(self, input, target):
		_, labels = target.max(dim=1)
		return nn.CrossEntropyLoss(weight=self.class_weights)(input, labels)
		
    
	def forward(self, y_proj, y_pred, label, label_vec):
        
		loss1 = self.contrastive_loss(y_proj.unsqueeze(1), label.squeeze(1))
		loss2 = self.cross_entropy_one_hot(y_pred, label_vec)
        
		return loss1 * config.alpha, loss2 * (1 - config.alpha)
