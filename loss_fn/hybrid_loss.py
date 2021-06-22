import torch
import torch.nn as nn

from loss_fn import contrastive_loss
import config

class HybridLoss(nn.Module):
	def __init__(self, alpha=0.5, temperature=0.07):
		super(HybridLoss, self).__init__()
		self.contrastive_loss = contrastive_loss.SupConLoss(temperature)
		self.alpha = alpha
    
	def cross_entropy_one_hot(self, input, target):
		_, labels = target.max(dim=1)
		return nn.CrossEntropyLoss()(input, labels)
		
    
	def forward(self, y_proj, y_pred, label, label_vec):
        
		contrastiveLoss = self.contrastive_loss(y_proj.unsqueeze(1), label.squeeze(1))
		entropyLoss = self.cross_entropy_one_hot(y_pred, label_vec)
        
		return contrastiveLoss * self.alpha, entropyLoss * (1 - self.alpha)
