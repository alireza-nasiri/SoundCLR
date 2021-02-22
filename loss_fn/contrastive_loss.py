'''
This implementation is from https://github.com/HobbitLong/SupContrast with small modifications.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class SupConLoss(nn.Module):
	"""Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
	It also supports the unsupervised contrastive loss in SimCLR"""
	def __init__(self, temperature=0.07):
		super(SupConLoss, self).__init__()
		self.temperature = temperature
        
        
	def forward(self, features, labels=None, mask=None):
		"""Compute loss for model. If both `labels` and `mask` are None,
		it degenerates to SimCLR unsupervised loss:
		https://arxiv.org/pdf/2002.05709.pdf
		Args:
		features: hidden vector of shape [bsz, n_views, ...].
		labels: ground truth of shape [bsz].
		mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
		has the same class as sample i. Can be asymmetric.
		Returns:
		A loss scalar.
		"""
		device = (torch.device('cuda')
			if features.is_cuda
			else torch.device('cpu'))

		if len(features.shape) < 3:
			raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
		if len(features.shape) > 3:
			features = features.view(features.shape[0], features.shape[1], -1)

		batch_size = features.shape[0]
		if labels is not None and mask is not None:
			raise ValueError('Cannot define both `labels` and `mask`')
		elif labels is None and mask is None:
			mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            
		elif labels is not None:
			labels = labels.contiguous().view(-1, 1)
			if labels.shape[0] != batch_size:
				raise ValueError('Num of labels does not match num of features')
            
			#here mask is of shape [bsz, bsz] and is one for one for [i,j] where label[i]=label[j]
			mask = torch.eq(labels, labels.T).float().to(device)
            
		else:
			mask = mask.float().to(device)

		contrast_count = features.shape[1] #number of positives per sample
        
		#contrast_features separates the features of different views of the samples and puts them in rows, so features of
		# shape of [50, 2, 128] becomes [100, 128]. we do this to be to calculate dot-product between each two views
		contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) 
        
        
		anchor_feature = contrast_feature
		anchor_count = contrast_count
        
		# compute logits - calculates the dot product of every two vectors divided by temperature
		anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        
		# for numerical stability  (some kind of normalization!)
		logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
		logits = anchor_dot_contrast - logits_max.detach()
        
        
        
		# tile mask as much as number of positives per sample
		mask = mask.repeat(anchor_count, contrast_count)
		# mask-out self-contrast cases
		logits_mask = torch.scatter(torch.ones_like(mask), 1, 
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device),0)
        
        
		mask = mask * logits_mask
        
		# compute log_prob
		exp_logits = torch.exp(logits) * logits_mask
		eps = 1e-30
		log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)+eps)
        
		# compute mean of log-likelihood over positive
		mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+eps)
        
		# loss
		loss = -  mean_log_prob_pos
			

		loss = loss.view(anchor_count, batch_size).mean()
        
		return loss 
