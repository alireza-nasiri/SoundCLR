import torch
import torch.nn as nn
import numpy as np
import os
import shutil
import datetime
import sys
import torch.nn.functional as F
import torchvision

import model_classifier
import model_projection

from utils import EarlyStopping, WarmUpExponentialLR
import config
import loss

if config.ESC_10:
        import dataset_ESC10 as dataset
elif config.ESC_50:
        import dataset_ESC50 as dataset
elif config.US8K:
        import dataset_US8K as dataset



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



model =torchvision.models.resnet50(pretrained=True).to(device)
model.fc = nn.Sequential(nn.Identity())


model = nn.DataParallel(model, device_ids=[0,1]) #
model = model.to(device)

projection_head = model_projection.ProjectionModel().to(device)

train_loader, val_loader = dataset.create_generators()


if config.US8K:
	#class_weights = dataset.getClassWeights()
	class_weights = torch.ones(config.class_numbers)
	class_weights = class_weights.to(device)
else:
        class_weights = torch.ones(config.class_numbers).to(device)



loss_fn = loss.SupConLoss(weights=class_weights)

optimizer = torch.optim.AdamW(list(model.parameters()) + list(projection_head.parameters()),
	lr=config.lr, weight_decay=1e-3) 
scheduler = WarmUpExponentialLR(optimizer, cold_epochs= 0, warm_epochs= config.warm_epochs, gamma=config.gamma)





root = './data/results/'
main_path = root + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
if not os.path.exists(main_path):
	os.mkdir(main_path)

shutil.copy('train_mainModel_supCon.py', main_path)





def hotEncoder(v):
	ret_vec = torch.zeros(v.shape[0], config.class_numbers).to(device)
	for s in range(v.shape[0]):
		ret_vec[s][v[s]] = 1
	return ret_vec



def train():
	num_epochs = 800
	with open(main_path + '/results.txt','w', 1) as output_file:
		mainModel_stopping = EarlyStopping(patience=300, verbose=True, log_path=main_path, output_file=output_file)

		print('*****', file=output_file)
		print('Supervised Contrastive Loss', file=output_file)
		print('temperature for the contrastive loss is {}'.format(config.temperature), file=output_file)
		if config.ESC_10:
			print('ESC_10', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold), file=output_file)
		elif config.ESC_50:
			print('ESC_50', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold), file=output_file)
		elif config.US8K:
			print('US8K', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold), file=output_file)

		print('number of freq masks are {} and their max length is {}'.format(config.freq_masks, config.freq_masks_width), file=output_file)
		print('number of time masks are {} and their max length is {}'.format(config.time_masks, config.time_masks_width), file=output_file)
		print('*****', file=output_file)
		
		for epoch in range(num_epochs):
			print(optimizer.param_groups[0]["lr"], file=output_file)

			model.train()
			projection_head.train()
        
			train_loss = []
           
			for x, label in train_loader:
				batch_loss = 0
				optimizer.zero_grad()
            
				x = x.to(device)
				label = label.to(device).unsqueeze(1)
				label_vec = hotEncoder(label)
            
				y_rep = model(x.float())
				y_rep = F.normalize(y_rep, dim=0)
            
				y_proj = projection_head(y_rep) 
				y_proj = F.normalize(y_proj, dim=0)
            
            
				batch_loss = loss_fn(y_proj.unsqueeze(1), label.squeeze(1))
            
            
				batch_loss.backward()
				train_loss.append(batch_loss.item() )
				optimizer.step()
            
        
			val_loss = []
			model.eval()
			projection_head.eval()
        
			with torch.no_grad():
				for val_x, val_label in val_loader:
					val_x = val_x.to(device)
					label = val_label.to(device).unsqueeze(1)
					label_vec = hotEncoder(label)
                
                
					y_rep = model(val_x.float())
					y_rep = F.normalize(y_rep, dim=0)
                
					y_proj = projection_head(y_rep) 
					y_proj = F.normalize(y_proj, dim=0)
                
					temp = loss_fn(y_proj.unsqueeze(1), label.squeeze(1))
                
					val_loss.append(temp.item() )
                
                
            
			scheduler.step()
        
        
			print("Epoch: {}/{}...".format(epoch+1, num_epochs),
				"Loss: {:.4f}...".format(np.mean(train_loss)),
				"Val Loss: {:.4f}".format(np.mean(val_loss)), file=output_file)
        
			mainModel_stopping(np.mean(val_loss), model, epoch+1)
			if mainModel_stopping.early_stop:
				print("Early stopping", file=output_file)
				return





train()








