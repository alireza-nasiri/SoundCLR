import torch
import torch.nn as nn
import numpy as np
import os
import shutil
import datetime
import sys
import torch.nn.functional as F
import torchvision
from torch.optim import lr_scheduler

from models import model_classifier
from models import model_projection
from utils.utils import EarlyStopping
import config

from utils import EarlyStopping, WarmUpExponentialLR
if config.ESC_10:
        import dataset_ESC10 as dataset
elif config.ESC_50:
        import dataset_ESC50 as dataset
elif config.US8K:
        import dataset_US8K as dataset


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



main_path = config.supCon_path_for_classifier

state_dict = torch.load( main_path + 'checkpoint.pt' )

pretrained_model =torchvision.models.resnet50(pretrained=True).to(device)
pretrained_model.fc = nn.Sequential(nn.Identity())


#to use multiple GPU cores for the model
pretrained_model = nn.DataParallel(pretrained_model, device_ids=[0, 1])
pretrained_model = pretrained_model.to(device)
pretrained_model.load_state_dict(state_dict)

pretrained_model = pretrained_model.eval()

classifier = model_classifier.Classifier().to(device)

train_loader, val_loader = dataset.create_generators()





optimizer = torch.optim.AdamW(list(classifier.parameters()), lr=config.lr,  weight_decay=1e-3)

scheduler = WarmUpExponentialLR(optimizer, cold_epochs= 0, warm_epochs= config.warm_epochs, gamma=0.995) 

# to save the parameters for the classifier
classifier_path = main_path + 'classifier/'
if not os.path.exists(classifier_path):
	os.mkdir(classifier_path)





def hotEncoder(v):
	ret_vec = torch.zeros(v.shape[0], config.class_numbers ).to(device)
	for s in range(v.shape[0]):
		ret_vec[s][v[s]] = 1
    
	return ret_vec




def cross_entropy_one_hot(input, target):
	_, labels = target.max(dim=1)
	ls = nn.CrossEntropyLoss()(input, labels)
	return ls




def train_classifier():
	num_epochs = 800

	with open(main_path + '/classifier_results.txt','w', 1) as output_file:
		classifier_stopping = EarlyStopping(patience=300, verbose=True, log_path=classifier_path, output_file=output_file)

		print('*****', file=output_file)
		print('classifier after sup_contrastive', file=output_file)
		
		if config.ESC_10:
			print('ESC_10', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold), file=output_file)
		elif config.ESC_50:
			print('ESC_10', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold), file=output_file)
		elif config.US8K:
			print('US8K', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.us8k_train_folds, config.us8k_test_fold), file=output_file)

		
		print('number of freq masks are {} and their max length is {}'.format(config.freq_masks, config.freq_masks_width), file=output_file)
		print('number of time masks are {} and their max length is {}'.format(config.time_masks, config.time_masks_width), file=output_file)
		print('*****', file=output_file)

		for epoch in range(num_epochs):
        
			classifier.train()
			train_loss = []
        
			train_corrects = 0
			train_samples_count = 0
        
			for _, x, label in train_loader:
				loss = 0
				optimizer.zero_grad()
            
				x = x.float().to(device)
				label = label.to(device).unsqueeze(1)
				label_vec = hotEncoder(label)
            
				y_rep = pretrained_model(x)
				y_rep = F.normalize(y_rep, dim=1)
            
				out = classifier(y_rep)
				loss = cross_entropy_one_hot(out, label_vec) 
				loss.backward()
				train_loss.append(loss.item() )
            
				optimizer.step()
            
				train_corrects += (torch.argmax(out, dim=1) == torch.argmax(label_vec, dim=1)).sum().item()
				train_samples_count += x.shape[0]
            
        
			val_loss = []
			val_acc = []
			val_corrects = 0
			val_samples_count = 0
        
			classifier.eval()
        
        
			with torch.no_grad():
				for _, val_x, val_label in val_loader:
					val_x = val_x.float().to(device)
					label = val_label.to(device).unsqueeze(1)
					label_vec = hotEncoder(label)
                
					y_rep = pretrained_model(val_x)
					y_rep = F.normalize(y_rep, dim=1)
                
					out = classifier(y_rep)
					temp = cross_entropy_one_hot(out, label_vec)
					val_loss.append(temp.item())
            
					val_corrects += (torch.argmax(out, dim=1) == torch.argmax(label_vec, dim=1)).sum().item()
					val_samples_count += val_x.shape[0]
            
			train_acc = train_corrects / train_samples_count
			val_acc = val_corrects / val_samples_count
        
			scheduler.step()
		
			print('\n', file=output_file)
			print("Epoch: {}/{}...".format(epoch+1, num_epochs),
                    		"Loss: {:.4f}...".format(np.mean(train_loss)),
                    		"Val Loss: {:.4f}".format(np.mean(val_loss)), file=output_file)
			print('train_acc is {:.4f} and val_acc is {:.4f}'.format(train_acc, val_acc), file=output_file)
        
			classifier_stopping(-val_acc, classifier, epoch+1)
			if classifier_stopping.early_stop:
				print("Early stopping", file=output_file)
				return  



if __name__ == "__main__":
	train_classifier()
