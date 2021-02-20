import torch
import torch.nn as nn
import numpy as np
import os
import shutil
import datetime
import sys
import torch.nn.functional as F
import torchvision
import librosa

import model_classifier
from utils import EarlyStopping, WarmUpExponentialLR
import config

if config.ESC_10:
	import dataset_ESC10 as dataset
elif config.ESC_50:
	import dataset_ESC50 as dataset
elif config.US8K:
	import dataset_US8K as dataset


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
print(torch.cuda.current_device())

main_model =torchvision.models.resnet50(pretrained=True).to(device)
#main_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

main_model.fc = nn.Sequential(nn.Identity())


main_model = nn.DataParallel(main_model, device_ids=[0,1])
main_model = main_model.to(device)
main_model = main_model.train()

classifier = model_classifier.Classifier().to(device)


train_loader, val_loader = dataset.create_generators()

if config.US8K:
	#class_weights = dataset.getClassWeights()
	class_weights = torch.ones(config.class_numbers).to(device)
	class_weights = class_weights.to(device)
else:
	class_weights = torch.ones(config.class_numbers).to(device)


root = './data/results/'
main_path = root + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
if not os.path.exists(main_path):
	os.mkdir(main_path)
shutil.copy('baseline_train.py', main_path )

classifier_path = main_path + '/' + 'classifier'
os.mkdir(classifier_path)



optimizer = torch.optim.AdamW(list(main_model.parameters())+ list(classifier.parameters()), 
                             lr=config.lr, weight_decay=1e-3)

scheduler = WarmUpExponentialLR(optimizer, cold_epochs= 0, warm_epochs= config.warm_epochs, gamma=config.gamma)



def hotEncoder(v):
	ret_vec = torch.zeros(v.shape[0], config.class_numbers).to(device)
	for s in range(v.shape[0]):
		ret_vec[s][v[s]] = 1
	return ret_vec




def cross_entropy_one_hot(input, target):
	_, labels = target.max(dim=1)
	return nn.CrossEntropyLoss(weight=class_weights)(input, labels)




def train():
	num_epochs = 800
	with open(main_path + '/results.txt','w', 1) as output_file:
		mainModel_stopping = EarlyStopping(patience=300, verbose=True, log_path=main_path, output_file=output_file)
		classifier_stopping = EarlyStopping(patience=300, verbose=False, log_path=classifier_path, output_file=output_file)

		print('*****', file=output_file)
		print('BASELINE', file=output_file)
		print('transfer - augmentation on both waves and specs - 3 channels', file=output_file)
		if config.ESC_10:
			print('ESC_10', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold), file=output_file)
		elif config.ESC_50:
			print('ESC_50', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold), file=output_file)
		elif config.US8K:
			print('US8K', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.us8k_train_folds, config.us8k_test_fold), file=output_file)


		print('number of freq masks are {} and their max length is {}'.format(config.freq_masks, config.freq_masks_width), file=output_file)
		print('number of time masks are {} and their max length is {}'.format(config.time_masks, config.time_masks_width), file=output_file)
		print('*****', file=output_file)
	


		for epoch in range(num_epochs):
			print('\n' + str(optimizer.param_groups[0]["lr"]), file=output_file)
        
			main_model.train()
			classifier.train()
        
			train_loss = []
			train_corrects = 0
			train_samples_count = 0
        
			for x, label in train_loader:
				loss = 0
				optimizer.zero_grad()
            
				inp = x.float().to(device)
				label = label.to(device).unsqueeze(1)
				label_vec = hotEncoder(label)
            
				y_rep = main_model(inp)
				y_rep = F.normalize(y_rep, dim=0)
            
				y_pred = classifier(y_rep)
            
				loss += cross_entropy_one_hot(y_pred, label_vec)
				loss.backward()
				train_loss.append(loss.item() )
				optimizer.step()
            
				train_corrects += (torch.argmax(y_pred, dim=1) == torch.argmax(label_vec, dim=1)).sum().item()
				train_samples_count += x.shape[0]
        
        
			val_loss = []
			val_corrects = 0
			val_samples_count = 0
        
			main_model.eval()
			classifier.eval()
        
			with torch.no_grad():
				for val_x, val_label in val_loader:
					inp = val_x.float().to(device)
					label = val_label.to(device)
					label_vec = hotEncoder(label)
                
					y_rep = main_model(inp)
					y_rep = F.normalize(y_rep, dim=0)

					y_pred = classifier(y_rep)
                
					temp = cross_entropy_one_hot(y_pred, label_vec)
					val_loss.append(temp.item() )
                
					val_corrects += (torch.argmax(y_pred, dim=1) == torch.argmax(label_vec, dim=1)).sum().item() 
					val_samples_count += val_x.shape[0]
        
        
			scheduler.step()
        
			train_acc = train_corrects / train_samples_count
			val_acc = val_corrects / val_samples_count
			print('\n', file=output_file)
			print("Epoch: {}/{}...".format(epoch+1, num_epochs), "Loss: {:.4f}...".format(np.mean(train_loss)),
				"Val Loss: {:.4f}".format(np.mean(val_loss)), file=output_file)
			print('train_acc is {:.4f} and val_acc is {:.4f}'.format(train_acc, val_acc), file=output_file)
			mainModel_stopping(-val_acc, main_model, epoch+1)
			classifier_stopping(-val_acc, classifier, epoch+1)
			if mainModel_stopping.early_stop:
				print("Early stopping", file=output_file)
				return



train()








