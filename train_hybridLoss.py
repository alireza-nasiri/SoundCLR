import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import shutil
import datetime
import sys

from models import model_classifier
from models import model_projection
from utils.utils import EarlyStopping, WarmUpExponentialLR
import config
from loss_fn import hybrid_loss

if config.ESC_10:
	import dataset_ESC10 as dataset
elif config.ESC_50:
	import dataset_ESC50 as dataset
elif config.US8K:
	import dataset_US8K as dataset




use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



model =torchvision.models.resnet50(pretrained=True).to(device)
model.fc = nn.Sequential(nn.Identity())

model = nn.DataParallel(model, device_ids=[0, 1]) 
model = model.to(device)


projection_layer = model_projection.ProjectionModel().to(device)
classifier = model_classifier.Classifier().to(device)

train_loader, val_loader = dataset.create_generators()


loss_fn = hybrid_loss.HybridLoss(alpha = config.alpha, temperature = config.temperature).to(device)


optimizer = torch.optim.AdamW(list(model.parameters())+list(projection_layer.parameters())+
                            list(classifier.parameters()), lr=config.lr, weight_decay=1e-3)

scheduler = WarmUpExponentialLR(optimizer, cold_epochs= 0, warm_epochs= config.warm_epochs, gamma=config.gamma)





# creating a folder to save the reports and models
root = './results/'
main_path = root + str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
if not os.path.exists(main_path):
	os.mkdir(main_path)

classifier_path = main_path + '/' + 'classifier'
os.mkdir(classifier_path)




def hotEncoder(v):
	ret_vec = torch.zeros(v.shape[0], config.class_numbers).to(device)
	for s in range(v.shape[0]):
		ret_vec[s][v[s]] = 1
	return ret_vec






def train_hybrid():
	num_epochs = 800
    
	with open(main_path + '/results.txt','w', 1) as output_file:
		mainModel_stopping = EarlyStopping(patience=300, verbose=True, log_path=main_path, output_file=output_file)
		classifier_stopping = EarlyStopping(patience=300, verbose=False, log_path=classifier_path, output_file=output_file)

		print('*****', file=output_file)
		print('HYBRID', file=output_file)
		print('alpha is {}'.format(config.alpha), file=output_file)
		print('temperature of contrastive loss is {}'.format(config.temperature), file=output_file)
		if config.ESC_10:
			print('ESC_10', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold), file=output_file)
		elif config.ESC_50:
			print('ESC_50', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold), file=output_file)
		elif config.US8K:
			print('US8K', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold), file=output_file)

		print('Freq mask number {} and length {}, and time mask number {} and length is{}'.format(config.freq_masks, config.freq_masks_width, config.time_masks, config.time_masks_width), file=output_file)

		print('*****', file=output_file)
        
		for epoch in range(num_epochs):
			print('\n' + str(optimizer.param_groups[0]["lr"]), file=output_file)

			model.train()
			projection_layer.train()
			classifier.train()

			train_loss = []
			train_loss1 = []
			train_loss2 = []
			train_corrects = 0
			train_samples_count = 0
			for _, x, label in train_loader:
				loss = 0
				optimizer.zero_grad()
				
				x = x.float().to(device)
				label = label.to(device).unsqueeze(1)
				label_vec = hotEncoder(label)
            
            
				y_rep = model(x)
				y_rep = F.normalize(y_rep, dim=1)
                
				y_proj = projection_layer(y_rep)
				y_proj = F.normalize(y_proj, dim=1)
                
				y_pred = classifier(y_rep)
            
                
				loss1, loss2 = loss_fn(y_proj, y_pred, label, label_vec)
				
                
				loss = loss1 + loss2
				torch.autograd.backward([loss1, loss2])
                
				train_loss.append(loss.item() )
				train_loss1.append(loss1.item())
				train_loss2.append(loss2.item())
                
				optimizer.step()
                
				train_corrects += (torch.argmax(y_pred, dim=1) == torch.argmax(label_vec, dim=1)).sum().item()
				train_samples_count += x.shape[0]
                
                
                
            
			val_loss = []
			val_loss1 = []
			val_loss2 = []
			val_corrects = 0
			val_samples_count = 0
                
			model.eval()
			projection_layer.eval()
			classifier.eval()
                
			with torch.no_grad():
				for _, val_x, val_label in val_loader:
					val_x = val_x.float().to(device)
					label = val_label.to(device).unsqueeze(1)
					label_vec = hotEncoder(label)
					y_rep = model(val_x)
					y_rep = F.normalize(y_rep, dim=1)
                        
					y_proj = projection_layer(y_rep)
					y_proj = F.normalize(y_proj, dim=1)
                        
					y_pred = classifier(y_rep)
                            
					loss1, loss2 = loss_fn(y_proj, y_pred, label, label_vec)
					

					loss = loss1 + loss2
                        
					val_loss.append(loss.item() )
					val_loss1.append(loss1.item())
					val_loss2.append(loss2.item())
                        
					val_corrects += (torch.argmax(y_pred, dim=1) == torch.argmax(label_vec, dim=1)).sum().item()
					val_samples_count += val_x.shape[0]
                        
			train_acc = train_corrects / train_samples_count
			val_acc = val_corrects / val_samples_count
                
			scheduler.step()

                   
			print("\nEpoch: {}/{}...".format(epoch+1, num_epochs), "Loss: {:.4f}...".format(np.mean(train_loss)), "Val Loss: {:.4f}".format(np.mean(val_loss)), file=output_file)
			print('train_loss1 is {:.4f} and train_loss2 is {:.4f}'.format(np.mean(train_loss1), np.mean(train_loss2)),file=output_file)
			print('val_loss1 is {:.4f} and val_loss2 is {:.4f}'.format(np.mean(val_loss1), np.mean(val_loss2)), file=output_file)
			print('train_acc is {:.4f} and val_acc is {:.4f}'.format(train_acc, val_acc), file=output_file)
        
			# add validation checkpoint for early stopping here
			mainModel_stopping(-val_acc, model, epoch+1)
			#proj_stopping(-val_acc, projection_layer, epoch+1)
			classifier_stopping(-val_acc, classifier, epoch+1)
			if mainModel_stopping.early_stop:
				print("Early stopping", file=output_file)
				return
    

if __name__ == "__main__":
	train_hybrid()


