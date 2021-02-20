ESC_10 = False
ESC_50 = True
US8K = False

ESC10_classIds = [0, 1, 10, 11, 12, 20, 21, 38, 40, 41]


if ESC_50:
	class_numbers = 50
else:
	class_numbers = 10


if ESC_10 or ESC_50:
	lr = 5e-4 #for ESC datasets: 5e-4, for US8K: 1e-4
	folds = 5
	train_folds = [1, 2, 3, 5]
	test_fold = [4]
else:
	lr = 1e-4 # for US8K
	fold = 10
	train_folds =[1, 2, 3, 4, 5, 6, 7, 8, 9]
	test_fold = [1]

supCon_path_for_classifier = './data/results/2020-12-22-10-42/'

temperature = 0.05
alpha = 0.5

freq_masks = 2
time_masks = 1
freq_masks_width = 32
time_masks_width = 32

epochs = 800
batch_size = 128
warm_epochs = 10
gamma = 0.98

