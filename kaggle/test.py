import torch 
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import architecture as cnn
import joblib
import os

import csv

np.random.seed(0)

#Normalize image
def get_test_input(image, input_dim=224, CUDA=True):
	img = cv2.imread(image)
	img_ = cv2.resize(img, (input_dim, input_dim)) 
    
    
    #Normalize and apply std for test example

	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	img_ = std * img_ + mean

	img_ =  img_.transpose((2, 0, 1))

	img_ = img_[np.newaxis,:,:,:]/255.0
    
    #values outside the interval are clipped to the interval edges
	img_ = np.clip(img_, 0, 1)

	img_ = torch.from_numpy(img_).float()
	img_ = Variable(img_)

	if CUDA:
		img_ = img_.cuda()
    
	return img_


#compare results with test set
def submission(test_dir, path_weights):

	weights = path_weights+'/weights.h5'

	#Saving class names
	with open(path_weights+'/class_names', "rb") as file:
		class_names = joblib.load(file)

	num_classes = len(class_names)

	print('class_names: '+str(class_names))

	# Device configuration
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print (device)

	print("Loading network.....")
	model = cnn.AlexNet(num_classes)
	model.load_state_dict(torch.load(weights))
	print("Network successfully loaded")

	#put the model in eval mode to disable dropout
	model = model.eval()

	#This takes a long time (it is done one time)
	model = model.to(device)

	#image_dataset = {datasets.ImageFolder(os.path.join(test_dir, 'test'))}
	#dataloaders = {torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
	                                             	#shuffle=True, num_workers=2)}


	#dataset_size = int(len(image_datasets))

	#class_names = image_dataset.classes

	dataset_size = len(os.listdir(test_dir))

	print(dataset_size)

	#Visualize random images
	#for i in range(1):
		# Get a batch of training data
	#	inputs, classes = next(iter(dataloaders))

		# Make a grid from batch
	#	out = utils.make_grid(inputs)

	#	imshow(out, title="test images")


	#Save prediction results in csv file 
	with open('test.csv', 'w') as csvfile:
		filewriter = csv.writer(csvfile, delimiter=',',
		                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

		filewriter.writerow(['id', 'label'])

		print('Iterate over test data')

		# Iterate over data.
		for img in range(1, dataset_size+1):

			#image = mpimg.imread(test_dir+'/'+str(img)+'.jpg')

			#print (image)
			image = test_dir+'/'+str(img)+'.jpg'
			inputs = get_test_input(image)

			#print(inputs.shape)

			#inputs = torch.from_numpy(image).float().to(device)
			#labels = labels.to(device)

			outputs = model(inputs)

			_, prediction = torch.max(outputs, 1)
			prediction = class_names[prediction]

			#print (prediction)

			filewriter.writerow([str(img), str(prediction)])



#get accracy by testing the first 100
def acc(truths, predictions):

	running_corrects = -1

	with open(truths, 'r') as t1, open(predictions, 'r') as t2:
		fileone = t1.readlines()
		filetwo = t2.readlines()

		i = 0
		for row in fileone:
			if (row == filetwo[i]):
				running_corrects += 1
			i += 1	

		print(i-1)
		print(running_corrects)

		acc = float(running_corrects) / (i-1)

		print(acc)



if __name__ == '__main__':

	test_dir = 'data/test'

	path_weights = 'weights/2019-02-03-18:35'

	submission(test_dir, path_weights)

	truths = 'sample_submission.csv'
	predictions = 'test.csv'

	acc(truths, predictions)
