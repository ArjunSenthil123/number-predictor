# This code is part of:
#
#  CMPSCI 370: Computer Vision, Spring 2023
#  University of Massachusetts, Amherst
#  Instructor: Subhransu Maji
#
#  Homework 5

import pickle
import numpy as np
import matplotlib.pyplot as plt



def visualizeNeighbors(imgs, topk_idxs, topk_distances, title):
	'''
	Visualize the query image as well as its nearest neighbors
	Input:
		imgs: a list or numpy array, with length k+1.
			imgs[0] is the query image with shape hxw
			imgs[k] is k-th nearest neighbor image
		topk_idxs: a list or numpy array, with length k+1.
			topk_idxs[k] is the index in training set of the k-th nearest image 
			topk_idxs[0] is the query image index in the test set
		topk_distances: a list or numpy array, with length k+1.
			topk_idxs[k] is the distance of the k-th nearest image to the query
			topk_idxs[0] is 0
	'''
	n = len(imgs)
	fig, axs = plt.subplots(1, n, figsize=(2 * n, 3))
	fig.suptitle(title)
	for k in range(n):
		if k == 0:
			ax_title = 'query: test_idx=%d' % topk_idxs[0]
		else:
			ax_title = '%d: idx=%d,d=%.2e' %(k, topk_idxs[k], topk_distances[k])
		axs[k].set_title(ax_title)
		axs[k].imshow(imgs[k], cmap='gray')
		axs[k].axis('off')
	fig.tight_layout()
	plt.show() 

	return 		

def euDistance(im1,im2):
	imdiff = im1 - im2
	imdiff2 = imdiff**2
	sum = np.sum(imdiff2)
	return sum**(.5)


def findMajorityLabel(dist,k,ytrain,i):
	
	corr5 = 0
	corr9 = 0
	for kk in range(k):
		ind_close_k = dist[i,kk]
		if(ytrain[ind_close_k]==5):
			corr5 = corr5 + 1
		else:
			corr9 = corr9+1
	if(corr5>corr9):
		label = 5
	else:
		label = 9
	
	return label







def computeAccuracy(ytrain,k,dist,ytest):

	r = len(ytest)
	acc = 0
	for i in range(r):
		labelCorr = ytest[i]
		labelEst = findMajorityLabel(dist,k,ytrain,i)
		if(labelCorr == labelEst):
			acc = acc + 1

	acc = (acc/r)

	return acc



def knn(visualize):
	data = pickle.load(open('data.pkl','rb'))
	
	distances = np.zeros((len(data['test']['y']), len(data['train']['y'])))
	num_test = len(data['test']['y'])
	num_train = len(data['train']['y'])

	for i in range(num_test):
		imtest = data['test']['x'][:,:,i]
		for j in range(num_train):
			imtrain = data['train']['x'][:,:,j]
			distances[i,j] = euDistance(imtest,imtrain)

	sortedarr = np.argsort(distances, axis = 1)

	#sort each row by index (smallest first)
	#--------------------------------------------------------------------------
	# Your implementation to calculate and sort distances
	#--------------------------------------------------------------------------

	if visualize:
		k = 5
		imgs = np.random.randint(2, size=(k+1, 28, 28))
		topk_idxs = [0] * (k+1)
		topk_distances = [0] * (k+1)
		for test_i in [10, 20, 110, 120]:
			imgs[0,:,:] = data['test']['x'][:,:,test_i]
			imgs[1,:,:] = data['train']['x'][:,:,sortedarr[test_i,0]]
			imgs[2,:,:] = data['train']['x'][:,:,sortedarr[test_i,1]]
			imgs[3,:,:] = data['train']['x'][:,:,sortedarr[test_i,2]]
			imgs[4,:,:] = data['train']['x'][:,:,sortedarr[test_i,3]]
			imgs[5,:,:] = data['train']['x'][:,:,sortedarr[test_i,4]]
			
			topk_idxs[0] = test_i
			#------------------------------------------------------------------
			# Prepare imgs, topk_idxs and topk_distances
			#------------------------------------------------------------------
			visualizeNeighbors(imgs, topk_idxs, topk_distances, 
				title='Test img %d: Top %d Neighbors' % (test_i, k))

	k_list = [1, 3, 5, 7, 9]
	accuracy_list = [0.0] * len(k_list)

	
	#--------------------------------------------------------------------------
	# Your implementation to calculate knn accuracy
	#--------------------------------------------------------------------------
	accuracy_list= []
	for k in k_list:
		
		acc = computeAccuracy(data['train']['y'],k,sortedarr,data['test']['y'])
		accuracy_list.append(acc)
	
				
	

	for k, acc in zip(k_list, accuracy_list):
		print('k=%d: accuracy=%.2f%%' % (k, acc * 100))

	return






if __name__ == "__main__":
	knn(visualize=True)
	