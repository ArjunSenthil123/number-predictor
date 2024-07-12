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


def accuracy_level1(x,y):
	numc5 = 0
	numc9 = 0
	cnt5 = 0
	cnt9 = 0
	h, w, d = x.shape
	for i in range(d):
		if(y[i]==5):
			cnt5 = cnt5+1
		else:
			cnt9 = cnt9+1
		if(((x[11,18,i] == 0) and y[i]==5)):
			numc5 = numc5 + 1
		if(((x[11,18,i] == 1) and y[i]==9)):
			numc9 = numc9 + 1

	accuracy = (numc5 + numc9)/(cnt5+cnt9) * 100
	accuracy5 = (numc5)/(cnt5) * 100
	accuracy9 = (numc9)/(cnt9) * 100
	print('num5: ',cnt5,' num9: ', cnt9)
	print('total accuracy: ', accuracy, ' accuracy of 5: ',accuracy5, 
       ' accuracy of 9: ', accuracy9)
	
	return(cnt5,cnt9)

def accuracy_level2(x,y):
	numc5 = 0
	numc9 = 0
	cnt5 = 0
	cnt9 = 0
	h, w, d = x.shape

	for i in range(d):
		if(y[i]==5):
			cnt5 = cnt5+1
		else:
			cnt9 = cnt9+1


		if(x[11,18,i] == 0):
			if ((x[12,20,i] == 0) and y[i]==5):
				numc5 = numc5 + 1
			if ((x[12,20,i] == 1) and y[i]==9):
				numc9 = numc9 + 1
		else:
			if ((x[11,14,i] == 1) and y[i]==5):
				numc5 = numc5 + 1
			if ((x[11,14,i] == 0) and y[i]==9):
				numc9 = numc9 + 1

	accuracy = (numc5 + numc9)/(cnt5+cnt9) * 100
	accuracy5 = (numc5)/(cnt5) * 100
	accuracy9 = (numc9)/(cnt9) * 100
	print('num5: ',cnt5,' num9: ', cnt9)
	print('total accuracy (level 2): ', accuracy, ' accuracy of 5: ',accuracy5, 
       ' accuracy of 9: ', accuracy9)
	
	return(cnt5,cnt9)


def scoreFeatures(x, y):
	scores = np.zeros(x.shape[:2])
	is_50 = np.zeros(x.shape[:2])

	#--------------------------------------------------------------------------
	# Calculate scores (Implement this)
	#--------------------------------------------------------------------------
	h, w, d = x.shape
	print('shape = ', h,w,d)
	for yy in range(h):
		for xx in range(w):
			#compute 0,1 in 5s
			sum_5 = 0
			sum_9 = 0
			cnt5 = 0
			cnt9 = 0
			for i in range(d):
				if(y[i]== 5):
					sum_5 = sum_5 + x[yy,xx,i]
					cnt5 = cnt5 + 1
				else:
					sum_9 = sum_9 + x[yy,xx,i]
					cnt9 = cnt9 + 1

			num_5_1 = sum_5
			num_5_0 = cnt5 - num_5_1
			correct_5_0 = num_5_0
			correct_5_1 = num_5_1
			num_9_1 = sum_9
			num_9_0 = cnt9 - sum_9
			correct_9_1 = num_9_1
			correct_9_0 = num_9_0
			#put score in scores array
			scores5091 = correct_5_0 + correct_9_1
			scores5190 = correct_5_1 + correct_9_0

			if(scores5091>scores5190):
				is_50[yy,xx] = 0
				scores[yy,xx] = scores5091
			else:
				is_50[yy,xx] = 1
				scores[yy,xx] = scores5190
			


	
	plt.figure(1)
	plt.imshow(scores, cmap ='gray')
	plt.figure(2)
	plt.imshow(is_50, cmap ='gray')
	plt.axis('off')
	plt.show()
	return scores


def main():
	data = pickle.load(open('data.pkl','rb'))
	scores = scoreFeatures(data['train']['x'], data['train']['y'])

	#--------------------------------------------------------------------------
	# Your implementation to answer questions on Decision Trees
	#--------------------------------------------------------------------------

	pos_max = np.where(scores == np.amax(scores))
	print(pos_max)
	
	y = pos_max[0][0]
	x = pos_max[1][0]
	print(y,x,scores[y,x])

	#get scores for test set 
	accuracy_level1(data['test']['x'], data['test']['y'])

	
	train_data_x = data['train']['x']
	train_data_y = data['train']['y']
	h,w,d = train_data_x.shape
	
	#count number of 0s and 1s
	numc5 = 0
	numc9 = 0
	for i in range(d):
		if(((train_data_x[11,18,i] == 0))):
			numc5 = numc5 + 1
		else:
			numc9 = numc9 + 1
	print('h,w,d ',h,w,d)
	arr_0 = np.zeros((h,w,numc5))
	print(arr_0.shape, 'shape')
	y_arr_0 = np.zeros(numc5)
	print('train data shape', train_data_x.shape)
	arr_1 = np.zeros((h,w,numc9))
	y_arr_1 = np.zeros(numc9)
	#from data train extract x and y such that x(11,18) = 0, (data_train0)
	cnt0 = 0
	cnt1 = 0
	for i in range(d):
		if((train_data_x[11,18,i] == 0)):
			arr_0[:,:,cnt0] = train_data_x[:,:,i]
			y_arr_0[cnt0] = train_data_y[i]
			cnt0 = cnt0+1
		else:
			arr_1[:,:,cnt1] = train_data_x[:,:,i]
			y_arr_1[cnt1] = train_data_y[i]
			cnt1 = cnt1+1
	print('num 0s = ', cnt0, ' num 1s = ', cnt1)
	#from data train extract x and y such that x(11,18) = 1, (data_train1)
	#find best pixel in data_train0 and if 5 should be a 0 or 1
	scores_0 = scoreFeatures(arr_0, y_arr_0)
	pos_max_0 = np.where(scores_0 == np.amax(scores_0))
	
	y0 = pos_max_0[0][0]
	x0 = pos_max_0[1][0]
	print('position and value of arr_0',y0,x0,scores_0[y0,x0])
	#find best pixel in data_train1 and if 5 should be a 0 or 1
	scores_1 = scoreFeatures(arr_1, y_arr_1)
	pos_max_1 = np.where(scores_1 == np.amax(scores_1))
	
	y1 = pos_max_1[0][0]
	x1 = pos_max_1[1][0]
	print('position and value of arr_1',y1,x1,scores_1[y1,x1])
	#write function accuracylevel2 given these new if statements
	
	accuracy_level2(data['test']['x'], data['test']['y'])


	return 


if __name__ == "__main__":
	main()
	