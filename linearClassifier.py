# This code is part of:
#
#  CMPSCI 370: Computer Vision, Spring 2023
#  University of Massachusetts, Amherst
#  Instructor: Subhransu Maji
#
#  Homework 5

import pickle
import numpy as np
from matplotlib import pyplot as plt

def softmax(z):
    return 1.0/(1+np.exp(-z))

def linearTrain(x, y):
    #Training parameters
    maxiter = 50
    lamb = 0.01
    eta = 0.01
    
    #Add a bias term to the features
    x = np.concatenate((x, np.ones((1, x.shape[1]))), axis=0)
    
    class_labels = np.unique(y)
    num_class = class_labels.shape[0]
    assert(num_class == 2) # Binary labels
    num_feats = x.shape[0]
    num_data = x.shape[1]
    
    true_prob = np.zeros(num_data)
    true_prob[y == class_labels[0]] = 1
    
    #Initialize weights randomly
    model = {}
    model['weights'] = np.random.randn(num_feats)*0.01
    # print('w', model['weights'].shape)
    #Batch gradient descent
    verbose_output = False
    for it in range(maxiter):
        prob = softmax(model['weights'].dot(x))
        delta = true_prob - prob
        gradL = delta.dot(x.T)
        model['weights'] = (1 - eta*lamb)*model['weights'] + eta*gradL
    model['classLabels'] = class_labels

    return model


def linearPredict(model, x):
    #Add a bias term to the features
    x = np.concatenate((x, np.ones((1, x.shape[1]))), axis=0)

    prob = softmax(model['weights'].dot(x))
    ypred = np.ones(x.shape[1]) * model['classLabels'][1]
    ypred[prob > 0.5] = model['classLabels'][0]

    return ypred


def testLinear():
    #--------------------------------------------------------------------------
    # Your implementation to answer questions on Linear Classifier
    #--------------------------------------------------------------------------
    data = pickle.load(open('data.pkl','rb'))
    xtrain = data['train']['x']
    ytrain = data['train']['y']
    xtest = data['test']['x']
    ytest = data['test']['y']
    print(xtrain.shape)


    xtrain_f = np.zeros((28*28,len(ytrain)))
    xtest_f = np.zeros((28*28,len(ytest)))

    print(xtrain_f.shape,xtest_f.shape)

    for y in range(len(ytrain)):
        img = xtrain[:,:,y]
        xtrain_f[:,y] = img.flatten()

    for y in range(len(ytest)):
        img = xtest[:,:,y]
        xtest_f[:,y] = img.flatten()
        
    model = linearTrain(xtrain_f,ytrain)
    ypred = linearPredict(model, xtest_f)

    
    cnt = 0
    for i in range(len(ypred)):
        if(ypred[i]==ytest[i]):
            cnt = cnt+1
    
    accuracy = (cnt/len(ypred)) * 100
    print('accuracy = ', accuracy)

    w = model["weights"]
    w2d = w[0:-1].reshape((28,28))
    
    wp=np.clip(w2d, 0, None)
    wn=np.clip(w2d, None, 0)
    plt.figure(1)
    plt.title('positive weight')
    plt.imshow(wp, cmap ='gray')
    plt.figure(2)
    plt.title('negative weight')
    plt.imshow(wn, cmap ='gray')

    plt.show()
    

    return





if __name__ == "__main__":
    testLinear()


