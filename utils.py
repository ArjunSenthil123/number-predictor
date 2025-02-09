# This code is part of:
#
#  CMPSCI 370: Computer Vision, Spring 2023
#  University of Massachusetts, Amherst
#  Instructor: Subhransu Maji
#
#  Homework 5

import numpy as np
import os
import errno
import matplotlib.pyplot as plt 
import scipy.io as spio
from skimage.feature import plot_matches


def showMatches(im1, im2, c1, c2, matches, title=""):
    disp_matches = np.array([np.arange(matches.shape[0]), matches]).T.astype(int)
    valid_matches = np.where(matches>=0)[0]
    disp_matches = disp_matches[valid_matches, :]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plot_matches(ax, im1, im2, 
            c1[[1, 0], :].astype(int).T, c2[[1,0], :].astype(int).T, disp_matches)
    ax.set_title(title)


#Thanks to mergen from https://stackoverflow.com/questions/7008608
def todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = todict(elem)
        else:
            dict[strg] = elem
    return dict


#data.mat is a dict with the following structure:
#   {train:
#       {x: array with raw data (image height, image width, number of images)}
#       {y: array with corresponding labels}},
#   {test:
#       {x: array with raw data (image height, image width, number of images)}
#       {y: array with corresponding labels}}
# def loadmat(path):
#     return todict(spio.loadmat(path,
#                                struct_as_record=False, 
#                                squeeze_me=True)['data'])


def imread(path):
    img = plt.imread(path).astype(float)
    
    #Remove alpha channel if it exists
    if img.ndim > 2 and img.shape[2] == 4:
        img = img[:, :, 0:3]
    #Puts images values in range [0,1]
    if img.max() > 1.0:
        img /= 255.0

    return img


def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    else:
        print("Directory {} already exists.".format(dirpath))

    
#Thanks to ali_m from https://stackoverflow.com/questions/17190649
def gaussian(hsize=3,sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    shape = (hsize, hsize)
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
