#!/home/yves/anaconda2/bin/python
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os, re, sys, pickle,time
import copy as cp
import cv2
import skimage
import scipy as sp
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.measure import label, regionprops
from skimage.color import label2rgb



def findpath():
    cDir = os.path.dirname(os.path.realpath(__file__))

    found = False
    while not found:
        cDir,ext = os.path.split(cDir) 
        if ext=='twoptb':
            found = False
            twoptb_path = cDir
            print 
            break
    return twoptb_path

twoptb_path = findpath()
sys.path.append(twoptb_path)

import twoptb as MP

def extract_pupil_from_stack(pupil_stack,thrsh='auto',n_Frames='auto',ms1=2e5,ms2=3e5,verbose=0):

    """ Input a numpy array of pupil images and returns parameters of ellipses fit to the pupil


    """

    if n_Frames=='auto':
        n_Frames = int(pupil_stack.shape[0])

    centre_ix = (np.array(pupil_stack[0].shape)/2).astype('int')

    if thrsh == 'auto':
        thrsh = np.mean(pupil_stack[:,centre_ix[0]-10:centre_ix[0]+10,centre_ix[1]-10:centre_ix[1]+10])*1.2
    else:
        thrsh = thrsh

    pupil_fits = []



    for ix in range(n_Frames):
        if verbose:
            sys.stdout.write("\r processing: %s/%s elapsed time: %s" %(ix,len(pupil),np.round(time.time() -st,decimals=1)))
            sys.stdout.flush()
        
        pup_im = cp.deepcopy(pupil_stack[ix])
            
        ellipse = _get_pupil((pup_im,thrsh))
            
            
        pupil_fits.append(ellipse)

    return pupil_fits



def _get_pupil(A):
    """ Function to extract pupil diameter from a single image

    Arguments:
    ==========================

    A:          tuple
                first element is an np.array containing the image of the pupil
                second entry is float containing threshold value of pupil brightness to include

    Returns:
    ==========================
    
    Ellipse:    list
                Parameters of an ellipse fit to the pupil on that trial

    Ims:        list
                intermediate images, processing steps


    """

    pup_im,thrsh = A
    pup_im[np.where(pup_im>thrsh*1.2)[0],np.where(pup_im>thrsh*1.2)[1]] = 0

    #pup_im[np.where(pup_im>120)[0],np.where(pup_im>120)[1]] = 0
    #val = skimage.filters.threshold_local(pup_im,55,offset=10)
    #mask = val
    val = skimage.filters.threshold_otsu(pup_im)
    mask = pup_im > val

    im2 = skimage.morphology.binary_opening(mask,skimage.morphology.disk(5))
    im2 = sp.ndimage.morphology.binary_fill_holes(im2)

    #plt.imshow(im2*pupil[ix])
    #plt.colorbar()
    im3 = skimage.morphology.binary_erosion(im2,skimage.morphology.disk(10))
    im3 = skimage.morphology.remove_small_objects(im3,min_size=20000)

    im4 = skimage.morphology.binary_dilation(im3,skimage.morphology.disk(10))
    im5 = skimage.morphology.binary_dilation(canny(im4),skimage.morphology.disk(3))
    #plt.imshow(im3)
    res = skimage.morphology.remove_small_objects(sp.ndimage.morphology.binary_fill_holes(im5),min_size=30000)
    _,cnt,_ = cv2.findContours(res.astype('uint8'),1,2)
    if len(cnt)>0:
        _,cnt,_ = cv2.findContours(res.astype('uint8'),1,2)
        ellipse = cv2.fitEllipse(cnt[0])
    else:
        ellipse = [(0,0),(0,0),0]

    return ellipse


def write_pupil_video(write_path,pupil,pupil_fit):

    """ Function to create a video where the pupil has been fit """

    height , width =  pupil[0].shape

    video = cv2.VideoWriter(write_path,cv2.VideoWriter_fourcc(*"MJPG"),10,(width,height))
    #pts = np.vstack([np.arange(len(pupil))/2,szs+200])
    for i in range(len(pupil_fit)):
        sys.stdout.write("\rframe %s" %(i))
        if pupil_fit[i][-1]==0:
            writeIM = cv2.cvtColor(pupil[i],cv2.COLOR_GRAY2RGB)

        else:
            writeIM = cv2.ellipse(cv2.cvtColor(pupil[i+offset],cv2.COLOR_GRAY2RGB),pupil_fit[i],(255,0,0),2)
        

        video.write(writeIM)

    cv2.destroyAllWindows()
    video.release()


def plot_pupil(pupil_stack,pupil_fit,ix=0):
    plt.imshow(cv2.ellipse(cv2.cvtColor(pupil_stack[ix],cv2.COLOR_GRAY2RGB),pupil_fit[ix],(255,0,0),2))
    plt.xticks([])
    plt.yticks([])

def evaluate_fitting_params(pupil_stack,n=10):

    """ Function to test parameters for fitting the pupil """


    pup_im = cp.deepcopy(pupil_stack[ix])

    centre_ix = np.array(pupil[0].shape)/2
    #thrsh = np.max(pupil[10][centre_ix[0]-10:centre_ix[0]+10,centre_ix[1]-10:centre_ix[1]+10])

    pup_im[np.where(pup_im>thrsh*1.2)[0],np.where(pup_im>thrsh*1.2)[1]] = 0
    #val = skimage.filters.threshold_local(pup_im,55,offset=10)
    #mask = val
    val = skimage.filters.threshold_otsu(pup_im)
    mask = pup_im > val
    #mask = sp.ndimage.morphology.binary_fill_holes(mask)

    im2 = skimage.morphology.binary_opening(mask,skimage.morphology.disk(5))
    im2 = sp.ndimage.morphology.binary_fill_holes(im2)
    #plt.imshow(im2*pupil[ix])
    #plt.colorbar()
    im3 = skimage.morphology.binary_erosion(im2,skimage.morphology.disk(10))
    im3 = skimage.morphology.remove_small_objects(im3,min_size=20000)

    im4 = skimage.morphology.binary_dilation(im3,skimage.morphology.disk(10))
    im5 = skimage.morphology.binary_dilation(canny(im4),skimage.morphology.disk(3))
    #plt.imshow(im3)
    res = skimage.morphology.remove_small_objects(sp.ndimage.morphology.binary_fill_holes(im5),min_size=30000)
    _,cnt,_ = cv2.findContours(res.astype('uint8'),1,2)
    ###################### In this block of code find the contour whose   ######################
    ###################### to the centre of the image centroid is closest ######################
    dst = []

    for i in cnt:
        M = cv2.moments(i)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        dst.append(np.abs(np.array([cY,cX])-centre_ix).sum())

    cnt_ix = np.argmin(np.array(dst))
    #########################################################################################
    ########################################################################################

    if len(cnt)>0:
        _,cnt,_ = cv2.findContours(res.astype('uint8'),1,2)
        ellipse = cv2.fitEllipse(cnt[cnt_ix])
    else:
        ellipse = [(0,0),(0,0),0]

    _plot_test_stack()


def _plot_test_stack():

    plt.figure(figsize=(16,8))
    plt.subplot(1,8,1)
    plt.imshow(pup_im,cmap='binary_r')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1,8,2)
    plt.imshow(mask)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1,8,3)
    plt.imshow(im2)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1,8,4)
    plt.imshow(im3)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1,8,5)
    plt.imshow(im4)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1,8,6)
    plt.imshow(im5)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1,8,7)
    plt.imshow(res)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1,8,8)
    plt.imshow(cv2.ellipse(cv2.cvtColor(pupil[ix],cv2.COLOR_GRAY2RGB),ellipse,(255,0,0),2))
    plt.xticks([])
    plt.yticks([])