from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import twoptb
import time
from skimage import morphology
import h5py
from IPython import display
import os
import pickle
import sys
from skimage.exposure import equalize_adapthist
import skimage

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
#sys.path.append(os.path.abspath())
import twoptb as MP

if __name__ =="__main__":

    rfC_fit,rad = pickle.load(open(sys.argv[1]))
    hdfF11 = h5py.File(sys.argv[2])

    #ded = [5,4,5] #for zoom 2
    bhdf11 =  hdfF11[hdfF11.keys()[0]][ u'registered_data']
    areas =bhdf11.keys()
    meanIm11 = bhdf11[areas[1]].attrs['mean_image']
    meanIm12 = bhdf11[areas[1]].attrs['mean_image']

    meanIm11 = equalize_adapthist(meanIm11/np.max(meanIm11),clip_limit=0.005)
    meanIm12 = equalize_adapthist(meanIm12/np.max(meanIm12),clip_limit=0.005)


    meanIM_TT = meanIm11

    ims = []
    ixs_sets = []
    for i in np.arange(5+rad,510-rad,2):
        for j in np.arange(50,450,2):
            t_ = meanIM_TT[i-rad:i+rad,j-rad:j+rad]
            if np.max(t_)==0:
                ims.append(np.concatenate([np.zeros(t_.shape).flatten(),[np.mean(t_)]]))
            else:
                ims.append(np.concatenate([(t_/np.max(t_)).flatten(),[np.mean(t_)]]))
            #print np.max(t_),
            ixs_sets.append([i,j])


    labels_pred = rfC_fit.predict_proba(np.array([i.flatten() for i in ims]))>.9
    bouton_pred_ixs = np.where(labels_pred==1)[0]

    mask = np.zeros([512,512])
    #print len(bouton_pred_ixs)
    for i in bouton_pred_ixs:
        mask[ixs_sets[i][0],ixs_sets[i][1]] = 1

    mask = morphology.dilation(mask,morphology.disk(5))
    mask2 = morphology.erosion(mask,morphology.disk(4))

    label_im = skimage.morphology.label(mask2)

    mask_fin = np.zeros([512,512,4])
    clrs = (np.random.randint(0,255,size=(3,len(np.unique(label_im))))/255.)
    for i in np.unique(label_im)[1:]:
        mask_temp = np.zeros([512,512])
        nIxs = np.where(label_im==i)
        nIxs_flat = np.ravel_multi_index(nIxs,mask.shape)
        mask_temp[nIxs[0],nIxs[1]] = 1
        mask_temp = morphology.dilation(mask_temp,morphology.disk(4))
        mask_fin = mask_fin + (np.dstack([mask_temp[:,:,np.newaxis]]*4)*
            np.concatenate([clrs[:,i],[0.2]])[np.newaxis,np.newaxis,:])


    mask = np.dstack([mask,mask*.4,mask*.2,mask*.2])
    #print mask_fin.shape
    pred_boutons = []
    for c in np.array(ixs_sets)[bouton_pred_ixs]:
        xc = int(np.round(c[1]))
        yc = int(np.round(c[0]))
        t_ = (meanIM_TT[yc-rad:yc+rad,xc-rad:xc+rad])
        pred_boutons.append(t_/np.max(t_))
        
    pred_boutons = np.array(pred_boutons)
    #print len(pred_boutons)

    print "Found %s neurons" %(len(np.unique(label_im))-1)
    plt.figure(figsize=(12,12))
    plt.imshow(meanIM_TT ,cmap='binary_r')
    plt.imshow(mask_fin)
    #plt.scatter(np.array([i[1] for i in ixs_sets])[bouton_pred_ixs],
    #            np.array([i[0] for i in ixs_sets])[bouton_pred_ixs],s=8)
    plt.show()