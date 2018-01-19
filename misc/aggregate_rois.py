
""" This script simply takes as input some folders with hdf5 datasets and 
    outputs the mean images as .jpg files. Can do multiple directions """


import sys
import h5py
import numpy as np
import os


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


if __name__=="__main__":

    n_basedirs = len(sys.argv) - 1
    
    im_path = os.path.join(sys.argv[1],'mean_images')
    if not os.path.isdir(im_path):
        os.mkdir(im_path)

    hdfPaths = []
    for dir_ix in range(1,1+n_basedirs):
        base_dir = sys.argv[dir_ix]
        for root, dirs, files in os.walk(base_dir):
            for fl in files:
                if fl.endswith(".h5"):
                     # print(os.path.join(root, fl)) 
                     hdfPaths.append(os.path.join(root,fl))


    ### Collect all the mean images and ROI locations
    for hdfPath in hdfPaths:
        hdf2 = h5py.File(hdfPath,'r',libver='latest')

        d = hdf2.keys()[0]
        for i,sess in enumerate(hdf2[d]['registered_data'].keys()):
            if i==0:
                meanIm2 = hdf2[d]['registered_data'][sess].attrs['mean_image']
            else:
                meanIm2 += hdf2[d]['registered_data'][sess].attrs['mean_image']

        meanIm2 /= i
        hdf2.close()
        plt.figure(figsize=(12,12))
        plt.imshow(meanIm2,cmap='binary_r',interpolation='None')

        l = 32
        plt.vlines(np.arange(0,512+l,l),0,512,color='w')
        plt.hlines(np.arange(0,512+l,l),0,512,color='w')

        pth = os.path.join(im_path,os.path.split(hdfPath)[-1][:-3]+'.jpg')
        plt.savefig(pth,dpi=100)
        plt.clf()