from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
import twoptb
import os
import copy as cp
from skimage import morphology
import skimage
import seaborn
import argparse
seaborn.set(style='ticks',font_scale=3)


if __name__=="__main__":



    parser = argparse.ArgumentParser(description=""" Merge ROIs drawn on different fields intended for use together with 
        run_roi_finder with the -sess argument set to -1
                                                    """)


    parser.add_argument("hdfPath", type=str,
                help="Full path to HDF5 file to open and search for ROIs in")

    parser.add_argument("-clean",type=int,dest='clean',default=65,
        help="Remove overly large ROIs? Specify size. Default size set to 65")

    parser.add_argument("-plot",type=bool,default=1,dest='plot',
                        help='show ROI mask after running?')


    args = parser.parse_args()

    to_clean = int(args.clean)


    with h5py.File(args.hdfPath,'r') as hdf:

        baseF = hdf[hdf.keys()[0]]['registered_data']
        areas = baseF.keys()

        rois1 = pickle.load(open(baseF[areas[0]].attrs['ROI_dataLoc'],'r'))


        allROIs = cp.deepcopy(rois1)
        added = 0
        not_added = 0
        for ar in areas[1:]:
            print(ar,)
            roisX = pickle.load(open(baseF[ar].attrs['ROI_dataLoc'],'r'))
            print(len(roisX['idxs']))
            for ij,c in enumerate(roisX['centres']):

                if np.any([np.sum((c - i)**2)<10 for i in allROIs['centres']]):
                    not_added += 1
                    pass
                else:
                    added += 1
                    for k in allROIs.keys():
                        allROIs[k].append(roisX[k][ij])
        if to_clean:
            lenS = []
            for i,pck in enumerate(allROIs['idxs']):
                if len(pck[0])>to_clean:
                    lenS.append(i)

            for k in allROIs.keys():
                for i in sorted(lenS,reverse=1):
                    del allROIs[k][i]




        if args.plot:

            mask = np.zeros([512,512])
            for i,j in allROIs['idxs']:
                mask[j,i] = 1

            roi_idxs = []
            clrs = np.random.randint(0,255,size=(3,len(allROIs['idxs'])))/255.
            kk = 0
            mask_temp = np.zeros([512,512])
            mask_fin = np.zeros([512,512,4])

            for nIxs in allROIs['idxs']:
              
                mask_temp[nIxs[1],nIxs[0]] = 1
                mask_fin = mask_fin + (np.dstack([mask_temp[:,:,np.newaxis]]*4)*
                    np.concatenate([clrs[:,kk],[0.2]])[np.newaxis,np.newaxis,:])
                mask_temp = np.zeros([512,512])
                kk += 1


            print ("Found %s neurons" %(len(allROIs['idxs'])))
            plt.figure(figsize=(16,16))
            plt.imshow(np.max(np.array([baseF[ar].attrs['mean_image'] for ar in areas]),axis=0) ,cmap='binary_r')
            plt.imshow(mask_fin)
            plt.show(block=1)

            for ar in areas:
                with open(baseF[ar].attrs['ROI_dataLoc'],'wb') as f:
                    pickle.dump(allROIs,f)
                print(1)      


