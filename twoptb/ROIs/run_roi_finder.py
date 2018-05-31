#!/home/yves/anaconda2/bin/python



from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage import morphology
import h5py
from IPython import display
import os
import pickle
import sys
from skimage.exposure import equalize_adapthist
import skimage
import argparse
import twoptb

def find_classifier(path,search_str):

    all_classS = os.listdir(path)

    class_path = []

    for classN in all_classS:
        if search_str in classN:
            print 'selected classifier: %s' %classN
            class_path.append(classN)


    if (len(class_path)==0):
        print "No roi_finders match the name you suggested available ones are:"

        for i in all_classS:
            print i
        raise Exception("Try again using one of these")

    elif (len(class_path)>1):
        print "The search string you used is not specific enough and has matched multiple roi_finders: here are availabel ones"

        for i in all_classS:
            print i
        raise Exception("Try again using one of these")
    else:
        pass

    return class_path[0]

if __name__ =="__main__":

    class_path = os.path.join(os.path.split(os.path.split(os.path.realpath(__file__))[0])[0],'classifiers')

    parser = argparse.ArgumentParser(description="""Automatically find ROIs based on a mean image example usage is:

python run_roi_finder -sess 0 -ded 3 2 3 classifier_name /path/to/hdf5.py

Optional arguments, i.e. those with a hyphen preceding may be omitted note that here
0 refers to sess and 3 2 3 refers to ded
=============================================================================""",
                                                    formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("classifier", type=str,
                    help="name of classifier, write help to get names of availabel classifiers")


    parser.add_argument("hdfPath", type=str,
                    help="Full path to HDF5 file to open and search for ROIs in")

    
    parser.add_argument("-sess",type=int,dest='sess',default=1,
                    help="Session to run automatic ROI finding on")

    parser.add_argument("-ded" ,type=int,nargs=3,dest='ded',
                help=("Parameters for morphological operations for drawing ROIs. Required if zoom level is not 1 or 2. ded is short for dilation erosion dilation, which are run morphological operations"))


    parser.add_argument("-thresh", type=float, dest='thresh', default=.925, help="Threshold recommended range 0.8-.98 the higher the threshold the harder the inlcusion threshold")

    args = parser.parse_args()


    pth0 = os.path.join(os.path.split(twoptb.__file__)[0],'classifiers')

    class_path = find_classifier(pth0,args.classifier)

    rfC_fit,rad = pickle.load(open(os.path.join(pth0,class_path)))
    hdfF11 = h5py.File(args.hdfPath)

    #ded = [5,4,5] #for zoom 2

    areaIx = args.sess
    bhdf11 =  hdfF11[hdfF11.keys()[0]][ u'registered_data']
    areas = bhdf11.keys()
    print "extracting from %s" %(areas[areaIx])
    meanIm11 = bhdf11[areas[areaIx]].attrs['mean_image']

    gInfoP = os.path.join(os.path.split(args.hdfPath)[0],'GRABinfos',os.path.split(bhdf11[areas[areaIx]].attrs['GRABinfo'])[-1])
    zoom = pickle.load(open(gInfoP,'r'))['scanZoomFactor']

    if zoom==2:
        ded = [5,4,5]
    elif zoom==1:
        ded = [3,2,2]
    else:
        ded = args.ded
    meanIm11 = equalize_adapthist(meanIm11/np.max(meanIm11),clip_limit=0.005)


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


    labels_pred = rfC_fit.predict_proba(np.array([i.flatten() for i in ims]))>args.thresh
    bouton_pred_ixs = np.where(labels_pred==1)[0]

    mask = np.zeros([512,512])
    #print len(bouton_pred_ixs)
    for i in bouton_pred_ixs:
        mask[ixs_sets[i][0],ixs_sets[i][1]] = 1

    mask = morphology.dilation(mask,morphology.disk(ded[0]))
    mask2 = morphology.erosion(mask,morphology.disk(ded[1]))

    label_im = skimage.morphology.label(mask2)

    mask_fin = np.zeros([512,512,4])

    ROI_attrs = {'centres': [],
                 'patches': [],
                 'idxs': [],
                 'masks':[],
                 'traces':[]
                 }

    roi_idxs = []
    clrs = (np.random.randint(0,255,size=(3,len(np.unique(label_im))))/255.)
    for i in np.unique(label_im)[1:]:
        mask_temp = np.zeros([512,512])
        nIxs = np.where(label_im==i)
        nIxs2 = np.where(mask_temp>0)

        if not np.logical_or(np.any(np.array(nIxs)<20),np.any(np.array(nIxs)>490)):
                #np.logical_or.reduce(np.any(np.array(nIxs[0]).flatten()<20),
                #                    np.any(np.array(nIxs[0]).flatten()>490),
                #                    np.any(np.array(nIxs[1]).flatten()<20),
                #                    np.any(np.array(nIxs[1]).flatten()>490)):
            #masks.append(mask)


            #roi_idxs.append(nIxs)

            nIxs_flat = np.ravel_multi_index(nIxs,mask.shape)
            mask_temp[nIxs[0],nIxs[1]] = 1



            mask_temp = morphology.dilation(mask_temp,morphology.disk(ded[2]))

            nIxs2 = np.where(mask_temp>0)

            ROI_attrs['idxs'].append([nIxs2[1],nIxs2[0]])
            ROI_attrs['centres'].append(np.flipud(np.mean(np.array(nIxs2),axis=1)))
            ROI_attrs['patches'].append(np.nan)

            #PRETTY SURE THESE TWO ARE THE CORRECT WAY AROUND BUT NOT 100%...
            xLims = [np.min(nIxs2[1])-10,np.max(nIxs2[1])+10]
            yLims = [np.min(nIxs2[0])-10,np.max(nIxs2[0])+10]

            ROI_attrs['masks'].append(mask_temp[yLims[0]:yLims[1],xLims[0]:xLims[1]])
            ROI_attrs['traces'].append([0])


            mask_fin = mask_fin + (np.dstack([mask_temp[:,:,np.newaxis]]*4)*
                np.concatenate([clrs[:,i],[0.2]])[np.newaxis,np.newaxis,:])
        else:
            pass


    mask = np.dstack([mask,mask*.4,mask*.2,mask*.2])
    #print mask_fin.shape
    print ROI_attrs['centres'][-1]

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

    yn = raw_input("save these ROIs?: (y/n)\nWarning will overwrite any you have drawn manually...")

    if yn=='y':

        fName = bhdf11[areas[areaIx]].name[1:].replace('/','-') + '_ROIs.p'
        #print os.path.join(self.Folder,fName)
        Folder = os.path.split(os.path.abspath(bhdf11[areas[areaIx]].file.filename))[0]

        FLOC = os.path.join(Folder,'ROIs',fName)
        with open(FLOC,'wb') as f:
            pickle.dump(ROI_attrs,f)

        bhdf11[areas[areaIx]].attrs['ROI_dataLoc'] = FLOC
        hdfF11.close()
        print 'saved ROIs for session %s' %(areas[areaIx])
    else:
        print "done!"



