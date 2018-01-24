
""" This script simply takes as input some folders with hdf5 datasets and 
    outputs the mean images as .jpg files. Can do multiple directions """

from __future__ import division
import sys
import h5py
import numpy as np
import os
import pickle
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
import copy as cp
import matplotlib.pyplot as plt

def findpath():
    cDir = os.path.dirname(os.path.realpath(__file__))

    found = False
    while not found:
        cDir,ext = os.path.split(cDir) 
        if ext=='twoptb':
            found = False
            twoptb_path = cDir
            break
    return twoptb_path

twoptb_path = findpath()
sys.path.append(twoptb_path)

import twoptb as MP
#________________ Helper Functions _________________________________

def get_gradsum(im):
    return np.sum(np.abs(np.gradient(im)))

def get_hdf_paths(n_basedirs,in_args):

    """ Returns list of all hdf paths"""
    hdfPaths = []
    for dir_ix in range(1,1+n_basedirs):
        base_dir = in_args[dir_ix]
        for root, dirs, files in os.walk(base_dir):
            for fl in files:
                if fl.endswith(".h5"):
                     # print(os.path.join(root, fl)) 
                     hdfPaths.append(os.path.join(root,fl))

    return hdfPaths


def get_patch_mask(roi_idxs,meanIm):
    centroid = 1



    return None
def get_overlap_ids(roi_locs,roiinfo2,thresh=.4):

    """ This returns cells where there is a lot of overlap 
        good/bad_idxs2 relates to roiinfo2, good_idxs relates to roi_locs




    """
    rois1 = range(len(roi_locs))
    rois2= range(len(roiinfo2['idxs']))
    roi_pairs = cartesian([rois1,rois2])

    overlap = np.zeros([len(rois1),len(rois2)])
    for i,j in roi_pairs:
        #these should be indices of rois
        roiset1 = set(map(tuple,np.array(roi_locs[i]).T.tolist()))
        roiset2 = set(map(tuple,np.array(roiinfo2['idxs'][j]).T.tolist()))

        overlap[i,j] = len(roiset1.intersection(roiset2))/np.min([len(roiset1),len(roiset2)])

    good_idxs1 = np.where(np.max(overlap,axis=1)>thresh)[0]
    bad_idxs1 = np.where(np.max(overlap,axis=1)<=thresh)[0]
    #print np.mean(np.max(overlap,axis=1))
    #print np.max(np.max(overlap,axis=1))
    good_idxs2 = np.argmax(overlap,axis=1)[good_idxs1]
    bad_idxs2 = np.argmax(overlap,axis=1)[bad_idxs1]



    estimated_pairs = [(i,j) for i,j in zip(good_idxs1,good_idxs2)]

    return estimated_pairs, good_idxs1, good_idxs2, bad_idxs1, bad_idxs2





def patch_register_roi_locs(meanIm1,meanIm2,roiinfo1,l=32):
    """ Function that takes as input two mean images and 
        roi coordinates on one of the images and returns 
        the ROI coordinates on a second image
        
        Arguments:
        ============================
        
        meanIm1:    np.array
                    mean image with known roi coordinates
        
        meanIm2:    np.array
                    mean image to map rois onto
                 
        roiinfo1:   dict
                    contains field idxs which is a list of arrays of 
                    roi coordinates on meanIm1
                
        l:          int
                    size of grid on which to motion register patches
        """
    
    xsz,ysz = meanIm1.shape

    mask_num = np.zeros([512,512])

    for n_,idxp in enumerate(roiinfo1['idxs']):
        #if n_ in table[table.columns[1]].tolist():
        mask_num[idxp[1],idxp[0]] = n_
        nPatch = xsz/float(l)

    #First register the whole images in rigid fashion to one another
    out = MP.image_registration.Register_Image(meanIm1,meanIm2)
    shift_all = out[0]
    regIm2 =  np.fft.ifftn(fourier_shift(np.fft.fftn(meanIm2), -shift_all)).real


    rois1 = []
    im_pairs = []
    roi_pairs = []
    roi_locs = cp.deepcopy(roiinfo1['idxs'])
    moved_rois = []
    shifts = []
    for i in range(int(nPatch)):


        sty = i*l
        for j in range(int(nPatch)):

            stx = j*l
            patch = meanIm1[stx:stx+l,sty:sty+l]
            bigpatch = patch#np.pad(patch,maxShift,mode='median')
            shift,_,_ = register_translation(bigpatch,
                                             regIm2[stx:stx+l,sty:sty+l]
                                             ,upsample_factor=5.)
            #print shift
            if np.any(shift>15):
                shift = np.array([0,0])

            rois_patch = np.unique(mask_num[stx:stx+l,sty:sty+l]).tolist()

            for roi in rois_patch:
                roi = int(roi)
                if roi not in moved_rois:
                    roi_locs[roi][0] = roi_locs[roi][0] - 1*int(shift[1]) + 1*int(shift_all[1])
                    roi_locs[roi][1] = roi_locs[roi][1] - 1*int(shift[0]) + 1*int(shift_all[0])

            moved_rois = moved_rois + rois_patch



    return roi_locs,shift_all



def load_data(hdfDir):

    """ This loads the data from hdf file
    
    Arguments:
    ===========================

    hdfDir:     str
                path to hdf file

    Returns:
    ==========================

    meanIm2:    np.array
                image array

    roiinfo     dict
                dictionary with properties of the rois

    """
    hdf2 = h5py.File(hdfDir,'r',libver='latest')
    print hdfDir
    d = hdf2.keys()[0]

    sharpness = []
    for iiii,sess in enumerate(hdf2[d]['registered_data'].keys()):
        #sharpness.append(get_gradsum(hdf2[d]['registered_data'][sess].attrs['mean_image']))
        sharpness.append(hdf2[d]['registered_data'][sess].attrs['mean_image'])
        #print 'jere'

        #if iiii==0:
        #    meanIm2 = hdf2[d]['registered_data'][sess].attrs['mean_image']
        #else:
        #    meanIm2 += hdf2[d]['registered_data'][sess].attrs['mean_image']


    #meanIm2 /= iiii
    #im_idx = np.argmax(np.asarray(sharpness))
    #bsess = (hdf2[d]['registered_data'].keys())[im_idx]
    #meanIm2 = hdf2[d]['registered_data'][bsess].attrs['mean_image']
    meanIm2 = np.max(np.array(sharpness),axis=0)
    print "image shape ", np.shape(meanIm2)

    roiInfoloc = hdf2[d]['registered_data'][hdf2[d]['registered_data'].keys()[0]].attrs['ROI_dataLoc']
    hdf2.close()
    print 'loading rois...'
    roiinfo = pickle.load(open(roiInfoloc))
    print 'done!'
    #plt.figure(figsize=(12,12))
    #plt.imshow(meanIm2,cmap='binary_r',interpolation='None')
    #plt.show(block=False)
    return meanIm2, roiinfo



def cartesian(pools):
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result


if __name__=="__main__":

    n_basedirs = len(sys.argv) - 1
    
    hdfPaths = get_hdf_paths(n_basedirs,sys.argv)
    #hdfPaths = ['/media/yves/imaging_yves/attention_data/tupac_attention_curated/28072017/processed/28072017_tupac_28072017/28072017_tupac_28072017.h5',
    #   '/media/yves/imaging_yves/tupac_learning/24072017/processed/24072017_tupac_24072017/24072017_tupac_24072017.h5',
    #   ]


    globalIM,globalROI = load_data(hdfPaths[0])

    backup = cp.deepcopy(globalROI)
    all_ROI = {'idxs': globalROI['idxs'],
               'centres': globalROI['centres'],
               'orig_index': range(len(globalROI['idxs']))}

    nCells_day = len(all_ROI['idxs'])

    all_ROI['sess'] = [hdfPaths[0]]*nCells_day

    #duplicate list is a 2 element list. The first entry is what number is it in the global list
    #the second entry is a dict. Fields are referenced by the day
    duplicate_list = []


    all_seen = 0
    all_seen += len(all_ROI['idxs'])
    ### Collect all the mean images and ROI locations
    #there is a bug somewhere
    roiinfo_set = [globalROI]


    ### This block of code runs this for the first day, getting an overall mask

    for hdfPath in hdfPaths[1:]:
        print hdfPath
        if 1:#'29' not in hdfPath:
            try:
                meanIm2,roiinfo2 = load_data(hdfPath)
                nCells_day = len(roiinfo2['idxs'])

                roiinfo2['sess'] = [hdfPath]*nCells_day

                roiinfo_set.append(roiinfo2)
                #print 'loaded'
                ## Returns the locations of ROIs of meanIm2 mapped onto globalIM
                roi_locs, shift_all = patch_register_roi_locs(meanIm2,globalIM,roiinfo2,l=32)
                #print 'regged'
                #bad_idxs2 are the rois that are present in meanIm2 that are not present in the global one
                estimated_pairs, good_idxs1, good_idxs2, bad_idxs1, bad_idxs2 = get_overlap_ids(roi_locs,
                                                                                                all_ROI,
                                                                                                thresh=.3)


                # Here run day refers to the day that is currently run in the loop
                #importantly, the curr duplicates are 
                curr_duplicates = [i[0] for i in duplicate_list]

                for runn_day, glob_day in estimated_pairs:
                    if glob_day in curr_duplicates:
                        ix_ = np.where(np.array(curr_duplicates)==glob_day)[0]
                        if len(ix_)==1:
                        #print type(duplicate_list[ix_][1])
                            duplicate_list[ix_][1][hdfPath] = runn_day
                        elif len(ix_)>0:
                            for ixx_ in ix_:
                                duplicate_list[ixx_][1][hdfPath] = runn_day

                    else:
                        duplicate_list.append([glob_day,{hdfPath:runn_day}])

                all_seen += len(roiinfo2['idxs'])
                print len(bad_idxs1)/float(len(good_idxs1)+len(bad_idxs1))
                #bad idxs1 should be the rois from roi_locs
                for i_,idx in enumerate(bad_idxs1):

                    
                    all_ROI['idxs'].append(roi_locs[idx])
                    all_ROI['centres'].append(np.mean(roi_locs[idx],axis=1))
                    all_ROI['sess'].append(hdfPath)
                    ### THE INDEX OF THIS ROI IN THE ORIGINAL SESSION; NOT SURE THIS IS CORRECT
                    ### MAYBE SHOULD BE bad_idxs2/1??
                    all_ROI['orig_index'].append(bad_idxs1[i_])   
                   
            except UnboundLocalError:
                print "!!!!! WARNING SESSION %s not loaded !!!!!" %hdfPath 
                pass


     ### This block of code runs over the other days like this
     #specifically, for each hdfPath you map all the ROIs over, unless

    print '\nRunning over other areas...\n'
    for hdfPath in hdfPaths[1:]:




        meanIm2,roiinfo2 = load_data(hdfPath)
        nCells_day = len(roiinfo2['idxs'])

        roiinfo2['sess'] = [hdfPath]*nCells_day

        roiinfo_set.append(roiinfo2)
        #print 'loaded'

        ## here, meanIm2 and globalIM are reversed relative to one another
        #Here the roi_locs are the locations of ALL Rois from all images shifted onto
        #the hdfPath image
        roi_locs, shift_all = patch_register_roi_locs(meanIm2,globalIM,all_ROI,l=32)


        #In this block of code deal with the bad_idx ROIs
        for i_ in range(len(all_ROI['sess'])):
            if all_ROI['sess'][i_]==hdfPath:

                roi_locs[i_] = roiinfo2['idxs'][all_ROI['orig_index'][i_]]



        #In this block of code deal with the duplicate ROIs
        #for i in

        #POTENTIAL BUG FROM ABOVE IS TWO MAP ON SAME DAY DOES IT OVERWRITE? YES BECAUSE PROBABLY CLOSE TOGETHER
        dupl_glob = [i[0] for i in duplicate_list]
        
        # the i_ here is the cells in the glbal ROI that have duplicates
        for ix1, i_ in enumerate(dupl_glob):
            if hdfPath in duplicate_list[ix1][1].keys():
                #ixx_ = np.where(np.array(dupl_glob)==i_)[0]
                #print type(ixx_)
                ixx_ = np.array([duplicate_list[ix1][1][hdfPath]]).astype('int')
                if len(ixx_)>0:
                    ixx_ = ixx_[0]
                #print type(duplicate_list[ixx_][1][hdfPath])
                #if type(ixx_)
                roi_locs[i_] = roiinfo2['idxs'][ixx_]

            #duplicate_list.append([glob_day,{hdfPath:runn_day}])  #here for refernce

        print len(roi_locs)
        plt.figure()

        maskNew = np.zeros([512,512,4])
        for n_,idxp in enumerate(roi_locs):
            xpos = np.clip(idxp[1].astype('int'),0,511)
            ypos = np.clip(idxp[0].astype('int'),0,511)

            maskNew[xpos,ypos,0] = 1
            maskNew[xpos,ypos,3] = 1

            #plt.text(np.mean(ypos),np.mean(xpos),str(n_),color='green')


        for n_,idxp in enumerate(roiinfo2['idxs']):
            xpos = np.clip(idxp[1].astype('int'),0,511)
            ypos = np.clip(idxp[0].astype('int'),0,511)

            maskNew[xpos,ypos,1] = 1
            maskNew[xpos,ypos,3] = 1



        plt.title(hdfPath)
        plt.imshow(meanIm2,cmap='binary_r',interpolation='None')
        plt.imshow(maskNew,alpha=.075,interpolation='None')





    #### END OF FOR LOOP #########
    print len(backup['idxs']),len(all_ROI['idxs']), all_seen
    #print duplicate_list
    maskNew = np.zeros([512,512,4])

    plt.figure()

    for n_,idxp in enumerate(all_ROI['idxs']):
        xpos = np.clip(idxp[1].astype('int'),0,511)
        ypos = np.clip(idxp[0].astype('int'),0,511)

        maskNew[xpos,ypos,0] = 1
        maskNew[xpos,ypos,3] = 1
        plt.text(np.mean(ypos),np.mean(xpos),str(n_),color='green')

    plt.imshow(globalIM,cmap='binary_r',interpolation='None')
    plt.imshow(maskNew,alpha=.075,interpolation='None')

    plt.show()


#mask1 =np.ma.masked_where(mask1==False,mask)


#As a test on the final version of this show the ROIs that are the same



