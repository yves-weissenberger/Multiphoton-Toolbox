
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
    
    hdfPaths = get_paths(n_basedirs,sys.argv)

    globalIM,globalROI = load_data(hdfPaths[0])


    ### Collect all the mean images and ROI locations
    for hdfPath in hdfPaths[1:]:
        
        meanIm2,roiinfo2 = load_data(hdfPath)

        ## Returns the locations of ROIs of meanIm2 mapped onto globalIM
        roi_locs = patch_register_roi_locs(meanIm2,globalIM,roiinfo2,mask_num,l=32)

        #bad_idxs2 are the rois that are present in meanIm2 that are not present in the global one
        estimated_pairs, good_idxs1, good_idxs2, bad_idxs1, bad_idxs2 = get_overlap_ids(roi_locs, globalROI, thresh=.5)






#________________ Helper Functions _________________________________


def get_paths(n_basedirs,in_args):

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


def get_overlap_ids(roi_locs,roiinfo2,thresh=.5):

    """ This returns cells where there is a lot of overlap 
        ???? Which one is which in the output
        good_idxs2 relates to roiinfo2, good_idxs relates to roi_locs

    """
    rois1 = range(len(roi_locs))
    rois1 = range(len(roiinfo2))

    overlap = np.zeros([len(rois1),len(rois2)])
    for i,j in roi_pairs:
        roiset1 = set(map(tuple,np.array(roi_locs[i]).T.tolist()))
        roiset2 = set(map(tuple,np.array(roiinfo2['idxs'][j]).T.tolist()))
        overlap[i,j] = len(roiset1.intersection(roiset2))/np.min([len(roiset1),len(roiset2)])

    good_idxs1 = np.where(np.max(overlap,axis=1)>thresh)[0]
    bad_idxs1 = np.where(np.max(overlap,axis=1)<=thresh)[0]

    good_idxs2 = np.argmax(overlap,axis=1)[good_idxs1]
    bad_idxs2 = np.argmax(overlap,axis=1)[good_idxs2]

    estimated_pairs = [(i,j) for i,j in zip(good_idxs1,good_idxs2)]

    return estimated_pairs, good_idxs1, good_idxs2, bad_idxs1, bad_idxs2





def patch_register_roi_locs(meanIm1,meanIm2,roiinfo1,mask_num,l=32):
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
            if np.any(shift>15):
                shift = np.array([0,0])

            rois_patch = np.unique(mask_num[stx:stx+l,sty:sty+l]).tolist()

            for roi in rois_patch:
                roi = int(roi)
                if roi not in moved_rois:
                    roi_locs[roi][0] = roi_locs[roi][0] - 1*int(shift[1]) + 1*int(shift_all[1])
                    roi_locs[roi][1] = roi_locs[roi][1] - 1*int(shift[0]) + 1*int(shift_all[0])

            moved_rois = moved_rois + rois_patch



    return roi_locs



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

    d = hdf2.keys()[0]

    for i,sess in enumerate(hdf2[d]['registered_data'].keys()):
        if i==0:
            meanIm2 = hdf2[d]['registered_data'][sess].attrs['mean_image']
        else:
            meanIm2 += hdf2[d]['registered_data'][sess].attrs['mean_image']


    meanIm2 /= i

    roiInfoloc = hdf2[d]['registered_data'][sess].attrs['ROI_dataLoc']
    hdf2.close()
    print 'loading rois...'
    roiinfo = pickle.load(open(roiInfoloc))
    print 'done!'
    return meanIm2, roiinfo



def cartesian(pools):
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result