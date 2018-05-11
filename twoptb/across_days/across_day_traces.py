#!/home/yves/anaconda2/bin/python


from __future__ import division
import h5py
import pickle
import copy as cp
import numpy as np
import sys
import os
from scipy.ndimage.morphology import binary_fill_holes
import matplotlib.pyplot as plt
import argparse
import twoptb as MP

def get_paths(n_basedirs,in_args):

    """ Returns list of all hdf paths"""
    roiPaths = []
    hdfPaths = []
    for dir_ix in range(n_basedirs):
        base_dir = in_args[dir_ix]
        for root, dirs, files in os.walk(base_dir):
            for fl in files:
                if fl.endswith("glob.p"):
                     # print(os.path.join(root, fl)) 
                     roiPaths.append(os.path.join(root,fl))
                if fl.endswith(".h5"):
                     # print(os.path.join(root, fl)) 
                     hdfPaths.append(os.path.join(root,fl))


    return [(i,j) for i,j in zip(roiPaths,hdfPaths)]



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

def extract_traces(areaFile,roiattrs):
        
    nROIs = len(roiattrs['idxs'])
    len_trace = areaFile.shape[0]


    traces = np.zeros([nROIs,len_trace])
    for idx in range(nROIs):
        sys.stdout.write('\r Extracting_Trace_from roi: %s' %idx)
        sys.stdout.flush()
        mpossx= roiattrs['idxs'][idx][0]
        mpossy = roiattrs['idxs'][idx][1]
        xLims = [np.min(mpossx)-10,np.max(mpossx)+10]
        yLims = [np.min(mpossy)-10,np.max(mpossy)+10]

        temp = areaFile[:,yLims[0]:yLims[1],xLims[0]:xLims[1]] *roiattrs['masks'][idx]
        temp = temp.astype('float64')
        temp[temp==0] = np.nan
                                                            
        traces[idx] = np.nanmean(temp,  axis=(1,2))
    return traces


       
def neuropil_correct(areaF,roi_attrs):
    


    nROIs = len(roiattrs['idxs'])
    len_trace = areaFile.shape[0]


    roiattrs['traces'] = np.zeros([nROIs,len_trace])
    roiattrs['neuropil_traces'] = np.zeros([nROIs,len_trace])
    roiattrs['corr_traces'] = np.zeros([nROIs,len_trace])
    for idx in range(nROIs):

        sys.stdout.write('\r Extracting_Trace_from roi: %s' %idx)
        sys.stdout.flush()

        mpossx= roi_attrs['idxs'][idx][0]
        mpossy = roi_attrs['idxs'][idx][1]



        xLims = [np.clip(np.min(mpossx)-10,0,510),np.clip(np.max(mpossx)+10,0,510)]
        yLims = [np.clip(np.min(mpossy)-10,0,510),np.clip(np.max(mpossy)+10,0,510)]

        temp_mask = np.zeros(areaFile.shape[1:])
        temp_mask[mpossx,mpossy] = 1
        temp_mask = binary_fill_holes(temp_mask).T
        mask = temp_mask[yLims[0]:yLims[1],xLims[0]:xLims[1]]
        print '___',np.sum(mask),

        im_mask = np.dstack([mask,mask*0,mask*0,mask*.2])
        #plt.imshow(np.mean(areaF[:,yLims[0]:yLims[1],xLims[0]:xLims[1]],axis=0),cmap='binary_r')
        #plt.imshow(im_mask)
        #plt.show()



        temp = areaF[:,yLims[0]:yLims[1],xLims[0]:xLims[1]] *np.abs(mask-1)
        temp = temp.astype('float64')
        temp[temp==0] = np.nan
        neuropil_trace = np.nanmean(temp,axis=(1,2))



        temp = areaF[:,yLims[0]:yLims[1],xLims[0]:xLims[1]] *mask
        temp = temp.astype('float64')
        temp[temp==0] = np.nan
        trace = np.nanmean(temp,axis=(1,2))
        corrected_trace = trace - .4*neuropil_trace

        roiattrs['traces'][idx] = trace
        roiattrs['neuropil_traces'][idx] = neuropil_trace
        roiattrs['corr_traces'][idx] = corrected_trace


    return roiattrs


def baseline_correct(roiattrs):
    import copy as cp
    """ Correct for drifting baseline"""
    nROIs = len(roiattrs['idxs'])
    cFrames = np.array(roiattrs['traces']).shape[1]

    roiattrs['dfF'] = np.zeros([nROIs,cFrames])
    roiattrs['raw_traces'] = cp.deepcopy(roiattrs['traces'])
    for idx in range(nROIs):
        sys.stdout.write('\r Baseline Correcting ROI: %s' %idx)
        sys.stdout.flush()

        roiattrs['traces'][idx] = roiattrs['traces'][idx] - MP.process_data.runkalman(roiattrs['traces'][idx],500000,5000)
        baseline = MP.process_data.runkalman(roiattrs['corr_traces'][idx],500000,5000)
        roiattrs['corr_traces'][idx] = roiattrs['corr_traces'][idx] - baseline
        roiattrs['dfF'][idx] = roiattrs2['corr_traces'][idx]/baseline
        if 'neuropil_traces' in roiattrs.keys():
            roiattrs['neuropil_traces'][idx] -= MP.process_data.runkalman(roiattrs['neuropil_traces'][idx],500000,5000)


    return roiattrs

def extract_spikes(roiattrs):

    """ Infer approximate spike rates """
    print "\nrunning spike extraction"
    import c2s

    frameRate = 25
    if 'corr_traces' in roiattrs.keys():
        trace_type = 'corr_traces'
    else:
        trace_type = 'traces'
    data = [{'calcium':np.array([i]),'fps': frameRate} for i in roiattrs[trace_type]]
    spkt = c2s.predict(c2s.preprocess(data),verbosity=0)

    nROIs = len(roiattrs['idxs'])
    cFrames = np.array(roiattrs['traces']).shape[1]

    spk_traces = np.zeros([nROIs,cFrames])
    spk_long = []
    for i in range(nROIs):
        spk_traces[i] = np.mean(spkt[i]['predictions'].reshape(-1,4),axis=1)
        spk_long.append(spkt[i]['predictions'])

    roiattrs['spike_inf'] = spk_traces
    roiattrs['spike_long'] = np.squeeze(np.array(spk_long))
    return roiattrs




if __name__=='__main__':

    npc = sys.argv[1]=='y'
    kf = sys.argv[2]=='y'
    n_basedirs = len(sys.argv[3:])
    print type(sys.argv[3:])

    paths = get_paths(n_basedirs,sys.argv[3:])
    print paths[0][1]
    print paths[0][0]
    #print os.path.split(paths[0][1])
    #across_day_traces = os.path.join(os.path.split)
    #if not os.path.isdir(im_path):
    #    os.mkdir(im_path)

    print "neuropil correct: %s" %npc
    print "kalman filter: %s" %kf
    #im_path = os.path.join(sys.argv[1],'reg_images')
    #            if not os.path.isdir(im_path):
    #                os.mkdir(im_path)

    #roi_pth = os.path.join(hdf_path[:-3],'ROIs')
    if 1:
        for roiP,hdf_path in paths:

            trace_path = os.path.join(os.path.split(roiP)[0],'glob_traces')
            if not os.path.isdir(trace_path):
                os.mkdir(trace_path)

            with h5py.File(hdf_path,'a',libver='latest') as hdf:
                keys = hdf.keys()
                Folder = os.path.split(os.path.abspath(hdf.filename))[0]
                roi_pth = os.path.join(Folder,'ROIs')
                print '\n\n'
                f = hdf[keys[0]]['registered_data']


                sessions = list((i for i in f.iterkeys()))
                n_sess = len(sessions)
                for idx,fn in enumerate(sessions):

                    roiattrs = pickle.load(open(roiP,'r'))

                    trace_path_sess = os.path.join(trace_path)

                    print '\nsession %s/%s: %s' %(idx,n_sess,fn)

                    areaFile = f[fn]
                    #if 'ROI_dataLoc' in areaFile.attrs.keys():
                    #    FLOC = areaFile.attrs['ROI_dataLoc']
                    #else:
                    #    Folder = os.path.split(os.path.abspath(areaFile.file.filename))[0]
                    #    fName = areaFile.name[1:].replace('/','-') + '_ROIs.p'
                    #    FLOC = os.path.join(Folder,'ROIs',fName)
                    #    areaFile.attrs['ROI_dataLoc'] = FLOC

                    #roiattrs = pickle.load(open(FLOC,'r'))
                    #traces = extract_traces(areaFile,roiattrs)
                    if npc:
                        roiattrs2 = neuropil_correct(areaFile,roiattrs)
                    else:
                        roiattrs2 = roiattrs
                        roiattrs2['corr_traces'] = roiattrs['traces']
                    print "\n"
                    if kf:
                        roiattrs2 = baseline_correct(roiattrs2)
                    roiattrs2 = extract_spikes(roiattrs2)
                    
                    sess_path = os.path.join(trace_path,fn)
                    if not os.path.isdir(sess_path):
                        os.mkdir(sess_path)


                    for t_type in ['traces','neuropil_traces','corr_traces','raw_traces','dfF','spike_inf']:
                        file_loca = os.path.join(sess_path,fn + '_' + t_type)
                        np.save(file_loca, roiattrs2[t_type], allow_pickle=True, fix_imports=True)


            print 'done!'
                    

