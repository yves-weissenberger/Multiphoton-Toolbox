from __future__ import division
import h5py
import pickle
import copy as cp
import numpy as np
import sys
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
#sys.path.append(os.path.abspath())
import twoptb as MP


def extract_traces(areaFile,roiattrs):
        
    nROIs = len(roiattrs['idxs'])
    len_trace = areaFile.shape[0]


    roiattrs['traces'] = np.zeros([nROIs,len_trace])
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
                                                            
        roiattrs['traces'][idx] = np.nanmean(temp,  axis=(1,2))
    return roiattrs


       
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
        xLims = [np.min(mpossx)-10,np.max(mpossx)+10]
        yLims = [np.min(mpossy)-10,np.max(mpossy)+10]
        temp = areaF[:,yLims[0]:yLims[1],xLims[0]:xLims[1]] *np.abs(roi_attrs['masks'][idx]-1)
        temp = temp.astype('float64')
        temp[temp==0] = np.nan
        neuropil_trace = np.nanmean(temp,axis=(1,2))



        temp = areaF[:,yLims[0]:yLims[1],xLims[0]:xLims[1]] *roi_attrs['masks'][idx]
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

        roiattrs['traces'][idx] = roiattrs['traces'][idx] - MP.process_data.runkalman(roiattrs['traces'][idx],50000)
        baseline = MP.process_data.runkalman(roiattrs['corr_traces'][idx],50000)
        roiattrs['corr_traces'][idx] = roiattrs['corr_traces'][idx] - baseline
        roiattrs['dfF'][idx] = roiattrs2['corr_traces'][idx]/baseline
        if 'neuropil_traces' in roiattrs.keys():
            roiattrs['neuropil_traces'][idx] -= MP.process_data.runkalman(roiattrs['neuropil_traces'][idx],5000)


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
    
    hdf_path = os.path.abspath(sys.argv[1])
    npc = sys.argv[2]=='y'
    kf = sys.argv[3]=='y'
    print "neuropil correct: %s" %npc
    print "kalman filter: %s" %kf
    
    #roi_pth = os.path.join(hdf_path[:-3],'ROIs')
    with h5py.File(hdf_path,'a',libver='latest') as hdf:
        keys = hdf.keys()
        Folder = os.path.split(os.path.abspath(hdf.filename))[0]
        roi_pth = os.path.join(Folder,'ROIs')
        print '\n\n'
        f = hdf[keys[0]]['registered_data']


        sessions = list((i for i in f.iterkeys()))
        n_sess = len(sessions)
        for idx,fn in enumerate(sessions):

            print '\narea %s/%s: %s' %(idx,n_sess,fn)

            areaFile = f[fn]
            if 'ROI_dataLoc' in areaFile.attrs.keys():

                FLOC = areaFile.attrs['ROI_dataLoc']
            else:
                Folder = os.path.split(os.path.abspath(areaFile.file.filename))[0]
                fName = areaFile.name[1:].replace('/','-') + '_ROIs.p'
                FLOC = os.path.join(Folder,'ROIs',fName)
                areaFile.attrs['ROI_dataLoc'] = FLOC

            roiattrs = pickle.load(open(FLOC,'r'))
            if npc:
                roiattrs2 = neuropil_correct(areaFile,roiattrs)
            else:
                roiattrs2 = roiattrs
                roiattrs2['corr_traces'] = roiattrs['traces']
            print "\n"
            if kf:
                roiattrs2 = baseline_correct(roiattrs2)
            roiattrs2 = extract_spikes(roiattrs2)
            with open(FLOC,'wb') as fi:
                pickle.dump(roiattrs2,fi)



    print 'done!'
                

