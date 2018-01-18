import h5py
import sys
import pickle
import copy as cp
import os
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



def extract_traces(areaFile,roiattrs):
        
    nROIs = len(roiattrs['idxs'])
    len_trace = areaFile.shape[0]
    print len_trace, '\n'


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



def save_ROIS(areaFile,ROI_attrs):

    ROI_attrs = extract_traces(areaFile,ROI_attrs)
    print os.path.abspath(areaFile.file.filename)
    Folder = os.path.split(os.path.abspath(areaFile.file.filename))[0]
    fName = areaFile.name[1:].replace('/','-') + '_ROIs.p'
    #print os.path.join(self.Folder,fName)
    FLOC = os.path.join(Folder,'ROIs',fName)
    with open(FLOC,'wb') as fi:
        pickle.dump(ROI_attrs,fi)

    areaFile.attrs['ROI_dataLoc'] = FLOC
    print 'ROI MASKS SAVED'





if __name__=='__main__':
    
    hdf_path = os.path.abspath(sys.argv[1])
    #roi_pth = os.path.join(hdf_path[:-3],'ROIs')
    with h5py.File(hdf_path,'a',libver='latest') as hdf:
        keys = hdf.keys()
        Folder = os.path.split(os.path.abspath(hdf.filename))[0]
        roi_pth = os.path.join(Folder,'ROIs')
        print '\n\n'
        f = hdf[keys[0]]['registered_data']


        all_sessions = list((i for i in f.iterkeys()))

        sessions = [i for i in all_sessions if ( 'ROI_dataLoc' in f[i].attrs.keys())]

        new_sess= []
        for i in sessions:
            if os.path.exists(f[i].attrs['ROI_dataLoc']):
                new_sess.append(i)
        sessions = new_sess

        #print sessions
        #if len(f[i].attrs['ROI_dataLoc'])!=0))

        #for idx,fn in enumerate(sessions):
            #print idx, fn
        #session = int(raw_input('Select Seed Session: '))

        roiFs = [os.path.join(roi_pth,i) for i in os.listdir(roi_pth)]
        lIdx = np.argmax([os.path.getsize(i) for i in roiFs])
        largest = roiFs[lIdx]
        lIdx = int(np.where([i in largest for i in sessions])[0])
        print "largest:", largest, lIdx, '\n\n'
        newest = max(roiFs , key = os.path.getctime)
        nIdx = int(np.where([i in newest for i in sessions])[0])
        print "newest:", newest 

        print lIdx,nIdx, "\n"
        #Idx = nIdx; should_be = newest
        if largest!=newest:
            resp = raw_input('how do you want to continue? (size/new/none):')
            print str(resp)

            if str(resp)=='size':
                print "using largest file"
                Idx = lIdx
                print [sessions[Idx]]
                should_be = largest
            elif str(resp)=='new':
                print "using newest file:",
                Idx = nIdx
                print [sessions[Idx]]

                should_be = newest
            else:
                raise "Largest File is not newest, be careful in erasing"
        #print newest,largest
        ROILoc = f[sessions[Idx]].attrs['ROI_dataLoc']
        print ROILoc
        if ROILoc!=should_be:

            raise "Something went wrong, wrong file has been selected"
        with open(ROILoc) as f_i:
                dat = pickle.load(f_i)

        for s in all_sessions:
            if s!=sessions[Idx]:
                print s
                save_ROIS(f[s],dat)

    print 'done!'
                



