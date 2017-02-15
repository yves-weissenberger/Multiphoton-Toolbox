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


def save_ROIS(areaFile,ROI_attrs):

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

        print sessions
        #if len(f[i].attrs['ROI_dataLoc'])!=0))

        #for idx,fn in enumerate(sessions):
            #print idx, fn
        #session = int(raw_input('Select Seed Session: '))

        roiFs = [os.path.join(roi_pth,i) for i in os.listdir(roi_pth)]
        lIdx = np.argmax([os.path.getsize(i) for i in roiFs])
        largest = roiFs[lIdx]
        print largest, '\n\n'
        newest = max(roiFs , key = os.path.getctime)
        print newest

        if largest!=newest:
            resp = str(raw_input('do you want to continue? (y/n):'))
            if resp=='y':
                print "using largest file"
            else:
                raise "Largest File is not newest, be careful in erasing"
        #print newest,largest
        ROILoc = f[sessions[lIdx]].attrs['ROI_dataLoc']
        print ROILoc
        if ROILoc!=largest:
            raise "Something went wrong, wrong file has been selected"
        with open(ROILoc) as f_i:
                dat = pickle.load(f_i)

        for s in all_sessions:
            if s!=all_sessions[lIdx]:
                print s
                save_ROIS(f[s],dat)

    print 'done!'
                



