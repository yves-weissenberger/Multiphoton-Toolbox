
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



def merge_files(sessionFile,mergeName,mergeFiles=None):
    import h5py
    import numpy as np
    
    # Sort out which files to merge
    if mergeFiles==None:
        areaKeys = (sessionFile['raw_data'].keys())
        for i,f in enumerate(sessionFile['raw_data'].keys()):
            print i,f
        mergeFiles = raw_input('Type ordered files to merge with comma separation (e.g. 0,2): ').split(',')
        mergeFiles = [areaKeys[int(i)] for i in mergeFiles if i!='']

    
    # Concatenate design matrices
    containsDM = []
    created_DM = False
    for area in mergeFiles:
        if not created_DM:
            mergeDM = sessionFile['raw_data'][area].attrs['trigger_DM']
            created_DM = True
        else:
            mergeDM = np.hstack([mergeDM,sessionFile['raw_data'][area].attrs['trigger_DM']])
        containsDM.append('trigger_DM' in (sessionFile['raw_data'][area].attrs.iterkeys()))
    
    assert all(containsDM), 'All files should have an attached design matrix if merge to be sensible'
    tot_len = 0
    
    file_handles = []
    file_names = []

    for area in mergeFiles:
        file_handles.append(sessionFile['raw_data'][area])
        tot_len += sessionFile['raw_data'][area].shape[0]
        frameShape = sessionFile['raw_data'][area].shape[1:]
        file_names.append(area)
    #print (tot_len,frameShape[0],frameShape[1])
    
    mergeFile = sessionFile['raw_data'].create_dataset(mergeName,
                                           shape=(tot_len,frameShape[0],frameShape[1]),
                                           dtype='int16',
                                           )
    
    #mergeFile =  sessionFile['raw_data'][mergeName]
    index = 0
    for fH in file_handles:
        print '\nAdding %s' %fH.name
        for page in fH:
            mergeFile[index,:,:] = page
            index += 1
            if np.remainder(index,2000)==0:
                print '.',
    
    print 'Done!'
    jFN = '___'.join(file_names)
    mergeFile.attrs['merged_files'] = jFN
    #copy file attributes across
    for key,value in fH.attrs.iteritems():
        if key=='trigger_DM':
            mergeFile.attrs[key] = mergeDM
        elif (key=='ROI_centres' or key=='ROI_patches'):
            mergeFile.attrs[key] = value
        else:
            mergeFile.attrs[key] = str(value)

    for area in mergeFiles:
        del sessionFile['raw_data'][area]