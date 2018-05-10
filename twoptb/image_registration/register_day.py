import sys
import os

def findpath():
    twoptbDir = os.path.dirname(os.path.realpath(__file__))

    found = False
    while not found:
        cDir,ext = os.path.split() 
        if ext='twoptb':
            found = False
            twoptb_path = cDir
            break
    return twoptb_path


sys.path.append(twoptb_path)

from motion_register import motion_register
import h5py
def register_day(fID,maxIter=5)

    for file_key in fID.keys()
        print ;file_key
        st = time.time()
        File = fID['raw_data'][file_key]
        regIms, shifts, tot_shifts = motion_register(File,maxIter)
        print fID[fID.keys()[0]].keys(), 'hello'
        if 'registered_data' not in fID[fID.keys()[0]].keys():
            fID.create_group('registered_data')
            fName = fID.filename
            fID.close()
            fID = h5py.File(fName,'a',libver='latest')
            print 'Creating registered data dataset'
        else:
            pass

        regFile = fID['registered_data'][file_key].create_dataset(name=file_key, data=regIms.astype('uint16'),
                                      chunks = (10,512,512), dtype='uint16')
        regFile.attrs['shifts'] = shifts
        regFile.attrs['mean_image'] = np.mean(regIms,axis=0)
        regFile.attrs['shifts_list'] = tot_shifts
        for key, value in File.attrs.iteritems():
            regFile.attrs[key] = str(value)
        print file_key + ' registration time: %s' %(time.time() - st)


    return fID
