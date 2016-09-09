from __future__ import division
import os
import scipy.io as spio
import tifffile
import numpy as np
import pickle
import h5py
import re
import time

def load_GRABinfo(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def get_triggers(matFilePth):
    if 'Tones2to64thirdOct' in matFilePth:
        matfile = spio.loadmat(matFilePth)
        stimattrs = {'stim_list': matfile['outDat'][0][0][2][:,0],
                     'stim_dur': matfile['outDat'][0][0][2][:,1][0],
                     'stim_levels': matfile['outDat'][0][0][2][:,5],
                     'stimScriptName': matfile['outDat'][0][0][-1][0],
                     'timestamp': matfile['outDat'][0][0][0][0],
                     'stimOrder': matfile['outDat'][0][0][1].T[0]}
    elif re.search(r'.*search_tones_outDat.*',matFilePth):
        stim_dict = load_GRABinfo(spio.loadmat(matFilePth,struct_as_record=False, squeeze_me=True)['outDat'])
        stimattrs = {'stim_list': np.fliplr(np.vstack([stim_dict['stimMat'][:,5],
                                               stim_dict['stimMat'][:,0]]).T),
                     'stimOrder': stim_dict['trialOrder'],
                     'timestamp': stim_dict['timeStamp'],
                     'stim_dur': stim_dict['stimMat'][1,1],
                     'stim_levels': stim_dict['stimMat'][:,5],
                     'stimScriptName': 'search_tones',
                     'stim_spacing': stim_dict['sweepLengthFrames']
                    }



        return stimattrs


def get_DM(stim_dict,framePeriod,nFrames):
    nFramesStim = int(np.floor(float(stim_dict['stim_dur'])/framePeriod))
    nStims = int(len(stim_dict['stim_list']))
    DM = np.zeros([nStims,nFrames])
    for i in range(1,1+nStims):
        stim_i_frames = np.where(stim_dict['stimOrder']==i)[0]
        for stim_presen in stim_i_frames:
            DM[i-1,stim_presen*stim_dict['stim_spacing']:stim_presen*stim_dict['stim_spacing']+nFramesStim] = 1
    return DM
     

def load_tiff_series(directory):
    file_sizes = [os.stat(directory + f).st_size for f in os.listdir(directory) if  '.tif' in f]
    allSame = file_sizes.count(file_sizes[0]) == len(file_sizes)
    if not allSame:
        print 'WARNING! FILES DIFFER IN SIZE'
        
        
    fnames = os.listdir(directory)
    GRABinfo = None
    stimattrs = None
    for idx, fname in enumerate(fnames):
        #print fname
        if '.tif' in fname:
            if idx==0:
                pth = directory + fname
                b = tifffile.TiffFile(pth)
                imageArr = b.series[0].asarray()
            else:
                pth = directory + fname
                b = tifffile.TiffFile(pth)
                imageArr = np.concatenate([imageArr,b.series[0].asarray()])
        elif 'GRABinfo' in fname:
            matFilePth = directory + fname
            matfile = spio.loadmat(matFilePth,struct_as_record=False, squeeze_me=True)['GRABinfo']
            GRABinfo = load_GRABinfo(matfile)

        elif 'outDat' in fname:
            if '._' not in fname:
                matFilePth = directory + fname
                print matFilePth
                stimattrs = get_triggers(matFilePth) 


        else:
            pass
        
        #print pth,
    return imageArr, allSame, GRABinfo, stimattrs





def add_raw_series(baseDir,file_Dirs,HDF_File,session_ID):
    i = 0
    HDF_PATH = HDF_File.filename
    for fDir in file_Dirs:
        Dir = baseDir + fDir + '/'  #+ '\\'
        st = time.time()
        File, allSame, GRABinfo, stimattrs = load_tiff_series(Dir)
        print 'Load Data %s Time: %s' %(i, time.time() - st)
        
        st = time.time()
        if i >0:
            HDF_File = h5py.File(HDF_PATH,'a',libver='latest')
        
        st = time.time()
        areaDSet = HDF_File[session_ID]['raw_data'].create_dataset(name=fDir,
                                                                   data=File.astype('uint16'),
                                                                   chunks=(10,512,512),
                                                                   dtype='uint16')
        print 'Write %s Time: %s' %(i, time.time() - st)
        
        if GRABinfo!=None:
            HDF_dir = re.findall(r'(.*\/).*\.h5',HDF_File.filename)


            gInfoLoc = str(HDF_dir[0])+str(fDir)+'_GRABinfo.p'

            with open(gInfoLoc,'wb') as grabI_f:
                pickle.dump(GRABinfo,grabI_f)

            areaDSet.attrs['GRABinfo'] = gInfoLoc
            #for key,value in GRABinfo.iteritems():
            #    areaDSet.attrs[key] = str(value)
            #    areaDSet.attrs['dropped_frames'] = not allSame


            framePeriod = float(GRABinfo['scanFramePeriod'])*1000
            nFrames = float(GRABinfo['acqNumFrames'])

            if stimattrs!=None:
                DM = get_DM(stimattrs,framePeriod,nFrames)  #get the designMatrix
                areaDSet.attrs['trigger_DM'] = DM
                areaDSet.attrs['trigger_DM'] = DM
                areaDSet.attrs['stim_list'] = stimattrs['stim_list']
                areaDSet.attrs['stimScriptName'] = str(stimattrs['stimScriptName'])
                areaDSet.attrs['stim_levels'] = stimattrs['stim_levels']
        else:
            print '!!!WARINING!!! NO GRABINFO WAS FOUND, FILE INCOMPLETE'
        st = time.time()
        HDF_File.close()

        i += 1
        HDF_File = h5py.File(HDF_PATH,'a',libver='latest')
