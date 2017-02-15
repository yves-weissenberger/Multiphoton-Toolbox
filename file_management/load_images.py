from __future__ import division
import os, sys, re, time, pickle
import scipy.io as spio
import tifffile
import numpy as np
import h5py



from twoptb.util import load_GRABinfo, progress_bar
from twoptb.file_management.load_events import get_triggers, import_imaging_behaviour, get_DM

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




     

def load_tiff_series(directory):
    file_sizes = [os.stat(os.path.join(directory,f)).st_size for f in os.listdir(directory) if  '.tif' in f]
    allSame = file_sizes.count(file_sizes[0]) == len(file_sizes)
    if not allSame:
        print 'WARNING! FILES DIFFER IN SIZE'
        
    
    fnames = os.listdir(directory); 
    nFiles = len(fnames)#variables for progress bar
    GRABinfo = None
    stimattrs = None
    sys.stdout.write('\r')

    for idx, fname in enumerate(fnames):



        #progress_bar(idx,nFiles)
        sys.stdout.write('\r')
        pStr = r"[%-" + str(nFiles) + r"s]  %d%%" 
        sys.stdout.write('\r' + pStr % ('.'*int(idx), np.round(100*np.divide(idx+1.,float(nFiles)))))
        sys.stdout.flush()
        #print "loading %s \n" %fname
        if '.tif' in fname:
            if idx==0:
                pth = os.path.join(directory,fname)
                b = tifffile.TiffFile(pth)
                imageArr = b.series[0].asarray()
            else:
                pth = os.path.join(directory,fname)
                b = tifffile.TiffFile(pth)
                imageArr = np.concatenate([imageArr,b.series[0].asarray()])
        elif 'GRABinfo' in fname:
            matFilePth = os.path.join(directory,fname)
            matfile = spio.loadmat(matFilePth,struct_as_record=False, squeeze_me=True)['GRABinfo']
            GRABinfo = load_GRABinfo(matfile)

        elif 'outDat' in fname:
            if '._' not in fname:
                matFilePth = os.path.join(directory,fname)
                print matFilePth
                stimattrs = get_triggers(matFilePth) 

        elif ('2AFC' in fname or '_data.txt' in fname):
            matFilePth = os.path.join(directory,fname)
            stimattrs = get_triggers(matFilePth)
        else:
            pass
        
        #print pth,
    return imageArr, allSame, GRABinfo, stimattrs





def add_raw_series(baseDir,file_Dirs,HDF_File,session_ID,get_DM=False):
    i = 0
    hdfPath = HDF_File.filename
    hdfDir = os.path.split(hdfPath)[0]

    gInfoDir = os.path.join(hdfDir,'GRABinfos')
    stimattrsDir = os.path.join(hdfDir,'stims')
    if not os.path.exists(gInfoDir):
        os.mkdir(gInfoDir)

    if not os.path.exists(stimattrsDir):
        os.mkdir(stimattrsDir)


    for fDir in file_Dirs:
        Dir = os.path.join(baseDir,fDir)
        st = time.time()
        print '\n loading %s \n' %os.path.split(fDir)[-1]
        File, allSame, GRABinfo, stimattrs = load_tiff_series(Dir)
        print 'Load Data time: %s' %(time.time() - st)
        
        st = time.time()
        if i >0:
            HDF_File = h5py.File(hdfPath,'a',libver='latest')
        
        st = time.time()
        areaDSet = HDF_File[session_ID]['raw_data'].create_dataset(name=fDir,
                                                                   data=File.astype('uint16'),
                                                                   chunks=(10,512,512),
                                                                   dtype='uint16')
        print 'Write %s Time: %s' %(i, time.time() - st)
        
        if GRABinfo!=None:
            #HDF_dir = re.findall(r'(.*\/).*\.h5',HDF_File.filename)


            gInfoLoc = os.path.join(gInfoDir,str(fDir)+ '_GRABinfo.p' )
            #gInfoLoc = os.path.join(str(HDF_dir[0])+str(fDir)+'_GRABinfo.p'

            with open(gInfoLoc,'wb') as grabI_f:
                pickle.dump(GRABinfo,grabI_f)

            areaDSet.attrs['GRABinfo'] = gInfoLoc


            framePeriod = float(GRABinfo['scanFramePeriod'])*1000
            nFrames = float(GRABinfo['acqNumFrames'])

            if stimattrs!=None:
                stimattrsLoc = os.path.join(stimattrsDir,str(fDir)+ '_GRABinfo.p' )
                if get_DM:
                    DM = get_DM(stimattrs,framePeriod,nFrames)  #get the designMatrix
                    stimattrs['DM'] = DM
                with open(stimattrsLoc,'wb') as saL:
                    pickle.dump(stimattrs,saL)

                areaDSet.attrs['stimattrs'] = stimattrsLoc
                #areaDSet.attrs['trigger_DM'] = DM
                #areaDSet.attrs['stim_list'] = stimattrs['stim_list']
                #areaDSet.attrs['stimScriptName'] = str(stimattrs['stimScriptName'])
                #areaDSet.attrs['stim_levels'] = stimattrs['stim_levels']
        else:
            print '!!!WARINING!!! NO GRABINFO WAS FOUND, FILE INCOMPLETE'
        st = time.time()
        HDF_File.close()

        i += 1
        HDF_File = h5py.File(hdfPath,'a',libver='latest')
