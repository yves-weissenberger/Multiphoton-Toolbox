from __future__ import division
import numpy as np
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from multiprocessing.dummy import Pool
import sys, os, time, re
import h5py
from twoptb.util import progress_bar




def register_dayData(HDF_File,session_ID,inRAM=True,poolSize=4,abs_loc='foo',common_ref=False):
    print abs_loc
    hdfPath = HDF_File.filename
    hdfDir = os.path.split(hdfPath)[0]
    global inRAM_flag
    inRAM_flag=inRAM

    if 'registered_data' not in HDF_File[session_ID].keys():
        HDF_File[session_ID].create_group('registered_data')
        HDF_File.close()
        HDF_File = h5py.File(hdfPath,'a',libver='latest')
        print 'Creating Registered Data Group'

    regData_dir = os.path.join(hdfDir,'regInfo')
    if not os.path.exists(regData_dir):
        os.mkdir(regData_dir)

    for f_idx, file_key in enumerate(HDF_File[session_ID]['raw_data'].keys()):

        if f_idx >0:
            HDF_File = h5py.File(hdfPath,'a',libver='latest')





        if file_key in  HDF_File[session_ID]['registered_data'].keys():
            del HDF_File[session_ID]['registered_data'][file_key]
            print 'deleting old version'

        st = time.time()
        try:
            #raw_file = np.array(HDF_File[session_ID]['raw_data'][file_key])
            raw_file = HDF_File[session_ID]['raw_data'][file_key]

            if common_ref and f_idx==0:
                print 'creating global reference'
                refIm_glob = np.mean(raw_file[:50],axis=0)[:,128:-128]
            elif not common_ref:
                refIm_glob = None

            #print 'file open %ss' %(time.time() - st)
            st = time.time()
            print '\n registering %s \n' %file_key
            if not inRAM:
                regFile = HDF_File[session_ID]['registered_data'].create_dataset(name=file_key,
                                                                                 shape=raw_file.shape,
                                                                                 chunks=(10,512,512),
                                                                                 dtype='uint16')
                st = time.time()
                shifts, tot_shifts = motion_register(raw_file,
                                                     regFile,
                                                     maxIter=1,
                                                     Crop=True,
                                                     inRAM=inRAM,
                                                     common_ref=common_ref,
                                                     refIm_glob=refIm_glob)
                regFile.attrs['mean_image'] = np.mean(regFile,axis=0)

            else:
            #________________________________________________________________
                regFile = None
                regIms, shifts, tot_shifts = motion_register(raw_file,
                                                             regFile,
                                                             maxIter=2,
                                                             Crop=True,
                                                             inRAM=inRAM,
                                                             poolSize=16,
                                                             common_ref=common_ref,
                                                             refIm_glob=None)

                print 'Motion Register Duration %ss' %(time.time() - st)
                #________________________________________________________________
                st = time.time()
                regFile = HDF_File[session_ID]['registered_data'].create_dataset(name=file_key,
                              	                                                 data=np.round(regIms).astype('uint16'),
                     	                                                         chunks=(10,512,512),
                                                                                 dtype='uint16')

                regFile.attrs['mean_image'] = np.mean(regIms.astype('uint16'),axis=0)


            

            regFile.attrs['regInfo'] = regData_dir

            regFile.attrs['GRABinfo'] = raw_file.attrs['GRABinfo']
            if 'stimattrs' in raw_file.attrs.keys():
                regFile.attrs['stimattrs'] = raw_file.attrs['stimattrs']
            if common_ref:
                regFile.attrs['common_ref'] = True
            else:
                regFile.attrs['common_ref'] = False


            build_registration_log(regFile,abs_loc=regData_dir,tot_shifts=tot_shifts)

            HDF_File.close()

            print 'file write in ram %ss' %(time.time() - st)
            HDF_File = h5py.File(hdfPath,'a',libver='latest')


        except IOError:
            print '!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!! \n %s could not be loaded. Skipping \n !!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!' %file_key
        HDF_File.close()

            #________________________________________________________________

	st = time.time()
	print 'Write to Disk time %ss:' %(time.time() - st)
    HDF_File = h5py.File(hdfPath,'a',libver='latest')

    return HDF_File

def build_registration_log(areaFile,abs_loc,tot_shifts):
    #file_loc = re.findall(r'(.*/).*\.h5',areaFile.file.filename)[0]
    
    fName = areaFile.name.replace('/','_')
    logF = os.path.join(abs_loc, str(fName) + str('_shifts.txt'))
    roi_pos_str = [str(i)+','+str(j)+'\n' for i,j in tot_shifts]
    with open(logF,'a') as logFile:
        for i in roi_pos_str:
            logFile.write(i)

    return None


def motion_register(imArray,regFile,maxIter=1,Crop=True,inRAM=True,poolSize=4,common_ref=False,refIm_glob=None):
    global inRAM_flag; inRAM_flag = inRAM
    if inRAM:
        pool = Pool(poolSize)

    global crop; crop = Crop

    global refIm
    if common_ref:
        print 'using common refence image \n'
        refIm = refIm_glob
    else:
        if crop==True:
        	refIm = np.mean(imArray[:50],axis=0)[:,128:-128]
        else:
            refIm = np.mean(imArray[:50],axis=0)

    converged = False
    iteration = 0
    if inRAM:
        imgList= [imArray[i] for i in range(imArray.shape[0])]
    #elif not inRAM:
    #    imgList= [(i,imArray[i]) for i in range(imArray.shape[0])]

    tot_shift = np.zeros([imArray.shape[0],2])
    while not converged:
        strt = time.time()

        if not inRAM:
            temp = []
            if iteration==0:
                nFrames = imArray.shape[0]; pFac = nFrames/50  #variables for progress bar
                for i,img in enumerate(imArray):

                    temp.append(register_image((i,img,regFile)))

                    ###### This is just a little progress bar
                    #sys.stdout.write('\r')
                    pStr = r"[%-50s]  %d%%" 
                    sys.stdout.write('\r' + pStr % ('.'*int(np.round(np.divide(i,pFac))), int(100*i/nFrames)))
                    sys.stdout.flush()

                    ###### This is just a little progress bar

                shifts = np.array(temp)
            elif iteration>0:
                #del imgList
                #imgList= [(i,regFile[i]) for i in range(imArray.shape[0])]

                for i,img in enumerate(regFile):
                    temp.append(register_image((i,img)))
                    if np.remainder(i,1000)==0:
                        print '.',
                shifts = np.array(temp)



        else:
            temp = pool.map(register_image,imgList)
            imgList = np.array([i[1] for i in temp])
            shifts = np.array([i[0] for i in temp])

        tot_shift += shifts
        iteration += 1
        mean_shift = np.mean(np.abs(shifts))
        print 'iteration: %s, mean shift: %s, duration: %s' %(iteration, mean_shift, time.time()-strt)
        
        if mean_shift<1:
            converged=True
        
        if iteration>=maxIter:
            print 'Convergence not-optimal'
            converged = True

        if inRAM:
            refIm = np.mean(imgList[:50],axis=0)
        else:
            refIm = np.mean(regFile[:50],axis=0)


        if crop==True:
            refIm = refIm[:,128:-128]

    if not inRAM:
        return shifts, tot_shift
    else:
        return np.array(imgList), shifts, tot_shift


def register_image(inp):
    if not inRAM_flag:
        image_idx = inp[0]
        image = inp[1]
        regFile = inp[2]
    else:
        image = inp


    if crop==True:
        shift, _, _ = register_translation(refIm,image[:,128:-128],upsample_factor=10)
    else:
        shift, _, _ = register_translation(refIm, image, upsample_factor=10)

    if np.sum(np.abs(shift))!=0:
        regIm =  np.fft.ifftn(fourier_shift(np.fft.fftn(image), shift)).real
    else:
        regIm = image

    #added for reduce memory test
    if not inRAM_flag:    
        regFile[image_idx] = regIm
        return shift
    else:
        return [shift,regIm]
