import numpy as np
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from multiprocessing.dummy import Pool
import time
import h5py



def register_dayData(HDF_File,session_ID):

    HDF_PATH = str(HDF_File.filename)


    if 'registered_data' not in HDF_File[session_ID].keys():
        HDF_File[session_ID].create_group('registered_data')
        HDF_File.close()
        HDF_File = h5py.File(HDF_PATH,'a',libver='latest')
        print 'Creating Registered Data Group'


    for f_idx, file_key in enumerate(HDF_File[session_ID]['raw_data'].keys()):

        if f_idx >0:
            HDF_File = h5py.File(HDF_PATH,'a',libver='latest')





        if file_key in  HDF_File[session_ID]['registered_data'].keys():
            del HDF_File[session_ID]['registered_data'][file_key]
            print 'deleting old version'

        st = time.time()
        try:
            raw_file = np.array(HDF_File[session_ID]['raw_data'][file_key])
            #print 'file open %ss' %(time.time() - st)
            #________________________________________________________________
            st = time.time()
            regIms, shifts, tot_shifts = motion_register(raw_file,2)
            print 'Motion Register Duration %ss' %(time.time() - st)
            #________________________________________________________________
            st = time.time()
            regFile = HDF_File[session_ID]['registered_data'].create_dataset(name=file_key,
                          	                                                 data=regIms.astype('int16'),
                 	                                                         chunks=(10,512,512),dtype='int16')

            HDF_File.close()
            HDF_File = h5py.File(HDF_PATH,'a',libver='latest')
            regFile =  HDF_File[session_ID]['registered_data'][file_key]
            raw_file = HDF_File[session_ID]['raw_data'][file_key]
            #print shifts

            #print 'success'
            regFile.attrs['shifts'] = shifts
            regFile.attrs['mean_image'] = np.mean(regIms.astype('int16'),axis=0)
            regFile.attrs['tot_shifts'] = tot_shifts
            for key,value in raw_file.attrs.iteritems():
                    if (key=='trigger_DM' or key=='ROI_centres' or key=='ROI_patches'):
                        regFile.attrs[key] = value
                    else:
            	       regFile.attrs[key] = str(value)
            print 'file write in ram %ss' %(time.time() - st)
        except IOError:
            print '!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!! \n %s could not be loaded. Skipping \n !!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!' %file_key
        HDF_File.close()

            #________________________________________________________________

        
	st = time.time()
	print 'Write to Disk time %ss:' %(time.time() - st)
    HDF_File = h5py.File(HDF_PATH,'a',libver='latest')

    return HDF_File


def motion_register(imArray,maxIter=5,Crop=True):
    
    pool = Pool(5)
    global crop
    crop = Crop
    imArray = imArray
    global refIm
    if crop==True:
    	refIm = np.mean(imArray[:50],axis=0)[128:-128,128:-128]
    else:
        refIm = np.mean(imArray[:50],axis=0)

    converged = False
    iteration = 0
    imgList= [imArray[i] for i in range(imArray.shape[0])]
    tot_shift = np.zeros([imArray.shape[0],2])
    while not converged:
        strt = time.time()
        temp = pool.map(register_image,imgList)
        imgList = [i[1] for i in temp]
        shifts = np.array([i[0] for i in temp])
        tot_shift += shifts
        iteration += 1
        mean_shift = np.mean(np.abs(shifts))
        print 'iteration: %s, mean shift: %s, duration: %s' %(iteration, mean_shift, time.time()-strt)
        
        if mean_shift<1:
            converged=True
        
        if iteration>maxIter:
            print 'Convergence not-optimal'
            converged = True

        if crop==True:
            refIm = np.mean(imgList[:50],axis=0)[128:-128,128:-128]
        else:
            refIm = np.mean(imgList[:50],axis=0)

    return np.array(imgList), shifts, tot_shift

def register_image(image):

    if crop==True:
        shift, _, _ = register_translation(refIm,image[128:-128,128:-128],upsample_factor=10)
    else:
        shift, _, _ = register_translation(refIm, image, upsample_factor=10)

    if np.sum(np.abs(shift))!=0:
        regIm =  np.fft.ifftn(fourier_shift(np.fft.fftn(image), shift)).real.astype('int16')
    else:
        regIm = image.astype('int16')
    return [shift,regIm]