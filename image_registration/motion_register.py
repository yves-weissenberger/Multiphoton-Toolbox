import numpy as np
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from multiprocessing.dummy import Pool
import time
import h5py



def copy_ROIs(HDF_File):

    return None

def register_dayData(HDF_File,session_ID,inRAM=True,poolSize=4):

    HDF_PATH = str(HDF_File.filename)
    global inRAM_flag
    inRAM_flag=inRAM

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
            #raw_file = np.array(HDF_File[session_ID]['raw_data'][file_key])
            raw_file = HDF_File[session_ID]['raw_data'][file_key]
            #print 'file open %ss' %(time.time() - st)
            st = time.time()

            if not inRAM:
                global regFile
                regFile = HDF_File[session_ID]['registered_data'].create_dataset(name=file_key,
                                                                                 shape=raw_file.shape,
                                                                                 chunks=(10,512,512),dtype='int16')
                st = time.time()
                shifts, tot_shifts = motion_register(raw_file,2,Crop=True,inRAM=inRAM)
                regFile.attrs['mean_image'] = np.mean(regFile,axis=0)

            else:
            #________________________________________________________________
                regIms, shifts, tot_shifts = motion_register(raw_file,
                                                             maxIter=2,
                                                             Crop=True,
                                                             inRAM=inRAM,
                                                             poolSize=16)

                print 'Motion Register Duration %ss' %(time.time() - st)
                #________________________________________________________________
                st = time.time()
                regFile = HDF_File[session_ID]['registered_data'].create_dataset(name=file_key,
                              	                                                 data=regIms.astype('int16'),
                     	                                                         chunks=(10,512,512),dtype='int16')
                regFile.attrs['mean_image'] = np.mean(regIms.astype('int16'),axis=0)


            

            regFile.attrs['shifts'] = shifts
            regFile.attrs['tot_shifts'] = tot_shifts
            for key,value in raw_file.attrs.iteritems():
                    if (key=='trigger_DM' or key=='ROI_centres' or key=='ROI_patches'):
                        regFile.attrs[key] = value
                    else:
            	       regFile.attrs[key] = str(value)

            build_registration_log(regFile)

            HDF_File.close()

            print 'file write in ram %ss' %(time.time() - st)
            HDF_File = h5py.File(HDF_PATH,'a',libver='latest')


        except IOError:
            print '!!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!! \n %s could not be loaded. Skipping \n !!!!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!' %file_key
        HDF_File.close()

            #________________________________________________________________

	st = time.time()
	print 'Write to Disk time %ss:' %(time.time() - st)
    HDF_File = h5py.File(HDF_PATH,'a',libver='latest')

    return HDF_File

def build_registration_log(areaFile):
    import re
    file_loc = re.findall(r'(.*/).*\.h5',areaFile.file.filename)[0]
    fName = areaFile.name.replace('/','_')

    logF = str(file_loc) + str(fName) + str('_shifts.txt')
    roi_pos_str = [str(i)+','+str(j)+'\n' for i,j in areaFile.attrs['tot_shifts']]
    with open(logF,'a') as logFile:
        for i in roi_pos_str:
            logFile.write(i)

    return None

def motion_register(imArray,maxIter=5,Crop=True,inRAM=True,poolSize=4):
    
    if inRAM:
        pool = Pool(poolSize)

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
    if inRAM:
        imgList= [imArray[i] for i in range(imArray.shape[0])]
    else:
        imgList= [(i,imArray[i]) for i in range(imArray.shape[0])]

    tot_shift = np.zeros([imArray.shape[0],2])
    print inRAM
    while not converged:
        strt = time.time()

        if not inRAM:
            temp = []
            for entry in imgList:
                temp.append(register_image(entry))
                if np.remainder(entry[0],1000)==0:
                    print '.',
            shifts = np.array(temp)
            print shifts.shape
        else:
            temp = pool.map(register_image,imgList)
            print inRAM_flag
            imgList = np.array([i[1] for i in temp])
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

        if inRAM:
            refIm = np.mean(imgList[:50],axis=0)
        else:
            refIm = np.mean(np.hstack([temp[i] for i in range(50)]),axis=0)

        if crop==True:
            refIm = refIm[128:-128,128:-128]

    if not inRAM:
        return shifts, tot_shift
    else:
        return np.array(imgList), shifts, tot_shift

def register_image(inp):
    if not inRAM_flag:
        image_idx = inp[0]
        image = inp[1]
    else:
        image = inp


    if crop==True:
        shift, _, _ = register_translation(refIm,image[128:-128,128:-128],upsample_factor=10)
    else:
        shift, _, _ = register_translation(refIm, image, upsample_factor=10)

    if np.sum(np.abs(shift))!=0:
        regIm =  np.fft.ifftn(fourier_shift(np.fft.fftn(image), shift)).real.astype('int16')
    else:
        regIm = image.astype('int16')

    #added for reduce memory test
    if not inRAM_flag:    
        regFile[image_idx] = regIm
        return shift
    else:
        return [shift,regIm]