from __future__ import division
import numpy as np
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from multiprocessing.dummy import Pool
import matplotlib.pyplot as plt
import sys, os, time, re
import h5py


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

from twoptb.util import progress_bar




def register_dayData(HDF_File,session_ID,inRAM=True,poolSize=4,abs_loc='foo',common_ref=False,show_ref_mean=1):
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

            frames = raw_file.shape[0]
            chunkSize = np.max(np.array([x for x in range(1, 11) if frames%x == 0]))

            #corrs_ = []
            #print frames
            #nChunks = np.floor(frames/50)
            #raw_file[:50*nChunks].reshape(-50,nChunks)
            #for j in raw_file[:frames]:
            #    corrs_.append(np. corrcoef(i.flatten(),j.flatten())[0,1])

            #ord_ = np.argsort(corrs_)
            

            if common_ref and f_idx==0:
                """print 'creating global reference'
                                                                #refIm_glob = np.mean(raw_file[ord_[-200:]],axis=0)[:,128:-128]
                                                                import Register_Image
                                                                ############# HERE TRY TO FIND A REFERENCE IMAGE ###############
                                                                refss = []
                                                                for ix_,as_ in enumerate(HDF_File[session_ID]['raw_data'].keys()):
                                                                    a_ = np.mean(HDF_File[session_ID]['raw_data'][as_][-500:],axis=0)
                                                                    refss.append(a_)
                                                                    plt.figure(ix_)
                                                                    plt.imshow(a_,interpolation='None',cmap='gray')
                                                                    plt.title(ix_)
                                                                plt.show(block=False)
                                                
                                                                sel_ref = int(raw_input("selected mean: "))"""

                ###########

                refss = []
                im_grads = []
                hdf_keys = HDF_File[session_ID]['raw_data'].keys()
                st = time.time()
                kix_ = hdf_keys[-1]
                temp = np.array(HDF_File[session_ID]['raw_data'][kix_])
                print 'load time:', np.round(time.time() - st,decimals=1)
                for ii in range(200):
                    #kix = np.random.randint(len(hdf_keys))
                    ixs = np.random.permutation(np.arange(HDF_File[session_ID]['raw_data'][kix_].shape[0]))[:500]

                    a_ = np.mean(temp[np.array(sorted(ixs)),:,:],axis=0)
                    refss.append(a_)
                    im_grads.append(np.sum(np.abs(np.gradient(refss[-1]))))
                    #print time.time() - st
                    print '.',

                ord_grads = np.argsort(np.array(im_grads))[-10:]
                #for ux in ord_grads:
                #    plt.figure()
                #    plt.title(ux)
                #    plt.imshow(refss[ux],interpolation='None',cmap='gray')
                #    plt.show(block=False)

                #sel_ref = int(raw_input("selected mean: "))
                refIm_glob = refss[ord_grads[-1]]
                if show_ref_mean:
                    plt.imshow(refIm_glob,cmap='binary_r')
                    plt.show(block=0)
                #np.mean(raw_file[:5000],axis=0)#raw_file[ord_[-100]].astype('float')
                #for i in ord_[-99:]:


                    #image_idx = inp[0]
                    #image = inp[1]
                    #regFile = inp[2]
                    #out=Register_Image.Register_Image(raw_file[i],refIm_glob)
                    #refIm_glob += out[1].astype('float')

                #refIm_glob /= 100.
                #plt.imshow(refIm_glob,cmap='gray')
                #plt.show()
                refIm_glob = refIm_glob[128:-128,128:-128]


            elif not common_ref:
                refIm_glob = None


           
            #print 'file open %ss' %(time.time() - st)
            st = time.time()
            print '\n registering %s \n' %file_key
            if not inRAM:
                regFile = HDF_File[session_ID]['registered_data'].create_dataset(name=file_key,
                                                                                 shape=raw_file.shape,
                                                                                 chunks=(chunkSize,512,512),
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
                                                                                 chunks=(chunkSize,512,512),
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
    with open(logF,'wb') as logFile:
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
            refIm = np.mean(imArray[:5000],axis=0)[128:-128,128:-128]
        else:
            refIm = np.mean(imArray[:5000],axis=0)

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

    if 'inRAM_flag' in globals():
        inRAM = inRAM_flag
    else:
        inRAM = True

    if 'refIm' in globals():
        refIm_ = refIm
    if 'crop' not in globals() and 'crop' not in locals():
        if inp[1].shape[0]>256:
            crop_ = True
            refIm_ = inp[0][128:-128,128:-128]
            upsample_factor = 10

        else:
            crop_ = False
            refIm_ = inp[0]
            upsample_factor = 1
        image = inp[1]

    else: 
        crop_ = crop
        upsample_factor = 10
        if not inRAM:
            image_idx = inp[0]
            image = inp[1]
            regFile = inp[2]
        else:
            image = inp

    #print refIm_.shape,
    if crop_==True:
        shift, _, _ = register_translation(refIm_,image[128:-128,128:-128],upsample_factor=upsample_factor)
    else:
        shift, _, _ = register_translation(refIm_, image, upsample_factor=upsample_factor)


    #set this 
    if np.any(shift>60):
        print "!!!!!!WARNING VERY LARGE SHIFTS!!!!!!"
        shift = np.array([0,0])

    
    if np.sum(np.abs(shift))!=0:
        regIm =  np.fft.ifftn(fourier_shift(np.fft.fftn(image), shift)).real
    else:
        regIm = image

   
    #added for reduce memory test
    if not inRAM:    
        regFile[image_idx] = regIm
        return shift
    else:
        return [shift,regIm]
