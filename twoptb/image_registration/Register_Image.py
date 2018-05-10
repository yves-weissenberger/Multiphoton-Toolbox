import numpy as np
from scipy.ndimage import fourier_shift
from skimage.feature import register_translation



def Register_Image(image,refIm,crop=False):

    if crop==True:    
        shift, _, _ = register_translation(refIm,image[128:-128,128:-128],upsample_factor=10)
    else:
        shift, _, _ = register_translation(refIm, image, upsample_factor=10)

    if np.sum(np.abs(shift))!=0:
        regIm =  np.fft.ifftn(fourier_shift(np.fft.fftn(image), shift)).real  
    else:
        regIm = image
    return [shift,regIm]