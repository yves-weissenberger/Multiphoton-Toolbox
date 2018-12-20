import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter


def getVmap(Ucell, sig):
    us = gaussian_filter(Ucell, [sig[0], sig[1], 0.],  mode='wrap')
    # compute log variance at each location
    V  = (us**2).mean(axis=-1)
    um = (Ucell**2).mean(axis=-1)
    um = gaussian_filter(um, sig,  mode='wrap')
    V  = V / um
    V  = V.astype('float64')
    return V, us


def get_sdmov(mov):
    ix = 0
    batch_size = 500
    nbins, npix = mov.shape
    sdmov = np.zeros(npix, 'float32')
    while 1:
        inds = ix + np.arange(0,batch_size)
        inds = inds[inds<nbins]
        if inds.size==0:
            break
        sdmov += np.sum(np.diff(mov[inds, :], axis = 0)**2, axis = 0)
        ix = ix + batch_size
    sdmov = (sdmov/nbins)**0.5
    sdmov = np.maximum(1e-10,sdmov)
    #sdmov = np.mean(np.diff(mov, axis = 0)**2, axis = 0)**.5
    return sdmov



def get_correlation_image_suite2p(mov):
    
    nF,nx,ny = mov.shape
    if nF<10000:
        nRd = 1
    elif nF<20000:
        nRd = 2
    elif nF<50000:
        nRd= 5
    elif nF<100000:
        nRd= 10
        
    mov = np.mean(mov.reshape(mov.shape[0]/nRd,nRd,nx,ny),axis=1)
    
    
    
    nbins, Lyc, Lxc = np.shape(mov)

    sig = 1./10. # PICK UP
    for j in range(nbins):
        mov[j,:,:] = ndimage.gaussian_filter(mov[j,:,:], sig)

    mov = np.reshape(mov, (-1,Lyc*Lxc))

    # compute noise variance across frames
    sdmov = get_sdmov(mov)
    mov /= sdmov
    cov = np.dot(mov,mov.transpose()) / float(mov.shape[1])
    cov = cov.astype('float32')
    
    
    u, s, v = np.linalg.svd(cov)

    u = u[:, :5000]
    U = np.dot(u.transpose(),mov)
    U = np.reshape(U, (-1,Lyc,Lxc))
    U = np.transpose(U, (1, 2, 0)).copy()

    V,us = getVmap(U,[.5,.5])
    return V