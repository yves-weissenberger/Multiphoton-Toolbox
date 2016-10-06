
from __future__ import division
import re, os, csv, time, sys
import itertools
import numpy as np
import scipy.io as spio







def get_DM(stim_dict,framePeriod,nFrames):
    nFramesStim = int(np.floor(float(stim_dict['stim_dur'])/framePeriod))
    nStims = int(len(stim_dict['stim_list']))
    DM = np.zeros([nStims,nFrames])
    for i in range(1,1+nStims):
        stim_i_frames = np.where(stim_dict['stimOrder']==i)[0]
        for stim_presen in stim_i_frames:
            DM[i-1,stim_presen*stim_dict['stim_spacing']:stim_presen*stim_dict['stim_spacing']+nFramesStim] = 1
    return DM


def cartesian_product_itertools(tup1, tup2):
    return list(product(tup1, tup2))


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


def copy_ROIs(HDF_File):

    return None



def progress_bar(idx,Tot,maxN=50):
    """ 
    Simple Progress bar that when called in loop will output something looking like this
    [....    ] 52%

    Args
    __________________________

    idx: int | float

        current index

    Tot: int | float

        total number of iterations to be completed

    maxN: int | float

        maxmimum numer of points to have between the square brackets




    """
    if Tot<maxN:
        maxN = Tot

    n_points = int(idx*np.round(Tot/maxN))
    sys.stdout.write('\r')
    pStr = r"[%-" + str(Tot) + r"s]  %d%%" 
    sys.stdout.write('\r' + pStr % ('.'*int(idx), np.round(100*np.divide(idx+1.,float(Tot)))))
    sys.stdout.flush()
    return None