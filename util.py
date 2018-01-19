
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
    dictA = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dictA[strg] = _todict(elem)
        else:
            dictA[strg] = elem
    return dictA


def copy_ROIs(hdf):

    return None

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            d[strg] = _todict(elem)
        elif isinstance(elem, np.ndarray):
            d[strg] = _tolist(elem)
        else:
            d[strg] = elem
    return d

def _tolist(ndarray):
    '''
    A recursive function which constructs lists from cellarrays
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem, np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list
data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
return _check_keys(data)

def _select_area(hdf):

    print hdf.filename
    print 'Sessions:'

    sessions = list((i for i in hdf.iterkeys()))
    for idx,f in enumerate(sessions):
        print idx, f 
    session = int(raw_input('Select Session Nr: '))

    sessions = list((i for i in hdf.iterkeys()))

    dataType = 0


    if 'registered_data' in hdf[sessions[session]].iterkeys():
        print 'Using registered Data'
        dataType = 'registered_data'
    else:
        print '\n!!!!!!!!!!WARNING!!!!!!!!!!!!\nUsing Raw Data\n!!!!!!!!!!WARNING!!!!!!!!!!!!'
        dataType = 'raw_data'



    areas = list((i for i in hdf[sessions[int(session)]][dataType].iterkeys()))
    for idx,f in enumerate(areas):
        print idx, f 


    areaID = int(raw_input('Select Area Nr: '))
    areaFile = hdf[sessions[session]][dataType][areas[areaID]]; 

    return areaFile, sessions[session], areas[areaID]


def progress_bar(idx,Tot,maxN=20):
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
    #sys.stdout.write('\r')
    pStr = "[%-" + str(Tot) + "s]  %d%%" 
    #sys.stdout.write("\r" + pStr % ('.'*int(n_points), np.round(100*np.divide(idx+1.,float(Tot)))))
    sys.stdout.write("\r"+str(idx)+"/"+str(Tot))
    sys.stdout.flush()
    return None