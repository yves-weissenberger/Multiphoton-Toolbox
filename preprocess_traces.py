#!/home/yves/anaconda2/bin/python

from __future__ import division
import h5py
import numpy as np
import c2s
import sys, os, re, pickle
sys.path.append('/home/yves/Documents/')

from twoptb.process_data import neuropil_correct, runkalman
from twoptb.util import _select_area, progress_bar





if __name__=="__main__":

    if len(sys.argv)==1:
        raise ValueError('first argument needs to be absolute or relative path to HDF file')  #wrong error type but cba
    else:
        hdfPath = sys.argv[1]


    with h5py.File(hdfPath, 'r', libver='latest') as hdf:

        areaF,session,area = _select_area(hdf)
        gInfo = pickle.load(open(hdf[session]['raw_data'][area].attrs['GRABinfo']))
        roiInfoLoc = areaF.attrs[u'ROI_dataLoc']
        roiInfo = pickle.load(open(roiInfoLoc))

        n_neurons = len(roiInfo['masks'])
        raw_traces = []; corr_traces = []; df_F = []
        #sys.stdout.write('\r')
        
        print 'Neuropil correcting and calculating df/F'
        sys.stdout.flush()
        for neuron_idx in range(n_neurons):
            progress_bar(neuron_idx+1,n_neurons)

            trace, corrected_trace, _ = neuropil_correct(areaF,roiInfo,neuron_idx)

            ffilt = runkalman(corrected_trace,meanwindow=100,RQratio=2000)
            df_F.append((corrected_trace - ffilt)/ffilt)
            raw_traces.append(trace)
            corr_traces.append(corrected_trace)

        print "running spike extraction... \n"

        #gInfo = pickle.load(open(hdf['raw_data'][session][area].attrs['GRABinfo']))
        frameRate = gInfo['scanFrameRate']
        #print np.array([corr_traces[0]]).shape
        for i in range(n_neurons):
            sys.stdout.write("\r running inference on cell: "+str(i)+"/"+str(n_neurons))
            sys.stdout.flush()
            data = [{'calcium':np.array([corr_traces[i]]),'fps': frameRate}]
        #data = [{'calcium':np.array([i]),'fps': frameRate} for i in corr_traces]
            inf = c2s.predict(c2s.preprocess(data),verbosity=0)

        print 'Saving Data'
        roiInfo['traces'] = np.array(raw_traces)
        roiInfo['corr_traces'] = np.array(corr_traces)
        roiInfo['df_F'] = np.array(df_F)
        roiInfo['spikerates'] = np.array([i['predictions'] for i in inf])
        with open(roiInfoLoc,'wb') as f:
            pickle.dump(roiInfo,f)
        print 'done!'









