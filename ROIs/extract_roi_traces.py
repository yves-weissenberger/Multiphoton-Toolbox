import h5py
import sys
import pickle
import copy as cp
import os
import numpy as np

def extract_traces(areaFile,roiattrs):
		
	nROIs = len(roiattrs['idxs'])
	len_trace = areaFile.shape[0]

	roiattrs['traces'] = np.zeros([nROIs,len_trace])
	for idx in range(nROIs):
		sys.stdout.write('\r Extracting_Trace_from roi: %s' %idx)
		sys.stdout.flush()
		mpossx= roiattrs['idxs'][idx][0]
		mpossy = roiattrs['idxs'][idx][1]
		xLims = [np.min(mpossx)-10,np.max(mpossx)+10]
		yLims = [np.min(mpossy)-10,np.max(mpossy)+10]

		temp = areaFile[:,yLims[0]:yLims[1],xLims[0]:xLims[1]] *roiattrs['masks'][idx]
		temp = temp.astype('float64')
		temp[temp==0] = np.nan
		                                                    
		roiattrs['traces'][idx] = np.nanmean(temp,  axis=(1,2))
	return roiattrs


       





if __name__=='__main__':
	
	hdf_path = os.path.abspath(sys.argv[1])
	
	#roi_pth = os.path.join(hdf_path[:-3],'ROIs')
	with h5py.File(hdf_path,'a',libver='latest') as hdf:
		keys = hdf.keys()
		Folder = os.path.split(os.path.abspath(hdf.filename))[0]
		roi_pth = os.path.join(Folder,'ROIs')
		print '\n\n'
		f = hdf[keys[0]]['registered_data']


		sessions = list((i for i in f.iterkeys()))
		for idx,fn in enumerate(sessions):

			print '\narea %s: %s' %(idx,fn)

			areaFile = f[fn]
			FLOC = areaFile.attrs['ROI_dataLoc']

			roiattrs = pickle.load(open(FLOC,'r'))

			roiattrs2 = extract_traces(areaFile,roiattrs)

			with open(FLOC,'wb') as fi:
				pickle.dump(roiattrs2,fi)



	print 'done!'
				

