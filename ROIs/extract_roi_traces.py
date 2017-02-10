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


       
def neuropil_correct(areaF,roi_attrs,idx):
    


	nROIs = len(roiattrs['idxs'])
	len_trace = areaFile.shape[0]


	roiattrs['traces'] = np.zeros([nROIs,len_trace])
	for idx in range(nROIs):

		mpossx= roi_attrs['idxs'][idx][0]
		mpossy = roi_attrs['idxs'][idx][1]
		xLims = [np.min(mpossx)-10,np.max(mpossx)+10]
		yLims = [np.min(mpossy)-10,np.max(mpossy)+10]
		temp = areaF[:,yLims[0]:yLims[1],xLims[0]:xLims[1]] *np.abs(roi_attrs['masks'][idx]-1)
		temp = temp.astype('float64')
		temp[temp==0] = np.nan
		neuropil_trace = np.nanmean(temp,axis=(1,2))



		temp = areaF[:,yLims[0]:yLims[1],xLims[0]:xLims[1]] *roi_attrs['masks'][idx]
		temp = temp.astype('float64')
		temp[temp==0] = np.nan
		trace = np.nanmean(temp,axis=(1,2))
		corrected_trace = trace - .7*neuropil_trace
    return trace, corrected_trace, neuropil_trace





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
			if 'ROI_dataLoc' in areaFile.attrs.keys():

				FLOC = areaFile.attrs['ROI_dataLoc']
			else:
				Folder = os.path.split(os.path.abspath(areaFile.file.filename))[0]
				fName = areaFile.name[1:].replace('/','-') + '_ROIs.p'
				FLOC = os.path.join(Folder,'ROIs',fName)
				areaFile.attrs['ROI_dataLoc'] = FLOC

			roiattrs = pickle.load(open(FLOC,'r'))

			roiattrs2 = extract_traces(areaFile,roiattrs)

			with open(FLOC,'wb') as fi:
				pickle.dump(roiattrs2,fi)



	print 'done!'
				

