from newkalman import runkalman
from twoptb.util import progress_bar
import numpy as np






#def extract_spikes(areaF,roi_attrs,idx)

def _extract_trace(areaF,roi_attrs,idx):

    mpossx= roi_attrs['idxs'][idx][0]
    mpossy = roi_attrs['idxs'][idx][1]
    xLims = [np.min(mpossx)-10,np.max(mpossx)+10]
    yLims = [np.min(mpossy)-10,np.max(mpossy)+10]
    temp = areaF[:,yLims[0]:yLims[1],xLims[0]:xLims[1]] *roi_attrs['masks'][idx]
    temp = temp.astype('float64')
    temp[temp==0] = np.nan
    trace = np.nanmean(temp,axis=(1,2))
    return trace

def neuropil_correct(areaF,roi_attrs,idx):
    
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



def get_dF_F_area(areaFile):
	traces = np.array(areaFile.attrs['roi_traces'])
	dFF = np.zeros(traces.shape)
	for idx, roi_trace in enumerate(traces):
		ffilt = MP.process_data.runkalman(roi_trace,meanwindow=100,RQratio=2000)
		dFF[idx] = (traces - ffilt)/ffilt
	return dFF


def extract_session_ROI_traces(session_file):

	rec_files_raw = session_file['raw_data'].keys()
	rec_files_regged = session_file['registered_data'].keys()
	if all([i in rec_files_raw for i in rec_files_regged]):
		f_type = 'registered_data'
		print 'Working with registered data'
		resp = 'y'
	else:
		f_type = 'raw_data'
		print 'Warning, you are extracting traces from raw_data that has not been motion registered'
		resp = raw_input('Are you sure you want to proceed? (type y or n):')
		if resp not in ['n','y']:
			raise ValueError('Please respond with "n" or "y"')

	if resp=='y':
		print 'Extracting traces from %s' %f_type
		for area_id in session_file[f_type].keys():
			area_file = session_file[f_type][area_id]
			if 'ROI_centres' in (area_file.attrs.iterkeys()):
				print 'processing %s' %area_id
				roi_traces = extract_area_ROI_traces(area_file)
				session_file[f_type][area_id].attrs['roi_traces'] = roi_traces
			else:
				print 'Warning: %s does not have any labelled ROIs' %area_id

	else:
		print 'Extraction aborted'


	return None




def extract_area_ROI_traces(areaFile,width=4):
	from numpy import mean, array, zeros
	nROIs = areaFile.attrs['ROI_centres'].shape[0]
	areaDset = array(areaFile)
	len_trace = areaDset.shape[0]

	ROI_Traces = zeros([nROIs,len_trace])

	for roi_idx in range(nROIs):
		centre0 =areaFile.attrs['ROI_centres'][roi_idx]
		if 'ROI_masks' not in (areaFile.attrs.iterkeys()):
			traces = areaDset[:,int(centre0[1])-width:int(centre0[1])+width,
			                int(centre0[0])-width:int(centre0[0])+width]
		else:
			mask_sh = masks.shape[0]
			traces = areaDset[:,int(centre0[1])-mask_sh:int(centre0[1])+mask_sh,
			                int(centre0[0])-mask_sh:int(centre0[0])+mask_sh]
			traces = traces*masks[roi_idx]

		Tr = mean(mean(traces,axis=1),axis=1)
		ROI_Traces[roi_idx] = Tr

	return ROI_Traces
#def get_ROI_masks_area(area_hdf):






#def extract_spike_probabilities(areaFile):

