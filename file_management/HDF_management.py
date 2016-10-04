def create_base_hdf(animal_ID,file_loc):

	import h5py
	import os
	if not os.path.isdir(os.path.join(file_loc,animal_ID)):
		os.mkdir(os.path.join(file_loc,animal_ID))

	file_path = os.path.join(file_loc,animal_ID,animal_ID + '.h5')
	file_exists = os.path.isfile(file_path)
	if file_exists:
		print 'File already exists if you proceed you will overwrite \n are you sure you would like to proceed'
		answer = raw_input('type yes if you would like to proceed otherwise press enter:  ')

		if answer=='yes':
				answer = raw_input('are you SURE (this will delete all work done on this file....):  ') 
 
	if (not file_exists or answer=='yes'):
		HDF_File = h5py.File(file_path,'w',libver='latest')
		#try:
		#	HDF_File = h5py.File(file_path,'w',libver='latest')
			#HDF_File.attrs['path'] = file_path
		#except IOError:
			#raise IOError('you idiot, the file is already open in your namespace =p')
	else:
		print 'File not overwritten, returning handle to existing file'
		HDF_File = h5py.File(file_path,'a',libver='latest')

	return HDF_File, file_path

#_____________________________________________________________________________________________


def load_hdf5(file_path):
	""" Literally just return the HDF5 file handle, really just in case
		one you forget h5py syntax """
	import h5py
	return h5py.File(file_path,'r',libver='latest')

#_____________________________________________________________________________________________



def add_session_groups(file_handle,session_ID):
	"""This adds the data for an entire imaging
	   day. Arguments:
	   file_handle: Handle to the HDF5 file
	   session_ID: Str containing the name of
	   that session"""
	import h5py
	import os


	#Argument checking
	assert type(session_ID) is str, 'session_ID needs to be a string'
	assert type(file_handle) is h5py._hl.files.File, 'file_handle needs to be a valid HDF5 File handle'

	dayGroup = file_handle.create_group(session_ID)
	dayGroup.create_group('raw_data')
	dayGroup.create_group('registered_data')

	return file_handle

	
