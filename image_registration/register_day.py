
from motion_register import motion_register

def register_day(fID,maxIter=5)

	for file_key in fID.keys()

		st = time.time()
		File = fID['raw_data'][file_key]
		regIms, shifts, tot_shifts = motion_register(File,maxIter)
		regFile = fID['registered_data'][file_key].create_dataset(name=file_key, data=regIms.astype('uint16'),
									  chunks = (10,512,512), dtype='uint16')
		regFile.attrs['shifts'] = shifts
		regFile.attrs['mean_image'] = np.mean(regIms,axis=0)
		regFile.attrs['shifts_list'] = tot_shifts
		for key, value in File.attrs.iteritems():
			regFile.attrs[key] = str(value)
		print file_key + ' registration time: %s %(time.time() - st)


	return fID
