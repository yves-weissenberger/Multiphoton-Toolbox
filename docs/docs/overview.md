#Overview

This page is intended to provide a quick overview of the workflow. For more information please see the [User Guide](/user_guide/data_conversion.md)

## Typical Set of function calls to preprocess data

Most important functions have (will soon have) help. To view the help run the function simply as:

First format data by calling 
	
	convert_to_hdf5.py path/to/datafolder/

Next motion register data

	motion_register_data.py path/to/hdf5.h5

Then draw ROIs either manually

	ROI_Drawer.py path/to/hdf5.h5

or automatically

	run_roi_finder.py finder_name path/to/hdf5.h5

followed by manul curation using the "ROI_Drawer"

Then, if the data contains several separate acquisition runs involving the same cells

	share_roiinfo.py path/to/hdf5.h5

Finally extract traces from the cells:

	extract_roi_traces.py path/to/hdf5.h5

## General Advice


The key functions defining the pipeline are run by calling (via command line)

	python /path/to/twoptb/twoptb/scripts/generic_script.py -h

