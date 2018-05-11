#Overview

This page is intended to provide a quick overview of the workflow. For more information please see the [User Guide](/user_guide/data_conversion.md)

## Typical Set of function calls to preprocess data

Most important functions have (will soon have) help. To view the help run the function simply as:

First format data by calling 
	
	python /path/to/twoptb/twoptb/scripts/convert_to_hdf5.py 

Next motion register data

	python /path/to/twoptb/twoptb/scripts/motion_register_data.py 

Then draw ROIs either manually

	python /path/to/twoptb/twoptb/ROIs/ROI_Drawer.py 

or automatically

	/path/to/twoptb/twoptb/ROIs/run_roi_finder.py

followed by manul curation using the "ROI_Drawer"

Then, if the data contains several separate acquisition runs involving the same cells

	/path/to/twoptb/twoptb/ROIs/share_roiinfo.py

Finally extract traces from the cells:

	/path/to/twoptb/twoptb/ROIs/extract_roi_traces.py 


## General Advice

If you know how to, I would strongly recommend adding the twoptb scripts and ROI paths to you python path as it will make calling scripts much quicker.


The main purpose of this toolbox is two-photon imaging data preprocessing. The key functions defining the pipeline are run by calling (via command line)

	python /path/to/twoptb/twoptb/scripts/generic_script.py -h

