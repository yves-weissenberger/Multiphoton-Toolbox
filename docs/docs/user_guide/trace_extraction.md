## Extracting ROI traces

Having drawn ROIs, traces may be extracted by running:

    extract_roi_traces.py /path/to/hdf5.h5 y y

where the two additional arguments specify whether to neuropil and baseline correct the data which should essentially always be done...
The traces will be extracted into the earlier creates pickle files in the ROI folder