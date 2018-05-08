#A toolbox (under development) for processing two-photon imaging data in python2.7

<h2> Includes complete pre-processing pipeline image registration GUIs for ROI extraction and mask drawing</h2>


<h3>Dependencies</h3>

<li>numpy</li>
<li>scipy</li>
<li>scikit-image</li>
* tifffile
* pyqt
* pyqtgraph (optional)



# How to run 

## 1. run convert_to_hdf5.py

## 2. run motion_register_data.py

## 3. draw ROIs using either

### 3.1. PyQtGraph_ROI_Drawer.py
    #### This is a GUI that enables manual drawing of ROIs

    Using this ROI drawer, ROIs are drawn on individual sessions that are grouped together in the hdf5 file.
    If multiple sessions share a common reference, the ROIs are typically be shared across acquisition runs
    to do this run 

    twoptb/ROIs/share_roiinfo.py /path/to/my_hdf5.py

    and the ROIs will be copied across acquitions. Importantly, only one session can serve as a seed to copy from,
    all other WILL BE OVERWRITTEN. 

### 3.2. run_roi_finder.py
    #### After having drawn ROIs on 'training' datasets, this
         automatic roi-drawing tool may be used by initially 
         training an ROI drawer using train_roi_finder.py
    

    Using this ROI drawer, ROIs are drawn on individual sessions that are grouped together in the hdf5 file.
    If multiple sessions share a common reference, the ROIs are typically be shared across acquisition runs
    to do this run 

    twoptb/ROIs/share_roiinfo.py /path/to/my_hdf5.py

    and the ROIs will be copied across acquitions. Importantly, only one session can serve as a seed to copy from,
    all other WILL BE OVERWRITTEN. 
