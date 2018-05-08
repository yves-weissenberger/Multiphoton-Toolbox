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

### 3.2. run_roi_finder.py
    #### After having drawn ROIs on 'training' datasets, this
         automatic roi-drawing tool may be used by initially 
         training an ROI drawer using train_roi_finder.py
    

