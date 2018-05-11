# ROI Definition Methods

## Automatic approach



### Training an ROI classifer


### Using a pre-trained ROI classifier

We have developed a simple algorithm for automatic roi definition. To run this algorithm, run:

    python /path/to/twoptb/twoptb/ROIs/run_roi_finder.py -sess -1 -ded 3 2 2 -thresh 0.96 /path/to/hdf5.h5 /path/to/twoptb/twoptb/classifiers/zoom1_GTMK.p


![Screenshot](ims/auto_roi.png)


## Manual curation


### Image of the GUI used for ROI and data curation

![Screenshot](ims/ROI_Drawer.png)
