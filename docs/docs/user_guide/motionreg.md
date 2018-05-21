# Motion registration

To run motion registration on a dataset that has been converted to hdf5, simply run

	motion_register_data.py 1 -show_ref 0 path/to/hdf_file.h5 

The first argument here is the common argument which specifies whether all data should be registered to the same mean-image. If The data are of the same cells, this should be set to 1 otherwise set to 0

The second argument just optionally (if set to 1) shows the selected mean image prior to running the registration.

Running this code will create an datasets in the hdf5 file containing the motion registered data. Additionally, it will write the required shifts down in into a text file in the regInfo folder in the directory containing the hdf5 file.

## Motion Registration Algorithm
Motion registration is implemented using the <a href="https://www.osapublishing.org/ol/abstract.cfm?uri=ol-33-2-156"> efficient subpixel registration algorithm</a> implemented in scikit learn.

## Mean image selection 
Mean images are computed by generating 1000 random mean images (from 500 randomly selected frames each) and selecting the one with the largest absolute gradient across all pixels in the image (using this as a proxy for sharpness).

