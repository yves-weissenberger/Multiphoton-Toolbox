# Data conversion

The code expects a certain directory structure when it comes to the storage of data. Firstly, it assumes that data are stored in the form of .tif files with the metadata required for reading them.  
	
    Expects certain folder structure e.g.
    /home
	/AIAK_2
	    /29082017
			/Area01
			    /Acq1.tif
			    GRABinfo.mat
			    outDat.mat
			    ...

			/Area02
			    /Acq1.tif
			    GRABinfo.mat
			    outDat.mat
			    ...

			/Area03
			    /Acq1.tif
			    GRABinfo.mat
			    outDat.mat
			    ...


## Linking stimulus scripts

As stated in the introduction a core feature of the toolbox is the centrality of the hdf5 file as a link for the multimodal data com
