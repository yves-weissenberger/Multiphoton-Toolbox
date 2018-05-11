# Data conversion

Prior to any preprocessing or analysis of the raw data, it is converted from the raw data format, (i.e. .tif) to HDF5 for convenience. The resulting HDF file serves as a central access point for the multimodal (i.e. two-photon imaging data, video-data, stimulus files etc.).  

Data is converted to HDF5 by running, in terminal 

	python /path/to/twoptb/twoptb/scripts/convert_to_hdf5.py /path/to/data_folder/

In order to preprocess data properly, the script requires a certain directory and file structure.  Firstly, it assumes that imaging data are stored in the form of .tif files with the metadata required for reading them. Additionally, it assumes that these .tif files are stored in directories containing GRABinfo files (which contain image acquisition parameters) and optionally stimulus script files (outDat.mat file in the example, see below for further details). Finally, it assumes that data are in a certain directory structure: 
	
    Expects certain folder structure e.g.
    /home
	/AIAK_2
	    /29082017
			/session01
			    /Acq1.tif
			    GRABinfo.mat
			    outDat.mat
			    ...

			/session02
			    /Acq1.tif
			    GRABinfo.mat
			    outDat.mat
			    ...

			/session03
			    /Acq1.tif
			    GRABinfo.mat
			    outDat.mat
			    ...


In the case of the above directory structure, the script above would be run as

	python /path/to/twoptb/twoptb/scripts/convert_to_hdf5.py /home/AIAK_2/

and would create a single HDF5 file with data from all acquired sessions. Importantly, to optimally use this toolbox, all sessions in the example above should either be acquisitions of the same group of cells, or should all be acquisitions of different groups of cells. 
Running this command would create a single HDF5 file from which raw-data, acquisition parameters and stimulus data could be convienetly and reproducibly accessed. 

## Linking stimulus scripts

A core feature of the toolbox is the centrality of the hdf5 file as a link for the multimodal data. For the analysis of most imaging datasets, additional information is required. This can be of the form of behavioural timestamps, simultaneously recorded video (currently not supported), or stimulus timestamps. Integrating this timestamped data with the HDF5 file requires processing of the raw form of the timestamp data. In order to link custom scripts to the HDF5 file, a readout script must be added to the "load_events.py" file which can be found in 

	twoptb/twoptb/file_management/load_events.py

After adding the appropriate function, the data must be directed to this function by adding the appropriate path to

	twoptb/twoptb/file_management/load_images.py

For example, to extract data from the ouDat.mat file in the example directory structure above "load_images.py" contains the following elif statement 

        elif 'outDat' in fname:
            if '._' not in fname:
                matFilePth = os.path.join(directory,fname)
                print matFilePth
                stimattrs = get_triggers(matFilePth) 

which directs processing of the outDat file to the function "get_triggers" which can be found in load_events.py.


## Output Folder Structure

Running the above script should give rise to the above directory structure
    /home
	/AIAK_2
	    /29082017
			/...
		/processed
			/29082017_AIAK_2
				29082017_AIAK_2.h5
				/GRABinfos
				/stims
				/regInfo
				/ROIs

		proc_log.txt

After this initial preprocessing step next is [motion registration](motionreg.md)