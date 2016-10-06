#!/home/yves/anaconda2/bin/python
#
#
#
#
#
#
# ==========================================



def convert_day_data_to_hdf5(base_path):
    exclude_list = ['cent','Cent','proc_log.txt','processed']


    #base_path = os.path.abspath('.')
    animal_ID = os.path.split(base_path)[-1]


    folders = os.listdir(base_path)
    print folders
    #print fs
    logLoc = os.path.join(base_path,'proc_log.txt')

    if 'proc_log.txt' not in folders:
        logF = open(logLoc,'wb')
        logF.write(animal_ID+'\n')

        logF.close()
        print '...starting log'
    #HDF_File,file_path = MP.file_management.create_base_hdf(animal_ID=animal_name,file_loc='/media/yves/Storage 2/')


    with open(logLoc) as logF:
    	
        for fold_nm in folders:
            print folders
            fold_dir = os.path.join(base_path,fold_nm)

            if all([crit not in fold_dir for crit in exclude_list]):

                if 'tonemapping' in fold_dir and fold_dir!=animal_ID+'_tonemapping':
                    fs = os.listdir(fold_dir)
                    fs = [i for i in fs if i !=(animal_ID + '_tonemapping')]
                    fs = [i for i in fs if i!=('proc_log.txt')]
                    print fs
                    session_ID = 'tonemapping'

                else:
                    session_ID = fold_nm
                    pass

                procDir = os.path.join(base_path,'processed')
                if not os.path.exists(procDir):
                    os.mkdir(procDir)

                HDF_File,file_path = MP.file_management.create_base_hdf(animal_ID=animal_ID,file_loc=procDir)
                HDF_File = MP.file_management.add_session_groups(file_handle = HDF_File,
                                                 session_ID=session_ID)

                print 'converting data to HDF5...'
                st = time.time()
                fs = os.listdir(fold_dir)

                MP.file_management.add_raw_series(baseDir=fold_dir,
                                                  file_Dirs=fs,
                                                  HDF_File=HDF_File,
                                                  session_ID=session_ID)

                print time.time() - st
    return None

if __name__ == "__main__":

    import os, sys, argparse, re, time
    import numpy as np
    import h5py
    import matplotlib.pyplot as plt
    sys.path.append('/home/yves/Documents/')
    import twoptb as MP

    if sys.argv[1]=='-help':
        print 'first argument specifies what folders to exclude'
    elif sys.argv[1]==None:
        base_path = os.path.abspath('.')
        print 'No explicit directory to load specified, using currenct directory \n %s' %base_path
    else:
        base_path = os.path.abspath(sys.argv[1])
        print  'converting data from \n %s \n to hdf5' %base_path


    convert_day_data_to_hdf5(base_path)