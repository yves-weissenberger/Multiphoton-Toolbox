#!/home/yves/anaconda2/bin/python

import os, sys, argparse, re, time
import numpy as np
import h5py
import matplotlib.pyplot as plt

import sys
import os

"""

Motion register data that has been converted to HDF5 format

To run this script, point to the hdf5 file that was created by running convery_to_hdf5.py

requires additional arguments:

The first additional argument is whether all acquisition runs should be registered to a common reference image
If so the second argument (the first [main] argument being the path to the hdf) should be 'common' otherwise '0'

The second additional argument is whether you would like to see the automatically selected reference image to
be used to register the data to.


"""







import twoptb as MP


parser = argparse.ArgumentParser(description="Motion register data that has been converted to hdf5")
parser.add_argument("hdfPath", type=str,
                help="Specify path to HDF5 file that should be motion registered")
parser.add_argument("common", type=str,
                help="Should all data be registered to a common reference image? \nOptions are 1 or 0")
parser.add_argument("-show_ref",action='store', type=bool,default=False,
                help="Show selected reference image before running registration?")
#parser.add_argument("-h",'--help','--h',required=False,help=helpm)
args = parser.parse_args()

fpath = os.path.abspath(args.hdfPath)
#print fpath
absPath = os.path.split(os.path.abspath(fpath))[0]
fName = os.path.split(os.path.abspath(fpath))[1]
print fName
if args.common==str(1):
  common_ref = 1
elif args.common==str(0):
  common_ref = 0
else:
  raise("ERROR common must either be 0 or 1")
print args.common
srm = args.show_ref
print "Using common reference %s" %common_ref

if srm==True:
  print 'Showing common refernce'
else:
  print("Not showing common reference")
#print os.path.split(absPath)[0]
############## Load the HDF File

absDir = os.path.split(absPath)[0]

procFDir = os.path.dirname(absDir)
print( absDir)
procFloc = os.path.join(procFDir,'proc_log.txt')

with open(procFloc,'a') as f:
    f.write('motion registered %s' %fName)

HDF_File = h5py.File(fpath,'a',libver='latest')
session_ID = HDF_File.keys()[0]

print('Session ID: %s \n' %session_ID)

areas = HDF_File[session_ID]['raw_data'].keys()

Areas = HDF_File[session_ID]['raw_data'].keys()

print('Registering areas:')
for a in Areas:
    print(a)
st = time.time()

#use inRAM if you have a lot of RAM. registering 40500 512x512 frames in
#parallel will eat around 120GB of RAM while without it will eat ~500MB
#
HDF_File = MP.image_registration.register_dayData(HDF_File=HDF_File,
                                                  session_ID=session_ID,
                                                  inRAM=False,
                                                  poolSize=16,
                                                  abs_loc=absPath,
                                                  common_ref=common_ref,
                                                  show_ref_mean=srm)
print time.time() - st

print 'Data Registered successfully: %s' %(all([i in (HDF_File[session_ID]['raw_data'].keys()) for i in HDF_File[session_ID]['registered_data'].keys()]))
