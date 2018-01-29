#!/home/yves/anaconda2/bin/python

import os, sys, argparse, re, time
import numpy as np
import h5py
import matplotlib.pyplot as plt

import sys
import os

def findpath():
    cDir = os.path.dirname(os.path.realpath(__file__))

    found = False
    while not found:
        cDir,ext = os.path.split(cDir) 
        if ext=='twoptb':
            found = False
            twoptb_path = cDir
            print 
            break
    return twoptb_path

twoptb_path = findpath()
sys.path.append(twoptb_path)
#sys.path.append('/home/yves/Documents/')
import twoptb as MP


fpath = sys.argv[1]
#print fpath
absPath = os.path.split(os.path.abspath(fpath))[0]
fName = os.path.split(os.path.abspath(fpath))[1]
print fName
common_ref = sys.argv[2]=='common'
print "using common reference %s" %common_ref
#print os.path.split(absPath)[0]
############## Load the HDF File

absDir = os.path.split(absPath)[0]

procFDir = os.path.dirname(absDir)
print absDir
procFloc = os.path.join(procFDir,'proc_log.txt')

with open(procFloc,'a') as f:
    f.write('motion registered %s' %fName)

HDF_File = h5py.File(fpath,'a',libver='latest')
session_ID = HDF_File.keys()[0]

print 'Session ID: %s \n' %session_ID

areas = HDF_File[session_ID]['raw_data'].keys()

Areas = HDF_File[session_ID]['raw_data'].keys()

print 'Registering areas:'
for a in Areas:
    print a
st = time.time()

#use inRAM if you have a lot of RAM. registering 40500 512x512 frames in
#parallel will eat around 120GB of RAM while without it will eat ~500MB
#
HDF_File = MP.image_registration.register_dayData(HDF_File=HDF_File,
                                                  session_ID=session_ID,
                                                  inRAM=False,
                                                  poolSize=16,
                                                  abs_loc=absPath,
                                                  common_ref=common_ref)
print time.time() - st

print 'Data Registered successfully: %s' %(all([i in (HDF_File[session_ID]['raw_data'].keys()) for i in HDF_File[session_ID]['registered_data'].keys()]))
