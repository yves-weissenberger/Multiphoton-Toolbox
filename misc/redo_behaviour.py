from __future__ import division
import re, os, csv, time, sys
import itertools
import pickle
import scipy.io as spio
import numpy as np


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
from twoptb.util import load_GRABinfo, get_DM
from twoptb.imports import pt1_self_loader



if __name__ == "__main__":
    pth = sys.argv[1]
    stimFs = os.listdir(os.path.join(pth,'stims'))

    root_path = sys.argv[1].split('processed')[0]

    data_folder = [i for i in os.listdir(root_path) if 'proc' not in i][0]

    data_path = os.path.join(root_path,data_folder)


    for i in stimFs:
        #print i
        acqD = re.findall(r'(.*)_GRAB*',i)[0]
        stim_txt = [j for j in os.listdir(os.path.join(data_path,acqD)) if '_data.txt' in j][0]
        stimtxtLoc = os.path.join(data_path,acqD,stim_txt)
        stimattrs = pt1_self_loader.load_pretraining1_self(stimtxtLoc)
        save_pth = os.path.join(pth,'stims',i)
        with open(save_pth,'wb') as saL:
            pickle.dump(stimattrs,saL)