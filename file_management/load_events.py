
from __future__ import division
import re, os, csv, time,sys
import itertools
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



def import_imaging_behaviour(pth,pretraining_type='2'):


    f = open(pth)
    cv = csv.reader(f)
    eventList = []
    for line in cv:
        eventList.append(line[0].split(':'))
    licks = []; rews = []; snd = []
    
    for entry in eventList:
        if entry[0]=='lick':
            licks.append(entry[1].split('_'))
        if entry[0]=='Sound':
            if (pretraining_type=='1' and 'catch_trials' not in pretraining_type):
                snd.append(entry[1].split('_'))

            elif (pretraining_type=='1' and 'catch_trials' in pretraining_type and 'multilevel' not in pretraining_type):
                print 'This has not been setup yet'
                raise Exception("Not implemented error")
            elif (pretraining_type=='1' and 'catch_trials' in pretraining_type and 'multilevel' in pretraining_type):
                raise Exception("Not implemented error")
            elif pretraining_type=='2':
                #print entry
                tSL = [int(entry[1][0]),entry[2][:2]] + entry[3].split('_')
                snd.append(tSL)

            else:
                tt = entry[1].split('_')
                tSL = [int(tt[0]),float(tt[1]),int(tt[2])]
                #tSL = [int(entry[1][0]),entry[2][:2]] + entry[3].split('_')
                snd.append(tSL)
        if entry[0]=='rew':
            rews.append(entry[1].split('_'))

    rew_R = []; rew_L = []; lick_R = []; lick_L = []; snds = []
    #extract rewards----------------------------------------------------------------------
    rewLst = []
    if pretraining_type=='1':
        for entry in rews:
            rewLst.append([float(entry[1]),int(entry[2])])
    else:
        for entry in rews:
            if (entry[0]=='R' or entry[0]=='1'):
                rew_R.append([float(entry[1]),int(entry[2])])
            elif (entry[0]=='L'or entry[0]=='2'):
                rew_L.append([float(entry[1]),int(entry[2])])

    #Extract Licks------------------------------------------------------------------------

    for entry in licks:
        if entry[0]=='R':
            lick_R.append([float(entry[1]),int(entry[2])])
        elif entry[0]=='L':
            lick_L.append([float(entry[1]),int(entry[2])])

    #Extract Stimuli----------------------------------------------------------------------
    if pretraining_type=='2':
        seen = set()
        levels = np.unique([float(i[1]) for i in snd])
        freqs = np.unique([float(i[2]) for i in snd])
        unique_stims = sorted([i for i in itertools.product(levels,freqs)],key=lambda x: x[1])
        nStims = len(unique_stims)
        stims = [[] for i in range(nStims)]
        for entry in snd:
            idx = unique_stims.index((float(entry[1]),float(entry[2])))
            stims[idx].append([float(entry[3]),int(entry[4])])
    elif pretraining_type=='1':
        stims = []
        for entry in snd:
            stims.append([float(entry[1]),int(entry[2])])
    elif pretraining_type=='legacy':
        stims = []
        for entry in snd:
            stims.append([int(entry[0]),float(entry[1]),int(entry[2])])
        
    #------------------------------------------------------------------------------------

    #------------------------------------------------------------------------------------
    #Create the dictionary of values

    timestamp_dict = {}
    if pretraining_type=='2':
        
        rew_R = np.array(rew_R); rew_L = np.array(rew_L); lick_R = np.array(lick_R); lick_L = np.array(lick_L)
        rew_R[:,0] *= 1000; rew_L[:,0] *= 1000; lick_R[:,0] *= 1000; lick_L[:,0] *= 1000
        timestamp_dict['rew_R'] = rew_R
        timestamp_dict['rew_L'] = rew_L
        
        for idx,entry in enumerate(stims):
            timestamp_dict['snd_F:' + str(unique_stims[idx][1])+',V:'+str(unique_stims[idx][0])] = np.array(entry)*1000
        timestamp_dict['rew_R'] = rew_R
        timestamp_dict['rew_L'] = rew_L

    elif pretraining_type=='1':
        
        rewLst = np.array(rewLst); lick_R = np.array(lick_R); lick_L = np.array(lick_L)
        if len(rewLst)>0:
            rewLst[:,0] *= 1000
        if len(lick_R)>0:
            lick_R[:,0] *= 1000
        if len(lick_L)>0:
            lick_L[:,0] *= 1000

        timestamp_dict['rew'] = rewLst
        stims = np.array(stims)
        if len(stims)>0:
            stims[:,0] *= 1000
        timestamp_dict['click'] = np.array(stims)
    elif pretraining_type=='legacy':
        #print rew_L
        rew_R = np.array(rew_R); rew_L = np.array(rew_L); lick_R = np.array(lick_R); lick_L = np.array(lick_L)
        rew_R[:,0] *= 1000; rew_L[:,0] *= 1000; lick_R[:,0] *= 1000; lick_L[:,0] *= 1000
        timestamp_dict['rew_R'] = rew_R
        timestamp_dict['rew_L'] = rew_L
        timestamp_dict['stim'] = np.array(stims)
        
    
    timestamp_dict['lick_R'] = lick_R
    timestamp_dict['lick_L'] = lick_L
    print "first entries e.g. data_dict['lick_R'][:,0] \nare the timestamps in milliseconds, second entries are frame numbers"
    return timestamp_dict



#---------------------------------------------------------------------------------------------------
def get_triggers(matFilePth,**kwargs):

    if ('Tones2to64thirdOct' in matFilePth or 'tones2to64thirdOct' in matFilePth):
        matfile = spio.loadmat(matFilePth)
        stimattrs = {'stim_list': matfile['outDat'][0][0][2][:,0],
                     'stim_dur': matfile['outDat'][0][0][2][:,1][0],
                     'stim_levels': matfile['outDat'][0][0][2][:,5],
                     'stimScriptName': matfile['outDat'][0][0][-1][0],
                     'timestamp': matfile['outDat'][0][0][0][0],
                     'stimOrder': matfile['outDat'][0][0][1].T[0],
                     'stim_spacing': int(matfile['outDat']['sweepLengthFrames'])}


    elif re.search(r'.*search_tones_outDat.*',matFilePth):
        stim_dict = load_GRABinfo(spio.loadmat(matFilePth,struct_as_record=False, squeeze_me=True)['outDat'])
        stimattrs = {'stim_list': np.fliplr(np.vstack([stim_dict['stimMat'][:,5],
                                               stim_dict['stimMat'][:,0]]).T),
                     'stimOrder': stim_dict['trialOrder'],
                     'timestamp': stim_dict['timeStamp'],
                     'stim_dur': stim_dict['stimMat'][1,1],
                     'stim_levels': stim_dict['stimMat'][:,5],
                     'stimScriptName': 'search_tones',
                     'stim_spacing': stim_dict['sweepLengthFrames']
                    }
    elif any(s in matFilePth for s in ('Marco','Ted')):

        stimattrs = import_imaging_behaviour(matFilePth,pretraining_type='legacy')

    elif 'Pretaining1_self' in matFilePth:
        #sys.path.append('../imports/')
        #print os.listdir('../imports/')
        #import pt1_self_loader
        stimattrs = pt1_self_loader.load_pretraining1_self(matFilePth)
        #import_imaging_behaviour(matFilePth,pretraining_type='1')
        
else:
    stimattrs = None



    return stimattrs