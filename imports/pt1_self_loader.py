# coding: utf-8


from __future__ import division
import numpy as np
import copy as cp
import os
import re
import csv



def _get_files(dataPath,mouseID,train_type,date):
    """ helper function for  load_day_behaviour """
    Fs = os.listdir(dataPath)
    Fs_filt1 = [f for f in Fs if mouseID in f]
    Fs_filt2 = [f for f in Fs_filt1 if date in f]
    Fs_filt3 = [f for f in Fs_filt2 if train_type in f]
    Fs_filt = [f for f in  Fs_filt3 if '.mat' not in f]

    Fs_filt.sort()
    return Fs_filt


# In[162]:

def _join_data_files(dicts,t_type='time'):
    """ helper function for  load_day_behaviour, 
        concatenates data dicts
    """
    if t_type=='time':
        dicts = [d[0] for d in dicts]
    else:
        dicts = [d[1] for d in dicts]
    
    nDicts = len(dicts)
    dictNs = [d['fName'] for d in dicts]
    order = sorted(range(nDicts), key=lambda k: dictNs[k])
    
    offset = 0
    
    bigDict = dicts[order[0]]
    maxT = np.max([np.max(bigDict[k]) for k in bigDict.keys() if (k!='fName'
        and len(bigDict[k])!=0)])

    offset = maxT
    for idx in order[1:]:
        
        #sys.stdout.write(idx + "\n")
        dd = dicts[idx]
        
        maxT = np.max([np.max(dd[k]) for k in dd.keys() if (k!='fName' and
            len(dd[k])!=0)])

        for k in bigDict.keys():
            if k!='fName':
                bigDict[k] = bigDict[k] + [i+offset for i in dd[k]]
        
        offset += maxT
    
    for k in bigDict.keys():
        if k!='fName':
            bigDict[k] = np.array(bigDict[k])
    return bigDict


# In[163]:

def _load_pretraining1_self(fPath):
    """ helper function for  load_day_behaviour loads
        behaviour data timestamps from individual files
    """
    with open(fPath, 'r') as fl:
        reader = csv.reader(fl)


        lickLs = []; lickRs = []; clicks = []; rews = [];
        lickLsF = []; lickRsF = []; clicksF = []; rewsF = []; 
        freeRrews = []; freeLrews = []; freeRrewsF = []; freeLrewsF = [];
        t_old = -1
        done=False
        for row in reader:
            if not done:
                if 'lick:L' in row[0]:
                    t, frameN = re.findall(r'.*[R,L]_(.*)',row[0])[0].split('_')

                    if float(frameN)>0:
                        if float(t)<float(t_old):
                            done = True    
                        else:
                            lickLs.append(float(t));lickLsF.append(float(frameN))


                if 'lick:R' in row[0]:
                    t, frameN = re.findall(r'.*[R,L]_(.*)',row[0])[0].split('_')

                    if float(frameN)>0:
                        if float(t)<float(t_old):
                            done = True
                        else:
                            lickRs.append(float(t));lickRsF.append(float(frameN))


                if 'Sound:click' in row[0]:
                    t, frameN = re.findall(r'.*click_(.*)',row[0])[0].split('_')

                    if float(frameN)>0:
                        if float(t)<float(t_old):
                            done = True 
                        else:
                            clicks.append(float(t));clicksF.append(float(frameN))

                if 'rew:RL' in row[0]:
                    t, frameN = re.findall(r'.*RL_(.*)',row[0])[0].split('_')

                    if float(frameN)>0:
                        if float(t)<float(t_old):
                            done = True
                        else:
                            rews.append(float(t));rewsF.append(float(frameN))
                if 'freeRew' in row[0]:
                    side,t,frameN = re.findall(r'.*freeRew:(.*)',row[0])[0].split('_')

                    if float(frameN)>0:
                        if float(t)<float(t_old):
                            done = True
                        else:
                            if side=='3':
                                freeRrews.append(float(t));freeRrewsF.append(float(frameN))
                            elif side=='4':
                                freeLrews.append(float(t));freeLrewsF.append(float(frameN))
                    

                t_old = cp.deepcopy(t)
                
    lickLs.sort(); lickRs.sort(); lickLsF.sort(); lickRsF.sort()
    clicks.sort(); clicksF.sort(); rews.sort(); rewsF.sort()
    freeRrews.sort(); freeRrews.sort(); freeLrews.sort(); freeLrewsF.sort()
    

    rewR = []
    rewR = rews + freeRrews
    rewR.sort()
    rewRF = []
    rewRF = rewsF + freeRrewsF; rewRF.sort()

    rewL = []
    rewL = rews + freeLrews; rewL.sort()
    rewLF = []
    rewLF = rewsF + freeLrewsF; rewLF.sort()

    
    data_t = {'lickL': lickLs,
              'lickR': lickRs,
              'clicks': clicks,
              'rews': rews,
              'freeR': freeRrews,
              'freeL': freeLrews,
              'rewR': rewR,
              'rewL': rewL,
              'fName': os.path.split(fPath)[-1]}
    
    
    data_F = {'lickL': lickLsF,
              'lickR': lickRsF,
              'clicks': clicksF,
              'rews': rewsF,
              'freeR': freeRrewsF,
              'freeL': freeLrewsF,
              'rewR': rewRF,
              'rewL': rewLF,
              'fName': os.path.split(fPath)[-1]}
    
    
    return data_t, data_F


# In[164]:

def load_day_behaviour(baseDir,mouseID,date,train_type,t_type='time'):
    
    """ Load entire behaviour session from multiple files concatenated
        
        Arguments:
        ==================================================
        baseDir:    str
                    location of the data files
        
        mouseID:    str
                    name of mouse

        date:       str
                    date looking for in form yyyymmdd e.g. 20161031

        train_type: type of pretretraining

        t_type:     str 
                    t_type=='time' if want the time-stamps in ms otherwise
                    returns in frames of imaging acquisition


    """
    
    #just get the data locations
    fs = _get_files(baseDir,mouseID,train_type,date)
        
    
    fs = [f for f in fs if os.stat(os.path.join(baseDir,f)).st_size>0]
    #sys.stdout.write(fs + "\n")
    dataDicts = [_load_pretraining1_self(os.path.join(baseDir,f)) for f in fs]
    data_by_sess = cp.deepcopy(dataDicts)
    allDict =_join_data_files(dataDicts,t_type=t_type)
    allDict['fName'] = mouseID + '_' + train_type + '_' + date


    return allDict, data_by_sess
