#!/home/yves/anaconda2/bin/python
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os, re, sys, pickle,time
from scipy.stats import f_oneway
from scipy.stats import sem
from matplotlib.patches import Rectangle   
import h5py
import seaborn
import matplotlib
from scipy.optimize import curve_fit

seaborn.set_style('whitegrid')
def findpath():
    cDir = os.path.dirname(os.path.realpath(__file__))

    found = False
    while not found:
        cDir,ext = os.path.split(cDir) 
        print ".",
        if ext=='twoptb':
            found = False
            twoptb_path = cDir
            print 
            break
    return twoptb_path

twoptb_path = findpath()
sys.path.append(twoptb_path)

import twoptb as MP

hdf_path = os.path.abspath(sys.argv[1])

if len(sys.argv>2):
    objective = sys.argv[2]
    if objective == 16:
        objective_multiplier = 1000./512.
    elif objective==20:
        objective_multiplier =  (1000./512.) * (16./20.)

hdf = h5py.File(hdf_path,'r+',libver='latest') #MP.file_management.load_hdf5(hdf_path,'wb')
#print hdf.keys()
tonemap = hdf[u'tonemapping']['registered_data']#hdf['tonemapping']['registered_data']
areas = tonemap.keys()
#print areas

def get_big_DM(x,n_back,rT,descriptor=None):
    """ Build the Big Design Matrix with
        all kinds of offsets """

    n_timePoints = x.shape[1]
    n_vars = x.shape[0]-1



    if type(rT)==int:
        rT = [rT]*n_vars


    if type(n_back)==int:
        nDims = 1 + n_back*(n_vars)
        DM = np.zeros([nDims,n_timePoints])
        n_back = [n_back]*n_vars
    elif (type(n_back)==list or type(n_back)==np.ndarray):
        nDims = 1 + np.sum(n_back) - np.sum(rT)
        DM = np.zeros([nDims,n_timePoints])


    DM_descriptor = ['offset']
    DM[0] = 1

    if type(descriptor)==type(None):
        descriptor = ['NA']*nVars

    dmidx = 1
    for i,nm in zip(range(n_vars),descriptor):
        n_backVar = n_back[i]
        for offset in range(n_backVar):

            if offset>=rT[i]:
                if offset==0:
                    dmi = x[i+1]
                else:
                    dmi = np.concatenate([[0]*offset,x[i+1,:-(offset)]])

                DM_descriptor.append(nm + str(offset))
                DM[dmidx] = dmi
                dmidx += 1
    return DM, np.array(DM_descriptor)


def get_DM(areaF):

    area_trace = np.mean(areaF,axis=(1,2))
    nT = len(area_trace)
    area_trace = zscore(area_trace)
    outDat = DM = pickle.load(open(areaF.attrs['stimattrs']))
    nStims = len(np.unique(outDat['stimOrder']))

    x = np.zeros([nStims+1,nT])
    ISI = outDat['stim_spacing']
    stimOrder = outDat['stimOrder']

    for idx,stim in enumerate(stimOrder):
        x[stim,idx*ISI] = 1

    lens = [40]*nStims
    x[0] = 1
    lens = [40]*nStims
    DM,desc = get_big_DM(x,lens,0,['offs','sw'] + [str(i) for i in range(nStims)])

    AT = MP.process_data.runkalman(area_trace,10)[:-100]
    DM = np.vstack([AT,DM])
    return DM

def ma(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def gauss_function(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))



def get_tuning_curves(areaF,centre=None):
    
    #load the associated data
    #print areaF.attrs.keys()
    grabI = pickle.load(open(os.path.join(os.path.join(os.path.split(hdf_path)[0],"GRABinfos"),os.path.split(areaF.attrs['GRABinfo'])[-1])))
    outDat = DM = pickle.load(open(os.path.join(os.path.join(os.path.split(hdf_path)[0],"stims"),os.path.split(areaF.attrs['stimattrs'])[-1])))

    #outDat = DM = pickle.load(open(areaF.attrs['stimattrs']))
    ROI_attrs = pickle.load(open(os.path.join(os.path.split(hdf.filename)[0],areaF.attrs['ROI_dataLoc']))) #this is monkey patch
    
    #absolute locations
    if centre==None:
        FOV_centre = np.array(grabI['xyzPosition'][:2])
    else:
        FOV_centre = centre
    #print FOV_centre
    ## so centres[0] is the x-coordinate and centres[1] is the y coordinate

    zoom_multiplier = grabI['scanZoomFactor']

    roi_centres = np.array(ROI_attrs['centres'])
    xPos = (-roi_centres[:,0]*objective_multiplier/zoom_multiplier) + FOV_centre[0]
    yPos = (roi_centres[:,1]*objective_multiplier/zoom_multiplier) + FOV_centre[1]
    absROI_pos = -np.vstack([xPos,yPos])
    
    
    
    n_neurons = len(ROI_attrs['traces'])
    resps = np.zeros([n_neurons,outDat['stim_list'].shape[0]])
    resp = np.zeros((n_neurons,outDat['stim_list'].shape[0],25))
    respall = np.zeros((n_neurons,outDat['stim_list'].shape[0],25,15))
    resperr = np.zeros((n_neurons,outDat['stim_list'].shape[0],25))
    
    for neuron in range(n_neurons):
        sys.stdout.write("\rprocessing neuron: %s/%s" %(neuron+1,n_neurons))
        sys.stdout.flush()
        if True:#'df_F' not in ROI_attrs.keys():

            ############ YVES EDIT NEUROPIL CORRECTION

            mpossx= ROI_attrs['idxs'][neuron][0]
            mpossy = ROI_attrs['idxs'][neuron][1]
            xLims = [np.min(mpossx)-10,np.max(mpossx)+10]
            yLims = [np.min(mpossy)-10,np.max(mpossy)+10]
            temp = areaF[:,yLims[0]:yLims[1],xLims[0]:xLims[1]] *np.abs(ROI_attrs['masks'][neuron]-1)
            temp = temp.astype('float64')
            temp[temp==0] = np.nan
            neuropil_trace = np.nanmean(temp,axis=(1,2))

            ##############END##########################


            trace = ROI_attrs['traces'][neuron] - .7*neuropil_trace
            filt_trace = MP.process_data.runkalman(trace,50000)
            use_trace = (trace - filt_trace)/filt_trace
        else:
            use_trace = ROI_attrs['df_F']


        stimSP = outDat['stim_spacing']
        istim = np.zeros((outDat['stim_list'].shape[0]))
        meanwin = [0,15]
        for idx,stim in enumerate(outDat['stimOrder']):
            istim[stim-1] += 1
            resp0 = use_trace[(idx*stimSP)-5:20+(idx*stimSP)] #average fluorescence from -5 to 20 frames post-stim (for plotting)
            resp00 = use_trace[(idx*stimSP):20+(idx*stimSP)] #average fluorescence 0-20 frames post-stim (only for the first stimulus)
            resp1 = use_trace[(idx*stimSP)+meanwin[0]:(idx*stimSP)+meanwin[1]] #average fluorescence 0-10 frames post-stim 
            resp2 = use_trace[(idx*stimSP)-5:(idx*stimSP)] #average fluorescence right before stim (5 frames)
            resp3 = use_trace[(idx*stimSP)+10:(idx*stimSP)+15] #average fluorescence right after stim (5 frames)
            if idx>0: 
                respall[neuron,stim-1,:,istim[stim-1]-1] = resp0 - np.mean(resp2)
                resp[neuron,stim-1,:] += resp0 - np.mean(resp2)
                resps[neuron,stim-1] += np.mean(resp1) - np.mean(resp2)

            else:
                respall[neuron,stim-1,:,istim[stim-1]-1] = np.lib.pad(resp00,(5,0),'minimum')
                resp[neuron,stim-1,:] += np.lib.pad(resp00,(5,0),'minimum')
                resps[neuron,stim-1] += np.mean(resp1)

        resps[neuron,:] = ma(resps[neuron,:],2) #smoothing of tuning curves
        #resperr[neuron,:,:] = sem(respall[neuron,:,:,:],axis=2) #compute standard error of the mean for raw stim-evoked responses
        resperr[neuron,:,:] = np.std(respall[neuron,:,:,:],axis=2) #compute standard deviation for raw stim-evoked responses

    
    tunstrength = []
    gaussians = []

    for n in resps:

        try:
            popt,_ = curve_fit(gauss_function,np.arange(len(n)),n)
            tunstrength.append(np.corrcoef(n,gauss_function(np.arange(len(n)),popt[0],popt[1],popt[2]))[0,1])
            
        except RuntimeError:
            tunstrength.append(0)
            popt = [np.nan]*3

        gaussians.append(popt)

    #good = ((np.array(good1)>.3 ).astype("int") * (np.array([i[0] for i in gaussians ])>0).astype("int")).astype("bool") #boolean telling us which ROIs to keep (must have positive gaussian with R^2 > 0.3)
    
    tunstrength = []
    tunstrength = (np.max(resps,axis=1) - np.mean(resps,axis=1))
    good = tunstrength>0.3
    BFs = np.argmax(resps,axis=1)
    gBFs = BFs[good]
    gPos = absROI_pos[:,good]
    return absROI_pos[:,:n_neurons],BFs,gBFs, gPos, resps, resp, resperr, tunstrength, good, FOV_centre, meanwin


#####################################################################
    #cs = 0
idx =0
iROI = []
igROI = []
Tcurves = []
Tresp = []
Tresperr = []
Tbfs = []
for area in areas:
    #centre = FOV_centres_mary[idx]
    centre = None

    areaF = tonemap[area]
    if ('Tono' in area or 'Tonotopy' in area or 'area' in area or 'Area' in area):


        print 'processing area %s:' %idx,

        print area
        if 'stimattrs' not in areaF.attrs.keys():
            areaF.attrs['stimattrs'] = hdf['tonemapping']['raw_data'][area].attrs['stimattrs']

        if 'ROI_dataLoc' in areaF.attrs.keys():
            if idx==0:
                absROI_pos,BFs,gBFs,gPos,Tcs,resp,resperr,goodN,iG,FOVcentre,meanwin = get_tuning_curves(areaF,centre=centre)
                Tbfs.append(BFs)
                Tcurves.append(Tcs)
                Tresp.append(resp)
                Tresperr.append(resperr)
                igROI=np.arange(len(gPos[0]))
                iROI=np.arange(len(absROI_pos[0]))
                iGood=iG
            else:
                T_absROI_pos,T_BFs,T_gBFs, T_gPos, Tcs, T_resp, T_resperr,T_goodN,T_iG,T_FOVcentre,meanwin = get_tuning_curves(tonemap[area],centre=centre)
                absROI_pos = np.hstack([absROI_pos,T_absROI_pos])
                Tbfs.append(T_BFs)
                BFs = np.concatenate([BFs,T_BFs])
                gBFs = np.concatenate([gBFs,T_gBFs])
                gPos = np.hstack([gPos,T_gPos])
                Tcurves.append(Tcs)
                Tresp.append(T_resp)
                Tresperr.append(T_resperr)
                goodN = np.concatenate([goodN,T_goodN])
                iGood = np.concatenate([iGood,T_iG])
                FOVcentre = np.vstack([FOVcentre,T_FOVcentre])
                igROI = np.hstack([igROI,np.arange(len(T_gPos[0]))])
                iROI = np.hstack([iROI,np.arange(len(T_absROI_pos[0]))])

            idx += 1
    else:
        pass

f = open('tono.pickle', 'wb') 
pickle.dump([absROI_pos,Tbfs,BFs,gBFs,gPos,Tcurves,Tresp,Tresperr,goodN,iGood,FOVcentre,igROI,iROI,areas,meanwin], f)
f.close()

cmapN = 'jet'
cmap = matplotlib.cm.ScalarMappable(cmap=cmapN )
norm = matplotlib.colors.Normalize(vmin=0, vmax=15)
c=cmap.to_rgba(np.flipud(np.arange(0,15,1))).reshape(15,1,4)
cc = np.tile(c,[1,4,1])
cc[:,:,3] *= cc[:,:,3]*np.linspace(1,0,num=4)[None,:]

cmap = matplotlib.cm.ScalarMappable(cmap=cmapN )
norm = matplotlib.colors.Normalize(vmin=0, vmax=15)
#cmap.set_array(BFs)

cl = cmap.to_rgba(BFs)
cl[:,-1] = 0.8

gcl = cmap.to_rgba(gBFs)
gcl[:,-1] = 0.8

cls = cl
cls[:,-1] = (goodN)/np.std(goodN)
nperc = np.percentile((goodN)/np.std(goodN),80)

cls[np.where(cls[:,-1]>=nperc),-1] = nperc
cls[:,-1] /= np.max(cls[:,-1])

cmap.set_array(cls)

st = locals()

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_axes([0.1,0.1,.7,.8])

#scat = ax1.scatter(-absROI_pos[1],absROI_pos[0],c=cls,
#            s=48,linewidth=.5,edgecolor=[.4]*3,cmap=cmap)
            # parametrize 's' to make marker-size vary with tuning strength
            
#scat = ax1.scatter(-absROI_pos[1],absROI_pos[0],c='none',
#            s=np.round((goodN**2)*200),linewidth=0.1*np.round(4+20*goodN),edgecolor=cl)

scat = ax1.scatter(-gPos[1],gPos[0],c='none',
            s=np.round((goodN**2)*200),linewidth=0.1*np.round(4+20*goodN),edgecolor=gcl)

#anotate with ROI nr
#for i, txt in enumerate(iROI):
#    ax1.annotate(txt, (-gPos[1][i],gPos[0][i]))

#plt.xlim(-2000,1000)
#plt.ylim(-2000,1000)

ax2 = fig.add_axes([.8,0.1,.2,.8])
ax2.imshow(cc,aspect=3)
ax2.grid()
plt.yticks([])
plt.xticks([])
plt.xlabel('Tuning Strength')
plt.ylabel('Best Frequency')
plt.yticks([])
#plt.xlim(-2000,1000)
#plt.ylim(-2000,1000)
plt.show(block=False)


plt.figure(figsize=(8,8))
ax = plt.gca()
#for i,area in enumerate(areas):
#    ax.add_patch(Rectangle((FOVcentre[i,1]+10, -FOVcentre[i,0]+10),500,470,facecolor='grey',alpha=0.3))
#    plt.annotate(i,(FOVcentre[i,1]+15, -FOVcentre[i,0]+15),color='grey',fontsize=12,horizontalalignment='left',verticalalignment='bottom')

plt.scatter(-absROI_pos[1],absROI_pos[0],c=BFs,
            s=36,linewidth=.5,edgecolor=[.4]*3,cmap=cmapN )

#anotate with ROI nr
#for i, txt in enumerate(iROI):
#    if iGood[i]:
#        plt.annotate(txt, (-absROI_pos[1][i],absROI_pos[0][i]), color='black', fontsize=8)
#    else:
#        plt.annotate(txt, (-absROI_pos[1][i],absROI_pos[0][i]), color='grey', fontsize=8) #comment this out if ROIs that aren't good shouldn't be annotated
    
plt.colorbar()
plt.show(block=False)
#plt.colorbar(scat)
#plt.savefig('~/Desktop/map_peter.svg')
#plt.xlim(-2000,1000)
#plt.ylim(-2000,1000)
#plt.savefig('/home/yves/Desktop/mary.png')

areaID=0
while areaID != -1:
    areaID = int(raw_input('Choose Area Nr: '))
    neuronID = int(raw_input('Choose ROI Nr: '))

    plt.figure(figsize=(5,3))
    ax3 = fig.add_axes([0.1,0.1,.7,.8])
    plt.plot(Tcurves[areaID][neuronID])
    plt.xlabel('Frequency')
    plt.ylabel('Response Strength')
    plt.show(block=False)