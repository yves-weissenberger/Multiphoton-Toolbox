#!/home/yves/anaconda2/bin/python


from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os, re, sys, pickle,time
import h5py
import seaborn
import matplotlib
seaborn.set_style('whitegrid')

sys.path.append('/home/yves/Documents/')
import twoptb as MP

hdf_path = os.path.abspath(sys.argv[1])
hdf = h5py.File(hdf_path,'r+',libver='latest') #MP.file_management.load_hdf5(hdf_path,'wb')
tonemap = hdf[u'20170201-213e']['registered_data']#hdf['tonemapping']['registered_data']
areas = tonemap.keys()
print areas
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

def get_tuning_curves(areaF,centre=None):
    
    
    
    #load the associated data
    #print areaF.attrs.keys()
    grabI = pickle.load(open(areaF.attrs['GRABinfo']))
    outDat = DM = pickle.load(open(areaF.attrs['stimattrs']))
    ROI_attrs = pickle.load(open(os.path.join(os.path.split(hdf.filename)[0],areaF.attrs['ROI_dataLoc']))) #this is monkey patch
    
    #absolute locations
    if centre==None:
        FOV_centre = np.array(grabI['xyzPosition'][:2])
    else:
        FOV_centre = centre
    #print FOV_centre
    ## so centres[0] is the x-coordinate and centres[1] is the y coordinate
    roi_centres = np.array(ROI_attrs['centres'])
    xPos = -roi_centres[:,0] + FOV_centre[0]
    yPos = roi_centres[:,1] + FOV_centre[1]
    absROI_pos = -np.vstack([xPos,yPos])
    
    
    
    n_neurons = len(ROI_attrs['traces'])
    resps = np.zeros([n_neurons,outDat['stim_list'].shape[0]])
    
    for neuron in range(n_neurons):
        #print neuron,
        if True:#'df_F' not in ROI_attrs.keys():
            trace = ROI_attrs['traces'][neuron]
            filt_trace = MP.process_data.runkalman(trace,50000)
            use_trace = (trace - filt_trace)/filt_trace
        else:
            use_trace = ROI_attrs['df_F']


        stimSP = outDat['stim_spacing']
        for idx,stim in enumerate(outDat['stimOrder']):
            if idx>0:
                resps[neuron,stim-1] += np.mean(use_trace[(idx*stimSP):10+(idx*stimSP)]) - np.mean(use_trace[(idx*45)-5:(idx*45)-3])
            else:
                resps[neuron,stim-1] += np.mean(use_trace[(idx*stimSP):10+(idx*stimSP)])

        resps[neuron,:] = ma(resps[neuron,:],2)



    good = (np.max(resps,axis=1) - np.mean(resps,axis=1))>.2
    BFs = np.argmax(resps,axis=1)
    gBFs = BFs[good]
    gPos = absROI_pos[:,good]
    return absROI_pos[:,:n_neurons],BFs,gBFs, gPos, resps[good], (np.max(resps,axis=1) - np.mean(resps,axis=1))








#####################################################################
    #cs = 0
idx =0
for area in areas:
    #centre = FOV_centres_mary[idx]
    centre = None

    areaF = tonemap[area]
    if ('Tonotopy' in area or 'area' in area or 'Area' in area):


        print 'processing area %s:' %idx,

        print area
        if 'stimattrs' not in areaF.attrs.keys():
            areaF.attrs['stimattrs'] = hdf['tonemapping']['raw_data'][area].attrs['stimattrs']

        if 'ROI_dataLoc' in areaF.attrs.keys():
            if idx==0:
                absROI_pos,BFs,gBFs,gPos,Tcs,goodN = get_tuning_curves(areaF,centre=centre)
            else:
                T_absROI_pos,T_BFs,T_gBFs, T_gPos, T_Tcs,T_goodN = get_tuning_curves(tonemap[area],centre=centre)
                absROI_pos = np.hstack([absROI_pos,T_absROI_pos])
                BFs = np.concatenate([BFs,T_BFs])
                gBFs = np.concatenate([gBFs,T_gBFs])
                gPos = np.hstack([gPos,T_gPos])
                Tcs =  np.vstack([Tcs,T_Tcs])
                goodN = np.concatenate([goodN,T_goodN])

            idx += 1
            print idx
    else:
        pass


cmapN = 'jet'
cmap = matplotlib.cm.ScalarMappable(cmap=cmapN )
norm = matplotlib.colors.Normalize(vmin=0, vmax=15)
c=cmap.to_rgba(np.flipud(np.arange(0,15,1))).reshape(15,1,4)
cc = np.tile(c,[1,4,1])
cc[:,:,3] *= cc[:,:,3]*np.linspace(1,0,num=4)[None,:]

cmap = matplotlib.cm.ScalarMappable(cmap=cmapN )
norm = matplotlib.colors.Normalize(vmin=0, vmax=15)
#cmap.set_array(BFs)

cls = cmap.to_rgba(BFs)
cls[:,-1] = (goodN)/np.std(goodN)
nperc = np.percentile((goodN)/np.std(goodN),80)

cls[np.where(cls[:,-1]>=nperc),-1] = nperc
cls[:,-1] /= np.max(cls[:,-1])

cmap.set_array(cls)


fig = plt.figure(figsize=(8,8))
ax1 = fig.add_axes([0.1,0.1,.7,.8])

scat = ax1.scatter(-absROI_pos[1],absROI_pos[0],c=cls,
            s=48,linewidth=.5,edgecolor=[.4]*3,cmap=cmap)

plt.xlim(-2000,1000)
plt.ylim(-2000,1000)

ax2 = fig.add_axes([.8,0.1,.2,.8])
ax2.imshow(cc,aspect=3)
ax2.grid()
plt.yticks([])
plt.xticks([])
plt.xlabel('Tuning Strength')
plt.ylabel('Best Frequency')
plt.yticks([])
plt.xlim(-2000,1000)
plt.ylim(-2000,1000)
plt.show(block=False)


plt.figure(figsize=(8,8))
plt.scatter(-gPos[1],gPos[0],c=gBFs,
            s=36,linewidth=.5,edgecolor=[.4]*3,cmap=cmapN )
plt.colorbar()
plt.show()
#plt.colorbar(scat)
#plt.savefig('~/Desktop/map_peter.svg')
plt.xlim(-2000,1000)
plt.ylim(-2000,1000)
#plt.savefig('/home/yves/Desktop/mary.png')
