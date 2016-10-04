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

hdf_path = sys.argv[1]
hdf = MP.file_management.load_hdf5(hdf_path)
tonemap = hdf['tonemapping']['registered_data']
areas = tonemap.keys()

def get_tuning_curves(areaF,centre=None):
    
    
    
    #load the associated data
    grabI = pickle.load(open(areaF.attrs['GRABinfo']))
    outDat = DM = pickle.load(open(areaF.attrs['outDat_path']))
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
    yPos = roi_centres[:,1]+FOV_centre[1]
    absROI_pos = np.vstack([xPos,yPos])
    
    
    
    n_neurons = len(ROI_attrs['traces'])
    resps = np.zeros([n_neurons,outDat['stim_list'].shape[0]])
    
    for neuron in range(n_neurons):
        #print neuron,
        trace = ROI_attrs['traces'][neuron]
        filt_trace = MP.process_data.runkalman(trace,2000)
        use_trace = (trace - filt_trace)/filt_trace
        for idx,stim in enumerate(outDat['stimOrder']):
            if idx>0:
                resps[neuron,stim-1] += np.mean(use_trace[(idx*45):6+(idx*45)]) - np.mean(use_trace[(idx*45)-4:(idx*45)])
            else:
                resps[neuron,stim-1] += np.mean(use_trace[(idx*45):6+(idx*45)])
                
    good = (np.max(resps,axis=1) - np.mean(resps,axis=1))>.5
    BFs = np.argmax(resps,axis=1)
    gBFs = BFs[good]
    gPos = absROI_pos[:,good]
    return absROI_pos[:,:n_neurons],BFs,gBFs, gPos, resps[good], (np.max(resps,axis=1) - np.mean(resps,axis=1))

    cs = 0
for idx,area in enumerate(areas):
    print 'processing area %s' %idx
    #centre = FOV_centres_mary[idx]
    centre = None
    if idx==0:
        absROI_pos,BFs,gBFs,gPos,Tcs,goodN = get_tuning_curves(tonemap[area],centre=centre)
    else:
        T_absROI_pos,T_BFs,T_gBFs, T_gPos, T_Tcs,T_goodN = get_tuning_curves(tonemap[area],centre=centre)
        absROI_pos = np.hstack([absROI_pos,T_absROI_pos])
        BFs = np.concatenate([BFs,T_BFs])
        gBFs = np.concatenate([gBFs,T_gBFs])
        gPos = np.hstack([gPos,T_gPos])
        Tcs =  np.vstack([Tcs,T_Tcs])
        goodN = np.concatenate([goodN,T_goodN])


cmap = matplotlib.cm.ScalarMappable(cmap='viridis')
norm = matplotlib.colors.Normalize(vmin=0, vmax=15)
c=cmap.to_rgba(np.flipud(np.arange(0,15,1))).reshape(15,1,4)
cc = np.tile(c,[1,4,1])
cc[:,:,3] *= cc[:,:,3]*np.linspace(1,0,num=4)[None,:]

cmap = matplotlib.cm.ScalarMappable(cmap='viridis')
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
plt.show(block=False)


plt.figure(figsize=(8,8))
plt.scatter(-gPos[1],gPos[0],c=gBFs,
            s=36,linewidth=.5,edgecolor=[.4]*3,cmap='viridis')
plt.colorbar()
plt.show()
#plt.colorbar(scat)
#plt.savefig('~/Desktop/map_peter.svg')
#plt.xlim(-2000,1000)
#plt.ylim(-2000,1000)
#plt.savefig('/home/yves/Desktop/mary.png')