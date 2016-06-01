from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.ptime as ptime
import pyqtgraph.console
import h5py
from pyqtgraph.dockarea import *




def add_ROIS_meanIm(Area):
    global ROI_mode
    ROI_mode = False
    global ROI_list
    ROI_loc = []
    press_loc = []
    patchSt = []
    ROI_list = []
    global Mask_list
    Mask_list = []
    patchAcc = []
    window_w = window_h = 10
    prev_on = False

    if 'ROI_centres' in [i for i in Area.attrs.iterkeys()]:
        press_loc = Area.attrs['ROI_centres'].tolist()
        prev_on = True
    else:
        press_loc = []
        prev_on = False
        
    if 'ROI_patches' in [i for i in Area.attrs.iterkeys()]:
        ROI_list = [Area.attrs['ROI_patches'][:,:,i] for i in range(Area.attrs['ROI_patches'].shape[2])]
    else:
        ROI_list = []
        prev_on = False

    if 'mean_image' in [i for i in Area.attrs.iterkeys()]:
        meanIm = Area.attrs['mean_image']
    else:
        print 'no mean image exists creating...'
        meanIm =  np.mean(Area[:-100],axis=0)
        print 'done!'

    global currROI
    currROI_L = []



    class Index(object):
        idx = len(ROI_list)
        ROI_mode = False
        shift_is_held = False
        
        
        #--------------------------------------------------------------------
        def next_roi(self, event):
            nROIs = len(ROI_list)
            currROI_L[0].remove()
            del currROI_L[0]
            if self.idx<(nROIs-1):
                self.idx += 1
            else:
                self.idx = nROIs-1
            
            if nROIs>0:
                ax2.imshow(ROI_list[self.idx],cmap='binary_r',interpolation='none')
                currROI = Circle([press_loc[self.idx][0],press_loc[self.idx][1]],6,fill=False,color='g',zorder=100)
                currROI_L.append(currROI)
                ax.add_patch(currROI)


            #print 'next_roi'
            
            
        #--------------------------------------------------------------------
        def prev_roi(self, event):
            nROIs = len(ROI_list)
            currROI_L[0].remove()
            del currROI_L[0]


            if self.idx>0:
                self.idx = self.idx - 1
            
            if nROIs>0:
                ax2.imshow(ROI_list[self.idx],cmap='binary_r',interpolation='none')
                currROI = Circle([press_loc[self.idx][0],press_loc[self.idx][1]],6,fill=False,color='g',zorder=100)
                currROI_L.append(currROI)
                ax.add_patch(currROI)
            #print 'next roi'
            
        #--------------------------------------------------------------------
        def remove_roi(self,event):
            nROIs = len(ROI_list) - 1
            if nROIs>0:
                del ROI_list[self.idx]
                del press_loc[self.idx]
                currROI_L[0].remove()
                del currROI_L[0]
                patchSt[self.idx].remove()
                del patchSt[self.idx]
                
                
        #--------------------------------------------------------------------
        def overlay_mask(self,event):
            ax2.imshow(Mask_list[self.idx])
            
        #--------------------------------------------------------------------
        def toggle_selector(self,event):
            #self.ROI_mode = True if self.ROI_mode==False else False
            #bROI_mode.label(str(self.ROI_mode))
            
            nROIs = len(ROI_list)
            #print nROIs
            if self.ROI_mode==False:
                self.ROI_mode = True
                bROI_mode.label.set_text('Stop ROI Selection')
            elif self.ROI_mode==True:
                self.ROI_mode = False
                bROI_mode.label.set_text('Enter ROI Selection')
                



        def onclick(self,event):
            if (event.name=='button_press_event'):
                if (event.inaxes==ax and
                (event.xdata<(meanIm.shape[0]-15) and event.xdata>10) and
                (event.ydata<(meanIm.shape[1]-15) and event.ydata>10)):
                    if event.button==3:
                        #print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                        #      (event.button, event.x, event.y, event.xdata, event.ydata))
                        press_loc.append([event.xdata, event.ydata])
                        #print press_loc
                        circ = Circle([event.xdata,event.ydata],6,fill=False,color=[.8,.2,.2])
                        ax.add_patch(circ)
                        patchSt.append(circ)
                        ROI_list.append(meanIm[event.ydata-window_h:event.ydata+window_h,
                                               event.xdata-window_w:event.xdata+window_w])

                        Mask_list.append(np.zeros([2*window_h,
                                                   2*window_w]))
                        nROIs = len(ROI_list) - 1
                        ax2.imshow(ROI_list[nROIs],cmap='binary_r')
                        #print len(currROI_L)
                        if len(currROI_L)==1:
                            currROI_L[0].remove()
                            del currROI_L[0]
                        if len(ROI_list)>1:
                            self.idx += 1
                            
                        self.idx = nROIs
                        currROI = Circle([press_loc[self.idx][0],press_loc[self.idx][1]],6,fill=False,color='g',zorder=100)
                        currROI_L.append(currROI)
                        ax.add_patch(currROI)
                        #print self.idx
               
        #-------------------------------------------
        #This is BAD CODE for ROI drawing but kind of works
        def on_motion(self,event):
            if event.button==1:
                
                if event.inaxes==ax2:
                    if self.shift_is_held == True:
                        print event.xdata, event.ydata, self.idx
                        
                        Mask_list[self.idx][int(event.ydata)-1:int(event.ydata)+1,
                                            int(event.xdata)-1:int(event.xdata)+1]= 1
        #-------------------------------------------        
            
            
            
        def on_key_press(self, event):
            if event.key == 'shift':
                self.shift_is_held = True

        def on_key_release(self, event):
            if event.key == 'shift':
                self.shift_is_held = False
            RGBA = np.zeros([2*window_h,
                                         2*window_w,
                                         4])
            RGBA[:,:,0] = Mask_list[self.idx]
            RGBA[:,:,3][np.where(Mask_list[self.idx]>0)] = 1
            ax2.cla()
            ax2.imshow(ROI_list[self.idx],cmap='binary_r')
            im2 = ax2.imshow(RGBA,alpha=.8)

                    



    import matplotlib
    class My_Axes(matplotlib.axes.Axes):
        name = "My_Axes"
        def drag_pan(self, button, key, x, y):
            #matplotlib.axes.Axes.drag_pan(self, button, 'x', x, y) # pretend key=='x'
            kk = None

    matplotlib.projections.register_projection(My_Axes)
        
                



    callback = Index()
                
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_axes([0.05, 0.1, 0.6, 1],projection="My_Axes")
    image = ax.imshow(meanIm,cmap='binary_r')

    ax2 = fig.add_axes([0.675, 0.675, .3, .3],projection="My_Axes")
    ax2.imshow(np.zeros([20,20]),cmap='binary_r')


    #------------------------------------------------------------------------------
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next_roi)


    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev_roi)


    axROI_mode = plt.axes([0.7, 0.525, 0.2, 0.04])
    bROI_mode = Button(axROI_mode, 'Select_ROIs')
    bROI_mode.on_clicked(callback.toggle_selector)


    rmvROI = plt.axes([0.7, 0.475, 0.2, 0.04])
    rmvROI = Button(rmvROI, 'Remove ROIs')
    rmvROI.on_clicked(callback.remove_roi)

    #------------------------------------------------------------------------------




    cid = image.figure.canvas.mpl_connect('button_press_event', callback.onclick)
    cid2 = image.figure.canvas.mpl_connect('motion_notify_event', callback.on_motion)
    image.figure.canvas.mpl_connect('key_press_event', callback.on_key_press)
    image.figure.canvas.mpl_connect('key_release_event', callback.on_key_release)
    #cid3 = image.figure.canvas.mpl_connect('button_press_event', callback.on_motion)

    def draw_circles(press_loc,ROI_list):
        i = 0
        for coords,patch in zip(press_loc,ROI_list):
            circ = Circle([coords[0],coords[1]],6,fill=False,color=[.8,.2,.2])
            ax.add_patch(circ)
            patchSt.append(circ)
            i+= 1
        currROI = Circle([press_loc[i-1][0],press_loc[i-1][1]],6,fill=False,color='g',zorder=100)
        currROI_L.append(currROI)
        ax.add_patch(currROI)
        
        return currROI_L, patchSt


    if prev_on:
        draw_circles(press_loc,ROI_list)



    plt.grid('off')
    plt.show()

    Area.attrs['ROI_patches'] = np.dstack(ROI_list)
    Area.attrs['ROI_centres'] = np.array(press_loc)

    return None


#def add_auto_masks_session(session_file):



#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------


def add_auto_masks_area(areaFile,addMasks=False):
    from skimage.morphology import binary_dilation, binary_erosion, disk
    from skimage import exposure
    from skimage.transform import hough_circle
    from skimage.morphology import convex_hull_image
    from skimage.feature import canny
    from skimage.draw import circle_perimeter

    import h5py


    MASKs = np.zeros(areaFile.attrs['ROI_patches'].shape)
    if 'ROI_masks' in (areaFile.attrs.iterkeys()):
        print 'Masks have already been created'
        awns = raw_input('Would you like to redo them from scratch: (answer y/n): ')
    else:
        awns = 'y'

    if awns=='y':
        MASKs = np.zeros(areaFile.attrs['ROI_patches'].shape)
        for i in range(areaFile.attrs['ROI_patches'].shape[2]):
            patch = areaFile.attrs['ROI_patches'][:,:,i]
            
            tt0 = exposure.equalize_hist(patch)
            tt = 255*tt0/np.max(tt0)
            thresh = 1*tt>0.3*255
            thresh2 = 1*tt<0.1*255

            tt[thresh] = 255
            tt[thresh2] = 0



            edges = canny(tt, sigma=2, low_threshold=20, high_threshold=30)
            try_radii = np.arange(3,5)
            res = hough_circle(edges, try_radii)
            ridx, r, c = np.unravel_index(np.argmax(res), res.shape)
            
            r, c, try_radii[ridx]
            image = np.zeros([20,20,4])
            cx, cy = circle_perimeter(c, r, try_radii[ridx]+2)
            
            try:
                image[cy, cx] = (220, 80, 20, 1)
                original_image = np.copy(image[:,:,3])
                chull = convex_hull_image(original_image)
                image[chull,:] = (255, 0, 0, .4)
                
            except:
                pass
            
            
            MASKs[:,:,i] = (1*image[:,:,-1]>0)
        if addMasks:
            areaFile.attrs['ROI_masks'] = MASKs
    else:
        pass    

    return np.array(MASKs)







#-------------------------------------------------------------------------------------------------------------


def MASK_DRAWER_GUI(areaFile,restart=False):
    import numpy as np
    from pyqtgraph.Qt import QtGui, QtCore
    from pyqtgraph import Qt
    import pyqtgraph as pg
    import time
    import sys
    import copy as cp
    class Visualizator(QtGui.QMainWindow):

        def __init__(self, ROI_patches,ROI_masks):
            QtGui.QMainWindow.__init__(self)
            self.idx = 0
            self.ROI_patches = ROI_patches
            self.masks = ROI_masks
            #self.masks = np.zeros(ROI_patches.shape)
            #initialise the main window
            w = QtGui.QWidget()
            layout = QtGui.QGridLayout()
            w.setLayout(layout)
            self.prevT = time.time()

            self.img = pg.ImageItem()
            self.img.setImage(self.ROI_patches[:,:,self.idx])


            self.img_ROI = pg.ImageItem()
            mask = np.zeros([self.masks.shape[0],self.masks.shape[1],3])
            mask[np.where(self.masks[:,:,self.idx])] = (1,0,0)
            self.img_ROI.setImage(mask)
            self.img_ROI.setOpacity(.2)
            self.tx = pg.TextItem('ROI Nr: ' + str(self.idx+1) + '/' + str(nROIs))

            
            """ This section contains the code to
                create and upgrade the histogram
                used to control the image """
            self.histLI = pg.HistogramLUTWidget(image=self.img,fillHistogram=False)
            self.histLI.autoHistogramRange=False
            
            self.vb = pg.ViewBox()
            self.vb.setAspectLocked(1)
            self.vb.addItem(self.img)
            self.vb.addItem(self.img_ROI)      
            self.vb.addItem(self.tx)
            grV1 = pg.GraphicsView()
            grV1.setCentralItem(self.vb)
            self.vb.scene().sigMouseMoved.connect(self.mouseMoved)


            btn1 = QtGui.QPushButton("Next ROI", self)
            btn2 = QtGui.QPushButton("Previous ROI", self)
            btn3 = QtGui.QPushButton("Save Progress", self)
            btn1.clicked.connect(self.buttonClicked)            
            btn2.clicked.connect(self.buttonClicked)
            btn3.clicked.connect(self.save_ROIS)

            layout.addWidget(grV1,0,0,7,8)
            layout.addWidget(btn1,9,1,1,1)
            layout.addWidget(btn2,9,0,1,1)
            layout.addWidget(btn3,9,4,1,1)
            layout.addWidget(self.histLI,0,8,7,2)
            self.setCentralWidget(w)
            self.show()
            #self.connect(self, Qt.SIGNAL('triggered()'), self.closeEvent
        def save_ROIS(self):
            arr = cp.deepcopy(np.array((self.masks)))
            areaFile.attrs['ROI_masks'] = arr
            print 'ROI MASKS SAVED'


        def closeEvent(self, event):
            print 'leaving now \n you have drawn %s ROIs' %self.masks.shape[0]
            event.accept() # let the window close
            #areaFile

        def buttonClicked(self):
            sender = self.sender()
            if sender.text()=='Next ROI':
                if self.idx<nROIs-1:
                    self.idx += 1
            elif sender.text()=='Previous ROI':
                if self.idx>=1:
                    self.idx -= 1
            self.tx.setText('ROI Nr: ' + str(self.idx+1) + '/' + str(nROIs))
            mask = np.zeros([self.masks.shape[0],self.masks.shape[1],3])
            #print self.masks.shape
            mask[np.where(self.masks[:,:,self.idx])] = (1,0,0)
            self.img_ROI.setImage(mask)

            self.img.setImage(self.ROI_patches[:,:,self.idx])

        def mouseMoved(self,e):

            if (time.time() - self.prevT)>0.01:
                a = self.img.mapFromScene(e)
                x_pos = a.x()
                y_pos = a.y()

                if  (y_pos<(self.masks.shape[1]-1) and  x_pos<(self.masks.shape[0]-1) and
                     y_pos>0                   and  x_pos>0):
                    
                    
                    
                    if y_pos>self.masks.shape[1]:
                        y_pos=self.masks.shape[1]
                    if x_pos>self.masks.shape[0]:
                        x_pos=self.masks.shape[0]

                    modifiers = QtGui.QApplication.keyboardModifiers()
                    if modifiers == (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier):
                        self.masks[:,:,self.idx][int(np.floor(x_pos)),int(np.floor(y_pos))] = 1
                        self.masks[:,:,self.idx][int(np.ceil(x_pos)),int(np.ceil(y_pos))] = 1

                        mask = np.zeros([self.masks.shape[0],self.masks.shape[1],3])
                        mask[np.where(self.masks[:,:,self.idx])] = (1,0,0)


                        self.img_ROI.setImage(mask)

                    elif (modifiers == QtCore.Qt.ControlModifier and
                    not modifiers == (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier)):
                        self.masks[:,:,self.idx][int(x_pos),int(y_pos)] = 1


                        mask = np.zeros([self.masks.shape[0],self.masks.shape[1],3])
                        mask[np.where(self.masks[:,:,self.idx])] = (1,0,0)
                        self.img_ROI.setImage(mask)

                    elif (modifiers == QtCore.Qt.AltModifier and
                    not modifiers == (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier)):
                        self.masks[:,:,self.idx][int(x_pos),int(y_pos)] = 0

                        mask = np.zeros([self.masks.shape[0],self.masks.shape[1],3])
                        mask[np.where(self.masks[:,:,self.idx])] = (1,0,0)
                        self.img_ROI.setImage(mask)

                    elif modifiers == (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier):
                        self.masks[:,:,self.idx][int(np.floor(x_pos)),int(np.floor(y_pos))] = 0
                        self.masks[:,:,self.idx][int(np.ceil(x_pos)),int(np.ceil(y_pos))] = 0

                        mask = np.zeros([self.masks.shape[0],self.masks.shape[1],3])
                        mask[np.where(self.masks[:,:,self.idx])] = (1,0,0)
                        self.img_ROI.setImage(mask)

                self.prevT = time.time()
                
    nROIs = areaFile.attrs['ROI_patches'].shape[2]
    #if restart==True:
        #areaFile.attrs['ROI_masks'] = np.zeros(areaFile.attrs['ROI_patches'].shape)

    if 'ROI_masks' not in (areaFile.attrs.iterkeys()):
        print 'no masks exist, creating empty ones'
        areaFile.attrs['ROI_masks'] = np.zeros(areaFile.attrs['ROI_patches'].shape)

    #roi_masks = cp.deepcopy(np.array(areaFile.attrs['ROI_masks'].astype('int')))
    app = QtGui.QApplication([])
    win = Visualizator(areaFile.attrs['ROI_patches'],areaFile.attrs['ROI_masks'])
    #app.aboutToQuit.connect(app.deleteLater)
    app.exec_()
    #print sys.exit(app.exec_())

    return app


#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------

def mask_helper_gui(areaFile):

    from skimage.morphology import binary_dilation, binary_erosion, disk, convex_hull_image
    from skimage import exposure
    from matplotlib.widgets import Button
    from skimage.draw import circle_perimeter

 
    nROIs =  areaFile.attrs['ROI_patches'].shape[2]
    mask_sh = areaFile.attrs['ROI_patches'].shape[0]
    if 'ROI_masks' in (areaFile.attrs.iterkeys()):
        masks = areaFile.attrs['ROI_masks']
        MASKs = np.zeros([nROIs,mask_sh,mask_sh,4])
        for mask_idx, mask in enumerate(masks):
            MASKs[mask_idx,mask] = (255, 0, 0, .4)
    else:
        MASKs = np.zeros([nROIs,mask_sh,mask_sh,4])

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    im = plt.imshow(np.zeros([20,20]),cmap='binary_r')
    ax = plt.gca()
    select_elem = disk(1)

    class Index(object):
        ind = 0

        def next_img(self, event):
            self.ind += 1 % nROIs
            data =  areaFile.attrs['ROI_patches'][:,:,self.ind]
            plt.title('ROI # %s' %self.ind)
            #im.set_data(data)
            ax.cla()
            ax.imshow(data,cmap='binary_r')
            ax.imshow(MASKs[self.ind])
            #plt.draw()

        def prev_img(self, event):
            if self.ind>=1:
                self.ind -= 1 % nROIs
            data =  areaFile.attrs['ROI_patches'][:,:,self.ind]
            plt.title('ROI # %s' %self.ind)
            #im.set_data(data)
            ax.cla()
            ax.imshow(data,cmap='binary_r')
            ax.imshow(MASKs[self.ind])

            plt.draw()
        
        def onclick(self,event):
            if event.inaxes==ax:
                if event.button==1:
                    if np.sum(MASKs[self.ind])==0:
                        
                        cx, cy = circle_perimeter(int(event.ydata), int(event.xdata), 3)
                        MASKs[self.ind][cx,cy] = (220, 80, 20, 1)
                        original_image = np.copy(MASKs[self.ind][:,:,3])
                        chull = convex_hull_image(original_image)
                        MASKs[self.ind][chull,:] = (255, 0, 0, .4)


                        ax.cla()
                        data =  areaFile.attrs['ROI_patches'][:,:,self.ind]
                        ax.imshow(data,cmap='binary_r')
                        ax.imshow(MASKs[self.ind])

                elif event.button==3:
                    MASKs[self.ind] = np.zeros(MASKs[self.ind].shape)

                    
                    data = areaFile.attrs['ROI_patches'][:,:,self.ind]

                    ax.cla()
                    ax.imshow(data,cmap='binary_r')
                    ax.imshow(MASKs[self.ind])
                    
                    
        def on_key_press(self,event):
            if event.key=='up':
                original_image = np.copy(MASKs[self.ind][:,:,3])
                new_image = binary_dilation(original_image,selem=select_elem)
                MASKs[self.ind][new_image>0] = (255, 0, 0, .4)
                data = areaFile.attrs['ROI_patches'][:,:,self.ind]
                ax.cla()
                ax.imshow(data,cmap='binary_r')
                ax.imshow(MASKs[self.ind])
                
            if event.key=='down':
                #print 'down'
                original_image = np.copy(MASKs[self.ind][:,:,3])
                new_image = binary_erosion(original_image,selem=select_elem)
                MASKs[self.ind][new_image==0] = (0, 0, 0, 0)
                
                
                data = areaFile.attrs['ROI_patches'][:,:,self.ind]
                ax.cla()
                ax.imshow(data,cmap='binary_r')
                ax.imshow(MASKs[self.ind])
                
                
                
            

    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next_img)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev_img)

    cid = im.figure.canvas.mpl_connect('button_press_event', callback.onclick)
    cid = im.figure.canvas.mpl_connect('key_press_event', callback.on_key_press)


    plt.show()
    areaFile.attrs['ROI_masks'] = 1*MASKs[:,:,:,-1]>0
    return None

#-------------------------------------------------------------------------------------------------------------




def view_area_video(areaFile):

    import sys
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph.ptime as ptime
    import pyqtgraph.console
    import h5py
    #from pyqtgraph.dockarea import *


    #_____________________________________________________________________________________________________ 
    b = areaFile
    nFrames, cols, rows = b.shape
    frame_idx = 0
    playVid = True
    isOverlayed = False
    roiCoordList = []
    roiList = []
    textItemList = []
    selectedROI = 0
    select_centre_pixel = False
    select_ROI = False
    remove_ROI = False
    #cols, rows = meanIm.shape
    m = np.mgrid[:cols,:rows]
    possx = m[0,:,:]# make the x pos array
    possy = m[1,:,:]# make the y pos array
    possx.shape = cols,rows
    possy.shape = cols,rows
    mask = np.concatenate([np.ones([cols,rows,1]),np.zeros([cols,rows,3])],2)
    global meanImage
    meanImage = False
    rolling_average = 0


    #___________________________________________________________________________________________________________


    """ 
        Initialise the User Interface
                                    """

    #create a window
    win = QtGui.QMainWindow()

    #and a grid layout
    layout =QtGui.QGridLayout()



    W = QtGui.QWidget()
    W.setLayout(layout)


    img = pg.ImageItem()

    #img.setImage(np.fliplr(meanIm.T),invertY=False,invertX=False,border='w',autoLevels=False)




    #-----------------------------------------------------------------
    vb = pg.ViewBox()
    vb.addItem(img)
    tx = pg.TextItem('29' + 'fps')
    tx2 = pg.TextItem('rolling average:' + '1 ')

    vb.addItem(tx)
    vb.addItem(tx2)

    vb.setAspectLocked(1)


    grV1 = pg.GraphicsView()
    #grV1.setFixedSize(512,512)
    grV1.setCentralItem(vb)
    layout.addWidget(grV1)
#___________________________________________________________________________________________________________



    """ In this section, describe keyboard interaction """


    def _keyPressEvent(ev):
        
        global playVid, playBtn, selectedROI,  select_ROI, remove_ROI
        #print ev.key()
        #print ev.key() == QtCore.Qt.Key_Space:
        if ev.key() == QtCore.Qt.Key_Space:
            playVid = not playVid
            if playVid:
                playBtn.text('Pause')
            else:
                playBtn.text('Play')
                
        if ev.key() == QtCore.Qt.Key_C:
            print selectedROI
            selectedROI = 0
            print selectedROI
        if ev.key() == QtCore.Qt.Key_S:
            select_ROI=True
        if ev.key() == QtCore.Qt.Key_R:
            remove_ROI=True
            


    W.keyPressEvent = _keyPressEvent




    def _keyReleaseEvent(ev):
        
        global select_ROI, remove_ROI
        if ev.key() == QtCore.Qt.Key_S:
            select_ROI=False
        if ev.key() == QtCore.Qt.Key_R:
            remove_ROI=False
        

        
    W.keyReleaseEvent = _keyReleaseEvent

    #___________________________________________________________________________________________________________

    """ Define the Outline of the PlotWidget and 
        create the timeLine that moves with the video """



    Gplt = pg.PlotWidget(background='w')
    Gplt.setFixedHeight(200)
    Gplt.setXRange(0,nFrames)
    #Gplt.setFixedWidth(512)

    #___________________________________________________________________________________________________________


    """
        Play Video Play Function      """
    def update_video():
        global frame_idx, vb, img, playVid, meanImage
        
        if playVid:
            img.setImage(np.fliplr(np.mean(b[frame_idx-rolling_average:frame_idx+rolling_average+1,:,:],axis=0).T),
                         autoLevels=False)
            timeLine.setPos(frame_idx)
            if frame_idx>=nFrames-1:
                frame_idx=1
            frame_idx += 1
        if meanImage:
            img.setImage(np.fliplr(meanIm.T),autolevels=False)
        else:
            img.setImage(np.fliplr(np.mean(b[frame_idx-rolling_average:frame_idx+rolling_average+1,:,:],axis=0).T),
                         autoLevels=False)
        
                
                

        
    IFI=30
    t2 = QtCore.QTimer()
    t2.timeout.connect(update_video)
    t2.start(IFI)

    ###___________________________________________________________________________________________________________________

    """ This section contains the code to
        create and upgrade the histogram
        used to control the image """
    histLI = pg.HistogramLUTWidget(image=img,fillHistogram=False)
    histLI.autoHistogramRange=False



    ###___________________________________________________________________________________________________________________

    """
        functions to update the timeLine 
                                        """
    timeLine = pg.InfiniteLine(pos=frame_idx,angle=90,movable=True)

    def time_line_changed():
        global frame_idx,timeLine, playVid
        wasPlaying = playVid
        while timeLine.isUnderMouse():
            playVid=False
            frame_idx = int(timeLine.getXPos())
            timeLine.setPos(frame_idx)
            img.setImage(np.fliplr(np.mean(b[frame_idx-rolling_average:frame_idx+rolling_average+1,:,:],axis=0).T),autoLevels=False)


        playVid = wasPlaying

    timeLine.sigDragged.connect(time_line_changed)
    Gplt.addItem(timeLine)


    ###___________________________________________________________________________________________________________________

    """ In this segment, update the ROI """




    #Overlay the first iamge
    img2 = pg.ImageItem()
    img2.setImage(mask)
    img2.setZValue(10)  # make sure this image is on top
    img2.setOpacity(0)  #Set opacity to one
    vb.addItem(img2)
    #img2.setLookupTable(pg.HistogramLUTItem(image=img2))

    ###___________________________________________________________________________________________________________________

    """ 
        These Buttons Control Playback of the Video
        accBtn speeds up playback by a factor of two
        decBtn slows playback by a factor of two
        playBtn plays and pauses the video  """


    accBtn = QtGui.QPushButton(r'>>')
    def double_speed(accBtn):
        global t2, IFI,tx
        if IFI>5:
            IFI = IFI/2
            t2.start(IFI)
            tx.setText(str(np.round(1000/IFI)) + 'fps')
    accBtn.clicked.connect(double_speed)


    decBtn = QtGui.QPushButton(r'<<')
    def half_speed(accBtn):
        global t2, IFI, tx
        IFI = IFI*2
        t2.start(IFI)
        tx.setText(str(np.round(1000/IFI)) + 'fps')
    decBtn.clicked.connect(half_speed)


    playBtn = QtGui.QPushButton('Pause')
    def play_vid():
        global playVid,playBtnStr,playBtn
        if playVid:
            playVid=False
            playBtnStr = 'Play'
        elif playVid==False:
            playVid=True
            playBtnStr = 'Pause'
        playBtn.setText(playBtnStr)
    playBtn.clicked.connect(play_vid)


    pROIBtn = QtGui.QPushButton('Start Pixel ROI Mode')
    def select_pixel_ROIs():
        
        global select_centre_pixel, pROIBtn 
        select_centre_pixel = not select_centre_pixel
        if select_centre_pixel:
            pROIBtn.text('Stop Pixel ROI Mode')
        elif select_centre_pixel==False:
            pROIBtn.text('Start Pixel ROI Mode')

    pROIBtn.clicked.connect(select_pixel_ROIs)




    meanImBtn = QtGui.QPushButton('Show Mean Image')
    def show_mean_image():
        global meanImage, playVid, meanImBtn
        meanImage = not meanImage
        if meanImage:
            meanImBtn.text('Show Movie')
            playVid = False
        elif not meanImage:
            meanImBtn.text('Show Mean Image')

    meanImBtn.clicked.connect(show_mean_image)


    incRollAvgBtn = QtGui.QPushButton('Increase Rolling Average')
    def incRollAvg():
        global rolling_average
        rolling_average += 1
        print rolling_average
        tx2.setText('rolling_average: %s' %(1 + rolling_average*2))
    incRollAvgBtn.clicked.connect(incRollAvg)

    decRollAvgBtn = QtGui.QPushButton('Decrease Rolling Average')
    def decRollAvg():
        global rolling_average
        print rolling_average
        if rolling_average>0:
            rolling_average -= 1
        tx2.setText('rolling_average: %s' %(1 + rolling_average*2))
    decRollAvgBtn.clicked.connect(decRollAvg)

    ###_______________________________________________________________________________________________________________

    """ 
        Control the Layout of the GUI
                                        """

    layout.addWidget(grV1,0,0,7,8)

    layout.addWidget(histLI,0,8,7,2)


    layout.addWidget(Gplt,10,0,3,8)
    layout.addWidget(decBtn,9,0,1,1)
    layout.addWidget(playBtn,9,1,1,1)
    layout.addWidget(accBtn,9,2,1,1)
    layout.addWidget(accBtn,9,2,1,1)




    layout.addWidget(roiBtn,9,8)
    layout.addWidget(maskBtn,9,9)
    layout.addWidget(pROIBtn,10,8)
    layout.addWidget(meanImBtn,10,9)
    layout.addWidget(incRollAvgBtn,11,9)
    layout.addWidget(decRollAvgBtn,11,8)



    win.setCentralWidget(W)

    #win.resize(1200,800)
    win.show()