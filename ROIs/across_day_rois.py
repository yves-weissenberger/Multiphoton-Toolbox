#!/home/yves/anaconda2/bin/python
import h5py
import sys
import os
import copy as cp
import pickle
import matplotlib.pyplot as plt


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


def get_roi_paths(n_basedirs,in_args):

    """ Returns list of all hdf paths"""
    roiPaths = []
    for dir_ix in range(1,1+n_basedirs):
        base_dir = in_args[dir_ix]
        for root, dirs, files in os.walk(base_dir):
            for fl in files:
                if fl.endswith("glob.p"):
                     # print(os.path.join(root, fl)) 
                     roiPaths.append(os.path.join(root,fl))

    return roiPaths

def MASK_DRAWER_GUI(roi_sets):
    import numpy as np
    from pyqtgraph.Qt import QtGui, QtCore
    from pyqtgraph import Qt
    import pyqtgraph as pg
    import sys, os, pickle, time
    import copy as cp
    from skimage.filters import gaussian as gaussian_filter
    from skimage import exposure
    from skimage.morphology import disk, dilation, erosion
    from skimage.filters.rank import median as median_filter
    from scipy.ndimage.morphology import binary_fill_holes
    from skimage.filters import scharr,sobel



    class Visualizator(QtGui.QMainWindow):

        def __init__(self, roi_sets):



            QtGui.QMainWindow.__init__(self)
            #sizePolicy = QtCore.QSizePolicy(QtCore.QSizePolicy.Preferred, QtCoreQSizePolicy.Preferred)
            #sizePolicy.setHeightForWidth(True)
            #self.setSizePolicy(sizePolicy)


            self.roi_idx = 0 
            self.roi_sets = cp.deepcopy(roi_sets)
            self.selected_window = None
            self.resize(1000,200)




            self.w = QtGui.QWidget()
            layout = QtGui.QGridLayout()
            self.w.setLayout(layout)
            #print '2'

            nCols = 7
            nRows = 1
            self.vbs = []
            self.grvs = []
            self.imgs = []
            self.masks = []
            self.click_funs = []
            self.frames = []
            #print '2.5'

            self.patch_size= [50,50]

            self.blueframe = np.dstack([np.zeros(self.patch_size),
                                    np.zeros(self.patch_size),
                                    np.pad(np.zeros([self.patch_size[0]-2]*2),1,'constant',constant_values=1),
                                    np.pad(np.zeros([self.patch_size[0]-2]*2),1,'constant',constant_values=1)
                                    ])

            self.redframe = np.dstack([ np.pad(np.zeros([self.patch_size[0]-2]*2),1,'constant',constant_values=1),
                                np.zeros(self.patch_size),
                                np.zeros(self.patch_size),
                                np.pad(np.zeros([self.patch_size[0]-2]*2),1,'constant',constant_values=1)
                                ])
            self.greenframe = np.dstack([np.zeros(self.patch_size),np.pad(np.zeros([self.patch_size[0]-2]*2),1,'constant',constant_values=1)]*2)

            self.drawOnDay = pg.TextItem('Drawn on day',color=[0,0,250])
            self.drawCopy = pg.TextItem('Copied',color=[250,0,0])

            self.drawnTexts = []
            for i in range(len(self.roi_sets)):

                nm = os.path.split(self.roi_sets[i][1])[1][:8]
                self.frameTxt = pg.TextItem(nm)
                self.frameTxt.setPos(0,55)



                self.imgs.append(pg.ImageItem(setAutoDownsample=0))
                self.masks.append(pg.ImageItem(setAutoDownsample=0))
                #print self.roi_sets[i][0]['isPresent'][self.roi_idx]
                #if self.roi_sets[i][0]['isPresent'][self.roi_idx]==0:
                #    fr = self.redframe
                #else:
                #    fr = self.greenframe
                self.frames.append(pg.ImageItem(setAutoDownsample=0))
                #self.frames[-1].setImage(fr)
                self.frameTxt.setParentItem(self.imgs[-1])

                #print '2.6'

                self.vbs.append(pg.ViewBox())


                self.vbs[-1].addItem(self.imgs[-1])
                self.vbs[-1].addItem(self.masks[-1])
                self.vbs[-1].addItem(self.frames[-1])


                self.grvs.append(pg.GraphicsView(useOpenGL=False))
                #print '2.8'
                self.grvs[-1].setCentralItem(self.vbs[-1])


                self.drawnTexts.append(pg.TextItem('Drawn on day',color=[0,0,250]))
                self.drawnTexts[-1].setParentItem(self.imgs[-1])
                self.drawnTexts[-1].setPos(0,50)



                #current column and row
                cCol= int(np.remainder(i,float(nCols)))
                cRow = int(np.floor(i/float(nCols)))
                layout.addWidget(self.grvs[-1],2*cRow,2*cCol,2,2)

                self.click_funs.append(lambda event,x=i: self.onClick(event,x))

                self.vbs[-1].scene().sigMouseClicked.connect(self.click_funs[-1])          #THIS IS IT!



            self.set_roi_images()
            self.setCentralWidget(self.w)

            self.show()
            #self.connect(self, Qt.SIGNAL('triggered()'), self.closeEvent




        def set_roi_images(self):
            for i,rois_ in enumerate(self.roi_sets):
                self.imgs[i].setImage(rois_[0]['patches'][self.roi_idx],autolevels=1)
                m_ = rois_[0]['masks'][self.roi_idx]
                m2_ = np.dstack([m_,np.zeros(m_.shape),np.zeros(m_.shape),m_])
                self.masks[i].setImage(m2_)
                self.masks[i].setOpacity(.2)
                if rois_[0]['isPresent'][self.roi_idx]==0:
                    fr = self.redframe
                else:
                    fr = self.greenframe
                self.frames[i].setImage(fr)
                if rois_[0]['drawn_onday'][self.roi_idx]:
                    self.drawnTexts[i].setText('Drawn On Day',color=[0,0,250])
                else:
                    self.drawnTexts[i].setText('Copied',color=[250,0,0])





        def onClick(self,event,window):
            if event.button()==1 and not event.double():
                modifiers = QtGui.QApplication.keyboardModifiers()
                if modifiers == QtCore.Qt.ShiftModifier:
                    self.roi_sets[window][0]['isPresent'][self.roi_idx] = 0
                    fr = self.redframe

                    self.frames[window].setImage(fr)
                else:
                    if window!=self.selected_window:
                        self.roi_sets[window][0]['isPresent'][self.roi_idx] = 1

                        fr = self.greenframe
                        self.frames[window].setImage(fr)
                    else:
                        pass
            elif event.button()==1 and event.double():
                for i_ in range(len(self.roi_sets)):

                    if  i_== window:
                        print 'hooo'
                        fr = self.blueframe
                        self.frames[i_].setImage(fr)
                    else:
                        if self.roi_sets[i_][0]['isPresent'][self.roi_idx]==0:
                            fr = self.redframe

                        else:
                            fr = self.greenframe
                        self.frames[i_].setImage(fr)


                    #self.frames[window].setImage(fr)
                    self.selected_window = window


        def update_mask(self):
            sz = 25
            rixs_ = self.roi_sets[self.selected_window][0]['idxs'][self.roi_idx]
            mask_big = np.zeros([512,512])


            mask_big[rixs_[1],rixs_[0]] = 1

            #plt.imshow(mask_big)
            #plt.show()
            #print len(self.roi_sets[self.selected_window][0]['centres'])
            centroid = self.roi_sets[self.selected_window][0]['centres'][self.roi_idx]
            mask = mask_big[centroid[1]-sz:centroid[1]+sz,centroid[0]-sz:centroid[0]+sz]
            #print sorted(rixs_[0])[:5]
            #print sorted(rixs_[1])[:5]
            #print sorted(np.where(mask_big)[0])[:5] 
            #print sorted(np.where(mask_big)[1])[:5]

            #print np.allclose(np.where(mask_big)[0][:5],rixs_[0][:5])


            #plt.figure()
            #plt.imshow(mask)
            #plt.figure()
            #plt.imshow(self.roi_sets[self.selected_window][0]['masks'][self.roi_idx])
            #plt.show()
            #print mask.shape
            self.roi_sets[self.selected_window][0]['masks'][self.roi_idx] = mask



        def update_mask_im(self):
            m_ = self.roi_sets[self.selected_window][0]['masks'][self.roi_idx]
            m2_ = np.dstack([m_,np.zeros(m_.shape),np.zeros(m_.shape),m_])
            self.masks[self.selected_window].setImage(m2_)
            #plt.imshow(m2_)
            #plt.show()
            #self.masks[self.selected_window].setOpacity(.2)

        def erode_mask(self):

            sz = 25
            rixs_ = self.roi_sets[self.selected_window][0]['idxs'][self.roi_idx]
            mask_big = np.zeros([512,512])

        def _save(self):
            print 'saving....'
            for rset in self.roi_sets:
                with open(rset[1],'wb') as f:
                    pickle.dump(rset[0],f)

            print 'saved!'




        def keyPressEvent(self,ev):
            modifiers = QtGui.QApplication.keyboardModifiers()

            key = ev.key()
            print key
            if modifiers == QtCore.Qt.ShiftModifier:
                if self.selected_window!=None:

                    if key==16777235:
                        print 'Up'
                        self.roi_sets[self.selected_window][0]['idxs'][self.roi_idx][0] += 1
                        self.update_mask()
                        self.update_mask_im()

                    elif key==16777234:
                        print 'Left'
                        self.roi_sets[self.selected_window][0]['idxs'][self.roi_idx][1] -= 1
                        self.update_mask()
                        self.update_mask_im()

                    elif key==16777236:
                        print 'Right'
                        self.roi_sets[self.selected_window][0]['idxs'][self.roi_idx][1] += 1
                        self.update_mask()
                        self.update_mask_im()

                    elif key==16777237:
                        print 'Down'
                        self.roi_sets[self.selected_window][0]['idxs'][self.roi_idx][0] -= 1
                        self.update_mask()
                        self.update_mask_im()

                    elif key==83:
                        print 'save'
                        self._save()

            else:
                if key==16777234:
                    self.roi_idx = np.clip(self.roi_idx-1,0,1e3).astype('int')
                    print 'Previous ROI: %s' %self.roi_idx

                    self.selected_window = None
                    self.set_roi_images()


                elif key==16777236:
                    self.roi_idx = np.clip(self.roi_idx+1,0,1e3).astype('int')
                    print 'Next ROI: %s' %self.roi_idx

                    self.selected_window = None
                    self.set_roi_images()









    app = QtGui.QApplication([])
    win = Visualizator(roi_sets)
    print sys.exit(app.exec_())

    return app

def sortkey(pth):
    return os.path.split(pth)[1][:8]

if __name__=="__main__":


    n_basedirs = len(sys.argv) - 1
    roiPaths = get_roi_paths(n_basedirs,sys.argv)
    roi_sets = []
    print 'loading ROI paths'

    for fpath in sorted(roiPaths,key=sortkey):
        with open(fpath) as f:
            roi_sets.append([pickle.load(f),fpath])



 
    app = MASK_DRAWER_GUI(roi_sets)
        #print sys.exit(app.exec_())

    #except:
    #    raise
    

    print 'HDF_File Closed, PyQt Closed'
