import h5py
import sys


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
            self.play_vid = False
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
            btn4 = QtGui.QPushButton("Clear ROI", self)
            btn1.clicked.connect(self.buttonClicked)            
            btn2.clicked.connect(self.buttonClicked)
            btn3.clicked.connect(self.save_ROIS)
            btn4.clicked.connect(self.buttonClicked)

            layout.addWidget(grV1,0,0,7,8)
            layout.addWidget(btn1,9,1,1,1)
            layout.addWidget(btn2,9,0,1,1)
            layout.addWidget(btn3,9,4,1,1)
            layout.addWidget(btn4,9,2,1,1)
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
            elif sender.text()=='Clear ROI':
            	self.masks[:,:,self.idx] = 0


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
    #app.exec_()
    print sys.exit(app.exec_())

    return app






if __name__=="__main__":
	hdfPath = sys.argv[1]
	with h5py.File(hdfPath, 'a', libver='latest') as HDF_File:
		try:
			print HDF_File.filename
			print 'Sessions:'

			sessions = list((i for i in HDF_File.iterkeys()))
			for idx,f in enumerate(sessions):
				print idx, f 
			session = int(raw_input('Select Session Nr:'))

			sessions = list((i for i in HDF_File.iterkeys()))
			#print sessions

			dataType = 0
			if 'registered_data' in HDF_File[sessions[session]].iterkeys():
				print 'Using registered Data'
				dataType = 'registered_data'
			else:
				print 'Using Raw Data'
				dataType = 'raw_data'



			areas = list((i for i in HDF_File[sessions[int(session)]][dataType].iterkeys()))
			for idx,f in enumerate(areas):
				print idx, f 

			areaID = int(raw_input('Select Area Nr:'))
			areaFile = HDF_File[sessions[session]][dataType][areas[areaID]]
			areaFile.attrs['ROI_patches']
			app = MASK_DRAWER_GUI(areaFile,restart=False)
			#print sys.exit(app.exec_())
		except:
			pass#draw_rois_manual(areFile)

	print 'HDF_File Closed, PyQt Closed'