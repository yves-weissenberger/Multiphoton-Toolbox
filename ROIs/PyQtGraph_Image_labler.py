#!/home/yves/anaconda2/bin/python
import h5py
import sys


def MASK_DRAWER_GUI(areaFile,restart=False,online_trace_extract=0):
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

    sys.path.append('/home/yves/Documents/Code/misc/')
    import cell_magic_wand




    class Visualizator(QtGui.QMainWindow):

        def __init__(self, areaFile,online_trace_extract,restart=False):



            QtGui.QMainWindow.__init__(self)
            self.Folder = os.path.split(os.path.abspath(areaFile.file.filename))[0]
            #print "\n %s \n" %os.path.split(self.Folder)[0]
            self.idx = 0
            self.roi_idx = 0 
            self.nROIs = 0
            self.ROI_patches = []
            self.masks = []
            #print self.masks.shape, ROI_masks.shape
            #self.masks[:,:,:ROI_masks.shape[2]] = ROI_masks
            #print "file is %s MB \n" %1e-6 * areaFile[0].itemsize * np.product(areaFile.shape)
            print 'loading data into RAM.... \n'
            #self.video = np.zeros(areaFile.shape,dtype='uint16')
            #self.video = np.asarray(areaFile.astpye('uint16'),dtype='uint16')
            print "done!\n"

            if 'mean_image' in areaFile.attrs.iterkeys():
                self.mean_image = areaFile.attrs['mean_image'].T
            else:
                print 'No mean Image provided, computing....'
                self.mean_image = np.mean(areaFile[:3000],axis=0).T


            self.mean_image /= np.max(self.mean_image)
            self.mean_image = exposure.equalize_adapthist(self.mean_image,clip_limit=.01)


            self.ROI_attrs = {'centres':[],
                              'patches':[],
                              'masks':[],
                              'idxs':[],
                              'traces':[]}
            self.play_video = False
            self.rolling_average = 3
            self.ROI_centres = []#areaFile.attrs['ROI_centres']
            self.video = areaFile
            self.nFrames = self.video.shape[0]
            self.frame_idx = self.rolling_average
            self.smoothing = .1
            self.maskalpha = 0.6
            self.mask = np.zeros([areaFile.shape[1],
                                 areaFile.shape[2],
                                 4])

            self.temp_mask = np.zeros([areaFile.shape[1],
                                 areaFile.shape[2]])

            self.show_mean_image = True
            #self.masks = np.zeros(ROI_patches.shape)
            #initialise the main window
            w = QtGui.QWidget()
            layout = QtGui.QGridLayout()
            w.setLayout(layout)
            self.prevT = time.time()



            self.img = pg.ImageItem(setAutoDownsample=False)
            self.mask_img = pg.ImageItem()
            print areaFile.shape
            self.img.setImage(self.mean_image)
            self.mask_img.setImage(self.mask)
            self.mask_img.setOpacity(self.maskalpha)

            self.frameTxt = pg.TextItem('Frame Nr: ' + str(self.frame_idx+1) + '/' + str(self.nFrames))
            
            """ This section contains the code to
                create and upgrade the histogram
                used to control the image """
            self.histLI = pg.HistogramLUTWidget(image=self.img,fillHistogram=False)
            self.histLI.autoHistogramRange=False
            
            self.roi_item = pg.EllipseROI([60, 10], [30, 20], pen=(3,9))
            self.vb = pg.ViewBox()
            self.vb.setAspectLocked(1)
            self.vb.addItem(self.img)
            self.vb.addItem(self.roi_item)
            self.vb.addItem(self.mask_img)
            #self.vb.addItem(self.img_ROI)      
            #self.vb.addItem(self.tx)
            self.vb.addItem(self.frameTxt)
            self.vb.setFixedWidth(512)
            grV1 = pg.GraphicsView(useOpenGL=False)
            grV1.setCentralItem(self.vb)
            #self.vb.scene().sigMouseMoved.connect(self.mouseMoved)
            #self.img.sigClicked.connect(self.test1)
            #self.sigClicked.connect(self.test1)
            self.vb.scene().sigMouseClicked.connect(self.onClick)          #THIS IS IT!

            self.Gplt = pg.PlotWidget(background='w')
            self.Gplt.setFixedHeight(200)
            self.Gplt.setXRange(0,self.nFrames)
            grV1.setFixedWidth(512)


            ############## INIT ROI IMAGES #######################
            cols,rows = areaFile.shape[1:3]
            m = np.mgrid[:cols,:rows]
            self.possx = m[0,:,:]# make the x pos array
            self.possy = m[1,:,:]# make the y pos array


            ############## INIT BUTTONS #######################
            btn1 = QtGui.QPushButton("Next ROI", self)
            btn2 = QtGui.QPushButton("Previous ROI", self)
            btn3 = QtGui.QPushButton("Save Progress", self)
            btn4 = QtGui.QPushButton("Clear ROI", self)
            btn5 = QtGui.QPushButton("Play Video", self)
            btn6 = QtGui.QPushButton("Extract ROI Traces",self)

            btn11 = QtGui.QPushButton("Show Mean Image", self)
            btn12 = QtGui.QPushButton("Hide Mask", self)


            btn1.setFixedWidth(110)
            btn1.setStyleSheet("font-size:11px;")
            btn2.setFixedWidth(110)
            btn2.setStyleSheet("font-size:11px;")
            btn3.setFixedWidth(110)
            btn3.setStyleSheet("font-size:11px;")
            btn4.setFixedWidth(110)
            btn4.setStyleSheet("font-size:11px;")
            btn5.setFixedWidth(110)
            btn5.setStyleSheet("font-size:11px;")
            btn6.setFixedWidth(110)
            btn6.setStyleSheet("font-size:11px;")
            btn11.setFixedWidth(110)
            btn11.setStyleSheet("font-size:11px;")
            btn12.setFixedWidth(110)
            btn12.setStyleSheet("font-size:11px;")


            btn1.clicked.connect(self.buttonClicked)            
            btn2.clicked.connect(self.buttonClicked)
            btn3.clicked.connect(self.save_ROIS)
            btn4.clicked.connect(self.buttonClicked)
            btn5.clicked.connect(self.buttonClicked)
            btn6.clicked.connect(self.buttonClicked)
            btn11.clicked.connect(self.buttonClicked)
            btn12.clicked.connect(self.buttonClicked)

            layout.addWidget(grV1,0,0,7,8)
            self.histLI.setFixedWidth(120)
            layout.addWidget(self.histLI,0,4,7,3)

            layout.addWidget(btn1,9,1,1,1)
            layout.addWidget(btn2,9,0,1,1)
            layout.addWidget(btn3,10,2,1,1)
            layout.addWidget(btn4,10,3,1,1)
            layout.addWidget(btn5,9,2,1,1)
            layout.addWidget(btn6,9,3,1,1)
            layout.addWidget(btn11,10,1,1,1)
            layout.addWidget(btn12,10,0,1,1)
            layout.addWidget(self.Gplt,11,0,3,8)

            self.setCentralWidget(w)
            if not restart:
                if 'ROI_dataLoc' in areaFile.attrs.iterkeys():
                    self._restore_prev_session()

            self.show()
            #self.connect(self, Qt.SIGNAL('triggered()'), self.closeEvent

        def save_ROIS(self):
            fName = areaFile.name[1:].replace('/','-') + '_ROIs.p'
            #print os.path.join(self.Folder,fName)
            FLOC = os.path.join(self.Folder,fName)
            with open(FLOC,'wb') as f:
                pickle.dump(self.ROI_attrs,f)

            areaFile.attrs['ROI_dataLoc'] = FLOC
            print 'ROI MASKS SAVED'

        def _restore_prev_session(self):

            """ restore  rois from the previous session"""
            areaFileLoc = os.path.split(os.path.abspath(areaFile.file.filename))[0]
            ROILoc = os.path.join(areaFileLoc,areaFile.attrs['ROI_dataLoc'])
            try:
                with open(ROILoc) as f:
                    dat = pickle.load(f)

                self.ROI_attrs = dat
                self.ROI_attrs['traces'] = list(dat['traces'])
                self.nROIs = len(dat['idxs'])
                for roiIdx in range(self.nROIs):
                    self.mask[self.ROI_attrs['idxs'][roiIdx][0],self.ROI_attrs['idxs'][roiIdx][1],0] = 1
                    self.mask[self.ROI_attrs['idxs'][roiIdx][0],self.ROI_attrs['idxs'][roiIdx][1],0] = 1
                    self.mask[self.ROI_attrs['idxs'][roiIdx][0],self.ROI_attrs['idxs'][roiIdx][1],3] = 1


                self.roi_idx = self.nROIs-1
                self.mask[self.ROI_attrs['idxs'][roiIdx][0],self.ROI_attrs['idxs'][roiIdx][1],0] = 0
                self.mask[self.ROI_attrs['idxs'][roiIdx][0],self.ROI_attrs['idxs'][roiIdx][1],1] = 1

                self.mask_img.setImage(self.mask,autoLevels=False,levels=[0,2])

                self.Gplt.plot(self.ROI_attrs['traces'][-1])
            except EOFError:
                print 'Warning old ROI file corrupted!'



        def onClick(self,ev):
            #print self.vb.mapSceneToView(ev.pos())
            if ev.button()==1 and ev.double():
                click_loc = np.array([self.vb.mapSceneToView(ev.pos()).x(),self.vb.mapSceneToView(ev.pos()).y()])
                self.proc_roi_region(add_region=True,c0=click_loc)
                self.mask_img.setImage(self.mask,autoLevels=False,levels=[0,2])
            elif ev.button()==2:
                # In this case, just select the roi

                click_loc = np.array([self.vb.mapSceneToView(ev.pos()).x(),self.vb.mapSceneToView(ev.pos()).y()])
                clickedROI = False
                roi_nr = None
                #print np.array(self.ROI_attrs['centres']).T
                val  = np.array(self.ROI_attrs['centres']).T - click_loc[:,None]
                #print val
                #print np.sum(np.abs(val),axis=0)
                #print "mindist %s" %np.min(np.sum(np.abs(val),axis=0))
                roi_nr = np.argmin(np.sum(np.abs(val),axis=0))

                #print roi_nr
                if np.min(np.sum(np.abs(val),axis=0))<10:
                    clickedROI=True
                print "prev_centre %s, click loc: %s" %(self.ROI_attrs['centres'][-2],click_loc)


                if clickedROI:
                    print 'clicked_ROI'
                    self.mask[:,:,1] = 0

                    self.mask[self.ROI_attrs['idxs'][self.roi_idx][0],self.ROI_attrs['idxs'][self.roi_idx][1],0] = 1


                    self.roi_idx = roi_nr
                    print self.roi_idx

                    try:
                        self.Gplt.plot(self.ROI_attrs['traces'][self.roi_idx])
                    except IndexError:
                        print 'Trace for roi: %s not extracted yet' %self.roi_idx 


                    self.mask[self.ROI_attrs['idxs'][self.roi_idx][0],self.ROI_attrs['idxs'][self.roi_idx][1],0] = 0
                    self.mask[self.ROI_attrs['idxs'][self.roi_idx][0],self.ROI_attrs['idxs'][self.roi_idx][1],1] = 1
                    self.mask_img.setImage(self.mask,autoLevels=False,levels=[0,2])
                    #self.proc_roi_region(add_region=False)
                    #self.mask_img.setImage(self.mask,autoLevels=False,levels=[0,2])
                else:
                    pass


        def proc_roi_region(self,add_region=True,c0=None):

            #click_loc = np.array([self.vb.mapSceneToView(ev.pos()).x(),self.vb.mapSceneToView(ev.pos()).y()])

            temp_mask = cell_magic_wand.cell_magic_wand(self.mean_image,np.flipud(c0),3,8,roughness=10)
            #mpossx = self.roi_item.getArrayRegion(self.possx,self.img).astype(int)
            #mpossx = mpossx[np.nonzero(binary_fill_holes(mpossx))]#get the x pos from ROI
            #mpossy = self.roi_item.getArrayRegion(self.possy,self.img).astype(int)
            #mpossy = mpossy[np.nonzero(binary_fill_holes(mpossy))]# get the y pos from ROI
            mpossx,mpossy = np.where(temp_mask)
            xLims = [np.min(mpossx)-10,np.max(mpossx)+10]
            yLims = [np.min(mpossy)-10,np.max(mpossy)+10]
            #xLims = [np.mean(mpossx)-20,np.mean(mpossx)+20]; yLims = [np.mean(mpossy)-20,np.mean(mpossy)+20]


            self.temp_mask = temp_mask
            #self.temp_mask = binary_fill_holes(self.temp_mask).T

            if add_region:
                self.ROI_attrs['centres'].append([np.mean(mpossx),np.mean(mpossy)])
                self.ROI_attrs['patches'].append(self.mean_image[yLims[0]:yLims[1],xLims[0]:xLims[1]])
                self.ROI_attrs['idxs'].append([np.where(temp_mask)])
                self.ROI_attrs['masks'].append(self.temp_mask[yLims[0]:yLims[1],xLims[0]:xLims[1]])
                if online_trace_extract:
                    temp = areaFile[:,yLims[0]:yLims[1],xLims[0]:xLims[1]] *self.ROI_attrs['masks'][-1]
                    temp = temp.astype('float64')
                    temp[temp==0] = np.nan
                    self.ROI_attrs['traces'].append(np.nanmean(temp,axis=(1,2)))
                    #self.ROI_attrs['mask_arr'].append(temp_mask)
                    self.Gplt.clear()
                    self.Gplt.plot(self.ROI_attrs['traces'][-1])
                else:
                    self.ROI_attrs['traces'].append([0])


                self.mask[:,:,0] += self.mask[:,:,1]
                self.mask[:,:,1] = 0
                self.mask[:,:,1] = self.temp_mask.T
                #self.mask[:,:,0] += self.temp_mask.T
                self.mask[:,:,3] += self.temp_mask.T
                self.nROIs += 1
                self.roi_idx = self.nROIs - 1

            else:
                self.mask[mpossx,mpossy,0] = 0
                self.mask[mpossx,mpossy,3] = 0
            self.temp_mask = np.zeros(self.temp_mask.shape) 



        def buttonClicked(self):
            sender = self.sender()
            if sender.text()=='Next ROI':
                if self.roi_idx<self.nROIs-1:
                    self.roi_idx += 1
                    print 'viewing roi: %s' %self.roi_idx

                    self.Gplt.clear()

                    try:
                        self.Gplt.plot(self.ROI_attrs['traces'][self.roi_idx])
                    except IndexError:
                        print 'Trace for roi: %s not extracted yet' %self.roi_idx 

                    self.mask[:,:,1] = 0
                    self.mask[self.ROI_attrs['idxs'][self.roi_idx-1][0],self.ROI_attrs['idxs'][self.roi_idx-1][1],0] = 1
                    self.mask[self.ROI_attrs['idxs'][self.roi_idx][0],self.ROI_attrs['idxs'][self.roi_idx][1],0] = 0

                    self.mask[self.ROI_attrs['idxs'][self.roi_idx][0],self.ROI_attrs['idxs'][self.roi_idx][1],1] = 1
                    self.mask_img.setImage(self.mask,autoLevels=False,levels=[0,2])
                    #self.mask[:,:,1] = self.temp_mask.T

            elif sender.text()=='Previous ROI':
                if self.roi_idx>=1:
                    self.roi_idx -= 1
                    print 'viewing roi: %s' %self.roi_idx

                    #self.frame_idx = self.rolling_average
                    self.Gplt.clear()
                    self.Gplt.plot(self.ROI_attrs['traces'][self.roi_idx])


                    self.mask[:,:,1] = 0
                    self.mask[self.ROI_attrs['idxs'][self.roi_idx+1][0],self.ROI_attrs['idxs'][self.roi_idx+1][1],0] = 1
                    self.mask[self.ROI_attrs['idxs'][self.roi_idx][0],self.ROI_attrs['idxs'][self.roi_idx][1],0] = 0

                    self.mask[self.ROI_attrs['idxs'][self.roi_idx][0],self.ROI_attrs['idxs'][self.roi_idx][1],1] = 1
                    self.mask_img.setImage(self.mask,autoLevels=False,levels=[0,2])


            elif sender.text()=='Clear ROI':
                self.mask[self.ROI_attrs['idxs'][self.roi_idx][0],self.ROI_attrs['idxs'][self.roi_idx][1],1] = 0
                self.mask[self.ROI_attrs['idxs'][self.roi_idx][0],self.ROI_attrs['idxs'][self.roi_idx][1],0] = 0
                self.mask[self.ROI_attrs['idxs'][self.roi_idx][0],self.ROI_attrs['idxs'][self.roi_idx][1],3] = 0



                for key in self.ROI_attrs.iterkeys():
                    del self.ROI_attrs[key][self.roi_idx]
                self.nROIs -= 1
                if self.roi_idx==self.nROIs:
                    self.roi_idx -= 1

                if self.roi_idx>0:
                    self.mask[self.ROI_attrs['idxs'][self.roi_idx][0],self.ROI_attrs['idxs'][self.roi_idx][1],0] = 0
                    self.mask[self.ROI_attrs['idxs'][self.roi_idx][0],self.ROI_attrs['idxs'][self.roi_idx][1],1] = 1
                
                self.mask_img.setImage(self.mask,autoLevels=False,levels=[0,2])
                self.Gplt.clear()



                
            elif sender.text()=='Hide Mask':
                if self.maskalpha==0:
                    self.maskalpha=0.6
                    self.mask_img.setOpacity(self.maskalpha)
                    #self.img_ROI.setOpacity(self.maskalpha)

                else:
                    self.maskalpha=0
                    self.mask_img.setOpacity(self.maskalpha)
                





    app = QtGui.QApplication([])
    win = Visualizator(areaFile,online_trace_extract,restart)
    print sys.exit(app.exec_())

    return app






if __name__=="__main__":


    if len(sys.argv)==1:
        raise ValueError('first argument needs to be absolute or relative path to HDF file')  #wrong error type but cba
    else:
        hdfPath = sys.argv[1]

    if len(sys.argv)>2:
        print sys.argv[2]
        online_trace_extract = bool(int(sys.argv[2]))
        
        print "%s extracting traces online" %('not' if online_trace_extract==False else '')
    else:
        online_trace_extract = 1
        print 'extracting traces online; may lead to performance issues. Set second argument to 0 to prevent'
    if len(sys.argv)>3:
        restart = sys.argv[3]
        print sys.argv[3]
        if restart:
            restart = int(raw_input('Are you sure you want to restart? All work on this area will be deleted (0/1): '))
    else:
        restart = 0
	with h5py.File(hdfPath, 'a', libver='latest') as HDF_File:
		try:
			print HDF_File.filename
			print 'Sessions:'

			sessions = list((i for i in HDF_File.iterkeys()))
			for idx,f in enumerate(sessions):
				print idx, f 
			session = int(raw_input('Select Session Nr: '))

			sessions = list((i for i in HDF_File.iterkeys()))

			dataType = 0
            

			if 'registered_data' in HDF_File[sessions[session]].iterkeys():
				print 'Using registered Data'
				dataType = 'registered_data'
			else:
				print '\n!!!!!!!!!!WARNING!!!!!!!!!!!!\nUsing Raw Data\n!!!!!!!!!!WARNING!!!!!!!!!!!!'
				dataType = 'raw_data'



			areas = list((i for i in HDF_File[sessions[int(session)]][dataType].iterkeys()))
			for idx,f in enumerate(areas):
				print idx, f 


			areaID = int(raw_input('Select Area Nr: '))
			areaFile = HDF_File[sessions[session]][dataType][areas[areaID]]; 
            #print '...into function'
			#areaFile.attrs['ROI_patches']


            

			app = MASK_DRAWER_GUI(areaFile,restart=0,online_trace_extract=online_trace_extract)
			#print sys.exit(app.exec_())
		except:
            #raise
            #print 'Something unexpected went wrong! :('
			raise

	print 'HDF_File Closed, PyQt Closed'
