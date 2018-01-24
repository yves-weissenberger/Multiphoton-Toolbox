#!/home/yves/anaconda2/bin/python
import h5py
import sys
import os
import copy as cp

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


def get_hdf_paths(n_basedirs,in_args):

    """ Returns list of all hdf paths"""
    hdfPaths = []
    for dir_ix in range(1,1+n_basedirs):
        base_dir = in_args[dir_ix]
        for root, dirs, files in os.walk(base_dir):
            for fl in files:
                if fl.endswith(".h5"):
                     # print(os.path.join(root, fl)) 
                     hdfPaths.append(os.path.join(root,fl))

    return hdfPaths

def MASK_DRAWER_GUI(areaFiles):
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

        def __init__(self, areaFile):



            QtGui.QMainWindow.__init__(self)

            self.Folder = os.path.split(os.path.abspath(areaFile.file.filename))[0]
            #if not os.path.isdir(os.path.join(self.Folder,'ROIs')):
            #    os.mkdir(os.path.join(self.Folder,'ROIs'))
            #print "\n %s \n" %os.path.split(self.Folder)[0]
            self.roi_idx = 0 

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


            #if 'max_image' in areaFile.attrs.iterkeys():
            #    self.mean_image = areaFile.attrs['max_image'].T
            #else:
            #    print 'No mean Image provided, computing....'
            #    self.mean_image = np.max(areaFile[:6000],axis=0).T

            #self.mean_image = self.mean_image + self.mean_image*sobel(self.mean_image)/50 #out
            self.mean_image /= np.max(self.mean_image)

            self.mean_image = exposure.equalize_adapthist(self.mean_image,clip_limit=.001)

            self.ROI_attrs = {'centres':[],
                              'patches':[],
                              'masks':[],
                              'idxs':[],
                              'traces':[]}

            self.show_mean_image = True
            #self.masks = np.zeros(ROI_patches.shape)
            #initialise the main window



            self.w = QtGui.QWidget()
            layout = QtGui.QGridLayout()
            self.w.setLayout(layout)
            #print '2'

            nCols = 4
            nRows = 2#np.ceil(len(areaFiles/float(nCols)))
            self.vbs = []
            self.grvs = []
            self.imgs = []
            self.masks = []
            self.click_funs = []
            self.frames = []
            #print '2.5'

            self.patch_size= [20,20]
            for i in range(8):

                self.imgs.append(pg.ImageItem(setAutoDownsample=True))
                self.masks.append(pg.ImageItem(setAutoDownsample=True))
                fr = np.dstack([np.zeros(self.patch_size),np.pad(np.zeros([self.patch_size[0]-2]*2),1,'constant',constant_values=1)]*2)
                self.frames.append(pg.ImageItem(setAutoDownsample=True))
                self.frames[-1].setImage(fr)
                #print '2.6'
                self.vbs.append(pg.ViewBox())

                self.vbs[-1].setAspectLocked(1)
                #print '2.7'
                #self.vbs[-1].addItem(self.imgs[-1])
                #self.vbs[-1].addItem(self.masks[-1])
                self.vbs[-1].addItem(self.frames[-1])
                self.grvs.append(pg.GraphicsView(useOpenGL=False))
                #print '2.8'
                self.grvs[-1].setCentralItem(self.vbs[-1])


                #current column and row
                cCol= int(np.remainder(i,float(nCols)))
                cRow = int(np.floor(i/float(nCols)))
                layout.addWidget(self.grvs[-1],2*cRow,2*cCol,2,2)

                self.click_funs.append(lambda event,x=i: self.onClick(event,x))

                self.vbs[-1].scene().sigMouseClicked.connect(self.click_funs[-1])          #THIS IS IT!




            self.setCentralWidget(self.w)

            self.show()
            #self.connect(self, Qt.SIGNAL('triggered()'), self.closeEvent

        def save_ROIS(self):
            fName = areaFile.name[1:].replace('/','-') + '_ROIs.p'
            #print os.path.join(self.Folder,fName)
            FLOC = os.path.join(self.Folder,'ROIs',fName)
            with open(FLOC,'wb') as f:
                pickle.dump(self.ROI_attrs,f)

            #areaFile.attrs['ROI_dataLoc'] = FLOC
            print 'ROI MASKS SAVED'

        def onClick(self,event,window):
            modifiers = QtGui.QApplication.keyboardModifiers()
            if modifiers == QtCore.Qt.ShiftModifier:
                fr = np.dstack([ np.pad(np.zeros([self.patch_size[0]-2]*2),1,'constant',constant_values=1),
                                np.zeros(self.patch_size),
                                np.zeros(self.patch_size),
                                np.pad(np.zeros([self.patch_size[0]-2]*2),1,'constant',constant_values=1)
                                ])

                self.frames[window].setImage(fr)
            else:

                fr = np.dstack([np.zeros(self.patch_size),np.pad(np.zeros([self.patch_size[0]-2]*2),1,'constant',constant_values=1)]*2)
                self.frames[window].setImage(fr)


        def keyPressEvent(self,ev):
            modifiers = QtGui.QApplication.keyboardModifiers()

            key = ev.key()
            if modifiers == QtCore.Qt.ShiftModifier:
                print self.roi_idx

                if key==16777235:
                    print 'Up'
                elif key==16777234:
                    print 'Left'
                elif key==16777236:
                    print 'Right'

                elif key==16777237:
                    print 'Down'

            else:
                if key==16777234:
                    print 'Previous ROI'
                elif key==16777236:
                    print 'Next ROI'






    app = QtGui.QApplication([])
    win = Visualizator(areaFile)
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
                if len(HDF_File[sessions[session]]['registered_data'])>0:
                    print 'Using registered Data'
                    dataType = 'registered_data'
                    #dataType = 'raw_data'
                else:
                    print '\n!!!!!!!!!!WARNING!!!!!!!!!!!!\nUsing Raw Data\n!!!!!!!!!!WARNING!!!!!!!!!!!!'
                    dataType = 'raw_data'

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

            fName = areaFile.name[1:].replace('/','-') + '_ROIs.p'
            Folder = os.path.split(os.path.abspath(areaFile.file.filename))[0]
            if not os.path.isdir(os.path.join(Folder,'ROIs')):
                os.mkdir(os.path.join(Folder,'ROIs'))

            #print os.path.join(self.Folder,fName)
            FLOC = os.path.join(Folder,'ROIs',fName)
            areaFile.attrs['ROI_dataLoc'] = FLOC

        except:
            #raise
            #print 'Something unexpected went wrong! :('
            raise
            
    with h5py.File(hdfPath, 'r', libver='latest') as HDF_File:
        #try:
        #    print 'here'
        areaFile = HDF_File[sessions[session]][dataType][areas[areaID]]; 
        app = MASK_DRAWER_GUI(areaFile)
            #print sys.exit(app.exec_())

        #except:
        #    raise
    

    print 'HDF_File Closed, PyQt Closed'
