#!usr/bin/env python
import sys
sys.path.append('/../../Classes/Tools')
from Geometry_Basepy import Geometry_Base
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
import skimage
try:
    import h5py
    h5_available = True
except:
    print("HDF5 bindings not found")
    h5_available = False

## \brief Multi-D Voxel geometry class
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
# \author Emily Vu, evu@sandia.gov, emilyhvu@umich.edu
# \author Dan Bolintineanu, dsbolin@sandia.gov
#
# Class for multi-D Voxel geometry 
#
class Geometry_Voxel(Geometry_Base):
    def __init__(self):
        super(Geometry_Voxel,self).__init__()
        self.flshowplot = False; self.flsaveplot = False
        self.GeomType = 'Voxel'
    
    def __str__(self):
        return str(self.__dict__)

    ## \brief Lets user define voxelization mixing parameters
    #
    # \param[in] flVoxelize, bool, whether to voxelize derivative geometry type when sample initialized
    # \param[in] numVoxels, list of ints, number of voxels to sample size of in x, y, and z directions
    # \returns creates self.flVoxelize and self.numVoxels
    def defineVoxelizationParams(self,flVoxelize=False,numVoxels=None,chunksize=None):
        assert isinstance(flVoxelize,bool)
        if flVoxelize:
            assert isinstance(numVoxels,list) and len(numVoxels)==3
            for i in numVoxels:
                assert isinstance(i,int) and i>0
        self.flVoxelize      = flVoxelize
        self.numVoxels       = numVoxels[:]    
        self.chunksize       = chunksize
        self.voxelSizeX      = (self.xbounds[1]-self.xbounds[0])/self.numVoxels[0]
        self.voxelSizeY      = (self.ybounds[1]-self.ybounds[0])/self.numVoxels[1]
        self.voxelSizeZ      = (self.zbounds[1]-self.zbounds[0])/self.numVoxels[2]

    ## \brief Initialize geometry parameters at start of each history
    # \returns nothing
    def _initializeHistoryGeometryMemory(self):
        pass

    ## \brief (Optionally) reads in voxel geometry from numpy files and (Always) selects Voxel version of samplePoint
    #
    # \returns initializes self.VoxelMatInds, self.xBoundaries, self.yBoundaries, self.zBoundaries, and sets self.samplePoint
    def _initializeSampleGeometryMemory(self):
        self.samplePoint = self.samplePoint_Voxel
        #later: read in file based on numerical progression and user-provided prefix

    ## \brief Finds material type at current location
    #
    # \returns sets self.CurrentMatInd
    def samplePoint_Voxel(self):
        #find indices of current voxel
        xIndex = int( np.floor( (self.Part.x + self.xbounds[0]) / (self.xbounds[1]-self.xbounds[0]) * self.numVoxels[0] ) )
        yIndex = int( np.floor( (self.Part.y + self.ybounds[0]) / (self.ybounds[1]-self.ybounds[0]) * self.numVoxels[1] ) )
        zIndex = int( np.floor( (self.Part.z + self.zbounds[0]) / (self.zbounds[1]-self.zbounds[0]) * self.numVoxels[2] ) )
        #set current material type
        self.CurrentMatInd = int(self.VoxelMatInds[xIndex][yIndex][zIndex])
 

    ## \brief Creates Voxel geometry by sampling from Other geometry type
    #
    # \param[in] NumRowsUpdateAt, int default 10000000 (not update), number of rows to update time estimates after
    # \returns creates self.VoxelMatInds as a voxel description of geometry and sets self.samplePoint
    def voxelizeGeometrySlow(self, NumRowsUpdateAt=10000000):
        assert isinstance(NumRowsUpdateAt,int) and NumRowsUpdateAt>0
        tstart = time.time()
        self.VoxelMatInds = np.zeros( (self.numVoxels[0], self.numVoxels[1], self.numVoxels[2]) )        
        xvals = np.linspace( self.xbounds[0] + self.voxelSizeX/2 , self.xbounds[1] - self.voxelSizeX/2, self.numVoxels[0])
        yvals = np.linspace( self.ybounds[0] + self.voxelSizeY/2 , self.ybounds[1] - self.voxelSizeY/2, self.numVoxels[1])
        zvals = np.linspace( self.zbounds[0] + self.voxelSizeZ/2 , self.zbounds[1] - self.voxelSizeZ/2, self.numVoxels[2])
        for i,ix in enumerate(xvals):
            self.Part.x = ix
            for j,jy in enumerate(yvals):
                self.Part.y = jy
                for k,kz in enumerate(zvals):
                    self.Part.z = kz
                    self.samplePoint()
                    self.VoxelMatInds[i,j,k] = self.CurrentMatInd
            if (i+1)%NumRowsUpdateAt==0:
                runtime = (time.time()-tstart)/60.0
                print("{}/{} slices rendered, eta: {:5.2f}/{:5.2f} min".format(i+1,len(xvals),runtime,runtime*self.numVoxels[0]/(i+1)))
        self.matInds = np.unique(self.VoxelMatInds)
        self.samplePoint = self.samplePoint_Voxel

    ## \brief Creates Voxel geometry by sampling from Other geometry type; vectorized
    #
    # \param[in] numVoxels, list of ints, number of voxels in each dimension
    # \param[in] chunksize, int, number of voxels to process at one time. Larger numbers require more memory, but result in faster runtime.
    #           By default, this is set to use approximately 2GB of memory.
    # \returns creates self.VoxelMatInds, voxel description of geometry
    def voxelizeGeometry(self):#, numVoxels=None, chunksize=None):
        tstart = time.time()
        self.VoxelMatInds = np.zeros( (self.numVoxels[0], self.numVoxels[1], self.numVoxels[2]) )
        self.voxelSizeX = ( self.xbounds[1] - self.xbounds[0] ) / self.numVoxels[0]
        self.voxelSizeY = ( self.ybounds[1] - self.ybounds[0] ) / self.numVoxels[1]
        self.voxelSizeZ = ( self.zbounds[1] - self.zbounds[0] ) / self.numVoxels[2]
        xvals = np.linspace( self.xbounds[0] + self.voxelSizeX/2 , self.xbounds[1] - self.voxelSizeX/2, self.numVoxels[0])
        yvals = np.linspace( self.ybounds[0] + self.voxelSizeY/2 , self.ybounds[1] - self.voxelSizeY/2, self.numVoxels[1])
        zvals = np.linspace( self.zbounds[0] + self.voxelSizeZ/2 , self.zbounds[1] - self.voxelSizeZ/2, self.numVoxels[2])
        xg, yg, zg = np.meshgrid(xvals, yvals, zvals, indexing='ij')
        all_points = np.vstack([xg.ravel().T, yg.ravel().T, zg.ravel().T]).T

        #Estimate chunksize based on assumption of using ~2GB of memory for the array
        #However, this does not ensure that no memory errors will occur, since the actual voxel array still
        #needs to be stored
        if self.chunksize is None:
            if self.GeomType == "BoxPoisson" or self.GeomType == "Voronoi":
                self.chunksize = np.min([int((2*1024**3)), len(all_points)*8])
            elif self.GeomType == "Markovian":
                self.chunksize = np.min([int(2*1024**3/(self.numPlanes*8)), len(all_points)])
            elif self.GeomType == "SphericalInclusions":
                self.chunksize = np.min([int(2*1024**3/(len(self.Centers)*8)), len(all_points)])
        
        VoxelMatInds = []
        nchunk = int(len(all_points)/self.chunksize)
        if len(all_points) % self.chunksize > 0:
            nchunk += 1 
        for i in np.arange(nchunk): 
            print("Processing batch ",i+1," of ",nchunk)              
            endpoint = np.min(((i+1)*self.chunksize, len(all_points)))
            VoxelMatInds.append(self.samplePointsFast(all_points[i*self.chunksize:endpoint,:]))

        self.VoxelMatInds = np.hstack([m for m in VoxelMatInds]).reshape(self.numVoxels)
        self.matInds = np.unique(self.VoxelMatInds)
        print('created voxel geometry, took {:5.2f} min'.format((time.time()-tstart)/60.0))    

    ## \brief Writes voxel data from numpy array to HDF5 format
    #
    # \param[in] filename, str, name of file to write data to (include .h5 extension)
    # \returns stores numpy data in .h5 file
    def writeVoxelMatIndsToH5(self, filename ):
        if not h5_available:
            print("HDF5 not available, data not saved")
            return
        assert isinstance(filename,str) and filename[-3:]=='.h5'
        tstart = time.time()        
        with h5py.File(f"{filename}", "w") as f:
            f.create_dataset("VoxelMatInds", data=self.VoxelMatInds.astype(np.uint8))
            f.attrs["voxelSizeX"] = self.voxelSizeX
            f.attrs["voxelSizeY"] = self.voxelSizeY
            f.attrs["voxelSizeZ"] = self.voxelSizeZ
            f.create_dataset("matInds", data=self.matInds.astype(np.uint8))
        print('wrote to '+filename+', took {:5.2f} min'.format((time.time()-tstart)/60.0))

    ## \brief Reads voxel data from HDF5 format to numpy array 
    #
    # \param[in] filename, str, name of file to read data from 
    # \returns stores numpy data as self.VoxelMatInds and number of voxels in each direction as self.numVoxels
    def readVoxelMatIndsFromH5(self, filename ):
        if not h5_available:
            print("HDF5 not available, data not read")
            return
        assert isinstance(filename,str) and filename[-3:]=='.h5'
        tstart = time.time()
        with h5py.File(f"{filename}", "r") as f:
            self.VoxelMatInds = f["VoxelMatInds"][:]
            self.voxelSizeX = f.attrs["voxelSizeX"]
            self.voxelSizeY = f.attrs["voxelSizeY"]
            self.voxelSizeZ = f.attrs["voxelSizeZ"]
            self.matInds = f["matInds"][:]
        self.numVoxels = self.VoxelMatInds.shape
        print('read from '+filename+', took {:5.2f} min'.format((time.time()-tstart)/60.0))

    ## \brief Writes voxel data from numpy array to .vtk format
    #
    # \param[in] filename, str, name of file to write data to (include .vtk extension)
    # \param[in] : binary, bool, optional flag to indicate whether to use binary or ASCII file format.
    #    The binary format may have portability issues, but files are much faster to read/write and smaller. 
    #    Default is 'True'.
    # \param[in] : spacing, list of float, optional voxel size in x/y/z. Defaults to [1.0, 1.0, 1.0]
    # \returns stores numpy data in .vtk file
    def writeVoxelMatIndsToVtk(self, filename, binary=True):
        assert isinstance(filename,str) and filename[-4:]=='.vtk'
        tstart = time.time()
        dx = self.voxelSizeX
        dy = self.voxelSizeY
        dz = self.voxelSizeZ

        arr = self.VoxelMatInds
        try   : nz = arr.shape[2]
        except: nz = 1 #To handle 2D case
                
        with open(filename,"wb") as f:        
            f.write(b"# vtk DataFile Version 2.0\n")            
            f.write(b"Time step 0\n")    
            if binary: f.write(b"BINARY\n")
            else     : f.write(b"ASCII\n")
            f.write(b"DATASET STRUCTURED_POINTS\n")
            f.write(bytes("DIMENSIONS "+str(arr.shape[0])+" "+str(arr.shape[1])+" "+str(nz)+"\n", 'utf-8'))
            f.write(bytes("SPACING "+str(dx)+" "+str(dy)+" "+str(dz)+"\n","utf-8"))
            f.write(b"ORIGIN 0 0 0\n")
            f.write(bytes("POINT_DATA "+str(arr.size)+"\n", 'utf-8'))
            f.write(b"SCALARS "+"VoxelMatInds".encode()+b" float\n")
            f.write(b"LOOKUP_TABLE default\n")
            flat = arr.flatten(order='F').astype(np.float32)
            if binary:
                if sys.byteorder=='little': flat.byteswap(True).tofile(f)
                else                      : flat.tofile(f)
            else:
                np.savetxt(f, flat)
            f.write(b"\n")
        print('wrote to '+filename+', took {:5.2f} min'.format((time.time()-tstart)/60.0))


    ## \brief Returns voxel realization in numpy array format
    #
    # \returns material types in array format
    def returnRealization(self):
        return self.VoxelMatInds

    ## \brief Creates plots of 2D slices of voxel data
    #
    # \param[in] slicedim, str, 'x','y', or 'z', dimension of slice to render
    # \param[in] sliceindices, list of ints, indices of 2D slices to plot
    # \param[in] flgray, bool, whether plot indices should be grayscale (or color)
    # \param[in] flshow, bool, whether to show plot(s) to user when calling this function
    # \param[in] flsave, bool, whether to save plot(s) to file
    # \param[in] imageNamePrefix, str, prefix of file name to save images to
    # \returns stores numpy data as self.VoxelMatInds and number of voxels in each direction as self.numVoxels
    def plotVoxelMatInds(self, slicedim=None, sliceindices=[0], flgray=True, flshow=True, flsave=False, imageNamePrefix=None ):
        assert slicedim in {'x','y','z'}
        assert isinstance(sliceindices,list)
        for i in sliceindices: assert isinstance(i,int) and i>=0
        assert isinstance(flgray,bool)
        assert isinstance(flsave,bool)
        if flsave: assert isinstance(imageNamePrefix,str)

        for sliceind in sliceindices:
            if slicedim=='x':
                if flgray: plt.imshow(self.VoxelMatInds[sliceind,:,:], cmap='gray', interpolation='none')
                else     : plt.imshow(self.VoxelMatInds[sliceind,:,:], interpolation='none')
                plt.xlabel('Z')
                plt.ylabel('Y')
            if slicedim=='y':
                if flgray: plt.imshow(self.VoxelMatInds[:,sliceind,:], cmap='gray', interpolation='none')
                else     : plt.imshow(self.VoxelMatInds[:,sliceind,:], interpolation='none')
                plt.xlabel('Z')
                plt.ylabel('X')
            if slicedim=='z':
                if flgray: plt.imshow(self.VoxelMatInds[:,:,sliceind], cmap='gray', interpolation='none')
                else     : plt.imshow(self.VoxelMatInds[:,:,sliceind], interpolation='none')
                plt.xlabel('Y')
                plt.ylabel('X')
    
            plt.colorbar() # add colored legend for data
            plt.title(slicedim.upper()+' = '+str(sliceind), fontdict = {'fontsize' : 28})
    
            if flsave: plt.savefig(imageNamePrefix+'_slicein_'+slicedim+'_sliceindex_'+str(sliceind)+'.pdf')
            if flshow: plt.show()
            plt.close()