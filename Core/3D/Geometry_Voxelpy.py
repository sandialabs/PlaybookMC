#!usr/bin/env python
import sys
sys.path.append('/../../Classes/Tools')
from RandomNumberspy import RandomNumbers
from Particlepy import Particle
from ClassToolspy import ClassTools
import numpy as np
import matplotlib.pyplot as plt
import time

## \brief Multi-D Voxel geometry class
# \author Emily Vu, evu@sandia.gov, emilyhvu@umich.edu
#
# Class for multi-D Voxel geometry 
#
class Geometry_Voxel(ClassTools):
    def __init__(self):
        super(Geometry_Voxel,self).__init__()
        self.flshowplot = False; self.flsaveplot = False
        self.GeomType = 'Voxel'
    
    def __str__(self):
        return str(self.__dict__)

    ## \brief Define outer boundaries of parallelapided geometry and solve volume
    #
    # \param[in] xbounds, ybounds, zbounds lists of two floats, boundaries of problem domain
    # \returns initializes self.xbounds, self.ybounds, self.zbounds, and calcs self.Volume
    def defineGeometryBoundaries(self,xbounds=None,ybounds=None,zbounds=None):
        assert isinstance(xbounds,list) and isinstance(ybounds,list) and isinstance(zbounds,list)
        assert len(xbounds)==2 and len(ybounds)==2 and len(zbounds)==2
        assert isinstance(xbounds[0],float) and isinstance(xbounds[1],float) and xbounds[0]<=xbounds[1]
        if self.Part.numDims<2: assert xbounds[0]==-np.inf and xbounds[1]==np.inf
        assert isinstance(ybounds[0],float) and isinstance(ybounds[1],float) and ybounds[0]<=ybounds[1]
        if self.Part.numDims<3: assert ybounds[0]==-np.inf and ybounds[1]==np.inf
        assert isinstance(zbounds[0],float) and isinstance(zbounds[1],float) and zbounds[0]<=zbounds[1]

        self.xbounds = xbounds
        self.ybounds = ybounds
        self.zbounds = zbounds

        self.Volume = self.zbounds[1]-zbounds[0]
        if self.Part.numDims==2: self.Volume *= (self.xbounds[1]-xbounds[0])
        if self.Part.numDims==3: self.Volume *= (self.ybounds[1]-ybounds[0])


    ## \brief Define boundary conditions
    #
    # \param[in] xBCs, yBCs, zBCs lists of two strings, 'vacuum' or 'reflective'
    # \returns initializes self.xBCs, self.yBCs, self.zBCs
    def defineBoundaryConditions(self,xBCs,yBCs,zBCs):
        assert isinstance(xBCs,list) and isinstance(yBCs,list) and isinstance(zBCs,list)
        assert len(xBCs)==2 and len(yBCs)==2 and len(zBCs)==2

        if self.Part.numDims<2: assert xBCs[0]==None and xBCs[1]==None
        else                  :
            assert xBCs[0]=='vacuum' or xBCs[0]=='reflective'
            assert xBCs[1]=='vacuum' or xBCs[1]=='reflective'

        if self.Part.numDims<3: assert yBCs[0]==None and yBCs[1]==None
        else                  :
            assert yBCs[0]=='vacuum' or yBCs[0]=='reflective'
            assert yBCs[1]=='vacuum' or yBCs[1]=='reflective'

        assert zBCs[0]=='vacuum' or zBCs[0]=='reflective'
        assert zBCs[1]=='vacuum' or zBCs[1]=='reflective'

        self.xBCs = xBCs
        self.yBCs = yBCs
        self.zBCs = zBCs
        

    ## \brief Defines Cross Sections
    #
    # \param[in] totxs, list of floats, list of total cross sections
    # \param scatxs, list of floats, list of scattering cross sections
    # \returns sets cross sections
    def defineCrossSections(self,totxs=None,scatxs=None):
        assert isinstance(totxs,list)
        self.nummats = len(totxs)
        for i in range(0,self.nummats):
            assert isinstance(totxs[i],float) and totxs[i]>=0.0
        self.totxs = totxs
        self.Majorant = max(totxs)
        if not scatxs==None:
            assert isinstance(scatxs,list) and len(scatxs)==self.nummats
            for i in range(0,self.nummats):
                assert isinstance(scatxs[i],float) and scatxs[i]>=0.0 and totxs[i]>=scatxs[i]
            self.scatxs = scatxs
            self.absxs = []
            self.scatrat = []
            for i in range(0,self.nummats):
                self.absxs.append( self.totxs[i]-self.scatxs[i] )
                self.scatrat.append( self.scatxs[i]/self.totxs[i] )

    
           
    ## \brief Associates random number object - must be instance of RandomNumbers class
    #
    # \param[in] Rng RandomNumbers object
    # \returns initializes self.Rng
    def associateRng(self,Rng):
        assert isinstance(Rng,RandomNumbers)
        self.Rng = Rng

    ## \brief Associates Particle object - must be instance of Particle class
    #
    # \param[in] Part Particle object
    # \returns initializes self.Part
    def associatePart(self,Part):
        assert isinstance(Part,Particle)
        self.Part = Part

        
    ## \brief samples random point within defined geometry 
    #
    # \returns concatonated Points and Materials vectors for each seed location 
    def _generateRandomPoint(self):
        x = self.Rng.uniform(self.xbounds[0],self.xbounds[1])
        y = self.Rng.uniform(self.ybounds[0],self.ybounds[1])
        z = self.Rng.uniform(self.zbounds[0],self.zbounds[1])        

        return [x,y,z]

    ## \brief prints Voxel realization
    #
    # \returns material types in array format
    def printRealization(self):
        return self.MatInds

        
    ## \brief Tests for a leakage or boundary reflection event, flags the first, evaluates the second
    #
    # \returns str 'reflect', 'transmit', 'sideleak', or 'noleak'
    def testForBoundaries(self):
        if self.isclose(self.Part.z,self.zbounds[0]):
            if   self.zBCs[0]==    'vacuum': return 'reflect'
            elif self.zBCs[0]=='reflective': self.Part.reflect_z()
        if self.isclose(self.zbounds[1],self.Part.z):
            if   self.zBCs[1]==    'vacuum': return 'transmit'
            elif self.zBCs[1]=='reflective': self.Part.reflect_z()

        if self.isclose(self.Part.x,self.xbounds[0]):
            if   self.xBCs[0]==    'vacuum': return 'sideleak'
            elif self.xBCs[0]=='reflective': self.Part.reflect_x()
        if self.isclose(self.xbounds[1],self.Part.x):
            if   self.xBCs[1]==    'vacuum': return 'sideleak'
            elif self.xBCs[1]=='reflective': self.Part.reflect_x()

        if self.isclose(self.Part.y,self.ybounds[0]):
            if   self.yBCs[0]==    'vacuum': return 'sideleak'
            elif self.yBCs[0]=='reflective': self.Part.reflect_y()
        if self.isclose(self.ybounds[1],self.Part.y):
            if   self.yBCs[1]==    'vacuum': return 'sideleak'
            elif self.yBCs[1]=='reflective': self.Part.reflect_y()

        return 'noleak'

    
    ## \brief Returns material type at location
    #
    # \returns material type of cell
    def returnMaterialType(self):
        x = int( np.floor( (self.Part.x + self.xbounds[0]) / (self.xbounds[1]-self.xbounds[0]) * self.numVoxels[0] ) )
        y = int( np.floor( (self.Part.y + self.ybounds[0]) / (self.ybounds[1]-self.ybounds[0]) * self.numVoxels[1] ) )
        z = int( np.floor( (self.Part.z + self.zbounds[0]) / (self.zbounds[1]-self.zbounds[0]) * self.numVoxels[2] ) )
        return int(self.MatInds[x][y][z])
 

    ## \brief Evaluates collision--chooses to accept or reject, then absorb or scatter
    #
    # \returns drives sampling of new points and returns str 'reject', 'absorb', or 'scatter'
    def evaluateCollision(self):
        cellMatType = self.returnMaterialType()
        if self.Rng.rand()>self.totxs[cellMatType]/self.Majorant:
            histRealization = self.printRealization()
            return 'reject' #accept or reject potential collision
        if self.Rng.rand()>self.scatrat[cellMatType]            :
            histRealization = self.printRealization() 
            return 'absorb'
        else: return 'scatter'
            
            
        return 'absorb' if self.Rng.rand()>self.scatrat[cellMatType] else 'scatter' #choose collision type

    ## \brief Returns distance to geometry boundary, only boundary for WMC-based approach
    #
    # \returns positive float, distance along streaming path to boundary
    def calculateDistanceToBoundary(self):
        if   self.isclose(self.Part.u,0.0): dbx = np.inf
        elif self.Part.u>0.0              : dbx = (self.xbounds[1]-self.Part.x)/self.Part.u
        else                              : dbx = (self.xbounds[0]-self.Part.x)/self.Part.u

        if   self.isclose(self.Part.v,0.0): dby = np.inf
        elif self.Part.v>0.0              : dby = (self.ybounds[1]-self.Part.y)/self.Part.v
        else                              : dby = (self.ybounds[0]-self.Part.y)/self.Part.v

        if   self.isclose(self.Part.w,0.0): dbz = np.inf
        elif self.Part.w>0.0              : dbz = (self.zbounds[1]-self.Part.z)/self.Part.w
        else                              : dbz = (self.zbounds[0]-self.Part.z)/self.Part.w

        return min(dbx,dby,dbz)




    ## \brief Creates Voxel geometry by sampling from Other geometry type
    #
    # \param[in] numVoxels, list of ints, number of voxels in each dimension
    # \param[in] GeomObj, PlaybookMC geometry object to sample points from
    # \param[in] timeUpdateFrequency, int, number of rows to update time estimates after
    # \returns creates self.MatInds, voxel description of geometry
    def createMatIndsFromOtherGeometry(self, numVoxels=None, GeomObj=None, timeUpdateFrequency=1):
        assert isinstance(numVoxels,list)
        for num in numVoxels: assert isinstance(num,int) and num>0
        assert isinstance(timeUpdateFrequency,int)
        self.numVoxels = numVoxels[:]
        tstart = time.time()
        self.MatInds = np.zeros( (self.numVoxels[0], self.numVoxels[1], self.numVoxels[2]) )
        xVoxelWidth = ( GeomObj.xbounds[1] - GeomObj.xbounds[0] ) / self.numVoxels[0]
        yVoxelWidth = ( GeomObj.ybounds[1] - GeomObj.ybounds[0] ) / self.numVoxels[1]
        zVoxelWidth = ( GeomObj.zbounds[1] - GeomObj.zbounds[0] ) / self.numVoxels[2]
        xvals = np.linspace( GeomObj.xbounds[0] + xVoxelWidth/2 , GeomObj.xbounds[1] - xVoxelWidth/2, self.numVoxels[0])
        yvals = np.linspace( GeomObj.ybounds[0] + yVoxelWidth/2 , GeomObj.ybounds[1] - yVoxelWidth/2, self.numVoxels[1])
        zvals = np.linspace( GeomObj.zbounds[0] + zVoxelWidth/2 , GeomObj.zbounds[1] - zVoxelWidth/2, self.numVoxels[2])
        for i,ix in enumerate(xvals):
            GeomObj.Part.x = ix
            for j,jy in enumerate(yvals):
                GeomObj.Part.y = jy
                for k,kz in enumerate(zvals):
                    GeomObj.Part.z = kz
                    GeomObj.samplePoint()
                    imat = GeomObj.CurrentMatInd
                    self.MatInds[i,j,k] = imat
            if i==0 or (i+1)%timeUpdateFrequency==0:
                runtime = (time.time()-tstart)/60.0
                print("{}/{} slices rendered, eta: {:5.2f}/{:5.2f} min".format(i+1,len(xvals),runtime,runtime*numVoxels[0]/(i+1)))
        print('created voxel geometry, took {:5.2f} min'.format((time.time()-tstart)/60.0))

    ## \brief Creates Voxel geometry by sampling from Other geometry type; vectorized
    #
    # \param[in] numVoxels, list of ints, number of voxels in each dimension
    # \param[in] GeomObj, PlaybookMC geometry object to sample points from
    # \param[in] chunksize, int, number of voxels to process at one time (larger numbers require more memory, but result in faster runtime)
    # \returns creates self.MatInds, voxel description of geometry
    def createMatIndsFromOtherGeometryFaster(self, numVoxels=None, GeomObj=None, chunksize=None):
        assert isinstance(numVoxels,list)
        for num in numVoxels: assert isinstance(num,int) and num>0
        self.numVoxels = numVoxels[:]
        tstart = time.time()
        self.MatInds = np.zeros( (self.numVoxels[0], self.numVoxels[1], self.numVoxels[2]) )
        xVoxelWidth = ( GeomObj.xbounds[1] - GeomObj.xbounds[0] ) / self.numVoxels[0]
        yVoxelWidth = ( GeomObj.ybounds[1] - GeomObj.ybounds[0] ) / self.numVoxels[1]
        zVoxelWidth = ( GeomObj.zbounds[1] - GeomObj.zbounds[0] ) / self.numVoxels[2]
        xvals = np.linspace( GeomObj.xbounds[0] + xVoxelWidth/2 , GeomObj.xbounds[1] - xVoxelWidth/2, self.numVoxels[0])
        yvals = np.linspace( GeomObj.ybounds[0] + yVoxelWidth/2 , GeomObj.ybounds[1] - yVoxelWidth/2, self.numVoxels[1])
        zvals = np.linspace( GeomObj.zbounds[0] + zVoxelWidth/2 , GeomObj.zbounds[1] - zVoxelWidth/2, self.numVoxels[2])
        xg, yg, zg = np.meshgrid(xvals, yvals, zvals, indexing='ij')
        all_points = np.vstack([xg.ravel().T, yg.ravel().T, zg.ravel().T]).T

        if chunksize is None:
            chunksize = len(all_points)            
        
        matinds = []
        nchunk = int(len(all_points)/chunksize)
        if len(all_points) % chunksize > 0:
            nchunk += 1 
        for i in np.arange(nchunk): 
            print("Processing batch ",i+1," of ",nchunk)              
            endpoint = np.min(((i+1)*chunksize, len(all_points)))
            matinds.append(GeomObj.samplePointsFast(all_points[i*chunksize:endpoint,:]))

        self.MatInds = np.hstack([m for m in matinds]).reshape(numVoxels)
        print('created voxel geometry, took {:5.2f} min'.format((time.time()-tstart)/60.0))
        
    # reads .npy format to numpy array - ripped from VoxelDataManip
    def readMatIndsFromNpy(self, readfile ):
        tstart = time.time()
        self.MatInds = np.load( readfile )
        self.numVoxels = [ len(self.MatInds[:,0,0]) , len(self.MatInds[0,:,0]) , len(self.MatInds[0,0,:]) ]
        print('read from '+readfile+', took {:5.2f} min'.format((time.time()-tstart)/60.0))

    # writes numpy array of data to .npy format - ripped from VoxelDataManip
    def writeMatIndsToNpy(self, writefile ):
        tstart = time.time()
        #np.save( writefile , self.MatInds )
        np.save( writefile , self.MatInds.astype(np.uint8) )
        print('wrote to '+writefile+', took {:5.2f} min'.format((time.time()-tstart)/60.0))


    # plots data - ripped from VoxelDataManip
    def plotMatInds(self, slicedim=None, sliceindices=[0], flgray=True, flshow=True, flsave=False, savefilenamepre=None ):
        assert slicedim=='x' or slicedim=='y' or slicedim=='z'
        assert isinstance(sliceindices,list)
        assert isinstance(flgray,bool)
        if flsave: assert isinstance(savefilenamepre,str)

        for sliceind in sliceindices:
            if slicedim=='x':
                if flgray: plt.imshow(self.MatInds[sliceind,:,:], cmap='gray', interpolation='none')
                else     : plt.imshow(self.MatInds[sliceind,:,:], interpolation='none')
                plt.xlabel('Z')
                plt.ylabel('Y')
            if slicedim=='y':
                if flgray: plt.imshow(self.MatInds[:,sliceind,:], cmap='gray', interpolation='none')
                else     : plt.imshow(self.MatInds[:,sliceind,:], interpolation='none')
                plt.xlabel('Z')
                plt.ylabel('X')
            if slicedim=='z':
                if flgray: plt.imshow(self.MatInds[:,:,sliceind], cmap='gray', interpolation='none')
                else     : plt.imshow(self.MatInds[:,:,sliceind], interpolation='none')
                plt.xlabel('Y')
                plt.ylabel('X')
    
            plt.colorbar() # add colored legend for data
            plt.title(slicedim.upper()+' = '+str(sliceind), fontdict = {'fontsize' : 28})
    
            if flsave: plt.savefig(savefilenamepre+'_slicein_'+slicedim+'_sliceindex_'+str(sliceind)+'.pdf')
            if flshow: plt.show()





