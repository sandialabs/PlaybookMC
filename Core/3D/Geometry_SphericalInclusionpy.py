#!usr/bin/env python
from Geometry_Voxelpy import Geometry_Voxel
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

## \brief Multi-D Spherical Inclusion geometry class
# \author Dan Bolintineanu, dsbolin@sandia.gov
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
# \author Emily Vu, evu@sandia.gov, emilyhvu@umich.edu
#
# Class for multi-D SphericalInclusion geometry 
#
class Geometry_SphericalInclusion(Geometry_Voxel):
    def __init__(self,flVerbose=False,abundanceModel='ensemble'):
        super(Geometry_SphericalInclusion,self).__init__()
        self.flshowplot = False
        self.flsaveplot = False
        self.GeomType = 'SphericalInclusions'
        assert isinstance(flVerbose,bool)
        self.flVerbose = flVerbose  
        assert abundanceModel in {'ensemble','sample'}      
        self.abundanceModel = abundanceModel
        
    
    def __str__(self):
        return str(self.__dict__)
    
    ## \brief Some geometries will need to overwrite this method definition, others won't.
    #
    # \returns nothing
    def _initializeHistoryGeometryMemory(self):
        pass

    ## \brief Defines spherical inclusions mixing parameters
    #
    # \param[in] sphereFrac, float, target fraction of volume to be inclusions
    # \param[in] sizeDistribution, string; 'Constant','Uniform','Exponential' the distribution of radii (can add any of the distributions in scipy.stats, currently these three available)
    # \param[in] radMin, float default 0.0; minimum sphere/circle radius to sample
    # \param[in] radAve, float; numerical input, currently either radius of all spheres or free parameter to exponential distribution of radii
    # \param[in] radMax, float default np.inf; maximum sphere/circle radius to sample
    # \param[in] sphereMatProbs, list of floats; probability of each material type within spheres
    # \param[in] xperiodic, bool default False; geometry is periodic in the x direction
    # \param[in] yperiodic, bool default False; geometry is periodic in the y direction
    # \param[in] zperiodic, bool default False; geometry is periodic in the z direction
    # \returns sets several parameters to describe spherical/circlular inclusion sampling
    def defineMixingParams(self,sphereFrac,sizeDistribution,radMin,radAve,radMax,sphereMatProbs=None,xperiodic=False,yperiodic=False,zperiodic=False):
        if not hasattr(self, "nummats"): self.nummats = len(sphereMatProbs)+1
        assert self.nummats == len(sphereMatProbs)+1
        assert isinstance(sphereFrac,float) and sphereFrac>0.0 and sphereFrac<1.0
        self.sphereFrac = sphereFrac

        assert sizeDistribution in {'Constant','Uniform','Exponential'}
        self.sizeDistribution = sizeDistribution

        assert isinstance(radMin,float) and radMin>=0.0
        assert isinstance(radAve,float) and radAve>=0.0
        assert isinstance(radMax,float) and radMax>=radMin
        if self.sizeDistribution=='Constant': assert radMin==radAve and radAve==radMax
        if self.sizeDistribution=='Uniform' : assert (radMin+radMax)/2==radAve
        self.radMin = radMin
        self.radAve = radAve
        self.radMax = radMax

        if sizeDistribution=='Constant'   : self.distribution = scipy.stats.uniform(loc=self.radAve, scale=0          )
        if sizeDistribution=='Uniform'    : self.distribution = scipy.stats.uniform(loc=self.radMin, scale=self.radMax)
        if sizeDistribution=='Exponential': self.distribution = scipy.stats.expon(                   scale=self.radAve)

        assert isinstance(sphereMatProbs,list) and np.sum(sphereMatProbs)==1.0
        for prob in sphereMatProbs: assert isinstance(prob,float) and prob>=0.0 and prob<=1.0
        self.sphereMatProbs = sphereMatProbs
        
        assert isinstance(xperiodic,bool) and isinstance(yperiodic,bool) and isinstance(zperiodic,bool)
        self.xperiodic = xperiodic
        self.yperiodic = yperiodic
        self.zperiodic = zperiodic

        self.prob      = np.multiply(sphereFrac,sphereMatProbs) 
        self.prob      = np.insert(self.prob, 0, 1.0 - sphereFrac)


    ## \brief Defines parameters for geometry generation
    #
    # Differences between fastRSA and generic grid methods:
    # - Grid size:
    #     * Fast RSA method (Brown et al): grid size is set to 2R/sqrt(3). This ensures that each grid cell 
    #        contains at most one sphere. The original method only applies to a constant radius. In the modified version here,
    #        the grid size is set as 2*R_min/sqrt(3). In both cases, a list of empty grid cells is maintained.
    #        However, for the case of polydispersity (R not equal for all spheres), this implies that 
    #           a. the grid overlap search could extend beyond the nearest neighbor grid cells; and 
    #           b. A sphere with R>R_min could extend across multiple grid cells. 
    #        As such, the tradeoff relative to larger grid cell sizes is fast addition of spheres to grid cells, at the
    #        cost of higher cost in searching neighboring grid cells for overlap. For R not constant, in addition to a list
    #         of empty grid cells, the list of all spheres belonging to each grid cell is maintained.
    #     * Generic grid method: grid size is a flexible parameter, with default set to 2*R_max. When a sphere of radius R is 
    #        inserted in the domain, it is added to any grid cells within R+R_max in all directions. 
    #        Instead of maintaining a list of empty grid cells as in the Fast RSA method, only a list of all spheres 
    #        associated with each grid cell is maintained.
    #
    # - Selection of insertion points
    #     * Fast RSA: A list of empty grid cells is maintained. At each insertion attempt, a cell is selected from this list
    #        at random. A point is then selected within this cell based on random uniform sampling.
    #     * Generic grid method: At each insertion attempt, a point is selected within the entire domain
    #        based on random uniform sampling. The point is then mapped to the appropriate grid cell.
    #
    # \param[in] maxPlacementAttempts, int; maximum number of times to try to place a sphere
    # \param[in] sphsampleModel, str; 'NoGrid','GenericGrid', or 'FastRSA'--model to sample spheres for geometry and return material at a point in the geometry.  'NoGrid' and 'GenericGrid' yield the same results, but 'NoGrid' tends to return these results faster for simpler problems, whereas 'GenericGrid' tends to return these results faster for more complex problems (as long as gridSize is chosen reasonably). 'FastRSA' is the grid-based approach discussed in BrownANS2006, is only applicable for constant-radius spheres, does not provide the exact same structures as the other methods with the same random number seed (due to the differences in the methods), and is the fastest of the three for some problems with medium to high sphere volume fraction.
    # \param[in] gridSize, float; size of grid cells for 'GenericGrid' method. Grid irrelevant for 'NoGrid', selected automatically for 'FastRSA', but can be set by user for 'GenericGrid'--if user doesn't set, will be set automatically to 2*R_max.          
    # \returns sets functions and values for generating spherical inclusion geometries
    def defineGeometryGenerationParams(self, maxPlacementAttempts=1000, sphSampleModel=None, gridSize=None):
        assert isinstance(maxPlacementAttempts,int) and maxPlacementAttempts>0
        self.maxPlacementAttempts = maxPlacementAttempts        
        assert sphSampleModel in {'NoGrid','GenericGrid','FastRSA'}
        self.sphSampleModel = sphSampleModel

        #assert Constant radius if using FastRSA
        if sphSampleModel=='FastRSA'    : assert self.radMin==self.radAve and self.radAve==self.radMax

        #check and set grid size
        if   sphSampleModel=='NoGrid'     : assert gridSize==None #there is no grid; gridSize is irrelevant
        elif sphSampleModel=='FastRSA'    : assert gridSize==None #grid size will be chosen based on problem parameters
        elif sphSampleModel=='GenericGrid': assert gridSize==None or (isinstance(gridSize,float) and gridSize>0.0)
        self.gridSize= gridSize

        if   sphSampleModel=='NoGrid'     :
            self.generateRandomPoint = self._generateRandomPoint
            self.checkSphereOverlap  = self._checkSphereOverlapNoGrid            
        elif sphSampleModel=='GenericGrid':
            self.generateRandomPoint = self._generateRandomPoint
            self.checkSphereOverlap  = self._checkSphereOverlapGrid
            self.addSphereToBins     = self._addSphereToBinsGrid                
        elif sphSampleModel=='FastRSA'    :
            self.generateRandomPoint = self._generateRandomPointFastRSA
            self.checkSphereOverlap  = self._checkSphereOverlapGridFastRSA
            self.addSphereToBins     = self._addSphereToBinsFastRSA

    ## \brief Samples spherical inclusion realization, voxelizes if selected
    #
    # \returns initializes self.Centers and self.MatInds; sets self.samplePoint; and voxelizes if selected
    def _initializeSampleGeometryMemory(self):   
        self.lx = self.xbounds[1]-self.xbounds[0]
        self.ly = self.ybounds[1]-self.ybounds[0]
        self.lz = self.zbounds[1]-self.zbounds[0]
        
        self._sampleSphereRadii()
        if self.sphSampleModel in {'GenericGrid','FastRSA'}: self._setupGrid()
        
        self.Centers = []        
        self.Radii = []
        for i,rad in enumerate(self.targetRadii):
            numAttempts = 0
            if self.flVerbose:
                print(f"Attempting to place sphere {i+1} of {len(self.targetRadii)}, radius = {rad}...")
            while True:
                numAttempts += 1
                x, y, z = self.generateRandomPoint(radius = rad)
                if self.checkSphereOverlap(x, y, z, rad) < 0:
                    if self.sphSampleModel in {'GenericGrid','FastRSA'}: self.addSphereToBins(x, y, z, rad, len(self.Centers))
                    self.Centers.append([x,y,z])
                    self.Radii.append(rad)
                    if self.flVerbose:
                        print(f"Successfully placed at {x,y,z} after {numAttempts} attempts")
                    break
                if numAttempts > self.maxPlacementAttempts:                    
                    raise Exception(f"Failed to place particle {i+1} with radius {rad}")
                
        self.Centers = np.array(self.Centers)          
        self.Radii = np.array(self.Radii)        
        vf = 4/3*np.pi*np.sum([rad**3 for rad in self.Radii])/(self.lx*self.ly*self.lz)
        if self.flVerbose:
            print(f"Placed a total of {len(self.Radii)} spheres, resulting in a final volume fraction of {vf}")
        
        # sample what material in each sphere according to user specified average material abundance in spheres
        self.MatInds   = []
        sphereMatChoices = np.arange(1, len(self.sphereMatProbs)+1)
        for _ in range(0,len(self.Radii)):
            self.MatInds.append(sphereMatChoices[self.Rng.choice(p=self.sphereMatProbs)])
        self.MatInds = np.array(self.MatInds)

        # set samplePoint
        if   self.sphSampleModel ==  'NoGrid'                : self.samplePoint = self._samplePointNoGrid
        elif self.sphSampleModel in {'GenericGrid','FastRSA'}: self.samplePoint = self._samplePointGrid
        # is selected, turn into voxel format
        if self.flVoxelize: self.voxelizeGeometry()



    ## \brief Returns random point in domain
    #
    # \returns x,y,z coordinates of random point
    def _generateRandomPoint(self, radius=0):                

        #Select a random point
        if self.xperiodic:
            xlo = self.xbounds[0]
            xhi = self.xbounds[1]
        else:
            xlo = self.xbounds[0]+radius
            xhi = self.xbounds[1]-radius
        x = self.Rng.uniform(xlo, xhi)   

        if self.yperiodic:
            ylo = self.ybounds[0]
            yhi = self.ybounds[1]
        else:
            ylo = self.ybounds[0]+radius
            yhi = self.ybounds[1]-radius
        y = self.Rng.uniform(ylo, yhi)   

        if self.zperiodic:
            zlo = self.zbounds[0]
            zhi = self.zbounds[1]
        else:
            zlo = self.zbounds[0]+radius
            zhi = self.zbounds[1]-radius
        z = self.Rng.uniform(zlo, zhi)   
        
        return x, y, z


    ## \brief Returns random point in domain using fast RSA method
    #
    # First selects an empty grid cell and then samples sphere center from within the cell.
    #
    # \returns x,y,z coordinates of random point
    def _generateRandomPointFastRSA(self, radius):        
        #Select an empty cell
        cellIndex = self.Rng.RngObj.choice(len(self.emptyCellList))
        emptyCell = self.emptyCellList[cellIndex]
        ix, iy, iz = self.cellIndexToCoords[emptyCell]        
        
        xlo = self.xbounds[0] + ix*self.gridSize
        ylo = self.ybounds[0] + iy*self.gridSize
        zlo = self.zbounds[0] + iz*self.gridSize

        xhi = xlo + self.gridSize
        yhi = ylo + self.gridSize
        zhi = zlo + self.gridSize

        #Select a random point within the cell
        if self.xperiodic:
            if xhi > self.xbounds[1]: xhi = self.xbounds[1]
        else:
            if ix == 0              : xlo = xlo+radius
            if ix == self.nbinx-1   : xhi = self.xbounds[1] - radius
        x = self.Rng.uniform(xlo, xhi)
        
        if self.yperiodic:            
            if yhi > self.ybounds[1]: yhi = self.ybounds[1]
        else:
            if iy == 0              : ylo = ylo+radius
            if iy == self.nbiny-1   : yhi = self.ybounds[1] - radius
        y = self.Rng.uniform(ylo, yhi)
        
        if self.zperiodic:            
            if zhi > self.zbounds[1]: zhi =  self.zbounds[1]
        else:
            if iz == 0              : zlo = zlo+radius
            if iz == self.nbinz-1   : zhi = self.zbounds[1] - radius
        z = self.Rng.uniform(zlo, zhi)

        return x, y, z
    
    
    ## \brief Returns material at specified location
    #
    # Distance calculation used with periodic boundaries uses the minimum image convention
    #
    # \returns material type of location
    def _samplePointNoGrid(self):
        dx                    = self.Centers[:,0]-self.Part.x                    
        if self.xperiodic: dx = dx - np.int32(2*dx/self.lx)*self.lx

        dy                    = self.Centers[:,1]-self.Part.y                   
        if self.yperiodic: dy = dy - np.int32(2*dy/self.ly)*self.ly

        dz                    = self.Centers[:,2]-self.Part.z                    
        if self.zperiodic: dz = dz - np.int32(2*dz/self.lz)*self.lz
        
        distances = np.sqrt(dx**2 + dy **2 + dz**2)
        overlapIndices  = np.where(self.Radii > distances)[0]
        if len(overlapIndices) > 0: self.CurrentMatInd = self.MatInds[overlapIndices[0]]
        else                      : self.CurrentMatInd = 0

    ## \brief Returns material at specified location, using a grid
    #
    # \returns material type of location.
    def _samplePointGrid(self):        
        sphere_index = self.checkSphereOverlap(self.Part.x, self.Part.y, self.Part.z, 0)
        if sphere_index >= 0: self.CurrentMatInd = self.MatInds[sphere_index]
        else                : self.CurrentMatInd = 0

    ## \brief Adds sphere to bins that it could potentially overlap
    #
    # This is specific to FastRSA, taking advantage of the radius being constant
    # by adding the sphere to a single bin.
    #
    # \param[in] x, y, x, float; coordinates of sphere center
    # \param[in] rad, float; sphere radius
    # \param[in] index, int; index of sphere in list of spheres        
    # \returns Updates spheres contained in grid cell
    def _addSphereToBinsFastRSA(self, x, y, z, rad, index):
        ix = int((x - self.xbounds[0]) / self.gridSize)
        iy = int((y - self.ybounds[0]) / self.gridSize)
        iz = int((z - self.zbounds[0]) / self.gridSize)
        gridIndex = self.cellCoordsToIndex[(ix,iy,iz)]
        self.grid[ix, iy, iz] = [[x, y, z, rad, index]]
        self.emptyCellList = np.delete(self.emptyCellList, self.emptyCellList == gridIndex)           
                   

    ## \brief Adds sphere to bins that it could potentially overlap
    #
    # This is generic to any bin size/radius combination, so bins could extend beyond
    # nearest neighbors, in all directions.
    # When a sphere is inserted, it is assigned to all bins that are within
    # its radius plus radMax. This makes it so that overlap checks can be done for
    # only the bin that new spheres are inserted in. An alternative scheme would have
    # a sphere assigned only to bins that are within its radius, and overlap checks would
    # then have to extend beyond the bin that a new sphere is inserted in. This is not
    # currently supported for generic grid styles, but could easily be added. However, it
    # would require additional user input, and it is not clear
    # what the optimal combination of bin assignment vs. overlap checks is.  
    #
    # \param[in] x0, y0, x0, float; coordinates of sphere center
    # \param[in] rad, float; sphere radius
    # \param[in] index, int; index of sphere in list of spheres        
    # \returns Updates spheres contained in grid cell, and returns list of grid cells that were updated
    def _addSphereToBinsGrid(self, x0, y0, z0, rad, index):
        grid_cells_added = []
        x = x0 - self.xbounds[0]
        y = y0 - self.ybounds[0]
        z = z0 - self.zbounds[0]
        maxOverlapDistance = rad+self.radMax 

        #Sphere is added to all grid cells where overlaps need to be checked.
        #ixlo/ixhi, iylo/iyhi, izlo/izhi are the extents of the grid subdomain,
        #as grid indices. Negative values of ixlo,iylo,izlo will be used to wrap
        #around periodic boundaries, if applicable

        #Grid subdomain extents in x
        ixlo = int((x - maxOverlapDistance)/self.gridSize)-1
        if x < maxOverlapDistance: #Needed due to potential truncation of grid cell at upper domain boundary
            ixlo -= 1
        ixhi = int((x + maxOverlapDistance)/self.gridSize)+1
        if ixlo < 0 and not self.xperiodic:
            ixlo = 0
        if ixhi >= self.nbinx and not self.xperiodic:
            ixhi = self.nbinx-1

        #Grid subdomain extens in y
        iylo = int((y - maxOverlapDistance)/self.gridSize)-1
        if y < maxOverlapDistance:
            iylo -= 1
        iyhi = int((y + maxOverlapDistance)/self.gridSize)+1
        if iylo < 0 and not self.yperiodic:
            iylo = 0
        if iyhi >= self.nbiny and not self.yperiodic:
            iyhi = self.nbiny-1

        #Bin extents in z
        izlo = int((z  - maxOverlapDistance)/self.gridSize)-1
        if z < maxOverlapDistance:
            izlo -= 1
        izhi = int((z + maxOverlapDistance)/self.gridSize)+1
        if izlo < 0 and not self.zperiodic:
            izlo = 0
        if izhi >= self.nbinz and not self.zperiodic:
            izhi = self.nbinz-1

        #Loop over bins
        for ix in np.arange(ixlo, ixhi+1):
            iix, xs = self._wrap_bins(ix, x, self.nbinx, self.lx, self.xperiodic)
            for iy in np.arange(iylo, iyhi+1):
                iiy, ys = self._wrap_bins(iy, y, self.nbiny, self.ly, self.yperiodic)
                for iz in np.arange(izlo, izhi+1):
                    iiz, zs = self._wrap_bins(iz, z, self.nbinz, self.lz, self.zperiodic)                    
                    xc = (iix + 0.5)*self.gridSize 
                    yc = (iiy + 0.5)*self.gridSize 
                    zc = (iiz + 0.5)*self.gridSize 
                    if (xs-xc)*(xs-xc) + (ys-yc)*(ys-yc) + (zs-zc)*(zs-zc) < (maxOverlapDistance + self.bindiag)*(maxOverlapDistance + self.bindiag):                   
                        #If sphere is within relevant distance of grid cell center, add sphere to grid cell
                        if self.grid[iix, iiy, iiz] is None:
                            self.grid[iix, iiy, iiz] = []                        
                        self.grid[iix, iiy, iiz].append([xs + self.xbounds[0], 
                                                         ys + self.ybounds[0], 
                                                         zs + self.zbounds[0],
                                                           rad, index])
                        grid_cells_added.append((iix, iiy, iiz))
        return grid_cells_added


    ## \brief Check if a given sphere overlaps any existing spheres
    #
    # Checks all possible pairs of spheres, hence scaling is N^2
    # Distance calculation used with periodic boundaries uses the minimum image convention
    #
    # \param[in] x, y, x, float; coordinates of sphere center
    # \param[in] rad, float: sphere radius    
    # \returns Index of sphere that is overlapped; -1 if no overlap
    def _checkSphereOverlapNoGrid(self, x, y, z, rad):
        if len(self.Centers) == 0: return -1
        centers = np.array(self.Centers)
        dx                    = centers[:,0]-x
        if self.xperiodic: dx = dx - np.int32(2*dx/self.lx)*self.lx

        dy                    = centers[:,1]-y
        if self.yperiodic: dy = dy - np.int32(2*dy/self.ly)*self.ly

        dz                    = centers[:,2]-z
        if self.zperiodic: dz = dz - np.int32(2*dz/self.lz)*self.lz

        distances = np.sqrt(dx**2 + dy**2 + dz**2)
        overlapIndices = distances < self.Radii + rad
        if np.any(overlapIndices): return np.where(overlapIndices)[0][0]
        else                     : return -1

    ## \brief Check if a given sphere overlaps any existing spheres
    #
    # Checks overlap between given sphere and all spheres in the bin that the
    # given sphere is located in, as well as neighboring bins. This is specific
    # to the fast RSA method, where only the nearest neighboring bins need to be 
    # checked.
    #
    # The sphere is checked against all grid cells where overlaps could occur.
    # Variables ixlo/ixhi, iylo/iyhi, and izlo/izhi are the extents of the grid
    # subdomain where overlap checking is relevant, as grid indices.
    # Negative values of ixlo, iylo, and izlo will be used to wrap around
    # periodic boundaries, if applicable.
    #
    # \param[in] x0, y0, z0, float; coordinates of sphere 
    # \param[in] rad, float: sphere radius    
    # \returns Index of sphere that is overlapped; -1 if no overlap
    def _checkSphereOverlapGridFastRSA(self, x0, y0, z0, rad):
        x = x0 - self.xbounds[0]
        y = y0 - self.ybounds[0]
        z = z0 - self.zbounds[0]

        ix = int(x/self.gridSize)
        iy = int(y/self.gridSize)        
        iz = int(z/self.gridSize)


        ixlo = ix - self.fastRSANumGridCheck
        ixhi = ix + self.fastRSANumGridCheck 
        if self.xperiodic:         ixlo -= 1     
        else             :
            if ixlo <  0         : ixlo  = 0
            if ixhi >= self.nbinx: ixhi  = self.nbinx-1

        iylo = iy - self.fastRSANumGridCheck
        iyhi = iy + self.fastRSANumGridCheck
        if self.yperiodic:         iylo -= 1
        else             :
            if iylo <  0         : iylo = 0
            if iyhi >= self.nbiny: iyhi = self.nbiny-1

        izlo = iz - self.fastRSANumGridCheck
        izhi = iz + self.fastRSANumGridCheck
        if self.zperiodic:         izlo -= 1
        else             :
            if izlo <  0         : izlo = 0
            if izhi >= self.nbinz: izhi = self.nbinz-1

        for ix in np.arange(ixlo, ixhi+1):
            iix, xs = self._wrap_bins(ix, x0, self.nbinx, self.lx, self.xperiodic)            
            for iy in np.arange(iylo, iyhi+1):
                iiy, ys = self._wrap_bins(iy, y0, self.nbiny, self.ly, self.yperiodic)
                for iz in np.arange(izlo, izhi+1):
                    iiz, zs = self._wrap_bins(iz, z0, self.nbinz, self.lz, self.zperiodic)
                    bin = self.grid[iix, iiy, iiz]  
                    if bin is None:
                        continue        
                    for particle in bin:            
                        px, py, pz, prad, index = particle
                        rx = xs - px                        
                        ry = ys - py
                        rz = zs - pz
                        r2 = np.sqrt(rx*rx + ry*ry + rz*rz)
                        if r2 < rad + prad:
                            return index
        return -1

    ## \brief Check if a given sphere overlaps any existing spheres
    #
    # Checks overlap between given sphere and all spheres in the bin that the
    # given sphere is located in.
    # Note: overlap needs to check only if center overlaps bin, 
    # since adding particle to bins covers range of maxrad+rad.
    #
    # \param[in] x, y, x, float; coordinates of sphere center
    # \param[in] rad, float; sphere radius    
    # \returns Index of sphere that is overlapped; -1 if no overlap
    def _checkSphereOverlapGrid(self, x, y, z, rad):
        ix = int((x - self.xbounds[0])//self.gridSize)
        iy = int((y - self.ybounds[0])//self.gridSize)        
        iz = int((z - self.zbounds[0])//self.gridSize)
        bin = self.grid[ix, iy, iz]        
        if bin is None:
            return -1        
        for particle in bin:            
            px, py, pz, prad, index = particle
            rx = x - px
            ry = y - py
            rz = z - pz            
            r2 = np.sqrt(rx*rx + ry*ry + rz*rz)
            if r2 < rad + prad:
                return index
        return -1
    
    ## \brief Wrap bin index and sphere coordinates around periodic boundaries
    #
    # Helper function for wrapping bins and sphere coordinates around periodic boundaries
    #
    # \param[in] ibin, int; index of bin along relevant direction
    # \param[in] coord, float; coordinate of sphere along relevant direction
    # \param[in] nbins, int; number of bins along relevant direction
    # \param[in] length, float; length of domain along relevant direction
    # \param[in] periodic, bool; periodicity flag along relevant direction
    # \returns Bin index, coordinate of sphere wrapped around periodic boundaries along relevant direction
    def _wrap_bins(self, ibin, coord, nbins, length, periodic):           
        if not periodic:
            return ibin, coord
        if ibin < 0:
            wbin = np.max((nbins + ibin, 0))
            wcoord = coord + length
        elif ibin >= nbins:
            wbin = np.min((ibin - nbins, nbins-1))
            wcoord = coord - length
        else:
            return ibin, coord
        return wbin, wcoord
    
    
    ## \brief Samples sphere radii from the user-specified distribution until desired volume fraction met
    #
    #  In the case of uniform and exponential distributions, the sphere radius of the last sphere sampled
    #  is adjusted to ensure that the exact user-specified volume fraction is achieved.
    #
    # \returns Sets the list of sphere radii that will be placed
    def _sampleSphereRadii(self):
        totSphVol  = 0.0
        radii = []        
        while True:
            flbreak  = False                                            #variable to end loop when True

            randstate = self.Rng.randint(1,100000)             
            [radius] = self.distribution.rvs(1, random_state = randstate) #sample radius
            if radius < self.radMin or radius > self.radMax: 
                continue #reject if larger or smaller than user allowing
            
            newSphVol = 4/3*np.pi*np.cumsum(radius**3) #compute new sphere/circle volume/area

            if (totSphVol+newSphVol)/self.Volume > self.sphereFrac:     #volume fraction exceeded, handle last sphere/circle
                flbreak = True
                if self.sizeDistribution in {'Uniform','Exponential'}:  #set last radius to size needed to get target volume fraction as in BrantleyMC2011; if 'Constant', accept last sphere as in BrantleyMC2011
                    volRemain = self.sphereFrac*self.Volume - totSphVol     #compute volume remaining
                    radius = (3/4*volRemain/np.pi)**(1/3)
                    if radius < self.radMin:
                        self.radMin = radius 
                        self._setupGrid()                       

            totSphVol += newSphVol
            radii.append( float(radius) )                                      #add radius to list
            if flbreak: 
                break
        self.targetRadii = np.sort(radii)[::-1]


    ## \brief Set up the grid data structure, for fast RSA and generic grid methods
    #
    #  \returns Sets the grid array, and other properties relevant to the grid.
    def _setupGrid(self):
        if   self.sphSampleModel=='GenericGrid':
            if self.gridSize is None: self.gridSize = self.radMax
        elif self.sphSampleModel=='FastRSA':
            self.gridSize = 2*np.min(self.targetRadii)/np.sqrt(3) # ensures each grid cell contains at most a single particle:
            self.fastRSANumGridCheck = np.int32(np.min(self.targetRadii)/self.gridSize+1) + 1 # number of grid neighbors to check

        self.lx = self.xbounds[1]-self.xbounds[0]
        self.ly = self.ybounds[1]-self.ybounds[0]
        self.lz = self.zbounds[1]-self.zbounds[0]
        
        self.nbinx = int((self.lx//self.gridSize)+1)
        self.nbiny = int((self.ly//self.gridSize)+1)
        self.nbinz = int((self.lz//self.gridSize)+1)
    
        self.grid = np.empty((self.nbinx, self.nbiny, self.nbinz), dtype=object)
        self.bindiag = np.sqrt(3)*self.gridSize        

        if self.sphSampleModel=='FastRSA':
            self.cellIndexToCoords = np.empty(self.nbinx*self.nbiny*self.nbinz, dtype=object)
            self.cellCoordsToIndex = {}
            self.emptyCellList = []
            i = 0
            for ix in np.arange(0, self.nbinx):
                for iy in np.arange(0, self.nbiny):
                    for iz in np.arange(0, self.nbinz):
                        self.cellIndexToCoords[i] = (ix, iy, iz)
                        self.cellCoordsToIndex[(ix,iy,iz)] = i
                        self.emptyCellList.append(i)
                        i = i + 1
            self.emptyCellList = np.array(self.emptyCellList)


    ## \brief Makes 2D plot of geometry
    #
    # \param[in] ipart, which particle history is being plotted 
    # \returns can show or print plot
    def plotRealization(self,ipart):
        x_mat = []; y_mat = []
        for imat in range(0,self.nummats):
            x_mat.append([])
            y_mat.append([])
        for ipt in range(0,len(self.Points)):
            x_mat[ self.MatInds[ipt] ] = self.Points[ipt][0]
            y_mat[ self.MatInds[ipt] ] = self.Points[ipt][1]

        plt.figure
        plt.title('ipart '+str(ipart+1)+'')
        plt.xlabel('x'); plt.ylabel('y')
        for imat in range(0,self.nummats):
            plt.scatter(x_mat[imat],y_mat[imat])
        plt.xlim( self.xbounds[0], self.xbounds[1]);  plt.ylim( self.ybounds[0], self.ybounds[1])
        if self.flsaveplot == True: plt.savefig(''+str(self.GeomType)+'_'+str(self.Part.numDims)+'D_Case-name-_Part'+str(ipart+1)+'.png')
        if self.flshowplot == True: plt.show()

    ## \brief Returns material at specified set of points
    #
    # This is a vectorized way to get material indices at a series of points in a geometry.
    # The primary intended use of this function is for fast voxelization. The vectorization across
    # many points does not use grids regardless of how the structure was generated; this sacrifices potential
    # efficiency of grid searches in favor of efficiency of vectorization across many points
    #
    # \param [in] points, NX3 numpy array of point coordinates, where N is number of points
    #
    # \returns material types at point locations
    def samplePointsFast(self, points):
        dx = self.Centers[:,0].reshape((-1,1))-points[:,0].reshape((1,-1))                   
        if self.xperiodic: #Set distance to nearest periodic image, aka Minimum Image Convention
            dx = dx - np.int32(2*dx/self.lx)*self.lx                    

        dy = self.Centers[:,1].reshape((-1,1))-points[:,1].reshape((1,-1))                         
        if self.yperiodic:
            dy = dy - np.int32(2*dy/self.ly)*self.ly                    

        dz = self.Centers[:,2].reshape((-1,1))-points[:,2].reshape((1,-1))      
        if self.zperiodic:
            dz = dz - np.int32(2*dz/self.lz)*self.lz                    
        
        distances = np.sqrt(dx**2 + dy **2 + dz**2)
        indices_of_particles, indices_of_points = np.where(distances < self.Radii.reshape((-1,1)))
        materials = np.zeros_like(points[:,0])
        materials[indices_of_points] = self.MatInds[indices_of_particles]
        return materials