#!usr/bin/env python
from Geometry_Voxelpy import Geometry_Voxel
import numpy as np
from scipy.special import gamma
import scipy.spatial
try:
    import pandas as pd
except:
    print("Pandas not available, some functionality may run slower")
    pass
## \brief Multi-D Voronoi geometry class
# \author Emily Vu, evu@sandia.gov, emilyhvu@umich.edu
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
# Class for multi-D Voronoi geometry 
#
class Geometry_Voronoi(Geometry_Voxel):
    def __init__(self):
        super(Geometry_Voronoi,self).__init__()
        self.seedsKDTree = None
        self.flshowplot = False; self.flsaveplot = False
        self.GeomType = 'Voronoi'
        self.abundanceModel = 'ensemble'
    
    def __str__(self):
        return str(self.__dict__)
        

    ## \brief Some geometries will need to overwrite this method definition, others won't.
    #
    # \returns nothing
    def _initializeHistoryGeometryMemory(self):
        pass

    ## \brief Samples cell seed locations to define geometry, voxelizes if selected
    #
    # \returns initializes self.SeedLocations and self.SeedMatIndices; sets self.samplePoint; and voxelizes if selected
    def _initializeSampleGeometryMemory(self):
        self._sampleNumberOfSeeds()
        self._sampleSeedLocations()
        self.seedsKDTree = None
        self.samplePoint = self.samplePoint_Voronoi
        if self.flVoxelize: self.voxelizeGeometry()

    ## \brief Defines mixing parameters (correlation length and material probabilities)
    # Note: laminf and rhoV notation and formula following LarmierJQSRT2017 different mixing types paper
    #
    # \param[in] laminf, float, correlation length of Markovian mixing
    # \param[in] prob, list of floats, list of material abundances (probabilities)
    # \returns sets corr length, mat. probs, and hyperplane density rhoV, calls computation of ave num of hyperplanes
    def defineMixingParams(self,laminf=None,prob=None):
        # Assert material chord lengths and slab length and store in object
        assert isinstance(laminf,float) and laminf>0.0
        if not hasattr(self, "nummats"):
            self.nummats = len(prob)
        assert isinstance(prob,list) and len(prob)==self.nummats
        for i in range(0,self.nummats): assert isinstance(prob[i],float) and prob[i]>=0.0 and prob[i]<=1.0
        assert self.isclose( np.sum(prob), 1.0 )
        self.laminf = laminf
        self.prob   = prob
        self.rhoV   = 4.0 / (256.0*np.pi/3.0)**(1.0/3.0) / gamma(5.0/3.0) / self.laminf

        
    ## \brief compute average number of seeds
    #
    # \param[in] 
    # \computes average number of seeds in defined geometry 
    def _sampleNumberOfSeeds(self):
        # Computer average number of seeds
        xLength = abs( self.xbounds[0]-self.xbounds[1] )
        yLength = abs( self.ybounds[0]-self.ybounds[1] )
        zLength = abs( self.zbounds[0]-self.zbounds[1] )
        assert xLength==yLength and xLength==zLength   #logic currently only for a cube
        aveNumSeeds = ( xLength * self.rhoV )**3.0
        self.numSeeds = self.Rng.poisson(aveNumSeeds)
        #print(self.numSeeds, "seeds sampled")
        

    ## \brief sample number of seeds from Poisson distribution and constructs Voronoi geometry
    #
    # \returns concatonated Points and Materials vectors for each seed location 
    def _sampleSeedLocations(self):
        self.SeedLocations = []
        self.SeedMatIndices= []
        # Sample seed location
        for i in range(0,self.numSeeds):
            coordinate = self._generateRandomPoint()
            self.SeedLocations.append( coordinate )
            self.SeedMatIndices.append( None )
        self.SeedLocations = np.asarray(self.SeedLocations)
        self.SeedMatIndices = np.array(self.SeedMatIndices)
    
    ## \brief Finds (or samples) material type at current location
    #
    # \returns sets self.CurrentMatInd
    def samplePoint_Voronoi(self):
        #find distances to Voronoi seed locations; find shortest distance
        xDists = self.SeedLocations[:,0]-self.Part.x
        yDists = self.SeedLocations[:,1]-self.Part.y
        zDists = self.SeedLocations[:,2]-self.Part.z
        minDistIndex = np.argmin( np.sqrt( xDists**2 + yDists**2 + zDists**2 ) )
        #check if material has been sampled in this cell; if not, sample material type
        if self.SeedMatIndices[minDistIndex] == None:
            self.SeedMatIndices[minDistIndex] = int(self.Rng.choice(p=self.prob))
        self.CurrentMatInd = self.SeedMatIndices[minDistIndex]

    ## \brief Samples a new point according to the selected rules
    #
    # \param[in] points, list of float, list of points to sample  
    # \returns scattering ratio of cell material type
    def samplePointsFast (self, points):
        if self.seedsKDTree is None:
            self.seedsKDTree = scipy.spatial.cKDTree(self.SeedLocations)
        #find index of nearest point
        min_dists, min_indices = self.seedsKDTree.query(points)
        min_indices = np.array(min_indices)
        materials = self.SeedMatIndices[min_indices]
        unassigned_inds = min_indices[materials == None]        
        if len(unassigned_inds) > 0:
            try:
                uniques = pd.unique(unassigned_inds)
            except:
                uniques, inds = np.unique(unassigned_inds, return_index=True)
                uniques = uniques[np.argsort(inds)] 
            #The above is just to preserve order and provide identical results to the 'slow' version.
            
            new_assignments = self.Rng.RngObj.choice(self.nummats, size=len(uniques), p=self.prob)
            self.SeedMatIndices[uniques] = new_assignments
            materials = self.SeedMatIndices[min_indices]
        return materials.astype(np.int16)