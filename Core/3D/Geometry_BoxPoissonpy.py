#!usr/bin/env python
from Geometry_Basepy import Geometry_Base
import numpy as np
import bisect as bisect
try:
    import pandas as pd
except:
    print("Pandas not available, some functionality may run slower")
    pass


## \brief Multi-D Box-Poisson geometry class
# \author Emily Vu, evu@sandia.gov, emilyhvu@umich.edu
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
# Class for multi-D Box-Poisson geometry 
#
class Geometry_BoxPoisson(Geometry_Base):
    def __init__(self):
        super(Geometry_BoxPoisson,self).__init__()
        self.flshowplot = False; self.flsaveplot = False
        self.kdtrees = False
        self.GeomType = 'BoxPoisson'
        self.abundanceModel = 'ensemble'

    def __str__(self):
        return str(self.__dict__)

    
    ## \brief Some geometries will need to overwrite this method definition, others won't.
    #
    # \returns nothing
    def _initializeHistoryGeometryMemory(self):
        pass

    ## \brief Initializes list of material types and coordinate of location, initializes xyz boundary locations
    #
    # Note: Numpy arrays don't start blank and concatenate, where lists do, therefore plan to use lists, but convert to arrays using np.asarray(l) if needed to use array format
    #
    # \returns initializes self.MatInds and self.xBoundaries, self.yBoundaries, self.zBoundaries
    def _initializeSampleGeometryMemory(self):
        self._sampleNumberOfHyperplanes()
        self._sampleHyperplaneLocations()

    ## \brief Defines mixing parameters (correlation length and material probabilities)
    # Note: laminf and rhoB notation and formula following LarmierJQSRT2017 different mixing types paper
    #
    # \param[in] laminf, float, correlation length of Markovian mixing
    # \param[in] prob, list of floats, list of material abundances (probabilities)
    # \returns sets corr length, mat. probs, and hyperplane density rhoB
    def defineMixingParams(self,laminf=None,prob=None):
        # Assert material chord lengths and slab length and store in object
        assert isinstance(laminf,float) and laminf>0.0
        assert isinstance(prob,list) and len(prob)==self.nummats
        for i in range(0,self.nummats): assert isinstance(prob[i],float) and prob[i]>=0.0 and prob[i]<=1.0
        assert self.isclose( np.sum(prob), 1.0 )
        self.laminf = laminf
        self.prob   = prob
        self.rhoB   = 2.0 / 3.0 / self.laminf

                
    ## \brief compute number of pseudo-interfaces
    #
    # \computes number of pseudo-interfaces in x, y, and z direction for defined geometry 
    def _sampleNumberOfHyperplanes(self):
        xLength = abs( self.xbounds[0]-self.xbounds[1] );  xAve = self.rhoB * xLength
        yLength = abs( self.ybounds[0]-self.ybounds[1] );  yAve = self.rhoB * yLength
        zLength = abs( self.zbounds[0]-self.zbounds[1] );  zAve = self.rhoB * zLength
        assert xLength==yLength and xLength==zLength  #logic currently only for a cube
        self.xPInterfaces = self.Rng.poisson(xAve)
        self.yPInterfaces = self.Rng.poisson(yAve)
        self.zPInterfaces = self.Rng.poisson(zAve)
        self.MatInds = np.full( ( self.xPInterfaces+1 , self.yPInterfaces+1 , self.zPInterfaces+1 ),None )
        #print(self.xPInterfaces,self.yPInterfaces,self.zPInterfaces, "planes sampled in x, y, and z")


    ## \brief sample location of pseudo-interfaces and constructs Box-Poisson geometry
    #
    # \returns boundary location vectors for each cartesianal direction 
    def _sampleHyperplaneLocations(self):
        self.xBoundaries = [self.xbounds[0],self.xbounds[1]]
        self.yBoundaries = [self.ybounds[0],self.ybounds[1]]
        self.zBoundaries = [self.zbounds[0],self.zbounds[1]]
        
        for x in range(0,self.xPInterfaces):
            location = self.Rng.uniform(self.xbounds[0],self.xbounds[1])
            self.xBoundaries.append(location)

        for y in range(0,self.yPInterfaces):
            location = self.Rng.uniform(self.ybounds[0],self.ybounds[1])
            self.yBoundaries.append(location)

        for z in range(0,self.zPInterfaces):
            location = self.Rng.uniform(self.zbounds[0],self.zbounds[1])
            self.zBoundaries.append(location)
            
        self.xBoundaries.sort()
        self.yBoundaries.sort()
        self.zBoundaries.sort()
        
    ## \brief Samples a new point according to the selected rules
    #
    # \returns scattering ratio of cell material type
    def samplePoint(self):
        #find index of nearest point
        x = self.Part.x
        y = self.Part.y
        z = self.Part.z

        xIndex = bisect.bisect(self.xBoundaries,x) - 1
        yIndex = bisect.bisect(self.yBoundaries,y) - 1
        zIndex = bisect.bisect(self.zBoundaries,z) - 1
        #check if material has been sampled. If not sample material type.
        self.CurrentMatInd = self.MatInds[xIndex,yIndex,zIndex]
        if self.CurrentMatInd==None:
            self.CurrentMatInd = int(self.Rng.choice(p=self.prob))
            self.MatInds[xIndex,yIndex,zIndex] = self.CurrentMatInd
        
    def samplePointsFast(self, points):
        #find index of nearest point 
        xinds = np.searchsorted(self.xBoundaries, points[:,0], side='right')-1
        yinds = np.searchsorted(self.yBoundaries, points[:,1], side='right')-1
        zinds = np.searchsorted(self.zBoundaries, points[:,2], side='right')-1
                
        array_of_indices = np.vstack((xinds, yinds, zinds)).T
        #check if material has been sampled. If not, sample material type.
        matTypes = self.MatInds[xinds, yinds, zinds]
        unassigned_inds = array_of_indices[matTypes == None]
        if len(unassigned_inds) > 0:
            nbx = len(self.xBoundaries)
            nby = len(self.yBoundaries)
            nbz = len(self.zBoundaries)
            nbyz = nby*nbz            
            single_ids = np.int64(xinds)*nbyz + np.int64(yinds)*nbz + np.int64(zinds)
            try:
                uniques = pd.unique(single_ids)
            except:
                uniques = np.unique(single_ids)
            ynb_plus_z = uniques % nbyz
            zi = ynb_plus_z % nbz
            yi = (ynb_plus_z - zi)/nbz
            xi = (uniques - ynb_plus_z)/nbyz
            
            xi = np.int16(xi)
            yi = np.int16(yi)
            zi = np.int16(zi)   
            #uniques = np.array(list(uniques.keys()))
            #uniques = np.unique(unassigned_inds, axis=0)
            new_assignments = self.Rng.RngObj.choice(self.nummats, size=len(uniques), p=self.prob)            
            #self.MatInds[uniques[:,0], uniques[:,1], uniques[:,2]] = new_assignments
            self.MatInds[xi, yi, zi] = new_assignments
            matTypes = self.MatInds[xinds, yinds, zinds]
        return matTypes.astype(np.int16)

