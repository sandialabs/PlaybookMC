#!usr/bin/env python
from Geometry_Basepy import Geometry_Base
import numpy as np
import math as math
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
class Geometry_Markovian(Geometry_Base):
    def __init__(self):
        super(Geometry_Markovian,self).__init__()
        self.flshowplot = False; self.flsaveplot = False
        self.GeomType = 'Markovian'
    
    def __str__(self):
        return str(self.__dict__)
    
    ## \brief Defines mixing parameters (correlation length and material probabilities)
    # Note: laminf and rhoP notation following LarmierJQSRT2017 different mixing types paper
    #
    # \param[in] laminf, float, correlation length of Markovian mixing
    # \param[in] prob, list of floats, list of material abundances (probabilities)
    # \returns sets corr length, mat. probs, and hyperplane density rhoP, calls computation of ave num of hyperplanes
    def defineMixingParams(self,laminf=None,prob=None):
        # Assert material chord lengths and slab length and store in object
        assert isinstance(laminf,float) and laminf>0.0
        assert isinstance(prob,list) and len(prob)==self.nummats
        for i in range(0,self.nummats): assert isinstance(prob[i],float) and prob[i]>=0.0 and prob[i]<=1.0
        assert self.isclose( np.sum(prob), 1.0 )
        self.laminf = laminf
        self.prob   = prob
        self.rhoP   = 1.0 / self.laminf


    ## \brief Initializes list of material types and coordinate of location, initializes xyz boundary locations
    #
    # Note: Numpy arrays don't start blank and concatenate, where lists do, therefore plan to use lists, but convert to arrays using np.asarray(l) if needed to use array format
    #
    # \returns initializes self.MatInds and self.xBoundaries, self.yBoundaries, self.zBoundaries
    def initializeGeometryMemory(self):
        self._sampleNumberOfHyperplanes()
        self._sampleHyperplaneLocations()

    ## \brief compute average number of hyper-planes
    #
    # \computes average number of hyper-planes used to sample number of planes (Poisson)
    def _sampleNumberOfHyperplanes(self):
        xLength = abs( self.xbounds[0]-self.xbounds[1] )
        yLength = abs( self.ybounds[0]-self.ybounds[1] )
        zLength = abs( self.zbounds[0]-self.zbounds[1] )
        assert xLength == yLength and xLength == zLength
        
        self.R = np.sqrt( xLength**2.0 + yLength**2.0 + zLength**2.0 )*0.5

        self.AveNumPlanes = 4.0 * self.rhoP * self.R
        self.numPlanes    = self.Rng.poisson(self.AveNumPlanes)
        #print(self.numPlanes, "planes sampled")


    ## \brief sample normal vector of hyper-plane
    #
    # \sample normal vector and compute x coordinate of point on plane 
    def _sampleHyperplaneLocations(self):
        self.Cells = {}
        self.r  = []
        self.n1 = []
        self.n2 = []
        self.n3 = []
        self.p1 = []
        for plane in range(0,self.numPlanes):
            rand1 = self.Rng.rand()
            rand2 = self.Rng.rand()

            r  = self.Rng.uniform(0.0,self.R)
            n1 = 1.0-2.0*rand1
            n2 = np.sqrt( 1.0-n1**2.0 ) * math.cos( 2.0*math.pi*rand2 )
            n3 = np.sqrt( 1.0-n2**2.0 ) * math.sin( 2.0*math.pi*rand2 )
            p1 = r/n1            

            self.r.append(r) 
            self.n1.append(n1)
            self.n2.append(n2)
            self.n3.append(n3)
            self.p1.append(p1)

    
    ## \brief Samples a new point according to the selected rules
    #
    # \returns scattering ratio of cell material type
    def samplePoint(self):
        #find index of nearest point
        self.x = self.Part.x
        self.y = self.Part.y
        self.z = self.Part.z

        keyCode = self._locateMarkovCell()

        #check if material has been sampled for specified cell. If not sample material type.
        if keyCode in self.Cells.keys():
            self.CurrentMatInd = self.Cells.get(keyCode)
        else:
            self.CurrentMatInd = int(self.Rng.choice(p=self.prob))
            self.Cells[keyCode] = self.CurrentMatInd

    ## \brief Samples a new point according to the selected rules
    #
    # \returns scattering ratio of cell material type
    def samplePointsFast(self, points):
        p1 = np.array(self.p1).reshape(-1,1)
        n1 = np.array(self.n1).reshape(-1,1)
        n2 = np.array(self.n2).reshape(-1,1)
        n3 = np.array(self.n3).reshape(-1,1)     
        
        relP1 = (points[:,0] - p1)
        relP2 = points[:,1].reshape(1,-1) #np.repeat(points[:,1].reshape((1,-1)), self.numPlanes, axis=0)
        relP3 = points[:,2].reshape(1,-1) #np.repeat(points[:,2].reshape((1,-1)), self.numPlanes, axis=0)
        
        n1s = np.repeat(n1, len(points), axis=1)
        
        #Using np.matmul instead of @ to keep things Python 2 compatible        
        location = np.multiply(n1s, relP1) + np.matmul(n2, relP2) + np.matmul(n3, relP3) 
        
        #print("Location matrix constructed, shape is ", location.shape)
        #Columns of location matrix correspond to distances from planes (rows of locations matrix)
        #Get unique keycodes indicating cells
        
        arr_bool = (location > 0).T
        rows = [tuple(k) for k in arr_bool]
        try:
            keycodes = pd.unique(rows)
        except:
            keycodes = np.unique(rows)
        #print("Got unique keycode entries, length is ",len(keycodes))
        #keycodes = np.array([k.__repr__() for k in keycodes])
        #keycodes = np.array(keycodes)
        
        #Find which keycodes have been assigned
        #assigned_cells_bool = np.in1d(keycodes, list(self.Cells.keys()))
        cells = list(self.Cells.keys())
        cellset = set(cells)
        keycodeset = set(keycodes)
        #Find assigned and unassigned keycodes
        assigned_cell_codes = cellset.intersection(keycodeset)       
        unassigned_cell_codes = keycodeset.difference(cellset)
        
        some_assigned = (len(assigned_cell_codes) > 0)
        some_unassigned = (len(unassigned_cell_codes) > 0)
        
        materials_dict = {} #np.zeros(len(rows))
        if some_assigned:            
            for k in assigned_cell_codes:
                materials_dict[k] = self.Cells[k]
        
        if some_unassigned:
            new_material_assignments = []
            for iassign in range(0,len(unassigned_cell_codes)):
                new_material_assignments.append( self.Rng.choice(p = self.prob) )
            for k,mat in zip(unassigned_cell_codes, new_material_assignments):
                self.Cells[k] = mat
                materials_dict[k] = mat          
        
        materials = [materials_dict[k] for k in rows]
        print("Full lookup complete")        
        return materials
           
    ## \brief determine relative location of particle with each plane
    #
    # \returns unique "key" code of cell defined by relative particle location
    def _locateMarkovCell(self):
        keyCode = "";
        for plane in range(0,self.numPlanes):
            relP1 = self.x - self.p1[plane]
            relP2 = self.y
            relP3 = self.z

            location = ( self.n1[plane]*relP1 ) + ( self.n2[plane]*relP2 ) + (self.n3[plane]*relP3)

            if   location >=0: keyCode += 'p'
            elif location < 0: keyCode += 'n'
                        
        return keyCode
