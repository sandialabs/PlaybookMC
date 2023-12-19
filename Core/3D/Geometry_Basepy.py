#!usr/bin/env python
import sys
sys.path.append('/../../Core/Tools')
from RandomNumberspy import RandomNumbers
from Particlepy import Particle
from ClassToolspy import ClassTools
import numpy as np
import matplotlib.pyplot as plt

## \brief Multi-D base geometry class - useless by itself, but other multi-D geometry classes inherit from it
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
# Base class for multi-D geometries
#
class Geometry_Base(ClassTools):
    def __init__(self):
        super(Geometry_Base,self).__init__()
        self.flshowplot = False; self.flsaveplot = False
    
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
        if self.GeomType=='Markovian':
            if not ( xbounds[0]==ybounds[0] and xbounds[0]==zbounds[0] and xbounds[1]==ybounds[1] and xbounds[1]==zbounds[1] and -xbounds[0]==xbounds[1] ):
                raise Exception("PlaybookMC's current 3D Markovian geometry implementation requires a cubic domain centered at the origin")

        self.xbounds = xbounds
        self.ybounds = ybounds
        self.zbounds = zbounds

        self.Volume = self.zbounds[1]-zbounds[0]
        if self.Part.numDims>=2: self.Volume *= (self.xbounds[1]-xbounds[0])
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

    ## \brief Tests for a leakage or boundary reflection event, flags the first, evaluates the second
    #
    # \returns str 'reflect', 'transmit', 'sideleak', or 'noleak'
    def testForBoundaries(self):
        if self.isclose(self.Part.z,self.zbounds[0]):
            self.Part.z = self.zbounds[0]
            if   self.zBCs[0]==    'vacuum': return 'reflect'
            elif self.zBCs[0]=='reflective': self.Part.reflect_z()
        if self.isclose(self.zbounds[1],self.Part.z):
            self.Part.z = self.zbounds[1]
            if   self.zBCs[1]==    'vacuum': return 'transmit'
            elif self.zBCs[1]=='reflective': self.Part.reflect_z()

        if self.isclose(self.Part.x,self.xbounds[0]):
            self.Part.x = self.xbounds[0]
            if   self.xBCs[0]==    'vacuum': return 'sideleak'
            elif self.xBCs[0]=='reflective': self.Part.reflect_x()
        if self.isclose(self.xbounds[1],self.Part.x):
            self.Part.x = self.xbounds[1]
            if   self.xBCs[1]==    'vacuum': return 'sideleak'
            elif self.xBCs[1]=='reflective': self.Part.reflect_x()

        if self.isclose(self.Part.y,self.ybounds[0]):
            self.Part.y = self.ybounds[0]
            if   self.yBCs[0]==    'vacuum': return 'sideleak'
            elif self.yBCs[0]=='reflective': self.Part.reflect_y()
        if self.isclose(self.ybounds[1],self.Part.y):
            self.Part.y = self.ybounds[1]
            if   self.yBCs[1]==    'vacuum': return 'sideleak'
            elif self.yBCs[1]=='reflective': self.Part.reflect_y()

        return 'noleak'


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


    ## \brief Samples new point, chooses to accept or reject, and chooses to absorb or scatter
    #
    # \returns drives sampling of new points and returns str 'reject', 'absorb', or 'scatter'
    # need to change to allow for standard tracking or Woodcock instead of just Woodcock
    # pseudocode: if woodcock, sample whether or not it is accepted. If it's not woodcock,
    # we just want to sample absorb or scatter. Will also do that if collision is accepted for woodcock.
    def evaluateCollision(self):
        if self.trackingType == "Woodcock": # is there a way to tell that we are using woodcock tracking at this point in the code??? If not, I think I need two functions
            self.samplePoint() #define new point
            if self.Rng.rand()>self.totxs[self.CurrentMatInd]/self.Majorant: return 'reject' #accept or reject potential collision
        return 'absorb' if self.Rng.rand()>self.scatrat[self.CurrentMatInd] else 'scatter' #choose collision type


    ## \brief samples random point within defined geometry 
    #
    # \returns list of floats that are x, y, z values
    def _generateRandomPoint(self):
        x = self.Rng.uniform(self.xbounds[0],self.xbounds[1])
        y = self.Rng.uniform(self.ybounds[0],self.ybounds[1])
        z = self.Rng.uniform(self.zbounds[0],self.zbounds[1])        

        return [x,y,z]







    ## \brief Define mixing parameters (correlation length and material probabilities)
    # To be provided by non-base class
    def defineMixingParams(self,laminf=None,prob=None):
        assert 1==0

    ## \brief Initializes list of material types and coordinate of location, initializes xyz boundary locations
    # To be provided by non-base class
    def initializeGeometryMemory(self):
        assert 1==0

    ## \brief Samples a new point according to the selected rules
    # To be provided by non-base class
    def samplePoint(self):
        assert 1==0
