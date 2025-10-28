#!usr/bin/env python
import sys
sys.path.append('/../../Classes/Tools')
from RandomNumberspy import RandomNumbers
import numpy as np
## \brief Class for multidimensional particle
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
class Particle(object):
    def __init__(self):
        pass

    def __str__(self):
        return str(self.__dict__)

    ## \brief Define dimensionality of transport, i.e., 1D, 2D, or 3D
    #
    # \param[in] dimensionality str, '1D', '2D', or '3D'
    # \returns initializes self.numDims int, 1,2,or 3
    def defineDimensionality(self,dimensionality=None):
        assert isinstance(dimensionality,str)
        assert dimensionality in {'1D','2D','3D'}
        if   dimensionality=='1D': self.numDims = 1
        elif dimensionality=='2D': self.numDims = 2
        elif dimensionality=='3D': self.numDims = 3

    ## \brief Define source range
    #
    # 'cuboid' source - Specifically referring to a rectangular cuboid in which particles are uniformly 
    # distributed.  Must define x, y, z ranges.  By selecting x, y, and/or z ranges in which the largest and
    # smallest extents are the same, can yield point, line, or plane sources.
    # 
    # 'annulus' source - Source defined over a subset of a disk within one or more regions defined by radii.
    # The source strength of each region is input and particles are uniformly sampled from within each region.
    # The user-input CDF is converted into a PDF for internal use.  The annulus source is normal to the z axis.
    # 
    # \param[in] sourceLocationType str, default 'cuboid'; 'cuboid' or 'annulus'
    # \param[in] xrange,yrange,zrange lists of two floats, upper and lower bounds of location ranges from which to initilize particles
    # \param[in] annulusCenter list of 3 floats, center location of annulus source
    # \param[in] annulusCDF, list of floats, cumulative distribution function of annulus region source strengths
    # \param[in] annulusRadii, list of floats, outer radii of each region in disk organized innermost to outermost
    # \returns initializes self.sourceLocationType, self.xrange, self.yrange, self.zrange, self.annulusCenter, self.annulusPDF, self.annulusRadii
    def defineSourcePosition(self,sourceLocationType='cuboid',xrange=None,yrange=None,zrange=None,annulusCenter=None,annulusCDF=None,annulusRadii=None):
        assert sourceLocationType=='cuboid' or sourceLocationType=='annulus'
        self.sourceLocationType = sourceLocationType

        if sourceLocationType=='cuboid':
            assert isinstance(xrange,list) and isinstance(yrange,list) and isinstance(zrange,list)
            assert len(xrange)==2 and len(yrange)==2 and len(zrange)==2
            assert isinstance(xrange[0],float) and isinstance(xrange[1],float) and xrange[0]<=xrange[1]
            if self.numDims<2: assert xrange[0]==0.0 and xrange[1]==0.0
            assert isinstance(yrange[0],float) and isinstance(yrange[1],float) and yrange[0]<=yrange[1]
            if self.numDims<3: assert yrange[0]==0.0 and yrange[1]==0.0
            assert isinstance(zrange[0],float) and isinstance(zrange[1],float) and zrange[0]<=zrange[1]

            self.xrange = xrange
            self.yrange = yrange
            self.zrange = zrange

        if sourceLocationType=='annulus':
            assert isinstance(annulusCenter,list)
            assert isinstance(annulusCenter[0],float) and isinstance(annulusCenter[1],float) and isinstance(annulusCenter[2],float)

            assert isinstance(annulusCDF,list) and isinstance(annulusRadii,list)
            assert len(annulusCDF)==len(annulusRadii)
            assert isinstance(annulusCDF[0],float) and isinstance(annulusRadii[0],float)
            assert annulusCDF[0]>=0.0 and annulusRadii[0]>=0.0
            self.annulusPDF = [ annulusCDF[0] ]
            for iregion in range(0,len(annulusCDF)-1):
                assert isinstance(annulusCDF[iregion+1],float) and isinstance(annulusRadii[iregion+1],float)
                assert annulusCDF[iregion]   <= annulusCDF[iregion+1]
                assert annulusRadii[iregion] <= annulusRadii[iregion+1]
                self.annulusPDF.append( annulusCDF[iregion+1] - annulusCDF[iregion] )
            assert annulusCDF[-1]==1.0

            self.annulusCenter = annulusCenter[:]
            self.annulusRadii  = [0.0] + annulusRadii[:]


    ## \brief Define source angular rule and/or angle
    #
    # 'beam' - Particles are monodirectional in the positive x direction.
    # 'boundary-isotropic' - Also known as a 'cosine-law' source, particles are distributed proportionally 
    #    to the cosine of the angle with respect to a reference direction, which is here always the
    #    positive z direction. This source models the boundary source for an external uniform isotropic
    #    flux field.
    # 'internal-isotropic' - Particles are distributed uniformly over all angles.
    #
    # \param[in] sourceAngleType str, 'beam', 'boundary-isotropic' or 'internal-isotropic'
    # \returns initializes self.sourceAngleType
    def defineSourceAngle(self,sourceAngleType=None):
        assert sourceAngleType=='beam' or sourceAngleType=='boundary-isotropic' or sourceAngleType=='internal-isotropic'
        self.sourceAngleType = sourceAngleType

    ## \brief Define particle scattering behavior
    #
    # 'isotropic-1Drod' - uniformly choose from positive or negative x direction
    # 'isotropic-2Drod' - uniformly choose from positive or negative x and y directions
    # 'isotropic-2Drod' - uniformly choose from positive or negative x, y, and z directions
    # 'isotropic' - uniformly distributed over all angles
    #
    # \param[in] scatteringtype str, 'isotropic-1Drod' 'isotropic-2Drod' 'isotropic-3Drod' 'isotropic'
    # \returns initializes self.ScatteringType
    def defineScatteringType(self,scatteringtype=None):
        assert scatteringtype in ('isotropic-1Drod','isotropic-2Drod','isotropic-3Drod','isotropic')
        self.ScatteringType = scatteringtype

    ## \brief Associates random number object - must be instance of RandomNumbers class
    #
    # \param[in] Rng RandomNumbers object
    # \returns initializes self.Rng
    def associateRng(self,Rng):
        assert isinstance(Rng,RandomNumbers)
        self.Rng = Rng


    ## \brief Sets directional variables based on current angles of travel
    #
    # \param self.mu, self.phi
    # \returns sets self.u, self.v, self.w
    def _set_uvw(self):
        sqrtterm = np.sqrt(1.0-self.mu**2)
        self.u = sqrtterm * np.cos(self.phi)
        self.v = sqrtterm * np.sin(self.phi)
        self.w = self.mu

    ## \brief Initializes particle based on choices specified
    #
    # \param[in] Rng RandomNumbers object
    # \returns sets self.x, self.y, self.z, self.weight, self.mu
    def initializeParticle(self):
        self.weight = 1.0

        if   self.sourceLocationType=='cuboid':
            self.x = self.xrange[0] + self.Rng.rand() * ( self.xrange[1] - self.xrange[0] )
            self.y = self.yrange[0] + self.Rng.rand() * ( self.yrange[1] - self.yrange[0] )
            self.z = self.zrange[0] + self.Rng.rand() * ( self.zrange[1] - self.zrange[0] )
        elif self.sourceLocationType=='annulus':
            theta  = 2.0 * np.pi * self.Rng.rand()
            region = self.Rng.choice(self.annulusPDF)
            inradius = self.annulusRadii[region]
            outradius= self.annulusRadii[region+1]
            radius = np.sqrt( self.Rng.rand() * (outradius**2 - inradius**2) + inradius**2)
            self.x = self.annulusCenter[0] + radius * np.cos(theta)
            self.y = self.annulusCenter[1] + radius * np.sin(theta)
            self.z = self.annulusCenter[2]

        if   self.sourceAngleType=='beam'              :
            self.mu  = 1.0
            self.phi = 0.0
        elif self.sourceAngleType=='boundary-isotropic':
            self.mu  = np.sqrt( self.Rng.rand() )
            self.phi = 2.0 * np.pi * self.Rng.rand()
        elif self.sourceAngleType=='internal-isotropic'         :
            self.mu  = 2.0 * self.Rng.rand()-1.0
            self.phi = 2.0 * np.pi * self.Rng.rand()
        self._set_uvw()
        

    ## \brief Calculates distance to collision
    #
    # \param[in] totxs float, value to compute distance to collision (or potential collision) with
    # \returns float, distance to collision
    def calculateDistanceToCollision(self,totxs):
        return -np.log( self.Rng.rand() ) / totxs

    ## \brief Computes new particle position based on streaming operation
    #
    # \param[in] dist float, distance particle streams along current path
    # \returns sets self.x, self.y, self.z
    def streamParticle(self,dist):
        if self.numDims>1: self.x = self.x + dist * self.u
        if self.numDims>2: self.y = self.y + dist * self.v
        self.z = self.z + dist * self.w

    ## \brief Chooses new angle of travel based on selected scattering rule
    #
    # \returns sets self.mu, angle relative to x of travel
    def scatterParticle(self):
        if  self.ScatteringType=='isotropic-1Drod': #travel only in z
            self.mu  = 1.0 if self.Rng.rand()<0.5 else -1.0
            self.phi = 0.0
        elif self.ScatteringType=='isotropic-2Drod': #travel directions in either only x or only z
            self.mu, self.phi = [ (1.0,0.0), (-1.0,0.0), (0.0,0.0), (0.0,np.pi) ][self.Rng.choice([0.25,0.25,0.25,0.25])] #choose and return a tuple from this list of four options
        elif self.ScatteringType=='isotropic-3Drod': #travel directions in either only x, only y, or only z
            self.mu, self.phi = [ (1.0,0.0), (-1.0,0.0), (0.0,0.0), (0.0,np.pi), (0.0,0.5*np.pi), (0.0,1.5*np.pi) ][self.Rng.choice([1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6])] #choose and return a tuple from this list of six options
        elif self.ScatteringType=='isotropic'    :
            self.mu  = 2.0 * self.Rng.rand()-1.0
            self.phi = 2.0 * np.pi * self.Rng.rand()
        self._set_uvw()

    ## \brief Reverses particle in x direction
    #
    # \returns sets self.u
    def reflect_x(self):
        self.u = - self.u

    ## \brief Reverses particle in y direction
    #
    # \returns sets self.v
    def reflect_y(self):
        self.v = - self.v

    ## \brief Reverses particle in z direction
    #
    # \returns sets self.w
    def reflect_z(self):
        self.w = - self.w
