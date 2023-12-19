#!usr/bin/env python
from Geometry_Basepy import Geometry_Base
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

## \brief Multi-D Spherical Inclusion geometry class
# \author Emily Vu, evu@sandia.gov, emilyhvu@umich.edu
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
# \author Dan Bolintineanu, dsbolin@sandia.gov
#
# Class for multi-D SphericalInclusion geometry 
#
class Geometry_SphericalInclusion(Geometry_Base):
    def __init__(self,flVerbose=False):
        super(Geometry_SphericalInclusion,self).__init__()
        self.flshowplot = False; self.flsaveplot = False
        self.GeomType = 'SphericalInclusions'
        assert isinstance(flVerbose,bool)
        self.flVerbose = flVerbose
    
    def __str__(self):
        return str(self.__dict__)
    
    ## \brief Defines spherical inclusions mixing parameters
    #
    # \param[in] sphereFrac, float, target fraction of volume to be inclusions
    # \param[in] sizeDistribution, string; 'Constant','Uniform','Exponential' the distribution of radii (can add any of the distributions in scipy.stats, currently these three available)
    # \param[in] radMin, float default 0.0; minimum sphere/circle radius to sample
    # \param[in] radAve, float; numerical input, currently either radius of all spheres or free parameter to exponential distribution of radii
    # \param[in] radMax, float default np.inf; maimum sphere/circle radius to sample
    # \param[in] matSphProbs, list of floats; probability of each material type within spheres
    # \param[in] matMatrix, int; material index of matrix material
    # \returns sets several parameters to describe spherical/circlular inclusion sampling
    def defineMixingParams(self,sphereFrac,sizeDistribution,radMin,radAve,radMax,matSphProbs=None,matMatrix=None):
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

        assert isinstance(matSphProbs,list) and np.sum(matSphProbs)==1.0
        for prob in matSphProbs: assert isinstance(prob,float) and prob>=0.0 and prob<=1.0
        assert isinstance(matMatrix,int) and matMatrix>=0
        self.matSphProbs = matSphProbs
        self.matMatrix   = matMatrix


    ## \brief Samples spherical inclusion realiation
    #
    # Radii sampled first, then spheres placed in decending order of size (helps ensure placement)
    #
    # \param[in] maxPlacementAttempts, int; maximum number of times to try to place a sphere
    # \returns initializes self.Points and self.MatInds 
    def initializeGeometryMemory(self,maxPlacementAttempts=1000):
        assert isinstance(maxPlacementAttempts,int) and maxPlacementAttempts>0

        # sample new spheres/circles until desired volume fraction met
        totSphVol  = 0.0
        self.Radii = []
        while True:
            flbreak  = False                                            #variable to end loop when True
            [radius] = self.distribution.rvs(1, random_state = self.Rng.randint(1,100000) ) #sample radius
            if radius < self.radMin or radius > self.radMax: continue #reject if larger or smaller than user allowing

            newSphVol = 4/3*np.pi*np.cumsum(radius**3) #compute new sphere/circle volume/area

            if (totSphVol+newSphVol)/self.Volume > self.sphereFrac:     #volume fraction exceeded, handle last sphere/circle
                flbreak = True
                if self.sizeDistribution in {'Uniform','Exponential'}:  #set last radius to size needed to get target volume fraction as in BrantleyMC2011; if 'Constant', accept last sphere as in BrantleyMC2011
                    volRemain = self.sphereFrac*self.Volume - totSphVol     #compute volume remaining
                    radius = (3/4*volRemain/np.pi)**(1/3)

            totSphVol += newSphVol
            self.Radii.append( float(radius) )                                      #add radius to list
            if flbreak: break

        # sort spheres/circles with larger first
        np.sort( self.Radii )

        # sample spheres/circles that are non-overlapping with each other and boundary
        self.Centers = []
        for isph,radius in enumerate( self.Radii ):
            nattempts = 0
            if self.flVerbose: print(f"Attempting to place sphere {isph} of {len(self.Radii)}, radius = {radius}...")
            while True:
                nattempts += 1
                flreject   = False

                # sample new sphere/circle center
                x = self.Rng.uniform(self.xbounds[0]+radius,self.xbounds[1]-radius)                
                y = self.Rng.uniform(self.ybounds[0]+radius,self.ybounds[1]-radius)                
                z = self.Rng.uniform(self.zbounds[0]+radius,self.zbounds[1]-radius)

                # reject newly sampled sphere if overlaps with previously sampled sphere
                for iold,center in enumerate( self.Centers ):
                    dist = np.sqrt( (center[0]-x)**2 + (center[1]-y)**2 + (center[2]-z)**2 )
                    if radius+self.Radii[iold]>dist:
                        flreject = True
                        break
                # reject sampled sphere location
                if flreject:
                    if nattempts > maxPlacementAttempts: raise Exception("Failed to place particle {isph} with radius {radius}")
                    continue
                
                # store sphere center in list
                self.Centers.append([x,y,z])
                if self.flVerbose: print(f"Successfully placed after {nattempts} attempts")
                break

        if self.flVerbose: print(f"Placed a total of {len(self.Centers)} spheres/circles, resulting in a final volume fraction of {totSphVol/self.Volume}")

        # sample what material in each sphere according to user specified average material abundance in spheres
        self.MatInds   = []
        for ibody in range(0,len(self.Radii)):
            self.MatInds.append( self.Rng.choice(p=self.matSphProbs) )


    ## \brief Returns material at specified location
    #
    # \returns material type of location
    def samplePoint(self):
        # search whether point in a sphere
        for ibody,center in enumerate( self.Centers ):
            dist = np.sqrt( (center[0]-self.Part.x)**2 + (center[1]-self.Part.y)**2 + (center[2]-self.Part.z)**2 )
            if self.Radii[ibody]>dist:
                self.CurrentMatInd = self.MatInds[ibody]; return
        # if not in a circle, then in matrix material
        self.CurrentMatInd = self.matMatrix; return







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
        if self.flsaveplot == True: plt.savefig(''+str(self.geomtype)+'_'+str(self.Part.numDims)+'D_Case-name-_Part'+str(ipart+1)+'.png')
        if self.flshowplot == True: plt.show()