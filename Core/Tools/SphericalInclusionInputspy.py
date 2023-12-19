#!usr/bin/env python
import numpy as np
## \brief Accesses spherical inclusion stochastic media benchmark inputs and solves mixing parameters from others
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
class SphericalInclusionInputs(object):
    def __init__(self):
        pass
    def __str__(self):
        return str(self.__dict__)

    ## \brief Solve for spherical inclusion parameters based on sphere average chord lengths and sphere fraction
    #
    # \param[in] sphereFrac float, domain volume fraction in spherical inclusions
    # \param[in] sphereLam float, average chord lengths in spheres
    # \returns solves for/sets self.sphereFrac, self.sphereLam, and matrixLam
    def solveSphericalInclusionParamsBasedOnSphereAveChordLength(self,sphereFrac,sphereLam):
        assert isinstance(sphereLam,float) and sphereLam>0.0
        assert isinstance(sphereFrac,float) and sphereFrac>=0.0 and sphereFrac<=1.0

        self.sphereFrac = sphereFrac
        self.sphereLam  = sphereLam
        self.matrixLam  = sphereLam * ( 1/sphereFrac - 1 )

    ## \brief Solves Brantley and Martos spherical inclusion benchmark inputs.
    #
    # Solves and stores as attributes BM input params based on case designations used in BrantleyMC2011.
    #
    # \param[in] CaseNumber str, number designation, i.e., '1', '2', '3'
    # \param[in] CaseVolFrac str, volume fraction, i.e., '0.05','0.10','0.15','0.20','0.25','0.30'
    # \param[in] sizeDistribution, string; 'Constant','Uniform','Exponential' the distribution of radii (can add any of the distributions in scipy.stats, currently these three available)
    # \returns solves for and sets Sigt, Scatrat, Sigs, matSphProbs, matMatrix, sphereFrac, lam, radAve, radMin, radAve, and radMax as attributes 
    def selectBMInputs(self, CaseNumber=None, CaseVolFrac=None, sizeDistribution=None ):
        assert isinstance(CaseNumber,str)
        assert CaseNumber in {'1','2','3'}
        assert isinstance(CaseVolFrac,str)
        assert CaseVolFrac in {'0.05','0.10','0.15','0.20','0.25','0.30'}
        assert sizeDistribution in {'Constant','Uniform','Exponential'}
        self.sizeDistribution = sizeDistribution

        self.Sigt    = [10.0/99.0,100.0/11.0]
        self.Scatrat = [0.9      ,0.9       ]
        self.Sigs = list( np.multiply( self.Sigt, self.Scatrat ) )

        self.matSphProbs = [0.0,1.0] #this means that there is 100% probability each sphere contains material with index one
        self.matMatrix   = 0         #this means that material with index zero is the matrix material

        if   CaseNumber=='1': self.solveSphericalInclusionParamsBasedOnSphereAveChordLength(sphereFrac=float(CaseVolFrac),sphereLam=11/40)
        elif CaseNumber=='2': self.solveSphericalInclusionParamsBasedOnSphereAveChordLength(sphereFrac=float(CaseVolFrac),sphereLam=11/20)
        elif CaseNumber=='3': self.solveSphericalInclusionParamsBasedOnSphereAveChordLength(sphereFrac=float(CaseVolFrac),sphereLam=11/10)

        if   sizeDistribution=='Constant'   : self.radAve = 3/4*self.sphereLam; self.radMin = self.radAve; self.radMax = self.radAve
        elif sizeDistribution=='Uniform'    : self.radAve = 1/2*self.sphereLam; self.radMin = 0.0        ; self.radMax = self.radAve * 2
        elif sizeDistribution=='Exponential': self.radAve = 1/4*self.sphereLam; self.radMin = 0.0        ; self.radMax = 10/6