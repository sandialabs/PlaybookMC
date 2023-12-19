#!usr/bin/env python
import sys
sys.path.append('/../../Classes/Tools')
from ClassToolspy import ClassTools
import numpy as np
import time
import matplotlib.pyplot as plt

## \brief Solves radiation transport quantities on 1D slab using MC method.
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
# Class for performing Monte Carlo particle transport.
# Class can perform Monte Carlo on a slab with defined material boundaries
# or using Woodcock Monte Carlo.
class MonteCarloParticleSolver(ClassTools):
    def __init__(self):
        super(MonteCarloParticleSolver,self).__init__()
        self.flTally             = False

    def __str__(self):
        return str(self.__dict__)

    ## \brief Enables user to set source angle and location
    #
    # 'beam' - Particles are monodirectional in the positive x direction.
    # 'boundary-isotropic' - Also known as a 'cosine-law' source, particles are distributed proportionally 
    #    to the cosine of the angle with respect to a reference direction, which is here always the
    #    positive x direction. This source models the boundary source for an external uniform isotropic
    #    flux field.
    # 'internal-isotropic' - Particles are distributed uniformly over all angles.
    #
    # \param[in] sourceType str, 'boundary-isotropic' or 'internal-isotropic' or 'beam'
    # \param[in] sourceLocationRange list of two floats, default [0.0,self.SlabLength], location bounds of uniformly distributed source
    # \param[in] sourceAngleOfIncidence float, default 1.0 (normally incident), cosine of angle of beam incidence
    # \returns sets self.sourceType, self.sourceLocationRange, and self.sourceAngleOfIncidence
    def defineSource(self,sourceType=None,sourceLocationRange=None,sourceAngleOfIncidence=None):
        assert sourceType=='boundary-isotropic' or sourceType=='internal-isotropic' or sourceType=='beam'
        if sourceType=='beam':
            if sourceAngleOfIncidence==None: sourceAngleOfIncidence = 1.0
            assert isinstance(sourceAngleOfIncidence,float)
            assert sourceAngleOfIncidence > 0.0 and sourceAngleOfIncidence <= 1.0
        if sourceType=='internal-isotropic':
            if sourceLocationRange==None: sourceLocationRange=[0.0,self.SlabLength]
            assert isinstance(sourceLocationRange,list) and len(sourceLocationRange)==2
            assert isinstance(sourceLocationRange[0],float) and isinstance(sourceLocationRange[1],float)
            assert self.isleq(0.0,sourceLocationRange[0]) and self.isleq(sourceLocationRange[0],sourceLocationRange[1]) and self.isleq(sourceLocationRange[1],self.SlabLength)
        self.sourceType             = sourceType
        self.sourceLocationRange    = sourceLocationRange
        self.sourceAngleOfIncidence = sourceAngleOfIncidence
        
    ## \brief Sets a link to instantiation of OneDSlab
    #
    # Links geometry to object and chooses either public or semi-private version of 'samplePoint'.
    # Could be generalized to accept other functions, but is not at this time.
    #
    # \param[in] slab object, instantiation of OneDSlabGeometry
    def defineMCGeometry(self,slab,slablength=None):
        self.Slab = slab
        assert callable(self.Slab.samplePoint_x) and callable(self.Slab.samplePoint_iseg)
        self.XSFunction   = self.Slab.samplePoint_x
        assert isinstance(slablength,float) and slablength>0.0
        self.SlabLength   = slablength
        self.GeomType     = 'MC'

    ## \brief Define a parameters needed for Woodcock Monte Carlo (WMC) transport.
    #
    # Defines a WMC total cross section ceiling, cross section function, and the arguments
    # needed to call the function.  The ceiling must be as large as or larger than the largest
    # total cross section which can be called using the function.  The function must be able to
    # return values for cross section types 'total', 'scatter', 'absorb', and 'scatrat', and
    # these values must be consistant with their meanings (at least at x=0.0).  The parameters
    # passed must pass some tests including calling the function for each cross section type at x=0.0.
    #
    # \param[in] ceiling float, ceiling value for WMC transport
    # \param[in] xsfunction CallableObject, function that returns cross section values at locations 'x'
    # \param     args arguments needed in xsfunction, e.g., (RealizationNumber,NumOfEigModes)
    def defineWMCGeometry(self,xsceiling=None,xsfunction=None,slablength=None,*args):
        #Make assertions on input, then make ceiling, function, and args pass some basic tests.
        assert isinstance(xsceiling,float) and xsceiling>0.0
        assert callable(xsfunction)
        assert isinstance(slablength,float) and slablength>0.0
        try:    totalxs  = xsfunction(0.0,'total',args)
        except: print('WMCGeometry function fail to call total cross section value'); raise
        try:    scatterxs = xsfunction(0.0,'scatter',args)
        except: print('WMCGeometry function fail to call scattering cross section value'); raise
        try:    absorbxs  = xsfunction(0.0,'absorb',args)
        except: print('WMCGeometry function fail to call absorption cross section value'); raise
        try:    scatrat   = xsfunction(0.0,'scatrat',args)
        except: print('WMCGeometry function fail to call scattering ratio value'); raise
        assert totalxs<=xsceiling
        assert scatterxs<=totalxs
        assert absorbxs<=totalxs and absorbxs+scatterxs==totalxs
        eps = 0.0000000001
        if not totalxs==0.0: assert scatterxs/totalxs<scatrat+eps and scatterxs/totalxs>scatrat-eps
        #Store function and parameters as attributes
        self.XSCeiling    = xsceiling
        self.XSFunction   = xsfunction
        self.SlabLength   = slablength
        self.args         = args
        self.GeomType     = 'WMC'

    ## \brief Associates FluxTallyObject with MC solver object for taking tallies.
    #
    # The two objects are associated in that the MC solver object can
    # call functions in the FluxTalObj, and the FluxTalObj can be called
    # independently--both will operate on the same object.
    #
    # Function also sets flag indicating to use flux tallies.  Maybe use
    # of this flag can be overwritten by a test for a FluxTalObj.
    #
    # \param[in] FluxTallyObject 'FluxTallies' object, instantiated flux tally object from PBMC
    # \returns sets object to self.FluxTalObj, sets self.flTally to True
    def associateFluxTallyObject(self,FluxTallyObject):
        self.FluxTalObj = FluxTallyObject
        self.flTally = True


    ## \brief Initializes internal random number object.
    #
    # Random numbers for Monte Carlo computation here use an instance of
    # numpy.random.RandomState.  The user can let numpy choose where in the pseudo-random
    # number chain to select numbers, or can specify.  If the user specifies (flUseSeed==True),
    # the sequence is re-seeded at the beginning of each particle history based on the initial
    # seed, the particle number, and the stride (i.e., localseed = seed + particlenum * stride)
    # This makes calcluations repeatable even when done in parts or in parallel.
    #
    # \param[in] flUseSeed bool, use specified seed and thus be repeatable (or let numpy choose seed
    #                           and be different each time?)
    # \param seed int, random number seed for random number sequence
    # \param stride int, perturbation of seed for each particle history
    def initializeRandomNumberObject(self,flUseSeed=None,seed=None,stride=None):
        assert isinstance(flUseSeed,bool)
        if seed==None  : seed   = 938901
        if stride==None: stride = 267
        assert isinstance(seed,int) and isinstance(stride,int)
        self.flUseSeed = flUseSeed
        self.Seed      = seed
        self.Stride    = stride
        self.Rng       = np.random.RandomState()


    ## \brief Initializes/resets tally of particles simulated, total time, and leakage tallies.
    #
    # \param sets self.NumOfParticles, self.TotalTime, self.T, self.R, and self.A to 0
    def _initializeInternalTallies(self):
        self.NumOfParticles = 0
        self.TotalTime      = 0.0
        self.T              = 0
        self.R              = 0
        self.A              = 0

    ## \brief If chosen, initializes random number seed before each history. Intent: private.
    #
    # Is called at beginning of each particle history.  If user chose to use seeds,
    # the history-specific seed is set.
    #
    # \param[in] ipart int, particle history number
    # \returns sets Rng attribute seed for particular particle history
    def _setRandomNumberSeed(self,ipart):
        if self.flUseSeed: self.Rng.seed( self.Seed + self.Stride * ipart )


    ## \brief Is the main driver for 'pushing' particles for MC or WMC solve.
    #
    # Makes some assertions to help ensure that options have been set to and
    # that the code is ready for the user to push particles.
    # Uses the chosen options and loops through the number of new particle histories,
    # calling the appropriate method to 'push' each history.
    #
    # \param[in] NumNewParticles int, number of particles to solve and add to any previously solved
    # \param[in] NumParticlesUpdateAt int, number of particles to print runtime updates at
    # \param flCompFluxQuants bool, default True, compute flux quantities, e.g., uncertainty and FOM?
    # \param[in] initializeTallies str, default 'first_hist'; 'first_hist' or 'no_hist' behavior to initialize/reset leakage tallies
    def pushParticles(self,NumNewParticles=None,NumParticlesUpdateAt=None,flCompFluxQuants=True,initializeTallies='first_hist'):
        assert isinstance(NumNewParticles,int) and NumNewParticles>0
        self.NumNewParticles = NumNewParticles
        if not NumParticlesUpdateAt==None: assert isinstance(NumParticlesUpdateAt,int) and NumParticlesUpdateAt>0
        self.NumParticlesUpdateAt = NumParticlesUpdateAt
        if isinstance(self.NumParticlesUpdateAt,int): self.printTimeUpdate = self._printTimeUpdate
        else                                        : self.printTimeUpdate = self._printNoTimeUpdate
        assert callable(self.XSFunction)
        assert isinstance(flCompFluxQuants,bool)
        assert initializeTallies=='first_hist' or initializeTallies=='no_hist'
        if   self.GeomType=='MC' : fPushParticle = self._pushMCParticle
        elif self.GeomType=='WMC': fPushParticle = self._pushWMCParticle
        self.tstart = time.time()
        if initializeTallies=='first_hist': self._initializeInternalTallies()
        for ipart in range(self.NumOfParticles,self.NumOfParticles+self.NumNewParticles):
            self._setRandomNumberSeed(ipart)
            if self.flTally: self.FluxTalObj._setRandomNumberSeed(ipart)
            if self.flTally: self.FluxTalObj._initializeHistoryTallies()
            fPushParticle()
            if self.flTally: self.FluxTalObj._foldHistoryTallies()
            self.printTimeUpdate(ipart)
        self.TotalTime      += time.time() - self.tstart
        self.NumOfParticles += self.NumNewParticles
        if self.flTally and flCompFluxQuants: self.FluxTalObj._computeFluxQuantities(self.TotalTime,self.NumOfParticles)

    ## \brief Dummy time update method. Intent: private.
    #
    # If time updates are not set, the time update method calls this method which does nothing.
    # \returns nothing
    def _printNoTimeUpdate(self,ipart):
        pass

    ## \brief Prints history number and time update including calculation ETA. Intent: private.
    #
    # If time updates are chosen, the time update method calls this method which returns the number
    # of particle histories run, the total number of histories, the runtime so far, and an estimate
    # of the total runtime.
    # \returns prints an update on runtime with estimate of total runtime
    def _printTimeUpdate(self,ipart):
        if (ipart+1) % self.NumParticlesUpdateAt == 0:
            totparts= self.NumOfParticles+self.NumNewParticles
            tottime = ( self.TotalTime + time.time() - self.tstart ) /60.0
            print('{}/{} histories, {:.2f}/{:.2f} min'.format(ipart+1,totparts,tottime,tottime/float(ipart+1)*totparts))

    ## \brief Returns source angle based on type of user choice
    #
    # \returns sampled source angle
    def _setParticleSourceAngle(self):
        if   self.sourceType=='boundary-isotropic': return np.sqrt( self.Rng.rand() )     #isotropic source at boundary
        elif self.sourceType=='internal-isotropic': return 2.0*self.Rng.rand()-1.0        #internal source
        elif self.sourceType=='beam'              : return self.sourceAngleOfIncidence    #beam source
        
    ## \brief initializes position of streaming particle
    #
    # \returns sets x
    def _initializePosition(self):
        if self.sourceType=='internal-isotropic': return self.Rng.uniform( self.sourceLocationRange[0],self.sourceLocationRange[1] )
        else                                    : return 0.0
        
    ## \brief Simulates one particle using slab-based Monte Carlo solver. Intent: private.
    #
    # Simulates one particle history using slab geometry defined by 'Slab' from 'OneDSlabpy.py'
    # and tallies transmittance, reflectance, and absorption.
    def _pushMCParticle(self):
        mu   = self._setParticleSourceAngle()
        x    = self._initializePosition()
        iseg = int( np.digitize( x, self.Slab.matbound ) ) - 1

        flbreak = False
        while True:
            oldx = x; oldiseg = iseg; oldmu = mu
            #solve distance to collision
            totxs= self.Slab.samplePoint_iseg(iseg,'total')
            dc= -np.log( self.Rng.rand() ) / totxs * mu
            #solve distance to boundary
            db = self.Slab.matbound[iseg+1] - x if mu >=0.0 else self.Slab.matbound[iseg] - x
            #stream particle, choose collision or boundary crossing
            if  self.isleq(abs(db),abs(dc)): dx = db; flcollide = False
            else                           : dx = dc; flcollide = True
            x = x + dx
            #evaluate collision or boundary crossing
            if flcollide:  #evaluate collision
                if self.Rng.rand()> self.Slab.samplePoint_iseg(iseg,'scatrat'):
                    self.A+=1; flbreak = True    #absorb
                else                                                          :
                    mu = 2.0*self.Rng.rand()-1.0 #scatter
            else        :  #evaluate boundary crossing
                if mu<0.0:
                    if iseg==0                       : self.R+=1; flbreak = True #tally R
                    x = self.Slab.matbound[iseg]
                    iseg -= 1
                else     :
                    if iseg==len(self.Slab.mattype)-1: self.T+=1; flbreak = True #tally T
                    iseg += 1
                    x = self.Slab.matbound[iseg]
            #tally flux
            if self.flTally: self.FluxTalObj._tallyFlux( oldx,x,oldmu,oldiseg,collinfo=(flcollide,totxs) )
            #if particle absorbed or leaked, kill history
            if flbreak: break

    ## \brief Simulates one particle using function-based Woodcock Monte Carlo solver. Intent: private.
    #
    # Simulates one particle history using geometry defined by a function which returns
    # necessary radiation transport quantities.
    # Currently uses normally incident beam source on left boundary and isotropic scattering
    # and tallies transmittance, reflectance, and absorption.
    def _pushWMCParticle(self):
        mu = self._setParticleSourceAngle()
        x  = self._initializePosition()

        flbreak = False
        while True:
            oldx = x; oldmu = mu
            #solve distance to potential collision
            dpc= -np.log( self.Rng.rand() ) / self.XSCeiling * mu
            #solve distance to boundary
            db = self.SlabLength-x if mu>=0.0 else -x
            #choose potential collision or boundary crossing
            if  self.isleq(abs(db),abs(dpc)): dx = db ; flcollide = False
            else                            : dx = dpc; flcollide = True
            #stream particle
            x = x + dx
            #evaluate collision or boundary crossing
            if flcollide:  #evaluate potential collision
                if self.Rng.rand() < self.XSFunction(x,'total',self.args)/self.XSCeiling:
                    #------#evaluate accepted collision
                    if self.Rng.rand()> self.XSFunction(x,'scatrat',self.args):
                        self.A+=1; flbreak = True    #absorb
                    else                                                      :
                        mu = 2.0*self.Rng.rand()-1.0 #scatter
            else        :  #evaluate boundary crossing (always leakage event)
                if   self.isclose(x,0.0)            : self.R+=1; x = 0.0            ; flbreak = True #tally R
                elif self.isclose(x,self.SlabLength): self.T+=1; x = self.SlabLength; flbreak = True #tally T
                else                                : assert(False)
            #tally flux
            if self.flTally: self.FluxTalObj._tallyFlux( oldx,x,oldmu,iseg=None,collinfo=(flcollide,self.XSCeiling) )
            #if particle absorbed or leaked, kill history
            if flbreak: break

            

    ## \brief Veneer for '_returnParticleKillMoments' that passes transmittance tally information
    #
    # \param flVerbose bool, default False, print to screen the data to return?
    # \returns ave, stdev, SEM, and FOM of transmittance
    def returnTransmittanceMoments(self,flVerbose=False):
        assert isinstance(flVerbose,bool)
        return self._returnParticleKillMoments(flVerbose,'Transmittance   ',self.T)

    ## \brief Veneer for '_returnParticleKillMoments' that passes reflectance tally information
    #
    # \param flVerbose bool, default False, print to screen the data to return?
    # \returns ave, stdev, SEM, and FOM of reflectance
    def returnReflectanceMoments(self,flVerbose=False):
        assert isinstance(flVerbose,bool)
        return self._returnParticleKillMoments(flVerbose,'Reflectance     ',self.R)

    ## \brief Veneer for '_returnParticleKillMoments' that passes absorption tally information
    #
    # \param flVerbose bool, default False, print to screen the data to return?
    # \returns ave, stdev, SEM, and FOM of absorption
    def returnAbsorptionMoments(self,flVerbose=False):
        assert isinstance(flVerbose,bool)
        return self._returnParticleKillMoments(flVerbose,'Absorption      ',self.A)

    ## \brief Computes and returns information on tallied transmittance, reflectance, or absorption.
    #
    # Computes mean, standard deviation, standard error of the mean (Monte Carlo uncertainty),
    # and figure-of-merit, prints them if chosen, and returns these values.
    #
    # \param flVerbose bool, default False, print to screen the data to return?
    # \param partkilltype str, used in part of the print message
    # \param tally int, tally of kill type of interest.
    #
    # \returns ave, stdev, SEM, and FOM of tally
    def _returnParticleKillMoments(self,flVerbose,partkilltype,tally):
        ave = float(tally)/self.NumOfParticles
        dev = np.sqrt( ave - ave**2 ) * np.sqrt(self.NumOfParticles/(self.NumOfParticles-1.0))
        SEM = dev/float(np.sqrt(self.NumOfParticles))
        FOM = 1.0 / ( ( SEM / ave )**2 * self.TotalTime )
        if flVerbose:
            print(        partkilltype +  'mean, dev:   {0:.8f}  {1:.8f}'.format(ave,dev))
            try:    print('              +-SEM , FOM:   {0:.8f}  {1:10g}'.format(SEM,FOM))
            except: print('              +-SEM      :   {0:.8f}'.format(SEM))
        return ave,dev,SEM,FOM
