#!usr/bin/env python
import sys
sys.path.append('/../../Classes/Tools')
from Talliespy import Tallies
import numpy as np
import time

## \brief Solves radiation transport quantities on 1D slab using MC method.
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
# Class for performing Monte Carlo particle transport.
# Class can perform Monte Carlo on a slab with defined material boundaries
# or using Woodcock Monte Carlo.
class MonteCarloParticleSolver(Tallies):
    # \param[in] NumParticlesPerSample int, default 1; number of particles per r.v. sample or batch; if ==1, tallies simplified
    def __init__(self,NumParticlesPerSample=1):
        super(MonteCarloParticleSolver,self).__init__()
        assert isinstance(NumParticlesPerSample,int) and NumParticlesPerSample>0
        self.NumParticlesPerSample = NumParticlesPerSample
        self.NumOfParticles = 0
        self.flTally        = False

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
    # \param[in] slab object, instantiation of OneDSlabGeometry
    def defineMCGeometry(self,slab):
        self.GeomType      = 'MC'
        self._initializeHistoryGeometryMemory = self._initializeGeometryMemory_pass
        self._initializeSampleGeometryMemory  = self._initializeGeometryMemory_pass
        self.fPushParticle = self._pushMCParticle
        self.Slab          = slab
        self.XSFunction    = self.Slab.samplePoint_x
        self.SlabLength    = self.Slab.s
        self.xmin          = 0.0
        self.xmax          = self.SlabLength
        self.nummats       = self.Slab.nummats
        self.abundanceModel= 'sample'


    ## \brief Sets a link to instantiation of OneDSlab
    #
    # \param[in] slab object, instantiation of OneDSlabGeometry
    def defineWMCGeometry(self,slab):
        self.GeomType      = 'WMC'
        self._initializeHistoryGeometryMemory = self._initializeGeometryMemory_pass
        self._initializeSampleGeometryMemory  = self._initializeGeometryMemory_pass
        self.fPushParticle = self._pushWMCParticle
        self.Slab          = slab
        self.XSFunction    = self.Slab.samplePoint_x
        self.SlabLength    = self.Slab.s
        self.xmin          = 0.0
        self.xmax          = self.SlabLength
        self.nummats       = self.Slab.nummats
        self.Majorant      = max(self.Slab.totxs)
        self.abundanceModel= 'sample'

    ## \brief Fulfills the syntax of initializing geometry memory when nothing needs initialized
    #
    # \returns nothing
    def _initializeGeometryMemory_pass(self):
        pass

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

    ## \brief If chosen, initializes random number seed before each history. Intent: private.
    #
    # Is called at beginning of each particle history.  If user chose to use seeds,
    # the history-specific seed is set.
    #
    # \param[in] ipart int, particle history number
    # \returns sets Rng attribute seed for particular particle history
    def _setRandomNumberSeed(self,ipart):
        if self.flUseSeed: self.Rng.seed( self.Seed + self.Stride * ipart )


    ## \brief Is the main driver for 'pushing' particles for MC Particle solver and SMC solvers.
    #
    # Primary driver for particles in MonteCarloParticleSolverpy.py and SpecialMonteCarloDriverspy.py
    # (where the latter contains capabilities for stochastic media transport).
    #
    # \param[in] NumNewParticles int, number of particles to solve and add to any previously solved
    # \param[in] NumParticlesUpdateAt int, number of particles to print runtime updates at
    # \param[in] flCompFluxQuants bool, default True, compute flux quantities, e.g., uncertainty and FOM?
    def pushParticles(self,NumNewParticles=None,NumParticlesUpdateAt=None,flCompFluxQuants=True):
        assert isinstance(NumNewParticles,int) and NumNewParticles>0
        self.NumNewParticles = NumNewParticles
        if not NumParticlesUpdateAt==None: assert isinstance(NumParticlesUpdateAt,int) and NumParticlesUpdateAt>0
        self.NumParticlesUpdateAt = NumParticlesUpdateAt
        if isinstance(self.NumParticlesUpdateAt,int): self.printTimeUpdate = self._printTimeUpdate
        else                                        : self.printTimeUpdate = self._printNoTimeUpdate
        assert isinstance(flCompFluxQuants,bool)
        if   self.GeomType == 'AlgC' and self.nummats>2: raise Exception("Current Algorithm C implementation only allows transport in binary stochastic media (i.e., cannot handle mixing of more than two material types)")

        self.tstart = time.time()
        for ipart in range(self.NumOfParticles,self.NumOfParticles+self.NumNewParticles):
            self._setRandomNumberSeed(ipart)
            #setup sample
            flStartSample = True if  ipart   %self.NumParticlesPerSample==0 else False #First history in a sample?
            if flStartSample:
                tstart_samp_setup = time.time()
                if self.NumParticlesPerSample>1 or ipart==0:
                    self._initializeSampleTallies() #start of sample (or samples if 1 hist/sample)
                self._initializeSampleGeometryMemory()
                #solve material fractions for geometry
                if   self.abundanceModel == 'ensemble': #SM material fractions are assumed atomic mix in each flux bin
                    self.Tals[-1]['SampleMatAbundance'] = np.repeat( np.transpose([self.prob]), self.numFluxBins, axis=1)
                elif self.abundanceModel == 'sample':   #SM material fractions solved for each sample and used in material-dependent flux tallies
                    self.Slab.solveMaterialTypeFractions(numbins=self.numFluxBins)
                    self.Tals[-1]['SampleMatAbundance'] = self.Slab.MatFractions
                tend_samp_setup   = time.time()
            #setup history
            self._initializeHistoryGeometryMemory()
            self._initializeHistoryFluxTallies()
            #simulate history
            self.fPushParticle()
            #postprocess history
            self._contribHistTalsToSampTals()
            self.printTimeUpdate(ipart)
            #postprocess sample
            flEndSample   = True if (ipart+1)%self.NumParticlesPerSample==0 else False #Last history in a sample?
            if flEndSample:
                tend_hists        = time.time()
                if self.NumParticlesPerSample>1: self._processSampleFluxTallies()
                tend_process_samp = time.time()
                self.Tals[-1]['SampleTime'] += tend_samp_setup - tstart_samp_setup + tend_process_samp - tend_hists
                self.Tals[-1]['MCTime']     += tend_hists      - tend_samp_setup
        self.TotalTime      += time.time() - self.tstart
        self.NumOfParticles += self.NumNewParticles

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
                    self.Tals[-1]['Absorb'] +=1; flbreak = True    #absorb
                else                                                          :
                    mu = 2.0*self.Rng.rand()-1.0 #scatter
            else        :  #evaluate boundary crossing
                if mu<0.0:
                    if iseg==0                       : self.Tals[-1]['Reflect'] +=1; flbreak = True #tally R
                    x = self.Slab.matbound[iseg]
                    iseg -= 1
                else     :
                    if iseg==len(self.Slab.mattype)-1: self.Tals[-1]['Transmit']+=1; flbreak = True #tally T
                    iseg += 1
                    x = self.Slab.matbound[iseg]
            #tally flux
            self._tallyFluxContribution(oldx,x,oldmu,self.Slab.mattype[oldiseg],flcollide,totxs)
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
            dpc= -np.log( self.Rng.rand() ) / self.Majorant * mu
            #solve distance to boundary
            db = self.SlabLength-x if mu>=0.0 else -x
            #choose potential collision or boundary crossing
            if  self.isleq(abs(db),abs(dpc)): dx = db ; flcollide = False
            else                            : dx = dpc; flcollide = True
            #stream particle
            x = x + dx
            #evaluate collision or boundary crossing
            if flcollide:  #evaluate potential collision
                material = self.XSFunction(x,'mattype')
                if self.Rng.rand() < self.Slab.totxs[material]/self.Majorant:
                    #------#evaluate accepted collision
                    if self.Rng.rand()> self.Slab.scatrat[material]:
                        self.Tals[-1]['Absorb'] +=1; flbreak = True    #absorb
                    else                                                      :
                        mu = 2.0*self.Rng.rand()-1.0 #scatter
            else        :  #evaluate boundary crossing (always leakage event)
                if   self.isclose(x,0.0)            : self.Tals[-1]['Reflect'] +=1; x = 0.0            ; flbreak = True #tally R
                elif self.isclose(x,self.SlabLength): self.Tals[-1]['Transmit']+=1; x = self.SlabLength; flbreak = True #tally T
                else                                : assert(False)
            #tally flux
            if flcollide: self._tallyFluxContribution(oldx,x,oldmu,material,flcollide,self.Majorant)
            #if particle absorbed or leaked, kill history
            if flbreak: break