#!usr/bin/env python
import sys
sys.path.append('/../../Classes/Tools')
from Talliespy import Tallies
from RandomNumberspy import RandomNumbers
from Particlepy import Particle
from Geometry_CLSpy import Geometry_CLS
from Geometry_CoPSpy import Geometry_CoPS
from Geometry_Markovianpy import Geometry_Markovian
from Geometry_Voronoipy import Geometry_Voronoi
from Geometry_Voxelpy import Geometry_Voxel
from Geometry_BoxPoissonpy import Geometry_BoxPoisson
from Geometry_SphericalInclusionpy import Geometry_SphericalInclusion
import numpy as np
import time

## \brief Generic driver for multi-D Monte Carlo rad trans solvers
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
# \author Dominic Lioce, dalioce@sandia.gov, liocedominic@gmail.com
#
# Associate the geometry, particle, random number object, and (possibly
# in future) tally object(s).  Leakage tallies, for now, handled in class.
# Eventually want to have cohorts and batches, but not at first.
class MonteCarloParticleSolver(Tallies):
    # \param[in] NumParticlesPerSample int, default 1; number of particles per r.v. sample or batch; if ==1, tallies simplified
    def __init__(self,NumParticlesPerSample=1):
        super(MonteCarloParticleSolver,self).__init__()
        assert isinstance(NumParticlesPerSample,int) and NumParticlesPerSample>0
        self.NumParticlesPerSample = NumParticlesPerSample
        self.NumOfParticles = 0
        self.flTally             = False

    def __str__(self):
        return str(self.__dict__)



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

    ## \brief Associates Geometry object - must be instance of Geometry_CoPS class #more options planned
    #
    # \param[in] Geom Geometry_CoPS object
    # \returns initializes self.Geom, initializes self.Geom.trackingType to specify Woodcock or standard tracking
    def associateGeom(self,Geom):
        assert isinstance(Geom,Geometry_CLS) or isinstance(Geom,Geometry_CoPS) or isinstance(Geom,Geometry_Markovian) or isinstance(Geom,Geometry_Voronoi) or isinstance(Geom,Geometry_Voxel) or isinstance(Geom,Geometry_BoxPoisson) or isinstance(Geom,Geometry_SphericalInclusion)
        self.Geom           = Geom
        self.GeomType       = self.Geom.GeomType
        self.xmin           = self.Geom.zbounds[0]
        self.xmax           = self.Geom.zbounds[1]
        self.SlabLength     = self.xmax - self.xmin
        self.y3Dmin         = self.Geom.ybounds[0]
        self.y3Dmax         = self.Geom.ybounds[1]
        self.x3Dmin         = self.Geom.xbounds[0]
        self.x3Dmax         = self.Geom.xbounds[1]
        self.nummats        = self.Geom.nummats
        self.abundanceModel = self.Geom.abundanceModel
        self.Geom.trackingType = 'standard' if isinstance(Geom,Geometry_CLS) else 'Woodcock'

    ## \brief Is the main driver for 'pushing' particles for MC or WMC solve.
    #
    # Makes some assertions to help ensure that options have been set to and
    # that the code is ready for the user to push particles.
    # Uses the chosen options and loops through the number of new particle histories,
    # calling the appropriate method to 'push' each history.
    #
    # \param[in] NumNewParticles int, number of particles to solve and add to any previously solved
    # \param[in] NumParticlesUpdateAt int, number of particles to print runtime updates at
    def pushParticles(self,NumNewParticles=None,NumParticlesUpdateAt=None):
        assert isinstance(NumNewParticles,int) and NumNewParticles>0
        self.NumNewParticles = NumNewParticles
        if not NumParticlesUpdateAt==None: assert isinstance(NumParticlesUpdateAt,int) and NumParticlesUpdateAt>0
        self.NumParticlesUpdateAt = NumParticlesUpdateAt
        if isinstance(self.NumParticlesUpdateAt,int): self.printTimeUpdate = self._printTimeUpdate
        else                                        : self.printTimeUpdate = self._printNoTimeUpdate

        self.tstart = time.time()
        for ipart in range(self.NumOfParticles,self.NumOfParticles+self.NumNewParticles):
            self.Rng.setSeedAtStride(istride=ipart)
            #setup sample
            flStartSample = True if  ipart   %self.NumParticlesPerSample==0 else False #First history in a sample?
            if flStartSample:
                tstart_samp_setup = time.time()
                if self.NumParticlesPerSample>1 or ipart==0:
                    self._initializeSampleTallies() #start of sample (or samples if 1 hist/sample)
                self.Geom._initializeSampleGeometryMemory()
                #solve material fractions for geometry
                if   self.abundanceModel == 'ensemble': #SM material fractions are assumed atomic mix in each flux bin
                    self.Tals[-1]['SampleMatAbundance'] = np.repeat( np.transpose([self.Geom.prob]), self.numFluxBins, axis=1)
                elif self.abundanceModel == 'sample':   #SM material fractions solved for each sample and used in material-dependent flux tallies
                    self.Geom.solveMaterialTypeFractions(numbins=self.numFluxBins,Rng=self.Rng,numSampPerBin=10000)
                    self.Tals[-1]['SampleMatAbundance'] = self.Geom.MatFractions
                elif self.abundanceModel == 'pre-sampled':
                    self.Tals[-1]['SampleMatAbundance'] = self.Geom.MatFractions
                tend_samp_setup   = time.time()
            #setup history
            self.Geom._initializeHistoryGeometryMemory()
            self._initializeHistoryFluxTallies()
            if self.flLeakTallies: self._initializeHistoryLeakageTallies()
            self.Part.initializeParticle()
            #simulate history
            killtype = self._pushMCParticle() if self.Geom.trackingType=='standard' else self._pushWMCParticle()
            #postprocess history
            if   killtype=='transmit':
                self.Tals[-1]['Transmit']    +=1
                if self.flLeakTallies: self._tallyTransmittance(self.Part.y,self.Part.x)
            elif killtype=='reflect' :
                self.Tals[-1]['Reflect']     +=1
                if self.flLeakTallies: self._tallyReflectance(self.Part.y,self.Part.x)
            elif killtype=='absorb'  : self.Tals[-1]['Absorb']      +=1
            elif killtype=='sideleak': self.Tals[-1]['SideLeakage'] +=1
            self._contribHistFluxTalsToSampTals()
            if self.flLeakTallies: self._contribHistLeakageTalsToSampTals()
            self.printTimeUpdate(ipart)
            #postprocess sample
            flEndSample   = True if (ipart+1)%self.NumParticlesPerSample==0 else False #Last history in a sample?
            if flEndSample:
                tend_hists        = time.time()
                if self.NumParticlesPerSample>1:
                    self._processSampleFluxTallies()
                    if self.flLeakTallies: self._processSampleLeakageTallies()
                tend_process_samp = time.time()
                self.Tals[-1]['SampleTime'] += tend_samp_setup - tstart_samp_setup + tend_process_samp - tend_hists
                self.Tals[-1]['MCTime']     += tend_hists      - tend_samp_setup
        self.TotalTime      += time.time() - self.tstart
        self.NumOfParticles += self.NumNewParticles

    ## \brief Simulates one particle using Woodcock Monte Carlo (WMC) solver. Intent: private.
    #
    # Simulates one particle history and tallies transmittance, reflectance, absorption, and flux.
    def _pushWMCParticle(self):
        while True:
            #Distance to collision (or potential collision for WMC), distance to boundary, min of these
            dc = self.Part.calculateDistanceToCollision(totxs=self.Geom.Majorant)
            db = self.Geom.calculateDistanceToBoundary()
            dmin = min(dc,db)
            #Stream particle
            self.Part.streamParticle(dist=dmin)
            #Evaluate pseudo-collision or stream-to-boundary event
            if dc<db:  #pseudo-collision
                #Chooses collision type.  For CoPS, samples point (needed in flux tallies).
                coltype = self.Geom.evaluateCollision()
                self._tallyFluxContribution(None,self.Part.z,None,self.Geom.CurrentMatInd,flcollide=True,streamingXS=self.Geom.Majorant)
                #Evaluate collision: 'reject' potential collision, 'absorb', or 'scatter'.
                if coltype=='absorb' : return 'absorb'
                if coltype=='scatter': self.Part.scatterParticle()
            else    :  #stream-to-boundary event
                #Determine boundary event type, reflect if needed, return leakage event type if particle escapes
                leaktype =  self.Geom.testForBoundaries()
                if not leaktype=='noleak': return leaktype

    ## \brief Simulates one particle using standard Monte Carlo (MC) solver. Intent: private.
    #
    # Simulates one particle history and tallies transmittance, reflectance, absorption, and flux.
    # Uses standard tracking for CLS algorithms and potential future benchmark/MC implementations.
    def _pushMCParticle(self): 
        killtype = None
        self.Geom.CurrentMatInd = self.Rng.choice(self.Geom.prob) #sets material index at the start of CLS based on volume fractions
        if self.Geom.CLSAlg=='LRP': #Sample dip and dim at start of LRP particle
            self.Geom.dip = self.Geom._sampleInterfaceDistance()
            self.Geom.dim = self.Geom._sampleInterfaceDistance()
        while True:
            #Distance to collision, distance to interface (CLS), distance to boundary, min of these
            dc = self.Part.calculateDistanceToCollision(totxs=self.Geom.totxs[self.Geom.CurrentMatInd])
            db = self.Geom.calculateDistanceToBoundary()
            if   self.Geom.CLSAlg == "CLS": di = self.Geom._sampleInterfaceDistance()
            elif self.Geom.CLSAlg == "LRP": di = self.Geom.dip
            if   self.Geom.fl1DEmulation  : di /= abs(self.Part.mu) # di projected to z-axis only if 1DEmulation
            dmin = min(dc,db,di)
            # Store particle properties for flux tallies
            oldz = self.Part.z; oldmu = self.Part.mu; oldmat = self.Geom.CurrentMatInd
            #Stream particle to new position
            self.Part.streamParticle(dist=dmin)
            #Tally total flux using track length
            if self.Geom.CLSAlg=='LRP': self.Geom._storeOldDirection()
            #Evaluate collision or stream-to-boundary event
            if dc==dmin:  #collision
                #Chooses collision type.
                coltype = self.Geom.evaluateCollision()
                #Evaluate collision: 'absorb' or 'scatter'.
                if coltype=='absorb' : killtype = 'absorb'
                if self.Geom.CLSAlg=="LRP": self.Geom._incrementDistances(dmin) #increment dip and dim for LRP
                if coltype=='scatter': self.Part.scatterParticle()
                if self.Geom.CLSAlg=="LRP": self.Geom._conditionallySwitchDistances() #switch dip and dim probabilistically
            elif db==dmin:  #stream-to-boundary event
                #Determine boundary event type, reflect if needed, return leakage event type if particle escapes
                leaktype =  self.Geom.testForBoundaries()
                if not leaktype=='noleak': killtype = leaktype
                if self.Geom.CLSAlg=='LRP': self.Geom._incrementDistances(dmin) #Reflected boundaries in LRP: increment dip and dim
            else:
                self.Geom._changeMat() #switch material at interface
                if self.Geom.CLSAlg=='LRP': #If LRP, sample new dip and set dim=0
                     self.Geom.dip = self.Geom._sampleInterfaceDistance(); self.Geom.dim = 0.0
            #Tally flux
            self._tallyFluxContribution(oldz,self.Part.z,oldmu,oldmat,dc==dmin,self.Geom.totxs[self.Geom.CurrentMatInd])
            if killtype != None: return killtype
                     

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
            print ('{}/{} histories, {:.2f}/{:.2f} min'.format(ipart+1,totparts,tottime,tottime/float(ipart+1)*totparts))

    ## \brief Computes and returns information on tallied flux.
    #
    # Computes mean, standard deviation, standard error of the mean (Monte Carlo uncertainty),
    # and figure-of-merit, prints them if chosen, and returns these values.
    #
    # \param flVerbose bool, default False, print to screen the data to return?
    # \param partkilltype str, used in part of the print message
    # \param tally int, tally of kill type of interest.
    #
    # \returns ave, stdev, SEM, and FOM of tally
    def _returnParticleKillMoments(self,flVerbose,partkilltype,talmom1,talmom2):
        ave = talmom1 / self.NumOfParticles
        dev = np.sqrt( talmom2/self.NumOfParticles - ave**2 ) * np.sqrt(self.NumOfParticles/(self.NumOfParticles-1.0))
        SEM = dev/float(np.sqrt(self.NumOfParticles))
        if not dev==0.0: FOM = 1.0 / ( ( dev / ave )**2 * self.TotalTime )
        else           : FOM = None
        if flVerbose:
            print(        partkilltype +  'mean, dev:   {0:.8f}  {1:.8f}'.format(ave,dev))
            try:    print('              +-SEM , FOM:   {0:.8f}  {1:10g}'.format(SEM,FOM))
            except: print('              +-SEM      :   {0:.8f}'.format(SEM))
        return ave,dev,SEM,FOM
