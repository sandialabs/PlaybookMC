#!usr/bin/env python
import sys
sys.path.append('/../../Classes/Tools')
from ClassToolspy import ClassTools
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
import matplotlib.pyplot as plt


## \brief Generic driver for multi-D Monte Carlo rad trans solvers
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
# \author Dominic Lioce, dalioce@sandia.gov, liocedominic@gmail.com
#
# Associate the geometry, particle, random number object, and (possibly
# in future) tally object(s).  Leakage tallies, for now, handled in class.
# Eventually want to have cohorts and batches, but not at first.
class MonteCarloParticleSolver(ClassTools):
    def __init__(self):
        super(MonteCarloParticleSolver,self).__init__()
        self.NumOfParticles      = 0
        self.TotalTime           = 0.0
        self.flTally             = False
        self.Tmom1 = 0.0; self.Rmom1 = 0.0; self.Amom1 = 0.0; self.Smom1 = 0.0; self.Fluxmom1 = 0.0 #transmit, reflect, absorb, sideleak, and flux first moment tallies
        self.Tmom2 = 0.0; self.Rmom2 = 0.0; self.Amom2 = 0.0; self.Smom2 = 0.0; self.Fluxmom2 = 0.0 #transmit, reflect, absorb, sideleak, and flux second moment tallies

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
        self.Geom = Geom
        self.Geom.trackingType = 'standard' if isinstance(Geom,Geometry_CLS) else 'Woodcock'

    ## \brief Associates FluxTallyObject with MC solver object for taking tallies.
    #
    # The two objects are associated in that the MC solver object can
    # call functions in the FluxTalObj, and the FluxTalObj can be called
    # independently--both will operate on the same object.
    #
    # Function also sets flag indicating to use flux tallies.  Maybe use
    # of this flag can be overwritten by a test for a FluxTalObj.
    #
    # \param[in] FluxTallyObject 'FluxTallies' object, instantiated flux tally object from PlaybookMC
    # \returns sets object to self.FluxTalObj, sets self.flTally to True
    def associateFluxTallyObject(self,FluxTallyObject):
        self.FluxTalObj = FluxTallyObject
        self.flTally = True


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
            self.histFluxmom1 = 0.0
            self.Rng.setSeedAtStride(istride=ipart)
            self.Part.initializeParticle()
            self.Geom.initializeGeometryMemory()
            if self.flTally: self.FluxTalObj._initializeHistoryTallies()
            killtype = self._pushMCParticle() if self.Geom.trackingType=='standard' else self._pushWMCParticle()
            if self.flTally: self.FluxTalObj._foldHistoryTallies()
            #add tallies here
            if   killtype=='transmit': self.Tmom1 += self.Part.weight; self.Tmom2 += self.Part.weight**2
            elif killtype=='reflect' : self.Rmom1 += self.Part.weight; self.Rmom2 += self.Part.weight**2
            elif killtype=='absorb'  : self.Amom1 += self.Part.weight; self.Amom2 += self.Part.weight**2
            elif killtype=='sideleak': self.Smom1 += self.Part.weight; self.Smom2 += self.Part.weight**2
            self.histFluxmom1 /= 1.0 #self.Geom.Volume ##Larmier appears not to normalize for volume
            self.Fluxmom1 += self.histFluxmom1; self.Fluxmom2 += self.histFluxmom1**2
##            if ipart+1 in self.Geom.particlestoplot: self.Geom.plotRealization(ipart)  ######## plot realization
            if isinstance(self.Geom,Geometry_CoPS):
                if self.Geom.flCollectPoints: self.Geom.appendNewPointsToCollection(ipart)
            self.printTimeUpdate(ipart)
        self.TotalTime += time.time() - self.tstart
        self.NumOfParticles += self.NumNewParticles
        if self.flTally: self.FluxTalObj._computeFluxQuantities(self.TotalTime,self.NumOfParticles)

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
                #Tally flux, both overall flux using 'self' and flux in bins using 'FluxTalObj'
                self.histFluxmom1 += self.Part.weight / self.Geom.Majorant
                if self.flTally:
                    assert self.isclose(self.Part.weight,1.0) #'FluxTallies' objects not currently setup for weighted particles
                    #                                                              iseg here for CoPS, will have to refactor some for other geometries
                    self.FluxTalObj._tallyFlux( oldx=None,x=self.Part.z,mu=None,iseg=self.Geom.CurrentMatInd,collinfo=(True,self.Geom.Majorant) )
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
            self.histFluxmom1 += dmin
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
            if self.flTally: self.FluxTalObj._tallyFlux( oldx=oldz,x=self.Part.z,mu=oldmu,iseg=oldmat,collinfo=(dc==dmin,self.Geom.totxs[self.Geom.CurrentMatInd]) )
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


    ## \brief Veneer for '_returnParticleKillMoments' that passes transmittance tally information
    #
    # \param flVerbose bool, default False, print to screen the data to return?
    # \returns ave, stdev, SEM, and FOM of transmittance
    def returnTransmittanceMoments(self,flVerbose=False):
        assert isinstance(flVerbose,bool)
        return self._returnParticleKillMoments(flVerbose,'Transmittance   ',self.Tmom1,self.Tmom2)

    ## \brief Veneer for '_returnParticleKillMoments' that passes reflectance tally information
    #
    # \param flVerbose bool, default False, print to screen the data to return?
    # \returns ave, stdev, SEM, and FOM of reflectance
    def returnReflectanceMoments(self,flVerbose=False):
        assert isinstance(flVerbose,bool)
        return self._returnParticleKillMoments(flVerbose,'Reflectance     ',self.Rmom1,self.Rmom2)

    ## \brief Veneer for '_returnParticleKillMoments' that passes side leakage tally information
    #
    # \param flVerbose bool, default False, print to screen the data to return?
    # \returns ave, stdev, SEM, and FOM of reflectance
    def returnSideLeakageMoments(self,flVerbose=False):
        assert isinstance(flVerbose,bool)
        return self._returnParticleKillMoments(flVerbose,'SideLeakage     ',self.Smom1,self.Smom2)

    ## \brief Veneer for '_returnParticleKillMoments' that passes absorption tally information
    #
    # \param flVerbose bool, default False, print to screen the data to return?
    # \returns ave, stdev, SEM, and FOM of absorption
    def returnAbsorptionMoments(self,flVerbose=False):
        assert isinstance(flVerbose,bool)
        return self._returnParticleKillMoments(flVerbose,'Absorption      ',self.Amom1,self.Amom2)

    ## \brief Veneer for '_returnParticleKillMoments' that passes flux tally information
    #
    # \param flVerbose bool, default False, print to screen the data to return?
    # \returns ave, stdev, SEM, and FOM of flux
    def returnFluxMoments(self,flVerbose=False):
        assert isinstance(flVerbose,bool)
        return self._returnParticleKillMoments(flVerbose,'Flux            ',self.Fluxmom1,self.Fluxmom2)


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
