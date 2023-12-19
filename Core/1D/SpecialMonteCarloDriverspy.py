#!usr/bin/env python
import sys
sys.path.append('/../../Core/Tools')
from MonteCarloParticleSolverpy import MonteCarloParticleSolver
from MarkovianInputspy import MarkovianInputs
import numpy as np
import operator
import time
import matplotlib.pyplot as plt
import warnings

## \brief Solves radiation transport quantities on 1D slab using collection of special Monte Carlo drivers.
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
# \author Emily Vu, evu@sandia.gov, emilyhvu@berkeley.edu
# \author Dominic Lioce, dalioce@sandia.gov, liocedominic@gmail.com
#
#  Special Monte Carlo Drivers include:
#   -Chord Length Sampling (CLS, Algorithm A)
#   -Local Realization Preserving (LRP, Algorithm B)
#   -Algorithm C
#   -Conditional Point Sampling (CoPS).
class SpecialMonteCarloDrivers(MonteCarloParticleSolver,MarkovianInputs):
    def __init__(self):
        super(SpecialMonteCarloDrivers,self).__init__()
        
    def __str__(self):
        return str(self.__dict__)

    ## \brief User sets recent memory and amnesia radius parameters for CoPS
    #
    # \param[in] recentMemory, int, number of points to store in recent memory
    # \param[in] amnesiaRadius, float, distance from remembered point in which to not commit new point to long-term memory
    # \param[in] flLongTermMemory, bool, whether or not long-term memory will be used
    # \returns sets self.recentMemory and self.amnesiaRadius
    def associateLimitedMemoryParam(self,recentMemory,amnesiaRadius,flLongTermMemory=True):
        assert (isinstance(recentMemory,int) or recentMemory==np.inf) and recentMemory  >= 0
        assert (isinstance(amnesiaRadius,float) and amnesiaRadius >= 0.0) or amnesiaRadius==None
        assert isinstance(flLongTermMemory,bool)
        if     flLongTermMemory and     amnesiaRadius==None: raise Exception("If long-term memory is enabled, a non-negative floating point value must be provided for the amnesia radius.")
        if not flLongTermMemory and not amnesiaRadius==None: warnings.warn("A value was provided for the amnesiaRadius which will not be used since long-term memory was not enabled")

        self.recentMemory     = recentMemory
        self.amnesiaRadius    = amnesiaRadius
        self.flLongTermMemory = flLongTermMemory


    # \brief Defines Markovian geometry parameters
    #
    # \param[in] totxs, list of floats, list of total cross sections
    # \param[in] lam, list of floats, list of chord lengths
    # \param[in] slablength, float, slab length
    # \param[in] scatxs, list of floats, list of scattering cross sections
    # \param[in] NaryType, str, 'volume-fraction' or 'uniform'
    # \returns sets cross sections, chord length, and slab length
    def defineMarkovianGeometry(self,totxs=None,lam=None,slablength=None,scatxs=None,NaryType=None):
        # Assert valid material properties and store in object, user may or may not specify scattering cross section values
        assert isinstance(totxs,list)
        self.nummats = len(totxs)
        for i in range(0,self.nummats):
            assert isinstance(totxs[i],float) and totxs[i]>=0.0
        self.totxs = totxs
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

        # Assert slab length and store in object
        assert isinstance(slablength,float) and slablength>0.0
        self.SlabLength = slablength

        # Solve and store material probabilities, mean chord length, and correlation length
        self.NaryType = NaryType
        assert isinstance(lam,list) and len(lam)==self.nummats
        for i in range(0,self.nummats): assert isinstance(lam[i],float) and lam[i]>0.0
        self.solveNaryMarkovianParamsBasedOnChordLengths(lam)
        if self.NaryType == 'uniform': del[self.lamc] #lamc is meaningless for non-volume-fraction transition matrices. Note: refactor to handle non-volume-fraction transition matrices better is in order.
        
    ## \brief Chooses CLS method
    # 
    # 3-Dimensional Emulation allows for the material to change as a function of 
    # distance traveled by a particle on the flight path, even if not in x direction. It aims to 
    # provide 3D results while maintaining the efficiency of 1D transport.
    #
    # \param[in] fl3DEmulation, bool, default False, flag for 3-Dimensional Emulation
    # \returns sets fl3DEmulation
    def chooseCLSsolver(self, fl3DEmulation = False):
        self.GeomType = 'CLS'
        assert isinstance(fl3DEmulation,bool)
        self.fl3DEmulation = fl3DEmulation
        
    ## \brief Chooses LRP method 
    # 
    # 3-Dimensional Emulation allows for the material to change as a function of 
    # distance traveled by a particle on the flight path, even if not in x direction. It aims to 
    # provide 3D results while maintaining the efficiency of 1D transport.
    #
    # \param[in] fl3DEmulation, bool, default False, flag for 3-Dimensional Emulation
    # \returns sets fl3DEmulation
    def chooseLRPsolver(self, fl3DEmulation = False):
        self.GeomType = 'LRP'
        assert isinstance(fl3DEmulation,bool)
        self.fl3DEmulation = fl3DEmulation

    ## \brief Chooses AlgC method 
    def chooseAlgCsolver(self):
        self.GeomType = 'AlgC'
        self.fl3DEmulation = False #3DEmulation not possible for AlgC
        
    ## \brief Chooses CoPS method
    #
    # Seeds can be used to create at least a realization family before starting the CoPS simulation.
    # If an amnesia ratius of at least half the spacing of the seeds is used, no new points will
    # be added to long-term memory during the CoPS simulation.
    # 
    # 3-Dimensional Emulation allows for the material to change as a function of 
    # distance traveled by a particle on the flight path, even if not in x direction. It aims to 
    # provide 3D results while maintaining the efficiency of 1D transport. This option can only
    # be used when conditional on one the most recent point for CoPS.
    #
    # \param[in] numcondprobpts, int, amount of points to use for CPF calculation. p-1 in CoPS notation
    # \param[in] numseeds, int, number of evenly spaced points to be sampled before the first history on each cohort
    # \param[in] fl3DEmulation, bool, default False, flag for 3-Dimensional Emulation
    def chooseCoPSsolver(self, numcondprobpts, numseeds = 0, fl3DEmulation = False):
        self.GeomType = 'CoPS'
        assert numcondprobpts == 0 or numcondprobpts == 1 or numcondprobpts == 2
        assert isinstance(numseeds,int) and numseeds>=0
        self.numCoPSSeeds = numseeds
        assert isinstance(fl3DEmulation,bool)
        if fl3DEmulation and numcondprobpts == 2:
            numcondprobpts = 1
            print('numcondprobpts changed to 1 in order to support 3DEmulation functionality')
        if fl3DEmulation and numseeds>0: raise Exception("Dimentional emulation and seeds cannot be used at the same time with CoPS")
        self.numcondprobpts = numcondprobpts
        self.fl3DEmulation = fl3DEmulation  

    ## \brief Is the main driver for 'pushing' particles for SMC solves.
    #
    # Makes some assertions to help ensure that options have been set to and
    # that the code is ready for the user to push particles.
    # Uses the chosen options and loops through the number of new particle histories,
    # calling the appropriate method to 'push' each history.
    #
    # \param[in] NumNewParticles int, number of particles to solve and add to any previously solved
    # \param[in] NumParticlesUpdateAt int, number of particles to print runtime updates at
    # \param[in] flCompFluxQuants bool, default True, compute flux quantities, e.g., uncertainty and FOM?
    # \param[in] initializeTallies str, default 'first_hist'; 'first_hist' or 'no_hist' behavior to initialize/reset leakage tallies
    # \param[in] initializeGeomMem str, default 'each_hist'; 'first_hist', 'each_hist', or 'no_hist' behavior to initialize/reset CoPS memory (shared memory==cohort)
    def pushParticles(self,NumNewParticles=None,NumParticlesUpdateAt=None,flCompFluxQuants=True,initializeTallies='first_hist',initializeGeomMem='each_hist'):
        assert isinstance(NumNewParticles,int) and NumNewParticles>0
        self.NumNewParticles = NumNewParticles
        if not NumParticlesUpdateAt==None: assert isinstance(NumParticlesUpdateAt,int) and NumParticlesUpdateAt>0
        self.NumParticlesUpdateAt = NumParticlesUpdateAt
        if isinstance(self.NumParticlesUpdateAt,int): self.printTimeUpdate = self._printTimeUpdate
        else                                        : self.printTimeUpdate = self._printNoTimeUpdate
        assert isinstance(flCompFluxQuants,bool)
        assert initializeTallies=='first_hist' or                                   initializeTallies=='no_hist'
        assert initializeGeomMem=='first_hist' or initializeGeomMem=='each_hist' or initializeGeomMem=='no_hist'
        if   self.GeomType == 'AlgC' and self.nummats>2: raise Exception("Current Algorithm C implementation only allows transport in binary stochastic media (i.e., cannot handle mixing of more than two material types)")
        if   self.GeomType in {'CLS','LRP','AlgC'}: fPushParticle = self._pushCLSParticle
        elif self.GeomType =='CoPS'               : fPushParticle = self._pushCoPSParticle
        self.tstart = time.time()
        if initializeTallies=='first_hist'                                  : self._initializeInternalTallies()
        if initializeGeomMem=='first_hist' or initializeGeomMem=='each_hist': self._initializeGeometryMemory()
        for ipart in range(self.NumOfParticles,self.NumOfParticles+self.NumNewParticles):
            self._setRandomNumberSeed(ipart)
            if self.flTally: self.FluxTalObj._setRandomNumberSeed(ipart)
            if self.flTally: self.FluxTalObj._initializeHistoryTallies()
            if initializeGeomMem=='each_hist': self._initializeGeometryMemory()
            fPushParticle()
            if self.flTally: self.FluxTalObj._foldHistoryTallies()
            self.printTimeUpdate(ipart)
        self.TotalTime      += time.time() - self.tstart
        self.NumOfParticles += self.NumNewParticles
        if self.flTally and flCompFluxQuants: self.FluxTalObj._computeFluxQuantities(self.TotalTime,self.NumOfParticles)

    ## \brief Sample new distance to interface. Intent: private.
    #
    # \param[in] mat ind, material index of which to sample a new chord length
    # \returns di float, distance to material interface
    def _newdi(self,mat):
        return self.lam[mat] * np.log(1/self.Rng.rand())

    ## \brief Veneer to change to different material. Intent: private.
    #
    # \param[in] mat ind, material index to choose the opposite of
    # \returns mat ind, index of different material after change
    def _changeMat(self,mat):
        if   self.NaryType == 'volume-fraction': local_probs = self.prob[:]
        elif self.NaryType == 'uniform'        : local_probs = np.ones(self.nummats)
        local_probs[mat] = 0.0                                         #give probability of zero to current material
        local_probs = np.divide( local_probs, np.sum(local_probs) )    #normalize probabilities
        mat = int( self.Rng.choice( self.nummats, p=local_probs ) )    #select material from any but current

        return mat

    ## \brief Switch interface distances for LRP and AlgC. Intent: private.
    #
    # Will also switch dipp and dimm for AlgC. If normal LRP,
    # dip and dim switch depending on direction with respect
    # to the x-axis. If 3DEmulation LRP, dip and dim switch prob-
    # abilistically based on scatter angle.
    #
    # \param[in] mu, direction of travel for particle
    # \returns swtiches dip/dim (LRP,AlgC) and dipp/dimm (AlgC)
    def _conditionallySwitchDistances(self):
        flSwitch = False
        if not self.fl3DEmulation:
            if np.sign(self.mu) != np.sign(self.oldmu): flSwitch = True
        else:
            phi = 2*np.pi*self.Rng.rand() #old direction treated as phi=0, new direction samples the difference in phi and uses that
            scattermu = np.dot([self.mu,np.sqrt(1.0-self.mu**2)*np.cos(phi),np.sqrt(1.0-self.mu**2)*np.sin(phi)],[self.oldmu,np.sqrt(1.0-self.oldmu**2),0])
            if scattermu < 2.0*self.Rng.rand() - 1.0: flSwitch = True

        if flSwitch                          : self.dip ,self.dim  = self.dim ,self.dip
        if flSwitch and self.GeomType=='AlgC': self.dipp,self.dimm = self.dimm,self.dipp

    ## \brief Switch interface distances for LRP and AlgC. Intent: private.
    #
    # \returns increments dim, dip, dimm, and dipp
    def _incrementDistances(self,dist):
        if not self.fl3DEmulation: 
            self.dip -= dist
            self.dim += dist
        elif   self.fl3DEmulation: 
            self.dip -= dist/abs(self.mu) #3DEmulation scales dip/dim increment to flight path rather than x-axis
            self.dim += dist/abs(self.mu)
        

    ## \brief Simulates one particle using CLS, LRP, or AlgC. Intent: private.
    #
    # Currently uses isotropic source on left boundary and isotropic scattering
    # and tallies transmittance, reflectance, and absorption.
    def _pushCLSParticle(self):
        self.mu  = self._setParticleSourceAngle()
        self.oldmu = self.mu
        x   = self._initializePosition()
        mat = int( self.Rng.choice( self.nummats, p=self.prob[:] ) ) #material designation
        if   self.GeomType=='LRP' : self.dip = self._newdi(mat); self.dim = self._newdi(mat)
        elif self.GeomType=='AlgC': self.dip = self._newdi(mat); self.dim = 0.0; self.dipp = self._newdi(self._changeMat(mat)); self.dimm = 0.0
        flbreak = False
        while True:
            oldx = x; oldmat = mat; flcollide = False
            #dist to interface, boundary, collision
            if   self.GeomType=='CLS'                         : di = self._newdi(mat) 
            elif self.GeomType=='LRP' or self.GeomType=='AlgC': di = self.dip
            if   self.fl3DEmulation                           : di *= abs(self.mu) #di scaled to flight path rather than x-axis
            db = self.SlabLength - x if self.mu>0.0 else x
            dc = -np.log( self.Rng.rand() ) / self.totxs[mat] * abs(self.mu)
            #stream particle, choose collision or boundary crossing
            dist = min(di,db,dc)
            x = x + dist if self.mu>=0.0 else x - dist
            #evaluate collision or boundary crossing
            if   db == dist: #boundary
                if   self.isclose(x,0.0)            : self.R+=1 
                elif self.isclose(x,self.SlabLength): self.T+=1
                else                                : assert(False)
                flbreak = True
            elif di == dist: #interface
                 mat = self._changeMat(mat)
                 if   self.GeomType=='LRP': #sample new interface for dip, set dim to 0
                     self.dip = self._newdi(mat); self.dim = 0.0
                 elif self.GeomType=='AlgC': #sample new interface for dipp, adjust all other stored interfaces as necessary
                     self.dimm = self.dim + self.dip; self.dim = 0.0; self.dip = self.dipp; self.dipp = self._newdi(self._changeMat(mat))
            else           : #collision
                flcollide = True
                if self.Rng.rand() > self.scatrat[mat]: self.A+=1; flbreak = True #absorb
                else: #scatter
                    if self.GeomType in {'LRP','AlgC'}: self._incrementDistances(dist)
                    self.oldmu = self.mu   #store old mu, sample new, switch dim/dip if direction switched
                    self.mu = 2.0*self.Rng.rand()-1.0
                    if self.GeomType in {'LRP','AlgC'}: self._conditionallySwitchDistances()
            #tally flux
            if self.flTally: self.FluxTalObj._tallyFlux(oldx=oldx,x=x,mu=self.oldmu,iseg=oldmat,collinfo=(flcollide,self.totxs[mat]) )
            #if particle absorbed or leaked, kill history
            if flbreak: break

    ## \brief Sets conditional probability for two point correlation method
    #
    # \param[in] r, float, distance from nearest point
    # \param[in] mat, int, material type
    def _TwoPtCorrelation(self,r,mat):
        self.condprobs = []
        for imat in range(0,self.nummats):
            if mat==imat: self.condprobs.append( self.prob[imat] + (1.0 - self.prob[imat]) *        np.exp(-r/self.lamc)  )
            else        : self.condprobs.append( self.prob[imat]                           * (1.0 - np.exp(-r/self.lamc)) ) 
            
    ## \brief Solves CoPS3PO conditional probabilities for an N-ary medium
    #
    # \param[in] r1, float, distance from first nearest point
    # \param[in] r2, float, distance from second nearest point
    # \param[in] mat1, int, material type of first nearest point
    # \param[in] mat2, int, material type of second nearest point
    def _ThreePtCorrelation(self,r1, r2, mat1, mat2):
        self.condprobs = [] #empty conditional probability list and fill with new values
        for imat in range(0,len(self.prob)):
            if       mat1==imat and     imat==mat2: self.condprobs.append( self._probAAA(self.prob[imat],r1,r2) )
            elif     mat1==imat and not imat==mat2: self.condprobs.append( self._probAAB(self.prob[imat],self.prob[mat2],r1,r2) )
            elif not mat1==imat and     imat==mat2: self.condprobs.append( self._probBAA(self.prob[imat],self.prob[mat1],r1,r2) )
            elif not mat1==imat and not imat==mat2: self.condprobs.append( self._probABC(self.prob[mat1],self.prob[imat],self.prob[mat2],r1,r2) )
        self.condprobs = np.divide( self.condprobs, np.sum(self.condprobs) )

    ## \brief Returns the conditional probability of having alpha, alpha, alpha (AAA)
    #
    # For 3 point correlations only 
    #
    # \param[in] pa, probability of material alpha
    # \param[in] r1, float, distance from first nearest point
    # \param[in] r2, float, distance from second nearest point
    def _probAAA(self,pa,r1,r2):
        paaa_0int = pa    * self._ComputePseudoFreq(r1,'zero') * self._ComputePseudoFreq(r2,'zero')
        paaa_1int = pa**2 * self._ComputePseudoFreq(r1,'atleastone') * self._ComputePseudoFreq(r2,'zero') + pa**2 * self._ComputePseudoFreq(r1,'zero') * self._ComputePseudoFreq(r2,'atleastone')
        paaa_2int = pa**3 * self._ComputePseudoFreq(r1,'atleastone') * self._ComputePseudoFreq(r2,'atleastone')
        return paaa_0int + paaa_1int + paaa_2int

    ## \brief Returns the conditional probability of having alpha, alpha, beta (AAB)
    #
    # For 3 point correlations only 
    #
    # \param[in] pa, probability of material alpha
    # \param[in] pb, probability of material beta
    # \param[in] r1, float, distance from first nearest point
    # \param[in] r2, float, distance from second nearest point
    def _probAAB(self,pa,pb,r1,r2):
        paab_1int = pa    * pb * self._ComputePseudoFreq(r1,'zero'      ) * self._ComputePseudoFreq(r2,'atleastone')
        paab_2int = pa**2 * pb * self._ComputePseudoFreq(r1,'atleastone') * self._ComputePseudoFreq(r2,'atleastone')
        return paab_1int + paab_2int

    ## \brief Returns the conditional probability of having beta, alpha, alpha (BAA)
    #
    # For 3 point correlations only 
    #
    # \param[in] pa, probability of material alpha
    # \param[in] pb, probability of material beta
    # \param[in] r1, float, distance from first nearest point
    # \param[in] r2, float, distance from second nearest point
    def _probBAA(self,pa,pb,r1,r2):
        pbaa_1int = pa    * pb * self._ComputePseudoFreq(r1,'atleastone') * self._ComputePseudoFreq(r2,'zero')
        pbaa_2int = pa**2 * pb * self._ComputePseudoFreq(r1,'atleastone') * self._ComputePseudoFreq(r2,'atleastone')
        return pbaa_1int + pbaa_2int

    ## \brief Returns the conditional probability of having alpha, beta, gamma (ABC)
    #
    # For 3 point correlations only
    # Also covers for the case of alpha, beta, alpha
    #
    # \param[in] pa, probability of material alpha
    # \param[in] pb, probability of material beta
    # \param[in] pc, probability of material gamma
    # \param[in] r1, float, distance from first nearest point
    # \param[in] r2, float, distance from second nearest point
    def _probABC(self,pa,pb,pc,r1,r2):
        pabc_2int = pa * pb * pc * self._ComputePseudoFreq(r1,'atleastone') * self._ComputePseudoFreq(r2,'atleastone')
        return pabc_2int

    ## \brief Returns conditional probability for absorption only problem
    #
    # \param[in] r1, float, distance from first nearest point
    # \param[in] r2, float, distance from second nearest point
    # \param[in] mat1, int, material type of first nearest point
    # \param[in] mat2, int, material type of second nearest point
    # \param[in] matM, int, material type of middle point  
    def _ComputeCondProb(self,r1,r2,mat1,mat2,matM):
        pt1 = self.prob[0]                                          if mat1 == 0 else self.prob[1]
        pt2 = self.prob[0]*self._ComputePseudoFreq(r1,'atleastone') if matM == 0 else self.prob[1]*self._ComputePseudoFreq(r1,'atleastone') 
        pt3 = self.prob[0]*self._ComputePseudoFreq(r2,'atleastone') if mat2 == 0 else self.prob[1]*self._ComputePseudoFreq(r2,'atleastone')
        if mat1 == matM: pt2 += self._ComputePseudoFreq(r1,'zero')
        if mat2 == matM: pt3 += self._ComputePseudoFreq(r2,'zero')
        return pt1*pt2*pt3

    ## \brief Returns frequency of pseudo-interfaces
    #
    # \param[in] r, float, distance from point
    # \param[in] case, char, specify case with zero or at least one pseudo-interface
    def _ComputePseudoFreq(self,r,case):
        assert case == 'zero' or case == 'atleastone'
        freq = np.exp(-r/self.lamc) if case == 'zero' else 1-np.exp(-r/self.lamc)
        return freq

    ## \brief Samples cohort "seeds" on a regular grid
    #
    # If only one seed, the seed is in the center of the domain.
    # If two or more seeds, the seeds are regularly spaced with seeds on the edges
    # of the domain (i.e., closed support).
    #
    # \returns initializes self.LongTermPoints, self.LongTermMatInds
    def _sampleCoPSSeeds(self):
        if  self.numCoPSSeeds == 1:
            self.LongTermPoints.append(      self.SlabLength / 2                                    )
            self.LongTermMatInds.append(     int(self.Rng.choice(self.nummats,p=self.prob[:]))      )
        elif self.numCoPSSeeds > 1:
            self.LongTermPoints.append(      0.0                                                    )
            self.LongTermMatInds.append(     int(self.Rng.choice(self.nummats,p=self.prob[:]))      )

            seedspacing = self.SlabLength / (self.numCoPSSeeds - 1)
            
            for i in range(1,self.numCoPSSeeds):
                self.LongTermPoints.append(  self.LongTermPoints[-1] + seedspacing                  )
                self._TwoPtCorrelation(          seedspacing , self.LongTermMatInds[-1]             )
                self.LongTermMatInds.append( int(self.Rng.choice(self.nummats,p=self.condprobs[:])) )


    ## \brief Initializes points and material indices--erases any memory
    #
    # Note: Numpy arrays don't start blank and concatenate, where lists do, therefore plan to use lists,
    # but convert to arrays using np.asarray(l) if needed to use array format
    # "LongTerm" refers to storage of point location and material index until the end of a cohort
    # "Recent" refers to storage of the N most recent points not committed to long term memory
    #
    # \returns initializes self.LongTermPoints, self.LongTermMatInds, self.RecentPoints, and self.RecentMatInds
    def _initializeGeometryMemory(self):
        if self.GeomType == "CoPS":
            self.RecentPoints   = []; self.RecentMatInds = []
            self.LongTermPoints = []; self.LongTermMatInds = []
            if self.numCoPSSeeds > 0: self._sampleCoPSSeeds()

    ## \brief Concatenates recent and long-term points and material indices for use in CoPS
    #
    # \returns sets self.MatPoints and self.MatInds
    def _prepMatPoints_and_Inds(self):
        self.MatPoints = self.RecentPoints  + self.LongTermPoints
        self.MatInds   = self.RecentMatInds + self.LongTermMatInds
   
    ## \brief Commits point locations and material indices to memory as user options dictate
    #
    # param[in] x, float, x position in realization
    # param[in] mat, int, material type at position x
    # param[in] r, float, distance to nearest pre-stored point
    def _appendMatPoint(self,x,mat):
        if self.flLongTermMemory:
            r = min( list(map(abs, list( np.subtract(self.MatPoints,x) )))) if len(self.MatPoints)>0 else np.inf
            if r > self.amnesiaRadius:
                self.LongTermPoints.append(   x  )
                self.LongTermMatInds.append( mat )
                return
        if self.recentMemory > 0:
            self.RecentPoints.append(   x  )
            self.RecentMatInds.append( mat )
            if len(self.RecentPoints) > self.recentMemory:
                del[ self.RecentPoints[ 0] ]
                del[ self.RecentMatInds[0] ]
 
    ## \brief Simulates one particle using function-based Woodcock Monte Carlo solver. Intent: private.
    #
    # Simulates one particle history using geometry defined by a function which returns
    # necessary radiation transport quantities.
    # Currently uses normally incident beam source on left boundary and isotropic scattering
    # and tallies transmittance, reflectance, and absorption.
    def _pushCoPSParticle(self):
        mu  = self._setParticleSourceAngle()
        x   = self._initializePosition()
        mat = None #if particle streams without colliding, this is needed to satisfy syntax
        self.XSCeiling = max(self.totxs) #majorant total cross section
            
        flbreak = False
        while True:
            #solve distance to potential collision and distance to boundary
            dpc = -np.log( self.Rng.rand() ) / self.XSCeiling * abs(mu)
            db  = self.SlabLength-x if mu>0.0 else x
            #choose potential collision or boundary crossing
            if   self.isleq(abs(db),abs(dpc)): dx = db ; flpcollide = False
            else                             : dx = dpc; flpcollide = True
            #stream particle
            x = x + dx if mu>=0.0 else x - dx
            #if external boundary crossed, tally and terminate
            if   self.isclose(x,0.0)            : self.R+=1.0; x = 0.0            ; flbreak = True
            elif self.isclose(x,self.SlabLength): self.T+=1.0; x = self.SlabLength; flbreak = True
            
            #evaluate potential collision
            if flpcollide:
                self._prepMatPoints_and_Inds()
                if  self.MatPoints == []:
                    self.condprobs = self.prob[:]
                    r = np.inf
                else:
                    #2 pt correlation
                    if self.numcondprobpts == 1:
                        dists   = list(map(abs, list( np.subtract(self.MatPoints,x) )))
                        index   = dists.index( min(dists) )
                        r       = dists[index] if not self.fl3DEmulation else dists[index]/abs(mu) #3DEmulation calculation handled here
                        mat     = self.MatInds[index]
                        self._TwoPtCorrelation(r,mat)
                    #3 pt correlation
                    elif self.numcondprobpts == 2:
                        dists   = list( np.divide( 1.0, np.subtract(self.MatPoints,x) ) )
                        indexL  = dists.index( min(dists) )
                        indexR  = dists.index( max(dists) )
                        #point on both sides
                        if dists[indexL]*dists[indexR] < 0.0:
                            r1    = abs( 1.0/dists[indexL] )
                            r2    = abs( 1.0/dists[indexR] )
                            mat1  = self.MatInds[indexL]
                            mat2  = self.MatInds[indexR]
                            self._ThreePtCorrelation(r1,r2,mat1,mat2)
                        #point on one side, dists is the reciprocal
                        else:
                            if abs(dists[indexL]) > abs(dists[indexR]): index = indexL
                            else                                      : index = indexR
                            r     = abs( 1.0/dists[index] )
                            mat   = self.MatInds[index]
                            self._TwoPtCorrelation(r,mat)
                #determine material
                mat = self.Rng.choice(self.nummats,p=self.condprobs[:])
                self._appendMatPoint(x,mat)
                #if distance to potential collision is chosen, determine if real collision
                if self.Rng.rand() <= self.totxs[mat]/self.XSCeiling: #accepted collision
                    if self.Rng.rand() > self.scatrat[mat]: self.A+=1.0; flbreak = True  #absorb
                    else                                  : mu = 2.0*self.Rng.rand()-1.0                #scatter      
            #tally flux
            oldx = x; oldmu = mu;
            if self.flTally: self.FluxTalObj._tallyFlux(oldx=oldx,x=x,mu=oldmu,iseg=mat,collinfo=(flpcollide,self.XSCeiling) )
            #if particle absorbed or leaked, kill history
            if flbreak: break
