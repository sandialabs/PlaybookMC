#!usr/bin/env python
import sys
sys.path.append('/../../Core/Tools')
from MonteCarloParticleSolverpy import MonteCarloParticleSolver
from MarkovianInputspy import MarkovianInputs
import numpy as np
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
    # \param[in] NumParticlesPerSample int, default 1; number of particles per r.v. sample or batch; if ==1, tallies simplified
    def __init__(self,NumParticlesPerSample=1):
        super(SpecialMonteCarloDrivers,self).__init__()
        assert isinstance(NumParticlesPerSample,int) and NumParticlesPerSample>0
        self.NumParticlesPerSample = NumParticlesPerSample
        
    def __str__(self):
        return str(self.__dict__)

    ## \brief User sets recent memory, amnesia radius, and presampled point parameters for CoPS
    #
    # Presampled points can be used to create a partially defined cohort before starting the CoPS simulation.
    # If flPreSampledOnly set to True, no new points will be added to long-term memory during the CoPS simulation
    # and long-term memory lookups will make use of the regular point spacing to be faster.
    #
    # \param[in] recentMemory, int, number of points to store in recent memory
    # \param[in] amnesiaRadius, float, distance from remembered point in which to not commit new point to long-term memory
    # \param[in] flLongTermMemory, bool, whether or not long-term memory will be used
    # \param[in] numpresampled, int, 0 or >=2, number of evenly spaced points to be sampled before the first history on each cohort
    # \param[in] flPreSampledOnly, bool, default False, only allow presampled points for long-term memory (and leverage regular spacing to get speedier lookups)?
    # \param[in] longTermMemoryMode, str 'on'; 'off','on','presampledonly'
    # \returns sets various values to define limited-memory behavior
    def associateLimitedMemoryParam(self,recentMemory,amnesiaRadius,numpresampled=0,longTermMemoryMode='on'):
        assert (isinstance(recentMemory,int) or recentMemory==np.inf) and recentMemory  >= 0
        self.recentMemory     = recentMemory
        assert (isinstance(amnesiaRadius,float) and amnesiaRadius >= 0.0) or amnesiaRadius==None
        self.amnesiaRadius    = amnesiaRadius
        assert isinstance(numpresampled,int) and (numpresampled==0 or numpresampled>=2)
        self.numPreSampledPts = numpresampled
        assert longTermMemoryMode in {'on','off','presampledonly'}
        self.longTermMemoryMode = longTermMemoryMode
        if longTermMemoryMode=='on'                       and     amnesiaRadius==None: raise Exception("If long-term memory is enabled, a non-negative floating point value must be provided for the amnesia radius.")
        if longTermMemoryMode in {'off','presampledonly'} and not amnesiaRadius==None: warnings.warn("A value was provided for the amnesiaRadius which will not be used with selected long-term memory options")
        if longTermMemoryMode=='presampledonly': assert(numpresampled>=2)
        if longTermMemoryMode=='presampledonly':
            self._findNearestLongTermPoint  = self._findNearestLongTermPoint_grid
            self._findNearestLongTermPoints = self._findNearestLongTermPoints_grid
            self._pointOnBothSides          = self._pointOnBothSides_grid
        else                    :
            self._findNearestLongTermPoint  = self._findNearestLongTermPoint_nogrid
            self._findNearestLongTermPoints = self._findNearestLongTermPoints_nogrid
            self._pointOnBothSides          = self._pointOnBothSides_nogrid
        if self.fl3DEmulation and numpresampled>0: raise Exception("Dimensional emulation and presampled points cannot be used at the same time with CoPS")

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
        self.xmin         = 0.0
        self.xmax         = self.SlabLength

        # Solve and store material probabilities, mean chord length, and correlation length
        self.NaryType = NaryType
        assert isinstance(lam,list) and len(lam)==self.nummats
        for i in range(0,self.nummats): assert isinstance(lam[i],float) and lam[i]>0.0
        self.solveNaryMarkovianParamsBasedOnChordLengths(lam)
        if self.NaryType == 'uniform': del[self.lamc] #lamc is meaningless for non-volume-fraction transition matrices. Note: refactor to handle non-volume-fraction transition matrices better is in order.

    ## \brief Chooses brute-force method and realization generation method
    #
    # Currently restricted to using standard tracker (as opposed to Woodcock tracker) method.
    # Could be generalized to enable either.
    #
    # \param[in] realGenMethod, str, default 'sampleChords'; 'sampleChords' or 'samplePseduInterfaces' to choose Markovian realization generation method
    # \param[in] abundanceModel, str, default 'ensemble'; use 'ensemble' or individual 'sample' material abundances to normalize material fraction in flux bins for material-dependent flux values
    def chooseBruteForcesolver(self, realGenMethod = 'sampleChords', abundanceModel = 'ensemble' ):
        self.GeomType = 'BruteForce'
        self._initializeHistoryGeometryMemory = self._initializeGeometryMemory_pass
        self._initializeSampleGeometryMemory  = self._initializeGeometryMemory_BruteForce
        self.fPushParticle = self._pushMCParticle
        assert realGenMethod in {'sampleChords','samplePseudoInterfaces'}
        if   realGenMethod=='sampleChords'          : self.populateRealization = self.Slab.populateMarkovRealization
        elif realGenMethod=='samplePseudoInterfaces': self.populateRealization = self.Slab.populateMarkovRealizationPseudo
        assert abundanceModel in {'ensemble','sample'}
        self.abundanceModel = abundanceModel

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
        self._initializeHistoryGeometryMemory = self._initializeGeometryMemory_pass
        self._initializeSampleGeometryMemory  = self._initializeGeometryMemory_pass
        self.fPushParticle = self._pushCLSParticle
        assert isinstance(fl3DEmulation,bool)
        self.fl3DEmulation = fl3DEmulation
        self.abundanceModel = 'ensemble'

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
        self._initializeHistoryGeometryMemory = self._initializeGeometryMemory_pass
        self._initializeSampleGeometryMemory  = self._initializeGeometryMemory_pass
        self.fPushParticle = self._pushCLSParticle
        assert isinstance(fl3DEmulation,bool)
        self.fl3DEmulation = fl3DEmulation
        self.abundanceModel = 'ensemble'

    ## \brief Chooses AlgC method 
    def chooseAlgCsolver(self):
        self.GeomType = 'AlgC'
        self._initializeHistoryGeometryMemory = self._initializeGeometryMemory_pass
        self._initializeSampleGeometryMemory  = self._initializeGeometryMemory_pass
        self.fPushParticle = self._pushCLSParticle
        self.fl3DEmulation = False #3DEmulation not possible for AlgC
        self.abundanceModel = 'ensemble'
        
    ## \brief Chooses CoPS method
    #
    # 3-Dimensional Emulation allows for the material to change as a function of 
    # distance traveled by a particle on the flight path, even if not in x direction. It aims to 
    # provide 3D results while maintaining the efficiency of 1D transport. This option can only
    # be used when conditional on one the most recent point for CoPS.
    #
    # \param[in] numcondprobpts, int, amount of points to use for CPF calculation. p-1 in CoPS notation
    # \param[in] fl3DEmulation, bool, default False, flag for 3-Dimensional Emulation
    def chooseCoPSsolver(self, numcondprobpts, fl3DEmulation = False):
        self.GeomType = 'CoPS'
        self._initializeHistoryGeometryMemory = self._initializeHistoryGeometryMemory_CoPS
        self._initializeSampleGeometryMemory  = self._initializeSampleGeometryMemory_CoPS
        self.fPushParticle = self._pushCoPSParticle
        assert isinstance(fl3DEmulation,bool)
        if fl3DEmulation and numcondprobpts == 2:
            numcondprobpts = 1
            print('numcondprobpts changed to 1 in order to support 3DEmulation functionality')
        self.numcondprobpts = numcondprobpts
        self.fl3DEmulation = fl3DEmulation  
        self.abundanceModel = 'ensemble'

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
        x   = self._initializePosition()
        mat = int( self.Rng.choice( self.nummats, p=self.prob[:] ) ) #material designation
        if   self.GeomType=='LRP' : self.dip = self._newdi(mat); self.dim = self._newdi(mat)
        elif self.GeomType=='AlgC': self.dip = self._newdi(mat); self.dim = 0.0; self.dipp = self._newdi(self._changeMat(mat)); self.dimm = 0.0
        flbreak = False
        while True:
            oldx = x; self.oldmu = self.mu; oldmat = mat; flcollide = False
            #dist to interface, boundary, collision
            if   self.GeomType=='CLS'                         : di = self._newdi(oldmat) 
            elif self.GeomType=='LRP' or self.GeomType=='AlgC': di = self.dip
            if   self.fl3DEmulation                           : di *= abs(self.mu) #di scaled to flight path rather than x-axis
            db = self.SlabLength - x if self.mu>0.0 else x
            dc = -np.log( self.Rng.rand() ) / self.totxs[oldmat] * abs(self.mu)
            #stream particle, choose collision or boundary crossing
            dist = min(di,db,dc)
            x = x + dist if self.mu>=0.0 else x - dist
            #evaluate collision or boundary crossing
            if   db == dist: #boundary
                if   self.isclose(x,0.0)            : self.Tals[-1]['Reflect'] +=1
                elif self.isclose(x,self.SlabLength): self.Tals[-1]['Transmit']+=1
                else                                : assert(False)
                flbreak = True
            elif di == dist: #interface
                 mat = self._changeMat(oldmat)
                 if   self.GeomType=='LRP': #sample new interface for dip, set dim to 0
                     self.dip = self._newdi(mat); self.dim = 0.0
                 elif self.GeomType=='AlgC': #sample new interface for dipp, adjust all other stored interfaces as necessary
                     self.dimm = self.dim + self.dip; self.dim = 0.0; self.dip = self.dipp; self.dipp = self._newdi(self._changeMat(mat))
            else           : #collision
                flcollide = True
                if self.Rng.rand() > self.scatrat[oldmat]: self.Tals[-1]['Absorb']+=1; flbreak = True #absorb
                else: #scatter
                    if self.GeomType in {'LRP','AlgC'}: self._incrementDistances(dist)
                    self.oldmu = self.mu   #store old mu, sample new, switch dim/dip if direction switched
                    self.mu = 2.0*self.Rng.rand()-1.0
                    if self.GeomType in {'LRP','AlgC'}: self._conditionallySwitchDistances()
            #tally flux
            self._tallyFluxContribution(oldx,x,self.oldmu,oldmat,flcollide,self.totxs[oldmat])
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

    ## \brief Samples "presampled" points for a cohort
    #
    # If only one presampled point, it is in the center of the domain.
    # If two or more, they are regularly spaced with points on the edges
    # of the domain (i.e., closed support).
    #
    # \returns initializes self.LongTermPoints, self.LongTermMatInds
    def _preSampleCoPSCohortPoints(self):
        self.LongTermPoints.append(      0.0                                                    )
        self.LongTermMatInds.append(     int(self.Rng.choice(self.nummats,p=self.prob[:]))      )

        self.presampledwidth = self.SlabLength / (self.numPreSampledPts - 1)
        
        for i in range(1,self.numPreSampledPts):
            self.LongTermPoints.append(  self.LongTermPoints[-1] + self.presampledwidth         )
            self._TwoPtCorrelation(      self.presampledwidth    , self.LongTermMatInds[-1]     )
            self.LongTermMatInds.append( int(self.Rng.choice(self.nummats,p=self.condprobs[:])) )


    ## \brief Samples realization of stochastic media for use with the 'brute force' solver method
    #
    # \returns calls methods in OneDSlab to sample new Markovian stochastic media realization
    def _initializeGeometryMemory_BruteForce(self):
        self.populateRealization(totxs=self.totxs[:],lam=self.lam[:],s=self.SlabLength,scatxs=self.scatxs[:],NaryType=self.NaryType)

    ## \brief Initializes short-term points and material indices for CoPS (starts history)--erases memory
    #
    # \returns initializes self.RecentPoints and self.RecentMatInds
    def _initializeHistoryGeometryMemory_CoPS(self):
        self.RecentPoints   = []; self.RecentMatInds = []

    ## \brief Initializes long-term memory points and material indices for CoPS (starts cohort)--erases memory
    #
    # \returns initializes self.LongTermPoints and self.LongTermMatInds
    def _initializeSampleGeometryMemory_CoPS(self):
        self.LongTermPoints = []; self.LongTermMatInds = []
        if self.numPreSampledPts > 0: self._preSampleCoPSCohortPoints()

    ## \brief Commits point locations and material indices to memory as user options dictate
    #
    # param[in] x, float, x position in realization
    # param[in] mat, int, material type at position x
    def _appendMatPoint(self,x,mat):
        if self.longTermMemoryMode=='on':
            if self.min_r > self.amnesiaRadius:
                self.LongTermPoints.append(   x  )
                self.LongTermMatInds.append( mat )
                return
        if self.recentMemory > 0:
            self.RecentPoints.append(   x  )
            self.RecentMatInds.append( mat )
            if len(self.RecentPoints) > self.recentMemory:
                del[ self.RecentPoints[ 0] ]
                del[ self.RecentMatInds[0] ]
 

    ## \brief Finds nearest point in recent memory to current location x
    # 
    # \param[out] r, float, distance to nearest point
    # \param[out] mat, int, index of material type of nearest point
    # \returns r, mat
    def _findNearestRecentPoint(self):
        if len(self.RecentPoints) > 0:
            dists   = list(map(abs, list( np.subtract(self.RecentPoints,self.x) )))
            index   = dists.index( min(dists) )
            r       = dists[index] if not self.fl3DEmulation else dists[index]/abs(self.mu) #3DEmulation calculation handled here
            mat     = self.RecentMatInds[index]
            return r,mat
        else:
            return np.inf,-1
        
    ## \brief Finds nearest point in long-term memory to current location x using general search approach
    # 
    # \param[out] r, float, distance to nearest point
    # \param[out] mat, int, index of material type of nearest point
    # \returns r, mat, sets self.min_r (minimum radius to long-term point, to compare amnesia radius with)
    def _findNearestLongTermPoint_nogrid(self):
        if len(self.LongTermPoints) > 0:
            dists   = list(map(abs, list( np.subtract(self.LongTermPoints,self.x) )))
            index   = dists.index( min(dists) )
            r       = dists[index] if not self.fl3DEmulation else dists[index]/abs(self.mu) #3DEmulation calculation handled here
            mat     = self.LongTermMatInds[index]
            self.min_r = r
            return r,mat
        else:
            self.min_r = np.inf
            return np.inf,-1

    ## \brief Finds nearest point in long-term memory to current location x using grid-based search approach
    #
    # Note: With current options, long-term memory is only expected to be on a regular grid
    # when using longTermMemoryMode == 'presampledonly'.
    # 
    # \param[out] r, float, distance to nearest point
    # \param[out] mat, int, index of material type of nearest point
    # \returns r, mat
    def _findNearestLongTermPoint_grid(self):
        index = int( np.round( self.x / self.presampledwidth ) )
        r       = abs( self.x - self.LongTermPoints[index] )
        mat     = self.LongTermMatInds[index]
        return r,mat
 
    ## \brief Finds nearest point in recent and long-term memory to current location x
    # 
    # \param[out] r, float, distance to nearest point
    # \param[out] mat, int, index of material type of nearest point
    # \returns r, mat
    def _findNearestPoint(self):
        pt_rec  = self._findNearestRecentPoint()
        pt_long = self._findNearestLongTermPoint()
        pt = pt_rec if pt_rec[0]<pt_long[0] else pt_long
        return pt

    ## \brief Returns whether there is at least one point in recent or long-term memory on each side of current point
    #
    # \returns bool
    def _pointOnBothSides_nogrid(self):
        pt_left  = min( self.LongTermPoints + self.RecentPoints )
        pt_right = max( self.LongTermPoints + self.RecentPoints )
        return True if pt_left<self.x and pt_right>self.x else False

    ## \brief Returns that there are points on both sides of the current point since there always is with the grid feature
    #
    # \returns True
    def _pointOnBothSides_grid(self):
        return True

    ## \brief Finds nearest point in recent memory on each side of current location x using general search approach
    #
    # \param[out] r1 and r2, float, distance to nearest point towards left and right
    # \param[out] mat1 and mat2, int, index of material type of nearest point towards left and right
    # \returns (r1, mat1), (r2, mat2), sets self.min_r (minimum radius to long-term point, to compare amnesia radius with)
    def _findNearestRecentPoints(self):
        if len(self.RecentPoints) > 0:
            dists   = np.divide( 1.0, np.subtract(self.RecentPoints,self.x) )
            indexL  = np.argmin(dists)
            indexR  = np.argmax(dists)
            r1    = abs( 1.0/dists[indexL] )
            r2    = abs( 1.0/dists[indexR] )
            mat1  = self.RecentMatInds[indexL]
            mat2  = self.RecentMatInds[indexR]
            self.min_r = min(r1,r2)
            return (r1,mat1),(r2,mat2)
        else:
            self.min_r = np.inf
            return (np.inf,-1),(np.inf,-1)
        
    ## \brief Finds nearest point in long-term memory on each side of current location x using general search approach
    #
    # \param[out] r1 and r2, float, distance to nearest point towards left and right
    # \param[out] mat1 and mat2, int, index of material type of nearest point towards left and right
    # \returns (r1, mat1), (r2, mat2)
    def _findNearestLongTermPoints_nogrid(self):
        if len(self.LongTermPoints) > 0:
            dists   = np.divide( 1.0, np.subtract(self.LongTermPoints,self.x) )
            indexL  = np.argmin(dists)
            indexR  = np.argmax(dists)
            r1    = abs( 1.0/dists[indexL] )
            r2    = abs( 1.0/dists[indexR] )
            mat1  = self.LongTermMatInds[indexL]
            mat2  = self.LongTermMatInds[indexR]
            return (r1,mat1),(r2,mat2)
        else:
            return (np.inf,-1),(np.inf,-1)

    ## \brief Finds nearest point in long-term memory on each side of current location x using grid-based search approach
    #
    # Note: With current options, long-term memory is only expected to be on a regular grid
    # when using longTermMemoryMode == 'presampledonly'.
    # 
    # \param[out] r1 and r2, float, distance to nearest point towards left and right
    # \param[out] mat1 and mat2, int, index of material type of nearest point towards left and right
    # \returns (r1, mat1), (r2, mat2)
    def _findNearestLongTermPoints_grid(self):
        indexL = int( np.floor( self.x / self.presampledwidth ) )
        r1 = self.x - self.LongTermPoints[indexL]
        r2 = self.LongTermPoints[indexL+1] - self.x
        mat1  = self.LongTermMatInds[indexL]
        mat2  = self.LongTermMatInds[indexL+1]
        return (r1,mat1),(r2,mat2)

    ## \brief Finds nearest point in long-term memory on each side of current location x
    #
    # \param[out] pt_left and pt_right, tuples, each tuple contains distance to nearest point to left or right and index of material type at that point
    # \returns pt_left, pt_right
    def _findNearestPoints(self):
        pt_rec_left ,pt_rec_right  = self._findNearestRecentPoints()
        pt_long_left,pt_long_right = self._findNearestLongTermPoints()
        pt_left  = pt_rec_left  if pt_rec_left[0] <pt_long_left[0]  else pt_long_left
        pt_right = pt_rec_right if pt_rec_right[0]<pt_long_right[0] else pt_long_right
        return pt_left,pt_right

    ## \brief Simulates one particle using function-based Woodcock Monte Carlo solver. Intent: private.
    #
    # Simulates one particle history using geometry defined by a function which returns
    # necessary radiation transport quantities.
    # Currently uses normally incident beam source on left boundary and isotropic scattering
    # and tallies transmittance, reflectance, and absorption.
    def _pushCoPSParticle(self):
        self.mu  = self._setParticleSourceAngle()
        self.x   = self._initializePosition()
        mat      = None #if particle streams without colliding, this is needed to satisfy syntax
        self.XSCeiling = max(self.totxs) #majorant total cross section
            
        flbreak = False
        while True:
            oldx = self.x; oldmu = self.mu
            #solve distance to potential collision and distance to boundary
            dpc = -np.log( self.Rng.rand() ) / self.XSCeiling * abs(self.mu)
            db  = self.SlabLength-self.x if self.mu>0.0 else self.x
            #choose potential collision or boundary crossing
            if   self.isleq(abs(db),abs(dpc)): dx = db ; flpcollide = False
            else                             : dx = dpc; flpcollide = True
            #stream particle
            self.x = self.x + dx if self.mu>=0.0 else self.x - dx
            #if external boundary crossed, tally and terminate
            if   self.isclose(self.x,0.0)            : self.Tals[-1]['Reflect'] +=1; self.x = 0.0            ; flbreak = True
            elif self.isclose(self.x,self.SlabLength): self.Tals[-1]['Transmit']+=1; self.x = self.SlabLength; flbreak = True
            
            #evaluate potential collision
            if flpcollide:
                if  len(self.RecentPoints) + len(self.LongTermPoints) == 0: #     independent (no points for new point to be correlated to)
                    self.min_r     = np.inf
                    self.condprobs = self.prob[:]
                elif self.numcondprobpts == 2 and self._pointOnBothSides(): #3-pt correlation (and at least one point on each side of new point)
                    (r1,mat1),(r2,mat2) = self._findNearestPoints()
                    self._ThreePtCorrelation(r1,r2,mat1,mat2)
                else                                                      : #2-pt correlation (new point correlated to nearest point)
                    r,mat = self._findNearestPoint()
                    self._TwoPtCorrelation(r,mat)
                #determine material
                mat = self.Rng.choice(self.nummats,p=self.condprobs[:])
                self._appendMatPoint(self.x,mat)
                #if distance to potential collision is chosen, determine if real collision
                if self.Rng.rand() <= self.totxs[mat]/self.XSCeiling: #accepted collision
                    if self.Rng.rand() > self.scatrat[mat]: self.Tals[-1]['Absorb']+=1; flbreak = True  #absorb
                    else                                  : self.mu = 2.0*self.Rng.rand()-1.0           #scatter      
            #tally flux
            self._tallyFluxContribution(oldx,self.x,oldmu,mat,flpcollide,self.XSCeiling)
            #if particle absorbed or leaked, kill history
            if flbreak: break