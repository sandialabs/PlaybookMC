#!usr/bin/env python
import sys
from Geometry_Basepy import Geometry_Base
sys.path.append('/../../Classes/Tools')
from ClassToolspy import ClassTools
from MarkovianInputspy import MarkovianInputs
import numpy as np
import matplotlib.pyplot as plt
import warnings

## \brief Multi-D CoPS geometry class
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
# A good chunk of what is in here would be general for geometries as well.
# Perhaps in the future those aspects should be abstracted a layer out
# and this class and other geometry classes should inherit from that one.
#
class Geometry_CoPS(Geometry_Base,ClassTools,MarkovianInputs):
    def __init__(self):
        super(Geometry_CoPS,self).__init__()
        self.flshowplot = False; self.flsaveplot = False
        self.flCollectPoints = False
        self.GeomType = 'CoPS'
        self.abundanceModel = 'ensemble'

    def __str__(self):
        return str(self.__dict__)
        
    ## \brief Initializes short-term points and material indices for CoPS (starts history)--erases memory
    #
    # Note: Numpy arrays don't start blank and concatenate, where lists do, therefore plan to use lists,
    # but convert to arrays using np.asarray(l) if needed to use array format
    #
    # \returns initializes self.RecentPoints and self.RecentMatInds
    def _initializeHistoryGeometryMemory(self):
        self.RecentPoints    = []
        self.RecentMatInds   = []

    ## \brief Initializes long-term memory points and material indices for CoPS (starts cohort)--erases memory
    #
    # Note: Numpy arrays don't start blank and concatenate, where lists do, therefore plan to use lists,
    # but convert to arrays using np.asarray(l) if needed to use array format
    #
    # \returns initializes self.LongTermPoints and self.LongTermMatInds
    def _initializeSampleGeometryMemory(self):
        self.LongTermPoints  = []
        self.LongTermMatInds = []

    ## \brief Defines mixing parameters (chord lengths and probabilities)
    #
    # \param[in] lam, list of floats, list of chord lengths
    # \returns sets chord length, probabilities, correlation length, seed locations
    def defineMixingParams(self,lam=None):
        # Assert material chord lengths and slab length and store in object
        assert isinstance(lam,list) and len(lam)==self.nummats
        for i in range(0,self.nummats): assert isinstance(lam[i],float) and lam[i]>0.0
        self.lam = lam

        # Solve and store material probabilities and correlation length
        self.solveNaryMarkovianParamsBasedOnChordLengths(self.lam)

    ## \brief Samples material index of new point, and stores or forgets new point, according to selected rules
    #
    # Calls other method to solve conditional probabilities of material index at new point based on previously
    # stored points.
    # Samples material index of new point based on those conditional probabilities.
    # Stores new points in LongTerm or Recent memory based on user-selected limited-memory options.
    #
    # \returns updates material index and point attributes
    def samplePoint(self):
        self.solveConditionalProbabilities()
        self.CurrentMatInd = self.Rng.choice(self.condprobs)     #make attribute to use in evaluateCollision
     
        if   self.flLongTermMemory and ( len(self.LongTermPoints)==0 or self.distNearestPoint > self.amnesiaRadius ): #if storing long-term memory and pass amnesia radius test, append point to long-term memory
            self.LongTermPoints.append( [self.Part.x,self.Part.y,self.Part.z] )
            self.LongTermMatInds.append( self.CurrentMatInd )
        elif self.recentMemory > 0:                                                                                   #else, if storing recent memory, append point to recent memory
            self.RecentPoints.append( [self.Part.x,self.Part.y,self.Part.z] )
            self.RecentMatInds.append( self.CurrentMatInd )
            if len(self.RecentMatInds) > self.recentMemory:                                                              #if recent memory now too long, delete oldest point
                del self.RecentPoints[0]
                del self.RecentMatInds[0]






    ## \brief Makes 2D plot of CoPS geometry
    #
    # \param[in] ipart, which particle history is being plotted 
    # \returns can show or print plot
    def plotRealization(self,ipart):
        x_mat = []; y_mat = []
        for imat in range(0,self.nummats):
            x_mat.append([])
            y_mat.append([])
        for ipt in range(0,len(self.LongTermPoints)):
            x_mat[ self.LongTermMatInds[ipt] ] = self.LongTermPoints[ipt][0]
            y_mat[ self.LongTermMatInds[ipt] ] = self.LongTermPoints[ipt][1]

        plt.figure
        plt.title('ipart '+str(ipart+1)+'')
        plt.xlabel('x'); plt.ylabel('y')
        for imat in range(0,self.nummats):
            plt.scatter(x_mat[imat],y_mat[imat])
        plt.xlim( self.xbounds[0], self.xbounds[1]);  plt.ylim( self.ybounds[0], self.ybounds[1])
        if self.flsaveplot == True: plt.savefig(''+str(self.geomtype)+'_'+str(self.Part.numDims)+'D_Case-name-_Part'+str(ipart+1)+'.png')
        if self.flshowplot == True: plt.show()


    ## \brief Defines CoPS geometry plotting options
    #
    # \param[in] particlestoplot, list of ints, list integer indices of realizations to plot
    # \param[in] flshowplot, bool, show plots to screen?
    # \param[in] flsaveplot, bool, save plots to files?
    # \returns sets user-selected options
    def defineCoPSGeomPlottingOptions(self,particlestoplot,flshowplot,flsaveplot):
        assert isinstance(particlestoplot,list)
        for i in range(0,len(particlestoplot)): isinstance(particlestoplot[i],int) and particlestoplot[i]>-1 or isinstance(particlestoplot,None)
        self.particlestoplot = particlestoplot
        self.flshowplot = flshowplot
        self.flsaveplot = flsaveplot    


    ## \brief Defines the function or method to be used to evaluate conditional probabilities based on governing points
    #
    # \param[in] conditionalProbEvaluator, function or method, function or method to call when using governing points to give probabilities
    # \returns sets chord length, probabilities, correlation length, seed locations
    def defineConditionalProbabilityEvaluator(self,conditionalProbEvaluator=None):
        assert callable(conditionalProbEvaluator) or conditionalProbEvaluator==None
        if conditionalProbEvaluator==None or conditionalProbEvaluator==self.computeConditionalProbabilitiesFromGoverningPointsUsingPseudoInterfaces:
            self.conditionalProbEvaluator = self.computeConditionalProbabilitiesFromGoverningPointsUsingPseudoInterfaces
        else                             :
            self.conditionalProbEvaluator = conditionalProbEvaluator
            #Brief test of chosen function (does not test all circumstances)
            try:
                testcondprobs = self.conditionalProbEvaluator( [0.0,0.0,0.0], [[0.1,0.3,0.5],[-0.3,2.4,3.0]], [0,1], [1.0/self.nummats]*self.nummats)
            except:
                assert 0==1#raise("Invalid conditional probability evaluator provided")
            assert len(testcondprobs)==self.nummats
            assert isinstance(testcondprobs[0],float) and isinstance(testcondprobs[1],float)
            for imat in range(0,self.nummats):
                assert testcondprobs[imat]>=0.0 and testcondprobs[imat]<=1.0
            assert self.isclose(np.sum(testcondprobs),1.0)


    ## \brief User specifies conditional sampling parameters
    #
    # \param[in] maxNumPoints int, maximum number of points use to compute conditional probability function
    # \param[in] maxDistance float, maximum distance from new point existing point may be to contribute to prob func
    # \param[in] exlusionMultiplier float, exclusion angle defined using triangle with 'opposite' length of exMult*lam[m]
    # \returns sets these parameters as object attributes
    def defineConditionalSamplingParameters(self,maxNumPoints=None,maxDistance=None,exclusionMultiplier=None):
        assert isinstance(maxNumPoints,int) and maxNumPoints>=0
        assert isinstance(maxDistance,float) and maxDistance>=0.0
        assert isinstance(exclusionMultiplier,float) and exclusionMultiplier>=0.0
        self.maxNumPoints        = maxNumPoints
        self.maxDistance         = maxDistance
        self.exclusionMultiplier = exclusionMultiplier

    ## \brief User specified limited memory parameters
    #
    # Defines limited memory options to use.  Recent memory is temporary memory in which the N most recently
    # sampled points are stored and used in CPF evaluations. Amensia radius is the distance from the closest
    # point currently stored in long-term memory that a point must be in order to be stored in long-term memory.
    # To entirely remove long-term memory, change flLongTermMemory to False.
    # Defaults of recentMemory=0, amnesiaRadius=0.0, and flLongTermMemory=True mean that all sampled points
    # are stored in long-term memory.
    # If recentMemory is to be utilized, amnesiaRadius also needs to be set to a higher value or flLongTermMemory
    # needs to be set to False so that at least some points will not be stored to long-term memory (and can be
    # stored in recent memory). The two can be used together with some points being stored to long-term
    # memory and others to temporary memory.
    # 
    # \param[in] recentMemory, int, maximum number of recently sampled point to hold in temporary memory
    # \param[in] amnesiaRadius, float, radius from existing long-term memory point in which to not store new point in long-term memory
    # \param[in] flLongTermMemory, bool, whether or not to accumulate long-term memory (does not control recent memory)
    # \returns sets these parameters as object attributes
    def defineLimitedMemoryParameters(self,recentMemory=0,amnesiaRadius=0.0,flLongTermMemory=True):
        assert (isinstance(recentMemory,int) or recentMemory==np.inf) and recentMemory  >= 0
        assert (isinstance(amnesiaRadius,float) and amnesiaRadius >= 0.0) or amnesiaRadius==None
        assert isinstance(flLongTermMemory,bool)
        if     flLongTermMemory and     amnesiaRadius==None: raise Exception("If long-term memory is enabled, a non-negative floating point value must be provided for the amnesia radius.")
        if not flLongTermMemory and not amnesiaRadius==None: warnings.warn("A value was provided for the amnesiaRadius which will not be used since long-term memory was not enabled")

        self.recentMemory     = recentMemory
        self.amnesiaRadius    = amnesiaRadius
        self.flLongTermMemory = flLongTermMemory

    ## \brief Choose Markovian or AM geometry and true 3D or 1D emulation to simulate with 3D CoPS
    #
    # For the AM CoPS geom type, mix all materials to be one atomically mixed material, set the number of
    # conditional points to be zero, then let CoPS run.
    # Algorithmically, this is different than classical AM, but mathematically it is equivalent.
    # Alternative methods by which to get CoPS to give AM results are to set maxNumPoints to 1
    # (so that conditional probability function evaluations only use one point--the new point)
    # or to make sure no sampled points are saved to short- or long-term memory, for exmaple by 
    # setting recentMemory to 0 and flLongTermMemory to False.
    #
    # 1DEmulation enables 3D CoPS to get 1D planar geometry CoPS results, but currently only works
    # for CoPSp-1 (i.e., recentMemory=1 and flLongTermMemory=False) and for a source oriented to 
    # travel in the z direction.
    #
    # \param[in] geomtype str, 'Markovian', 'AM'
    # \param[in] fl1DEmulation bool, True will only allow material to change along z-dimension (emulates 1D slab behavior)
    # \returns sets self.geomtype and self.fl1DEmulation
    def defineCoPSGeometryType(self,geomtype,fl1DEmulation=False):
        assert geomtype=='Markovian' or geomtype=='AM'
        self.geomtype = geomtype
        assert hasattr(self, 'lamc')
        assert isinstance(fl1DEmulation,bool)
        #if using atomic mix CoPS, create single material with infinite correlation length
        if self.geomtype=='AM':
            assert hasattr(self, 'maxNumPoints')
            self.totxs   = [ np.sum( np.multiply( self.totxs  , self.prob ) ) ]
            self.Majorant= self.totxs[0]
            self.scatxs  = [ np.sum( np.multiply( self.scatxs , self.prob ) ) ]
            self.absxs   = [ self.totxs[0] - self.scatxs[0] ]
            self.scatrat = [ self.scatxs[0] / self.totxs[0] ]
            self.nummats = 1
            self.maxNumPoints = 0
            self.lam     = [100000.0]
            self.lamc    =  100000.0
            self.prob    = [ 1.0 ]
        if fl1DEmulation==True: assert geomtype=='Markovian'
        self.fl1DEmulation = fl1DEmulation

    ## \brief Tests whether candidate points should be excluded based on governing points
    #
    # Use the Law of Cosines to compute angle between the candidate point and each governing point.
    # If any such angle is less than the governing point's exclusion angle, exclude candidate point, else accept.
    #
    # \param[in] candpt list of floats, x,y,z coordinates of candidate point
    # \param[in] dist_candpt float, distance from new point to candidate point
    # \returns bool, whether or not the candidate point should be excluded based on governing opints
    def testForExclusion(self,candpt,dist_candpt,exclusionAngle):
	#distance to governing point
        dist_govpt = np.sqrt( (self.govpoints[-1][0]-self.Part.x)**2 + (self.govpoints[-1][1]-self.Part.y)**2 + (self.govpoints[-1][2]-self.Part.z)**2 )
        #distance between governing point and proposed point
        dist_pts   = np.sqrt( (self.govpoints[-1][0]-candpt[0]  )**2 + (self.govpoints[-1][1]-candpt[1]  )**2 + (self.govpoints[-1][2]-candpt[2]  )**2 )
        #angle between governing point and proposed point
        arccos_arg = ( dist_govpt**2 + dist_candpt**2 - dist_pts**2 ) /  ( 2.0 * dist_govpt * dist_candpt )
        if   arccos_arg<-0.999999999: angle = np.pi
        elif arccos_arg> 0.999999999: angle = 0.0
        else                        : angle = np.arccos( arccos_arg )
        #if angle less than exclusion angle, return True to exlusion, otherwise return False to exclusion
        if angle<exclusionAngle: return True
        else                   : return False

    ## \brief Returns the exclusion angle of a governing point relative to the new point
    #
    # \param[in] m int, index of material type of governing point in question
    # \param[in] dist float, distance from new point to governing point in question
    # \returns float, angle of exclusion for governing point and new point [in radians]
    def returnExclusionAngle(self,m,dist):
        L = self.lam[m] * self.exclusionMultiplier
        return np.arcsin( L / np.sqrt( L**2 + dist**2 ) )

    ## \brief Searches through all defined points and chooses governing points based on user-input selection params
    #
    # Computes the distance between the new point and each pre-existing point
    # Starting with the nearest pre-existing point, test whether to accept as governing
    # point or exclude.
    #
    # \returns sets self.govpoints and self.govmatinds, governing points and their material indices
    def selectGoverningPoints(self):
        #compute distance from new point to each pre-existing point in long-term and recent memory
        allpoints = np.asarray( self.LongTermPoints[:] + self.RecentPoints[:] )
        allmatinds=            self.LongTermMatInds[:] + self.RecentMatInds[:]
        distances =                                   np.power(np.subtract(allpoints[:,0],self.Part.x),2)
        distances =                np.add( distances, np.power(np.subtract(allpoints[:,1],self.Part.y),2) )
        distances = list( np.sqrt( np.add( distances, np.power(np.subtract(allpoints[:,2],self.Part.z),2) ) ) )

        #search through all pre-existing points using exclusion rules to choose governing points
        self.govpoints = []; self.govmatinds = []; self.govdists = []
        while True:      #while loop to test and accept successive points
            #select next candidate point
            idmin = distances.index( min(distances) )         #find index of smallest distance
            self.distNearestPoint = distances[idmin]          #store as attribute for use with amnesia radius
            if self.distNearestPoint>=self.maxDistance: break #stop the search for governing points if next nearest point too far (will trigger if all points already searched since value will be np.inf)

            #add candidate point to governing point list, exclude points based on it
            self.govpoints.append(   allpoints[idmin][:] )
            self.govmatinds.append( allmatinds[idmin]    )
            self.govdists.append(    distances[idmin]    )

            #if gotten enough governing points, exit
            if len(self.govpoints)==self.maxNumPoints: break

            #after first point chosen, exclude all points beyond the maximum distance
            if len(self.govpoints)==1:
                for ipt in range(0,len(distances)):
                    if distances[ipt]<np.inf and distances[ipt]>self.maxDistance:
                        distances[ipt] = np.inf

            #mark points that are now excluded based on new governing point
            exclusionAngle = self.returnExclusionAngle( allmatinds[idmin],distances[idmin] )
            distances[idmin] = np.inf
            for ipt in range(0,len(distances)):
                if distances[ipt]<np.inf:
                    if self.testForExclusion( allpoints[ipt][:],distances[ipt],exclusionAngle ): distances[ipt] = np.inf
                

    ## \brief Returns frequency of at least one psuedo-interface between two points in a Markovian-mixed medium
    #
    # \param[in] igovpt int, which governing point to compute the probability with regard to
    # \returns float, probability of having at least one pseudo-interface
    def freqAtLeastOnePseudointerface_Markovian(self,igovpt):
        if not self.fl1DEmulation: return 1.0 - np.exp( - self.govdists[igovpt]                        / self.lamc )
        else                     : return 1.0 - np.exp( - self.govdists[igovpt] * np.abs(self.Part.mu) / self.lamc )

    ## \brief Computes conditional probabilities from governing points
    #
    # \returns sets condprob list of floats, probabilities of selecting each material type
    def computeConditionalProbabilitiesFromGoverningPointsUsingPseudoInterfaces(self,locpoint,govpoints,govmatinds,abundprob):
        compliments = np.ones( self.nummats )
        #for loop through governing points, collect compliments of new point in same region as each material type
        for igovpt in range(0,len(govpoints)):
            imat = govmatinds[igovpt]
            compliments[imat] *= self.freqAtLeastOnePseudointerface_Markovian(igovpt)
        #compute fraction of combination space for which new point is not in the same cell as any governing points
        frac_newregion = np.prod( compliments )
        #compute fraction of combination space for which new point is in the same cell as governing points of each type
        frac_sameregion = []
        for imat in range(0,self.nummats):
            frac_sameregion.append( frac_newregion/compliments[imat] - frac_newregion )
        #compute probability of sharing cell with no governing point or governing point(s) of each type
        frac_valid = np.sum(frac_sameregion) + frac_newregion
        prob_newregion = frac_newregion / frac_valid
        prob_sameregion= np.divide(frac_sameregion,frac_valid)
        #compute probabilities of sampling each material type
        return np.add( prob_sameregion, np.multiply(prob_newregion,abundprob) )


    ## \brief Computes conditional probabilities from governing points using CoPS-2 errorless CPF
    #
    # \returns sets condprob list of floats, probabilities of selecting each material type
    def computeConditionalProbabilitiesErrorless(self,locpoint,govpoints,govmatinds,abundprob):
        # if there are 2 points, we want to do the full thing. Otherwise, I think we want to just use the other function
        if len(self.govdists) != 2:
            return computeConditionalProbabilitiesFromGoverningPointsUsingPseudoInterfaces(locpoint,govpoints,govmatinds,abundprob)
        #get distances in order to compute probabilities
        imat = govmatinds[igovpt]
        ra = self.govdists[0]
        rb = self.govdists[1]
        rc = np.sqrt((self.govpoints[0][0] - self.govpoints[1][0])**2 + (self.govpoints[0][1] - self.govpoints[1][1])**2 + (self.govpoints[0][2] - self.govpoints[1][2])**2) #distance formula
        ## Calculate probabilities of no material interface along a single side
        Pa = np.exp(-ra/self.lamc)
        Pb = np.exp(-rb/self.lamc)
        Pc = np.exp(-rc/self.lamc)
        ## Probabilities of no material interface along two sides (derived)
        Pab = np.sqrt(Pa*Pb/Pc)
        Pac = np.sqrt(Pa*Pc/Pb)
        Pbc = np.sqrt(Pb*Pc/Pa)
        ## Probabilities of various triangle configurations
        ## In order: none, ab, ac, bc, rest
        P1 = Pab*Pac*Pbc
        P2 = Pac*Pbc - P1
        P3 = Pab*Pbc - P1
        P4 = Pab*Pac - P1
        P5 = 1 - (P1 + P2 + P3 + P4)
        config_probs = {}
        ## Calculate probabilities of each material configuration
        ## Below is from CoPS3PO in 1D. I think we need to add an extra material configuration, probABA. 
        ## I can email you a pdf scan of my work on figuring out which triangle is which. Hopefully it's right
        condprobs = []
        for imat in range(0,self.nummats):
            if       mat1==imat and     imat==mat2: condprobs.append( self._probAAA(self.prob[imat],P1,P2,P3,P4,P5) )
            elif     mat1==imat and not imat==mat2: condprobs.append( self._probAAB(self.prob[imat],self.prob[mat2],P1,P2,P3,P4,P5) )
            elif not mat1==imat and     imat==mat2: condprobs.append( self._probBAA(self.prob[imat],self.prob[mat1],P1,P2,P3,P4,P5) )
            elif not mat1==imat and not imat==mat2: condprobs.append( self._probABC(self.prob[mat1],self.prob[imat],self.prob[mat2],P1,P2,P3,P4,P5) )
        condprobs = np.divide( condprobs, np.sum(condprobs) )

        ## Everything below here is a remnant of the CompareStencilAnswers script, it may or may not be needed
        probsum = 0
        if mat1 == mat2 and mat2 == mat3:
                probsum += probs[mat1] * P1
        if mat1 == mat2:
                probsum += probs[mat1]*probs[mat3] * P4
        if mat1 == mat3:
                probsum += probs[mat1]*probs[mat2] * P2
        if mat2 == mat3:
                probsum += probs[mat2]*probs[mat1] * P3
        probsum += probs[mat1]*probs[mat2]*probs[mat3] * P5
        config_probs[govmatinds[0],govmatinds[1],mat3] = probsum
        # we want this to return a list the length of nummats with conditional probabilities inside I think
			

        return condprobs


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


    ## \brief Finds governing points and solves conditional probabilities (approximation to true conditional probability function)
    #
    # Responsible for calling method to downselect from points to choose governing points (if necessary) and
    # responsible for calling function/method to provide conditional probabilities based on governing points.
    #
    # \returns sets self.condprob list of floats, probabilities of selecting each material type
    def solveConditionalProbabilities(self):
        #If need to, select governing points
        if len(self.LongTermPoints)+len(self.RecentPoints)==0 or self.maxNumPoints==0: self.condprobs = self.prob[:]; return                                                                                          #If no points defined yet or user chose not to depend on points, use material abundances
        else                                                                         : self.selectGoverningPoints()                                                                                                   #Else use the downselection scheme to select the governing points
        #Use governing points to evaluate conditional probabilities
        if len(self.govpoints)==0                                                    : self.condprobs = self.prob[:]; return                                                                                          #If no points are governing points (i.e., all points beyond user-defined maxiumum distance), use material abundances
        else                                                                         : self.condprobs = self.conditionalProbEvaluator([self.Part.x,self.Part.y,self.Part.z],self.govpoints,self.govmatinds,self.prob) #Else use the selected governing points to compute conditional probabilities


    ## \brief Select whether to collect and print points CoPS creates
    #
    # \param[in] flCollectPoints bool, default False, Collect geometry points?
    # \param[in] pointsNpyFileName str, default 'CoPSPoints', Name of file to store points data in
    # \returns sets internal versions of these parameters
    def selectWhetherPrintPoints(self,flCollectPoints=False,pointsNpyFileName='CoPSPoints'):
        assert isinstance(flCollectPoints,bool)
        self.flCollectPoints = flCollectPoints
        if self.flCollectPoints:
            assert isinstance(pointsNpyFileName,str)
            self.pointsNpyFileName = pointsNpyFileName