#!usr/bin/env python
import sys
from Geometry_Basepy import Geometry_Base
sys.path.append('/../../Classes/Tools')
from ClassToolspy import ClassTools
from MarkovianInputspy import MarkovianInputs
from CPF_MarkovianAnalyticpy import CPF_MarkovianAnalytic
from CPF_MarkovianCombinationpy import CPF_MarkovianCombination
from CPF_MultipleIndicatorCoKrigingpy import CPF_MultipleIndicatorCoKriging
import numpy as np
import matplotlib.pyplot as plt
import warnings

## \brief Multi-D CoPS geometry class
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
# \author Alec Shelley, ams01@stanford.edu
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
        if self.flsaveplot == True: plt.savefig(''+str(self.GeomType)+'_'+str(self.Part.numDims)+'D_Case-name-_Part'+str(ipart+1)+'.png')
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


#########################################################################################################################################
    ## \brief Sets method for conditional probability evaluation
    #
    # \param[in] conditionalProbEvaluator, str, 'MarkovianAnalytic', 'MarkovianCombination', or 'MultipleIndicatorCoKriging'
    # \returns sets self.CPF
    def defineConditionalProbabilityEvaluator(self,conditionalProbEvaluator=None):
        if   conditionalProbEvaluator=='MarkovianAnalytic'         : self.CPF = CPF_MarkovianAnalytic()
        elif conditionalProbEvaluator=='MarkovianCombination'      : self.CPF = CPF_MarkovianCombination()
        elif conditionalProbEvaluator=='MultipleIndicatorCoKriging': self.CPF = CPF_MultipleIndicatorCoKriging()
        else : raise Exception("Please choose 'MarkovianAnalytic' or 'MarkovianCombination' or 'MICK' for conditionalProbEvaluator")

    ## \brief Veneer to interface with method of the CPF class method of the same name
    # \returns passes params to method of the CPF class of the same name
    def defineMixingParams(self,*args):
        self.CPF.defineMixingParams(*args)
        self.prob = self.CPF.prob[:]
        self.lam  = self.CPF.lam[:]

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
        else                                                                         : self.condprobs = self.CPF.returnConditionalProbabilities([self.Part.x,self.Part.y,self.Part.z],self.govpoints,self.govmatinds) #Else use the selected governing points to compute conditional probabilities
#########################################################################################################################################

    ## \brief User specifies conditional sampling parameters
    #
    # \param[in] maxNumPoints int, maximum number of points use to compute conditional probability function
    # \param[in] maxDistance float, maximum distance from new point existing point may be to contribute to prob func
    # \param[in] exlusionMultiplier float, exclusion angle defined using triangle with 'opposite' length of exMult*lam[m]
    # \param[in] flRefillToMaxPoints bool, keep points that would otherwise be excluded to meet maxNumPoints if possible (can be useful when CPF evaluation has little to no error)
    # \returns sets these parameters as object attributes
    def defineConditionalSamplingParameters(self,maxNumPoints=None,maxDistance=None,exclusionMultiplier=None,flRefillToMaxPoints=False):
        assert isinstance(maxNumPoints,int) and maxNumPoints>=0
        assert isinstance(maxDistance,float) and maxDistance>=0.0
        assert isinstance(exclusionMultiplier,float) and exclusionMultiplier>=0.0
        assert isinstance(flRefillToMaxPoints,bool)
        self.maxNumPoints        = maxNumPoints
        self.maxDistance         = maxDistance
        self.exclusionMultiplier = exclusionMultiplier
        self.flRefillToMaxPoints = flRefillToMaxPoints

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

        if self.flRefillToMaxPoints:
            #if point refill option active and number of points <= maxNumPoints, use all points
            if  len(distances)<=self.maxNumPoints:
                self.govpoints  = [list(pt) for pt in allpoints]
                self.govmatinds = allmatinds[:]
                self.govdists   = distances[:]
                if self.govdists: self.distNearestPoint = min(self.govdists)
                return
            #if point refill option active and number of points > maxNumPoints, create pristine copy of point distances for use in later refill operation
            distances0 = distances[:]

        #search through all pre-existing points using exclusion rules to choose governing points
        self.govpoints = []; self.govmatinds = []; self.govdists = []
        while True:      #while loop to test and accept successive points
            #select next candidate point
            idmin = distances.index( min(distances) )         #find index of smallest distance
            self.distNearestPoint = distances[idmin]          #store as attribute for use with amnesia radius
            if self.distNearestPoint>=self.maxDistance: break #stop the search for governing points if next nearest point too far (will trigger if all points already searched since value will be np.inf)

            #add candidate point to governing point list, exclude points based on it
            self.govpoints.append(   list(allpoints[idmin][:]) )
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
    
        #if point refill option active and current number of governing points < maxNumPoints, 'refill' by adding points back until maxNumPoints acquired
        if self.flRefillToMaxPoints and len(self.govpoints)<self.maxNumPoints:
            # find all not yet chosen
            extras = [i for i in range(len(distances0))
                        if list(allpoints[i]) not in self.govpoints]
            # sort by distance (closest first)
            extras.sort(key=lambda i: distances0[i])
            # append just enough to hit the quota
            for idx in extras[: self.maxNumPoints - len(self.govpoints)]:
                self.govpoints.append(list(allpoints[idx]))
                self.govmatinds.append(allmatinds[idx])
                self.govdists.append(distances0[idx])


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