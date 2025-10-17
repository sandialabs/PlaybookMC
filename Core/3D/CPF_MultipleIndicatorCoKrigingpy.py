from CPF_Basepy import CPF_Base
import VoxelSpatialStatisticspy
import numpy as np

## \brief Conditional probability function evaluator for multiple indicator co-kriging (MICK) approach
# \author Dan Bolintineanu, dsbolin@sandia.gov
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
class CPF_MultipleIndicatorCoKriging(CPF_Base):
    def __init__(self):
        super(CPF_MultipleIndicatorCoKriging,self).__init__()
        #Defaults for parameters that can be set via defineConditionalSamplingParameters
        self.flMickNormalize = True
        self.anisotropic = False
        self.ndim = 3

    def __str__(self):
        return str(self.__dict__)
    
    ## \brief Defines mixing parameters (S2s and probabilities)
    #
    # \param[in] stats, StatMetrics object; or, nmat X nmat array of callables and nmat X 1 list of prob
    # \returns sets nmat X nmat array of callable S2 objects
    def defineMixingParams(self, *args):
        if isinstance(args[0], VoxelSpatialStatisticspy.StatMetrics):            
            stats = args[0]
            if not hasattr(stats, "SplineFitsToS2RadialAverage"):
                stats.fitSplinesToS2RadialAverage()
            self.s2 = stats.SplineFitsToS2RadialAverages
            self.prob = np.array(list(stats.MaterialAbundances.values()))
            self.nmat = len(stats.matInds)
        else: #Expects array of S2 callables for all combinations of materials, material abundances
            assert len(args) == 2
            self.s2 = args[0]
            self.prob = args[1]
            self.nmat = len(self.prob)
            for i in np.arange(self.nmat):
                for j in np.arange(self.nmat):
                    assert callable(self.s2[i,j])
        self.lam = np.zeros(self.nmat) #Dummy definition to work with Geometry_CoPSpy.defineMixingParams base class function
            
    ## \brief Set parameters associated with MICK calculation
    #
    #\param[in] flAnisotropic : bool, flag indicating whether anisotropy is relevant
    #\param[in] flMickNormalize: bool, flag indicating whether or not to normalize probability estimates.
    #\param[in] ndim: int, dimensionality of the problem
    def defineConditionalSamplingParameters(self, flAnisotropic=False, flMickNormalize=True, ndim=3):
        self.anisotropic = flAnisotropic
        self.flMickNormalize = flMickNormalize
        self.ndim = ndim
        
    ## \brief Computes conditional probabilities from governing points using MICK approach
    #
    # \returns sets condprob list of floats, probabilities of selecting each material type
    def returnConditionalProbabilities(self,locpoint,govpoints,govmatinds):
        locpoint = np.array(locpoint)
        govpoints = np.array(govpoints)
        govmatinds = np.array(govmatinds)

        m = self.nmat-1
        num_points = len(govpoints)
        K = np.empty((num_points*m, num_points*m))
        B = np.empty((num_points*m, m))  
        #Check that all governing points have the correct dimensionality
        for p in govpoints:
            p = np.array(p)  
            assert len(p) == self.ndim
        #Loop over points, material combinations to populate MICK matrices
        for i, i_point in enumerate(govpoints):
            distance_i0 = self.distance_function(i_point, locpoint)
            #print("d_i0:", distance_i0)
            for I in np.arange(0,m):
                for J in np.arange(0,m):
                    B[i*m+I, J] = self.s2[I,J](distance_i0) - self.prob[I]*self.prob[J]                                                                        
                    for j, j_point in enumerate(govpoints):  
                        if i != j:          
                            distance_ij = self.distance_function(i_point, j_point)
                            K[i*m+I, j*m+J] = self.s2[I,J](distance_ij) - self.prob[I]*self.prob[J]
                        else:
                            if I == J:
                                K[i*m+I, j*m+J] = self.prob[I]*(1-self.prob[I])
                            else:
                                K[i*m+I, j*m+J] = -self.prob[I]*self.prob[J]
        #Computation of weights for MICK estimator        
        w = np.linalg.inv(K)@B    
        #One-hot encoding for final evaluation of MICK estimate of probabilities
        codes = np.zeros((num_points, m))
        for c,p in zip(codes, govmatinds):
            if p != self.nmat-1:
                c[p] = 1
        probabilities = np.copy(self.prob[:-1])
        for i in np.arange(0, num_points):
            weights = w[m*i:m*i+m,:]            
            probabilities += np.matmul(codes[i]-self.prob[:-1], weights)      

        if self.flMickNormalize: #Enforces constraints of probabilities due to small errors in MICK estimates
            probabilities[probabilities < 0] = 0
            probabilities[probabilities > 1] = 1
            ptot = np.sum(probabilities)
            if ptot >= 1:
                last = 0
            else:
                last = 1-ptot
            probabilities = np.append(probabilities, last)
        return probabilities 

    ## \brief Returns distance measure for MICK calculations
    #
    #  The distance measure is the argument for S2 evaluation in MICK calculations. For the isotropic case, this is simply the distance
    #  between two points; for general anisotropy, the vector between two points is returned.
    #
    # \returns distance or vector between points
    def distance_function(self, p1, p2):
        vec = p1-p2
        if self.anisotropic: return vec
        else               : return np.linalg.norm(vec)    
