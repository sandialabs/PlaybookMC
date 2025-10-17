#!usr/bin/env python
from CPF_Basepy import CPF_Base
from MarkovianInputspy import MarkovianInputs
import numpy as np

## \brief Conditional probability function evaluator for model where independent contributions from each point are combined to approximate multi-point function
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
class CPF_MarkovianCombination(CPF_Base,MarkovianInputs):
    def __init__(self):
        super(CPF_MarkovianCombination,self).__init__()

    def __str__(self):
        return str(self.__dict__)
        
    ## \brief Defines mixing parameters (chord lengths and probabilities)
    #
    # \param[in] lam, list of floats, list of chord lengths
    # \returns sets chord length, probabilities, correlation length, seed locations
    def defineMixingParams(self,*args):
        # Assert material chord lengths and slab length and store in object
        lam = args[0]
        assert isinstance(lam,list)
        self.nummats = len(lam)
        for i in range(0,self.nummats): assert isinstance(lam[i],float) and lam[i]>0.0
        self.lam = lam

        # Solve and store material probabilities and correlation length
        self.solveNaryMarkovianParamsBasedOnChordLengths(self.lam)

    ## \brief Computes conditional probabilities from governing points
    #
    # \returns sets condprob list of floats, probabilities of selecting each material type
    def returnConditionalProbabilities(self,locpoint,govpoints,govmatinds):
        govpoints = np.asarray(govpoints)
        self.govdists =                                       np.power(np.subtract(govpoints[:,0],locpoint[0]),2)
        self.govdists =                np.add( self.govdists, np.power(np.subtract(govpoints[:,1],locpoint[1]),2) )
        self.govdists = list( np.sqrt( np.add( self.govdists, np.power(np.subtract(govpoints[:,2],locpoint[2]),2) ) ) )

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
        return np.add( prob_sameregion, np.multiply(prob_newregion,self.prob) )

    ## \brief Returns frequency of at least one psuedo-interface between two points in a Markovian-mixed medium
    #
    # \param[in] igovpt int, which governing point to compute the probability with regard to
    # \returns float, probability of having at least one pseudo-interface
    def freqAtLeastOnePseudointerface_Markovian(self,igovpt):
        return 1.0 - np.exp( - self.govdists[igovpt] / self.lamc )
