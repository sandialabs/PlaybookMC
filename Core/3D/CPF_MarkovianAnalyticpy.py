#!/usr/bin/env python
from CPF_Basepy import CPF_Base
from MarkovianInputspy import MarkovianInputs
from CPF_MarkovianAnalyticHelperpy import color_distribution, color_sample
import numpy as np

## \brief Analytic N-point conditional probability function evaluator for 1,2, or 3 dimensions
# \author Alec Shelley, ams01@stanford.edu
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
# 
# \param flSolveDistribution, bool, default True, Solve distribution of colors (or just sample it)
#
class CPF_MarkovianAnalytic(CPF_Base, MarkovianInputs):
    def __init__(self,flSolveDistribution=True):
        super(CPF_MarkovianAnalytic, self).__init__()
        assert isinstance(flSolveDistribution,bool)
        self.flSolveDistribution = flSolveDistribution

    def __str__(self):
        return str(self.__dict__)
        
    ## \brief Defines mixing parameters (if needed for compatibility with PlaybookMC)
    #
    # \param[in] lam, list of floats, list of chord lengths
    # \returns sets chord length, probabilities, correlation length, seed locations
    def defineMixingParams(self, *args):
        lam = args[0]
        assert isinstance(lam, list)
        self.nummats = len(lam)
        for i in range(self.nummats):
            assert isinstance(lam[i], float) and lam[i] > 0.0
        self.lam = lam

        # Solve and store material probabilities and correlation length
        self.solveNaryMarkovianParamsBasedOnChordLengths(self.lam)

    ## \brief Computes conditional probabilities using the analytic solution
    #
    # \param[in] locpoint, np.array, shape(d), coordinates of the point whose CPF we compute
    # \param[in] govpoints, list of np.array, length(n), coordinates of governing points
    # \param[in] govmatinds, list, shape(n) material indices of the governing points
    # \param[in] Rng, RandomNumberspy object, PlaybookMC random number object to pass to CPF_MarkovianAnalyticHelperpy for use in color_sample function
    # \returns np.array of floats, shape(m) probabilities of selecting each material type
    def returnConditionalProbabilities(self, locpoint, govpoints, govmatinds, Rng=None):
        govpoints = np.asarray(govpoints)/self.lamc
        locpoint = locpoint/self.lamc
        #flattens govmatinds to a list if it is a list of lists with one element
        govmatinds = [mat[0] if isinstance(mat, list) and len(mat) == 1\
                       else mat for mat in govmatinds]
        
        # Compute the color distribution via analytic solution; Stack the unknown point with known points, Convert list to tuple for consistency, Use stored material probabilities
        if self.flSolveDistribution: return color_distribution( np.vstack((govpoints, locpoint)), tuple(govmatinds), tuple(self.prob) )
        # Sample from the color distribution (and record sample by saying the conditional probability is 100% that color)
        else                       : return color_sample(       np.vstack((govpoints, locpoint)), tuple(govmatinds), tuple(self.prob), Rng )

if __name__ == "__main__":
    print("=== DEBUGGING CPF_MarkovianAnalytic: Case '1b' ===")
    
    # Load the 1b benchmark case
    inputs = MarkovianInputs()
    inputs.selectALPInputs("1b")

    lam = np.array(inputs.lam)
    fractions = lam / np.sum(lam)

    print(inputs)
    print(f"Chord lengths (λ): {lam.tolist()}")
    print(f"Volume fractions:  {fractions.tolist()}")