#!/usr/bin/env python
from CPF_Basepy import CPF_Base
from MarkovianInputspy import MarkovianInputs
from CPF_MarkovianAnalyticHelperpy import color_distribution
import numpy as np

## \brief Analytic N-point conditional probability function evaluator for 1,2, or 3 dimensions
# \author Alec Shelley, ams01@stanford.edu
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
# 
class CPF_MarkovianAnalytic(CPF_Base, MarkovianInputs):
    def __init__(self):
        super(CPF_MarkovianAnalytic, self).__init__()

    def __str__(self):
        return str(self.__dict__)
        
    ## \brief Defines mixing parameters (if needed for compatibility with PlaybookMC)
    #
    # \param[in] lam, list of floats, list of chord lengths
    # \returns sets chord length, probabilities, correlation length, seed locations
    def defineMixingParams(self, *args):
        lam = args[0][0]
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
    # \returns np.array of floats, shape(m) probabilities of selecting each material type
    def returnConditionalProbabilities(self, locpoint, govpoints, govmatinds):
        govpoints = np.asarray(govpoints)/self.lamc
        locpoint = locpoint/self.lamc
        #flattens govmatinds to a list if it is a list of lists with one element
        govmatinds = [mat[0] if isinstance(mat, list) and len(mat) == 1\
                       else mat for mat in govmatinds]
        
        # Compute the color distribution using the pre-existing function
        color_probs = color_distribution(
            np.vstack((govpoints, locpoint)),  # Stack the unknown point with known points
            tuple(govmatinds),                 # Convert list to tuple for consistency
            tuple(self.prob)                    # Use stored material probabilities
        )
        return color_probs

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