#!usr/bin/env python
import sys
sys.path.append('/../../Classes/Tools')
from ClassToolspy import ClassTools
import numpy as np

## \brief Base class for conditional probability functions which return probability of material type at one point based on known material type at others
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
class CPF_Base(ClassTools):
    def __init__(self):
        super(CPF_Base,self).__init__()

    def __str__(self):
        return str(self.__dict__)
        
    ## \brief Defines mixing parameters to use in conditional probability function evaluations
    # To be provided by non-base class
    def defineMixingParams(self):
        assert 1==0

    ## \brief Computes conditional probabilities from governing points
    # To be provided by non-base class
    def returnConditionalProbabilities(self):
        assert 1==0