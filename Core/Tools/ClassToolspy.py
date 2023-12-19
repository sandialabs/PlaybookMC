#!usr/bin/env python
import numpy as np

## \brief Tools to be used in several classes, can inherit from here.
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
# Tool 1: Evaluate floating point comparisons within tolerance
# Tool 2: (could put random number object here)
class ClassTools(object):
    def __init__(self):
        self.rtol                = 0.000000000001 #roundoff tolerance, 10^(-12)

    def __str__(self):
        return str(self.__dict__)

    ## \brief Evaluate whether floats are equal within tolerance
    #
    # \param[in] a,b floats, floats to compare
    # \returns boolean
    def isclose(self,a,b):
        return a < b+self.rtol and a > b-self.rtol

    ## \brief Evaluate whether a is <= b within tolerance
    #
    # \param[in] a float, number that might be smaller within tolerance
    # \param[in] b float, number that might be larger  within tolerance
    # \returns boolean
    def isleq(self,a,b):
        if   a<b              : return True
        elif self.isclose(a,b): return True
        else                  : return False

    ## \brief Evaluate whether a is >= b but within tolerance equal to b
    #
    # \param[in] a float, number that might be smaller within tolerance
    # \param[in] b float, number that might be larger  within tolerance
    # \returns boolean
    def isclose_and_geq(self,a,b):
        if a<=b and a<=b+self.rtol: return True
        else                      : return False
