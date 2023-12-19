#!usr/bin/env python
## \file RandomCoefficientsAndBoundariesBenchmarks.py
#  \brief Example driver to make use of RandCoefsBoundsBenchModule.
#
# The RandCoefsBoundsBenchModule computes moments of the uncollided transmittance through
# a slab of material for a normally incident beam source for which total cross section values,
# material boundary locations, or both are uncertain with a uniform distribution.
# The mathematical models encoded in the module and the inputs to the model used here are 
# presented in OlsonANS2019 where they were used to provide truth solutions to numerical
# experiments exploring variance deconvolution.
# The models provide analytic results for uncertain total cross sections or boundary locations,
# however, the results for problems involving both sources of uncertainty are semi-analytic
# in the sense that numerical integration is used to evaluate the mathematical models.
#
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Benchmarks')
import RandCoefsBoundsBenchModule as Bench
import numpy as np
import pandas as pd

#Define total cross sections, material boundaries, and the magnitude of their uniformly distributed variability
TotXSAves      = [0.9, 0.15, 0.6]
TotXSVars      = [0.7, 0.12, 0.5]

MatBoundAves = [0.0, 2.0 , 5.0 , 6.0]
MatBoundVars = [0.0, 1.75, 0.95, 0.0]

#Translate average material boundaries to segment lengths
SegLens = Bench.MatBoundsToSegLens(MatBoundAves)

#Compute transmittance distribution descriptors based on random coefficients
print('\n                 ***** Random Coefficients *****')
print('                     mom1/mean     mom2      st. dev     variance')
mom1 = Bench.returnMom_UnRandCoefs(TotXSAves,TotXSVars,SegLens,moment=1) 
mom2 = Bench.returnMom_UnRandCoefs(TotXSAves,TotXSVars,SegLens,moment=2)
print('Solve from moments:  {:0.6f},   {:0.6f},   {:0.6f},   {:0.6f}'.format(mom1,mom2,np.sqrt(mom2-mom1**2),mom2-mom1**2))
mean,dev = Bench.returnMeanDev_UnRandCoefs(TotXSAves,TotXSVars,SegLens)
print('Solve directly    :  {:0.6f},      N/A  ,   {:0.6f},   {:0.6f}'.format(mom1,dev,dev**2))

#Compute transmittance distribution descriptors based on random material boundaries
print('\n              ***** Random Material Boundaries *****')
print('                     mom1/mean     mom2      st. dev     variance')
mom1 = Bench.returnMom_UnRandBounds(TotXSAves,MatBoundAves,MatBoundVars,moment=1)
mom2 = Bench.returnMom_UnRandBounds(TotXSAves,MatBoundAves,MatBoundVars,moment=2)
print('Solve from moments:  {:0.6f},   {:0.6f},   {:0.6f},   {:0.6f}'.format(mom1,mom2,np.sqrt(mom2-mom1**2),mom2-mom1**2))
mean,dev = Bench.returnMeanDev_UnRandBounds(TotXSAves,MatBoundAves,MatBoundVars)
print('Solve directly    :  {:0.6f},      N/A  ,   {:0.6f},   {:0.6f}'.format(mom1,dev,dev**2))

#Compare transmittance distribution descriptors based on random coefficients and material boundaries
print('\n         ***** Random Coefficients and Material Boundaries *****')
print('                     mom1/mean     mom2      st. dev     variance')
mom1,mom1err = Bench.returnMom_UnRandCoefsBounds(TotXSAves,TotXSVars,MatBoundAves,MatBoundVars,moment=1)
mom2,mom2err = Bench.returnMom_UnRandCoefsBounds(TotXSAves,TotXSVars,MatBoundAves,MatBoundVars,moment=2)
print('Solve from moments:  {:0.6f},   {:0.6f},   {:0.6f},   {:0.6f}'.format(mom1,mom2,np.sqrt(mom2-mom1**2),mom2-mom1**2))
mean,dev,meanerr,mom2err2 = Bench.returnMeanDev_UnRandCoefsBounds(TotXSAves,TotXSVars,MatBoundAves,MatBoundVars)
print('Solve directly    :  {:0.6f},      N/A  ,   {:0.6f},   {:0.6f}'.format(mom1,dev,dev**2))
print('Integration error :  {:0.6f},   {:0.6f},      N/A  ,      N/A  \n'.format(mom1err,mom2err))