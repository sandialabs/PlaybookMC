#!usr/bin/env python
## \file RandCoefsBoundsBenchModule.py
#  \brief Module containing functions with analytic and semi-analytic benchmarks.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
import numpy as np
import scipy.integrate as scint


################################################ General ################################################

## \brief Translates material bounds to segment lengths
#
# \param[in] MatBounds list of floats, boundaries of 1D material geometry
# \returns SigLens list of floats, segment lengths
def MatBoundsToSegLens(MatBounds):
    SegLens = []
    for i in range(0,len(MatBounds)-1):
        SegLens.append( MatBounds[i+1]-MatBounds[i] )
    return SegLens



########################################## Random Coefficents ###########################################

## \brief Computes desired moment of transmittance for random coefficient problem
#
# \param[in] TotXSAves list of floats, average total cross section values
# \param[in] TotXSVars list of floats, magnitude of uniform variation in cross section values
# \param[in] SegLengs list of floats, length of each material segment
# \param[in] moment int, moment to calculate
# \returns transmittance moment, float
def returnMom_UnRandCoefs(TotXSAves,TotXSVars,SegLens,moment):
    assert isinstance(TotXSAves,list) and isinstance(TotXSVars,list) and isinstance(SegLens,list)
    numdims = len(TotXSAves)
    assert numdims==len(TotXSAves) and numdims==len(SegLens)
    for idim in range(0,numdims):
        assert isinstance(TotXSAves[idim],float) and isinstance(TotXSVars[idim],float) and isinstance(SegLens[idim],float)
    assert isinstance(moment,int) and moment>0

    product = 1.0
    for idim in range(0,numdims):
        product *= np.exp( - moment * TotXSAves[idim] * SegLens[idim] )
        if TotXSVars[idim] * SegLens[idim] == 0.0:  #lim x->0 sinh(0)/0 = 1.0
            product *= 1.0
        else:
            product *= np.sinh(  moment * TotXSVars[idim] * SegLens[idim] )
            product /=        (  moment * TotXSVars[idim] * SegLens[idim] )
    return product

## \brief Computes mean and standard deviation of transmittance for random coefficient problem
#
# \param[in] TotXSAves list of floats, average total cross section values
# \param[in] TotXSVars list of floats, magnitude of uniform variation in cross section values
# \param[in] SegLengs list of floats, length of each material segment
# \returns mean, dev, floats
def returnMeanDev_UnRandCoefs(TotXSAves,TotXSVars,SegLens):
    assert isinstance(TotXSAves,list) and isinstance(TotXSVars,list) and isinstance(SegLens,list)
    numdims = len(TotXSAves)
    assert numdims==len(TotXSAves) and numdims==len(SegLens)
    for idim in range(0,numdims):
        assert isinstance(TotXSAves[idim],float) and isinstance(TotXSVars[idim],float) and isinstance(SegLens[idim],float)

    mean = returnMom_UnRandCoefs(TotXSAves,TotXSVars,SegLens,moment=1)
    mom2 = returnMom_UnRandCoefs(TotXSAves,TotXSVars,SegLens,moment=2)
    dev  = np.sqrt( mom2 - mean**2 )
    return mean,dev




########################################### Random Boundaries ###########################################

## \brief Computes desired moment of transmittance for random boundaries problem
#
# \param[in] TotXSs list of floats, total cross section values
# \param[in] MatBoundAves list of floats, average material boundary locations
# \param[in] MatBoundVars list of floats, magnitude of uniform variation in material boundary locations--first and last must be zero
# \param[in] moment int, moment to calculate
# \returns transmittance moment, float
def returnMom_UnRandBounds(TotXSs,MatBoundAves,MatBoundVars,moment):
    assert isinstance(TotXSs,list) and isinstance(MatBoundAves,list) and isinstance(MatBoundVars,list)
    numdims = len(TotXSs)-1
    assert numdims==len(MatBoundAves)-2 and numdims==len(MatBoundVars)-2
    for idim in range(0,numdims+1):
        assert isinstance(TotXSs[idim],float)
    for idim in range(0,numdims+2):
        assert isinstance(MatBoundAves[idim],float) and isinstance(MatBoundVars[idim],float)
    assert MatBoundVars[0]==0.0 and MatBoundVars[-1]==0.0
    assert isinstance(moment,int) and moment>0

    product = 1.0
    for idim in range(0,numdims+1):
        product *= np.exp( - moment * TotXSs[idim] * (MatBoundAves[idim+1] - MatBoundAves[idim]) )
    for idim in range(1,numdims+1):
        if (TotXSs[idim] - TotXSs[idim-1]) * MatBoundVars[idim] == 0.0:  #lim x->0 sinh(0)/0 = 1.0
            product *= 1.0
        else:
            product *= np.sinh(  moment * (TotXSs[idim] - TotXSs[idim-1]) * MatBoundVars[idim] )
            product /=        (  moment * (TotXSs[idim] - TotXSs[idim-1]) * MatBoundVars[idim] )
    return product

## \brief Computes mean and standard deviation of transmittance for random boundaries problem
#
# \param[in] TotXSs list of floats, total cross section values
# \param[in] MatBoundAves list of floats, average material boundary locations
# \param[in] MatBoundVars list of floats, magnitude of uniform variation in material boundary locations--first and last must be zero
# \returns mean, dev, floats
def returnMeanDev_UnRandBounds(TotXSs,MatBoundAves,MatBoundVars):
    assert isinstance(TotXSs,list) and isinstance(MatBoundAves,list) and isinstance(MatBoundVars,list)
    numdims = len(TotXSs)-1
    assert numdims==len(MatBoundAves)-2 and numdims==len(MatBoundVars)-2
    for idim in range(0,numdims+1):
        assert isinstance(TotXSs[idim],float)
    for idim in range(0,numdims+2):
        assert isinstance(MatBoundAves[idim],float) and isinstance(MatBoundVars[idim],float)
    assert MatBoundVars[0]==0.0 and MatBoundVars[-1]==0.0

    mean = returnMom_UnRandBounds(TotXSs,MatBoundAves,MatBoundVars,moment=1)
    mom2 = returnMom_UnRandBounds(TotXSs,MatBoundAves,MatBoundVars,moment=2)
    dev  = np.sqrt( mom2 - mean**2 )
    return mean,dev




################################### Random Boundaries and Coefficents ###################################

## \brief Arbitrary-D integrand for random boundaries and coefficients problem
#
# \param[in] *args tuple, first in tuple are values of random variables, last four are TotXSAves, TotXSVars, MatBoundAves, and moment
# \returns mean, dev, floats
def IntegrandND(*args):
    TotXSAves    = args[-5]
    TotXSVars    = args[-4]
    MatBoundAves = args[-3]
    MatBoundVars = args[-2]
    moment       = args[-1]

    numdims = len(args)-5
    normalization = 1.0
    for idim in range(0,numdims):
        MatBoundAves[idim+1] = args[idim]
        normalization *= 2.0 * MatBoundVars[idim+1]

    SegLens = MatBoundsToSegLens(MatBoundAves)
    return returnMom_UnRandCoefs(TotXSAves,TotXSVars,SegLens,moment)/normalization

## \brief Computes desired moment of transmittance for problem with random coefficients and random boundaries
#
# This function performs numerical integration for the random boundaries component of
# the integration over the analytic integration of the random coefficients part of the problem.
#
# \param[in] TotXSAves list of floats, average total cross section values
# \param[in] TotXSVars list of floats, magnitude of uniform variation in cross section values
# \param[in] MatBoundAves list of floats, average material boundary locations
# \param[in] MatBoundVars list of floats, magnitude of uniform variation in material boundary locations--first and last must be zero
# \param[in] moment int, moment to calculate
# \returns transmittance moment and numerical integration error estimate on that value, floats
def returnMom_UnRandCoefsBounds(TotXSAves,TotXSVars,MatBoundAves,MatBoundVars,moment):
    assert isinstance(TotXSAves,list) and isinstance(TotXSVars,list)
    numcoefdims = len(TotXSAves)
    assert numcoefdims==len(TotXSAves)
    for idim in range(0,numcoefdims):
        assert isinstance(TotXSAves[idim],float) and isinstance(TotXSVars[idim],float)

    assert isinstance(MatBoundAves,list) and isinstance(MatBoundVars,list)
    numbounddims = len(MatBoundAves)-2
    assert numbounddims==len(MatBoundVars)-2
    for idim in range(0,numbounddims+2):
        assert isinstance(MatBoundAves[idim],float) and isinstance(MatBoundVars[idim],float)
    assert MatBoundVars[0]==0.0 and MatBoundVars[-1]==0.0

    assert isinstance(moment,int) and moment>0

    ranges = []
    for idim in range(1,numbounddims+1):
        ranges.append([MatBoundAves[idim]-MatBoundVars[idim],MatBoundAves[idim]+MatBoundVars[idim]])

    return scint.nquad(IntegrandND,ranges,args=(TotXSAves,TotXSVars,MatBoundAves[:],MatBoundVars,moment))

## \brief Computes mean and standard deviation of transmittance for random boundaries problem
#
# This function performs numerical integration for the random boundaries component of
# the integration over the analytic integration of the random coefficients part of the problem.
#
# \param[in] TotXSAves list of floats, average total cross section values
# \param[in] TotXSVars list of floats, magnitude of uniform variation in cross section values
# \param[in] MatBoundAves list of floats, average material boundary locations
# \param[in] MatBoundVars list of floats, magnitude of uniform variation in material boundary locations--first and last must be zero
# \returns mean, dev, numerical integration error of mean and mom2, all floats
def returnMeanDev_UnRandCoefsBounds(TotXSAves,TotXSVars,MatBoundAves,MatBoundVars):
    assert isinstance(TotXSAves,list) and isinstance(TotXSVars,list)
    numcoefdims = len(TotXSAves)
    assert numcoefdims==len(TotXSAves)
    for idim in range(0,numcoefdims):
        assert isinstance(TotXSAves[idim],float) and isinstance(TotXSVars[idim],float)

    assert isinstance(MatBoundAves,list) and isinstance(MatBoundVars,list)
    numbounddims = len(MatBoundAves)-2
    assert numbounddims==len(MatBoundVars)-2
    for idim in range(0,numbounddims+2):
        assert isinstance(MatBoundAves[idim],float) and isinstance(MatBoundVars[idim],float)
    assert MatBoundVars[0]==0.0 and MatBoundVars[-1]==0.0

    mean,meanerr = returnMom_UnRandCoefsBounds(TotXSAves,TotXSVars,MatBoundAves,MatBoundVars,moment=1)
    mom2,mom2err = returnMom_UnRandCoefsBounds(TotXSAves,TotXSVars,MatBoundAves,MatBoundVars,moment=2)
    dev          = np.sqrt( mom2 - mean**2 )
    return mean,dev,meanerr,mom2err