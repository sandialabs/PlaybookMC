#!usr/bin/env python
## \file ConditionalPointSampling-FullMemory.py
#  \brief Simple version of CoPS with full memory of sampled points intented to help communicate how it works.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
## -------------------------------------------------------------------------- ##
# In this script we solve leakage values for an isotropic boundary source on a
# 1D slab of binary Markovian mixed material using CoPS with full memory.  Full memory
# means that if the material at a location is sampled, it is remembered and potentially
# used in future conditional probability function evaluations.  CoPS2 uses the nearest
# point to define the probability of getting each possible material type at a new
# point.  CoPS3 uses the nearest point on each side of the new point (when there are
# only points on one side of the new point, defaulting to CoPS2). CoPS3, as presented
# here, is sometimes called CoPS3PO in the literature (i.e.,VuJQSRT2021). The "PO" is
# used to denote the use of the CPF in "ThreePtCondProbFunc" and yields stochastic
# media transport without (bias) error in 1D stochastic media with Markovian mixing.
# Inputs and benchmarks for nine published benchmark problems are provided, but the
# user can set other input.
## -------------------------------------------------------------------------- ##
import numpy as np
import time

## Define user options
## -------------------------------------------------------------------------- ##
# Set number of particles to simulate and how often you want a runtime update
# Choose CoPS2 or CoPS3
# Set problem description parameters, the Adams, Larsen, and Pomraning problems are provided as input options, choose one by uncommenting or create your own input
numparts = 1000
numprint = 100
CoPSversion = 'CoPS3' #'CoPS2' or 'CoPS3'

#total cross sections;            scattering ratios;   average chord lengths, slab thickness; benchmark values: average reflection/transimission + uncertainty (VuJQSRT2021, L=10)
#Sigt = [10.0/99.0, 100.0/11.0 ]; Scatrat = [0.0,1.0]; lam = [0.99,0.11];     L = 10.0;       Rbench = 0.4360; RbenchUnc = 0.0005; Tbench = 0.0149; TbenchUnc = 0.0001 #1a
#Sigt = [10.0/99.0, 100.0/11.0 ]; Scatrat = [1.0,0.0]; lam = [0.99,0.11];     L = 10.0;       Rbench = 0.0850; RbenchUnc = 0.0003; Tbench = 0.00167 ; TbenchUnc = 0.00005 #1b
Sigt = [10.0/99.0, 100.0/11.0 ]; Scatrat = [0.9,0.9]; lam = [0.99,0.11];     L = 10.0;       Rbench = 0.4778; RbenchUnc = 0.0005; Tbench = 0.0163; TbenchUnc = 0.0001 #1c
#Sigt = [10.0/99.0, 100.0/11.0 ]; Scatrat = [0.0,1.0]; lam = [9.9 ,1.1 ];     L = 10.0;       Rbench = 0.2373; RbenchUnc = 0.0005; Tbench = 0.0980; TbenchUnc = 0.0003 #2a
#Sigt = [10.0/99.0, 100.0/11.0 ]; Scatrat = [1.0,0.0]; lam = [9.9 ,1.1 ];     L = 10.0;       Rbench = 0.2876; RbenchUnc = 0.0005; Tbench = 0.1953; TbenchUnc = 0.0004 #2b
#Sigt = [10.0/99.0, 100.0/11.0 ]; Scatrat = [0.9,0.9]; lam = [9.9 ,1.1 ];     L = 10.0;       Rbench = 0.4326; RbenchUnc = 0.0005; Tbench = 0.1870; TbenchUnc = 0.0004 #2c
#Sigt = [2.0/101.0, 200.0/101.0]; Scatrat = [0.0,1.0]; lam = [5.05,5.05];     L = 10.0;       Rbench = 0.6904; RbenchUnc = 0.0005; Tbench = 0.1639; TbenchUnc = 0.0004 #3c
#Sigt = [2.0/101.0, 200.0/101.0]; Scatrat = [1.0,0.0]; lam = [5.05,5.05];     L = 10.0;       Rbench = 0.0364; RbenchUnc = 0.0002; Tbench = 0.0762; TbenchUnc = 0.0003 #3b
#Sigt = [2.0/101.0, 200.0/101.0]; Scatrat = [0.9,0.9]; lam = [5.05,5.05];     L = 10.0;       Rbench = 0.4452; RbenchUnc = 0.0005; Tbench = 0.1042; TbenchUnc = 0.0003 #3c

## Process inputs
## -------------------------------------------------------------------------- ##
XSCeiling = max(Sigt); percents = [ lam[0]/sum(lam), lam[1]/sum(lam) ]; lamc = lam[0]*lam[1] / sum(lam)

## Functions evaluating CoPS points and probabilities
## -------------------------------------------------------------------------- ##
# uses nearest point on each side of new point to compute conditional probability for material 0 (exact for 1D, Markovian)
def ThreePtCondProbFunc(r1, r2, mat1, mat2, percents, lamc):
    if   mat1==0 and mat2==0: return 1.0 - ( percents[1]*( 1.0-np.exp(-r1/lamc) ) ) * (                     1.0-np.exp(-r2/lamc)   ) / ( 1.0 - percents[1]/(percents[1]-1.0) * np.exp(-(r1+r2)/lamc) ) # pi(m=0|{0,0},{r1,r2})
    elif mat1==1 and mat2==1: return       ( percents[0]*( 1.0-np.exp(-r1/lamc) ) ) * (                     1.0-np.exp(-r2/lamc)   ) / ( 1.0 - percents[0]/(percents[0]-1.0) * np.exp(-(r1+r2)/lamc) ) # pi(m=0|{1,1},{r1,r2})
    elif mat1==0 and mat2==1: return       (               1.0-np.exp(-r2/lamc)   ) * ( 1.0 - percents[1]*( 1.0-np.exp(-r1/lamc) ) ) / ( 1.0 -                                 np.exp(-(r1+r2)/lamc) ) # pi(m=0|{0,1},{r1,r2})
    elif mat1==1 and mat2==0: return       (               1.0-np.exp(-r1/lamc)   ) * ( 1.0 - percents[1]*( 1.0-np.exp(-r2/lamc) ) ) / ( 1.0 -                                 np.exp(-(r2+r1)/lamc) ) # pi(m=0|{1,0},{r1,r2})

## -------------------------------------------------------------------------- ##
# uses nearest point to compute conditional probability for material 0 (exact for 1D, Markovian if points only on one side)
def TwoPtCondProbFunc(r,mat,percents,lamc):
    if mat==0: return percents[0] + percents[1]*np.exp(-r/lamc) # pi(m=0|0,r)
    else     : return percents[0] - percents[0]*np.exp(-r/lamc) # pi(m=0|1,r)

## -------------------------------------------------------------------------- ##
# solves for nearest point and calls two point conditional probability function to return probability of sampling material 0
def returnCoPS2Probability(x, matpts, mattypes, percents, lamc):
    # solve for nearest point
    dists = list( map(abs, list( np.subtract(matpts,x) )) ) # solve distance to all points
    index = dists.index( min(dists) )                # solve index of nearest point
    r     = dists[index]                             # get distance to nearest point
    mat   = mattypes[index]                          # get type of nearest point
    # sample new material type
    return TwoPtCondProbFunc(r,mat,percents,lamc)

## -------------------------------------------------------------------------- ##
# collects nearest point on each side and calls three point conditional probability function to return probability of sampling material 0
def returnCoPS3Probability(indexL, indexR, dists, mattypes, percents, lamc):
    r1     = abs( 1.0/dists[indexL] );   mat1 = mattypes[indexL]  # the left point
    r2     = abs( 1.0/dists[indexR] );   mat2 = mattypes[indexR]  # the right point
    return ThreePtCondProbFunc(r1, r2, mat1, mat2, percents, lamc)

## -------------------------------------------------------------------------- ##
# selects CoPS2 evaluation if CoPS2 chosen or if there are only points on one side of the new point, otherwise selects CoPS3
def samplePtConditionally(x, matpts, mattypes, percents, lamc, CoPSversion):
    # if using CoPS2, evaluate using CoPS2
    if CoPSversion=='CoPS2': return returnCoPS2Probability( x, matpts, mattypes, percents, lamc )
    # test if there is at least one point on each side
    dists     = list( np.divide( 1.0, np.subtract(matpts,x) ) )
    indexL    = dists.index( min(dists) )
    indexR    = dists.index( max(dists) )
    # if all points only on one side, evaluate using CoPS2, else evaluate using CoPS3
    if dists[indexL]*dists[indexR] > 0.0: return returnCoPS2Probability( x, matpts, mattypes, percents, lamc )
    else                                : return returnCoPS3Probability( indexL, indexR, dists, mattypes, percents, lamc )
## -------------------------------------------------------------------------- ##

#####
Seed = 43925
Stride = 267
Rng = np.random.RandomState()
#####

## -------------------------------------------------------------------------- ##
## Begin Running problem
# initialize runtime clock and tallies
tstart = time.time()
R = 0;T = 0

# cycle through all particles
for ipart in range(0,numparts):
    Rng.seed(Seed + Stride * ipart)

    # initialize semi-isotropic boundary source; cycle through operations until history terminated
    x = 0.0; mu= np.sqrt( Rng.rand() ); matpts = []; mattypes = []
    while True:

        # stream to potential collision site
        distPotentialCol = -np.log( Rng.rand() ) / XSCeiling * mu
        x = x + distPotentialCol

        # if particle leaks, tally and terminate particle
        if x<=0.0: R+=1; break
        if x>=L  : T+=1; break

        # get probability of material 0, sample new point material type, store new point material type and location
        if len(matpts)==0: localProbOf0 = percents[0]
        else             : localProbOf0 = samplePtConditionally( x, matpts, mattypes, percents, lamc, CoPSversion )
        mattypes.append( 0 if Rng.rand()<localProbOf0 else 1 )
        matpts.append( x )

        # sample if pseudo-collision is a true collision, if so, evaluate it
        if Rng.rand() > Sigt[mattypes[-1]]/XSCeiling: continue # pseudo-collision rejected
        else:                                                  # pseudo-collision accepted as collision
            if Rng.rand() > Scatrat[mattypes[-1]]: break       # particle absorbed
            else: mu = 2.0*Rng.rand()-1.0                      # particle isotropically scattered

    # runtime updates
    if ipart>0 and ipart%numprint==0:
        ttot = (time.time()-tstart)/60.0
        esttot  = ttot/ipart*numparts
        print(ipart,'/',numparts,' parts, ',ttot,'/',esttot,' min')

# compute stats on reflectance and transmittance
# note - since tallies are 0s and 1s, second moment is same as first moment making uncertainty computation simpler than usual
averageRefl     = float(R)/float(numparts)
uncertaintyRefl = np.sqrt( ( averageRefl - averageRefl**2 ) / float(numparts - 1) )
averageTran     = float(T)/float(numparts)
uncertaintyTran = np.sqrt( ( averageTran - averageTran**2 ) / float(numparts - 1) )

try:
    print('\n--- Benchmark ---')
    print('Reflectance          :',Rbench,' +- ',RbenchUnc)
    print('Transmittance        :',Tbench,' +- ',TbenchUnc)
except: pass

print('\n--- ',CoPSversion,' ---')
print('Reflectance          :',averageRefl,' +- ',uncertaintyRefl)
print('Transmittance        :',averageTran,' +- ',uncertaintyTran)


print('\nTotal runtime: ',(time.time()-tstart)/60.0,' min')