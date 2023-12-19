#!usr/bin/env python
## \file ConditionalPointSampling-RecentMemory.py
#  \brief Simple version of CoPS with recent memory of sampled points intented to help communicate how it works.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
## -------------------------------------------------------------------------- ##
# In this script we solve leakage values for an isotropic boundary source on a
# 1D slab of binary Markovian mixed material using CoPS with recent memory. Recent
# memory means that we remember a certain number of recently sampled points and
# potentially use them in conditional probability function evaluations. This script
# specifically demonstrates remembering the one most recently sampled point.  In
# OlsonMC2023, it was argued and demonstrated that this usage of CoPS, in 1D stochastic
# media with Markov mixing statistics, yields equivalent transport quantities as 
# Chord Length Sampling. Inputs and benchmarks for nine published benchmark problems
# are provided, but the user can set other input.
## -------------------------------------------------------------------------- ##
import numpy as np
import time

## Define user options
## -------------------------------------------------------------------------- ##
# Set number of particles to simulate and how often you want a runtime update
# Set problem description parameters, the Adams, Larsen, and Pomraning problems are provided as input options, choose one by uncommenting or create your own input
numparts = 10000
numprint = 1000

#total cross sections;            scattering ratios;   average chord lengths, slab thickness; benchmark values: average reflection/transimission + uncertainty (VuJQSRT2021, L=10)
#Sigt = [10.0/99.0, 100.0/11.0 ]; Scatrat = [0.0,1.0]; lam = [0.99,0.11];     L = 10.0;       R_CLS = 0.3778; R_CLSUnc = 0.0005; T_CLS = 0.0262; T_CLSUnc = 0.0002 #1a
#Sigt = [10.0/99.0, 100.0/11.0 ]; Scatrat = [1.0,0.0]; lam = [0.99,0.11];     L = 10.0;       R_CLS = 0.0585; R_CLSUnc = 0.0002; T_CLS = 0.00156 ; T_CLSUnc = 0.00004 #1b
Sigt = [10.0/99.0, 100.0/11.0 ]; Scatrat = [0.9,0.9]; lam = [0.99,0.11];     L = 10.0;       R_CLS = 0.3691; R_CLSUnc = 0.0005; T_CLS = 0.0236; T_CLSUnc = 0.0002 #1c
#Sigt = [10.0/99.0, 100.0/11.0 ]; Scatrat = [0.0,1.0]; lam = [9.9 ,1.1 ];     L = 10.0;       R_CLS = 0.1802; R_CLSUnc = 0.0004; T_CLS = 0.1284; T_CLSUnc = 0.0003 #2a
#Sigt = [10.0/99.0, 100.0/11.0 ]; Scatrat = [1.0,0.0]; lam = [9.9 ,1.1 ];     L = 10.0;       R_CLS = 0.2187; R_CLSUnc = 0.0004; T_CLS = 0.1789; T_CLSUnc = 0.0004 #2b
#Sigt = [10.0/99.0, 100.0/11.0 ]; Scatrat = [0.9,0.9]; lam = [9.9 ,1.1 ];     L = 10.0;       R_CLS = 0.2894; R_CLSUnc = 0.0005; T_CLS = 0.1946; T_CLSUnc = 0.0004 #2c
#Sigt = [2.0/101.0, 200.0/101.0]; Scatrat = [0.0,1.0]; lam = [5.05,5.05];     L = 10.0;       R_CLS = 0.6075; R_CLSUnc = 0.0005; T_CLS = 0.2409; T_CLSUnc = 0.0004 #3c
#Sigt = [2.0/101.0, 200.0/101.0]; Scatrat = [1.0,0.0]; lam = [5.05,5.05];     L = 10.0;       R_CLS = 0.0240; R_CLSUnc = 0.0002; T_CLS = 0.0756; T_CLSUnc = 0.0003 #3b
#Sigt = [2.0/101.0, 200.0/101.0]; Scatrat = [0.9,0.9]; lam = [5.05,5.05];     L = 10.0;       R_CLS = 0.3266; R_CLSUnc = 0.0005; T_CLS = 0.1194; T_CLSUnc = 0.0004 #3c

## Process inputs
## -------------------------------------------------------------------------- ##
XSCeiling = max(Sigt); percents = [ lam[0]/sum(lam), lam[1]/sum(lam) ]; lamc = lam[0]*lam[1] / sum(lam)


## -------------------------------------------------------------------------- ##
# uses nearest point to compute conditional probability for material 0
def TwoPtCondProbFunc(r,mat,percents,lamc):
    if mat==0: return percents[0] + percents[1]*np.exp(-r/lamc) # pi(m=0|0,r)
    else     : return percents[0] - percents[0]*np.exp(-r/lamc) # pi(m=0|1,r)

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
    x = 0.0; mu= np.sqrt( Rng.rand() ); matpt = None; mattype = None
    while True:

        # stream to potential collision site
        distPotentialCol = -np.log( Rng.rand() ) / XSCeiling * mu
        x = x + distPotentialCol

        # if particle leaks, tally and terminate particle
        if x<=0.0: R+=1; break
        if x>=L  : T+=1; break

        # get probability of material 0, sample new point material type, store new point material type and location
        if   matpt==None: localProbOf0 = percents[0]
        else            : localProbOf0 = TwoPtCondProbFunc(abs(x-matpt),mattype,percents,lamc)
        mattype = 0 if Rng.rand()<localProbOf0 else 1
        matpt   =                x 

        # sample if pseudo-collision is a true collision, if so, evaluate it
        if Rng.rand() > Sigt[mattype]/XSCeiling: continue # pseudo-collision rejected
        else:                                                  # pseudo-collision accepted as collision
            if Rng.rand() > Scatrat[mattype]: break       # particle absorbed
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
    print('\n--- CLS ---')
    print('Reflectance          :',R_CLS,' +- ',R_CLSUnc)
    print('Transmittance        :',T_CLS,' +- ',T_CLSUnc)
except: pass

print('\n--- CoPS with recent memory of one point ---')
print('Reflectance          :',averageRefl,' +- ',uncertaintyRefl)
print('Transmittance        :',averageTran,' +- ',uncertaintyTran)


print('\nTotal runtime: ',(time.time()-tstart)/60.0,' min')