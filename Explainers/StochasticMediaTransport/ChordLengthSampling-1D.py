#!usr/bin/env python
## \file ChordLengthSampling-1D.py
#  \brief Standalone script that performs Chord Length Sampling (CLS) in binary stochastic media in 1D.
#  Meant as a tool for education/basic run time comparisons.
#  \author Dominic Lioce, dalioce@sandia.gov, liocedominic@gmail.com
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import numpy as np
import time

## Initialize material variables
materials = [0,1]
lambdas = [99/100,11/100] #mean chord lengths of each material
probs = [lambdas[0]/sum(lambdas),lambdas[1]/sum(lambdas)] #volume fractions
totxs = [10/99,100/11] #total cross section of each material
scatterFraction = [0,1] #what fraction of total xs is scattering?

## Initialize transport/geometry variables
R = 0; T = 0; A = 0 #leakage tallies
L = 10
numParticles = 10000
tstart = time.time()

for i in range(0,numParticles):
    if i%(numParticles/10) == 0: print(i,"/",numParticles) #print progress update
    #define initial particle properties
    x = 0
    mu = np.sqrt( np.random.rand() ) #boundary isotropic source
    #select material index from volume fractions, imat
    imat = np.random.choice(materials,p=probs)
    while True:
        #find each distance (in the x direction) for the possible interactions
        distanceToBoundary  = L - x if mu >= 0 else x
        distanceToCollision = - np.log( np.random.rand() ) / totxs[imat] * abs(mu)
        distanceToInterface = - np.log( np.random.rand() ) * lambdas[imat]
        #particle travels the minimum of these sampled distances
        dist = min(distanceToBoundary,distanceToCollision,distanceToInterface)
        x = x + dist if mu >= 0 else x - dist #move particle to interaction site
        if   dist == distanceToBoundary:
            if   mu <  0: R += 1 #tally reflection
            elif mu >= 0: T += 1 #tally transmission
            break #stop simulating transport for this particle
        elif dist == distanceToCollision:
            if np.random.rand() <= scatterFraction[imat]: #scatter
                mu = 2*np.random.rand() - 1 #resample direction isotropically
            else: #absorption
                A += 1 #tally absorption, stop transport for this particle
                break
        elif dist == distanceToInterface: #switch materials if interface is sampled
            if   imat == 0: imat = 1
            elif imat == 1: imat = 0
t = time.time() - tstart
#calculate moments of leakage values
T /= numParticles
R /= numParticles
A /= numParticles
uncT = np.sqrt((T - T**2) / (numParticles-1)) #Standard error of the mean
uncR = np.sqrt((R - R**2) / (numParticles-1))
uncA = np.sqrt((A - A**2) / (numParticles-1))

print('Transmittance: ',T,'+/-',uncT)
print('Reflectance  : ',R,'+/-',uncR)
print('Absorption   : ',A,'+/-',uncA)
print('Total time   : ', t,'seconds')