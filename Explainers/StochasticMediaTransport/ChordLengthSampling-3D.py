#!usr/bin/env python
## \file ChordLengthSampling-3D.py
#  \brief Standalone script that performs Chord Length Sampling (CLS) in binary stochastic media in 3D.
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
    #define initial particle properties. Particles start on the minus z face travelling upward
    x = np.random.uniform(-L/2,L/2)
    y = np.random.uniform(-L/2,L/2)
    z = -L/2
    mu = np.sqrt( np.random.rand() ) #boundary isotropic source
    phi = 2.0 * np.pi * np.random.rand() #phi uniform on (0,2pi)
    u = np.sqrt(1.0 - mu**2) * np.cos(phi) #unit direction vector x component
    v = np.sqrt(1.0 - mu**2) * np.sin(phi) #unit direction vector y component
    w = mu                                 #unit direction vector z component
    #select material index from volume fractions, imat
    imat = np.random.choice(materials,p=probs)
    while True:
        #find each distance for the possible interactions
        #start by calculating distance to boundary in every direction
        if   u > 0: distanceToBoundary_x = ( L/2 - x)/u
        else      : distanceToBoundary_x = (-L/2 - x)/u
        if   v > 0: distanceToBoundary_y = ( L/2 - y)/v
        else      : distanceToBoundary_y = (-L/2 - y)/v
        if   w > 0: distanceToBoundary_z = ( L/2 - z)/w
        else      : distanceToBoundary_z = (-L/2 - z)/w
        distanceToBoundary  = min(distanceToBoundary_x,distanceToBoundary_y,distanceToBoundary_z)
        distanceToCollision = - np.log( np.random.rand() ) / totxs[imat]
        distanceToInterface = - np.log( np.random.rand() ) * lambdas[imat]
        #particle travels the minimum of these sampled distances
        dist = min(distanceToBoundary,distanceToCollision,distanceToInterface)
        x += dist * u #move particle to interaction site
        y += dist * v
        z += dist * w
        if   dist == distanceToBoundary:
            if z >=  L/2: T += 1; break #tally transmission and break
            if z <= -L/2: R += 1; break #tally reflection and break
            if x >=  L/2: u = -u #reflecting boundary
            if x <= -L/2: u = -u #reflecting boundary
            if y >=  L/2: v = -v #reflecting boundary
            if y <= -L/2: v = -v #reflecting boundary
        elif dist == distanceToCollision:
            if np.random.rand() <= scatterFraction[imat]: #scatter
                mu  = 2.0 * np.random.rand() - 1 #resample direction isotropically
                phi = 2.0 * np.random.rand() * np.pi
                u = np.sqrt(1.0 - mu**2) * np.cos(phi) #redefine direction vector
                v = np.sqrt(1.0 - mu**2) * np.sin(phi)
                w = mu
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
