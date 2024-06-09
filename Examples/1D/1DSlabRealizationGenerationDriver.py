#!usr/bin/env python
## \file 1DSlabRealizationGenerationDriver.py
#  \brief Example script generating and performing operations with stochastic media 1D realizations.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools')
from MarkovianInputspy import MarkovianInputs
sys.path.append('../../Core/1D')
from OneDSlabpy import Slab
import numpy as np

#Slab generation options
numreals      = 5000      #number of realizations to generate and analyze
ChordLengths  = [1.0,0.3]
slablength    = 10.0
generatorType = 'SampleChords'  #'SampleChords', 'Pseudointerfaces', 'Cells' - The first two are alternative methods for sampling 1D stochastic media with Markovian mixing, the third has different mixing behavior but can be yield attributes similar to Markovian (as done below).

#Particle cross sections and tranport emulator options
Sigt     = [0.1,0.5]; Sigs=[0.0,0.0]
numhists = 10   #number of particle histories to emulate per realization with uncollided transmittance emulator

#Set up slab object, tallies, and Markovian mixing values
slab = Slab()
slab.initializeRandomNumbers(flUseSeed=False,seed=876543)
aveMat0Frac = 0.0; aveMat1Frac = 0.0; aveOptThick = 0.0; aveAnTrans  = 0.0; aveNumTrans = 0.0
MarkVals = MarkovianInputs()
MarkVals.solveNaryMarkovianParamsBasedOnChordLengths(lam=ChordLengths)

#Sample realizations and tally observed attributes
for ireal in range(0,numreals):
    #Generate realization
    if   generatorType=='SampleChords'    : slab.populateMarkovRealization(totxs=Sigt,lam=ChordLengths,s=slablength,scatxs=Sigs,NaryType='volume-fraction')
    elif generatorType=='Pseudointerfaces': slab.populateMarkovRealizationPseudo(totxs=Sigt,lam=ChordLengths,s=slablength,scatxs=Sigs)
    elif generatorType=='Cells'           : slab.populateFixedCellLengthRealization(totxs=Sigt,prob=MarkVals.prob[:],numcells=round(slablength/MarkVals.lamc),s=slablength,scatxs=Sigs)
    else: raise Exception("Please select 'SampleChords', 'Pseudointerfaces', 'Cells' as the generator type")

    #Solve material fractions in evenly sized bins
    slab.solveMaterialTypeFractions(numbins=1)
    aveMat0Frac += slab.MatFractions[0][0] / numreals
    aveMat1Frac += slab.MatFractions[1][0] / numreals

    #Solve optical thickness
    optthick = slab.solveOptThick()
    aveOptThick += optthick / numreals

    #Solve uncollided transmittance analytically using optical thickness or MC particle emulator 
    aveAnTrans  += np.exp( - optthick ) / numreals
    aveNumTrans += slab.emulateUncolMCTrans(numhists=100,mu=1.0) / numreals


#Process and print tallies
print()
print('Fraction of Material 0: observed ; theoretical            :',aveMat0Frac,';',MarkVals.prob[0])
print('Fraction of Material 1: observed ; theoretical            :',aveMat1Frac,';',MarkVals.prob[1])
print('Average optical thickness                                 :',aveOptThick)
print('Average uncollided transmittance (analytic      transport):',aveAnTrans)
print('Average uncollided transmittance (num. emulator transport):',aveNumTrans)