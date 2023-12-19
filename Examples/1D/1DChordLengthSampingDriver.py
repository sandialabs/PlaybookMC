#!usr/bin/env python
## \file 1DChordLengthSamplingDriver.py
#  \brief Example driver for running 1D Chord Length Sampling (or accuracy-enhanced derivaties).
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#  \author Dominic Lioce, dalioce@sandia.gov, liocedominic@gmail.com
import sys
sys.path.append('../../Core/Tools')
from FluxTalliespy import FluxTallies
from MarkovianInputspy import MarkovianInputs
sys.path.append('../../Core/1D')
from SpecialMonteCarloDriverspy import SpecialMonteCarloDrivers
import numpy as np

#Prepare inputs
numparticles  = 1000
numpartupdat  = 100
CLSvariant    = 'AlgC' #'CLS', 'LRP', or 'AlgC' - Chord Length Sampling (CLS), the Local Realization Preserving (LRP), or "Algorithm C" (AlgC).  The latter are successively memory-improved versions of the first; they are referred to as Algorithms A, B, and C in ZimmermanANS1991 (and by the names here in VuJQSRT2021)
case          = '3a'  #'1a','1b','1c','2a','2b','2c','3a','3b','3c'; Problem from Adams, Larsen, and Pomraning (ALP) benchmark set
flfluxtallies = False

#Load problem parameters
CaseInp = MarkovianInputs()
CaseInp.selectALPInputs( case )
print(''); print(case)

#Setup transport
SMCsolver = SpecialMonteCarloDrivers()
SMCsolver.defineSource(sourceType='boundary-isotropic',sourceLocationRange=[0.0,0.0])
SMCsolver.initializeRandomNumberObject(flUseSeed=False, seed=None)
SMCsolver.defineMarkovianGeometry(totxs=CaseInp.Sigt[:], lam=CaseInp.lam[:], slablength=10.0, scatxs=CaseInp.Sigs[:], NaryType='volume-fraction')
if   CLSvariant == 'CLS' : SMCsolver.chooseCLSsolver(fl3DEmulation=False)
elif CLSvariant == 'LRP' : SMCsolver.chooseLRPsolver(fl3DEmulation=False)
elif CLSvariant == 'AlgC': SMCsolver.chooseAlgCsolver()
else                     : raise Exception("Please choose 'CLS', 'LRP', or 'AlgC' for the variable CLSvariant")
#If selected, instantiate and associate flux tally object
if flfluxtallies:
    FTal = FluxTallies()
    FTal.setFluxTallyOptions(numMomsToTally=2)
    FTal.defineSMCGeometry(slablength=10.0,xmin=0.0)
    SMCsolver.associateFluxTallyObject(FTal)
    FTal.setupFluxTallies(numTallyBins=100,flMaterialDependent=True,numMaterials=2)
    if FTal.flMaterialDependent:
        FTal.defineMaterialTypeMethod( FTal._return_iseg_AsMaterialType )
        matfractions = np.ones((SMCsolver.nummats,FTal.numTallyBins))
        for imat in range(0,SMCsolver.nummats):
            matfractions[imat,:] = np.multiply(matfractions[imat,:],SMCsolver.prob[imat])
        FTal.defineMaterialTypeFractions( matfractions )
        FTal.defineFluxTallyMethod( FTal._tallyMaterialBasedTrackLengthFlux )
    else:
        FTal.defineFluxTallyMethod( FTal._tallyTrackLengthFlux )

#Perform transport
SMCsolver.pushParticles(numparticles,numpartupdat)

#Compute leakage tallies, print to screen if flVerbose=True
tmean,tdev,tSEM,tFOM = SMCsolver.returnTransmittanceMoments(flVerbose=True)
rmean,rdev,rSEM,rFOM = SMCsolver.returnReflectanceMoments(flVerbose=True)
amean,adev,aSEM,aFOM = SMCsolver.returnAbsorptionMoments(flVerbose=True)

#If selected, print, read, and plot flux tallies (currently must print and read to plot, refactor to not require printing and reading to plot desired)
if flfluxtallies:
    FTal.printFluxVals(filename='1D'+CLSvariant+'Flux'+case,flFOMs=True)
    FTal.readFluxVals( filename='1D'+CLSvariant+'Flux'+case)
    FTal.plotFlux(flMaterialDependent=True)