#!usr/bin/env python
## \file 1DConditionalPointSamplingDriver.py
#  \brief Example driver for running 1D Conditional Point Sampling
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
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
CoPSvariant   = 'CLS-accuracy'  #'AM-accuracy', 'CLS-accuracy', 'moderate-accuracy', or 'errorless-accuracy' - CoPS has several free parameters that enable it to operate differently. In this script, this variable enables user selection between four sets of options that are described in a more detail when this variable is used.
case          = '3a'  #'1a','1b','1c','2a','2b','2c','3a','3b','3c'; Problem from Adams, Larsen, and Pomraning (ALP) benchmark set
flfluxtallies = False

#Load problem parameters
CaseInp = MarkovianInputs()
CaseInp.selectALPInputs( case )
print(''); print(case)

#Setup transport
CaseInp.selectALPInputs( case )
SMCsolver = SpecialMonteCarloDrivers()
SMCsolver.defineSource(sourceType='boundary-isotropic',sourceLocationRange=[0.0,0.0])
SMCsolver.initializeRandomNumberObject(flUseSeed=False, seed=None)
SMCsolver.defineMarkovianGeometry(totxs=CaseInp.Sigt[:], lam=CaseInp.lam[:], slablength=10.0, scatxs=CaseInp.Sigs[:], NaryType='volume-fraction')
if   CoPSvariant == 'AM-accuracy'       : numcondprobpts = 0; recentmemory = 0; amnesiaradius = None; fllongtermmemory = False #With these options, new points are sampled based only on material abundance (and no previously sampled points).  In OlsonMC2023, it was argued and demonstrated that when running CoPS with these options yields equivalent transport solutions as the atomic mix (AM) approximation.
elif CoPSvariant == 'CLS-accuracy'      : numcondprobpts = 1; recentmemory = 1; amnesiaradius = None; fllongtermmemory = False #With these options, only the most recently sampled point is remembered and used in conditional probability evluations for the next point. In OlsonMC2023, it was argued and demonstrated that when running CoPS with these options, for stochastic media with Markovian mixing statistics, yields equivalent transport solutions as Chord Length Sampling.
elif CoPSvariant == 'moderate-accuracy' : numcondprobpts = 1; recentmemory = 1; amnesiaradius = 0.01; fllongtermmemory = True  #With these options, a sampled point is remembered in long-term memory as long as it is at least 0.01 units from the nearest point which is already stored in long-term memory.  Additionally, the most recently sampled point that was not stored in long-term memory is stored in short-term memory. The nearest point from both sets of memory to a newly sampled point is used in conditional probability evaluations for the next point. This choice of recent memory and amnesia radius was explored in VuNSE2022.
elif CoPSvariant == 'errorless-accuracy': numcondprobpts = 2; recentmemory = 0; amnesiaradius = 0.0 ; fllongtermmemory = True  #With these options, all sampled points are remembered in long-term memory. The nearest point on each side of the new point (if there is one on each side) are used in conditional probability evaluations for the next point. The conditional probability function that is errorless for colinear points in stochastic media with Markovian mixing, described in VuJQSRT2021, is used with these options and yields errorless (i.e., no bias compared to realization-based benchmark calculations) results. Running CoPS in 1D with these options is sometimes called CoPS3PO (as in VuJQSRT2021).
else                                    : raise Exception("Please choose 'CLS-accuracy', 'moderate-accuracy', or 'errorless-accuracy' for CoPSvariant")
SMCsolver.chooseCoPSsolver(numcondprobpts=numcondprobpts,fl3DEmulation=False)
SMCsolver.associateLimitedMemoryParam(recentMemory=recentmemory,amnesiaRadius=amnesiaradius,flLongTermMemory=fllongtermmemory)

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
        FTal.defineFluxTallyMethod( FTal._tallyMaterialBasedCollisionFlux )
    else:
        FTal.defineFluxTallyMethod( FTal._tallyCollisionFlux )

#Perform transport
SMCsolver.pushParticles(numparticles,numpartupdat,initializeTallies='first_hist',initializeGeomMem='each_hist')

#Compute leakage tallies, print to screen if flVerbose=True
tmean,tdev,tSEM,tFOM = SMCsolver.returnTransmittanceMoments(flVerbose=True)
rmean,rdev,rSEM,rFOM = SMCsolver.returnReflectanceMoments(flVerbose=True)
amean,adev,aSEM,aFOM = SMCsolver.returnAbsorptionMoments(flVerbose=True)

#If selected, print, read, and plot flux tallies (currently must print and read to plot, refactor to not require printing and reading to plot desired)
if flfluxtallies:
    FTal.printFluxVals(filename='1DCoPSFlux'+case,flFOMs=True)
    FTal.readFluxVals( filename='1DCoPSFlux'+case)
    FTal.plotFlux(flMaterialDependent=True)