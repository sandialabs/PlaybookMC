#!usr/bin/env python
## \file 1DConditionalPointSamplingDriver.py
#  \brief Example driver for running 1D Conditional Point Sampling (CoPS).
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools')
from MarkovianInputspy import MarkovianInputs
sys.path.append('../../Core/1D')
from SpecialMonteCarloDriverspy import SpecialMonteCarloDrivers

#Prepare inputs
numparticles  = 1000
numpartupdat  = 100
numpartsample = 20   #with only short-term memory, batch size; with long-term memory, cohort size
numpartitions = 25    #number of partitions of samples - used to provide r.v. statistical precision
CoPSvariant   = 'high-accuracy'  #'AM-accuracy', 'CLS-accuracy', 'moderate-accuracy', 'high-accuracy', or 'errorless-accuracy' - CoPS has several free parameters that enable it to operate differently. In this script, this variable enables user selection between five sets of options that are described in more detail when this variable is used.
case          = '3a'  #'1a','1b','1c','2a','2b','2c','3a','3b','3c'; Problem from Adams, Larsen, and Pomraning (ALP) benchmark set
numtalbins    = 8

#Load problem parameters
CaseInp = MarkovianInputs()
CaseInp.selectALPInputs( case )
print(''); print(case)

#Setup transport
CaseInp.selectALPInputs( case )
SMCsolver = SpecialMonteCarloDrivers(numpartsample)
SMCsolver.defineSourcePosition('left-boundary')
SMCsolver.defineSourceAngle('boundary-isotropic')
SMCsolver.initializeRandomNumberObject(flUseSeed=True, seed=1)
SMCsolver.defineMarkovianGeometry(totxs=CaseInp.Sigt[:], lam=CaseInp.lam[:], slablength=10.0, scatxs=CaseInp.Sigs[:], NaryType='volume-fraction')
if   CoPSvariant == 'AM-accuracy'       : numcondprobpts = 0; recentmemory = 0; amnesiaradius = None; longTermMemoryMode = 'off'           ; numpresampled=0    #With these options, new points are sampled based only on material abundance (and no previously sampled points).  In OlsonMC2023, it was argued and demonstrated that when running CoPS with these options yields equivalent transport solutions as the atomic mix (AM) approximation.
elif CoPSvariant == 'CLS-accuracy'      : numcondprobpts = 1; recentmemory = 1; amnesiaradius = None; longTermMemoryMode = 'off'           ; numpresampled=0    #With these options, only the most recently sampled point is remembered and used in conditional probability evluations for the next point. In OlsonMC2023, it was argued and demonstrated that when running CoPS with these options, for stochastic media with Markovian mixing statistics, yields equivalent transport solutions as Chord Length Sampling.
elif CoPSvariant == 'moderate-accuracy' : numcondprobpts = 1; recentmemory = 1; amnesiaradius = 0.01; longTermMemoryMode = 'on'            ; numpresampled=0    #With these options, a sampled point is remembered in long-term memory as long as it is at least 0.01 units from the nearest point which is already stored in long-term memory.  Additionally, the most recently sampled point that was not stored in long-term memory is stored in short-term memory. The nearest point from both sets of memory to a newly sampled point is used in conditional probability evaluations for the next point. This choice of recent memory and amnesia radius was explored in VuNSE2022.
elif CoPSvariant == 'high-accuracy'     : numcondprobpts = 1; recentmemory = 0; amnesiaradius = None; longTermMemoryMode = 'presampledonly'; numpresampled=1000 #With these options, long-term memory is sampled at the start of each cohort and remains constant for the duration of the cohort, there is no recent memory, and only the simple 2-pt conditional probaiblity function is used. If the number of points is increased, accuracy is increased.
elif CoPSvariant == 'errorless-accuracy': numcondprobpts = 2; recentmemory = 0; amnesiaradius = 0.0 ; longTermMemoryMode = 'on'            ; numpresampled=0    #With these options, all sampled points are remembered in long-term memory. The nearest point on each side of the new point (if there is one on each side) are used in conditional probability evaluations for the next point. The conditional probability function that is errorless for colinear points in stochastic media with Markovian mixing, described in VuJQSRT2021, is used with these options and yields errorless (i.e., no bias compared to realization-based benchmark calculations) results. Running CoPS in 1D with these options is sometimes called CoPS3PO (as in VuJQSRT2021).
else                                    : raise Exception("Please choose a valid option for the variable CoPSvariant")
SMCsolver.chooseCoPSsolver(numcondprobpts=numcondprobpts,fl3DEmulation=False)
SMCsolver.associateLimitedMemoryParam(recentMemory=recentmemory,amnesiaRadius=amnesiaradius,numpresampled=numpresampled,longTermMemoryMode=longTermMemoryMode)
SMCsolver.selectFluxTallyOptions(numFluxBins=numtalbins,fluxTallyType='Collision')

#Perform transport
SMCsolver.pushParticles(numparticles,numpartupdat)

#Compute leakage tallies, print to screen if flVerbose=True
SMCsolver.processSimulationFluxTallies()
tmean,tdev,tmeanSEM,tdevSEM = SMCsolver.returnTransmittanceMoments(flVerbose=True,NumStatPartitions=numpartitions)
rmean,rdev,rmeanSEM,rdevSEM = SMCsolver.returnReflectanceMoments(flVerbose=True,NumStatPartitions=numpartitions)
amean,adev,ameanSEM,adevSEM = SMCsolver.returnAbsorptionMoments(flVerbose=True,NumStatPartitions=numpartitions)
fmean,fdev,fmeanSEM,fdevSEM = SMCsolver.returnWholeDomainFluxMoments(flVerbose=True,NumStatPartitions=numpartitions)
if SMCsolver.longTermMemoryMode=='presampledonly':
    print()
    SMCsolver.returnTransmittanceRuntimeAnalysis(flVerbose=True,NumStatPartitions=numpartitions)
    SMCsolver.returnReflectanceRuntimeAnalysis(flVerbose=True,NumStatPartitions=numpartitions)
    SMCsolver.returnAbsorptionRuntimeAnalysis(flVerbose=True,NumStatPartitions=numpartitions)
print()
SMCsolver.returnRuntimeValues(flVerbose=True)

SMCsolver.plotFlux(flMaterialDependent=True,flshow=True,flsave=False,fileprefix='CoPSProb')