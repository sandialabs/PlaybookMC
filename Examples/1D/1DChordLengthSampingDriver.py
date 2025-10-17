#!usr/bin/env python
## \file 1DChordLengthSamplingDriver.py
#  \brief Example driver for running 1D Chord Length Sampling (and accuracy-enhanced derivaties).
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#  \author Dominic Lioce, dalioce@sandia.gov, liocedominic@gmail.com
import sys
sys.path.append('../../Core/Tools')
from MarkovianInputspy import MarkovianInputs
sys.path.append('../../Core/1D')
from SpecialMonteCarloDriverspy import SpecialMonteCarloDrivers

#Prepare inputs
numparticles  = 10000
numpartupdat  = 1000
numpartsample = 2   #batch size
numpartitions = 25    #number of partitions of samples - used to provide r.v. statistical precision
CLSvariant    = 'CLS' #'CLS', 'LRP', or 'AlgC' - Chord Length Sampling (CLS), the Local Realization Preserving (LRP), or "Algorithm C" (AlgC).  The latter are successively memory-improved versions of the first; they are referred to as Algorithms A, B, and C in ZimmermanANS1991 (and by the names here in VuJQSRT2021)
case          = '3a'  #'1a','1b','1c','2a','2b','2c','3a','3b','3c'; Problem from Adams, Larsen, and Pomraning (ALP) benchmark set
numtalbins    = 8
fluxTallyType = 'TrackLength' #'TrackLength' or 'Collision'

#Load problem parameters
CaseInp = MarkovianInputs()
CaseInp.selectALPInputs( case )
print(''); print(case)

#Setup transport
SMCsolver = SpecialMonteCarloDrivers(numpartsample)
SMCsolver.defineSourcePosition('left-boundary')
SMCsolver.defineSourceAngle('boundary-isotropic')
SMCsolver.initializeRandomNumberObject(flUseSeed=True, seed=1)
SMCsolver.defineMarkovianGeometry(totxs=CaseInp.Sigt[:], lam=CaseInp.lam[:], slablength=10.0, scatxs=CaseInp.Sigs[:], NaryType='volume-fraction')
if   CLSvariant == 'CLS' : SMCsolver.chooseCLSsolver(fl3DEmulation=False)
elif CLSvariant == 'LRP' : SMCsolver.chooseLRPsolver(fl3DEmulation=False)
elif CLSvariant == 'AlgC': SMCsolver.chooseAlgCsolver()
else                     : raise Exception("Please choose 'CLS', 'LRP', or 'AlgC' for the variable CLSvariant")
SMCsolver.selectFluxTallyOptions(numFluxBins=numtalbins,fluxTallyType=fluxTallyType)

#Perform transport
SMCsolver.pushParticles(numparticles,numpartupdat)

#Compute leakage tallies, print to screen if flVerbose=True
SMCsolver.processSimulationFluxTallies()
tmean,tdev,tmeanSEM,tdevSEM = SMCsolver.returnTransmittanceMoments(flVerbose=True,NumStatPartitions=numpartitions)
rmean,rdev,rmeanSEM,rdevSEM = SMCsolver.returnReflectanceMoments(flVerbose=True,NumStatPartitions=numpartitions)
amean,adev,ameanSEM,adevSEM = SMCsolver.returnAbsorptionMoments(flVerbose=True,NumStatPartitions=numpartitions)
fmean,fdev,fmeanSEM,fdevSEM = SMCsolver.returnWholeDomainFluxMoments(flVerbose=True,NumStatPartitions=numpartitions)
print()
SMCsolver.returnRuntimeValues(flVerbose=True)

SMCsolver.plotFlux(flMaterialDependent=True,flshow=True,flsave=False,fileprefix='CLSProb')