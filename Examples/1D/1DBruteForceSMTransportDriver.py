#!usr/bin/env python
## \file 1DBruteForceSMTransportDriver.py
#  \brief Example driver script for 1D transport with stochastic media slab realizations.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools')
from MarkovianInputspy import MarkovianInputs
sys.path.append('../../Core/1D')
from SpecialMonteCarloDriverspy import SpecialMonteCarloDrivers
from OneDSlabpy import Slab
import numpy as np

#Prepare inputs
numparticles  = 10000
numpartupdat  = 1000
numpartsample = 2   #number of particle histories per realization
numpartitions = 25    #number of partitions of samples - used to provide r.v. statistical precision
Slablength    = 10.0
case          = '3a'  #'1a','1b','1c','2a','2b','2c','3a','3b','3c'; Problem from Adams, Larsen, and Pomraning (ALP) benchmark set
realGenMethod = 'sampleChords' #'sampleChords' or 'samplePseudoInterfaces' - which method to use to generate 1D realizations of Markovain stochastic media
numtalbins    = 8
fluxTallyType = 'TrackLength' #'TrackLength' or 'Collision'
abundanceModel= 'sample'  #'ensemble' or 'sample'

#Load problem parameters
CaseInp = MarkovianInputs()
CaseInp.selectALPInputs( case )
print(''); print(case)

#Set up geometry; instantiate slab by defining materials, material cell boundaries, and the material type in each cell
slab = Slab()
slab.initializeRandomNumbers(flUseSeed=True,seed=876543)
slab.s       = Slablength
slab.nummats = 2
MarkVals = MarkovianInputs()
MarkVals.solveNaryMarkovianParamsBasedOnChordLengths(lam=CaseInp.lam[:])

#Instantiate Monte Carlo solver object, connect with slab geometry object, choose source, and initialize random number seed
SMCsolver = SpecialMonteCarloDrivers(numpartsample)
SMCsolver.defineSource('boundary-isotropic')
SMCsolver.initializeRandomNumberObject(flUseSeed=True,seed=1357)
SMCsolver.defineMarkovianGeometry(totxs=CaseInp.Sigt[:], lam=CaseInp.lam[:], slablength=Slablength, scatxs=CaseInp.Sigs[:], NaryType='volume-fraction')
SMCsolver.defineMCGeometry(slab)
SMCsolver.chooseBruteForcesolver(realGenMethod=realGenMethod,abundanceModel=abundanceModel)
SMCsolver.selectFluxTallyOptions(numFluxBins=numtalbins,fluxTallyType=fluxTallyType)

#Perform transport
SMCsolver.pushParticles(NumNewParticles=numparticles,NumParticlesUpdateAt=numpartupdat)

#Return transport results, also print to screen if "flVerbose==True"
SMCsolver.processSimulationFluxTallies()
tmean,tdev,tmeanSEM,tdevSEM = SMCsolver.returnTransmittanceMoments(flVerbose=True,NumStatPartitions=numpartitions)
rmean,rdev,rmeanSEM,rdevSEM = SMCsolver.returnReflectanceMoments(flVerbose=True,NumStatPartitions=numpartitions)
amean,adev,ameanSEM,adevSEM = SMCsolver.returnAbsorptionMoments(flVerbose=True,NumStatPartitions=numpartitions)
fmean,fdev,fmeanSEM,fdevSEM = SMCsolver.returnWholeDomainFluxMoments(flVerbose=True,NumStatPartitions=numpartitions)
print()
SMCsolver.returnTransmittanceRuntimeAnalysis(flVerbose=True,NumStatPartitions=numpartitions)
SMCsolver.returnReflectanceRuntimeAnalysis(flVerbose=True,NumStatPartitions=numpartitions)
SMCsolver.returnAbsorptionRuntimeAnalysis(flVerbose=True,NumStatPartitions=numpartitions)
print()
SMCsolver.returnRuntimeValues(flVerbose=True)

SMCsolver.plotFlux(flMaterialDependent=True)