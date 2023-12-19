#!usr/bin/env python
## \file 1DBruteForceSMTransportDriver.py
#  \brief Example driver script for 1D transport with stochastic media slab realizations.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools')
from FluxTalliespy import FluxTallies
from MarkovianInputspy import MarkovianInputs
sys.path.append('../../Core/1D')
from MonteCarloParticleSolverpy import MonteCarloParticleSolver
from OneDSlabpy import Slab
import numpy as np

#Prepare inputs
numparticles  = 10000
numpartupdat  = 1000
Slablength    = 10.0
case          = '3a'  #'1a','1b','1c','2a','2b','2c','3a','3b','3c'; Problem from Adams, Larsen, and Pomraning (ALP) benchmark set
flfluxtallies = True

#Load problem parameters
CaseInp = MarkovianInputs()
CaseInp.selectALPInputs( case )
print(''); print(case)


#Set up geometry; instantiate slab by defining materials, material cell boundaries, and the material type in each cell
slab = Slab()
slab.initializeRandomNumbers(flUseSeed=True,seed=876543)
aveMat0Frac = 0.0; aveMat1Frac = 0.0; aveOptThick = 0.0; aveAnTrans  = 0.0; aveNumTrans = 0.0
MarkVals = MarkovianInputs()
MarkVals.solveNaryMarkovianParamsBasedOnChordLengths(lam=CaseInp.lam[:])

#Instantiate Monte Carlo solver object, connect with slab geometry object, choose source, and initialize random number seed
MCsolver = MonteCarloParticleSolver()
MCsolver.defineSource('boundary-isotropic')
MCsolver.initializeRandomNumberObject(flUseSeed=True,seed=1357)
MCsolver.defineMCGeometry(slab,Slablength)

#Sample a new realization, push one particle, and repeat (Notes: 1) The current flux tally capability does not work well with this approach, 2) if utilized differently, you can simulate multiple histories per realization, but that approach requires more data management than is here to get MC tallies correct, and 3) time estimates here are not accurate since measured time is only time simulating particles, i.e., time spent generating realizations is not accounted for when running in this way)
for ipart in range(0,numparticles):
    initializetallies = 'first_hist' if ipart==0 else 'no_hist'
    slab.populateMarkovRealization(totxs=CaseInp.Sigt[:],lam=CaseInp.lam[:],s=Slablength,scatxs=list(np.multiply(CaseInp.Scatrat[:],CaseInp.Sigt[:])),NaryType='volume-fraction')
    MCsolver.pushParticles(NumNewParticles=1,NumParticlesUpdateAt=numpartupdat,initializeTallies=initializetallies)

#Return transport results, also print to screen if "flVerbose==True"
tmean,tdev,tSEM,tFOM = MCsolver.returnTransmittanceMoments(flVerbose=True)
rmean,rdev,rSEM,rFOM = MCsolver.returnReflectanceMoments(flVerbose=True)
amean,adev,aSEM,aFOM = MCsolver.returnAbsorptionMoments(flVerbose=True)