#!usr/bin/env python
## \file 1DMonteCarloTransportDriver.py
#  \brief Examples script for running 1D Monte Carlo transport.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools')
sys.path.append('../../Core/1D')
from MonteCarloParticleSolverpy import MonteCarloParticleSolver
from OneDSlabpy import Slab

#Monte Carlo options
numparticles  = 10000
numpartupdat  = 1000
numpartsample = 5   #batch size
numpartitions = 20    #number of partitions of samples - used to provide r.v. statistical precision
trackerType   = "standard"  #"standard" or "Woodcock"
numtalbins    = 8
fluxTallyType = 'TrackLength' #'TrackLength' or 'Collision'

#Set up geometry; instantiate slab by defining materials, material cell boundaries, and the material type in each cell
slab = Slab()
slab.imprintSlab(totxs   =[ 0.01, 0.9, 0.3],
                 scatxs  =[0.005, 0.6, 0.1],
                 matbound=[  0.0, 1.0, 2.0, 3.0, 4.32, 5.0],
                 mattype =[     0,   1,   1,   0,    2 ])

#Instantiate Monte Carlo solver object, connect with slab geometry object, choose source, and initialize random number seed
MCsolver = MonteCarloParticleSolver(numpartsample)
if   trackerType=="standard": MCsolver.defineMCGeometry(slab)
elif trackerType=="Woodcock": MCsolver.defineWMCGeometry(slab)
else                        : raise Exception("Invalid tracker type--\""+trackerType+"\"--chosen; use \"standard\" or \"Woodcock\"")
MCsolver.defineSourcePosition('left-boundary')
MCsolver.defineSourceAngle('boundary-isotropic')
MCsolver.initializeRandomNumberObject(flUseSeed=True,seed=1)
MCsolver.selectFluxTallyOptions(numFluxBins=numtalbins,fluxTallyType=fluxTallyType)

#Push particles
MCsolver.pushParticles(numparticles,numpartupdat)

#Return transport results, also print to screen if "flVerbose==True"
tmean,tdev,tmeanSEM,tdevSEM = MCsolver.returnTransmittanceMoments(flVerbose=True,NumStatPartitions=numpartitions)
rmean,rdev,rmeanSEM,rdevSEM = MCsolver.returnReflectanceMoments(flVerbose=True,NumStatPartitions=numpartitions)
amean,adev,ameanSEM,adevSEM = MCsolver.returnAbsorptionMoments(flVerbose=True,NumStatPartitions=numpartitions)
MCsolver.processSimulationFluxTallies()
print()
MCsolver.returnRuntimeValues(flVerbose=True)

MCsolver.plotFlux(flMaterialDependent=True,flshow=True,flsave=False,fileprefix='MCProb')