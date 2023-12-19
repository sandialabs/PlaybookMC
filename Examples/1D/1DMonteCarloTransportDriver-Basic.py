#!usr/bin/env python
## \file 1DMonteCarloTransportDriver-Basic.py
#  \brief Example script for running 1D Monte Carlo transport (with no internal flux tallies)
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools')
sys.path.append('../../Core/1D')
from MonteCarloParticleSolverpy import MonteCarloParticleSolver
from OneDSlabpy import Slab

#Monte Carlo options
numparticles  = 50000      #number of particles to run
numpartupdat  = 5000       #number of particles to get runtime updates at
trackerType   = "Woodcock" #"standard" or "Woodcock"

#Set up geometry; instantiate slab by defining materials, material cell boundaries, and the material type in each cell
slab = Slab()
slab.imprintSlab(totxs   =[ 0.01, 0.9, 0.3],
                 scatxs  =[0.005, 0.6, 0.1],
                 matbound=[  0.0, 1.0, 2.0, 3.0, 4.32, 5.0],
                 mattype =[     0,   1,   1,   0,    2 ])

#Instantiate Monte Carlo solver object, connect with slab geometry object, choose source, and initialize random number seed
MCsolver = MonteCarloParticleSolver()
if   trackerType=="standard": MCsolver.defineMCGeometry(slab,slab.s)
elif trackerType=="Woodcock": MCsolver.defineWMCGeometry(max(slab.totxs),slab.samplePoint_x,slab.s)
else                        : raise Exception("Invalid tracker type--\""+trackerType+"\"--chosen; use \"standard\" or \"Woodcock\"")
MCsolver.defineSource('boundary-isotropic')
MCsolver.initializeRandomNumberObject(flUseSeed=True,seed=1)

#Push particles
MCsolver.pushParticles(numparticles,numpartupdat)

#Return transport results, also print to screen if "flVerbose==True"
tmean,tdev,tSEM,tFOM = MCsolver.returnTransmittanceMoments(flVerbose=True)
rmean,rdev,rSEM,rFOM = MCsolver.returnReflectanceMoments(flVerbose=True)
amean,adev,aSEM,aFOM = MCsolver.returnAbsorptionMoments(flVerbose=True)