#!usr/bin/env python
## \file 1DMonteCarloTransportDriver-WithTallies.py
#  \brief Examples script for running 1D Monte Carlo transport with flux tallies.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools')
from FluxTalliespy import FluxTallies
sys.path.append('../../Core/1D')
from MonteCarloParticleSolverpy import MonteCarloParticleSolver
from OneDSlabpy import Slab

#Monte Carlo options
numparticles  = 10000
numpartupdat  = 1000
trackerType   = "Woodcock"  #"standard" or "Woodcock"

#Flux tally options
numtalbins    = 50
TallyType     = 'Collision' #'TrackLength','Collision','PBTL' (point-based track-length tallies)
flmatdep      = True
numPBTLtalpts = 2
if TallyType not in ('TrackLength','Collision','PBTL') : raise Exception("Please choose 'TrackLength', 'Collision', or 'PBTL' as your tally type")
if trackerType=='Woodcock' and TallyType=='TrackLength': raise Exception("For a Woodcock tracker, please choose collision or PBTL tallies")

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

#Set up flux tallies
FTal = FluxTallies()
FTal.setFluxTallyOptions(numMomsToTally=4)
FTal.defineMCGeometry(slab,slab.s)
FTal.initializeRandomNumberObject(flUseSeed=False)
MCsolver.associateFluxTallyObject(FTal)
FTal.setupFluxTallies(numTallyBins=numtalbins,flMaterialDependent=flmatdep,numMaterials=slab.nummats)
if FTal.flMaterialDependent:
    if   TallyType == 'TrackLength': FTal.defineFluxTallyMethod( FTal._tallyMaterialBasedTrackLengthFlux )
    elif TallyType == 'PBTL'       : FTal.defineFluxTallyMethod( FTal._tallyMaterialBasedPointBasedTrackLengthFlux,numPBTLtalpts )
    elif TallyType == 'Collision'  : FTal.defineFluxTallyMethod( FTal._tallyMaterialBasedCollisionFlux )
    FTal.defineMaterialTypeMethod( FTal._returnSlabMaterialType )
    FTal.Slab.solveMaterialTypeFractions( numbins=FTal.numTallyBins )
    FTal.defineMaterialTypeFractions( matfractions=FTal.Slab.MatFractions )
else                  :
    if   TallyType == 'TrackLength': FTal.defineFluxTallyMethod( FTal._tallyTrackLengthFlux )
    elif TallyType == 'PBTL'       : FTal.defineFluxTallyMethod( FTal._tallyPointBasedTrackLengthFlux,numPBTLtalpts )
    elif TallyType == 'Collision'  : FTal.defineFluxTallyMethod( FTal._tallyCollisionFlux )

#Push particles
MCsolver.pushParticles(numparticles,numpartupdat)

#Return transport results, also print to screen if "flVerbose==True"
tmean,tdev,tSEM,tFOM = MCsolver.returnTransmittanceMoments(flVerbose=True)
rmean,rdev,rSEM,rFOM = MCsolver.returnReflectanceMoments(flVerbose=True)
amean,adev,aSEM,aFOM = MCsolver.returnAbsorptionMoments(flVerbose=True)

#Print, read, and plot flux tallies (currently must print and read to plot, refactor to not require printing and reading to plot desired)
FTal.printFluxVals(filename='1DTMCSolverFlux',flFOMs=True)
FTal.readFluxVals( filename='1DTMCSolverFlux')
FTal.plotFlux(flMaterialDependent=True)