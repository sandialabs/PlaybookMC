#!usr/bin/env python
## \file 3DBruteForceSMTransportDriver-TesselationGeoms
#
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools')
from RandomNumberspy import RandomNumbers
from MarkovianInputspy import MarkovianInputs
from FluxTalliespy import FluxTallies
sys.path.append('../../Core/3D')
from MonteCarloParticleSolverpy import MonteCarloParticleSolver
from Particlepy import Particle
from Geometry_Markovianpy import Geometry_Markovian
from Geometry_Voronoipy import Geometry_Voronoi
from Geometry_BoxPoissonpy import Geometry_BoxPoisson
import numpy as np


#Prepare inputs
numparticles  = 1000
numpartupdat  = 100
Solvervariant ='Markovian_realizations' #'Markovian_realizations','BoxPoisson_realizations','Voronoi_realizations'
Geomsize      = 10.0
case          = '3a'  #'1a','1b','1c','2a','2b','2c','3a','3b','3c'; Problem from Adams, Larsen, and Pomraning (ALP) benchmark set
flfluxtallies = False

#Load problem parameters
CaseInp = MarkovianInputs()
CaseInp.selectALPInputs( case )
print(''); print(case)

#Setup random number geneator
Rng = RandomNumbers(flUseSeed=True,seed=11119,stridelen=None)

#Setup multi-D particle object
Part = Particle()
Part.defineDimensionality(dimensionality='3D')
Part.defineParticleInitAngle(initangletype='boundary-isotropic')
Part.defineParticleInitPosition(xrange=[-Geomsize/2,Geomsize/2],yrange=[-Geomsize/2,Geomsize/2],zrange=[-Geomsize/2,-Geomsize/2]) #Sample particle position on negative z face
Part.defineScatteringType(scatteringtype='isotropic')
Part.associateRng(Rng)

#Setup geometry
if   Solvervariant=='Markovian_realizations' : Geom = Geometry_Markovian()
elif Solvervariant=='BoxPoisson_realizations': Geom = Geometry_BoxPoisson()
elif Solvervariant=='Voronoi_realizations'   : Geom = Geometry_Voronoi()
else                                         : raise Exception("For Solvervariant, please choose 'Markovian_realizations', 'BoxPoisson_realizations', or 'Voronoi_realizations'")
Geom.associateRng(Rng)
Geom.associatePart(Part)
Geom.defineGeometryBoundaries(xbounds=[-Geomsize/2,Geomsize/2],ybounds=[-Geomsize/2,Geomsize/2],zbounds=[-Geomsize/2,Geomsize/2])
Geom.defineBoundaryConditions(xBCs=['reflective','reflective'],yBCs=['reflective','reflective'],zBCs=['vacuum','vacuum'])
Geom.defineCrossSections(totxs=CaseInp.Sigt[:],scatxs=CaseInp.Sigs[:])

MarkInp = MarkovianInputs()
MarkInp.solveNaryMarkovianParamsBasedOnChordLengths( lam=CaseInp.lam[:] )
Geom.defineMixingParams( laminf=CaseInp.lamc, prob=CaseInp.prob[:] )
Geom.initializeGeometryMemory()

#Instantiate and associate the general Monte Carlo particle solver
NDMC = MonteCarloParticleSolver()
NDMC.associateRng(Rng)
NDMC.associatePart(Part)
NDMC.associateGeom(Geom)

## If selected, instantiate and associate flux tally object
if flfluxtallies:
    FTal = FluxTallies()
    FTal.setFluxTallyOptions(numMomsToTally=2)
    FTal.defineSMCGeometry(slablength=Geom.zbounds[1]-Geom.zbounds[0],xmin=Geom.zbounds[0])
    NDMC.associateFluxTallyObject(FTal)
    FTal.setupFluxTallies(numTallyBins=100,flMaterialDependent=True,numMaterials=Geom.nummats)
    if FTal.flMaterialDependent:
        FTal.defineFluxTallyMethod( FTal._tallyMaterialBasedCollisionFlux )
        FTal.defineMaterialTypeMethod( FTal._return_iseg_AsMaterialType )
        matfractions = np.ones((Geom.nummats,FTal.numTallyBins))
        for imat in range(0,Geom.nummats):
            matfractions[imat,:] = np.multiply(matfractions[imat,:],Geom.prob[imat])
        FTal.defineMaterialTypeFractions( matfractions )
    else                  :
        FTal.defineFluxTallyMethod( FTal._tallyCollisionFlux )

#Run particle histories
NDMC.pushParticles(NumNewParticles=numparticles,NumParticlesUpdateAt=numpartupdat)

#Compute tallies, return values from function and print to screen
tmean,tdev,tSEM,tFOM = NDMC.returnTransmittanceMoments(flVerbose=True)
rmean,rdev,rSEM,rFOM = NDMC.returnReflectanceMoments(flVerbose=True)
amean,adev,aSEM,aFOM = NDMC.returnAbsorptionMoments(flVerbose=True)
smean,sdev,sSEM,sFOM = NDMC.returnSideLeakageMoments(flVerbose=True)
fmean,fdev,fSEM,fFOM = NDMC.returnFluxMoments(flVerbose=True)

#If selected, print, read, and plot flux tallies (currently must print and read to plot, refactor to not require printing and reading to plot desired)
if flfluxtallies:
    FTal.printFluxVals(filename='3D'+Solvervariant+'Flux'+case,flFOMs=True)
    FTal.readFluxVals( filename='3D'+Solvervariant+'Flux'+case)
    FTal.plotFlux(flMaterialDependent=True)