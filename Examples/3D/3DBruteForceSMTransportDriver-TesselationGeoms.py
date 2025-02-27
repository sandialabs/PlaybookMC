#!usr/bin/env python
## \file 3DBruteForceSMTransportDriver-TesselationGeoms
#  \brief Example driver script for multi-D transport with tesselation geometries.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools')
from RandomNumberspy import RandomNumbers
from MarkovianInputspy import MarkovianInputs
sys.path.append('../../Core/3D')
from MonteCarloParticleSolverpy import MonteCarloParticleSolver
from Particlepy import Particle
from Geometry_Markovianpy import Geometry_Markovian
from Geometry_Voronoipy import Geometry_Voronoi
from Geometry_BoxPoissonpy import Geometry_BoxPoisson

#Prepare inputs
numparticles  = 1000
numpartupdat  = 100
numpartsample = 1   #number of particle histories per realization
numpartitions = 25    #number of partitions of samples - used to provide r.v. statistical precision
Solvervariant = 'BoxPoisson_realizations' #'Markovian_realizations','BoxPoisson_realizations','Voronoi_realizations'
Geomsize      = 10.0
case          = '3b'  #'1a','1b','1c','2a','2b','2c','3a','3b','3c'; Problem from Adams, Larsen, and Pomraning (ALP) benchmark set
numtalbins    = 8
flVoxelize    = False   #turn realizations into voxel versions of themselves?
numVoxels     = [10]*3  #number of voxels in each of three directions

#Load problem parameters
CaseInp = MarkovianInputs()
CaseInp.selectALPInputs( case )
print(''); print(case)

#Setup random number generator
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
else                                         : raise Exception("For Solver variant, please choose 'Markovian_realizations', 'BoxPoisson_realizations', or 'Voronoi_realizations'")
Geom.associateRng(Rng)
Geom.associatePart(Part)
Geom.defineGeometryBoundaries(xbounds=[-Geomsize/2,Geomsize/2],ybounds=[-Geomsize/2,Geomsize/2],zbounds=[-Geomsize/2,Geomsize/2])
Geom.defineBoundaryConditions(xBCs=['reflective','reflective'],yBCs=['reflective','reflective'],zBCs=['vacuum','vacuum'])
Geom.defineCrossSections(totxs=CaseInp.Sigt[:],scatxs=CaseInp.Sigs[:])
Geom.defineVoxelizationParams(flVoxelize=flVoxelize,numVoxels=numVoxels)

MarkInp = MarkovianInputs()
MarkInp.solveNaryMarkovianParamsBasedOnChordLengths( lam=CaseInp.lam[:] )
Geom.defineMixingParams( laminf=CaseInp.lamc, prob=CaseInp.prob[:] )

#Instantiate and associate the general Monte Carlo particle solver
NDMC = MonteCarloParticleSolver(numpartsample)
NDMC.associateRng(Rng)
NDMC.associatePart(Part)
NDMC.associateGeom(Geom)
NDMC.selectFluxTallyOptions(numFluxBins=numtalbins,fluxTallyType='Collision')

#Run particle histories
NDMC.pushParticles(NumNewParticles=numparticles,NumParticlesUpdateAt=numpartupdat)

#Compute tallies, return values from function and print to screen
NDMC.processSimulationFluxTallies()
tmean,tdev,tmeanSEM,tdevSEM = NDMC.returnTransmittanceMoments(flVerbose=True,NumStatPartitions=numpartitions)
rmean,rdev,rmeanSEM,rdevSEM = NDMC.returnReflectanceMoments(flVerbose=True,NumStatPartitions=numpartitions)
amean,adev,ameanSEM,adevSEM = NDMC.returnAbsorptionMoments(flVerbose=True,NumStatPartitions=numpartitions)
smean,sdev,smeanSEM,sdevSEM = NDMC.returnSideLeakageMoments(flVerbose=True,NumStatPartitions=numpartitions)
fmean,fdev,fmeanSEM,fdevSEM = NDMC.returnWholeDomainFluxMoments(flVerbose=True,NumStatPartitions=numpartitions)
print()
NDMC.returnTransmittanceRuntimeAnalysis(flVerbose=True,NumStatPartitions=numpartitions)
NDMC.returnReflectanceRuntimeAnalysis(flVerbose=True,NumStatPartitions=numpartitions)
NDMC.returnAbsorptionRuntimeAnalysis(flVerbose=True,NumStatPartitions=numpartitions)
print()
NDMC.returnRuntimeValues(flVerbose=True)

NDMC.plotFlux(flMaterialDependent=True)