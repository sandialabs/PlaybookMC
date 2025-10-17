#!usr/bin/env python
## \file 3DBruteForceTransportDriver-UserDefinedVoxelGeometry
#  \brief Example driver script for multi-D transport with a user-defined voxel geometry.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
import numpy as np
sys.path.append('../../Core/Tools')
from RandomNumberspy import RandomNumbers
sys.path.append('../../Core/3D')
from MonteCarloParticleSolverpy import MonteCarloParticleSolver
from Particlepy import Particle
from Geometry_Voxelpy import Geometry_Voxel

#Prepare inputs
numparticles  = 100000
numpartupdat  = 10000
numpartsample = 100    #number of particle histories per realization
numpartitions = 25   #number of partitions of samples - used to provide r.v. statistical precision
GeomLengths   = [10.0,10.0,6.0]
numtalbins    = 60
numleakbins   = 19
numVoxels     = [5,5,3]  #number of voxels in each of three directions

flshow     = True         #show results plots when run
flsave     = False        #save results plots to file
fileprefix = 'AnnulusProb' #file prefix for saved plots

#Setup random number generator
Rng = RandomNumbers(flUseSeed=True,seed=11119,stridelen=None)

#Setup multi-D particle object
Part = Particle()
Part.defineDimensionality(dimensionality='3D')
Part.defineSourceAngle(sourgeAngleType='boundary-isotropic')
Part.defineSourcePosition(sourceLocationType='annulus',annulusCenter=[-0.0,-0.0,-GeomLengths[2]/2],annulusCDF=[0.0,0.15,0.15,1.0],annulusRadii=[1.0,1.0,4.5,4.5]) #Sample particle position from rings on negative z face
Part.defineScatteringType(scatteringtype='isotropic-3Drod')
Part.associateRng(Rng)

#Setup geometry
Geom = Geometry_Voxel()
Geom.associateRng(Rng)
Geom.associatePart(Part)
Geom.defineGeometryBoundaries(xbounds=[-GeomLengths[0]/2,GeomLengths[0]/2],ybounds=[-GeomLengths[1]/2,GeomLengths[1]/2],zbounds=[-GeomLengths[2]/2,GeomLengths[2]/2])
Geom.defineBoundaryConditions(xBCs=['vacuum','vacuum'],yBCs=['vacuum','vacuum'],zBCs=['vacuum','vacuum'])
Geom.defineCrossSections(totxs=[1.0,0.02],scatxs=[0.95,0.01])
GeomThroughDepth = [[0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1]]
Geom.initializeUserDefinedVoxelGeometry(VoxelMatInds=[GeomThroughDepth]*5,numfluxbins=numtalbins,numSampPerFluxBin=1)

#Instantiate and associate the general Monte Carlo particle solver
NDMC = MonteCarloParticleSolver(numpartsample)
NDMC.associateRng(Rng)
NDMC.associatePart(Part)
NDMC.associateGeom(Geom)
NDMC.selectFluxTallyOptions(numFluxBins=numtalbins,fluxTallyType='Collision')
NDMC.selectLeakageTallyOptions(numLeakageBins_x=numleakbins,numLeakageBins_y=numleakbins)

#Run particle histories
NDMC.pushParticles(NumNewParticles=numparticles,NumParticlesUpdateAt=numpartupdat)

#Compute tallies, return values from function and print to screen
NDMC.processSimulationFluxTallies()
NDMC.processSimulationLeakageTallies()
tmean,tdev,tmeanSEM,tdevSEM = NDMC.returnTransmittanceMoments(flVerbose=True,NumStatPartitions=numpartitions)
rmean,rdev,rmeanSEM,rdevSEM = NDMC.returnReflectanceMoments(flVerbose=True,NumStatPartitions=numpartitions)
amean,adev,ameanSEM,adevSEM = NDMC.returnAbsorptionMoments(flVerbose=True,NumStatPartitions=numpartitions)
smean,sdev,smeanSEM,sdevSEM = NDMC.returnSideLeakageMoments(flVerbose=True,NumStatPartitions=numpartitions)
fmean,fdev,fmeanSEM,fdevSEM = NDMC.returnWholeDomainFluxMoments(flVerbose=True,NumStatPartitions=numpartitions)
print()
NDMC.returnRuntimeValues(flVerbose=True)

NDMC.plotFlux(flMaterialDependent=False,flshow=flshow,flsave=flsave,fileprefix=fileprefix)
NDMC.plotTransmittance(    flgray=False,flshow=flshow,flsave=flsave,fileprefix=fileprefix)
NDMC.plotReflectance(      flgray=False,flshow=flshow,flsave=flsave,fileprefix=fileprefix)