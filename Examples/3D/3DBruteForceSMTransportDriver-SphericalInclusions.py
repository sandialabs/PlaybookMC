#!usr/bin/env python
## \file 3DBruteForceSMTransportDriver-SphericalInclusions.py
#  \brief Example driver script for multi-D transport with spherical-inclusion geometries.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#  \author Dan Bolintineanu, dsbolin@sandia.gov
import sys
sys.path.append('../../Core/Tools')
from RandomNumberspy import RandomNumbers
from SphericalInclusionInputspy import SphericalInclusionInputs
from CompareOutputDataAnalysispy import CompareOutputDataAnalysis
sys.path.append('../../Core/3D')
from MonteCarloParticleSolverpy import MonteCarloParticleSolver
from Particlepy import Particle
from Geometry_SphericalInclusionpy import Geometry_SphericalInclusion
import pandas as pd

#Prepare inputs
numparticles  = 200
numpartupdat  = 10
numpartsample = 10   #number of particle histories per realization
numpartitions = 5    #number of partitions of samples - used to provide r.v. statistical precision
raddist       = 'Constant' #'Constant', 'Uniform', 'Exponential'
Geomsize      = 10.0
case          = '2'        #'1','2','3'; case number from BrantleyMC2011 and BrantleyANS2014
volfrac       = '0.20'     #'0.05','0.10','0.15','0.20','0.25','0.30'; volume fraction of spheres as a string
sphSampleModel= 'FastRSA'   #'NoGrid', 'GenericGrid', or 'FastRSA'
abundanceModel= 'ensemble' #'ensemble' or 'sample'
numtalbins    = 20
flVoxelize    = False   #turn realizations into voxel versions of themselves?
numVoxels     = [10]*3  #number of voxels in each of three directions

#Load problem parameters
CaseInp = SphericalInclusionInputs()
CaseInp.selectBMInputs( case,volfrac,raddist )
print(''); print('Case:',case,' Volume Fraction:',volfrac,' Radius Distrubution:',raddist )

#Load published values
Coda = CompareOutputDataAnalysis()
if   raddist=="Constant"   : Coda.readCSV(filePath='../../../playbookmc/Counterparts/BrantleyMC2011/BrantleyMC2011_Spheres_Constant.csv', type='reference',  name='Brant')
elif raddist=="Uniform"    : Coda.readCSV(filePath='../../../playbookmc/Counterparts/BrantleyMC2011/BrantleyMC2011_Spheres_Uniform.csv', type='reference',  name='Brant')
elif raddist=="Exponential": Coda.readCSV(filePath='../../../playbookmc/Counterparts/BrantleyANS2014/BrantleyANS2014_Spheres_Exponential.csv', type='reference',  name='Brant')

#Setup random number geneator
Rng = RandomNumbers(flUseSeed=True,seed=11119,stridelen=None)

#Setup multi-D particle object
Part = Particle()
Part.defineDimensionality(dimensionality='3D')
Part.defineSourceAngle(sourgeAngleType='boundary-isotropic')
Part.defineSourcePosition(sourceLocationType='cuboid',xrange=[-Geomsize/2,Geomsize/2],yrange=[-Geomsize/2,Geomsize/2],zrange=[-Geomsize/2,-Geomsize/2]) #Sample particle position on negative z face
Part.defineScatteringType(scatteringtype='isotropic')
Part.associateRng(Rng)

#Setup geometry
Geom = Geometry_SphericalInclusion(flVerbose=False,abundanceModel=abundanceModel)
Geom.associateRng(Rng)
Geom.associatePart(Part)
Geom.defineGeometryBoundaries(xbounds=[-Geomsize/2,Geomsize/2],ybounds=[-Geomsize/2,Geomsize/2],zbounds=[-Geomsize/2,Geomsize/2])
Geom.defineBoundaryConditions(xBCs=['reflective','reflective'],yBCs=['reflective','reflective'],zBCs=['vacuum','vacuum'])
Geom.defineCrossSections(totxs=CaseInp.Sigt[:],scatxs=CaseInp.Sigs[:])
Geom.defineMixingParams(sphereFrac=CaseInp.sphereFrac,radMin=CaseInp.radMin,radAve=CaseInp.radAve,radMax=CaseInp.radMax,
                        sizeDistribution=CaseInp.sizeDistribution,sphereMatProbs=CaseInp.sphereMatProbs,
                        xperiodic=False,yperiodic=False,zperiodic=False)
Geom.defineGeometryGenerationParams(maxPlacementAttempts=10000, sphSampleModel=sphSampleModel, gridSize=None)
Geom.defineVoxelizationParams(flVoxelize=flVoxelize,numVoxels=numVoxels)

#Instantiate and associate the general Monte Carlo particle solver
NDMC = MonteCarloParticleSolver(numpartsample)
NDMC.associateRng(Rng)
NDMC.associatePart(Part)
NDMC.associateGeom(Geom)
NDMC.selectFluxTallyOptions(numFluxBins=numtalbins,fluxTallyType='Collision')

#Run particle histories
NDMC.pushParticles(NumNewParticles=numparticles,NumParticlesUpdateAt=numpartupdat)

#Print published values to screen
if   raddist=="Constant"   : print("\n--BrantleyMC2011 values (read from plot, uncertainty ~10%--)")
elif raddist=="Uniform"    : print("\n--BrantleyMC2011 values (read from plot, uncertainty ~20%--)")
elif raddist=="Exponential": print("\n--BrantleyANS2014 values (read from plot, uncertainty ~10%--)")
print("Transmittance               :  ",Coda.reference['Brant'][case+'-'+volfrac]['Trans'])
print("Reflectance                 :  ",Coda.reference['Brant'][case+'-'+volfrac]['Refl'],'\n')

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

NDMC.plotFlux(flMaterialDependent=True,flshow=True,flsave=False,fileprefix='SphereProb')