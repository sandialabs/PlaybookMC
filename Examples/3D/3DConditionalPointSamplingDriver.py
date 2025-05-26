#!usr/bin/env python
## \file 3DConditionalPointSamplingDriver.py
#  \brief Example driver script for multi-D transport with Conditional Point Sampling (CoPS).
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#  \author Alec Shelley, ams01@stanford.edu
import sys
sys.path.append('../../Core/Tools')
from RandomNumberspy import RandomNumbers
from MarkovianInputspy import MarkovianInputs
sys.path.append('../../Core/3D')
from MonteCarloParticleSolverpy import MonteCarloParticleSolver
from Particlepy import Particle
from Geometry_CoPSpy import Geometry_CoPS
import numpy as np


#Prepare inputs
numparticles  = 100
numpartupdat  = 10
numpartsample = 2   #with only short-term memory, batch size; with long-term memory, cohort size
numpartitions = 5   #number of partitions of samples - used to provide r.v. statistical precision
CoPSvariant   = 'higher-accuracy'  #'AM-accuracy', 'CLS-accuracy', 'moderate-accuracy', 'high-accuracy', or 'higher-accuracy' - CoPS has several free parameters that enable it to operate differently (as a different 'CoPS variant'). This variable enables easy user selection between five example CoPS variants that are described in a more detail where the variable is used.
Geomsize      = 10.0  #Edge length of cubic 3D geometry
case          = '1b'  #'1a','1b','1c','2a','2b','2c','3a','3b','3c'; Problem from Adams, Larsen, and Pomraning (ALP) benchmark set
numtalbins    = 8

#Load problem parameters
CaseInp = MarkovianInputs()
CaseInp.selectALPInputs( case )
print(''); print(case)

#Setup random number geneator
Rng = RandomNumbers(flUseSeed=True,seed=54321,stridelen=None)

#Setup multi-D particle object
Part = Particle()
Part.defineDimensionality(dimensionality='3D')
Part.defineParticleInitAngle(initangletype='boundary-isotropic')
Part.defineParticleInitPosition(xrange=[-Geomsize/2,Geomsize/2],yrange=[-Geomsize/2,Geomsize/2],zrange=[-Geomsize/2,-Geomsize/2]) #Sample particle position on negative z face
Part.defineScatteringType(scatteringtype='isotropic')
Part.associateRng(Rng)

#Setup geometry
Geom = Geometry_CoPS()
Geom.associateRng(Rng)
Geom.associatePart(Part)
Geom.defineGeometryBoundaries(xbounds=[-Geomsize/2,Geomsize/2],ybounds=[-Geomsize/2,Geomsize/2],zbounds=[-Geomsize/2,Geomsize/2])
Geom.defineBoundaryConditions(xBCs=['reflective','reflective'],yBCs=['reflective','reflective'],zBCs=['vacuum','vacuum'])
Geom.defineCrossSections(totxs=CaseInp.Sigt[:],scatxs=CaseInp.Sigs[:])


if   CoPSvariant == 'AM-accuracy'      : maxnumpoints = 0; recentmemory = 0; amnesiaradius = None; fllongtermmemory = False; conditionalProbEvaluator='MarkovianAnalytic'           ; exclusionMultiplier = 0.0 #With these options, new points are sampled based only on material abundance (and no previously sampled points). In OlsonMC2023, it was argued and demonstrated that this variant of CoPS yields equivalent transport solutions as the atomic mix (AM) approximation.
elif CoPSvariant == 'CLS-accuracy'     : maxnumpoints = 1; recentmemory = 1; amnesiaradius = None; fllongtermmemory = False; conditionalProbEvaluator='MarkovianIndependentContribs'; exclusionMultiplier = 0.0 #With these options, only the most recently sampled point is remembered and used in conditional probability evluations for the next point. In OlsonMC2023, it was argued and demonstrated that this variant of CoPS, for stochastic media with Markovian mixing statistics, yields equivalent transport solutions as Chord Length Sampling.
elif CoPSvariant == 'moderate-accuracy': maxnumpoints = 1; recentmemory = 1; amnesiaradius = 0.01; fllongtermmemory = True ; conditionalProbEvaluator='MarkovianIndependentContribs'; exclusionMultiplier = 1.0 #With these options, a sampled point is remembered in long-term memory if it is at least 0.01 units from the nearest point which is already stored in long-term memory.  Additionally, the most recently sampled point that was not stored in long-term memory is stored in short-term memory. The nearest point from both sets of memory to a newly sampled point is used in conditional probability evaluations for the next point. The conditional probability model used is the "independent contributions" model. This choice of recent memory and amnesia radius was explored in VuNSE2022.
elif CoPSvariant == 'high-accuracy'    : maxnumpoints = 3; recentmemory = 1; amnesiaradius = 0.01; fllongtermmemory = True ; conditionalProbEvaluator='MarkovianIndependentContribs'; exclusionMultiplier = 1.0 #With these options, a sampled point is remembered in long-term memory as long as it is at least 0.01 units from the nearest point which is already stored in long-term memory.  Additionally, the most recently sampled point that was not stored in long-term memory is stored in short-term memory. The nearest three points to a newly sampled point, drawn from both sets of memory, that aren't excluded based on the exclusion angle, are used in conditional probability evaluations for the next point. The conditional probability model used is the "independent contributions" model. Considering more than one point noticeably increases runtime, but also increases accuracy.
elif CoPSvariant == 'higher-accuracy'  : maxnumpoints = 3; recentmemory = 1; amnesiaradius = 0.01; fllongtermmemory = True ; conditionalProbEvaluator='MarkovianAnalytic'           ; exclusionMultiplier = 0.0 #With these options, a sampled point is remembered in long-term memory as long as it is at least 0.01 units from the nearest point which is already stored in long-term memory.  Additionally, the most recently sampled point that was not stored in long-term memory is stored in short-term memory. The nearest three points to a newly sampled point, drawn from both sets of memory are used in conditional probability evaluations for the next point. The conditional probability model used is the "analytic" model. The analytic model exactly represents N-point relationships in Markovian media, but grows in computational expense with more points more quickly than the "independent contributions" model.
else                                   : raise Exception("Please choose 'AM-accuracy', 'CLS-accuracy', 'moderate-accuracy', 'high-accuracy', or 'higher-accuracy' for CoPSvariant")
Geom.defineConditionalProbabilityEvaluator(conditionalProbEvaluator=conditionalProbEvaluator)
Geom.defineMixingParams(CaseInp.lam[:])
Geom.defineConditionalSamplingParameters(maxNumPoints=maxnumpoints,maxDistance=np.sqrt(3.0)*Geomsize+1.0,exclusionMultiplier=exclusionMultiplier) #These options use up to two governing points for CPF evaluations, exclude no points based on distance (this distance a little greater than largest possible distance in cubic geometry of side length Geomsize), and select a moderate angular exclusion multiplier (as explored in OlsonMC2019; that guards against using multiple points that are very close together as governing points)
Geom.defineLimitedMemoryParameters(recentMemory=recentmemory, amnesiaRadius=amnesiaradius, flLongTermMemory=fllongtermmemory)
Geom.defineCoPSGeometryType(geomtype='Markovian') #geomtype: 'Markovian', 'AM'

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
NDMC.returnRuntimeValues(flVerbose=True)

NDMC.plotFlux(flMaterialDependent=True)