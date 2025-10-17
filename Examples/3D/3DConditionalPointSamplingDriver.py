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
from Geometry_Markovianpy import Geometry_Markovian
from VoxelSpatialStatisticspy import StatMetrics
import numpy as np
## Brief overview of CoPS
# In Conditional Point Sampling (CoPS), particles are streamed through a stochastic medium using 
# Woodcock/delta tracking.  The material at pseudo-collision sites, which are points in space, is
# sampled according to a conditional probaiblity function (CPF), which is conditioned on geometric data,
# typically on material abundance and material type at previously sampled points.  A unique CoPS variant
# is defined by selecting
#   1) which sampled points will be remembered and for how long to be used in future conditional
#      evaluations (a memory scheme)
#   2) which of the previously sampled points will be used in a particular conditional probability
#      evaluation (a point-downselection scheme), and
#   3) which model will be used to compute the conditional probability of a new point based on the
#      downselected points (a CPF).
# Several pre-packaged memory schemes and the CPFs currently implemented in PlaybookMC are available 
# in this script by selecting values of CoPS_Memory and CoPS_CPF.  Typical point-downselection 
# options are chosen in this script based on these values.  These are meant to serve as examples of 
# defining CoPS variants from which a user can select whatever selection of values is of interest.

#Prepare inputs
numparticles  = 10000
numpartupdat  = 1000
numpartsample = 2   #with only short-term memory, batch size; with long-term memory, cohort size
numpartitions = 5   #number of partitions of samples - used to provide r.v. statistical precision
CoPS_Memory   = 'Recent-Memory'     #'Unconditional','Recent-Memory','Amnesia-Radius','Hybrid-Memory','Full-Memory'; Choose prepackaged CoPS memory scheme
CoPS_CPF      = 'MarkovianAnalytic' #'MarkovianAnalytic','MarkovianCombination','MICK-Analytic','MICK-Numeric'; Choose CoPS conditional probability function (CPF)
Geomsize      = 10.0  #Edge length of cubic 3D geometry
case          = '3a'  #'1a','1b','1c','2a','2b','2c','3a','3b','3c'; Problem from Adams, Larsen, and Pomraning (ALP) benchmark set
numtalbins    = 8

if CoPS_Memory not in {'Unconditional','Recent-Memory','Amnesia-Radius','Hybrid-Memory','Full-Memory'}:
    raise Exception("Please choose 'Unconditional','Recent-Memory','Amnesia-Radius','Hybrid-Memory', or 'Full-Memory' for CoPS_Memory")
if CoPS_CPF not in {'MarkovianAnalytic','MarkovianCombination','MICK-Analytic','MICK-Numeric'}:
    raise Exception("Please choose 'MarkovianAnalytic','MarkovianCombination','MICK-Analytic','MICK-Numeric' for CoPS_CPF")

#Load problem parameters
CaseInp = MarkovianInputs()
CaseInp.selectALPInputs( case )
print(''); print(case)

#Setup random number geneator
Rng = RandomNumbers(flUseSeed=True,seed=54321,stridelen=None)

#Setup multi-D particle object
Part = Particle()
Part.defineDimensionality(dimensionality='3D')
Part.defineSourceAngle(sourgeAngleType='boundary-isotropic')
Part.defineSourcePosition(sourceLocationType='cuboid',xrange=[-Geomsize/2,Geomsize/2],yrange=[-Geomsize/2,Geomsize/2],zrange=[-Geomsize/2,-Geomsize/2]) #Sample particle position on negative z face
Part.defineScatteringType(scatteringtype='isotropic')
Part.associateRng(Rng)

#Setup geometry
CoPSGeom = Geometry_CoPS()
CoPSGeom.associateRng(Rng)
CoPSGeom.associatePart(Part)
CoPSGeom.defineGeometryBoundaries(xbounds=[-Geomsize/2,Geomsize/2],ybounds=[-Geomsize/2,Geomsize/2],zbounds=[-Geomsize/2,Geomsize/2])
CoPSGeom.defineBoundaryConditions(xBCs=['reflective','reflective'],yBCs=['reflective','reflective'],zBCs=['vacuum','vacuum'])
CoPSGeom.defineCrossSections(totxs=CaseInp.Sigt[:],scatxs=CaseInp.Sigs[:])

#Select CoPS memory options
if   CoPS_Memory=='Unconditional' : recentmemory = 0; fllongtermmemory = False; amnesiaradius = None
elif CoPS_Memory=='Recent-Memory' : recentmemory = 1; fllongtermmemory = False; amnesiaradius = None
elif CoPS_Memory=='Amnesia-Radius': recentmemory = 0; fllongtermmemory = True ; amnesiaradius = 0.01
elif CoPS_Memory=='Hybrid-Memory' : recentmemory = 1; fllongtermmemory = True ; amnesiaradius = 0.01
elif CoPS_Memory=='Full-Memory'   : recentmemory = 0; fllongtermmemory = True ; amnesiaradius = 0.0
CoPSGeom.defineLimitedMemoryParameters(recentMemory=recentmemory, amnesiaRadius=amnesiaradius, flLongTermMemory=fllongtermmemory)

#Select CoPS point-downselection options
if   CoPS_Memory=='Unconditional'                    : maxnumpoints = 0; exclusionMultiplier = 0.0; flRefillToMaxPoints = False
elif CoPS_Memory=='Recent-Memory'                    : maxnumpoints = 1; exclusionMultiplier = 0.0; flRefillToMaxPoints = False
elif CoPS_Memory in {'Amnesia-Radius','Hybrid-Memory','Full-Memory'}:
    if   CoPS_CPF=='MarkovianAnalytic'               : maxnumpoints = 3; exclusionMultiplier = 0.0; flRefillToMaxPoints = True
    elif CoPS_CPF=='MarkovianCombination'            : maxnumpoints = 3; exclusionMultiplier = 1.0; flRefillToMaxPoints = False
    elif CoPS_CPF in {'MICK-Analytic','MICK-Numeric'}: maxnumpoints = 3; exclusionMultiplier = 0.0; flRefillToMaxPoints = True
CoPSGeom.defineConditionalSamplingParameters(maxNumPoints=maxnumpoints,maxDistance=Geomsize,exclusionMultiplier=exclusionMultiplier,flRefillToMaxPoints=flRefillToMaxPoints)

#Select CoPS CPF type
if   CoPS_CPF=='MarkovianAnalytic'               : conditionalProbEvaluator='MarkovianAnalytic'
elif CoPS_CPF=='MarkovianCombination'            : conditionalProbEvaluator='MarkovianCombination'
elif CoPS_CPF in {'MICK-Analytic','MICK-Numeric'}: conditionalProbEvaluator='MultipleIndicatorCoKriging'
CoPSGeom.defineConditionalProbabilityEvaluator(conditionalProbEvaluator=conditionalProbEvaluator)

#Define mixing parameters (based on CoPS CPF type chosen)
if   CoPS_CPF in {'MarkovianAnalytic','MarkovianCombination'}: CoPSGeom.defineMixingParams(CaseInp.lam[:])
elif CoPS_CPF=='MICK-Analytic'                                       :
    CaseInp.generateAnalyticalS2CallableArray()
    CoPSGeom.defineMixingParams(CaseInp.AnalyticalS2CallableArray, CaseInp.prob) 
elif CoPS_CPF=='MICK-Numeric':
    prefix = "CoPSDriverExample_ALP_"+case #File name prefix for geometry and statistics files for numeric MICK
    try: #Try to read abundances and S2s from files
        Stats = StatMetrics()
        Stats.readMaterialAbundancesFromText(prefix+"_mat_abunds.txt")
        Stats.readS2FromCSV(prefix)
    except Exception as e: #If abundances or S2s are not available, generate geometry and calculate stats
        print(f"Failed to read statistics, with exception: {e}")
        print(f"Generating new geometry and computing statistics from it")

        Geomsize  = 10.0
        numVoxels = [400]*3   # number of voxels in each of three directions--increase for better numerical accuracy
        RealizationRng = RandomNumbers(flUseSeed=True,seed=12345,stridelen=None)
        RealizationGeom = Geometry_Markovian()
        RealizationGeom.defineMixingParams(laminf=CaseInp.lamc, prob=CaseInp.prob[:])
        RealizationGeom.associateRng(RealizationRng)
        RealizationGeom.defineGeometryBoundaries(xbounds=[-Geomsize*1.2,Geomsize*1.2],ybounds=[-Geomsize*1.2,Geomsize*1.2],zbounds=[-Geomsize*1.2,Geomsize*1.2]) #Creating a realization larger than the target domain to get a broader survey of material behavior
        RealizationGeom.defineVoxelizationParams(flVoxelize=True,numVoxels=numVoxels)
        RealizationGeom._initializeSampleGeometryMemory()
        RealizationGeom.writeVoxelMatIndsToH5(filename = prefix+'.h5')    

        Stats = StatMetrics(RealizationGeom)    
        #Calculate and plot S2 spatial statistics
        Stats.calculateMaterialAbundances(flVerbose=True)
        Stats.writeMaterialAbundancesToText(f"{prefix}_mat_abunds.txt")
        Stats.calculateS2()
        Stats.writeS2ToCSV(prefix) 

    CoPSGeom.defineMixingParams(Stats)

#Instantiate and associate the general Monte Carlo particle solver
NDMC = MonteCarloParticleSolver(numpartsample)
NDMC.associateRng(Rng)
NDMC.associatePart(Part)
NDMC.associateGeom(CoPSGeom)
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

NDMC.plotFlux(flMaterialDependent=True,flshow=True,flsave=False,fileprefix='CoPSProb')