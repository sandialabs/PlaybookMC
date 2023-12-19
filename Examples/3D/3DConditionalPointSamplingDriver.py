#!usr/bin/env python
## \file 3DConditionalPointSamplingDriver.py
#  \brief Example driver script for multi-D transport with Conditional Point Sampling (CoPS).
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools')
from RandomNumberspy import RandomNumbers
from MarkovianInputspy import MarkovianInputs
from FluxTalliespy import FluxTallies
sys.path.append('../../Core/3D')
from MonteCarloParticleSolverpy import MonteCarloParticleSolver
from Particlepy import Particle
from Geometry_CoPSpy import Geometry_CoPS
import numpy as np


#Prepare inputs
numparticles  = 1000
numpartupdat  = 100
CoPSvariant   = 'CLS-accuracy'  #'AM-accuracy', 'CLS-accuracy', 'moderate-accuracy', or 'errorless-accuracy' - CoPS has several free parameters that enable it to operate differently. In this script, this variable enables user selection between four sets of options that are described in a more detail when this variable is used.
Geomsize      = 10.0  #Edge length of cubic 3D geometry
case          = '3a'  #'1a','1b','1c','2a','2b','2c','3a','3b','3c'; Problem from Adams, Larsen, and Pomraning (ALP) benchmark set
flfluxtallies = False

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

Geom.defineMixingParams(lam=CaseInp.lam[:])
if   CoPSvariant == 'AM-accuracy'      : maxnumpoints = 0; recentmemory = 0; amnesiaradius = None; fllongtermmemory = False #With these options, new points are sampled based only on material abundance (and no previously sampled points).  In OlsonMC2023, it was argued and demonstrated that when running CoPS with these options yields equivalent transport solutions as the atomic mix (AM) approximation.
elif CoPSvariant == 'CLS-accuracy'     : maxnumpoints = 1; recentmemory = 1; amnesiaradius = None; fllongtermmemory = False #With these options, only the most recently sampled point is remembered and used in conditional probability evluations for the next point. In OlsonMC2023, it was argued and demonstrated that when running CoPS with these options, for stochastic media with Markovian mixing statistics, yields equivalent transport solutions as Chord Length Sampling
elif CoPSvariant == 'moderate-accuracy': maxnumpoints = 1; recentmemory = 1; amnesiaradius = 0.01; fllongtermmemory = True  #With these options, a sampled point is remembered in long-term memory as long as it is at least 0.01 units from the nearest point which is already stored in long-term memory.  Additionally, the most recently sampled point that was not stored in long-term memory is stored in short-term memory. The nearest point from both sets of memory to a newly sampled point is used in conditional probability evaluations for the next point. This choice of recent memory and amnesia radius was explored in VuNSE2022.
elif CoPSvariant == 'high-accuracy'    : maxnumpoints = 3; recentmemory = 1; amnesiaradius = 0.01; fllongtermmemory = True  #With these options, a sampled point is remembered in long-term memory as long as it is at least 0.01 units from the nearest point which is already stored in long-term memory.  Additionally, the most recently sampled point that was not stored in long-term memory is stored in short-term memory. The nearest three points to a newly sampled point, drawn from both sets of memory, that aren't excluded based on the exclusion angle, are used in conditional probability evaluations for the next point. Considering more than one point noticeably increases runtime, but also increases accuracy.
else                                   : raise Exception("Please choose 'CLS-accuracy', 'moderate-accuracy', or 'high-accuracy' for CoPSvariant")
Geom.defineConditionalSamplingParameters(maxNumPoints=maxnumpoints,maxDistance=np.sqrt(3.0)*Geomsize+1.0,exclusionMultiplier=1.0) #These options use up to two governing points for CPF evaluations, exclude no points based on distance (this distance a little greater than largest possible distance in cubic geometry of side length Geomsize), and select a moderate angular exclusion multiplier (as explored in OlsonMC2019; that guards against using multiple points that are very close together as governing points)
Geom.defineLimitedMemoryParameters(recentMemory=recentmemory, amnesiaRadius=amnesiaradius, flLongTermMemory=fllongtermmemory)
Geom.defineCoPSGeometryType(geomtype='Markovian',fl1DEmulation=False) #geomtype: 'Markovian', 'AM'
Geom.defineConditionalProbabilityEvaluator(Geom.computeConditionalProbabilitiesFromGoverningPointsUsingPseudoInterfaces)
Geom.initializeGeometryMemory()

#Instantiate and associate the general Monte Carlo particle solver
NDMC = MonteCarloParticleSolver()
NDMC.associateRng(Rng)
NDMC.associatePart(Part)
NDMC.associateGeom(Geom)

#If selected, instantiate and associate flux tally object
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
    FTal.printFluxVals(filename='3DCoPSFlux'+case,flFOMs=True)
    FTal.readFluxVals( filename='3DCoPSFlux'+case)
    FTal.plotFlux(flMaterialDependent=True)