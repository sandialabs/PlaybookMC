#!usr/bin/env python
## \file 3DBruteForceSMTransportDriver-SphericalInclusions.py
#  \brief Example driver script for multi-D transport with tesselation geometries (currently must have one history per realization).
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools')
from RandomNumberspy import RandomNumbers
from SphericalInclusionInputspy import SphericalInclusionInputs
from FluxTalliespy import FluxTallies
sys.path.append('../../Counterparts/BrantleyMC2011')
sys.path.append('../../Counterparts/BrantleyANS2014')
sys.path.append('../../Core/3D')
from MonteCarloParticleSolverpy import MonteCarloParticleSolver
from Particlepy import Particle
from Geometry_SphericalInclusionpy import Geometry_SphericalInclusion
import numpy as np
import pandas as pd

# plot 3D
# return volume fractions

# voxelize via the base geometry - print voxels to file, sample point for transport based on voxels 

#Prepare inputs
numparticles  = 100
numpartupdat  = 10
raddist       = 'Constant' #'Constant', 'Uniform', 'Exponential'
Geomsize      = 10.0
case          = '2'        #'1','2','3'; case number from BrantleyMC2011 and BrantleyANS2014
volfrac       = '0.05'     #'0.05','0.10','0.15','0.20','0.25','0.30'; volume fraction of spheres as a string

flfluxtallies = False

#Load problem parameters
CaseInp = SphericalInclusionInputs()
CaseInp.selectBMInputs( case,volfrac,raddist )
print(''); print('Case:',case,' Volume Fraction:',volfrac,' Radius Distrubution:',raddist )

#Load published values
if   raddist=="Constant"   : Brant = pd.read_csv('../../Counterparts/BrantleyMC2011/BrantleyMC2011_Spheres_Constant.csv',index_col=0,skiprows=1,encoding='unicode_escape')
elif raddist=="Uniform"    : Brant = pd.read_csv('../../Counterparts/BrantleyMC2011/BrantleyMC2011_Spheres_Uniform.csv',index_col=0,skiprows=1,encoding='unicode_escape')
elif raddist=="Exponential": Brant = pd.read_csv('../../Counterparts/BrantleyANS2014/BrantleyANS2014_Spheres_Exponential.csv',index_col=0,skiprows=1,encoding='unicode_escape')

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
Geom = Geometry_SphericalInclusion(flVerbose=False)
Geom.associateRng(Rng)
Geom.associatePart(Part)
Geom.defineGeometryBoundaries(xbounds=[-Geomsize/2,Geomsize/2],ybounds=[-Geomsize/2,Geomsize/2],zbounds=[-Geomsize/2,Geomsize/2])
Geom.defineBoundaryConditions(xBCs=['reflective','reflective'],yBCs=['reflective','reflective'],zBCs=['vacuum','vacuum'])
Geom.defineCrossSections(totxs=CaseInp.Sigt[:],scatxs=CaseInp.Sigs[:])

Geom.defineMixingParams(sphereFrac=CaseInp.sphereFrac,radMin=CaseInp.radMin,radAve=CaseInp.radAve,radMax=CaseInp.radMax,sizeDistribution=CaseInp.sizeDistribution,matSphProbs=CaseInp.matSphProbs,matMatrix=CaseInp.matMatrix)
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

#Print published values to screen
if   raddist=="Constant"   : print("\n--BrantleyMC2011 values (read from plot, uncertainty ~10%--)")
elif raddist=="Uniform"    : print("\n--BrantleyMC2011 values (read from plot, uncertainty ~20%--)")
elif raddist=="Exponential": print("\n--BrantleyANS2014 values (read from plot, uncertainty ~10%--)")
print("Transmittance            :  ",Brant['Case'+case+'-Frac'+volfrac]['Trans'])
print("Reflectance              :  ",Brant['Case'+case+'-Frac'+volfrac]['Refl'],'\n')

if   raddist=="Constant"   : Brant = pd.read_csv('../../Counterparts/BrantleyMC2011/BrantleyMC2011_Spheres_Constant.csv',index_col=0,skiprows=1,encoding='unicode_escape')
elif raddist=="Uniform"    : Brant = pd.read_csv('../../Counterparts/BrantleyMC2011/BrantleyMC2011_Spheres_Uniform.csv',index_col=0,skiprows=1,encoding='unicode_escape')
elif raddist=="Exponential": Brant = pd.read_csv('../../Counterparts/BrantleyANS2014/BrantleyANS2014_Spheres_Exponential.csv',index_col=0,skiprows=1,encoding='unicode_escape')


#Compute tallies, return values from function and print to screen
tmean,tdev,tSEM,tFOM = NDMC.returnTransmittanceMoments(flVerbose=True)
rmean,rdev,rSEM,rFOM = NDMC.returnReflectanceMoments(flVerbose=True)
amean,adev,aSEM,aFOM = NDMC.returnAbsorptionMoments(flVerbose=True)
smean,sdev,sSEM,sFOM = NDMC.returnSideLeakageMoments(flVerbose=True)
fmean,fdev,fSEM,fFOM = NDMC.returnFluxMoments(flVerbose=True)

#If selected, print, read, and plot flux tallies (currently must print and read to plot, refactor to not require printing and reading to plot desired)
if flfluxtallies:
    FTal.printFluxVals(filename='3DSpheresFlux',flFOMs=True)
    FTal.readFluxVals( filename='3DSpheresFlux')
    FTal.plotFlux(flMaterialDependent=True)