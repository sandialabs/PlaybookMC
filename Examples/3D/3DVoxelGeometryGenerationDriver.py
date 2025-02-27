#!usr/bin/env python
## \file 3DVoxelGeometryGenerationDriver.py
#  \brief Example driver script for sampling a geometry, converting it to voxel format, and storing.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#  \author Dan Bolintineanu, dsbolin@sandia.gov
import sys
sys.path.append('../../Core/Tools')
from RandomNumberspy import RandomNumbers
sys.path.append('../../Core/3D')
from Geometry_Markovianpy import Geometry_Markovian
from Geometry_Voronoipy import Geometry_Voronoi
from Geometry_BoxPoissonpy import Geometry_BoxPoisson
from Geometry_SphericalInclusionpy import Geometry_SphericalInclusion

# Select geometry type
GeomType  = 'SphericalInclusion' # Choose 'Markovian','BoxPoisson','Voronoi', of 'SphericalInclusion'
Geomsize  = 10.0
numVoxels = [100]*3   # number of voxels in each of three directions

#Setup random number generator and multi-D particle object
Rng = RandomNumbers(flUseSeed=True,seed=14935,stridelen=None)

#Setup geometry
if  GeomType in {'Markovian','BoxPoisson','Voronoi'}:
    if   GeomType=='Markovian' : Geom = Geometry_Markovian()
    elif GeomType=='BoxPoisson': Geom = Geometry_BoxPoisson()
    elif GeomType=='Voronoi'   : Geom = Geometry_Voronoi()
    Geom.defineMixingParams(laminf = 1.0, prob=[0.1,0.4,0.5])
elif GeomType=='SphericalInclusion'   : 
    Geom = Geometry_SphericalInclusion()
    Geom.defineMixingParams(sphereFrac = 0.2, sizeDistribution='Uniform', radMin=0.5, radAve=1.0, radMax=1.5,
                            sphereMatProbs = [0.25,0.75], xperiodic = False, yperiodic=False, zperiodic=False)
    Geom.defineGeometryGenerationParams(sphSampleModel='GenericGrid')

Geom.associateRng(Rng)
Geom.defineGeometryBoundaries(xbounds=[-Geomsize/2,Geomsize/2],ybounds=[-Geomsize/2,Geomsize/2],zbounds=[-Geomsize/2,Geomsize/2])
Geom.defineVoxelizationParams(flVoxelize=True,numVoxels=numVoxels)

#Sample the geometry, voxelize it, and store to h5 and VTK files
Geom._initializeSampleGeometryMemory()
if not Geom.flVoxelize: Geom.voxelizeGeometry() #If flVoxelize==True, this method is called in _initializeSampleGeometryMemory
Geom.writeVoxelMatIndsToH5(filename = 'ExampleGeom.h5')    
Geom.writeVoxelMatIndsToVtk(filename = 'ExampleGeom.vtk') #For Paraview visualization