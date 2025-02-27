#!usr/bin/env python
## \file 3DVoxelStatisticsDriver.py
#  \brief Example driver script for computing spatial statistics metrics on voxel geometry.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#  \author Dan Bolintineanu, dsbolin@sandia.gov
#
# To generate example voxel data file to compute on, run 3DVoxelGeometryGenerationDriver.py.
import sys
sys.path.append('../../Core/Tools')
from RandomNumberspy import RandomNumbers
sys.path.append('../../Core/3D')
from Geometry_Voxelpy import Geometry_Voxel
from VoxelSpatialStatisticspy import StatMetrics

# Specify .h5 file name prefix to read from
prefix = 'ExampleGeom'

# Read in voxel geometry from h5 format and initialize statistical metrics class
Geom = Geometry_Voxel()
Geom.readVoxelMatIndsFromH5(filename = prefix+'.h5')
Stats = StatMetrics(Geom)
print()


#Calculate material abundances
Stats.calculateMaterialAbundances(flVerbose=True)
print()

#Calculate and plot chord lengths distributions (CLDs)
Stats.calculateChordLengthDistributions(flExcludeEdgeChords=True,flReturnChordLengthPDFs=True,flVerbose=True)
Stats.plotCLDResults(outputfilename=f"{prefix}_cld.png",plotStyle='nonzero',flNormalizeCounts=True,flShowPlot=True)
print()

#Calculate and plot S2 spatial statistics
Stats.calculateS2()
Stats.plotS2Results(outputfilename=f"{prefix}_s2.png",flPlotLimits=True,flPlotStandardYAxis=True,flShowPlot=True)


#Demo writing and reading data
Stats.writeMaterialAbundancesToText(f"{prefix}_mat_abunds.txt")
Stats.readMaterialAbundancesFromText(f"{prefix}_mat_abunds.txt")
Stats.writeCLDSummaryToText(f"{prefix}_cld_summary.txt")
Stats.readCLDSummaryFromText(f"{prefix}_cld_summary.txt")
Stats.writeCLDToCSV(prefix)
Stats.readCLDFromCSV(prefix)
Stats.writeS2ToCSV(prefix) 
Stats.readS2FromCSV(prefix)