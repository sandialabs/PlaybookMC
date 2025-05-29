#!usr/bin/env python
## \file CompareOutputDataAnalysisDriver.py
#  \brief Example driver script for CompareOutputDataAnalysis class. 
#  \author Revolution Rivera-Felix, rfriver@sandia.gov, revolutionriverafelix@gmail.com
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools') 
from CompareOutputDataAnalysispy import CompareOutputDataAnalysis

# Initialize object
Coda = CompareOutputDataAnalysis()

# Read in data using CODA   
Coda.readCSV(filePath='../../../playbookmc/Counterparts/VuMC2021_N-ary/VuMC2021_Bench_VolFrac_leakage.csv', type='reference',  name='Bench') # reference dataset
Coda.readCSV(filePath='../../../playbookmc/Counterparts/OlsonMC2023/1D_Quaternary_AM_leakage.csv',          type='comparison', name='AM')   # comparison dataset
Coda.readCSV(filePath='../../../playbookmc/Counterparts/VuMC2021_N-ary/VuMC2021_CLS_VolFrac_leakage.csv',   type='comparison', name='CLS')   # comparison dataset
Coda.readCSV(filePath='../../../playbookmc/Counterparts/VuMC2021_N-ary/VuMC2021_LRP_VolFrac_leakage.csv',   type='comparison', name='LRP')   # comparison dataset

# Compute differences; can specify if want magnitude of differences and whether differences are relative to reference values
Coda.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Refl')
Coda.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Trans')
# Save difference values to CSV files
Coda.datasetToCSV(type='comparison', dataset='AM' , fileName=None, dataIdentifiers=('Sign', 'Rel'), flPrintData=False) #Can print only select difference values
Coda.datasetToCSV(type='comparison', dataset='CLS', fileName=None, dataIdentifiers=('Sign', 'Rel'), flPrintData=False) #Can print only select difference values
Coda.datasetToCSV(type='comparison', dataset='LRP', fileName=None, dataIdentifiers=  None         , flPrintData=False) #Can print outputs and difference values

# Plot outputs and differences
Coda.plotOutputsAcrossCases(    quantity='Refl' ,                                  fileName='ReflectanceValues'  , flShowPlot=False, flSavePlot=True, yLim=None, plotTitle=None)
Coda.plotOutputsAcrossCases(    quantity='Trans',                                  fileName='TransmittanceValues', flShowPlot=False, flSavePlot=True, yLim=None, plotTitle=None)
Coda.plotDifferencesAcrossCases(quantity='Refl' , dataIdentifiers=('Sign', 'Rel'), fileName='ReflectanceErrors'  , flShowPlot=True, flSavePlot=True, yLim=None, plotTitle=None)
Coda.plotDifferencesAcrossCases(quantity='Trans', dataIdentifiers=('Sign', 'Rel'), fileName='TransmittanceErrors', flShowPlot=True, flSavePlot=True, yLim=None, plotTitle=None)

# Compute error metrics, save to CSV, and print to screen if specified
Coda.computeDifferenceMetrics(rowIdentifier='Diff')
Coda.datasetToCSV(type='metrics', dataset='AM' , fileName=None, dataIdentifiers=None, flPrintData=True)
Coda.datasetToCSV(type='metrics', dataset='CLS', fileName=None, dataIdentifiers=None, flPrintData=True)
Coda.datasetToCSV(type='metrics', dataset='LRP', fileName=None, dataIdentifiers=None, flPrintData=True)