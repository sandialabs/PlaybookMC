#!usr/bin/env python
## \file CounterpartsAgreementAssessment.py
#  \brief Testing agreements between Counterparts data using CompareOutputDataAnalysis class. 
#  \author Revolution Rivera-Felix, rfriver@sandia.gov, revolutionriverafelix@gmail.com

import sys
sys.path.append('../../Core/Tools') 
from CompareOutputDataAnalysispy import CompareOutputDataAnalysis

benchmarkSets = ["1DALPMeans", "1DALPVariances", "3DALPMeans", "1DVBOKMeans", "SphericalInclusions"] # "1DALPMeans", "1DALPVariances", "3DALPMeans", "1DVBOKMeans", "SphericalInclusions"; Select benchmark set(s) to compare via plots
flAgreements  = True           # Plot Agreements
flMethods     = True           # Plot Methods

if "1DALPMeans" in benchmarkSets:
    # Mean Values on 1D ALP/Binary Benchmark Problems
    if flAgreements:
        ## 1D Benchmark Means Agreement
        _1DBenchMeans = CompareOutputDataAnalysis() 
        _1DBenchMeans.readCSV(filePath='../../../playbookmc/Counterparts/VuJQSRT2021/VuJQSRT2021_Bench_means.csv',                        type='reference',  name='VuJQSRT2021 Bench means')
        _1DBenchMeans.readCSV(filePath='../../../playbookmc/Counterparts/BrantleyJQSRT2011/1DBrantleyJQSRT2011_Bench_SuiteI_leakage.csv', type='comparison', name='1D BrantleyJQSRT2011 Bench means')  
        _1DBenchMeans.readCSV(filePath='../../../playbookmc/Counterparts/VuJQSRT2021/VuJQSRT2021_CoPS3PO_means.csv',                      type='comparison', name='VuJQSRT2021 CoPS3PO means') 
        ## 1D AM Means Agreement
        _1DAMMeans = CompareOutputDataAnalysis() 
        _1DAMMeans.readCSV(   filePath='../../../playbookmc/Counterparts/VuJQSRT2021/VuJQSRT2021_AM_means.csv',                           type='reference',  name='VuJQSRT2021 AM means') 
        _1DAMMeans.readCSV(   filePath='../../../playbookmc/Counterparts/BrantleyJQSRT2011/1DBrantleyJQSRT2011_AM_SuiteI_leakage.csv',    type='comparison', name='1D BrantleyJQSRT2011 AM means')   
        _1DAMMeans.readCSV(   filePath='../../../playbookmc/Counterparts/OlsonMC2023/1D_Binary_CoPSp-0_leakage.csv',                      type='comparison', name='1D Binary CoPSp-0 means')   
        ## 1D CLS Means Agreement
        _1DCLSMeans = CompareOutputDataAnalysis()
        _1DCLSMeans.readCSV(  filePath='../../../playbookmc/Counterparts/LarmierJQSRT2018_CLS/1DLarmierJQSRT2018_CLS_SuiteI_leakage.csv', type='reference',  name='LarmierJQSRT2018 CLS means') 
        _1DCLSMeans.readCSV(  filePath='../../../playbookmc/Counterparts/VuJQSRT2021/VuJQSRT2021_CLS_means.csv',                          type='comparison', name='VuJQSRT2021 CLS means')   
        _1DCLSMeans.readCSV(  filePath='../../../playbookmc/Counterparts/BrantleyJQSRT2011/1DBrantleyJQSRT2011_CLS_SuiteI_leakage.csv',   type='comparison', name='1D BrantleyJQSRT2011 CLS means') 
        _1DCLSMeans.readCSV(  filePath='../../../playbookmc/Counterparts/VuNSE2022/1DVuNSE2022_CoPSp-1_leakage.csv',                      type='comparison', name='VuNSE2022 CoPSp-1 means') 
        # Compute Differences
        _1DBenchMeans.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Refl')
        _1DAMMeans.computeDifferences(   SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Refl')
        _1DCLSMeans.computeDifferences(  SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Refl')
        _1DBenchMeans.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Trans')
        _1DAMMeans.computeDifferences(   SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Trans')
        _1DCLSMeans.computeDifferences(  SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Trans')
        # Plot Outputs and Differences 
        _1DBenchMeans.plotDifferencesAcrossCases(quantity='Refl' , dataIdentifiers=('Sign', 'Rel'), fileName='1D Benchmark Means Agreement - Reflectance'  , flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D Benchmark Means Agreement - Reflectance')
        _1DBenchMeans.plotDifferencesAcrossCases(quantity='Trans', dataIdentifiers=('Sign', 'Rel'), fileName='1D Benchmark Means Agreement - Transmittance', flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D Benchmark Means Agreement - Transmittance')
        _1DAMMeans.plotDifferencesAcrossCases(   quantity='Refl' , dataIdentifiers=('Sign', 'Rel'), fileName='1D AM Means Agreement - Reflectance'  ,        flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D AM Means Agreement - Reflectance')
        _1DAMMeans.plotDifferencesAcrossCases(   quantity='Trans', dataIdentifiers=('Sign', 'Rel'), fileName='1D AM Means Agreement - Transmittance',        flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D AM Means Agreement - Transmittance')
        _1DCLSMeans.plotDifferencesAcrossCases(  quantity='Refl' , dataIdentifiers=('Sign', 'Rel'), fileName='1D CLS Means Agreement - Reflectance'  ,       flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D CLS Means Agreement - Reflectance')
        _1DCLSMeans.plotDifferencesAcrossCases(  quantity='Trans', dataIdentifiers=('Sign', 'Rel'), fileName='1D CLS Means Agreement - Transmittance',       flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D CLS Means Agreement - Transmittance')

    if flMethods: 
        # Methods Comparison
        _1DMethodsMeans = CompareOutputDataAnalysis() 
        _1DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/VuJQSRT2021/VuJQSRT2021_Bench_means.csv',                        type='reference',  name='VuJQSRT2021 Bench means')
        _1DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/VuJQSRT2021/VuJQSRT2021_AM_means.csv',                           type='comparison', name='VuJQSRT2021 AM means') 
        _1DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/LarmierJQSRT2018_CLS/1DLarmierJQSRT2018_CLS_SuiteI_leakage.csv', type='comparison', name='LarmierJQSRT2018 CLS means') 
        _1DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/VuJQSRT2021/VuJQSRT2021_LRP_means.csv',                          type='comparison', name='VuJQSRT2021 LRP means')
        _1DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/VuJQSRT2021/VuJQSRT2021_AlgC_means.csv',                         type='comparison', name='VuJQSRT2021 AlgC means')
        _1DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/VuJQSRT2021/VuJQSRT2021_CoPS2_means.csv',                        type='comparison', name='VuJQSRT2021 CoPS2 means')
        _1DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/VuJQSRT2021/VuJQSRT2021_CoPS3_means.csv',                        type='comparison', name='VuJQSRT2021 CoPS3 means')
        # Compute Differences
        _1DMethodsMeans.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Refl')
        _1DMethodsMeans.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Trans')
        # Plot Outputs and Differences 
        _1DMethodsMeans.plotDifferencesAcrossCases(quantity='Refl' , dataIdentifiers=('Sign', 'Rel'), fileName='1D Methods Means Agreement - Reflectance'  , flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D Methods Means Agreement - Reflectance')
        _1DMethodsMeans.plotDifferencesAcrossCases(quantity='Trans', dataIdentifiers=('Sign', 'Rel'), fileName='1D Methods Means Agreement - Transmittance', flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D Methods Means Agreement - Transmittance')



if "1DALPVariances" in benchmarkSets:
    # Variance Values on 1D ALP/Binary Benchmark Problems
    if flAgreements == True:
        ## 1D Benchmark Variances Agreement
        _1DBenchVar = CompareOutputDataAnalysis()
        _1DBenchVar.readCSV(filePath='../../../playbookmc/Counterparts/AdamsJQSRT1989/AdamsJQSRT1989_Bench_variances.csv', type='reference',  name='AdamsJQSRT1989 Bench variances')
        _1DBenchVar.readCSV(filePath='../../../playbookmc/Counterparts/VuJQSRT2021/VuJQSRT2021_CoPS3PO_variances.csv',     type='comparison', name='VuJQSRT2021 CoPS3PO variances')  
        # Compute Differences
        _1DBenchVar.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Refl')
        _1DBenchVar.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Trans')
        # Plot Outputs and Differences 
        _1DBenchVar.plotDifferencesAcrossCases(quantity='Refl' , dataIdentifiers=('Sign', 'Rel'), fileName='1D Benchmark Variances Agreement - Reflectance'  , flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D Benchmark Variances Agreement - Reflectance')
        _1DBenchVar.plotDifferencesAcrossCases(quantity='Trans', dataIdentifiers=('Sign', 'Rel'), fileName='1D Benchmark Variances Agreement - Transmittance', flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D Benchmark Variances Agreement - Transmittance')
        
    if flMethods: 
        ## 1D Methods Variances Agreement
        _1DMethodsVar = CompareOutputDataAnalysis()
        _1DMethodsVar.readCSV(filePath='../../../playbookmc/Counterparts/AdamsJQSRT1989/AdamsJQSRT1989_Bench_variances.csv', type='reference',  name='AdamsJQSRT1989 Bench variances')
        _1DMethodsVar.readCSV(filePath='../../../playbookmc/Counterparts/VuJQSRT2021/VuJQSRT2021_CoPS2_variances.csv',       type='comparison', name='VuJQSRT2021 CoPS2 variances')
        _1DMethodsVar.readCSV(filePath='../../../playbookmc/Counterparts/VuJQSRT2021/VuJQSRT2021_CoPS3_variances.csv',       type='comparison', name='VuJQSRT2021 CoPS3 variances')
        # Compute Differences
        _1DMethodsVar.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Refl')
        _1DMethodsVar.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Trans')
        # Plot Outputs and Differences 
        _1DMethodsVar.plotDifferencesAcrossCases(quantity='Refl' , dataIdentifiers=('Sign', 'Rel'), fileName='1D Methods Variances Agreement - Reflectance'  , flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D Methods Variances Agreement - Reflectance')
        _1DMethodsVar.plotDifferencesAcrossCases(quantity='Trans', dataIdentifiers=('Sign', 'Rel'), fileName='1D Methods Variances Agreement - Transmittance', flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D Methods Variances Agreement - Transmittance')
        

if "3DALPMeans" in benchmarkSets:
    # Means Values on 3D ALP/Binary Benchmark Problems
    if flAgreements == True:
        ## 3D CLS Means Agreement 
        _3DCLSMeans = CompareOutputDataAnalysis()
        _3DCLSMeans.readCSV(filePath='../../../playbookmc/Counterparts/LarmierJQSRT2018_CLS/3DLarmierJQSRT2018_CLS_SuiteI_leakage.csv', type='reference',  name='3D LarmierJQSRT2018 CLS means')
        _3DCLSMeans.readCSV(filePath='../../../playbookmc/Counterparts/BrantleyANS2017/3DBrantleyANS2017_CLS_leakage.csv',              type='comparison', name='3D BrantleyANS2017 CLS means')
        _3DCLSMeans.readCSV(filePath='../../../playbookmc/Counterparts/OlsonMC2023/3D_Binary_CoPSp-1_leakage.csv',                      type='comparison', name='3D Binary CoPSp-1 means')
        # Compute Differences
        _3DCLSMeans.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Refl')
        _3DCLSMeans.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Trans')
        # Plot Outputs and Differences 
        _3DCLSMeans.plotDifferencesAcrossCases(quantity='Refl' , dataIdentifiers=('Sign', 'Rel'), fileName='3D CLS Means Agreement - Reflectance'  , flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='3D CLS Means Agreement - Reflectance')
        _3DCLSMeans.plotDifferencesAcrossCases(quantity='Trans', dataIdentifiers=('Sign', 'Rel'), fileName='3D CLS Means Agreement - Transmittance', flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='3D CLS Means Agreement - Transmittance')
        
    if flMethods: 
        ## 3D Methods Means Agreement
        _3DMethodsMeans = CompareOutputDataAnalysis()
        _3DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/LarmierJQSRT2018_CLS/3DLarmierJQSRT2018_Bench_SuiteI_leakage.csv',      type='reference',  name='3D LarmierJQSRT2018 Bench means')
        _3DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/BrantleyANS2017/3DBrantleyANS2017_AM_leakage.csv',                      type='comparison', name='3D BrantleyANS2017 AM means')
        _3DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/LarmierJQSRT2018_CLS/3DLarmierJQSRT2018_CLS_SuiteI_leakage.csv',        type='comparison', name='3D LarmierJQSRT2018 CLS means')
        _3DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/BrantleyANS2017/3DBrantleyANS2017_LRP_leakage.csv',                     type='comparison', name='3D BrantleyANS2017 LRP means')
        _3DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/LarmierJQSRT2018_PBS/3DLarmierJQSRT2018_BoxPoisson_SuiteI_leakage.csv', type='comparison', name='3D LarmierJQSRT2018 BoxPoisson means')
        _3DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/LarmierJQSRT2018_PBS/3DLarmierJQSRT2018_PBS-1_SuiteI_leakage.csv',      type='comparison', name='3D LarmierJQSRT2018 PBS-1 means')
        _3DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/LarmierJQSRT2018_PBS/3DLarmierJQSRT2018_PBS-2_SuiteI_leakage.csv',      type='comparison', name='3D LarmierJQSRT2018 PBS-2 means')
        _3DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/OlsonMC2019/OlsonMC2019_CoPS2_leakage.csv',                             type='comparison', name='OlsonMC2019 CoPS2 means')
        _3DMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/OlsonMC2019/OlsonMC2019_CoPS4_leakage.csv',                             type='comparison', name='OlsonMC2019 CoPS4 means')
        # Compute Differences
        _3DMethodsMeans.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Refl')
        _3DMethodsMeans.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Trans')
        # Plot Outputs and Differences 
        _3DMethodsMeans.plotDifferencesAcrossCases(quantity='Refl' , dataIdentifiers=('Sign', 'Rel'), fileName='3D Methods Means Agreement - Reflectance'  , flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='3D Methods Means Agreement - Reflectance')
        _3DMethodsMeans.plotDifferencesAcrossCases(quantity='Trans', dataIdentifiers=('Sign', 'Rel'), fileName='3D Methods Means Agreement - Transmittance', flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='3D Methods Means Agreement - Transmittance')


if "1DVBOKMeans" in benchmarkSets:
    # Means Values on 1D VBOK/Quaternary Benchmark Problems
    if flAgreements == True:
        ## 1D Quaternary Benchmark Means Agreement 
        _1DQuatAMMeans = CompareOutputDataAnalysis()
        _1DQuatAMMeans.readCSV(filePath='../../../playbookmc/Counterparts/OlsonMC2023/1D_Quaternary_AM_leakage.csv',         type='reference',  name='1D Quaternary AM means')
        _1DQuatAMMeans.readCSV(filePath='../../../playbookmc/Counterparts/OlsonMC2023/1D_Quaternary_CoPSp-0_leakage.csv',    type='comparison', name='1D Quaternary CoPSp-0 means')
        ## 1D Quaternary CLS Means Agreement 
        _1DQuatCLSMeans = CompareOutputDataAnalysis()
        _1DQuatCLSMeans.readCSV(filePath='../../../playbookmc/Counterparts/VuMC2021_N-ary/VuMC2021_CLS_VolFrac_leakage.csv', type='reference',  name='VuMC2021 CLS means')
        _1DQuatCLSMeans.readCSV(filePath='../../../playbookmc/Counterparts/OlsonMC2023/1D_Quaternary_CoPSp-1_leakage.csv',   type='comparison', name='1D Quaternary CoPSp-1 means')
        # Compute Differences
        _1DQuatAMMeans.computeDifferences( SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Refl')
        _1DQuatCLSMeans.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Refl')
        _1DQuatAMMeans.computeDifferences( SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Trans')
        _1DQuatCLSMeans.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Trans')
        # Plot Outputs and Differences 
        _1DQuatAMMeans.plotDifferencesAcrossCases( quantity='Refl' , dataIdentifiers=('Sign', 'Rel'), fileName='1D Quaternary AM Means Agreement - Reflectance'  ,  flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D Quaternary AM Means Agreement - Reflectance')
        _1DQuatAMMeans.plotDifferencesAcrossCases( quantity='Trans', dataIdentifiers=('Sign', 'Rel'), fileName='1D Quaternary AM Means Agreement - Transmittance',  flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D Quaternary AM Means Agreement - Transmittance')
        _1DQuatCLSMeans.plotDifferencesAcrossCases(quantity='Refl' , dataIdentifiers=('Sign', 'Rel'), fileName='1D Quaternary CLS Means Agreement - Reflectance'  , flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D Quaternary CLS Means Agreement - Reflectance')
        _1DQuatCLSMeans.plotDifferencesAcrossCases(quantity='Trans', dataIdentifiers=('Sign', 'Rel'), fileName='1D Quaternary CLS Means Agreement - Transmittance', flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D Quaternary CLS Means Agreement - Transmittance')
        
    if flMethods: 
        ## 1D Quaternary Methods Means Agreement
        _1DQuatMethodsMeans = CompareOutputDataAnalysis()
        _1DQuatMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/VuMC2021_N-ary/VuMC2021_Bench_VolFrac_leakage.csv', type='reference',  name='VuMC2021 Bench VolFrac means')
        _1DQuatMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/OlsonMC2023/1D_Quaternary_AM_leakage.csv',          type='comparison', name='1D Quaternary AM means')
        _1DQuatMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/VuMC2021_N-ary/VuMC2021_CLS_VolFrac_leakage.csv',   type='comparison', name='VuMC2021 CLS means')
        _1DQuatMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/VuMC2021_N-ary/VuMC2021_LRP_VolFrac_leakage.csv',   type='comparison', name='VuMC2021 LRP means')
        # Compute Differences
        _1DQuatMethodsMeans.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Refl')
        _1DQuatMethodsMeans.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Trans')
        # Plot Outputs and Differences 
        _1DQuatMethodsMeans.plotDifferencesAcrossCases(quantity='Refl' , dataIdentifiers=('Sign', 'Rel'), fileName='1D Quaternary Methods Means Agreement - Reflectance'  , flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D Quaternary Methods Means Agreement - Reflectance')
        _1DQuatMethodsMeans.plotDifferencesAcrossCases(quantity='Trans', dataIdentifiers=('Sign', 'Rel'), fileName='1D Quaternary Methods Means Agreement - Transmittance', flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='1D Quaternary Methods Means Agreement - Transmittance')


if "1DVBOKMeans" in benchmarkSets:
    # Means Values on Spherical Inclusions Benchmark Problems
    if flAgreements == True:
        print("There is no Agreements data for Spherical Inluclusions, only Methods.")

    if flMethods: 
        ## Spherical Inclusions Methods Means Agreement
        SphereMethodsMeans = CompareOutputDataAnalysis()
        SphereMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/BrantleyMC2011/BrantleyMC2011_Spheres_Constant.csv',      type='reference',  name='BrantleyMC2011 Spheres Constant')
        SphereMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/BrantleyANS2014/BrantleyANS2014_Spheres_Exponential.csv', type='comparison', name='BrantleyANS2014 Spheres Exponential')
        SphereMethodsMeans.readCSV(filePath='../../../playbookmc/Counterparts/BrantleyMC2011/BrantleyMC2011_Spheres_Uniform.csv',       type='comparison', name='BrantleyMC2011 Spheres Uniform')
        # Compute Differences
        SphereMethodsMeans.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Refl')
        SphereMethodsMeans.computeDifferences(SignedOrUnsignedDiffs='signed', AbsoluteOrRelativeDiffs='relative', quantity='Trans')
        # Plot Outputs and Differences 
        SphereMethodsMeans.plotDifferencesAcrossCases(quantity='Refl' , dataIdentifiers=('Sign', 'Rel'), fileName='Spherical Inclusions Means Agreement - Reflectance'  ,  flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='Spherical Inclusions Methods Means Agreement - Reflectance')
        SphereMethodsMeans.plotDifferencesAcrossCases(quantity='Trans', dataIdentifiers=('Sign', 'Rel'), fileName='Spherical Inclusions Means Agreement - Transmittance',  flShowPlot=True, flSavePlot=True, yLim=None, plotTitle='Spherical Inclusions Methods Means Agreement - Transmittance')