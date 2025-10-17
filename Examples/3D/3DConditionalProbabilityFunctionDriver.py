#!usr/bin/env python
## \file 3DConditionalProbabilityFunctionDriver.py
#  \brief Example driver script for evaluting point-wise Probability functions.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#  \author Dan Bolintineanu, dsbolin@sandia.gov
import sys
sys.path.append('../../Core/Tools')
from RandomNumberspy import RandomNumbers
sys.path.append('../../Core/3D')
from MarkovianInputspy import MarkovianInputs
from CPF_MarkovianAnalyticpy import CPF_MarkovianAnalytic
from CPF_MarkovianCombinationpy import CPF_MarkovianCombination
from CPF_MultipleIndicatorCoKrigingpy import CPF_MultipleIndicatorCoKriging
from Geometry_Markovianpy import Geometry_Markovian
from VoxelSpatialStatisticspy import StatMetrics

#Select which CPFs to compare
flCPF_Analytic     = True
flCPF_IndContrib   = True
flCPF_MICK_Numeric = True
flCPF_MICK_Analytic= True

#Define mixing (assuming Markovian mixing model)
Mark = MarkovianInputs()
Mark.solveNaryMarkovianParamsBasedOnProbAndCorrLength(lamc=1.0,prob=[0.1,0.4,0.5]) #Note: Same mixing values as in 3DVoxelGeometryGenerationDriver

#Set up CPF evaluators
if flCPF_Analytic: #Analytic CPF
    CPF_Ana = CPF_MarkovianAnalytic()
    CPF_Ana.defineMixingParams(Mark.lam)
if flCPF_IndContrib: #Combination CPF
    CPF_Ind = CPF_MarkovianCombination()
    CPF_Ind.defineMixingParams(Mark.lam)
if flCPF_MICK_Numeric: # Mick CPF with numerical abundances and S2s
    prefix = "CPFDriverExample"

    try: #Try to read abundances and S2s from files
        Stats = StatMetrics()
        Stats.readMaterialAbundancesFromText(f"{prefix}_mat_abunds.txt")
        Stats.readS2FromCSV(prefix)
    except Exception as e: #If abundances or S2s are not available, read in the geometry, calculate stats
        print(f"Failed to read statistics, with exception: {e}")
        print(f"Generating new geometry and computing statistics from it")

        Geomsize  = 10.0
        numVoxels = [100]*3    #If abundances or S2s are not available, generate geometry and calculate stats
        Rng = RandomNumbers(flUseSeed=True,seed=13425,stridelen=None)
        Geom = Geometry_Markovian()
        Geom.defineMixingParams(laminf=Mark.lamc, prob=Mark.prob[:])
        Geom.associateRng(Rng)
        Geom.defineGeometryBoundaries(xbounds=[-Geomsize/2,Geomsize/2],ybounds=[-Geomsize/2,Geomsize/2],zbounds=[-Geomsize/2,Geomsize/2])
        Geom.defineVoxelizationParams(flVoxelize=True,numVoxels=numVoxels)
        Geom._initializeSampleGeometryMemory()
        Geom.writeVoxelMatIndsToH5(filename = prefix+'.h5')    

        Stats = StatMetrics(Geom)    
        #Calculate and plot S2 spatial statistics
        Stats.calculateMaterialAbundances(flVerbose=True)
        Stats.writeMaterialAbundancesToText(f"{prefix}_mat_abunds.txt")
        Stats.calculateS2()
        Stats.writeS2ToCSV(prefix) 


    CPF_MICK_NumericalS2 = CPF_MultipleIndicatorCoKriging()
    CPF_MICK_NumericalS2.defineMixingParams(Stats) #Simply pass it a Stats object. If not already available, will run spline fits.
if flCPF_MICK_Analytic: #Mick CPF with analytical abundances and S2s
    Mark.generateAnalyticalS2CallableArray()
    CPF_MICK_AnalyticalS2 = CPF_MultipleIndicatorCoKriging()
    CPF_MICK_AnalyticalS2.defineMixingParams(Mark.AnalyticalS2CallableArray, Mark.prob) 


#Set new point at origin and governing points to be material types 0, 1, and 2.
new_pt_loc         =  [ 0, 0, 0]
existing_pts_types = [[      0    ],[      1    ],[      2    ]]

#Solve conditional probabilities with various locations for three three governing points
print("\n|------------------------------------ Easy problem variants --------------------------------------|")
existing_pts_locs  = [[-2.0,  0,  0],[ 1.0,-1.0, 0],[ 1.0, 1.0, 0.0]]
print("-- Points spread far apart")
if flCPF_Analytic     : print("     Probs predicted by Analytic                  model:",CPF_Ana.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_IndContrib   : print("     Probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_MICK_Numeric : print("     Probs predicted by MICK w/ numerical S2      model:",CPF_MICK_NumericalS2.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_MICK_Analytic: print("     Probs predicted by MICK w/ analytical S2     model:",CPF_MICK_AnalyticalS2.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))


existing_pts_locs  = [[-0.1,  0,  0],[ 1.0,-1.0, 0],[ 1.0, 1.0, 0.0]]
print("\n-- One point near")
if flCPF_Analytic     : print("     Probs predicted by Analytic                  model:",CPF_Ana.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_IndContrib   : print("     Probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_MICK_Numeric : print("     Probs predicted by MICK w/ numerical S2      model:",CPF_MICK_NumericalS2.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_MICK_Analytic: print("     Probs predicted by MICK w/ analytical S2     model:",CPF_MICK_AnalyticalS2.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))

existing_pts_locs  = [[-2.0,  0,  0],[ 0.2,0.3, 0],[ 0.2,-0.4, 0.0]]
print("\n-- Two points near, wide spacing")
if flCPF_Analytic     : print("     Probs predicted by Analytic                  model:",CPF_Ana.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_IndContrib   : print("     Probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_MICK_Numeric : print("     Probs predicted by MICK w/ numerical S2      model:",CPF_MICK_NumericalS2.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_MICK_Analytic: print("     Probs predicted by MICK w/ analytical S2     model:",CPF_MICK_AnalyticalS2.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))

print("\n|---------------------------------- Moderate problem variant -------------------------------------|")
existing_pts_locs  = [[-2.0,  0,  0],[ 0.3,0.15, 0],[ 0.4,-0.15, 0.0]]
print("-- Two points near, moderate spacing")
if flCPF_Analytic     : print("     Probs predicted by Analytic                  model:",CPF_Ana.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_IndContrib   : print("     Probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_MICK_Numeric : print("     Probs predicted by MICK w/ numerical S2      model:",CPF_MICK_NumericalS2.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_MICK_Analytic: print("     Probs predicted by MICK w/ analytical S2     model:",CPF_MICK_AnalyticalS2.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))


print("\n|------------------------------------ Hard problem variants -------------------------------------|")
existing_pts_locs  = [[-0.1,  0,  0],[ -0.11,0, 0],[ 1.0, 1.0, 0.0]]
print("-- Two points near, one behind the other")
if flCPF_Analytic     : print("     Probs predicted by Analytic                  model:",CPF_Ana.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_IndContrib   : print("     Probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_MICK_Numeric : print("     Probs predicted by MICK w/ numerical S2      model:",CPF_MICK_NumericalS2.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_MICK_Analytic: print("     Probs predicted by MICK w/ analytical S2     model:",CPF_MICK_AnalyticalS2.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))


existing_pts_locs  = [[-0.1,  0,  0],[ -0.11,0.01, 0],[ -0.11, 0, 0.01]]
print("\n-- All points near each other")
if flCPF_Analytic     : print("     Probs predicted by Analytic                  model:",CPF_Ana.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_IndContrib   : print("     Probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_MICK_Numeric : print("     Probs predicted by MICK w/ numerical S2      model:",CPF_MICK_NumericalS2.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
if flCPF_MICK_Analytic: print("     Probs predicted by MICK w/ analytical S2     model:",CPF_MICK_AnalyticalS2.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))

print()