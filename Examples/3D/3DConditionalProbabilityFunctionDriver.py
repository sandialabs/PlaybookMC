#!usr/bin/env python
## \file 3DConditionalProbabilityFunctionDriver.py
#  \brief Example driver script for evaluting point-wise Probability functions.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools')
sys.path.append('../../Core/3D')
from MarkovianInputspy import MarkovianInputs
from CPF_MarkovianAnalyticpy import CPF_MarkovianAnalytic
from CPF_MarkovianIndependentContribspy import CPF_MarkovianIndependentContribs

#Define mixing (assuming Markovian mixing model)
Mark = MarkovianInputs()
Mark.solveNaryMarkovianParamsBasedOnProbAndCorrLength(lamc=1.0,prob=[0.1,0.4,0.5]) #Note: Same mixing values as in 3DVoxelGeometryGenerationDriver

#Set up CPF evaluators
CPF_Ana = CPF_MarkovianAnalytic()
CPF_Ana.defineMixingParams((Mark.lam,))
CPF_Ind = CPF_MarkovianIndependentContribs()
CPF_Ind.defineMixingParams((Mark.lam,))

#Set new point at origin and governing points to be material types 0, 1, and 2.
new_pt_loc         =  [ 0, 0, 0]
existing_pts_types = [[      0    ],[      1    ],[      2    ]]

#Solve conditional probabilities with various locations for three three governing points
print("\n|------------------------------------ Easy problem variants --------------------------------------|")
existing_pts_locs  = [[-2.0,  0,  0],[ 1.0,-1.0, 0],[ 1.0, 1.0, 0.0]]
print("-- Points spread far apart")
print("     Probs predicted by Analytic                  model:",CPF_Ana.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
print("     Probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))

existing_pts_locs  = [[-0.1,  0,  0],[ 1.0,-1.0, 0],[ 1.0, 1.0, 0.0]]
print("\n-- One point near")
print("     Probs predicted by Analytic                  model:",CPF_Ana.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
print("     Probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))

existing_pts_locs  = [[-2.0,  0,  0],[ 0.2,0.3, 0],[ 0.2,-0.4, 0.0]]
print("\n-- Two points near, wide spacing")
print("     Probs predicted by Analytic                  model:",CPF_Ana.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
print("     Probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))


print("\n|---------------------------------- Moderate problem variant -------------------------------------|")
existing_pts_locs  = [[-2.0,  0,  0],[ 0.3,0.15, 0],[ 0.4,-0.15, 0.0]]
print("-- Two points near, moderate spacing")
print("     Probs predicted by Analytic                  model:",CPF_Ana.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
print("     Probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))


print("\n|------------------------------------ Hard problem variants -------------------------------------|")
existing_pts_locs  = [[-0.1,  0,  0],[ -0.11,0, 0],[ 1.0, 1.0, 0.0]]
print("-- Two points near, one behind the other")
print("     Probs predicted by Analytic                  model:",CPF_Ana.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
print("     Probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))

existing_pts_locs  = [[-0.1,  0,  0],[ -0.11,0.01, 0],[ -0.11, 0, 0.01]]
print("\n-- All points near each other")
print("     Probs predicted by Analytic                  model:",CPF_Ana.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
print("     Probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))
print()