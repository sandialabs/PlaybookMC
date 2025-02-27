#!usr/bin/env python
## \file 3DConditionalProbabilityFunctionDriver.py
#  \brief Example driver script for evaluting point-wise conditional probability functions.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools')
sys.path.append('../../Core/3D')
from CPF_MarkovianIndependentContribspy import CPF_MarkovianIndependentContribs
import numpy as np

#Define mixing (assuming Markovian mixing model)
lam = [0.7,0.5,0.3] #average material chord lengths

#Set up CPF evaluator
CPF_Ind = CPF_MarkovianIndependentContribs()
CPF_Ind.defineMixingParams((lam,))


#Test problem where Independent Contribs model expected to do pretty well (since points angularly far apart)
new_pt_loc         =  [ 0, 0, 0]
existing_pts_locs  = [[  1,  0,  0],[-1.2,0.1, 0],[ -1, -1,  0]]
existing_pts_types = [[      0    ],[      1    ],[      2    ]]
print("\n**** Original test problem ****")
print("Conditional probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))


#Variation of test problem where Independent Contribs model gives same answer as original version but shouldn't
#since the second point is largely behind the first, an effect this model does not account for
new_pt_loc         =  [ 0, 0, 0]
existing_pts_locs  = [[  1,  0,  0],[1.2,0.1,  0],[ -1, -1,  0]]
existing_pts_types = [[      0    ],[      1    ],[      2    ]]
print("\n**** Stressing problem variant ****")
print("Conditional probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))


#Variation of test problem where Independent Contribs model expected to do pretty well (since points angularly far apart)
new_pt_loc         =  [ 0, 0, 0]
existing_pts_locs  = [[0.1,  0,  0],[-1.2,0.1, 0],[ -1, -1,  0]]
existing_pts_types = [[      0    ],[      1    ],[      2    ]]
print("\n**** Near proximity problem variant ****")
print("Conditional probs predicted by Independent Contributions model:",CPF_Ind.returnConditionalProbabilities(new_pt_loc,existing_pts_locs,existing_pts_types))