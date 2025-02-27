#!usr/bin/env python
import sys
from Geometry_Basepy import Geometry_Base
sys.path.append('/../../Classes/Tools')
from ClassToolspy import ClassTools
from MarkovianInputspy import MarkovianInputs
import numpy as np
import matplotlib.pyplot as plt

## \brief Multi-D CLS geometry class
# \author Dominic Lioce, dalioce@sandia.gov, liocedominic@gmail.com
#
class Geometry_CLS(Geometry_Base,ClassTools,MarkovianInputs):
    def __init__(self):
        super(Geometry_CLS,self).__init__()
        self.GeomType = 'CLS'
        self.abundanceModel = 'ensemble'

    def __str__(self):
        return str(self.__dict__)
        
    ## \brief Some geometries will need to overwrite this method definition, others won't.
    #
    # \returns nothing
    def _initializeHistoryGeometryMemory(self):
        pass

    ## \brief Some geometries will need to overwrite this method definition, others won't.
    #
    # \returns nothing
    def _initializeSampleGeometryMemory(self):
        pass

    ## \brief Defines mixing parameters (chord lengths and probabilities)
    #
    # \param[in] lam, list of floats, list of chord lengths
    # \returns sets chord length, probabilities, correlation length, seed locations
    def defineMixingParams(self,lam=None):
        # Assert material chord lengths and slab length and store in object
        assert isinstance(lam,list) and len(lam)==self.nummats
        for i in range(0,self.nummats): assert isinstance(lam[i],float) and lam[i]>0.0
        self.lam = lam

        # Solve and store material probabilities and correlation length
        self.solveNaryMarkovianParamsBasedOnChordLengths(self.lam)

    ## \brief Choose CLS or LRP and whether to perform true 3D transport or emulate 1D planar behavior
    #
    # Either Chord Length Sampling (CLS) or an accuracy improved variance of CLS known as the Limited Realization
    # Preserving (LRP) method can be selected.
    # Secondly, perform 3D stochastic media transport by setting fl1DEmulation to false or
    # make the algorithm emulate 1D planar results by setting fl1DEmulation to true. 
    #
    # \param[in] CLSAlg, str, allows user to choose between either CLS or LRP algorithms
    # \param[in] fl1DEmulation, bool, True will only allow material to change along z-dimension (emulates 1D results)
    # \returns sets fl1DEmulation and CLSAlg
    def defineCLSGeometryType(self,CLSAlg="CLS",fl1DEmulation=False):
        assert CLSAlg == "CLS" or CLSAlg == "LRP"
        self.CLSAlg = CLSAlg
        assert isinstance(fl1DEmulation,bool)
        self.fl1DEmulation = fl1DEmulation

    ## \brief Samples material index at start of particle transport based on volume fractions
    #
    # \returns updates material index and point attributes
    def samplePoint(self):
        pass     # sample which material we are in initially from volume fractions

    ## \brief Sample the distance to an interface. Intent: private.
    #
    # \returns di, distance to interface after random sampling
    def _sampleInterfaceDistance(self):
        return self.lam[self.CurrentMatInd] * np.log(1/self.Rng.rand())
     
    ## \brief Veneer to change the material. Intent: private.
    #
    # \param[in] CurrentMatInd, material index to choose the opposite of
    # \returns CurrentMatInd, index of new material after change   
    def _changeMat(self):
        local_probs = self.prob[:]
        local_probs[self.CurrentMatInd] = 0.0                          #give probability of zero to current material
        local_probs = np.divide( local_probs, np.sum(local_probs) )    #normalize probabilities
        self.CurrentMatInd = int( self.Rng.choice( p=local_probs ) )    #select material from any but current

    ## \brief Increment dim and dip for LRP. Intent: private.
    #
    # \param[in] dist, sampled distance by which dip and dim are incremented
    # \returns sets dip, distance to interface in the positive direction after increment   
    # \returns sets dim, distance to interface in the negative (minus) direction after increment   
    def _incrementDistances(self,dist):
        if not self.fl1DEmulation:
            self.dip -= dist
            self.dim += dist
        elif self.fl1DEmulation:
            self.dip -= dist * abs(self.Part.mu) #if 1DEmulation, scale to change along z-axis
            self.dim += dist * abs(self.Part.mu)

    ## \brief Store old direction vector for LRP. Intent: private.
    #
    # \returns sets self.Part.oldu, self.Part.oldv, self.Part.oldw
    def _storeOldDirection(self):
        self.Part.oldu = self.Part.u; self.Part.oldv = self.Part.v; self.Part.oldw = self.Part.w

    ## \brief Conditionally switch dim and dip for LRP. Intent: private.
    #
    # For regular LRP, this is done probabilistically based on scatter 
    # angle. For 1DEmulation, it is done based on direction of travel in z.
    #
    # \returns switches dim and dip.
    def _conditionallySwitchDistances(self):
        flSwitch = False
        if self.fl1DEmulation:
            if np.sign(self.Part.w) != np.sign(self.Part.oldw): flSwitch = True
        else                 :
            scattermu = np.dot([self.Part.u,self.Part.v,self.Part.w],[self.Part.oldu,self.Part.oldv,self.Part.oldw])
            if scattermu < 2.0 * self.Rng.rand()-1.0: flSwitch = True
        
        if flSwitch: self.dip,self.dim = self.dim,self.dip



