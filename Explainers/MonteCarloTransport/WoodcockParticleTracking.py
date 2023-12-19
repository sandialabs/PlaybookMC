#!usr/bin/env python
## \file WoodcockParticleTracking.py
#  \brief Example of Woodcock tracking intended to help communicate how it works.
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
# Problem is a three-material, 1D slab geoemtry problem with a beam source.
# Particles absorb or isotropically scatter.
# Leakage tallies and internal flux tallies are taken, the latter at all pseudo-collisions.
# Particle number and cross section values can be changed by the user.
import numpy as np

#Particle Number
numParticles = 10000

#Materials
TotXS     = [2.0,0.3,1.0]  #Total cross sections
ScatRat   = [0.2,0.4,0.9]  #Scattering ratios

#Geometry
MatBounds = [0.0,1.0,2.0,3.0]                                 #for simplicity, also using these material boundaries as flux boundaries
MatTypes  = [   0 , 1 , 2   ]

#tallies
T =  0                                                        #transmittance
R =  0                                                        #reflectance
Fmom1 = [0.0,0.0,0.0]                                         #first moment of flux tallies
Fmom2 = [0.0,0.0,0.0]                                         #second moment of flux tallies
assert MatBounds[1]-MatBounds[0] == MatBounds[2]-MatBounds[1] #since we are using material cells as flux cells and assuming they are the same size, make sure the user of this script doesn't change the relative sizes
assert MatBounds[1]-MatBounds[0] == MatBounds[3]-MatBounds[2]

#solve for majorant cross section
MajXS = max(TotXS)

#define rounding constant
eps = 0.0000000000001

def returnMaterialCell(x,MatBounds):
    if   x > MatBounds[0] and x <= MatBounds[1]: return 0
    elif x > MatBounds[1] and x <= MatBounds[2]: return 1
    elif x > MatBounds[2] and x <= MatBounds[3]: return 2

for ipart in range(0,numParticles):                                    #loop over histories
    #initialize particle
    x  = 0.0; mu = 1.0                                                 #beam, boundary source
    Fhist = [0.0,0.0,0.0]                                              #within-history flux tally
    flbreak = False
    while True:                                                        #loop for life of history
        db = ( MatBounds[-1]-x if mu>=0.0 else x ) / abs( mu )         #distance to boundary
        dpc= -np.log( np.random.random() ) / MajXS                     #distance to pseudo-collision
        #determine what happens at end of streaming operation
        if db < dpc: dx = db ; flPseudoCollide = False
        else       : dx = dpc; flPseudoCollide = True
        #stream particle
        x += dx * mu
        #evaluate pseudo-collision or boundary crossing
        if flPseudoCollide:
            icell = returnMaterialCell(x,MatBounds)
            #either accept or reject pseudo-collision based on ratio of local total cross section to majorant cross section
            if np.random.random() < TotXS[icell] / MajXS:
                #pseudo-collision accepted as real collision, if rejected, particle continues with same directional heading
                if np.random.random() > ScatRat[icell]: flbreak = True                      #absorption
                else                                  : mu = 2.0 * np.random.random() - 1.0 #isotropic scatter
        else              :
            if   x <= eps              : R += 1; flbreak = True
            elif x >= MatBounds[-1]-eps: T += 1; flbreak = True
            else                       : raise Exception("Tried to leak particle that is not close to leakage boundary")
        #can take tallies such as collision flux tallies (taken at each pseudo-collision site) here
        if flPseudoCollide: Fhist[icell] += 1.0                                             #collision flux tallies can be taken at each pseudo-collision.  Flux must then be normalized by the majorant cross section.
        if flbreak        : break                                                           #if particle was killed, exit particle loop
    #normalize history tallies and contribute to overall tallies (not doing leakage this way here since it is a special case and can be done without this step here)
    normalizationFactor = MajXS * (MatBounds[icell+1]-MatBounds[icell])                     #cell volume normalization can be done like this since all flux cells are the same size
    Fhist = np.divide( Fhist, normalizationFactor   )                                       #normalization can be done at different steps in the algorithm.  Volume and cross section normalization here and history number normalization in the next step is one logical choice
    Fmom1 = np.add(    Fmom1, np.divide(           Fhist     , numParticles ) )
    Fmom2 = np.add(    Fmom2, np.divide( np.power( Fhist, 2 ), numParticles ) )

#process tallies to yield uncertainties and print
Tmom1 = float(T)/numParticles; Tmom2 = Tmom1                           #the first and second moments are only the same for some special tallies including transmittance and reflectance
Tunc  = np.sqrt( ( Tmom2 - Tmom1**2 ) / (float(numParticles) - 1) )    #compute the standard error of the mean--the statistical uncertainty--on the tally using the sample standard deviation
print('\nTransmittance:',Tmom1,' +- ',Tunc)

Rmom1 = float(R)/numParticles; Rmom2 = Rmom1
Runc  = np.sqrt( ( Rmom2 - Rmom1**2 ) / (float(numParticles) - 1) )
print('Reflectance  :',Rmom1,' +- ',Runc)

sampleStatsNormalization = ( 1.0 / (float(numParticles) - 1) )
Func = np.sqrt( np.multiply( np.subtract( Fmom2, np.power( Fmom1, 2 ) ), sampleStatsNormalization ) )
print('Flux[0]      :',Fmom1[0],' +- ',Func[0])
print('Flux[1]      :',Fmom1[1],' +- ',Func[1])
print('Flux[2]      :',Fmom1[2],' +- ',Func[2])