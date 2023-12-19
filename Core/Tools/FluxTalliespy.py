#!usr/bin/env python
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from ClassToolspy import ClassTools

## \brief Takes 1D cell-based flux tallies.
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
# Class for collecting 1D tallies with analog MC scheme.
# Can handle track-length tallies and point-based track-length tallies.
# Plans include adding the ability to take collision tallies and work
# with non-one particle weights.
class FluxTallies(ClassTools):
    def __init__(self):
        super(FluxTallies,self).__init__()
        self._tallyFlux          = self._tallyNothing
        self.flMaterialDependent = False
        self.flUseSeed           = False

    def __str__(self):
        return str(self.__dict__)

    ## \brief Choose FluxTallies option(s)
    #
    # Currently only option here is the number of moments to tally.
    # If choose to take flux tallies, tallying at least two moments is required
    # so that uncertanties can be provided (and all code will work).
    #
    # \param[in] numMomsToTally int, number of flux moments to tally
    # \returns sets self.numMomsToTally int, number of flux moments to tally
    def setFluxTallyOptions(self,numMomsToTally=2):
        assert isinstance(numMomsToTally,int) and numMomsToTally>=2
        self.numMomsToTally = numMomsToTally

    ## \brief Sets a link to instantiation of OneDSlab
    #
    # Links geometry to object and chooses either public or semi-private version of 'samplePoint'.
    # Could be generalized to accept other functions, but is not at this time.
    # The geometry needs to be the same defined for the MC transport object an
    # instantiation of this class is associated to.  Perhaps in the future I could
    # have the MC transport class call this instead of the user.
    #
    # \param[in] slab object, instantiation of OneDSlabGeometry
    # \param[in] slablength float, length of the slab
    # \param[in] xmin float, default 0.0, left boundary of geometry, default for 1D PlaybookMC
    def defineMCGeometry(self,slab,slablength=None,xmin=0.0):
        self.Slab = slab
        assert callable(self.Slab.samplePoint_x) and callable(self.Slab.samplePoint_iseg)
        #self.XSFunction   = self.Slab.samplePoint_x
        assert isinstance(slablength,float) and slablength>0.0 and slablength<np.inf
        self.SlabLength   = slablength
        assert isinstance(xmin,float)
        self.xmin         = xmin
        self.xmax         = xmin + self.SlabLength
        self.GeomType     = 'MC'

    ## \brief Define a parameters needed for Woodcock Monte Carlo (WMC) transport.
    #
    # Defines a WMC total cross section ceiling, cross section function, and the arguments
    # needed to call the function.  The ceiling must be as large as or larger than the largest
    # total cross section which can be called using the function.  The function must be able to
    # return values for cross section types 'total', 'scatter', 'absorb', and 'scatrat', and
    # these values must be consistant with their meanings (at least at x=0.0).  The parameters
    # passed must pass some tests including calling the function for each cross section type at x=0.0.
    # The geometry needs to be the same defined for the MC transport object an
    # instantiation of this class is associated to.  Perhaps in the future I could
    # have the MC transport class call this instead of the user.
    #
    # \param[in] ceiling float, ceiling value for WMC transport
    # \param[in] xsfunction CallableObject, function that returns cross section values at locations 'x'
    # \param[in] xmin float, default 0.0, left boundary of geometry, default for 1D PlaybookMC
    # \param     args arguments needed in xsfunction, e.g., (RealizationNumber,NumOfEigModes)
    def defineWMCGeometry(self,xsceiling=None,xsfunction=None,slablength=None,*args):
        #Make assertions on input, then make ceiling, function, and args pass some basic tests.
        assert isinstance(xsceiling,float) and xsceiling>0.0
        assert callable(xsfunction)
        assert isinstance(slablength,float) and slablength>0.0
        assert isinstance(xmin,float)
        try:    totalxs  = xsfunction(0.0,'total',args)
        except: print('WMCGeometry function fail to call total cross section value'); raise
        try:    scatterxs = xsfunction(0.0,'scatter',args)
        except: print('WMCGeometry function fail to call scattering cross section value'); raise
        try:    absorbxs  = xsfunction(0.0,'absorb',args)
        except: print('WMCGeometry function fail to call absorption cross section value'); raise
        try:    scatrat   = xsfunction(0.0,'scatrat',args)
        except: print('WMCGeometry function fail to call scattering ratio value'); raise
        assert totalxs<=xsceiling
        assert scatterxs<=totalxs
        assert absorbxs<=totalxs and absorbxs+scatterxs==totalxs
        if not totalxs==0.0: assert self.isclose( scatterxs/totalxs, scatrat )
        #Store function and parameters as attributes
        self.XSCeiling    = xsceiling
        self.XSFunction   = xsfunction
        self.SlabLength   = slablength
        self.xmin         = xmin
        self.xmax         = xmin + self.SlabLength
        self.args         = args
        self.GeomType     = 'WMC'

    ## \brief Sets slab length and geometry type for Special MC solvers
    #
    # Intended for use when a 1DSlab object is not the geometry but instead
    # the geometry is determined on the fly.  Currently in mind are the
    # CLS, LRP, Alg. C, CoPS, and DRS solvers.
    #
    # \param[in] slablength float, length of the slab
    # \param[in] xmin float, default 0.0, left boundary of geometry, default for 1D PlaybookMC
    # \returns sets self.SlablLength and self.GeomType
    def defineSMCGeometry(self,slablength=None,xmin=0.0):
        assert isinstance(slablength,float) and slablength>0.0
        self.SlabLength   = slablength
        assert isinstance(xmin,float)
        self.xmin         = xmin
        self.xmax         = xmin + self.SlabLength
        self.GeomType     = 'SMC'


    ## \brief Initializes internal random number object.
    #
    # Random numbers for sampling of tally locations here use an instance of
    # numpy.random.RandomState.  The user can let numpy choose where in the pseudo-random
    # number chain to select numbers, or can specify.  If the user specifies (flUseSeed==True),
    # the sequence is re-seeded at the beginning of each particle history based on the initial
    # seed, the particle number, and the stride (i.e., localseed = seed + particlenum * stride)
    # This makes calcluations repeatable even when done in parts or in parallel.
    #
    # \param[in] flUseSeed bool, use specified seed and thus be repeatable (or let numpy choose seed
    #                           and be different each time?)
    # \param seed int, random number seed for random number sequence
    # \param stride int, perturbation of seed for each particle history
    def initializeRandomNumberObject(self,flUseSeed=None,seed=None,stride=None):
        assert isinstance(flUseSeed,bool)
        if seed==None  : seed   = 292837
        if stride==None: stride = 533
        assert isinstance(seed,int) and isinstance(stride,int)
        self.flUseSeed = flUseSeed
        self.Seed      = seed
        self.Stride    = stride
        self.Rng       = np.random.RandomState()

    ## \brief If chosen, initializes random number seed before each history. Intent: private.
    #
    # Is called at beginning of each particle history.  If user chose to use seeds,
    # the history-specific seed is set.
    #
    # \param[in] ipart int, particle history number
    # \returns sets Rng attribute seed for particular particle history
    def _setRandomNumberSeed(self,ipart):
        if self.flUseSeed: self.Rng.seed( self.Seed + self.Stride * ipart )


    ## \brief Instantiates and initializes flux tally attributes.
    #
    # Instantiates and initializes parameters and numpy arrays in which to
    # take flux tallies
    #
    # \param[in] numTallyBins int, number of bins to tally flux in
    # \param[in] flMaterialDependent bool, use material-dependent tallies?
    # \param[in] numMaterials int, number of materials tallies are taken for
    # \returns sets attributes for taking flux tallies
    def setupFluxTallies(self,numTallyBins=10,flMaterialDependent=True,numMaterials=None):
        #check input and store attributes
        assert isinstance(numTallyBins,int) and numTallyBins>=1
        self.numTallyBins = numTallyBins
        self.TallyBinSize = self.SlabLength/self.numTallyBins
        assert isinstance(flMaterialDependent,bool)
        self.flMaterialDependent = flMaterialDependent
        if flMaterialDependent: assert isinstance(numMaterials,int) and numMaterials>=1
        self.numMaterials = numMaterials
        self.FluxXVals              = np.linspace( self.xmin + self.SlabLength/(2.0*float(self.numTallyBins)), self.xmax - self.SlabLength/(2.0*float(self.numTallyBins)), self.numTallyBins )
        self.FluxMatBounds          = np.linspace( self.xmin, self.xmax, self.numTallyBins+1 )
        #initialize material-independent, bin flux tallies
        self.HistoryFluxMeanTallies = np.zeros(self.numTallyBins)
        self.FluxMomTallies         = np.zeros((self.numMomsToTally,self.numTallyBins))
        if flMaterialDependent:
            #initialize material-dependent, bin flux tallies
            self.HistoryFluxMaterialMeanTallies = np.zeros((self.numMaterials,self.numTallyBins))
            self.FluxMaterialMomTallies         = np.zeros((self.numMaterials,self.numMomsToTally,self.numTallyBins))

    ## \brief Defines the method to use for flux tallies.
    #
    # If flux tallies are selected, asserts that the function passed is callable and one of the
    # approved methods for tallying flux.
    # Assigns tally method to 'self._tallyFlux'.
    # If using point-based, track-length tallies, stores the first 'args' entry as the number of
    # locations at which to tally the flux per track-length.
    #
    # \param[in] tallymethod callable, function to use to tally flux
    # \param[in] *args tuple, any additional arguments required by tally method
    # \returns sets 'self._tallyFlux' and if PBTL tallies selected, number of points to tally at
    def defineFluxTallyMethod(self,tallymethod,*args):
        assert callable(tallymethod)
        assert isinstance(self.flMaterialDependent,bool) #must be defined with self.setupTallies first
        if self.GeomType == 'WMC'  : assert (tallymethod==self._tallyPointBasedTrackLengthFlux    or tallymethod==self._tallyMaterialBasedPointBasedTrackLengthFlux or tallymethod==self._tallyCollisionFlux or tallymethod==self._tallyMaterialBasedCollisionFlux)
        if self.flMaterialDependent: assert (tallymethod==self._tallyMaterialBasedTrackLengthFlux or tallymethod==self._tallyMaterialBasedPointBasedTrackLengthFlux or tallymethod==self._tallyMaterialBasedCollisionFlux)
        else                       : assert (tallymethod==self._tallyTrackLengthFlux              or tallymethod==self._tallyPointBasedTrackLengthFlux              or tallymethod==self._tallyCollisionFlux)
        self._tallyFlux    = tallymethod
        self.tallyFluxArgs = args
        if self._tallyFlux == self._tallyPointBasedTrackLengthFlux or\
           self._tallyFlux == self._tallyMaterialBasedPointBasedTrackLengthFlux:
           self.numOfPointsToTallyAt = args[0]

    ## \brief Default tally method: tallies nothing.  Intent: private.
    #
    # If no tally method defined by user, this is called, effectively not taking flux tallies.
    # \returns nothing
    def _tallyNothing(self,oldx,x,mu,iseg,collinfo):
        pass

    ## \brief Tallies material-independent flux in traditional track-length way.  Intent: private.
    #
    # Tallies flux for a track length based on entire track length in traditional way, marching
    # through each flux tally bin in which the track-length exists.
    #
    # \param[in] oldx float, location of beginning of track length
    # \param[in] x float, location of end of track length
    # \param[in] mu float, cosine of angle of particle travel
    # \param[in] iseg int, not needed for material-independent flux tallies
    # \param[in] collinfo dummy, not used in routine, value passed to fit form for tally methods.
    # \returns tallies flux contributions in material-independent tally attributes
    def _tallyTrackLengthFlux(self,oldx,x,mu,iseg,collinfo=None):
        #determine bins of tally
        smx = min(oldx,x); smibin = int( (smx-self.xmin) / self.TallyBinSize )
        lgx = max(oldx,x); lgibin = int( (lgx-self.xmin) / self.TallyBinSize ); lgibin = max(smibin,lgibin-1) if self.isclose_and_geq( lgx, self.FluxMatBounds[lgibin] ) else lgibin
        #if in same bin, tally, else tally individually the first and last and loop for all in between
        if smibin==lgibin:
            self.HistoryFluxMeanTallies[smibin] += abs((lgx-smx) / (self.TallyBinSize * mu) )
        else             :
            self.HistoryFluxMeanTallies[smibin] += abs((self.FluxMatBounds[smibin+1] - smx) / (self.TallyBinSize * mu) )
            self.HistoryFluxMeanTallies[lgibin] += abs((lgx  -  self.FluxMatBounds[lgibin]) / (self.TallyBinSize * mu) )
            for ibin in range(smibin+1,lgibin):
                self.HistoryFluxMeanTallies[ibin] += abs(self.TallyBinSize                  / (self.TallyBinSize * mu) )

    ## \brief Tallies material-independent and dependent flux in traditional track-length way.  Intent: private.
    #
    # Tallies flux for a track length based on entire track length in traditional way, marching
    # through each flux tally bin in which the track-length exists.
    #
    # \param[in] oldx float, location of beginning of track length
    # \param[in] x float, location of end of track length
    # \param[in] mu float, cosine of angle of particle travel
    # \param[in] iseg int, used either to get material type or as material type (SMC and ND CoPS)
    # \param[in] collinfo dummy, not used in routine, value passed to fit form for tally methods.
    # \returns tallies flux contributions in material-independent and material-dependent tally attributes
    def _tallyMaterialBasedTrackLengthFlux(self,oldx,x,mu,iseg,collinfo=None):
        #determine bins of tally
        smx = min(oldx,x); smibin = int( (smx-self.xmin) / self.TallyBinSize )
        lgx = max(oldx,x); lgibin = int( (lgx-self.xmin) / self.TallyBinSize ); lgibin = max(smibin,lgibin-1) if self.isclose_and_geq( lgx, self.FluxMatBounds[lgibin] ) else lgibin
        #if in same bin, tally, else tally individually the first and last and loop for all in between
        imat = self._returnMaterialType((lgx-smx)/2.0,iseg)
        if smibin==lgibin:
            self.HistoryFluxMeanTallies[smibin]               += abs((lgx-smx) / (self.TallyBinSize * mu                                 ) )
##            print 'imat,smibin:',imat,smibin
            self.HistoryFluxMaterialMeanTallies[imat,smibin]  += abs((lgx-smx) / (self.TallyBinSize * mu * self.MatFractions[imat,smibin]) )
        else             :
            self.HistoryFluxMeanTallies[smibin]               += abs((self.FluxMatBounds[smibin+1] - smx) / (self.TallyBinSize * mu                                 ) )
            self.HistoryFluxMeanTallies[lgibin]               += abs((lgx  -  self.FluxMatBounds[lgibin]) / (self.TallyBinSize * mu                                 ) )
##            print 'imat,smibin:',imat,smibin
            self.HistoryFluxMaterialMeanTallies[imat,smibin]  += abs((self.FluxMatBounds[smibin+1] - smx) / (self.TallyBinSize * mu * self.MatFractions[imat,smibin]) )
            self.HistoryFluxMaterialMeanTallies[imat,lgibin]  += abs((lgx  -  self.FluxMatBounds[lgibin]) / (self.TallyBinSize * mu * self.MatFractions[imat,lgibin]) )
            for ibin in range(smibin+1,lgibin):
                self.HistoryFluxMeanTallies[ibin]             += abs(self.TallyBinSize                    / (self.TallyBinSize * mu                                 ) )
                self.HistoryFluxMaterialMeanTallies[imat,ibin]+= abs(self.TallyBinSize                    / (self.TallyBinSize * mu * self.MatFractions[imat,ibin]  ) )

    ## \brief Tallies material-independent flux using PBTL tallies.  Intent: private.
    #
    # Tallies flux for a track length using point-based, track-length tallies.
    # 1/N of the flux is distributed in each of N locations randomly sampled from the track length.
    #
    # \param[in] oldx float, location of beginning of track length
    # \param[in] x float, location of end of track length
    # \param[in] mu float, cosine of angle of particle travel
    # \param[in] iseg int, not needed for material-independent flux tallies
    # \param[in] collinfo dummy, not used in routine, value passed to fit form for tally methods.
    # \returns tallies flux contributions in material-independent tally attributes
    def _tallyPointBasedTrackLengthFlux(self,oldx,x,mu,iseg=None,collinfo=None):
        fluxcont = abs((oldx-x)/mu) / ( self.numOfPointsToTallyAt * self.TallyBinSize )
        smx = min(oldx,x)
        lgx = max(oldx,x)
        for i in range(0,self.numOfPointsToTallyAt):
            tallyloc = smx + self.Rng.rand() * ( lgx - smx )
            ibin = int( (tallyloc-self.xmin) / ( self.TallyBinSize ) ); ibin = min( ibin, self.numTallyBins-1 )
            self.HistoryFluxMeanTallies[ibin] += fluxcont

    ## \brief Tallies material-independent and dependent flux using PBTL tallies.  Intent: private.
    #
    # Tallies flux for a track length using point-based, track-length tallies.
    # 1/N of the flux is distributed in each of N locations randomly sampled from the track length.
    # Tallies are taken in material-independent way and material-dependent way using same randomly
    # sampled tally locations.
    #
    # \param[in] oldx float, location of beginning of track length
    # \param[in] x float, location of end of track length
    # \param[in] mu float, cosine of angle of particle travel
    # \param[in] iseg int, used either to get material type or as material type (SMC and ND CoPS)
    # \param[in] collinfo dummy, not used in routine, value passed to fit form for tally methods.
    # \returns tallies flux contributions in material-independent and material-dependent tally attributes
    def _tallyMaterialBasedPointBasedTrackLengthFlux(self,oldx,x,mu,iseg=None,collinfo=None):
        fluxcont = abs((oldx-x)/mu) / ( self.numOfPointsToTallyAt * self.TallyBinSize )
        smx = min(oldx,x)
        lgx = max(oldx,x)
        for i in range(0,self.numOfPointsToTallyAt):
            tallyloc = smx + self.Rng.rand() * ( lgx - smx )
            ibin = int( (tallyloc-self.xmin) / ( self.TallyBinSize ) ); ibin = min( ibin, self.numTallyBins-1 )
            self.HistoryFluxMeanTallies[ibin] += fluxcont
            imat = self._returnMaterialType(tallyloc,iseg=iseg)
            self.HistoryFluxMaterialMeanTallies[imat,ibin] += fluxcont / self.MatFractions[imat,ibin]

    ## \brief Tallies material-independent flux using collision tallies.  Intent: private.
    #
    # \param[in] oldx float, not needed for collision tallies
    # \param[in] x float, location of end of track length
    # \param[in] mu float, not needed for collision tallies
    # \param[in] iseg int, not needed for material-independent flux tallies
    # \param[in] collinfo tuple of bool and float, whether to take collision tally, and cross section used to determine collision
    # \returns tallies flux contributions in material-independent tally attributes
    def _tallyCollisionFlux(self,oldx,x,mu,iseg=None,collinfo=None):
        if not collinfo[0]: pass #no collision tally for streaming operations
        else              :
            fluxcont = 1.0 / ( collinfo[1] * self.TallyBinSize )
            ibin = int( (x-self.xmin) / ( self.TallyBinSize ) ); ibin = min( ibin, self.numTallyBins-1 )
            self.HistoryFluxMeanTallies[ibin] += fluxcont

    ## \brief Tallies material-independent and dependent flux using collision tallies.  Intent: private.
    #
    # \param[in] oldx float, not needed for collision tallies
    # \param[in] x float, location of end of track length
    # \param[in] mu float, not needed for collision tallies
    # \param[in] iseg int, used either to get material type or as material type (SMC and ND CoPS)
    # \param[in] collinfo tuple of bool and float, whether to take collision tally, and cross section used to determine collision
    # \returns tallies flux contributions in material-independent and material-dependent tally attributes
    def _tallyMaterialBasedCollisionFlux(self,oldx,x,mu,iseg=None,collinfo=None):
        if not collinfo[0]: pass #no collision tally for streaming operations
        else              :
            fluxcont = 1.0 / ( collinfo[1] * self.TallyBinSize )
            ibin = int( (x-self.xmin) / ( self.TallyBinSize ) ); ibin = min( ibin, self.numTallyBins-1 )
            self.HistoryFluxMeanTallies[ibin] += fluxcont

            imat = self._returnMaterialType(x,iseg=iseg)
            self.HistoryFluxMaterialMeanTallies[imat,ibin] += fluxcont / self.MatFractions[imat,ibin]

    ## \brief Sets function that will be used to call material type at location or segment.
    #
    # Asserts that passed function is callable and that flag for material dependent flux tallies
    # has been set to True.
    # If assertions pass, sets passed method to be 'self._returnMaterialType'.
    # 'self._returnSlabMaterialType', 'self._returnDKLMaterialType_RedisualBased', and
    # 'self._returnDKLMaterialType_CollapsedLinearInterpBased' are three functions in this
    # class that can be used as the passed function.  Other functions of this form would
    # be appropriate as well.
    #
    # \param[in] mattypemethod, callable, function which accepts 'x' and 'iseg' and returns material type
    #
    # \returns sets self._returnMaterialType, method, for returning material type based on location or mat. segment
    def defineMaterialTypeMethod(self,mattypemethod):
        assert callable(mattypemethod)
        try   : self.flMaterialDependent
        except: print('Material dependent flux tallies not selected: don\'t need to define self._returnMaterialType.'); raise
        self._returnMaterialType = mattypemethod

    ## \brief Returns mat. type at loc. 'x' or seg. 'iseg' in OneDSlab geometry type.  Intent: private.
    #
    # Veneer to call 'returnMaterialType' in 'OneDSlab' object that is an attribute of an object of
    # this class.  More efficient if segment known, but reasonably efficient if based on 'x'.
    #
    # \param[in] x float, location at which material type is desired
    # \param[in] iseg int, slab segment for which material type is desired
    # returns materialindex int, index of material at 'iseg' or 'x'.
    def _returnSlabMaterialType(self,x,iseg):
        return self.Slab.returnMaterialType(x,iseg)

    ## \brief Returns 'iseg' as material type.  Used in SpecialMCDrivers.  Intent: private.
    #
    # SpecialMCDrivers methods have constantly changing or not fully defined material geometries,
    # however, the FluxTallies object still requires a function with arguments of x and iseg
    # to be able to return material type.  This function does that if the material type is passed
    # as 'iseg'.  Since SMC methods aren't using iseg anyway, this is a sufficient substitution.
    #
    # \param[in] x float, dummy variable to maintain same form here
    # \param[in] iseg int, material type which will be returned
    # returns materialindex int, index of material, passed in as 'iseg'
    def _return_iseg_AsMaterialType(self,x,iseg):
        return iseg

    ## \brief Returns mat. type at loc. 'x' in DiscontinuousKL geometry type.  Intent: private.
    #
    # Intended to be a veneer to call a function in 'DiscontinuousKL' which uses the residual at
    # a location 'x' to return material type.  Not functional yet.  'iseg' is not used.
    #
    # \param[in] x float, location at which material type is desired
    # \param[in] iseg dummy, not used, only included to fit common form
    # returns materialindex int, index of material at 'x'.
    def _returnDKLMaterialType_ResidualBased(self,x,iseg=None):
        if self._sampleKLResidual() >= 0.0: return 1
        else                              : return 0

    ## \brief Returns mat. type at loc. 'x' in DiscontinuousKL geometry type.  Intent: private.
    #
    # Intended to be a veneer to call a function in a new class that will hold geometry information
    # based on a collapsed DKL linear interpolation scheme.  Not functional yet.  'iseg' is not used.
    #
    # \param[in] x float, location at which material type is desired
    # \param[in] iseg dummy, not used, only included to fit common form
    # returns materialindex int, index of material at 'x'.
    def _returnDKLMaterialType_CollapsedLinearInterpBased(self,x,iseg):
        pass
        
    ## \brief Estimates material type tractions in each flux bin by sampling
    #
    # Estimates material fractions in each flux tally cell by Monte Carlo sampling for
    # any geometry which returns material type at a location x. Currently the user
    # specified how many samples are taken in each bin.  Perhaps this could be handled
    # more smartly in the future, though this is an approximate method and the best
    # way to solve this values in most cases is probably by directly solving for them.
    #
    # \param numsamples int, def 10000, number of samples to use in each bin
    # \returns sets self.MatFractions, numpy array, fraction of each mat in each flux tally bin
    def estimateMaterialTypeFractions(self,numsamples=10000):
        #Instantiate and initialize material fractions 'array'
        self.MatFractions = np.zeros((self.numMaterials,self.numTallyBins))
        #Cycle through flux cells and use Monte Carlo to tally material fractions
        assert isinstance(numsamples,int) and numsamples>0
        for ifluxseg in range(0,self.numTallyBins):
            smx = self.FluxMatBounds[ifluxseg  ]
            lgx = self.FluxMatBounds[ifluxseg+1]
            for i in range(0,numsamples):
                xloc = smx + self.Rng.rand() * ( lgx - smx )
                imat = self._returnMaterialType(xloc,None)
                self.MatFractions[imat,ifluxseg] += 1.0/numsamples
        self.assertConservationOfMatInFractions()

    ## \brief Sets passed material fractions array as attribute
    #
    # Asserts that it is a numpy array with right dimensionality
    # and has conservation of material fractions in each flux cell
    #
    # \param[in] matfractions 2D numpy array, material fractions in each flux cell
    # \returns sets self.MatFractions, 2D numpy array, material fractions in each flux cell
    def defineMaterialTypeFractions(self,matfractions=None):
        assert isinstance(matfractions,np.ndarray)
        assert len(matfractions[:,0])==self.numMaterials
        assert len(matfractions[0,:])==self.numTallyBins
        self.MatFractions = matfractions
        self.assertConservationOfMatInFractions()

    ## \brief Asserts that material fractions add to 1 in each flux cell
    def assertConservationOfMatInFractions(self):
        for ifluxseg in range(0,self.numTallyBins):
            tot = 0.0
            for imat in range(0,self.numMaterials):
                tot += self.MatFractions[imat,ifluxseg]
            assert self.isclose(tot,1.0)


    ## \brief Initializes flux tally array for a history.  Intent: private.
    #
    # \returns sets history accumulated flux values to 0.0 for new history
    def _initializeHistoryTallies(self):
        self.HistoryFluxMeanTallies.fill(0.0)
        if self.flMaterialDependent: self.HistoryFluxMaterialMeanTallies.fill(0.0)

    ## \brief Fold history moment tallies into tallies from all histories. Intent: private.
    #
    # \returns updates overall computation flux tallies
    def _foldHistoryTallies(self):
        for imom in range(0,self.numMomsToTally):
            self.FluxMomTallies[imom,:] = np.add( self.FluxMomTallies[imom,:], np.power(self.HistoryFluxMeanTallies,imom+1) )
        if self.flMaterialDependent:
            for imom in range(0,self.numMomsToTally):
                self.FluxMaterialMomTallies[:,imom,:] = np.add( self.FluxMaterialMomTallies[:,imom,:], np.power(self.HistoryFluxMaterialMeanTallies,imom+1) )

    ## \brief Computes flux quantities from flux tallies. Intent: private.
    #
    # From flux moment tallies compute flux mean, standard deviation,
    # Monte Carlo uncertainties ("standard error of the mean"), and figures-of-merit.
    #
    # \param[in] tottime float, total runtime (for FOM calcs)
    # \param[in] numparts int, number of particle histories to normalize by--enables
    #                          calculation of quantities midway through particle simulation
    # \returns sets attributes with flux information
    def _computeFluxQuantities(self,tottime=None,numparts=None):
        #compute normalized moments
        self.FluxMoms    = np.divide( self.FluxMomTallies,  numparts )
        #compute material independent mean, standard deviation, uncertainty, and FOM
        self.FluxMeans   = self.FluxMoms[0,:]
        self.FluxDevs    = np.zeros(self.numTallyBins)
        self.FluxUncerts = np.zeros(self.numTallyBins)
        self.FluxFOM     = np.zeros(self.numTallyBins)
        for ibin in range(0,self.numTallyBins):
            self.FluxDevs[ibin]     = np.sqrt( self.FluxMoms[1,ibin] - self.FluxMoms[0,ibin]**2 ) * np.sqrt(float(numparts)/(float(numparts)-1.0))
            self.FluxUncerts[ibin]  = self.FluxDevs[ibin] / np.sqrt( float(numparts) )
            self.FluxFOM[ibin]      = 1.0 / ( ( self.FluxUncerts[ibin] / self.FluxMoms[0,ibin] )**2 * tottime ) if self.FluxMoms[0,ibin]>0.0 else 0.0
        #compute material dependent mean, standard deviation, uncertainty, and FOM
        if self.flMaterialDependent:
            self.FluxMaterialMoms   = np.divide( self.FluxMaterialMomTallies , numparts )
            self.FluxMaterialMeans  = self.FluxMaterialMoms[:,0,:]
            self.FluxMaterialDevs   = np.zeros((self.numMaterials,self.numTallyBins))
            self.FluxMaterialUncerts= np.zeros((self.numMaterials,self.numTallyBins))
            self.FluxMaterialFOM    = np.zeros((self.numMaterials,self.numTallyBins))
            for imat in range(0,self.numMaterials):
                for ibin in range(0,self.numTallyBins):
                    self.FluxMaterialDevs[imat][ibin]    = np.sqrt( self.FluxMaterialMoms[imat,1,ibin] - self.FluxMaterialMoms[imat,0,ibin]**2 ) * np.sqrt(float(numparts)/(float(numparts)-1.0))
                    self.FluxMaterialUncerts[imat][ibin] = self.FluxMaterialDevs[imat][ibin] / np.sqrt( float(numparts) )
                    self.FluxMaterialFOM[imat][ibin]     = 1.0/ ( ( self.FluxMaterialUncerts[imat][ibin] / self.FluxMaterialMoms[imat,0,ibin] )**2 * tottime ) if self.FluxMaterialMoms[imat,0,ibin]>0.0 else 0.0
        


    ## \brief Collects flux values in dictionary.
    #
    # Saves various flux values in a dictionary for use in printing to a .csv file or other
    # data interrogation/manipuluation. Flux cell centers, mean values, and computed SEM
    # (aka, 1-sigma "uncertainties") always collected in dictionary. The optional prefix
    # is front-appended to all key names.  This can be useful if you want to merge this
    # flux dictionary with another with meaningfully different key names. 
    #
    # \param prefix str, prefix for all key names
    # \param flDevs bool, store computed flux standard deviation values in the dictionary?
    # \param flFOMs bool, store computed flux figures-of-merit in the dictionary?
    # \param flMoments bool, store computed flux moments (other than mean and SEM) in the dictionary?
    # \returns collects values in a dictionary
    def putFluxValsInDictionary(self,prefix='',flDevs=False,flFOMs=False,flMoments=False):
        assert isinstance(prefix,str)
        assert isinstance(flDevs,bool) and isinstance(flFOMs,bool) and isinstance(flMoments,bool)

        self.fluxvalues = {}
        self.fluxvalues[prefix+"Flux Cell Centers"] = self.FluxXVals[:]
        self.fluxvalues[prefix+"Flux Means"]        = self.FluxMeans[:]
        self.fluxvalues[prefix+"Flux Uncerts"]      = self.FluxUncerts
        if flDevs: self.fluxvalues[prefix+"Flux Devs"]         = self.FluxDevs
        if flFOMs: self.fluxvalues[prefix+"Flux FOM"]          = self.FluxFOM
        if self.flMaterialDependent:
            for imat in range(0,self.numMaterials):
                self.fluxvalues[prefix+"Flux Means, Mat "+str(imat)]   = self.FluxMaterialMeans[imat][:]
                self.fluxvalues[prefix+"Flux Uncerts, Mat "+str(imat)] = self.FluxMaterialUncerts[imat][:]
                if flDevs: self.fluxvalues[prefix+"Flux Devs, Mat "+str(imat)]    = self.FluxMaterialDevs[imat][:]
                if flFOMs: self.fluxvalues[prefix+"Flux FOM, Mat "+str(imat)]     = self.FluxMaterialFOM[imat][:]

        if flMoments:
            for imom in range(1,len(self.FluxMoms[:,0])):
                self.fluxvalues[prefix+"Flux Moment "+str(imom+1)] = self.FluxMoms[imom,:]
            if self.flMaterialDependent:
                for imat in range(0,self.numMaterials):
                    for imom in range(1,self.numMomsToTally):
                        self.fluxvalues[prefix+"Flux Moment "+str(imom+1)+", Mat "+str(imat)] = self.FluxMaterialMoms[imat,imom,:]

    ## \brief Print flux cell centers and solved flux values to a file.
    #
    # Loads various, computed flux values (moments, uncertainty, FOM, etc.)
    # into a pandas dataframe and prints to .csv format. Flux cell centers, mean values,
    # and computed SEM (aka, 1-sigma "uncertainties") always collected in dictionary.
    #
    # \param filename str, name of file in which to store data
    # \param flDevs bool, print computed flux standard deviation values?
    # \param flFOMs bool, print computed flux figures-of-merit?
    # \param flMoments bool, print computed flux moments (other than mean and SEM)?
    # \returns prints values to a .csv file
    def printFluxVals(self,filename='default_name',flDevs=False,flFOMs=False,flMoments=False):
        assert isinstance(filename,str)
        assert isinstance(flDevs,bool) and isinstance(flFOMs,bool) and isinstance(flMoments,bool)

        self.putFluxValsInDictionary('',flDevs,flFOMs,flMoments)

        self.fluxvalues_df = pd.DataFrame(self.fluxvalues)
        self.fluxvalues_df.to_csv(filename+'.csv')


    ## \brief Read flux cell centers and other flux values from a .csv file.
    #
    # Reads flux data from a .csv file into a pandas dataframe.
    #
    # \param filename str, name of file from which to read data
    # \returns stores data as local attributes
    def readFluxVals(self,filename=''):
        assert isinstance(filename,str)

        self.fluxvalues_df = pd.read_csv(filename+'.csv')

        
    ## \brief Plots flux mean, rel. stdev, MC uncertainty, and FOM for flux tallies.
    #
    # Plots either material-independent flux or material dependent flux (one plot
    # for each material).
    #
    # \param flMaterialDependent bool, default True, plot material-dependent values
    # \returns 'shows' plots
    def plotFlux(self,flMaterialDependent=True):
        assert isinstance(flMaterialDependent,bool)
        plt.figure()
        plt.title('Flux and Relative Standard Deviation of Flux in Domain')
        plt.ylabel(r'$\phi(x)$, $\sigma_\phi(x)/\phi(x)$, or $FOM_\phi(x)$')
        plt.xlabel(r'$x$')
        plt.errorbar(self.fluxvalues["Flux Cell Centers"],self.fluxvalues["Flux Means"],self.fluxvalues["Flux Uncerts"],color='b',fmt=',-',lw=0.5,label=r'$\phi(x)$')
        plt.plot(self.fluxvalues["Flux Cell Centers"],np.divide(self.fluxvalues["Flux Uncerts"],self.fluxvalues["Flux Means"]),color='c',label=r'$\sigma_\phi(x)/\phi(x)$')
        plt.plot(self.fluxvalues["Flux Cell Centers"],self.fluxvalues["Flux FOM"],'g.-',label=r'$FOM_\phi(x)$')
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.show()
        if flMaterialDependent and self.flMaterialDependent:
            for imat in range(0,self.numMaterials):
                plt.figure()
                plt.title('Flux and Relative Standard Deviation of Flux in Material '+str(imat)+' in Domain')
                plt.ylabel(r'$\phi(x)$, $\sigma_\phi(x)/\phi(x)$, or $FOM_\phi(x)$')
                plt.xlabel(r'$x$')
                plt.errorbar(self.fluxvalues["Flux Cell Centers"],self.fluxvalues["Flux Means, Mat "+str(imat)],self.fluxvalues["Flux Uncerts, Mat "+str(imat)],color='b',fmt=',-',lw=0.5,label=r'$\phi(x)$')
                plt.plot(self.fluxvalues["Flux Cell Centers"],np.divide(self.fluxvalues["Flux Uncerts, Mat "+str(imat)],self.fluxvalues["Flux Means, Mat "+str(imat)]),color='c',label=r'$\sigma_\phi(x)/\phi(x)$')
                plt.plot(self.fluxvalues["Flux Cell Centers"],self.fluxvalues["Flux FOM, Mat "+str(imat)],'g.-',label=r'$FOM_\phi(x)$')
                plt.legend(loc = 'best')
                plt.yscale('log')
                binsize = self.FluxXVals[1] - self.FluxXVals[0]
                plt.xlim([ self.FluxXVals[0]-binsize/2.0 , self.FluxXVals[-1]+binsize/2.0 ])
                plt.show()
