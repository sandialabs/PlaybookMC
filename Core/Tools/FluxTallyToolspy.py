#!usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from ClassToolspy import ClassTools

## \brief Tools to be used by the Tallies class in taking flux tallies.
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
# Can handle track-length and collision tallies.
class FluxTallyTools(ClassTools):
    def __init__(self):
        super(FluxTallyTools,self).__init__()

    def __str__(self):
        return str(self.__dict__)


    ## \brief Enables selection of flux tally options
    #
    # \param numFluxBins int, number of evenly spaced flux tally bins
    # \param fluxTallyType str, 'TrackLength' or 'Collision' - type of tallies to take
    # \returns sets flux parameters and tally function and prepares flux grid
    def selectFluxTallyOptions(self,numFluxBins=1,fluxTallyType='Collision'):
        assert isinstance(numFluxBins,int) and numFluxBins>0
        assert fluxTallyType in {'TrackLength','Collision'}
        if fluxTallyType=='TrackLength' and self.GeomType in {'WMC','CoPS','Markovian','BoxPoisson','Voronoi'}: raise Exception("Currently, only collision (i.e., not track-length) tallies cannot be used with a Woodcock tracker.")
        ##  Perform additional checks here to make sure matFractions is valid
        self.numFluxBins   = numFluxBins
        self.sizeFluxBins  = self.SlabLength/self.numFluxBins
        if   fluxTallyType=='TrackLength': self._tallyFluxContribution = self._tallyTrackLengthFlux
        elif fluxTallyType=='Collision'  : self._tallyFluxContribution = self._tallyCollisionFlux
        self.centerFluxLocations = np.linspace( self.xmin + self.sizeFluxBins/2, self.xmax - self.sizeFluxBins/2, self.numFluxBins   )
        self.edgeFluxLocations   = np.linspace( self.xmin                      , self.xmax                      , self.numFluxBins+1 )


    ## \brief Initializes flux tallies for new sample (to be called by Talliespy.py)
    #
    # \returns prepares tally values for next sample
    def _initializeSampleFluxTallies(self):
        self.Tals[-1]['SampleMatDepFluxMeans'] = np.zeros((self.nummats,self.numFluxBins))
        self.Tals[-1]['SampleMatDepFluxMom2s'] = np.zeros((self.nummats,self.numFluxBins))
        self.Tals[-1]['SampleMatAbundance']    = np.zeros((self.nummats,self.numFluxBins))
        self.Tals[-1]['SampleFluxMeans']       = np.zeros((             self.numFluxBins))
        self.Tals[-1]['SampleFluxMom2s']       = np.zeros((             self.numFluxBins))
        self.Tals[-1]['SampleFluxMean']        = 0.0
        self.Tals[-1]['SampleFluxMom2']        = 0.0

    ## \brief Initializes flux tallies for new history
    #
    # \returns initializes material-dependent flux tallies for history
    def _initializeHistoryFluxTallies(self):
        self.HistFluxTals = np.zeros((self.nummats,self.numFluxBins))

    ## \brief Contribute material-dependent history tallies into various sample-level flux tallies
    def _contribHistFluxTalsToSampTals(self):
        #material-dependent flux tallies
        self.Tals[-1]['SampleMatDepFluxMeans'] = np.add(self.Tals[-1]['SampleMatDepFluxMeans'],          self.HistFluxTals   )
        self.Tals[-1]['SampleMatDepFluxMom2s'] = np.add(self.Tals[-1]['SampleMatDepFluxMom2s'], np.power(self.HistFluxTals,2))
        
        #material-independent flux tallies
        HistFluxTals = np.sum( np.multiply(self.HistFluxTals,self.Tals[-1]['SampleMatAbundance']), axis=0 )  #[material,ibin]
        self.Tals[-1]['SampleFluxMeans']       = np.add(self.Tals[-1]['SampleFluxMeans']      ,               HistFluxTals   )
        self.Tals[-1]['SampleFluxMom2s']       = np.add(self.Tals[-1]['SampleFluxMom2s']      , np.power(     HistFluxTals,2))

        #whole-domain flux tally
        HistFluxTal = np.sum( np.multiply(HistFluxTals,self.sizeFluxBins) ) / self.SlabLength
        self.Tals[-1]['SampleFluxMean']        += HistFluxTal   
        self.Tals[-1]['SampleFluxMom2']        += HistFluxTal**2

    ## \brief Processes sample flux tally values, e.g., computes mean values and statistical uncertianty on those values
    def _processSampleFluxTallies(self):
        numparticles = self.NumParticlesPerSample if self.NumParticlesPerSample>1 else self.NumOfParticles
        #material-dependent flux tallies
        self.Tals[-1]['SampleMatDepFluxMeans'] = np.divide( self.Tals[-1]['SampleMatDepFluxMeans'],numparticles )
        self.Tals[-1]['SampleMatDepFluxMom2s'] = np.divide( self.Tals[-1]['SampleMatDepFluxMom2s'],numparticles )
        self.Tals[-1]['SampleMatDepFluxSEMs']  = self._computeSEMs(self.Tals[-1]['SampleMatDepFluxMeans'],self.Tals[-1]['SampleMatDepFluxMom2s'],numparticles)

        #material-independent flux tallies
        self.Tals[-1]['SampleFluxMeans']       = np.divide( self.Tals[-1]['SampleFluxMeans']      ,numparticles )
        self.Tals[-1]['SampleFluxMom2s']       = np.divide( self.Tals[-1]['SampleFluxMom2s']      ,numparticles )
        self.Tals[-1]['SampleFluxSEMs']        = self._computeSEMs(self.Tals[-1]['SampleFluxMeans']      ,self.Tals[-1]['SampleFluxMom2s']     ,numparticles)

        #whole-domain flux tally
        self.Tals[-1]['SampleFluxMean']        = np.divide( self.Tals[-1]['SampleFluxMean']       ,numparticles )
        self.Tals[-1]['SampleFluxMom2']        = np.divide( self.Tals[-1]['SampleFluxMom2']       ,numparticles )
        self.Tals[-1]['SampleFluxSEM']         = self._computeSEMs(self.Tals[-1]['SampleFluxMean']       ,self.Tals[-1]['SampleFluxMom2']      ,numparticles)


    ## \brief Process flux tallies from set of samples
    #
    # Processing is different based on whether there is only one history per sample (for which tallies are
    # taken as if they are in the same sample) vs. if there are at least two histories per sample.
    def processSimulationFluxTallies(self):
        numSamples = len(self.Tals)

        if self.NumParticlesPerSample >1:
            self.MatDepFluxMeans = np.zeros((self.nummats,self.numFluxBins))
            self.MatDepFluxMom2s = np.zeros((self.nummats,self.numFluxBins))
            matAbundances = self.returnListFromTalsDictionary('SampleMatAbundance') # When using 'sample' material abundance model, weight material-dependent flux tally contributions in each bin based on the material abundance seen in each sample.  The weights are each 1/numSamples when using 'ensemble' material abundance model.  This the weighting approach here solves for either method (though could be simplified when using the 'ensemble' material abudance model).
            abundWgts = matAbundances / sum(matAbundances)
            for imat in range(0,self.nummats):
                for isamp in range(0,numSamples):
                    self.MatDepFluxMeans[imat,:] = np.add( self.MatDepFluxMeans[imat,:],np.multiply(abundWgts[isamp][imat],         self.Tals[isamp]['SampleMatDepFluxMeans'][imat,:]    ))
                    self.MatDepFluxMom2s[imat,:] = np.add( self.MatDepFluxMom2s[imat,:],np.multiply(abundWgts[isamp][imat],np.power(self.Tals[isamp]['SampleMatDepFluxMeans'][imat,:],2) ))
            self.MatDepFluxSEMs = self._computeSEMs(self.MatDepFluxMeans,self.MatDepFluxMom2s,numSamples)
            
            self.FluxMeans = np.zeros(self.numFluxBins)
            self.FluxMom2s = np.zeros(self.numFluxBins)
            for isamp in range(0,numSamples):
                self.FluxMeans = np.add( self.FluxMeans,         self.Tals[isamp]['SampleFluxMeans']    )
                self.FluxMom2s = np.add( self.FluxMom2s,np.power(self.Tals[isamp]['SampleFluxMeans'],2) )
            self.FluxMeans = np.divide( self.FluxMeans,numSamples )
            self.FluxMom2s = np.divide( self.FluxMom2s,numSamples )
            self.FluxSEMs       = self._computeSEMs(self.FluxMeans      ,self.FluxMom2s      ,numSamples)

            self.FluxMean = 0.0
            self.FluxMom2 = 0.0
            for isamp in range(0,numSamples):
                self.FluxMean += self.Tals[isamp]['SampleFluxMean']
                self.FluxMom2 += self.Tals[isamp]['SampleFluxMean']**2
            self.FluxMean = self.FluxMean/numSamples
            self.FluxMom2 = self.FluxMom2/numSamples
            self.FluxSEM        = self._computeSEMs(self.FluxMean       ,self.FluxMom2       ,numSamples)
        else:
            self._processSampleFluxTallies()
            self.MatDepFluxMeans = self.Tals[-1]['SampleMatDepFluxMeans']
            self.MatDepFluxMom2s = self.Tals[-1]['SampleMatDepFluxMom2s']
            self.MatDepFluxSEMs  = self.Tals[-1]['SampleMatDepFluxSEMs']

            self.FluxMeans       = self.Tals[-1]['SampleFluxMeans']
            self.FluxMom2s       = self.Tals[-1]['SampleFluxMom2s']
            self.FluxSEMs        = self.Tals[-1]['SampleFluxSEMs']

            self.FluxMean        = self.Tals[-1]['SampleFluxMean']
            self.FluxMom2        = self.Tals[-1]['SampleFluxMom2']
            self.FluxSEM         = self.Tals[-1]['SampleFluxSEM']

    ## \brief Plot flux as a function of depth through a 1D or 3D geometry
    #
    # \param[in] flMaterialDependent bool, default True; whether or not to plot material-dependent flux (in addition to the material-independent flux)
    # \param[in] flshow bool, default True; show reflectance plot to user?
    # \param[in] flsave bool, default False; save reflectance plot to file?
    # \param[in] fileprefix str, default 'Prefix'; prefix to file name if saving plot
    # \returns shows plot to user
    def plotFlux(self,flMaterialDependent=True,flshow=True,flsave=False,fileprefix='Prefix'):
        assert isinstance(flMaterialDependent,bool)
        assert isinstance(flshow,bool)
        assert isinstance(flsave,bool)
        if flsave: assert isinstance(fileprefix,str)

        plt.figure()
        plt.title('Flux')
        plt.ylabel(r'$\phi(x)$')
        plt.xlabel(r'$x$')
        plt.errorbar(self.centerFluxLocations,self.FluxMeans,self.FluxSEMs,color='k',fmt=',-',lw=1.0,label=r'Flux')
        if flMaterialDependent:
            for imat in range(0,self.nummats):
                plt.errorbar(self.centerFluxLocations,self.MatDepFluxMeans[imat,:],self.MatDepFluxSEMs[imat,:],fmt=',-',lw=0.5,label=r'Flux, Mat. '+str(imat))
        plt.xlim([ self.xmin, self.xmax ])
        plt.legend(loc = 'best')
        plt.yscale('log')
        plt.grid(which='both',axis='both')

        if flsave: plt.savefig(fileprefix+'_Flux.png')
        if flshow: plt.show()
        plt.close()

    ## \brief Tally a track-length flux contribution to history flux values
    #
    # \param[in] xstart float, location of start of particle track in primary geometry direction
    # \param[in] xend float, location of end of particle track in primary geometry direction
    # \param[in] mu float, cosine of angle of particle track compared to primary geometry direction
    # \param[in] material float, material type experienced by particle on particle track
    # \param[in] flcollide N/A, passed only to fulfill syntax for collision tallies
    # \param[in] streamingXS N/A, passed only to fulfill syntax for collision tallies
    # \returns contributes to material-dependent history flux tallies
    def _tallyTrackLengthFlux(self,xstart,xend,mu,material,flcollide,streamingXS):
        #Orient x values and determine tally bins
        smx = min(xstart,xend); smibin = int( (smx-self.xmin) / self.sizeFluxBins )
        lgx = max(xstart,xend); lgibin = int( (lgx-self.xmin) / self.sizeFluxBins )
        lgibin = max(smibin,lgibin-1) if self.isclose_and_geq( lgx, self.edgeFluxLocations[lgibin] ) else lgibin #if lgx at or just to right of flux boundary, the computed lgibin will be one larger than should be (with the exception that smx is also at or to right of flux cell bin, which could happen with a very short streaming path, in which case lgibin should not be decremented and should be equal to smibin)
        #Make tally contribution(s)
        if smibin==lgibin: #If track within one tally cell
            self.HistFluxTals[material,smibin]   += abs((lgx                -               smx) / (self.sizeFluxBins * mu * self.Tals[-1]['SampleMatAbundance'][material,smibin] ) )
        else             : #If track in more than one cell
            self.HistFluxTals[material,smibin]   += abs((self.edgeFluxLocations[smibin+1] - smx) / (self.sizeFluxBins * mu * self.Tals[-1]['SampleMatAbundance'][material,smibin]) )
            self.HistFluxTals[material,lgibin]   += abs((lgx  -  self.edgeFluxLocations[lgibin]) / (self.sizeFluxBins * mu * self.Tals[-1]['SampleMatAbundance'][material,lgibin]) )
            for ibin in range(smibin+1,lgibin):
                self.HistFluxTals[material,ibin] += abs(             self.sizeFluxBins           / (self.sizeFluxBins * mu * self.Tals[-1]['SampleMatAbundance'][material,  ibin]) )

    ## \brief Tally a collision flux contribution to history flux values
    #
    # \param[in] xstart N/A, passed only to fulfill syntax for track-length tallies
    # \param[in] xend float, location of end of particle track in primary geometry direction
    # \param[in] mu N/A, passed only to fulfill syntax for track-length tallies
    # \param[in] material float, material type experienced by particle at location xend
    # \param[in] flcollide bool, True or False, did this particle track end in a collision or pseudo-collision (vs. non-collision event)?
    # \param[in] streamingXS float, cross section used in determining location of collision or pseudo-collision (e.g., total or Majorant cross section)
    # \returns contributes to material-dependent history flux tallies
    def _tallyCollisionFlux(self,xstart,xend,mu,material,flcollide,streamingXS):
        if not flcollide: pass #no collision tally for some streaming operations
        else            :
            fluxcont = 1.0 / ( streamingXS * self.sizeFluxBins )
            ibin = int( (xend-self.xmin) / ( self.sizeFluxBins ) ); ibin = min( ibin, self.numFluxBins-1 )
            self.HistFluxTals[material,ibin] += fluxcont / self.Tals[-1]['SampleMatAbundance'][material,ibin]