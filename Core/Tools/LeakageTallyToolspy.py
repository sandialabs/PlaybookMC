#!usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from ClassToolspy import ClassTools

## \brief Tools to be used by the Tallies class in taking domain leakage tallies.
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
# Can handle surface tallies.
class LeakageTallyTools(ClassTools):
    def __init__(self):
        super(LeakageTallyTools,self).__init__()
        self.flLeakTallies = False

    def __str__(self):
        return str(self.__dict__)


    ## \brief Enables selection of leakage tally options
    #
    # \param numLeakageBins_y int, number of evenly spaced leakage tally bins on the reflective and transmittive boundaries in the y direction
    # \param numLeakageBins_x int, number of evenly spaced leakage tally bins on the reflective and transmittive boundaries in the z direction
    # \returns sets leakage parameters and prepares leakage grid
    def selectLeakageTallyOptions(self,numLeakageBins_y=1,numLeakageBins_x=1):
        assert isinstance(numLeakageBins_y,int) and numLeakageBins_y>0
        self.numLeakageBins_y  =      numLeakageBins_y
        self.sizeLeakageBins_y = ( self.y3Dmax - self.y3Dmin ) / self.numLeakageBins_y
        self.centerLeakageLocations_y = np.linspace( self.y3Dmin + self.sizeLeakageBins_y/2, self.y3Dmax - self.sizeLeakageBins_y/2, self.numLeakageBins_y   )
        self.edgeLeakageLocations_y   = np.linspace( self.y3Dmin                           , self.y3Dmax                           , self.numLeakageBins_y+1 )

        assert isinstance(numLeakageBins_x,int) and numLeakageBins_x>0
        self.numLeakageBins_x  =      numLeakageBins_x
        self.sizeLeakageBins_x = ( self.x3Dmax - self.x3Dmin ) / self.numLeakageBins_x
        self.centerLeakageLocations_x = np.linspace( self.x3Dmin + self.sizeLeakageBins_x/2, self.x3Dmax - self.sizeLeakageBins_x/2, self.numLeakageBins_x   )
        self.edgeLeakageLocations_x   = np.linspace( self.x3Dmin                           , self.x3Dmax                           , self.numLeakageBins_x+1 )

        self.flLeakTallies = True

    ## \brief Initializes leakage tallies for new sample (to be called by Talliespy.py)
    #
    # \returns prepares reflectance and transmittance tally values for next sample
    def _initializeSampleLeakageTallies(self):
        self.Tals[-1]['SampleReflectanceMeans']   = np.zeros((self.numLeakageBins_y,self.numLeakageBins_x))
        self.Tals[-1]['SampleReflectanceMom2s']   = np.zeros((self.numLeakageBins_y,self.numLeakageBins_x))
        self.Tals[-1]['SampleTransmittanceMeans'] = np.zeros((self.numLeakageBins_y,self.numLeakageBins_x))
        self.Tals[-1]['SampleTransmittanceMom2s'] = np.zeros((self.numLeakageBins_y,self.numLeakageBins_x))

    ## \brief Initializes leakage tallies for new history
    #
    # \returns initializes reflectance and transmittance tallies for history
    def _initializeHistoryLeakageTallies(self):
        self.HistReflectanceTals   = np.zeros((self.numLeakageBins_y,self.numLeakageBins_x))
        self.HistTransmittanceTals = np.zeros((self.numLeakageBins_y,self.numLeakageBins_x))

    ## \brief Contribute leakage tallies into sample-level leakage tallies
    def _contribHistLeakageTalsToSampTals(self):
        #gridded leakage tallies
        self.Tals[-1]['SampleReflectanceMeans']   = np.add(self.Tals[-1]['SampleReflectanceMeans']  ,          self.HistReflectanceTals   )
        self.Tals[-1]['SampleReflectanceMom2s']   = np.add(self.Tals[-1]['SampleReflectanceMom2s']  , np.power(self.HistReflectanceTals,2))
        self.Tals[-1]['SampleTransmittanceMeans'] = np.add(self.Tals[-1]['SampleTransmittanceMeans'],          self.HistTransmittanceTals   )
        self.Tals[-1]['SampleTransmittanceMom2s'] = np.add(self.Tals[-1]['SampleTransmittanceMom2s'], np.power(self.HistTransmittanceTals,2))        

    ## \brief Processes sample flux tally values, e.g., computes mean values and statistical uncertianty on those values
    def _processSampleLeakageTallies(self):
        numparticles = self.NumParticlesPerSample if self.NumParticlesPerSample>1 else self.NumOfParticles

        self.Tals[-1]['SampleReflectanceMeans'] = np.divide( self.Tals[-1]['SampleReflectanceMeans'],numparticles )
        self.Tals[-1]['SampleReflectanceMom2s'] = np.divide( self.Tals[-1]['SampleReflectanceMom2s'],numparticles )
        self.Tals[-1]['SampleReflectanceSEMs']  = self._computeSEMs(self.Tals[-1]['SampleReflectanceMeans'],self.Tals[-1]['SampleReflectanceMom2s'],numparticles)

        self.Tals[-1]['SampleTransmittanceMeans'] = np.divide( self.Tals[-1]['SampleTransmittanceMeans'],numparticles )
        self.Tals[-1]['SampleTransmittanceMom2s'] = np.divide( self.Tals[-1]['SampleTransmittanceMom2s'],numparticles )
        self.Tals[-1]['SampleTransmittanceSEMs']  = self._computeSEMs(self.Tals[-1]['SampleTransmittanceMeans'],self.Tals[-1]['SampleTransmittanceMom2s'],numparticles)

    ## \brief Process leakage tallies from set of samples
    #
    # Processing is different based on whether there is only one history per sample (for which tallies are
    # taken as if they are in the same sample) vs. if there are at least two histories per sample.
    def processSimulationLeakageTallies(self):
        numSamples = len(self.Tals)

        if self.NumParticlesPerSample >1:
            self.ReflectanceMeans = np.zeros((self.numLeakageBins_y,self.numLeakageBins_x))
            self.ReflectanceMom2s = np.zeros((self.numLeakageBins_y,self.numLeakageBins_x))
            for isamp in range(0,numSamples):
                self.ReflectanceMeans[:,:] = np.add( self.ReflectanceMeans[:,:],         self.Tals[isamp]['SampleReflectanceMeans'][:,:]    )
                self.ReflectanceMom2s[:,:] = np.add( self.ReflectanceMom2s[:,:],np.power(self.Tals[isamp]['SampleReflectanceMeans'][:,:],2) )
            self.ReflectanceMeans = np.divide( self.ReflectanceMeans,numSamples )
            self.ReflectanceMom2s = np.divide( self.ReflectanceMom2s,numSamples )
            self.ReflectanceSEMs = self._computeSEMs(self.ReflectanceMeans,self.ReflectanceMom2s,numSamples)

            self.TransmittanceMeans = np.zeros((self.numLeakageBins_y,self.numLeakageBins_x))
            self.TransmittanceMom2s = np.zeros((self.numLeakageBins_y,self.numLeakageBins_x))
            for isamp in range(0,numSamples):
                self.TransmittanceMeans[:,:] = np.add( self.TransmittanceMeans[:,:],         self.Tals[isamp]['SampleTransmittanceMeans'][:,:]    )
                self.TransmittanceMom2s[:,:] = np.add( self.TransmittanceMom2s[:,:],np.power(self.Tals[isamp]['SampleTransmittanceMeans'][:,:],2) )
            self.TransmittanceMeans = np.divide( self.TransmittanceMeans,numSamples )
            self.TransmittanceMom2s = np.divide( self.TransmittanceMom2s,numSamples )
            self.TransmittanceSEMs = self._computeSEMs(self.TransmittanceMeans,self.TransmittanceMom2s,numSamples)

        else:
            self._processSampleLeakageTallies()
            self.ReflectanceMeans = self.Tals[-1]['SampleReflectanceMeans']
            self.ReflectanceMom2s = self.Tals[-1]['SampleReflectanceMom2s']
            self.ReflectanceSEMs  = self.Tals[-1]['SampleReflectanceSEMs']

            self.TransmittanceMeans = self.Tals[-1]['SampleTransmittanceMeans']
            self.TransmittanceMom2s = self.Tals[-1]['SampleTransmittanceMom2s']
            self.TransmittanceSEMs  = self.Tals[-1]['SampleTransmittanceSEMs']

    ## \brief Tally a reflectance tally contribution to history leakage values
    #
    # \param[in] y float, y location of reflectance event
    # \param[in] z float, z location of reflectance event
    # \returns contributes to reflectance history tallies
    def _tallyReflectance(self,y,z):
        ibin_y = int( (y-self.y3Dmin) / ( self.sizeLeakageBins_y ) ); ibin_y = min( ibin_y, self.numLeakageBins_y-1 )
        ibin_x = int( (z-self.x3Dmin) / ( self.sizeLeakageBins_x ) ); ibin_x = min( ibin_x, self.numLeakageBins_x-1 )
        self.HistReflectanceTals[ibin_y,ibin_x] += 1.0

    ## \brief Tally a transmittance tally contribution to history leakage values
    #
    # \param[in] y float, y location of transmittance event
    # \param[in] x float, x location of transmittance event
    # \returns contributes to transmittance history tallies
    def _tallyTransmittance(self,y,x):
        ibin_y = int( (y-self.y3Dmin) / ( self.sizeLeakageBins_y ) ); ibin_y = min( ibin_y, self.numLeakageBins_y-1 )
        ibin_x = int( (x-self.x3Dmin) / ( self.sizeLeakageBins_x ) ); ibin_x = min( ibin_x, self.numLeakageBins_x-1 )
        self.HistTransmittanceTals[ibin_y,ibin_x] += 1.0

    ## \brief Plot flux as a function of depth through a 1D or 3D geometry
    #
    # \param[in] flgray bool, default False; plot in grayscale instead of color?
    # \param[in] flshow bool, default True; show reflectance plot to user?
    # \param[in] flsave bool, default False; save reflectance plot to file?
    # \param[in] fileprefix str, default 'Prefix'; prefix to file name if saving plot
    # \returns creates plot and shows to user and/or saves in file based on user inputs
    def plotReflectance(self,flgray=False,flshow=True,flsave=False,fileprefix='Prefix'):
        assert isinstance(flgray,bool)
        assert isinstance(flshow,bool)
        assert isinstance(flsave,bool)
        if flsave: assert isinstance(fileprefix,str)

        if flgray: plt.imshow(self.ReflectanceMeans[:,:], cmap='gray', interpolation='none')
        else     : plt.imshow(self.ReflectanceMeans[:,:], interpolation='none')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar() # add colored legend for data
        plt.title('Reflectance', fontdict = {'fontsize' : 28})

        if flsave: plt.savefig(fileprefix+'_Reflectance.png')
        if flshow: plt.show()
        plt.close()

    ## \brief Plot flux as a function of depth through a 1D or 3D geometry
    #
    # \param[in] flgray bool, default False; plot in grayscale instead of color?
    # \param[in] flshow bool, default True; show transmittance plot to user?
    # \param[in] flsave bool, default False; save transmittance plot to file?
    # \param[in] fileprefix str, default 'Prefix'; prefix to file name if saving plot
    # \returns creates plot and shows to user and/or saves in file based on user inputs
    def plotTransmittance(self,flgray=False,flshow=True,flsave=False,fileprefix='Prefix'):
        assert isinstance(flgray,bool)
        assert isinstance(flshow,bool)
        assert isinstance(flsave,bool)
        if flsave: assert isinstance(fileprefix,str)

        if flgray: plt.imshow(self.TransmittanceMeans[:,:], cmap='gray', interpolation='none')
        else     : plt.imshow(self.TransmittanceMeans[:,:], interpolation='none')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.colorbar() # add colored legend for data
        plt.title('Transmittance', fontdict = {'fontsize' : 28})

        if flsave: plt.savefig(fileprefix+'_Transmittance.png')
        if flshow: plt.show()
        plt.close()
