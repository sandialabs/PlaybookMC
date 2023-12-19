#!usr/bin/env python
import sys
sys.path.append('/../../Classes/Tools')
from MarkovianInputspy import MarkovianInputs
from ClassToolspy import ClassTools
import numpy as np
import matplotlib.pyplot as plt
#
# Two classes in this file: 'Markov_real', and 'Markov_reals' (collection of the first)
#

## \brief Creates a material slab
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
# Class for creating a material slab consisting of material bounds, material types,
# and material properties as a function of the material type.
class Slab(ClassTools,MarkovianInputs):
    def __init__(self):
        super(Slab,self).__init__()
        self.mattype = []
        self.matbound= []

    def __str__(self):
        return str(self.__dict__)

    ## \brief Initialize random number object with or without seed
    #
    # If 'flUseSeed' is True, realization sampling will be repeatable, as long as the same
    # seed is used.  Different values of 'seed' will yield different repeatable sets of samples.
    # 
    # \param[in] flUseSeed bool, should random numbers use seed (and thus be repeatable)?
    # \param[in] seed int, user-passed value for random seed
    # \returns initializes self.Rng numpy random number object
    def initializeRandomNumbers(self,flUseSeed=None,seed=None):
        assert isinstance(flUseSeed,bool)
        if flUseSeed==True:
            assert isinstance(seed,int)
            if seed%2==0: seed +=1  #conventional wisdom says odd random seeds are best
        self.Rng = np.random.RandomState()        
        if flUseSeed: self.Rng.seed( seed )

    ## \brief External user specifies slab parameters. 
    #
    # User passes minimal material information.  Method asserts that parameters are valid
    # (i.e., cross sections are positive, the right number of material boundaries are specified, etc.),
    # and stores as attributes specifications for the slab.
    #
    # \param[in] totxs list of total cross section values in order of material type index
    # \param[in] matbound list of material boundaries
    # \param[in] mattype list of material types, must be one shorter than length of material bounds and contain a number of material types less than or equal to the number of total cross sections specified
    # \param     scatxs optional, list of scattering cross section values, if specified, absorption cross sections and scattering ratios will also be computed and stored as attributes
    def imprintSlab(self,totxs=None,matbound=None,mattype=None,scatxs=None):
        # Assert valid material properties and store in object
        assert isinstance(totxs,list)
        self.nummats = len(totxs)
        self.totxs = totxs
        for i in range(0,self.nummats): assert isinstance(totxs[i],float) and totxs[i]>=0.0
        
        if not scatxs==None:
            self.scatxs  = scatxs
            self.absxs   = []
            self.scatrat = []
            for i in range(0,self.nummats): assert isinstance(scatxs[i],float) and scatxs[i]>=0.0 and totxs[i]>=scatxs[i]
            for i in range(0,self.nummats):
                self.absxs.append(   self.totxs[i]-self.scatxs[i] )
                self.scatrat.append( self.scatxs[i]/self.totxs[i] )
        #Assert valid material bounds and assignments and store in object
        assert isinstance(matbound,list) and isinstance(mattype,list) and len(matbound)==len(mattype)+1
        for i in range(0,len(mattype)):
            assert isinstance(matbound[i],float) and isinstance(mattype[i],int)
            assert matbound[i]<matbound[i+1]
            assert mattype[i]>=0 and mattype[i]<self.nummats
        assert isinstance(matbound[-1],float)
        self.matbound = matbound
        self.mattype = mattype
        self.s = matbound[-1]


    ## \brief Imprints slab parameters provided by user to object.
    #
    # User passes scattering and absorption average and variation values.  Values for a realization are
    # computed as a function of these values, currently by sampling from a uniform distribution centered
    # at the described average value and with a total width equal to 2 times the variation specified.
    # Slab bounds and material assignments must also be given, and a slab is created.
    #
    # \param[in] avescatxs list of average scattering cross section values
    # \param[in] varscatxs list of variation parameters for scattering cross section values
    # \param[in] aveavsxs list of average absorption cross section values
    # \param[in] varabsxs list of variation parameters for absorption cross section values
    # \param[in] matbound list of material boundaries
    # \param[in] mattype list of material types, must be one shorter than length of material bounds and contain a number of material types less than or equal to the number of total cross sections specified
    def populateRandCoefRealization(self,avescatxs=None,varscatxs=None,aveabsxs=None,varabsxs=None,matbound=None,mattype=None):
        # Assert valid material properties
        assert isinstance(avescatxs,list) and isinstance(varscatxs,list) and isinstance(aveabsxs,list) and isinstance(varabsxs,list)
        self.nummats = len(avescatxs)
        assert len(varscatxs)==self.nummats and len(aveabsxs)==self.nummats and len(varabsxs)==self.nummats 
        for i in range(0,self.nummats):
            assert isinstance(avescatxs[i],float) and isinstance(varscatxs[i],float) and varscatxs[i]>=0.0 and avescatxs[i]>=varscatxs[i] and isinstance(aveabsxs[i],float) and isinstance(varabsxs[i],float) and varabsxs[i]>=0.0 and aveabsxs[i]>=varabsxs[i]
        # Sample material properties for this realization and store locally, assumes uniform distribution currently
        self.scatxs = []
        self.absxs  = []
        self.totxs  = []
        self.scatrat= []
        for i in range(0,self.nummats):
            self.scatxs.append( (self.Rng.uniform()-0.5+avescatxs[i]) * varscatxs[i] )
            self.absxs.append(  (self.Rng.uniform()-0.5+aveabsxs[i])  * varabsxs[i]  )
            self.totxs.append(   self.scatxs[i] + self.absxs[i] )
            self.scatrat.append( self.scatxs[i] / self.totxs[i] )

        #Assert valid material bounds and assignments and store in object
        assert isinstance(matbound,list) and isinstance(mattype,list) and len(matbound)==len(mattype)+1
        for i in range(0,len(mattype)):
            assert isinstance(matbound[i],float) and isinstance(mattype[i],int)
            assert matbound[i]<matbound[i+1]
            assert mattype[i]>=0 and mattype[i]<self.nummats
        assert isinstance(matbound[-1],float)
        self.matbound = matbound
        self.mattype = mattype
        self.s = matbound[-1]


    ## \brief Create realization of stochastic media with Markovian mixing statistics.
    #
    # Creates a realization of Markovian media by chord length sampling for one material
    # at a time.  This is distinguished from the method of generating pseudo-interfaces
    # and then filling in materials afterwards.
    #
    # \param[in] totxs list of total cross section values
    # \param[in] lam list of average material chord lengths
    # \param[in] s float, the length of the slab
    # \param[in] NaryType char or array, 'volume-fraction' 'uniform' or a transition matrix; if special case will become transition matrix
    # \param     scatxs list, optional, list of scattering cross section values, if specified, absorption cross sections and scattering ratios will also be computed and stored as attributes
    def populateMarkovRealization(self,totxs=None,lam=None,s=None,scatxs=None,NaryType='volume-fraction'):
        # Assert valid material properties and store in object, user may or may not specify scattering cross section values
        assert isinstance(totxs,list)
        self.nummats = len(totxs)
        for i in range(0,self.nummats): assert isinstance(totxs[i],float) and totxs[i]>=0.0
        self.totxs   = totxs        
        if not scatxs==None:
            assert isinstance(scatxs,list) and len(scatxs)==self.nummats
            for i in range(0,self.nummats): assert isinstance(scatxs[i],float) and scatxs[i]>=0.0 and totxs[i]>=scatxs[i]
            self.scatxs  = scatxs
            self.absxs   = []
            self.scatrat = []
            for i in range(0,self.nummats):
                self.absxs.append(   self.totxs[i]-self.scatxs[i] )
                self.scatrat.append( self.scatxs[i]/self.totxs[i] )

        # Assert material chord lengths and slab length and store in object
        assert isinstance(lam,list) and len(lam)==self.nummats
        assert isinstance(s ,float) and s>0.0
        for i in range(0,self.nummats): assert isinstance(lam[i],float) and lam[i]>0.0
        self.lam = lam
        self.s   = s

        # Assert valid NaryType input, if special case, populate array
        if NaryType=='volume-fraction':
            self.solveNaryMarkovianParamsBasedOnChordLengths(self.lam)
            NaryType = []
            for i in range(0,self.nummats):
                NaryType.append([])
            for i in range(0,self.nummats):
                for j in range(0,self.nummats):
                    if i==j: NaryType[i].append( 0.0 )
                    else   : NaryType[i].append( self.prob[j]/(1.0 - self.prob[i]) )
        if NaryType=='uniform':
            NaryType = []
            for i in range(0,self.nummats):
                NaryType.append([])
            for i in range(0,self.nummats):
                for j in range(0,self.nummats):
                    if i==j: NaryType[i].append( 0.0 )
                    else   : NaryType[i].append( 1.0 / (self.nummats - 1.0) )
        assert isinstance(NaryType,list)
        for i in range(0,len(NaryType)):
            assert isinstance(NaryType[i],list)
            assert len(NaryType[i])==len(NaryType)
            assert self.isclose( np.sum(NaryType[i]), 1.0 )
            for j in range(0,len(NaryType[i])):
                assert isinstance(NaryType[i][j],float) and self.isleq(0.0,NaryType[i][j]) and self.isleq(NaryType[i][j],1.0)
        self.NaryType = NaryType
        # solve for chord number frequency and material abundance
        A=np.append(np.transpose(self.NaryType)-np.identity(self.nummats),[[1]*self.nummats],axis=0)
        b0 = [0]*self.nummats; b0.append(1)
        b=np.transpose(np.array(b0))
        self.chordfreq = list( np.linalg.solve(np.transpose(A).dot(A), np.transpose(A).dot(b)) )
        #########print('self.chordfreq:',self.chordfreq)
        self.prob = []
        for i in range(0,self.nummats):
            self.prob.append( self.chordfreq[i]*self.lam[i] / np.dot(self.chordfreq,self.lam) )

        # Generate a realization of Markovian-mixed media
        self.mattype  = []
        self.matbound = [0.0]

        #attach first segment
        curmat = int( self.Rng.choice( self.nummats, p=self.prob ) )
        self.mattype.append(curmat)
        self.matbound.append(self.matbound[-1] + self.lam[curmat]*np.log(1.0/self.Rng.uniform()))

        #either truncate end method or attach subsequent segments
        if self.matbound[-1]>=self.s: self.matbound[-1] = self.s
        else:
            while True: #sample next material either by abundance or with equal probability
                local_probs = self.NaryType[curmat]
#                local_probs[curmat] = 0.0                                      #give probability of zero to current material
#                local_probs = np.divide( local_probs, np.sum(local_probs) )    #normalize probabilities
                curmat = int( self.Rng.choice( self.nummats, p=local_probs ) ) #select material from any but current

                self.mattype.append(curmat)
                self.matbound.append( self.matbound[-1] + self.lam[curmat]*np.log(1.0/self.Rng.uniform()))
                #last segment, truncate (if needed) and end loop            
                if self.matbound[-1]>=self.s: self.matbound[-1] = self.s; break


    ## \brief Create realization of stochastic media with Markovian mixing statistics.
    #
    # Creates a realization of Markovian media by generating pseudo-interfaces
    # and filling in materials afterwards.
    #
    # \param[in] totxs list of total cross section values
    # \param[in] lam list of average material chord lengths
    # \param[in] s float, the length of the slab
    # \param     scatxs list, optional, list of scattering cross section values, if specified, absorption cross sections and scattering ratios will also be computed and stored as attributes
    def populateMarkovRealizationPseudo(self,totxs=None,lam=None,s=None,scatxs=None):
        assert isinstance(totxs,list)
        self.nummats = len(totxs)
        for i in range(0,self.nummats): assert isinstance(totxs[i],float) and totxs[i]>=0.0
        self.totxs   = totxs
        if not scatxs==None:
            assert isinstance(scatxs,list) and len(scatxs)==self.nummats
            for i in range(0,self.nummats): assert isinstance(scatxs[i],float) and scatxs[i]>=0.0 and totxs[i]>=scatxs[i]
            self.scatxs  = scatxs
            self.absxs   = []
            self.scatrat = []
            for i in range(0,self.nummats):
                self.absxs.append(   self.totxs[i]-self.scatxs[i] )
                self.scatrat.append( self.scatxs[i]/self.totxs[i] )

        # Assert material chord lengths and slab length and store in object
        assert isinstance(lam,list) and len(lam)==self.nummats
        assert isinstance(s, float) and s>0.0
        for i in range(0,self.nummats): assert isinstance(lam[i],float) and lam[i]>0.0
        self.lam = lam
        self.s   = s

        # Solve and store material probabilities and correlation length
        self.solveNaryMarkovianParamsBasedOnChordLengths(self.lam)

        # Solve and store average chord length, lamave = \sum_{i=1}^N lam_i*p_i
        self.lamave = np.sum( np.multiply( self.lam, self.prob ) )

        # Pseudo-interfaces within domain
        numpseudoints = int( self.Rng.poisson(self.s/self.lamc) )

        # Generate a realization of Markovian-mixed media
        #sample material boundaries
        orig_matbound= [0.0]
        for i in range(0,numpseudoints):
            orig_matbound.append( self.Rng.uniform( low=0.0 , high=self.s ) )
        orig_matbound.append(self.s)
        orig_matbound = list( np.sort( orig_matbound ) )
        #sample material types
        orig_mattype = []
        for i in range(0,numpseudoints+1):
            orig_mattype.append( int( self.Rng.choice( self.nummats, p=self.prob ) ) )


        # Store realization with the not-needed psuedo-interfaces removed
        self.matbound = [0.0]
        self.mattype = [ orig_mattype[0] ]
        for i in range( 0 , len(orig_mattype)-1 ):
            if not orig_mattype[i]==orig_mattype[i+1]:
                self.matbound.append( orig_matbound[i+1] )
                self.mattype.append( orig_mattype[i+1] )
        self.matbound.append( self.s )
        

    ## \brief Create realization of stochastic media with fixed cell lengths.
    #
    # Creates realization for which the user choses the number of evenly sized regions, or
    # cells, the realization is made up of and the method samples which material type
    # is in each material segment based on user-supplied probabilities.
    #
    # \param[in] totxs list of total cross section values
    # \param[in] prob list of probability of each material
    # \param[in] numcells int, number of fixed-size material segments
    # \param[in] s float, the length of the slab
    # \param     scatxs list, optional, list of scattering cross section values, if specified, absorption cross sections and scattering ratios will also be computed and stored as attributes
    def populateFixedCellLengthRealization(self,totxs=None,prob=None,numcells=None,s=None,scatxs=None):
        # Assert valid material properties and store in object, user may or may not specify scattering cross section values
        assert isinstance(totxs,list)
        self.nummats = len(totxs)
        for i in range(0,self.nummats):
            assert isinstance(totxs[i],float) and totxs[i]>=0.0
        self.totxs = totxs
        if not scatxs==None:
            assert isinstance(scatxs,list) and len(scatxs)==self.nummats
            for i in range(0,self.nummats):
                assert isinstance(scatxs[i],float) and scatxs[i]>=0.0 and totxs[i]>=scatxs[i]
            self.scatxs = scatxs
            self.absxs = []
            self.scatrat = []
            for i in range(0,self.nummats):
                self.absxs.append( self.totxs[i]-self.scatxs[i] )
                self.scatrat.append( self.scatxs[i]/self.totxs[i] )

        # Assert material piece lengths and slab length and store in object
        assert isinstance(prob,list) and len(prob)==self.nummats
        for i in range(0,self.nummats): assert isinstance(prob[i],float) and prob[i]>=0.0 and prob[i]<1.0
        assert sum(prob)==1.0
        self.prob = prob
        assert isinstance(s,float) and s>0.0
        self.s = s

        self.matbound = list(np.linspace(0.0,self.s,numcells+1))
        self.mattype = []
        for i in range(0,numcells):
            self.mattype.append( self.Rng.choice( self.nummats , p=prob ) )


    ## \brief Imprint semi-infinite slab description to object.
    #
    # User passes minimal material information.  Method asserts that parameters are valid
    # (i.e., cross sections are positive, the right number of material boundaries are specified, etc.),
    # and stores as attributes specifications for the slab.
    #
    # \param[in] totxs list of total cross section value for semi-inf slab
    # \param scatxs list of scattering cross section value for semi-inf slab
    def imprintSemiInfSlab(self,totxs=None,scatxs=None):
        # Assert valid material properties and store in object
        assert isinstance(totxs,list) and len(totxs)==1 and isinstance(totxs[0],float) and totxs[0]>0.0
        self.nummats = len(totxs)
        self.totxs = totxs
        if not scatxs==None:
            assert isinstance(scatxs,list) and len(scatxs)==1 and isinstance(scatxs[0],float) and scatxs[0]>=0.0 and totxs[0]>=scatxs[0]
            self.scatxs  =  scatxs
            self.absxs   = [ self.totxs[0]  - self.scatxs[0] ]
            self.scatrat = [ self.scatxs[0] / self.totxs[0]  ]
        self.matbound = [0.0,float('Inf')]
        self.mattype = [0]
        self.s = self.matbound[-1]


    ## \brief Solves the optical thickness of the slab and saves as an attribute.
    #
    # No parameters required, but slab must be defined. Method stores optical thickness
    # as attribute of class and returns it.
    #
    # returns OptThick Optical thickness of slab.
    def solveOptThick(self):
        assert not np.isinf(self.s)
        assert len(self.matbound)>0
        OptThick = 0.0
        for i,imat in enumerate(self.mattype):
            OptThick += (self.matbound[i+1]-self.matbound[i]) * self.totxs[imat]
        self.OptThick = OptThick
        return OptThick

    ## \brief Uses slab optical thickness to emulate Monte Carlo transport for uncollided transmittance.
    #
    # Solve analytic uncollided transmittance using optical thickness and 
    # then use that uncollided transmittance as the probability of success
    # for sampling the number of successes from numhists number of trials.
    # Since trials (histories) are independent, have the same probability
    # of success, and either success or don't (outcome of 0 or 1), 
    # these are Bernoulli trials and the number of successes (transmittances)
    # can be sampled with the binomial distribution.
    #    
    # \param[in] numhists, int, number of uncollided histories to sample
    # \param[in] mu, float, default 1.0, cosine of angle of particle travel
    # returns tmean, float, computed transmittance.
    def emulateUncolMCTrans(self,numhists,mu=1.0):
        assert isinstance(numhists,int)
        assert numhists>0
        assert isinstance(mu,float)
        assert self.isleq(0.0,mu) and self.isleq(mu,1.0)
        self.solveOptThick()
        an_trans = np.exp( - self.OptThick / mu )
        return self.Rng.binomial( n=numhists , p=an_trans ) / numhists

    ## \brief Solves material type fractions in user-specified number of bins.
    #
    # Solves material fractions in each flux tally cell exactly for 'OneDSlab' geometry
    # type.  Intended to be called by 'self.solveMaterialTypeFractions'.
    #
    # \returns sets self.MatFractions, numpy arrays, fraction of each mat in each flux tally bin
    def solveMaterialTypeFractions(self,numbins=None):
        assert isinstance(numbins,int) and numbins>0
        self.numBins = numbins
        self.BinBounds = np.linspace( 0.0, self.s, self.numBins+1 )
        binsize = self.s/self.numBins
        #Instantiate and initialize material fractions 'array'
        self.MatFractions = np.zeros((self.nummats,self.numBins))
        #Compute and store volume fractions of each material in each bin
        ifluxseg = 0
        #Cycle through each material segment
        for iseg in range(0,len(self.mattype)):
            imat     = self.returnMaterialType(None,iseg)
            segstart = self.matbound[iseg]
            segend   = self.matbound[iseg+1]
            #For each material segment, tally material type in flux tally cells
            while True:
                smallloc = max(segstart,self.BinBounds[ifluxseg  ])
                largeloc = min(segend  ,self.BinBounds[ifluxseg+1])
                self.MatFractions[imat,ifluxseg] += (largeloc-smallloc)/binsize
                if   segend> self.BinBounds[ifluxseg+1]: ifluxseg += 1
                elif segend==self.BinBounds[ifluxseg+1]: ifluxseg += 1; break
                elif segend< self.BinBounds[ifluxseg+1]:                break


    ## \brief Returns desired quantity of slab segment.
    #
    # Asserts valid input and calls '_samplePoint' to return the desired
    # quantity as a function of slab segment.
    #
    # \param[in] iseg int, segment of slab to return information about
    # \param xstype string, default 'total', type of cross section to return ('total', 'scatter', 'absorb', 'scatrat')
    # returns SelectedXSVal
    def samplePoint_iseg(self,iseg=None,xstype='total'):
        assert isinstance(iseg,int) and iseg>=0 and iseg<=len(self.mattype)
        assert  xstype=='total' or xstype=='scatter' or xstype=='absorb' or xstype=='scatrat' or xstype=='mattype'
        if not (xstype=='total' or xstype=='mattype'): assert hasattr(self,'scatxs')
        return self._samplePoint(x=None,xstype=xstype,currentmu=None,Seg=iseg,flexactseg=True)

    ## \brief Returns material type at either material segment or location.
    #
    # If 'iseg' is an int, return material type in segment.
    # If 'iseg' is not an int (like None) and 'x' is a float, return material type at location
    # 'x' in slab.
    # If neither of these are true, raise an exception with a message.
    #
    # \param[in] x float, location at which material type is desired
    # \param[in] iseg int, slab segment for which material type is desired
    # returns materialindex, int
    def returnMaterialType(self,x,iseg):
        if   isinstance(iseg,int): return self.samplePoint_iseg(iseg=iseg,xstype='mattype')
        elif isinstance(x,float) : return self.samplePoint_x(   x   =x   ,xstype='mattype')
        else                     : print('Invalid input to returnMaterialType in OneDSlabpy'); raise


    ## \brief Returns desired quantity at a point in a slab.
    #
    # Asserts valid input and calls '_samplePoint' to return the desired
    # quantity as a function of 'x'.
    #
    # \param[in] x float, point in slab from left boundary
    # \param xstype string, default 'total', type of cross section to return ('total', 'scatter', 'absorb', 'scatrat')
    # \param currentmu float, direction of particle, if positive heading right if negative heading left
    # \param oldseg int, last known segment of particle location
    # \param flbinarysearch bool, default of True uses faster 'binary' segment search method
    # returns SelectedXSVal
    def samplePoint_x(self,x=None,xstype='total',currentmu=None,oldseg=None,flbinarysearch=True):
        assert isinstance(x,float) and x>=self.matbound[0] and x<=self.matbound[-1]
        assert  xstype=='total' or xstype=='scatter' or xstype=='absorb' or xstype=='scatrat' or xstype=='mattype'
        if not (xstype=='total' or xstype=='mattype'): assert hasattr(self,'scatxs')
        if not currentmu==None and not currentmu==() : assert isinstance(currentmu,float) and isinstance(oldseg,int) and abs(currentmu)<=1.0 and abs(currentmu)>=0.0 and oldseg>=0 and oldseg<=len(self.mattype)
        return self._samplePoint(x,xstype,currentmu,oldseg,flexactseg=False,flbinarysearch=flbinarysearch)

    ## \brief Returns desired quantity at a point or in a segment of a slab. Intent: private.
    #
    # Return the desired quantity either as a function of 'x' or 'Seg'.
    # In the first case, the segment of the slab is not known, but the location 'x'
    # is.  The segment containing 'x' is found and the desired quantity returned.
    # A bound on the segment may be specified to decrease search time.
    # In the second case, the segment is already known, and the desired value is returned.
    #
    # \param[in] x float, point in slab from left boundary
    # \param xstype string, default 'total', type of cross section to return ('total', 'scatter', 'absorb', 'scatrat')
    # \param currentmu float, direction of particle, if positive heading right if negative heading left
    # \param Seg int, last known segment of particle location
    # \param flexactoldseg bool, default of False, if True claims that 'Seg' is current segment
    # \param flbinarysearch bool, default of True uses faster 'binary' segment search method
    # returns SelectedXSVal
    def _samplePoint(self,x=None,xstype='total',currentmu=None,Seg=None,flexactseg=False,flbinarysearch=True):
        if flexactseg:
            iseg = Seg
        else:
            segrightof = 0                   if not isinstance(currentmu,float) or currentmu< 0.0 else Seg
            segleftof  = len(self.mattype)-1 if not isinstance(currentmu,float) or currentmu>=0.0 else Seg
            if not np.isinf(self.s): iseg = self._binarySegmentFinder(x,segrightof,segleftof) if flbinarysearch else self._basicSegmentFinder(x,segrightof,segleftof) #should updating using np.digitize
            elif   np.isinf(self.s): iseg = 0
        if   xstype=='total'  : return self.totxs[self.mattype[iseg]]
        elif xstype=='scatter': return self.scatxs[self.mattype[iseg]]
        elif xstype=='absorb' : return self.absxs[self.mattype[iseg]]
        elif xstype=='scatrat': return self.scatrat[self.mattype[iseg]]
        elif xstype=='mattype': return self.mattype[iseg]

            
    ## \brief Checks if 'x' is in slab segment 'iseg'.
    #
    # \param[in] x float, point in slab from left boundary
    # \param[in] iseg int, segment being investigated
    # \returns bool True if 'x' is in 'iseg'
    def _IsxIniseg(self,x,iseg):
        return self.matbound[iseg]<x and x<=self.matbound[iseg+1]

    ## \brief Uses binary search to find index of cell containing 'x'. Intent: private.
    #
    # This is the faster method of the two SegmentFinders, but is less staightforward and
    # therefore more difficult to understand and debug.
    #
    # \param[in] x float, point in slab from left boundary
    # \param segrightof int, declaring that 'x' is in this segment or further right
    # \param segleftof int, declaring that 'x' is in this segment or further left
    # \returns iseg segment of slab containing 'x'
    def _binarySegmentFinder(self,x,segrightof,segleftof):
        if x==0.0: return 0
        ilow  = segrightof
        ihigh = segleftof
        while True:
            #print('ilow:',ilow,' ihigh:',ihigh)
            iseg = int( (ilow + ihigh) / 2 )
            if self._IsxIniseg(x,iseg): return iseg
            assert not ilow==ihigh #if this happens search has failed due to bounds given or other cause
            ilow  = max(iseg,ilow+1)  if x>self.matbound[iseg+1] else ilow
            ihigh = ihigh if ilow==iseg              else iseg

    ## \brief Uses sequential search to find index of cell containing 'x'. Intent: private.
    #
    # This is the slower method of the two SegmentFinders, but is more staightforward and
    # therefore easier to understand and debug.
    #
    # \param[in] x float, point in slab from left boundary
    # \param segrightof int, declaring that 'x' is in this segment or further right
    # \param segleftof int, declaring that 'x' is in this segment or further left
    # \returns iseg segment of slab containing 'x'
    def _basicSegmentFinder(self,x,segrightof,segleftof):
        if x==0.0: return 0
        if not segleftof==len(self.mattype):
            for iseg in range(segleftof,-1,-1):
                if self._IsxIniseg(x,iseg): return iseg
            assert not True #if this happens search has failed due to bound given or other cause
        for iseg in range(segrightof-1,len(self.mattype)):
            if self._IsxIniseg(x,iseg): return iseg
        assert not True #if this happens search has failed due to bound given or other cause

                
    ## \brief Plots cross-section profile of slab.
    #
    # Intended for visualizing and debugging, not meant for publication-level plotting.
    #
    # \param numpts int, number of points to plot realization at.
    # \param xstype string, default 'total', type of cross section to return ('total', 'scatter', 'absorb', 'scatrat')
    def plotSlab(self,numpts=400,xstype='total'):
        assert not np.isinf(self.s)
        assert isinstance(numpts,int) and numpts>1
        assert xstype=='total' or xstype=='scatter' or xstype=='absorb' or xstype=='scatrat'
        if not xstype=='total': assert hasattr(self,'scatxs')
        plt.figure()
        plt.title('Cross-Section Profile of Slab')
        plt.xlabel(r'$x$')
        plt.ylabel(r'Cross Section Value')
        xvec = np.linspace(0.0,self.s,numpts)
        xsvals = []
        for x in xvec:
            xsvals.append( self.samplePoint_x(x,xstype) )
        plt.plot(xvec,xsvals)
        plt.grid(True)
        xsvalsspread = max(xsvals)-min(xsvals)
        plt.ylim( min(xsvals)-0.1*xsvalsspread,max(xsvals)+0.1*xsvalsspread )
        plt.show()