#!usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
## \brief Accesses Markovian media benchmark inputs and solves mixing parameters from others
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
# \author Emily Vu
#
class MarkovianInputs(object):
    def __init__(self):
        pass
    def __str__(self):
        return str(self.__dict__)

    ## \brief Solve for N-ary parameters based on average chord lengths
    #
    # \param[in] lam list of floats, average chord lengths of materials
    # \returns solves for/sets, self.lam, self.lamc, self.prob
    def solveNaryMarkovianParamsBasedOnChordLengths(self,lam):
        assert isinstance(lam,list) and len(lam)>1
        self.nummats = len(lam)
        for i in range(0,self.nummats):
            assert isinstance(lam[i],float) and lam[i]>0.0
        self.lam = lam
        self.lamc = float(self.nummats - 1) / np.sum( np.divide(1.0,lam) )
        for i in range(0,self.nummats):
            if self.lam[i] < self.lamc:
                print('\n**This combination of chord lengths invalid for N-ary Markovian-mixed media,**')
                print('**material '+str(i)+' average chord length too small compared to others**')
                raise
        self.prob = list( np.subtract( 1.0 , np.divide( self.lamc, self.lam ) ) )

    ## \brief Solve for N-ary parameters based on material abundances and correlation length
    #
    # \param[in] prob list of floats, material abundances
    # \param[in] lamc float, correlation length
    # \returns solves for/sets, self.lam, self.lamc, self.prob
    def solveNaryMarkovianParamsBasedOnProbAndCorrLength(self,prob,lamc):
        assert isinstance(prob,list) and len(prob)>1
        self.nummats = len(prob)
        for i in range(0,self.nummats):
            assert isinstance(prob[i],float) and prob[i]>0.0 and prob[i]<1.0
        self.prob = prob
        assert isinstance(lamc,float) and lamc>0.0
        self.lamc = lamc
        self.lam = list( np.divide( self.lamc, np.subtract( 1.0, self.prob ) ) )



    ## \brief Solves Adams, Larsen, and Pomraning binary benchmark inputs.
    #
    # Solves and stores as attributes ALP input params based on case designations used in BrantleyJQSRT2011.
    #
    # \param[in] CaseDesig str, letter and number designation, e.g., '1a', '3b', etc.
    # \returns solves for and sets Sigt, Scatrat, lam, lamc as attributes
    def selectALPInputs(self, CaseDesig=None ):
        assert isinstance(CaseDesig,str)
        assert len(CaseDesig)==2
        assert CaseDesig[0] in ('1','2','3')
        assert CaseDesig[1] in ('a','b','c')

        if   CaseDesig[0]=='1': self.Sigt = [10.0/99.0, 100.0/11.0 ]; self.lam = [0.99,0.11]
        elif CaseDesig[0]=='2': self.Sigt = [10.0/99.0, 100.0/11.0 ]; self.lam = [9.9 ,1.1 ]
        elif CaseDesig[0]=='3': self.Sigt = [2.0/101.0, 200.0/101.0]; self.lam = [5.05,5.05]

        if   CaseDesig[1]=='a': self.Scatrat = [0.0,1.0]
        elif CaseDesig[1]=='b': self.Scatrat = [1.0,0.0]
        elif CaseDesig[1]=='c': self.Scatrat = [0.9,0.9]

        self.Sigs = list( np.multiply( self.Sigt, self.Scatrat ) )

        self.solveNaryMarkovianParamsBasedOnChordLengths(self.lam)

    ## \brief Solves Vu, Brantley, Olson, and Kiedrowski quaternary benchmark inputs.
    #
    # Solves and stores as attributes VBOK input params based on case designations used in VuMC2021.
    # Note: The 'volume-fraction' mixing type is type compatible with multi-D Markovian and
    # the most straightforward CoPS CPFs.  The 'uniform' mixing type can still be realized in 1D,
    # easily adapted to with CLS, and CoPS CPFs could likely be (but as of yet haven't been) derived for.
    #
    # \param[in] CaseDesig int, number designation
    # \param[in] MixingType str, default 'volume-fraction'; 'volume-fraction' or 'uniform', N-ary mixing Markovian transition type
    # \returns solves for and sets Sigt, Scatrat, lam, lamc as attributes
    def selectVBOKInputs(self, CaseDesig=None, MixingType='volume-fraction' ):
        assert isinstance(CaseDesig,str)
        assert len(CaseDesig)==2
        assert CaseDesig[0] in ('1','2','3')
        assert CaseDesig[1] in ('a','b','c','d')
        assert MixingType=='volume-fraction' or MixingType=='uniform'

        if   CaseDesig[0]=='1': self.Sigt = [10.0/99.0, 100.0/11.0 , 10.0/99.0, 100.0/11.0 ]
        elif CaseDesig[0]=='2': self.Sigt = [2.0/101.0, 200.0/101.0, 2.0/101.0, 200.0/101.0]
        elif CaseDesig[0]=='3': self.Sigt = [10.0/99.0, 100.0/11.0 , 2.0/101.0, 200.0/101.0]

        if   MixingType=='volume-fraction':
            if   CaseDesig[0]=='1': self.lam = [ 110.0/101 ,  110.0/109 ,  11.0/2 ,  11.0/10]
            elif CaseDesig[0]=='2': self.lam = [ 101.0/20  ,  101.0/20  , 101.0/20, 101.0/20]
            elif CaseDesig[0]=='3': self.lam = [1680.0/1021, 1680.0/1109, 112.0/41, 112.0/41]
        elif MixingType=='uniform':
            if   CaseDesig[0]=='1': self.lam = [     0.99 ,      0.11   ,     9.9 ,     1.1 ]
            elif CaseDesig[0]=='2': self.lam = [     5.05 ,      5.05   ,     5.05,     5.05]
            elif CaseDesig[0]=='3': self.lam = [     0.99 ,      0.11   ,     5.05,     5.05]

        if   CaseDesig[1]=='a': self.Scatrat = [1.0,0.0,1.0,0.0]
        elif CaseDesig[1]=='b': self.Scatrat = [0.0,1.0,0.0,1.0]
        elif CaseDesig[1]=='c': self.Scatrat = [0.9,0.9,0.9,0.9]
        elif CaseDesig[1]=='d': self.Scatrat = [0.0,0.0,0.0,0.0]

        self.Sigs = list( np.multiply( self.Sigt, self.Scatrat ) )

        self.solveNaryMarkovianParamsBasedOnChordLengths(self.lam)
