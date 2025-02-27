#!usr/bin/env python
import numpy as np

## \brief Class for random number object
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
# Class is essentially a veneer for numpy's random number object, but
# makes often desired functionality easy to use with a couple intuitive
# commands.
class RandomNumbers():
    ## \brief Initializes internal random number object.
    #
    # Random numbers for Monte Carlo computation here use an instance of
    # numpy.random.RandomState.  The user can let numpy choose where in the pseudo-random
    # number chain to select numbers, or can specify.  If the user specifies (flUseSeed==True),
    # the sequence is re-seeded at the beginning of each stride by calling setSeedAtStride using
    # the seed, stridelen, and stride number, i.e., localseed = seed + stridelen * istride.
    # A stride can be a particle history, set of geometries, or anything else.
    # This makes calcluations repeatable even when done in parts or in parallel.
    #
    # \param[in] flUseSeed bool, use specified seed and thus be repeatable (or let numpy choose seed
    #                           and be different each time?)
    # \param seed int, seed for random number sequence
    # \param stridelen int, perturbation of seed for each particle history
    def __init__(self,flUseSeed=None,seed=None,stridelen=None):
        assert isinstance(flUseSeed,bool)
        if seed==None     : seed   = 938901
        if stridelen==None: stridelen = 267
        assert isinstance(seed,int) and isinstance(stridelen,int)
        self.flUseSeed = flUseSeed
        self.Seed      = seed
        self.Stridelen = stridelen
        self.RngObj    = np.random.RandomState()
        self.setSeedAtStride(1)
        
    def __str__(self):
        return str(self.__dict__)

    ## \brief Initializes random number seed before each stride
    #
    # \param[in] istride int, stride number
    # \returns sets RngObj attribute seed for particular stride
    def setSeedAtStride(self,istride):
        if self.flUseSeed: self.RngObj.seed( self.Seed + self.Stridelen * istride )

    ## \brief Returns next random number`
    #
    # \returns returns next random number
    def rand(self):
        return self.RngObj.rand()

    ## \brief Returns index of selection based on discrete PDF
    #
    # \param[in] p list of floats, discrete PDF to sample from
    # \returns returns index of selection
    def choice(self,p):
        return int(self.RngObj.choice(len(p), p=p))


    ## \brief Returns next random number
    #
    # \returns returns next random number
    def uniform(self,low,high):
        return self.RngObj.uniform(low,high)


    ## \brief Returns sample from Poisson distribution
    #
    # \param[in] lam float, expected value
    # \returns sample
    def poisson(self,lam):
        return self.RngObj.poisson(lam)


    ## \brief Returns sample from binomial distribution
    #
    # A sample from a binomial distribution is also known
    # as the number of successes from a Bernoulli trial. 
    #
    # \param[in] n int, number of trials, 0<n
    # \param[in] p float, probability of success, 0<p<1
    # \returns numsuccesses
    def binomial(self,n,p):
        return self.RngObj.binomial(n,p)

    ## \brief Returns a sampled random integer
    #
    # \param[in] low int, low end random integer to generate
    # \param[in] high int, high end of random integer to generate
    # \returns integer
    def randint(self,low,high):
        return int( self.RngObj.randint(low,high) )