#!usr/bin/env python
import numpy as np
from FluxTallyToolspy import FluxTallyTools
from LeakageTallyToolspy import LeakageTallyTools

## \brief Infrastructure for leakage tallies.
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
#
# Assumes particle weights of 1.
class Tallies(FluxTallyTools,LeakageTallyTools):
    def __init__(self):
        super(Tallies,self).__init__()
        self.Tals = []
        self.TotalTime = 0.0
        self.NumOfParticles = 0

    def __str__(self):
        return str(self.__dict__)

    ## \brief Instantiate new sample; sample can be stochastic media or UQ sample or batch
    #
    # \returns adds new dictionary to 'self.Tals' list of dictionaries
    def _initializeSampleTallies(self):
        self.Tals.append( {'Transmit': 0, 'Reflect': 0, 'Absorb': 0, 'SideLeakage': 0,'MCTime': 0,'SampleTime':0} )
        self.CurrentSampleRuntime = 0.0 #This variable is useful for separating MC vs. Sample runtime when there is only one history per sample (tallied as only one sample)
        self._initializeSampleFluxTallies()
        if self.flLeakTallies: self._initializeSampleLeakageTallies()

    ## \brief Accesses values in the Tals dictionary and returns them as a list
    #
    # \param whichList str, name of values to retrieve from dictionary
    # \returns list of values
    def returnListFromTalsDictionary(self,whichList):
        vals = []
        for isamp in range(0,len(self.Tals)):
            vals.append( self.Tals[isamp][whichList] )
        return vals[:]

    ## \brief Computes and returns total, Monte Carlo, and sample runtime values
    #
    # A sample can be either a collection of particle histories nominally identical to 
    # those in other samples or a collection of particle histories simulated with a unique
    # value of one or more uncertain inputs (such as a stochastic media realization).
    # In each case, the sample runtime is the time to prepare the sample (e.g., sample the
    # realization) and collect and process tallies at the sample level.
    #
    # \param[in]  flVerbose bool, default False, print to screen the data to return?
    # \param[out] self.TotalTime float, total runtime of calculation in seconds
    # \param[out] TimeForHists float, total runtime of simulating all particle histories in seconds
    # \param[out] TimePerHist float, average runtime per particle history in seconds, C_{MC} in OlsonANS2018
    # \param[out] TimeForSamps float, total runtime of preparing and tallying samples in seconds
    # \param[out] TimePerSamp float, average runtime per sample prepared and tallied in seconds, C_{RS} in OlsonANS2018
    # \returns self.TotalTime, MCRuntime, TimePerHist, SampleRuntime, TimePerSamp
    def returnRuntimeValues(self,flVerbose=False):
        assert isinstance(flVerbose,bool)
        MCtime_vals     = self.returnListFromTalsDictionary('MCTime')
        Sampletime_vals = self.returnListFromTalsDictionary('SampleTime')

        TimeForHists = sum(MCtime_vals)
        TimePerHist  = TimeForHists/self.NumOfParticles
        TimeForSamps = sum(Sampletime_vals)
        TimePerSamp  = np.mean(Sampletime_vals)
        if self.NumParticlesPerSample==1: TimePerSamp /= self.NumOfParticles #in this case, all sample runtime data collected in same place
        if flVerbose:
            print('Total  time               :  {0:13.8f}'.format(self.TotalTime))
            print('MC     time; tot, per hist:  {0:13.8f}  {1:13.8f}'.format(TimeForHists,TimePerHist))
            print('Sample time; tot, per samp:  {0:13.8f}  {1:13.8f}'.format(TimeForSamps,TimePerSamp))
        return self.TotalTime, TimeForHists, TimePerHist, TimeForSamps, TimePerSamp


    ## \brief Drives the computation of transmittance mean, sample transmittance variance, and precision statistics on these quantities
    #
    # \param flVerbose bool, default False, print the computed data to screen?
    # \param NumStatPartitions int, default None, number of partitions for deterermining MC statistics on computed variance 
    # \returns tmean,tdev (square root of sample variance), tmeanSEM,tdevSEM
    def returnTransmittanceMoments(self,flVerbose=False,NumStatPartitions=None):
        assert isinstance(flVerbose,bool)
        if len(self.Tals)==1: return self._returnParticleKillMoments1HistPerSamp(flVerbose,'Transmittance   ',self.Tals[-1]['Transmit']/self.NumOfParticles)
        else:
            tran_vals = self.returnListFromTalsDictionary('Transmit')
            tran_vals = list( np.divide( tran_vals, self.NumParticlesPerSample ) ) #compute averages
            tran_MCSEM_vals = self._computeSEMs(tran_vals,tran_vals,self.NumParticlesPerSample)
            return self._returnParticleKillMomentsGT1HistPersamp(flVerbose,'Transmittance   ',tran_vals,tran_MCSEM_vals,NumStatPartitions)

    ## \brief Drives the computation of reflectance mean, sample reflectance variance, and precision statistics on these quantities
    #
    # \param flVerbose bool, default False, print the computed data to screen?
    # \param NumStatPartitions int, default None, number of partitions for deterermining MC statistics on computed variance 
    # \returns rmean,rdev (square root of sample variance), rmeanSEM,rdevSEM
    def returnReflectanceMoments(self,flVerbose=False,NumStatPartitions=None):
        assert isinstance(flVerbose,bool)
        if len(self.Tals)==1: return self._returnParticleKillMoments1HistPerSamp(flVerbose,'Reflectance     ',self.Tals[-1]['Reflect']/self.NumOfParticles)
        else:
            refl_vals = self.returnListFromTalsDictionary('Reflect')
            refl_vals = list( np.divide( refl_vals, self.NumParticlesPerSample ) ) #compute averages
            refl_MCSEM_vals = self._computeSEMs(refl_vals,refl_vals,self.NumParticlesPerSample)
            return self._returnParticleKillMomentsGT1HistPersamp(flVerbose,'Reflectance     ',refl_vals,refl_MCSEM_vals,NumStatPartitions)

    ## \brief Drives the computation of absoprtion mean, sample absoprtion variance, and precision statistics on these quantities
    #
    # \param flVerbose bool, default False, print the computed data to screen?
    # \param NumStatPartitions int, default None, number of partitions for deterermining MC statistics on computed variance 
    # \returns amean,adev (square root of sample variance), ameanSEM,adevSEM
    def returnAbsorptionMoments(self,flVerbose=False,NumStatPartitions=None):
        assert isinstance(flVerbose,bool)
        if len(self.Tals)==1: return self._returnParticleKillMoments1HistPerSamp(flVerbose,'Absorption      ',self.Tals[-1]['Absorb']/self.NumOfParticles)
        else:
            abso_vals = self.returnListFromTalsDictionary('Absorb')
            abso_vals = list( np.divide( abso_vals, self.NumParticlesPerSample ) ) #compute averages
            abso_MCSEM_vals = self._computeSEMs(abso_vals,abso_vals,self.NumParticlesPerSample)
            return self._returnParticleKillMomentsGT1HistPersamp(flVerbose,'Absorption      ',abso_vals,abso_MCSEM_vals,NumStatPartitions)

    ## \brief Drives the computation of side leakage mean, sample side leakage variance, and precision statistics on these quantities
    #
    # \param flVerbose bool, default False, print the computed data to screen?
    # \param NumStatPartitions int, default None, number of partitions for deterermining MC statistics on computed variance 
    # \returns amean,adev (square root of sample variance), ameanSEM,adevSEM
    def returnSideLeakageMoments(self,flVerbose=False,NumStatPartitions=None):
        assert isinstance(flVerbose,bool)
        if len(self.Tals)==1: return self._returnParticleKillMoments1HistPerSamp(flVerbose,'SideLeakage     ',self.Tals[-1]['SideLeakage']/self.NumOfParticles)
        else:
            side_vals = self.returnListFromTalsDictionary('SideLeakage')
            side_vals = list( np.divide( side_vals, self.NumParticlesPerSample ) ) #compute averages
            side_MCSEM_vals = self._computeSEMs(side_vals,side_vals,self.NumParticlesPerSample)
            return self._returnParticleKillMomentsGT1HistPersamp(flVerbose,'SideLeakage     ',side_vals,side_MCSEM_vals,NumStatPartitions)

    ## \brief Drives the computation of whole-domain flux mean, sample whole-domain flux variance, and precision statistics on these quantities
    #
    # \param flVerbose bool, default False, print the computed data to screen?
    # \param NumStatPartitions int, default None, number of partitions for deterermining MC statistics on computed variance 
    # \returns fmean,fdev (square root of sample variance), fmeanSEM,fdevSEM
    def returnWholeDomainFluxMoments(self,flVerbose=False,NumStatPartitions=None):
        assert isinstance(flVerbose,bool)
        if len(self.Tals)==1:
            ave = self.FluxMean
            var = self.FluxSEM*self.NumOfParticles
            SEM = self.FluxSEM
            if flVerbose:
                print('WholeDomFlux    mean, totvar,       :   {0:.8f}  {1:.8f}'.format(ave,var))
                print('              +-SEM                 :   {0:.8f}'.format(SEM))
            return ave,var,SEM,None
        else:
            flux_mom1s = self.returnListFromTalsDictionary('SampleFluxMean')
            flux_SEMs  = self.returnListFromTalsDictionary('SampleFluxSEM')
            return self._returnParticleKillMomentsGT1HistPersamp(flVerbose,'WholeDomFlux    ',flux_mom1s,flux_SEMs,NumStatPartitions)

    ## \brief Computes and returns information on tallied leakage values when 1 history per sample.
    #
    # Computes mean, total variance, and standard error of the mean (Monte Carlo precision)
    # for leakage tallies when only one history per sample.
    #
    # \param flVerbose bool, default False, print values to screen?
    # \param partkilltype str, used in part of the print message
    # \param ave int, average of tally of kill type of interest
    # \returns ave, var, SEM, None (to fit syntax of sister method, _returnParticleKillMomentsGT1HistPersamp)
    def _returnParticleKillMoments1HistPerSamp(self,flVerbose,partkilltype,ave):
        var = ( ave - ave**2 ) * (self.NumOfParticles/(self.NumOfParticles-1.0))
        SEM = np.sqrt(var)/float(np.sqrt(self.NumOfParticles))
        if flVerbose:
            print(partkilltype +  'mean, totvar,       :   {0:.8f}  {1:.8f}'.format(ave,var))
            print('              +-SEM                 :   {0:.8f}'.format(SEM))
        return ave,var,SEM,None
    
    ## \brief Computes and returns information on tallied leakage values when >1 history per sample.
    #
    # Computes mean, unbiased sample standard deviation (via variance deconvolution, see ClementsJQSRT2024),
    # and the standard error (statistical precision) of these tallies.
    # Partitions are groupings of samples.
    # - There must be the snaem number of particles per sample in order to get reliable Monte Carlo statistics.
    #    The method will provide a user warning, but still run, if this condition is not met.
    # - There must be at least two partitions to compute precision estimates on computed values. There need
    #    to be more partitions to get decent statistics.  The method asserts that there is at least one.
    # - Each partition must have the same (whole number) number of samples in order to get meaningful precision
    #    estimates. The method will raise one of two exceptions if this condition is not met.
    # - There must be at least two samples per partition to perform variance deconvolution within the partition.
    #    The method compute statistics for the mean, but will not estimate the sample variance if this condition not met.
    #
    # \param flVerbose bool, default False, print values to screen?
    # \param partkilltype str, used in part of the print message
    # \param tal_vals list of ints or floats, tally of kill type of interest on each sample
    # \param tal_MCSEM_vals list of floats, standard error of the mean for kill type on each sample
    # \param NumStatPartitions int, default None, number of statistical partitions for deterermining MC statistics on computed variance (when more than one history per sample)
    # \returns ave,rvvar, aveSEM,rvvarSEM
    def _returnParticleKillMomentsGT1HistPersamp(self,flVerbose,partkilltype,tal_vals,tal_MCSEM_vals,NumStatPartitions):
        if not self.NumOfParticles%self.NumParticlesPerSample==0:
            print('***************************************************************************************')
            print('** Warning: MC statistics not rigorous since variable number of histories per sample **')
            print('***************************************************************************************')
        assert isinstance(NumStatPartitions,int) and NumStatPartitions>0
        if not len(tal_vals)>=NumStatPartitions   : raise Exception("Need to specify at least as many samples as partitions of samples. Currently "+str(len(tal_vals))+" samples and "+str(NumStatPartitions)+" partitions.")
        if not len(tal_vals) %NumStatPartitions==0: raise Exception("Need the number of partitions to be an integer factor of the numer of samples. Currently "+str(len(tal_vals))+" samples and "+str(NumStatPartitions)+" partitions.")
        flAtLeastTwoSamplesPerPartition = False if len(tal_vals)==NumStatPartitions else True

        numSampPerPartition = int(len(tal_vals)/NumStatPartitions)
        aves=[]; rvvars=[]
        for ipartition in range(0,NumStatPartitions):
            istart =   ipartition       * numSampPerPartition
            iend   = ( ipartition + 1 ) * numSampPerPartition
            #Compute mean on sample
            aves.append( np.average(tal_vals[istart:iend]) )
            #Compute random variable variance using variance deconvolution
            if flAtLeastTwoSamplesPerPartition:
                totvar   = np.var(                                 tal_vals[istart:iend], ddof=1)
                aveMCvar = np.average( np.multiply( np.power(tal_MCSEM_vals[istart:iend],2) , self.NumParticlesPerSample) ) #average MC variance from samples (back-compute from MC SEM on each realization and average) 
                rvvars.append( totvar - aveMCvar/self.NumParticlesPerSample )  #Variance deconvolution - remove average MC noise from polluted variance estimate to yield unbiased estimate of varianced caused by random variables

        ave   = np.mean(aves)  ; aveSEM   = np.std(  aves,ddof=1)/np.sqrt(NumStatPartitions)

        if flAtLeastTwoSamplesPerPartition:
            rvvar = np.mean(rvvars); rvvarSEM = np.std(rvvars,ddof=1)/np.sqrt(NumStatPartitions)
            if flVerbose:
                print(partkilltype +  'mean,  rvvar:   {0:.8f}  {1:.8f}'.format(ave   ,rvvar   ))
                print('       +- SEM               :   {0:.8f}  {1:.8f}'.format(aveSEM,rvvarSEM))
            return ave,rvvar, aveSEM,rvvarSEM
        else:
            if flVerbose:
                print(partkilltype +  'mean        :   {0:.8f}                     '.format(ave   ))
                print('       +- SEM               :   {0:.8f}                     '.format(aveSEM))
            return ave,None, aveSEM,None


    ## \brief Provides warning message and returns None values if user tries to perform runtime analysis with only one sample
    #
    # \returns None,None,None,None,None
    def returnRuntimeAnalysisAndNonesWithWarning(self):
        print('***************************************************************************************')
        print('** Warning: Need more than one sample to perform runtime analysis; returning Nones   **')
        print('***************************************************************************************')
        return None,None,None,None,None

    ## \brief Drives the performance of runtime analysis for transmittance (primarily based on OlsonANS2018)
    #
    # \param flVerbose bool, default False, print the computed data to screen?
    # \param NumStatPartitions int, default None, number of partitions for computing sample variance
    # \returns MCdev,TimePerHist,RVdev,TimePerSamp,optN_eta
    def returnTransmittanceRuntimeAnalysis(self,flVerbose=False,NumStatPartitions=None):
        assert isinstance(flVerbose,bool)
        if len(self.Tals)==1: return self.returnRuntimeAnalysisAndNonesWithWarning()
        tran_vals = self.returnListFromTalsDictionary('Transmit')
        tran_vals = list( np.divide( tran_vals, self.NumParticlesPerSample ) ) #compute averages
        return self._returnRuntimeAnalysis(flVerbose,'Transmittance   ',tran_vals,NumStatPartitions)

    ## \brief Drives the performance of runtime analysis for reflectance (primarily based on OlsonANS2018)
    #
    # \param flVerbose bool, default False, print the computed data to screen?
    # \param NumStatPartitions int, default None, number of partitions for computing sample variance
    # \returns MCdev,TimePerHist,RVdev,TimePerSamp,optN_eta
    def returnReflectanceRuntimeAnalysis(self,flVerbose=False,NumStatPartitions=None):
        assert isinstance(flVerbose,bool)
        if len(self.Tals)==1: return self.returnRuntimeAnalysisAndNonesWithWarning()
        refl_vals = self.returnListFromTalsDictionary('Reflect')
        refl_vals = list( np.divide( refl_vals, self.NumParticlesPerSample ) ) #compute averages
        return self._returnRuntimeAnalysis(flVerbose,'Reflectance     ',refl_vals,NumStatPartitions)

    ## \brief Drives the performance of runtime analysis for absorption (primarily based on OlsonANS2018)
    #
    # \param flVerbose bool, default False, print the computed data to screen?
    # \param NumStatPartitions int, default None, number of partitions for computing sample variance
    # \returns MCdev,TimePerHist,RVdev,TimePerSamp,optN_eta
    def returnAbsorptionRuntimeAnalysis(self,flVerbose=False,NumStatPartitions=None):
        assert isinstance(flVerbose,bool)
        if len(self.Tals)==1: return self.returnRuntimeAnalysisAndNonesWithWarning()
        abso_vals = self.returnListFromTalsDictionary('Absorb')
        abso_vals = list( np.divide( abso_vals, self.NumParticlesPerSample ) ) #compute averages
        return self._returnRuntimeAnalysis(flVerbose,'Absorption      ',abso_vals,NumStatPartitions)

    ## \brief Drives the performance of runtime analysis for side leakage (primarily based on OlsonANS2018)
    #
    # \param flVerbose bool, default False, print the computed data to screen?
    # \param NumStatPartitions int, default None, number of partitions for computing sample variance
    # \returns MCdev,TimePerHist,RVdev,TimePerSamp,optN_eta
    def returnSideLeakageRuntimeAnalysis(self,flVerbose=False,NumStatPartitions=None):
        assert isinstance(flVerbose,bool)
        if len(self.Tals)==1: return self.returnRuntimeAnalysisAndNonesWithWarning()
        side_vals = self.returnListFromTalsDictionary('SideLeakage')
        side_vals = list( np.divide( side_vals, self.NumParticlesPerSample ) ) #compute averages
        return self._returnRuntimeAnalysis(flVerbose,'SideLeakage     ',side_vals,NumStatPartitions)

    ## \brief Performs runtime analysis on selected quantity. 
    #
    #
    # \param[in] flVerbose bool, default False, print the computed data to screen?
    # \param[in] partkilltype str, used in part of the print message
    # \param[in] tal_vals list of ints or floats, tally of kill type of interest on each sample
    # \param[in] NumStatPartitions int, default None, number of partitions for computing sample variance
    # \param[out] MCdev float, square root of the average Monte Carlo variance (\sqrt{N_\eta E_{\xi}[\sigma_\eta^2]} in ClementsJQSRT2024, similar to \sigma_{MC} in OlsonANS2018--believe should average variance then take root as here instead of take roots and then averages as in OlsonANS2018, the two are similar with N_\eta large, but diverge when N_\eta is small)
    # \param[out] TimePerHist float, average runtime per particle history in seconds (C_{MC} in OlsonANS2018)
    # \param[out] RVdev float, random variable standard deviation (sqrt(Var_{\xi}[Q]) is notation from ClementsJQSRT2024, \sigma_{RS} is used OlsonANS2018)
    # \param[out] TimePerSamp float, average runtime per sample prepared and tallied in seconds (C_{RS} in OlsonANS2018)
    # \param[out] optN_eta float, computed optimal number of histories per sample (N_\eta in ClementsJQSRT2024, N in OlsonANS2018)
    # \returns MCdev,TimePerHist,RVdev,TimePerSamp,optN_eta
    def _returnRuntimeAnalysis(self,flVerbose,partkilltype,tal_vals,NumStatPartitions):
        assert isinstance(NumStatPartitions,int) and NumStatPartitions>1

        #Prep computation distribution parameters 
        N_eta = self.NumParticlesPerSample #number of histories per sample (N_eta is notation from ClementsJQSRT2024, N is used in OlsonANS2018)
        N_xi  = len(self.Tals)             #number of samples (N_xi is notation from ClementsJQSRT2024, R is used in OlsonANS2018)

        #Prep standard deviation values
        kill_MCSEM_vals = self._computeSEMs(tal_vals,tal_vals,self.NumParticlesPerSample)

        MCvars= np.power( np.multiply( kill_MCSEM_vals, np.sqrt(N_eta) ), 2)
        MCdev = np.sqrt( np.mean(MCvars) ) #square root of the average Monte Carlo variance

        ave,rvvar, aveSEM,rvvarSEM = self._returnParticleKillMomentsGT1HistPersamp(False,partkilltype,tal_vals,kill_MCSEM_vals,NumStatPartitions)
        RVdev = np.sqrt(rvvar)             #random variable standard deviation

        #Prep runtime values
        tott, MCt, TimePerHist, Samplet, TimePerSamp = self.returnRuntimeValues(False)

        #Compute optimal number of histories per sample
        optN_eta = MCdev / RVdev * np.sqrt( TimePerSamp / TimePerHist )
        

        #Same cost, greater precision (formulas from OlsonANS2018)
        R_tbud = self.TotalTime / ( TimePerSamp + MCdev/RVdev * np.sqrt(TimePerSamp*TimePerHist) )
        u_tbud  = ( np.sqrt(TimePerSamp)*RVdev + np.sqrt(TimePerHist)*MCdev ) / np.sqrt(self.TotalTime)

        #Same precision, less cost (formulas from OlsonANS2018)
        R_utol =   ( RVdev**2 + RVdev*MCdev*np.sqrt(TimePerHist/TimePerSamp) ) / aveSEM**2
        t_utol = ( ( np.sqrt(TimePerSamp)*RVdev + np.sqrt(TimePerHist)*MCdev ) / aveSEM    ) **2

        #print values to screen
        if flVerbose:
            print(partkilltype +  'MC   dev, t :   {0:.8f}  {1:.8f}     optN_eta:{2:6.1f}     Reduce SEM     by ~{3:.0f}% for same cost      with {4:7.0f} samples'.format(MCdev,TimePerHist,optN_eta,(aveSEM-u_tbud)/aveSEM*100,R_tbud))
            print('                Samp dev, t :   {0:.8f}  {1:.8f}                         Reduce runtime by ~{2:.0f}% for same precision with {3:7.0f} samples'.format(RVdev,TimePerSamp,(self.TotalTime-t_utol)/self.TotalTime*100,R_utol))
        return MCdev,TimePerHist,RVdev,TimePerSamp,optN_eta