#!usr/bin/env python
## \file RetrievePublishedSMTransportValues.py
#  \brief Example driver to read Counterpart data for ALP and VBOK benchmark sets.
#
#  Note: While the below demonstrates how to access some of the data collected in csv files in Counterparts, it
#  only makes use of a sample of the data currently in Counterparts for these benchmark problems and is not
#  formatted in the prettiest way.  Desired future functionality include a more complete dataset from
#  publications in Counterparts and a module or class in Core that streamlines accessing the data and provides it 
#  for the user in a more aesthetically pleasing way.
#
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import sys
sys.path.append('../../Core/Tools')
from MarkovianInputspy import MarkovianInputs
sys.path.append('../../Counterparts/BrantleyJQSRT2011')
sys.path.append('../../Counterparts/LarmierJQSRT2018_CLS')
sys.path.append('../../Counterparts/VuMC2021_CLS')
import pandas as pd


#Select the problem and solver for which to retrieve published data
benchmarkSet  = 'ALP'  #'ALP','VBOK'; Adams, Larsen, and Pomraning (ALP) binary media benchmark set or Vu, Brantley, Olson, and Kiedrowski quaternary media benchmark set
dimensionality= '1D'   #'1D','3D'
cases         = ['3a']   #'1a','1b','1c','2a','2b','2c','3a','3b',or'3c' from ALP; '1a','1b','1c','1d','2a','2b','2c','2d','3a','3b','3c',or'3d' from VBOK
methods       = ['Bench','AM','CLS']  #'Bench', 'AM', or 'CLS'


#Instantiate inputs class
CaseInp = MarkovianInputs()

#Read published results from csv files
Brant11     = pd.read_csv('../../Counterparts/BrantleyJQSRT2011/1DBrantleyJQSRT2011_leakage.csv',index_col=0,skiprows=2,encoding='unicode_escape')
Larm1DCLS   = pd.read_csv('../../Counterparts/LarmierJQSRT2018_CLS/1DLarmierJQSRT2018_CLS_leakage.csv',index_col=0)

Larm3DCLS   = pd.read_csv('../../Counterparts/LarmierJQSRT2018_CLS/3DLarmierJQSRT2018_CLS_leakage.csv',index_col=0)
Larm3DBench = pd.read_csv('../../Counterparts/LarmierJQSRT2018_CLS/3DLarmierJQSRT2018_Bench_leakage.csv',index_col=0)

Vu1DCLS     = pd.read_csv('../../Counterparts/VuMC2021_CLS/1DVuMC2021_CLS_leakage.csv',index_col=0)


#Define function that can print results in a standard format where standard error of the mean (SEM) values are aligned with values of interest
def printOutputsScreen(paper,tave,tunc,rave,runc,fave=None,func=None):
    print('     As reported in',paper+':')
    print(' Transmittance   mean:   {0:.8f}'.format(tave))
    try   : print('               +-SEM :   {0:.8f}'.format(tunc))
    except: pass
    print(' Reflectance     mean:   {0:.8f}'.format(rave))
    try   : print('               +-SEM :   {0:.8f}'.format(runc))
    except: pass
    if not fave==None:
        print(' Flux            mean:   {0:.8f}'.format(fave))
        print('               +-SEM :   {0:.8f}'.format(func))
    print()

#Loop through selected cases and methods and print results
for case in cases:
    if   benchmarkSet=='ALP' : CaseInp.selectALPInputs(  case )
    elif benchmarkSet=='VBOK': CaseInp.selectVBOKInputs( case )
    print('\n********************    ',benchmarkSet,'case',case,'    ********************')
    print('Case defined by the following values:',CaseInp,'\n')
    for method in methods:
        print('--------------------    Solved using',method,'    --------------------')
        if benchmarkSet=='ALP':
            if dimensionality=='1D':
                if method=='AM':
                    printOutputsScreen(paper='BrantleyJQSRT2011',tave=Brant11[case+'-SuiteI-AM-L10']['Trans'],tunc=None,rave=Brant11[case+'-SuiteI-AM-L10']['Refl'],runc=None)
                if method=='CLS':
                    printOutputsScreen(paper='BrantleyJQSRT2011',tave=Brant11[case+'-SuiteI-CLS-L10']['Trans'],tunc=None,rave=Brant11[case+'-SuiteI-CLS-L10']['Refl'],runc=None)
                    printOutputsScreen(paper='LarmierJQSRT2018_CLS',tave=Larm1DCLS[case+'-1D-SuiteI-CLS-L10']['Trans'],tunc=Larm1DCLS[case+'-1D-SuiteI-CLS-L10']['Trans_unc'],rave=Larm1DCLS[case+'-1D-SuiteI-CLS-L10']['Refl'],runc=Larm1DCLS[case+'-1D-SuiteI-CLS-L10']['Refl_unc'],fave=Larm1DCLS[case+'-1D-SuiteI-CLS-L10']['Flux'],func=Larm1DCLS[case+'-1D-SuiteI-CLS-L10']['Flux_unc'])
            elif dimensionality=='3D':
                if method=='Bench':
                    printOutputsScreen(paper='LarmierJQSRT2018_CLS',tave=Larm3DBench[case+'-3D-SuiteI-Bench-L10']['Trans'],tunc=Larm3DBench[case+'-3D-SuiteI-Bench-L10']['Trans_unc'],rave=Larm3DBench[case+'-3D-SuiteI-Bench-L10']['Refl'],runc=Larm3DBench[case+'-3D-SuiteI-Bench-L10']['Refl_unc'],fave=Larm3DBench[case+'-3D-SuiteI-Bench-L10']['Flux'],func=Larm3DBench[case+'-3D-SuiteI-Bench-L10']['Flux_unc'])
                if method=='CLS':
                    printOutputsScreen(paper='LarmierJQSRT2018_CLS',tave=Larm3DCLS[case+'-3D-SuiteI-CLS-L10']['Trans'],tunc=Larm3DCLS[case+'-3D-SuiteI-CLS-L10']['Trans_unc'],rave=Larm3DCLS[case+'-3D-SuiteI-CLS-L10']['Refl'],runc=Larm3DCLS[case+'-3D-SuiteI-CLS-L10']['Refl_unc'],fave=Larm3DCLS[case+'-3D-SuiteI-CLS-L10']['Flux'],func=Larm3DCLS[case+'-3D-SuiteI-CLS-L10']['Flux_unc'])
        elif benchmarkSet=='VBOK':
            if dimensionality=='1D':
                if method=='CLS':
                    printOutputsScreen(paper='VuMC2021',tave=Vu1DCLS[case+'-volfrac-CLS-L10']['Trans'],tunc=Vu1DCLS[case+'-volfrac-CLS-L10']['Trans_unc'],rave=Vu1DCLS[case+'-volfrac-CLS-L10']['Refl'],runc=Vu1DCLS[case+'-volfrac-CLS-L10']['Refl_unc'])