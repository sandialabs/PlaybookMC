#!usr/bin/env python
## \file CompareOutputDataAnalysispy.py
#  \brief Compares output data for difference and plotting analysis.  
#  \author Revolution Rivera-Felix, rfriver@sandia.gov, revolutionriverafelix@gmail.com
#  \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
import pandas as pd
import numpy as np
import warnings 
import matplotlib.pyplot as plt


class CompareOutputDataAnalysis(object):
    def __init__(self):
        self.reference  = {}
        self.comparison = {}
        self.metrics    = {} 
        self.markers    = ['o', '^', 's', 'D', 'v', 'p', '+', '1', 'x', '*', 'h']
        self.shorthand_mapping = {
                                "Sign"  : "Signed",
                                "Unsign": "Unsigned",
                                "Rel"   : "Relative",
                                "Abs"   : "Absolute",
                                "Diff"  : "Difference"}

    ## \brief Read in CSV data, name it, and add to class-defined dictionaries
    #
    #  Only one reference dataset can be defined at a time. 
    #  Multiple comparison datasets can be defined at a time. 
    #  When using comparison capabilities in this class, 
    #  each of the comparison datasets will be compared to the reference dataset.  
    #  It is recommended that a new object be created for each reference dataset
    #  a user wants to work with.
    # 
    # \param[in] filePath string, local path to file read in by pandas
    # \param[in] type string, choose between reference or comparison dataset
    # \param[in] name string, optional parameter to name file data (otherwise name will be full file path)
    # \returns Sets as attributes the dictionaries of reference and comparison datasets 
    def readCSV(self, filePath, type, name=None):
        if type not in ('reference', 'comparison'): raise Exception ("\'type\' must be either \'reference\' or \'comparison\'.")
        if name==None: name = filePath  
        assert isinstance(name, str)
        if type=="reference": 
            if len(self.reference) > 0: 
                warnings.warn("Only one reference dataset allowed at a time; new reference dataset has overwritten old reference dataset. Comparison datasets may contain values computed with different reference datasets. To ensure the clarity and purity of the computed values, create a new object for every reference dataset you want to use.")
                self.reference.clear()
            self.reference[name]  = pd.read_csv(filePath, comment="#", index_col=[0])
        elif type=="comparison": 
            self.comparison[name] = pd.read_csv(filePath, comment="#", index_col=[0])

    ## \brief Compute difference between the reference dataset and each comparison dataset 
    #
    # Note that rows with uncertainty values must have a row index that follows the format Quantity_unc (e.g. "Refl_unc")  
    # or else the propogation of uncertainty values will be skipped for the current operation.
    #
    # \param[in] SignedOrUnsignedDiffs str, 'signed' or 'unsigned', computes signed or unsigned differences
    # \param[in] AbsoluteOrRelativeDiffs string, 'absolute' or 'relative', computes absolute or relative differences
    # \param[in] quantity string, row of the datasets the user wants to work with (e.g. 'Refl' or 'Trans')
    # \returns Adds difference values and difference uncertainty values as new rows to the comparison dataset(s)
    def computeDifferences(self, SignedOrUnsignedDiffs, AbsoluteOrRelativeDiffs, quantity):
        if len(self.reference)  == 0: warnings.warn("There is no reference dataset. A reference dataset is necessary to compute differences with the comparison datasets. Results of this function call may not return expected values.")
        if len(self.comparison) == 0: warnings.warn("There is no comparison dataset. At least one comparison dataset is required to compute differences. Results of this function call may not return expected values.")
        if SignedOrUnsignedDiffs   not in {'signed'  , 'unsigned'}: raise Exception ("\'SignedOrUnsignedDiffs\' must be either \'relative\' or \'absolute\'.")
        if AbsoluteOrRelativeDiffs not in {'relative', 'absolute'}: raise Exception ("\'AbsoluteOrRelativeDiffs\' must be either \'relative\' or \'absolute\'.")
        assert isinstance(quantity, str)
        identifier  = ' Sign' if SignedOrUnsignedDiffs   == 'signed'   else ' Unsign'
        identifier += ' Abs'  if AbsoluteOrRelativeDiffs == 'absolute' else ' Rel'
        identifier += ' Diff'
        # Get values from reference dataset according to quantity
        refRow                = self.reference[next(iter(self.reference))].loc[quantity] 
        flReferenceUncPresent = self.reference[next(iter(self.reference))].index.str.contains(quantity+"_unc").any()
        if flReferenceUncPresent: refUncRow = self.reference[next(iter(self.reference))].loc[quantity+"_unc"]
        else                    : warnings.warn(f"The Reference Dataset does not have any uncertainty values associated with the {quantity} values. The propogation of error uncertainty has been skipped for this operation.")
        for key in self.comparison: 
            # Compute differences
            compRow = self.comparison[key].loc[quantity]
            diff    = compRow - refRow
            if AbsoluteOrRelativeDiffs == 'relative': diff = diff / refRow
            if SignedOrUnsignedDiffs   == 'unsigned': diff = np.abs(diff)
            self.comparison[key].loc[quantity+identifier]  = diff 
            # Compute uncertainties of differences
            if flReferenceUncPresent: 
                flCompUncPresent = self.comparison[key].index.str.contains(quantity+"_unc").any()
                if flCompUncPresent: 
                    compUncRow = self.comparison[key].loc[quantity+"_unc"]
                    if AbsoluteOrRelativeDiffs   == 'relative': unc = np.sqrt((compUncRow/refRow)**2 + ((compRow * refUncRow)/refRow**2)**2)
                    elif AbsoluteOrRelativeDiffs == 'absolute': unc = np.sqrt(refUncRow**2 + compUncRow**2)
                    self.comparison[key].loc[quantity+identifier+" Unc"] = unc
            else: warnings.warn(f"{key} does not have any uncertainty values associated with the {quantity} values. The propogation of error uncertainty has been skipped for this operation.")       

    ## \brief Compute metrics (i.e. mean absolute, root mean squared, and max absolute) of the difference values in the comparison DataFrames 
    #
    # Comparison datasets with no difference values will be skipped with warning given. 
    # 
    # \param[in] rowIdentifier str, default 'Diff', used to search datasets for difference values
    # \returns Adds difference metrics values as new row(s) to the comparison dataset(s)
    def computeDifferenceMetrics(self, rowIdentifier='Diff'): 
        if len(self.comparison) == 0: raise Exception("There is no comparison dataset. At least one comparison dataset is required to compute differences. Results of this function call may not return expected values.")
        assert isinstance(rowIdentifier, str) 
        for key in self.comparison: 
            # Create temp DataFrame with rows containing the rowIdentifier string in the row index
            tempDF = self.comparison[key][self.comparison[key].index.str.contains(rowIdentifier)& ~self.comparison[key].index.str.contains('Unc')]  
            if tempDF.empty == True: 
                warnings.warn(f" The \'{key}\' dataset does not contain any difference values. The metrics computations have been skipped for this dataset.")
                continue
            # Apply lambda functions to the rows to compute metrics
            diffDF     = pd.DataFrame({
                'meanDiff'  : tempDF.apply(lambda row: np.mean(np.abs( row   )), axis=1),
                'RMSDiff'   : tempDF.apply(lambda row: np.sqrt(np.mean(row**2)), axis=1), 
                'maxAbsDiff': tempDF.apply(lambda row: np.max( np.abs( row   )), axis=1)
            }) 
            self.metrics[key] = diffDF

    ## \brief Plots the output values specified by the user 
    # 
    # \param[in] quantity str, default None (e.g. Refl, Trans), used to identify and collect data from the DataFrames
    # \param[in] fileName str, default 'outputs', used to specify file name of plots generated
    # \param[in] flShowPlot bool, default True, used to display plot to user after calculations
    # \param[in] flSavePlot bool, default True, used to automatically save generated plots to a .png file at the location of the driver script
    # \param[in] yLim tuple of floats, default None (e.g. (-1, 1.5)), sets the y-axis limits of the plot
    # \param[in] plotTitle str, default None, allows user to specify a title for the plot (otherwise one will be generated automatically)
    # \returns Displays and saves an output plot dependent on user specifications
    def plotOutputsAcrossCases(self, quantity=None, fileName='outputs', flShowPlot=True, flSavePlot=True, yLim=None, plotTitle=None): 
        if quantity == None: raise Exception ('Please specify which quantity you want to plot (e.g. \'Refl\' or \'Trans\').')
        plt.figure()
        xAxis = self.comparison[next(iter(self.comparison))].columns.tolist()
        markerCounter = 0
        for key in self.comparison:
            # Slice comparison DataFrames based on user parameters to prepare for plotting 
            rows    = self.comparison[key][self.comparison[key].index.str.contains(quantity)  & ~self.comparison[key].index.str.contains('Diff') & ~self.comparison[key].index.str.contains('unc')]
            uncRows = self.comparison[key][self.comparison[key].index.str.contains(quantity)  & ~self.comparison[key].index.str.contains('Diff') &  self.comparison[key].index.str.contains('unc')]
            # Adding error bars & plotting 
            if rows.empty: warnings.warn(f"There are no {quantity} values in the {key} dataset.") 
            else: 
                for idx, row in rows.iterrows(): 
                    if idx+"_unc" in uncRows.index: yerr = uncRows.loc[idx+"_unc"].tolist()
                    else                          : yerr = None
                    expandedIdx = " ".join([self.shorthand_mapping.get(word, word) for word in idx.split()]) # Expands the short-hand version of the legend
                    plt.errorbar(xAxis, row, yerr=yerr, marker=self.markers[markerCounter % len(self.markers)], markersize=5, label=f'{key} - {expandedIdx}') 
                    markerCounter += 1
                if plotTitle: plt.title(plotTitle,              fontsize=25)
                else        : 
                    title = quantity+' '+'Leakage'
                    expandedTitle = " ".join([self.shorthand_mapping.get(word, word) for word in title.split()]) # Expands the short-hand version of the title
                    plt.title(expandedTitle, fontsize=25)
                plt.ylabel('Leakage Values', fontsize=20)

        # Check if any data points have been plotted
        check = plt.gca()
        if len(check.lines) == 0: raise Exception ('None of the datasets have difference values according to the specified identifiers. No plot will be displayed.')
        # Continue plotting
        plt.xlabel('Case',   fontsize=20)
        if yLim: plt.ylim(yLim)
        plt.legend(fontsize=10)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        if flSavePlot: plt.savefig(fileName+'.png')
        if flShowPlot: plt.show()
        plt.clf()
        plt.close()

    ## \brief Plots the difference values specified by the user 
    # 
    # \param[in] quantity str, default None (e.g. Refl, Trans), used to identify and collect data from the DataFrames
    # \param[in] dataIdentifiers tuple of strings, default None (e.g. ('Sign', 'Rel')), strings to search for in the row indexes of the datasets 
    # \param[in] fileName str, default 'differences', used to specify file name of plots generated
    # \param[in] flShowPlot bool, default True, used to display plot to user after calculations
    # \param[in] flSavePlot bool, default True, used to automatically save generated plots to a .png file at the location of the driver script
    # \param[in] yLim tuple of floats, default None (e.g. (-1, 1.5)), sets the y-axis limits of the plot
    # \param[in] plotTitle str, default None, allows user to specify a title for the plot (otherwise one will be generated automatically)
    # \returns Displays and saves a difference plot dependent on user specifications
    def plotDifferencesAcrossCases(self, quantity=None, dataIdentifiers=None, fileName='differences', flShowPlot=True, flSavePlot=True, yLim=None, plotTitle=None): 
        if quantity == None: raise Exception ('Please specify which quantity you want to plot (e.g. \'Refl\' or \'Trans\').')
        
        plt.figure()
        xAxis = self.comparison[next(iter(self.comparison))].columns.tolist()
        markerCounter = 0
        for key in self.comparison:
            # Slice comparison DataFrames based on user parameters to prepare for plotting 
            diffRows    = self.comparison[key][self.comparison[key].index.str.contains(quantity)  & self.comparison[key].index.str.contains('Diff') & ~self.comparison[key].index.str.contains('Unc')]
            diffUncRows = self.comparison[key][self.comparison[key].index.str.contains(quantity)  & self.comparison[key].index.str.contains('Diff') &  self.comparison[key].index.str.contains('Unc')]
            # Filter rows based on dataIdentifiers and update diffRows/diffUncRows  
            diffCondition    = pd.Series(True, index=diffRows.index)
            diffUncCondition = pd.Series(True, index=diffUncRows.index)
            if not dataIdentifiers: continue 
            else: 
                for stringVar in dataIdentifiers: 
                    diffCondition    &= diffRows.index.str.contains(stringVar)
                    diffUncCondition &= diffUncRows.index.str.contains(stringVar)
                diffRows    = diffRows[diffCondition]
                diffUncRows = diffUncRows[diffUncCondition]

            # Adding error bars & plotting 
            if diffRows.empty: warnings.warn(f"There are no {quantity} Difference values in the {key} dataset with the dataIdentifiers: {dataIdentifiers}.") 
            else: 
                for idx, row in diffRows.iterrows(): 
                    if idx+" Unc" in diffUncRows.index: yerr = diffUncRows.loc[idx+" Unc"].tolist()
                    else                              : yerr = None
                    expandedIdx = " ".join([self.shorthand_mapping.get(word, word) for word in idx.split()]) # Expands the short-hand version of the legend
                    plt.errorbar(xAxis, row, yerr=yerr, marker=self.markers[markerCounter % len(self.markers)], markersize=5, label=f'{key} - {expandedIdx}') 
                    markerCounter += 1
                if plotTitle: plt.title(plotTitle,           fontsize=25)
                else        : 
                    title = quantity+' '+'Diff'
                    expandedTitle = " ".join([self.shorthand_mapping.get(word, word) for word in title.split()]) # Expands the short-hand version of the title
                    plt.title(expandedTitle, fontsize=25)
                plt.ylabel('Difference Values', fontsize=20)

        # Check if any data points have been plotted
        check = plt.gca()
        if len(check.lines) == 0: raise Exception ('None of the datasets have difference values according to the specified identifiers. No plot will be displayed.')
        # Continue plotting
        first, last = self.comparison[next(iter(self.comparison))].columns[0], self.comparison[next(iter(self.comparison))].columns[-1]
        plt.plot([first,last],[0,0],color='black',linestyle='--',linewidth=0.8)
        plt.xlabel('Case',   fontsize=20)
        if yLim: plt.ylim(yLim)
        plt.legend(fontsize=10)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        if flSavePlot: plt.savefig(fileName+'.png')
        if flShowPlot: plt.show()
        plt.clf()
        plt.close()

    ## \brief Save data as a .csv file
    #
    #  If no dataIdentifiers are specified, the whole dataset will be saved to a .csv file.
    # 
    # \param[in] type string, default 'comparison', determines whether to search in self.comparison or self.metrics
    # \param[in] dataset string, default None, finds the key of the dataset within the dictionary
    # \param[in] fileName str, default None, allows user to specify a title for the .csv file (otherwise one will be generated automatically)
    # \param[in] dataIdentifiers tuple of strings, default None, optionally used to specify rows of data the user wants and filters out the rest
    # \param[in] flPrintData bool, default False, whether to print specified data to screen
    # \returns Prints and saves a comparison dataset to a .csv file  
    def datasetToCSV(self, type='comparison', dataset=None, fileName=None, dataIdentifiers=None, flPrintData=False):
        if type not in ('comparison', 'metrics'): raise Exception('\'type\' must either be \'comparison\' or \'metrics\'.')
        if type == 'comparison':                  dictionary = self.comparison
        else:                                     dictionary = self.metrics
        if dataset not in dictionary:             raise Exception(f'{dataset} does not exist in the dictionary.')
 
        # Separate values and uncertainty values in specified DataFrame
        rows       = dictionary[dataset][~dictionary[dataset].index.str.contains('Unc')]
        uncRows    = dictionary[dataset][ dictionary[dataset].index.str.contains('Unc')]
        # Filter rows and uncRows based on dataIdentifiers  
        if dataIdentifiers:
            condition    = pd.Series(True, index=rows.index)
            uncCondition = pd.Series(True, index=uncRows.index)
            for stringVar in dataIdentifiers: 
                condition    &= rows.index.str.contains(stringVar)
                uncCondition &= uncRows.index.str.contains(stringVar)
            rows    = rows[condition]
            uncRows = uncRows[uncCondition] 
        # Concatenate rows and uncRows then save to a .csv file
        result = pd.concat([rows, uncRows], axis=0)
        if flPrintData: print(f'{dataset}: \n', result)
        if fileName == None: fileName = dataset+'_'+type
        if not result.empty: result.to_csv(fileName+'.csv')
        else: warnings.warn("There are no values for the specified data identifiers--CSV file not created.")