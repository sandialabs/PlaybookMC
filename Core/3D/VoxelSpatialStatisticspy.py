#!usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy
import skimage
try:
    import h5py
    fl_h5_available = True
except:
    print("HDF5 bindings not found")
    fl_h5_available = False
   
# \brief Compute spatial statistics quantities from voxel geometry class
# \author Dan Bolintineanu, dsbolin@sandia.gov
# \author Aaron Olson, aolson@sandia.gov, aaronjeffreyolson@gmail.com
# 
# Calculates, plots, and outputs abundances, chord length distributions, and S2 autocorrelation metrics for voxel-based geometry
class StatMetrics():
        
    ## Associate voxel geometry
    def __init__(self, Geom=None):
        if Geom is not None:
            self.Geom = Geom
            self.matInds = self.Geom.matInds 
        else:
            print("Initializing StatMetrics object without an associated voxel geometry. For most uses, material indices should be read in from a material abundances text file via readMaterialAbundancesFromText.")
                   
    
    # \brief Calculates material abundances for voxel data
    #
    # \param flVerbose bool, default False, print values to screen?
    # \returns Sets material abundances dictionary, where keys are "Material <index>", and values are material abundances
    def calculateMaterialAbundances(self,flVerbose=False):    
        assert isinstance(flVerbose,bool)
        self.MaterialAbundances = {}       
        for matIndex in self.matInds:
            self.MaterialAbundances[matIndex] = np.sum(self.Geom.VoxelMatInds==matIndex)/self.Geom.VoxelMatInds.size
        if flVerbose: print(f"Computed material abundances: {self.MaterialAbundances}")
    
    # \brief Writes material abundances to text file
    #
    # \param [in]: string; name of text file
    # \returns Output of material abundances to text file
    def writeMaterialAbundancesToText(self, filename):
        with open(filename, "w") as f:
            for k,v in self.MaterialAbundances.items():
                f.write(f"Material {k}: {v}\n")            
    
    # \brief Reads material abundances from text file
    #
    # \param [in]: string; name of text file
    # \returns Sets self.MaterialAbundances dictionary
    def readMaterialAbundancesFromText(self, filename):
        self.MaterialAbundances = {}   
        flSetMatInds = False
        if not hasattr(self, 'matInds'):
            self.matInds = []
            flSetMatInds = True
        with open(filename, "r") as f:
            for line in f.readlines():
                words = line.strip().split()   
                matIndex = int(words[1][:-1])
                if flSetMatInds:
                    self.matInds.append(matIndex)
                self.MaterialAbundances[matIndex] = float(words[-1])
    

    # \brief Calculates chord lengths for voxel structures
    #
    # This function outputs mean chord length and number of chords for all materials in all directions.
    # Chords that cross the domain edges can optionally be excluded.
    # Histograms of the chord lengths for each material in each direction can also be optionally returned.
    # The results are stored in a dictionary, where keys indicate material indices, axis directions, and relevant quantities. For keys 
    # with no direction, results correspond to chord statistics across all three axis directions.
    # Values in the results dictionary are scalars for 'mean' and 'num_chords', corresponding to the mean chord length in the given direction.
    # If the 'flReturnChordLengthPDFs' flag is True, then a tuple is also returned that stores (chord length values, relative frequencies), i.e.
    # a histogram of chord lengths. 
    # For example:
    #    results["average_chord_length_material1_x"] stores the average chord length for material 1 in the x direction
    #    results["average_chord_length_material2"] stores the average chord length for material 2 across all directions
    #    results["chord_length_pdf_material1_x"] stores the (lengths, relative frequencies) for all chords in material 1 in the x direction
    #    results["chord_length_pdf_material1"] stores the (lengths, relative frequencies) for all chords in material 1 across all direction
    # etc.
    #   
    # \param[in] flExcludeEdgeChords, bool, Indicate whether or not to ignore chords that touch domain edges. 
    #            Default is True (ignore/exclude)
    # \param[in] flReturnChordLengthPDFs, bool, Indicate whether to return histograms (probability density function estimates)
    # \param flVerbose bool, default False, print values to screen?
    #
    # \returns results, dictionary of all results. Keys indicate material indices, axis directions, and relevant quantities, values are scalar quantities or tuples of arrays 
    def calculateChordLengthDistributions(self, flExcludeEdgeChords=True, flReturnChordLengthPDFs=True, flVerbose=False):
        assert isinstance(flExcludeEdgeChords,bool)
        assert isinstance(flReturnChordLengthPDFs,bool)
        assert isinstance(flVerbose,bool)
        arr = self.Geom.VoxelMatInds
    
        if np.any(np.array(arr.shape) == 1):
            raise Exception("At least one axis has dimension of one")
        results = {}
        tstart = time.time()

        selem_dict = {} #Dictionary of structuring elements for connected components labeling of chords in x/y/z
        for axis in ["x", "y", "z"]:
            s = np.zeros((3,3,3))
            if   axis == "x": s[0,1,1] = s[2,1,1] = 1        
            elif axis == "y": s[1,0,1] = s[1,2,1] = 1        
            elif axis == "z": s[1,1,0] = s[1,1,2] = 1                
            selem_dict[axis] = s

        all_lengths = []
        for mat in self.matInds:                
            all_lengths_mat = []                    
            all_counts_mat = 0
            for axis, voxelSize in zip(["x", "y", "z"], [self.Geom.voxelSizeX, self.Geom.voxelSizeY, self.Geom.voxelSizeZ]) :
                print(f"Calculating chord lengths for material {mat}, {axis} direction")
                s = selem_dict[axis]
                labels, _ = scipy.ndimage.label(arr == mat, structure=s)
                if flExcludeEdgeChords: chords = skimage.segmentation.clear_border(labels)
                else                  : chords = labels
                regprops = skimage.measure.regionprops(chords)
                lengths = np.array([r.filled_area for r in regprops])*voxelSize           
                all_lengths.append(lengths)
                all_lengths_mat.append(lengths)
                all_counts_mat += len(lengths)
        
                results[f"average_chord_length_material{mat}_{axis}"] = np.mean(lengths)
                results[f"number_of_chords_material{mat}_{axis}"] = len(lengths)

                if flReturnChordLengthPDFs:
                    length_values, counts = np.unique(lengths, return_counts=True)
                    results[f"chord_length_pdf_material{mat}_{axis}"] = (length_values, counts)

            #Also return quantities across all directions
            all_lengths_mat = np.hstack(all_lengths_mat)
            results[f"average_chord_length_material{mat}_all_directions"] = np.mean(all_lengths_mat)
            results[f"number_of_chords_material{mat}_all_directions"] = all_counts_mat
            if flReturnChordLengthPDFs:
                length_values, counts = np.unique(all_lengths_mat, return_counts=True)
                results[f"chord_length_pdf_material{mat}_all_directions"] = (length_values, counts)
            
        #Add zero-count entries for all unique chord length values, where absent
        all_lengths = np.hstack(all_lengths)
        for axis in ["x", "y", "z", "all_directions"]:
            for mat in self.matInds:
                length_values, counts = results[f"chord_length_pdf_material{mat}_{axis}"]
                length_values_absent = np.setdiff1d(all_lengths, length_values)
                counts_absent = np.zeros_like(length_values_absent)
                new_length_values = np.hstack([length_values, length_values_absent])
                new_counts = np.hstack([counts, counts_absent])
                sortargs = np.argsort(new_length_values)
                results[f"chord_length_pdf_material{mat}_{axis}"] = [new_length_values[sortargs], new_counts[sortargs]]
        self.CLDResults = results
        
        #Populate additional dictionaries for convenience
        self.AverageChordLengthOverall = {}
        self.AverageChordLengthX = {}
        self.AverageChordLengthY = {}
        self.AverageChordLengthZ = {}
        for mat in self.matInds:
            self.AverageChordLengthOverall[f"Material {mat}"] = self.CLDResults[f"average_chord_length_material{mat}_all_directions"]
            self.AverageChordLengthX[f"Material {mat}"] = self.CLDResults[f"average_chord_length_material{mat}_x"]
            self.AverageChordLengthY[f"Material {mat}"] = self.CLDResults[f"average_chord_length_material{mat}_y"]
            self.AverageChordLengthZ[f"Material {mat}"] = self.CLDResults[f"average_chord_length_material{mat}_z"]

        runtime = (time.time()-tstart)/60.0
        print(f"Chord length calculation took a total of {runtime:5.2f} minutes.")
        if flVerbose: print(f"Computed average chord lengths:{self.AverageChordLengthOverall}")

    # Write chord length summary data to a text file.
    #                
    # Write the summary of chord length calculation results to a single text file. This includes the average chord length and
    # the number of chords for each material in each direction.
    #
    # \param [in] fileName Name of text file. Typically with a .txt extension, but not required   
    # \returns Writes chord length summary data to text file.
    def writeCLDSummaryToText(self, fileName):
        with open(fileName, "w") as f:
            for i in self.matInds:
                for direction in ["x", "y", "z", "all_directions"]:                    
                    key = f"average_chord_length_material{i}_{direction}"
                    if direction == "all_directions": direction_string = "all directions"
                    else                            : direction_string = f"{direction} direction"
                    f.write(f"Mean chord length in material {i}, {direction_string}: = {self.CLDResults[key]}\n")
                    key = f"number_of_chords_material{i}_{direction}"
                    f.write(f"Number  of chords in material {i}, {direction_string}: = {self.CLDResults[key]}\n")                    

    # Write chord length distributions to a set of CSV files.
    #                
    # This functions writes the chord length vs count data for all directions, one file per direction.
    #
    # The user-specified fileNamePrefix determines the file name pattern, as follows: if the user enters a string that 
    # ends with .csv, this is removed. Files are then generated with names such as fileNamePrefix_cld_x.csv, 
    # fileNamePrefix_cld_y.csv, fileNamePrefix_cld_z.csv, fileNamePrefix_cld_all_directions.csv 
    # Each file contains five columns, one for the chord length values and the others for the total counts in x,y,z and overall
    #
    # \param [in] fileNamePrefix Prefix for CSV files.     
    # \returns Writes chord length probability distribution data to CSV files
    def writeCLDToCSV(self, fileNamePrefix):
        if fileNamePrefix[-4:] == ".csv":
            fileNamePrefix = fileNamePrefix[:-4]
        
        for direction in ["x", "y", "z", "all_directions"]:
            arr = []
            fmt_list = [] #For formatting the file
            with open(f"{fileNamePrefix}_cld_{direction}.csv", "w") as f:            
                for mat in self.matInds:                               
                    if mat == self.matInds[0]:
                        f.write(f"Chord length, ")
                        fmt_list.append("%13.4f")
                        arr.append(self.CLDResults[f"chord_length_pdf_material{mat}_{direction}"][0])                                    
                    if mat == self.matInds[-1]: f.write(f"Count in mat. {mat}")
                    else                      : f.write(f"Count in mat. {mat}, ")
                    fmt_list.append("%15d")
                    arr.append(self.CLDResults[f"chord_length_pdf_material{mat}_{direction}"][1])                           
                f.write("\n")
                arr = np.vstack(arr).T
                np.savetxt(f, arr, fmt=fmt_list, delimiter=", ")                      

    # Read chord length distributions from a set of CSV files.
    #                
    # This functions reads the chord length vs count data for all combinations of materials and directions from CSV files
    # produced by the writeCLDToCSV function.   
    #
    # The user-specified fileNamePrefix should match the corresponding call to writeCLDToCSV that generated the files to be read, i.e.
    # files should have the pattern fileNamePrefix_cld_x.csv, fileNamePrefix_cld_y.csv, etc.
    #
    # \param [in] fileNamePrefix . Prefix for CSV files. Must match call to writeCLDToCSV that generated the files of interest.     
    # \returns Reads chord length probability distribution data from CSV files
    def readCLDFromCSV(self, fileNamePrefix):
        if not hasattr(self, 'CLDResults'):
            self.CLDResults = {}
        for direction in ["x", "y", "z", "all_directions"]:            
            with open(f"{fileNamePrefix}_cld_{direction}.csv", "r") as f:
                arr = np.loadtxt(f, skiprows=1, delimiter=",")
                for i,mat in enumerate(self.matInds):
                    self.CLDResults[f"chord_length_pdf_material{mat}_{direction}"] = (arr[:,0].astype(np.float32), arr[:,1+i].astype(np.int32))
        
     
    # Read chord length summary data from a text file.
    #                
    # Read the summary of chord length calculation results from a single text file. This includes the average chord length and
    # the number of chords for each material in each direction.
    #
    # \param [in] fileName .. Name of text file. 
    # \returns Reads chord length summary data from text file, sets self.CLDResults
    def readCLDSummaryFromText(self, fileName):
        if not hasattr(self, 'CLDResults'):
            self.CLDResults = {}
        with open(fileName, "r") as f:
            for i in self.matInds:
                for direction in ["x", "y", "z", "all_directions"]:                    
                    key = f"average_chord_length_material{i}_{direction}"
                    result = float(f.readline().strip().split()[-1])
                    self.CLDResults[key] = result

                    key = f"number_of_chords_material{i}_{direction}"
                    result = int(f.readline().strip().split()[-1])
                    self.CLDResults[key] = result

        #Populate additional dictionaries for convenience
        self.AverageChordLengthOverall = {}
        self.AverageChordLengthX = {}
        self.AverageChordLengthY = {}
        self.AverageChordLengthZ = {}
        for mat in self.matInds:
            self.AverageChordLengthOverall[f"Material {mat}"] = self.CLDResults[f"average_chord_length_material{mat}_all_directions"]
            self.AverageChordLengthX[f"Material {mat}"] = self.CLDResults[f"average_chord_length_material{mat}_x"]
            self.AverageChordLengthY[f"Material {mat}"] = self.CLDResults[f"average_chord_length_material{mat}_y"]
            self.AverageChordLengthZ[f"Material {mat}"] = self.CLDResults[f"average_chord_length_material{mat}_z"]
    
    # Plot summary of chord length distributions.
    #
    # This will plot the directionally-dependent and overall chord length distributions.
    # The figure generated has as many subplots as materials, with each subplot containing x, y, z and 
    # overall chord length distributions for the given material.
    # 
    #
    # \param [in] outputfilename, string; name of output file to save figure to, should include appropriate image format extension. If not specified, the figure is not saved.
    # \param [in] figWidth, float; width of figure in inches
    # \param [in] figHeight, float; height of figure in inches
    # \param [in] plotStyle, string; One of 'nonzero', 'all', or 'binned'.
    #                'all': Plots include all chord length values, including those with zero counts in each direction
    #                'nonzero': Plots include only chord length values that have nonzero counts in each direction. This is the default.
    #                'binned' : Plots are based on histograms, with 'bins' carrying through to the numpy.histogram 'bins' parameter
    # \param [in] numBins, int or list of float or string; Carries through to the 'bins' parameter of numpy.histogram if plotStyle is 'binned'. If an integer, this is the number of bins (smaller number is coarser binning)
    # \param [in] flNormalizeCounts, bool; Indicate whether or not data are normalized by total number of chord counts in each direction. Default is False.
    # \param [in] flShowPlot, bool; Show plot when run method?
    def plotCLDResults(self, outputfilename=None, figWidth=12, figHeight=4, plotStyle="nonzero", numBins=20, flNormalizeCounts=False,flShowPlot=False):
        assert isinstance(figWidth,int) and figWidth>0
        assert isinstance(figHeight,int) and figHeight>0
        assert plotStyle in {'all','nonzero','binned'}
        if plotStyle=='binned': assert isinstance(numBins,int) and numBins>0
        assert isinstance(flNormalizeCounts,bool)
        assert isinstance(outputfilename,str) or outputfilename==None
        assert isinstance(flShowPlot,bool)
        fig, axs = plt.subplots(1, len(self.matInds), figsize=(figWidth, figHeight))
        for i, ax in enumerate(axs):
            for direction in ["x", "y", "z", "all_directions"]:
                key = f"chord_length_pdf_material{self.matInds[i]}_{direction}"
                lvals, counts = self.CLDResults[key]
                if plotStyle == "all":
                    x = lvals
                    y = counts
                elif plotStyle == "nonzero":
                    x = lvals[counts != 0]
                    y = counts[counts != 0]                    
                elif plotStyle == "binned":
                    all_values = np.repeat(lvals, counts.astype(np.int32))
                    binvals, binedges = np.histogram(all_values, bins = numBins)
                    y = binvals
                    x = 0.5*(binedges[1:]+binedges[:-1])
                if flNormalizeCounts:
                    y = y/np.sum(y)
                if direction == "all_directions":
                    linewidth = 2
                    color = 'k'
                    label = "all directions"
                else:
                    linewidth = 1
                    color = None
                    label = direction
                ax.plot(x, y, '--x', linewidth=linewidth, color=color, label=label)
                ax.legend()
                ax.set_title(f"Material {self.matInds[i]}")
                ax.set_xlabel("Chord length (length units)")
                if flNormalizeCounts: ax.set_ylabel("Freq.")
                else              : ax.set_ylabel("Counts")
        
        plt.tight_layout()
        if flShowPlot: plt.show()
        if outputfilename is not None: plt.savefig(outputfilename)
        plt.close()


    # \brief Calculates radial average of 3D array data, centered at the array center
    #
    # This is used primarily for calculating radially averaged S2.
    #
    # \returns r, a(r), both as np.arrays. r contains the radial distance values, a(r) contains the average values of the array 
    #  at the corresponding radial distances.
    @staticmethod
    def _radial_profile(data):
        data = np.array(data)
        center = np.array(data.shape)//2
        inds = np.indices((data.shape))
        r = np.sqrt((inds[2] - center[2])**2 + 
                    (inds[1] - center[1])**2 +
                    (inds[0] - center[0])**2)        
        
        radvals, inverse_inds, counts = np.unique(r.ravel(), return_inverse=True, return_counts=True)
        sum_by_r = np.bincount(inverse_inds, weights=data.ravel())
        radialprofile = sum_by_r/np.where(counts == 0, 1, counts)        
        return radvals, radialprofile
    
    # \brief Calculates S2 for voxel data
    #
    # The S2_i, or two-point correlation function, is most commonly defined as the probability that two points separated by a distance r both belong
    # to a given material index i. A slightly more general definition expands this definition to different
    # combinations of materials as well as different vector directions. 
    # For instance, the common definition above, denoted S2_1_1_radial(r), is the probability that two points separated by distance r
    # both belong to material index 1. 
    # An example of directional dependence as well as inclusion of other material index combinations is S2_0_1_x(x), which
    # denotes the probability that for two points separated by distance x in the x direction, one point belongs to material 0, 
    # the other to material 1 (the ordering is not relevant in a stationary medium, i.e. S2_0_1_x = S2_1_0_x). 
    # For more information/background, see e.g. Jiao, Yang, F. H. Stillinger, and S. Torquato. "Modeling 
    # heterogeneous materials via two-point correlation functions: Basic principles." Physical Review E, 76.3 (2007): 031110)).
    #
    # This function computes all such possibilites of materials, including radially averaged quantities, dependence on coordinate axis directions x,
    # and optionally full vector dependence. Results are stored in a Python dictionary, with keys indicating the relevant material indices and direction, 
    # ("<index1>_<index2>_<direction>", e.g. "1_1_radial" denotes the S2 in the first example above, "0_1_x" denotes the S2 in the second example). Values in the
    # dictionary for "x","y","z" and "radial" are tuples, where the first entry in the tuple is an array of length values (either radial distances, or distances in 
    # the relevant axis coordinate direction), and the second entry in the tuple stores the correspoding S2 values. The exception to this is the (optional) "vector" key, 
    # which stores only the full vector array, such that a given entry in the array represents the vector with origin at the array center, i.e. (arr.shape//2)
    #
    # \param[in] flStoreFullVectorS2, bool, Indicate whether to store the full vector S2 arrays. These are arrays of the same size
    #            as the voxelMatInds array, and can result in very large datasets for large voxel structures.
    #
    # \returns Sets S2, a dictionary where keys are material indices and direction ("<index>_<index>_<direction>"), and values are
    # tuples of numpy arrays, with first entry storing distances, second entry storing S2 values at distances.
    def calculateS2(self, flStoreFullVectorS2=False):    
        assert isinstance(flStoreFullVectorS2,bool)
        arr = self.Geom.VoxelMatInds

        import scipy
        fft = scipy.fft.fftn
        ifft = scipy.fft.ifftn
        
        if not np.isclose(self.Geom.voxelSizeX, self.Geom.voxelSizeY) or not np.isclose(self.Geom.voxelSizeY, self.Geom.voxelSizeZ):
            print("S2 calculation for non-cubic voxels not currently implemented, results for radial profiles will not be accurate")

        tstart = time.time()

        self.S2Results = {}        
        for i in self.matInds:
            ai = (arr == i).astype(np.uint8)
            aifft = fft(ai)
            for j in self.matInds:
                if i > j: #Since S2_i_j = S2_j_i, only calculate S2_i_j for i <=j
                    continue
                aj = (arr == j).astype(np.uint8)
                print(f"Calculating S2 for material combination {i}, {j}")

                if i == j: #Avoid duplicating FFT for performance
                    ajfft = aifft
                else:
                    ajfft = fft(aj)
                c0 = np.real(ifft(aifft*np.conjugate(ajfft)))/aifft.size
                s2vec = np.fft.fftshift(c0)
                s2x = s2vec[s2vec.shape[0]//2:, s2vec.shape[1]//2, s2vec.shape[2]//2]
                self.S2Results[f"S2_{i}_{j}_x"] = (np.arange(0,len(s2x))*self.Geom.voxelSizeX, s2x)
                s2y = s2vec[s2vec.shape[0]//2, s2vec.shape[1]//2:, s2vec.shape[2]//2]
                self.S2Results[f"S2_{i}_{j}_y"] = (np.arange(0,len(s2y))*self.Geom.voxelSizeY, s2y)
                s2z = s2vec[s2vec.shape[0]//2, s2vec.shape[1]//2, s2vec.shape[2]//2:]
                self.S2Results[f"S2_{i}_{j}_z"] = (np.arange(0,len(s2z))*self.Geom.voxelSizeZ, s2z)
                r, s2r = self._radial_profile(s2vec)
                self.S2Results[f"S2_{i}_{j}_radial"] = [r*self.Geom.voxelSizeX, s2r]
                if flStoreFullVectorS2: self.S2Results[f"S2_{i}_{j}_vector"] = s2vec  
        runtime = (time.time()-tstart)/60.0
        print(f"S2 calculation took a total of {runtime:5.2f} minutes.")

    # Plot summary of two-point correlation functions
    #
    # This will plot the directionally-dependent and radially averaged S2s for all combinations of materials.
    # The figure generated has number_of_materials X number_of_materials subplots, with each subplot containing x, y, z and 
    # radially-averaged S2.
    #
    # \param [in] outputfilename, string; Name of output file to save figure to, should include appropriate image format extension. If not specified, the figure is not saved.
    # \param [in] figWidth, float; Width of figure in inches
    # \param [in] figWidth, float; Height of figure in inches
    # \param [in] flPlotLimits, bool; Indicate whether to add labels to limits of S2 at r=0 and r->infinity. Default False.
    # \param [in] flPlotStandardYAxis, bool; Make all subplots have same y-axis limits?
    # \param [in] flShowPlot, bool; Show plot when run method?
    def plotS2Results(self, outputfilename=None, figWidth=12, figHeight=10, flPlotLimits=False, flPlotStandardYAxis=False,flShowPlot=True):
        assert isinstance(figWidth,int) and figWidth>0
        assert isinstance(figHeight,int) and figHeight>0
        assert isinstance(flPlotLimits,bool)
        assert isinstance(flPlotStandardYAxis,bool)
        assert isinstance(outputfilename,str) or outputfilename==None
        assert isinstance(flShowPlot,bool)
        fig, axs = plt.subplots(len(self.matInds), len(self.matInds), figsize=(figWidth, figHeight))
        xlim = None    
        for i,imat in enumerate(self.matInds):
            for j,jmat in enumerate(self.matInds):
                if i > j: continue
                for ax in [axs[i,j], axs[j,i]]:
                    try: #In case only radially-averaged S2s are available
                        x, s2x = self.S2Results[f"S2_{imat}_{jmat}_x"]
                        ax.plot(x, s2x, label="x")
                        y, s2y = self.S2Results[f"S2_{imat}_{jmat}_y"]
                        ax.plot(y, s2y, label="y")
                        z, s2z = self.S2Results[f"S2_{imat}_{jmat}_z"]
                        ax.plot(z, s2z, label="z")
                        xlim = np.max([x.max(), y.max(), z.max()])
                    except Exception as e:
                        print(e)               
                    r, s2r = self.S2Results[f"S2_{imat}_{jmat}_radial"]
                    ax.plot(r, s2r, label="radial", color='k', linewidth=2)
                    if flPlotLimits:
                        xlims = ax.get_xlim()
                        if not hasattr(self,'MaterialAbundances'): self.calculateMaterialAbundances()
                        phi2 = self.MaterialAbundances[i]*self.MaterialAbundances[j]
                        if i == j:
                            phi = self.MaterialAbundances[i]
                            ax.plot(xlims, [phi2, phi2], '--k', label=f"$\\lim_{{r \\rightarrow \\infty}}\\.S_2 = \\phi_{{{imat}}}^2 = $"+f"{phi2:.3f}", alpha=0.5)
                            ax.plot(xlims, [phi, phi], '--r', label=f"$\\lim_{{r \\rightarrow 0}}\\.S_2 = \\phi_{{{imat}}} = $"+f"{phi:.3f}", alpha=0.5)
                        else:
                            ax.plot(xlims, [phi2, phi2], '--k', label=f"$\\lim_{{r \\rightarrow \\infty}}\\.S_2 = \\phi_{{{imat}}}\\phi_{{{jmat}}} = $"+f"{phi2:.3f}", alpha=0.5)
                        ax.set_xlim(xlims)
                    ax.set_title(f"$S_2$, {imat}-{jmat}")
                    ax.set_xlabel("Distance (length units)")
                    ax.set_xlim([0, xlim])
                    if flPlotStandardYAxis: ax.set_ylim([0,max(self.MaterialAbundances.values())*1.1])
                    ax.set_ylabel("S$_2$")
                    ax.grid(linestyle='--', linewidth=0.5, color='gray')
                    ax.legend()
                    if i == j: break
        plt.tight_layout()
        if flShowPlot: plt.show()
        if outputfilename is not None: plt.savefig(outputfilename)
        plt.close()

    # Fit splines to radial averages of S2
    #
    # This generates an array of callables, where each entry i,j is a callable (spline fit) that returns the radially-averaged
    # S2 for materials i,j, with distance argument r. This is the format required by MICK CPF predictions.
    #
    # \returns Sets the S2SplineFitsToRadialAverages array
    def fitSplinesToS2RadialAverage(self):
        import scipy.interpolate        
        nmat = len(self.matInds)
        s2_splines = np.zeros((nmat, nmat)).astype(object)
        for i in self.matInds:
            for j in self.matInds:                
                r, s2r = self.S2Results[f"S2_{min((i,j))}_{max([i,j])}_radial"]                                    
                nmat = len(self.matInds)
                spl = scipy.interpolate.UnivariateSpline(r, s2r, s=0)
                s2_splines[i, j] = s2_splines[j,i] = spl
        self.SplineFitsToS2RadialAverages = s2_splines

    # Write x,y,z and radially-averaged S2 for all combinations of materials to a set of CSV files.
    #                
    # Since the number of rows is potentially different for each direction, four separate CSV files are generated: one for 
    # each direction x, y, z, and a fourth for the radial average. The user-specified fileNamePrefix determines the file name pattern,
    # as follows: if the user enters a string that ends with .csv, this is removed. Files are then generated with names such as 
    # fileNamePrefix_s2_x.csv, fileNamePrefix_s2_y.csv, fileNamePrefix_s2_z.csv, fileNamePrefix_s2_radial.csv. 
    # Each file contains columns for the distance and the corresponding S2 values, for all material combinations. Material combination
    # labels such as S2_1_1_x, etc., are printed in the first row of the CSV file.
    #
    # \param [in] fileNamePrefix Prefix for CSV files. 
    #
    # \returns Writes S2 data to CSV files
    def writeS2ToCSV(self, fileNamePrefix):
        if fileNamePrefix[-4:] == ".csv":
            fileNamePrefix = fileNamePrefix[:-4]
        for direction in ["x", "y", "z", "radial"]:
            arr = []
            with open(f"{fileNamePrefix}_s2_{direction}.csv", "w") as f:
                for i in self.matInds:
                    for j in self.matInds:
                        if i > j: #Since S2_i_j = S2_j_i, only calculate S2_i_j for i <=j
                            continue
                        if i == self.matInds[0] and i == j:                            
                            f.write(f"distance, ")
                            arr.append([self.S2Results[f"S2_{i}_{j}_{direction}"][0]])
                        f.write(f"S2_{i}_{j}_{direction}, ")
                        arr.append(self.S2Results[f"S2_{i}_{j}_{direction}"][1])
                        
                f.write("\n")                
                arr = np.vstack(arr).T
                np.savetxt(f, arr, fmt='%10.5f', delimiter=", ")

    
    # Read x,y,z and radially-averaged S2 for all combinations of materials from a set of CSV files.
    #                
    # Since the number of rows is potentially different for each direction, four separate CSV files are needed.
    #
    # \param [in] fileNamePrefix Prefix for CSV files. This is everything except the _s2_x.csv, _s2_y.csv etc. portion of the file name.
    #
    # \returns Sets S2Results from CSV files
    def readS2FromCSV(self, fileNamePrefix):
        if not hasattr(self, "S2Results"):
            self.S2Results = {}
        for direction in ["x", "y", "z", "radial"]:
            arr = []
            with open(f"{fileNamePrefix}_s2_{direction}.csv", "r") as f:
                keystrings = f.readline().split(", ")[:-1] #Get keys
                keys = []
                for keystring in keystrings:
                    if "S2" in keystring:
                        keys.append(keystring)

                arr = np.loadtxt(f, delimiter=",")            
                distances = arr[:,0]
                s2values = arr[:,1:]                
                for k, s2 in zip(keys, s2values.T):                
                    self.S2Results[k] = [distances, s2]

    # Read radially-averaged S2 for all combinations of materials from a CSV file.       
    #
    # This is provided to simplify input for MICK CPF calculations, which do not require directional (x,y,z) S2 results.
    #
    # \param [in] fileName. Full name of file containing radially-averaged S2.
    #
    # \returns Sets S2Results[S2_*_*_radial] from CSV file
    def readS2RadialAverageFromCSV(self, fileName):
        if not hasattr(self, "S2Results"):
            self.S2Results = {}        
        arr = []
        with open(f"{fileName}", "r") as f:
            keystrings = f.readline().split(", ")[:-1] #Get keys
            keys = []
            for keystring in keystrings:
                if "S2" in keystring:
                    keys.append(keystring)

            arr = np.loadtxt(f, skiprows=1, delimiter=",")
            distances = arr[:,0::2].T
            s2values = arr[:,1::2].T                
            for k, d, s2 in zip(keys, distances, s2values):
                self.S2Results[k] = [d, s2]

    # Write vector S2 data to HDF5 file
    #
    # The HDF5 file will contain datasets (3D arrays) for all combinations of materials. The voxel size in X/Y/Z is also stored
    # as an attribute.
    #                
    # \param [in] File name to store vector S2 data into
    # \returns Writes vector S2 data to file.
    def writeVectorS2ToHDF5(self, fileName):
        if not fl_h5_available:
            print("HDF5 not available, data not saved")
            return
        with h5py.File(f"{fileName}", "w") as f:
            for i in self.matInds:
                for j in self.matInds:
                    if i > j: #Since S2_i_j = S2_j_i, only calculate S2_i_j for i <=j
                        continue
                    try:
                        s2vec = self.S2Results[f"S2_{i}_{j}_vector"]
                    except Exception as e:
                        print(e)
                        print(f"Cannot find vector S2 for material combination {i},{j}. You may need to rerun calculateS2\
                            with the flStoreFullVectorS2=True setting")
                        return
                    f.create_dataset(name=f"S2_{i}_{j}_vector", data=s2vec)
            f.attrs["VoxelSizeX"] = self.Geom.VoxelSizeX
            f.attrs["VoxelSizeY"] = self.Geom.VoxelSizeY
            f.attrs["VoxelSizeZ"] = self.Geom.VoxelSizeZ

    # Read vector S2 data from HDF5 file
    #                    
    # \param [in] File name to read vector S2 data from
    # \returns Sets S2Results vector data
    def readVectorS2FromHDF5(self, fileName):
        if not fl_h5_available:
            print("HDF5 not available, data not read")
            return
        if not hasattr(self, "S2Results"):
            self.S2Results = {}
        with h5py.File(f"{fileName}", "r") as f:
            for i in self.matInds:
                for j in self.matInds:
                    if i > j: #Since S2_i_j = S2_j_i, only calculate S2_i_j for i <=j
                        continue
                    try:
                        s2vec = f[f"S2_{i}_{j}_vector"][:]
                        self.S2Results[f"S2_{i}_{j}_vector"] = s2vec
                    except Exception as e:
                        print(e)
                        print(f"Cannot find vector S2 for material combination {i},{j} in file")                    