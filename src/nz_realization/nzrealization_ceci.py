from ceci import PipelineStage
#from ceci_example.types import NpyFile, HDFFile
# We need to define NpyFile etc we want to use inside ceci_example.types, Or not??? get_input don't use this function and we open in run ourselves

class PreparePZRealizationsPipe(PipelineStage):
    """
    Generates LHC sampling points for the uncertainty beside sample variance and shot noise.
    Saves to a NPZfile with LHCsampling points.
    """

    name = "PreparePZRealizationsPipe"
    inputs = [("deepfield_zeropoint", NpyFile)]
    outputs = [("LHC_samples", NpyFile)]
    parallel = False
    
    config_options = {
        "shot_noise": StageParameter(bool, False,
            msg="If you want to add shot noise from the limited number of galaxies in redshift and deep sample "),
        "sample_variance": StageParameter(bool, False,
            msg="If you want to add sample variance from the limited area of redshift and deep field. Must add shot nose if you add sample variance" ),
        "photometric_zeropoint_deep": StageParameter(bool, False,
            msg="If you want to add photometric zero point uncertainty due to the deep field zero point offsets " ),
        "redshift_sample_uncertainty": StageParameter(bool, False,
            msg="The bias and uncertainty due to the use of photometric redshfit calibration sample" ), #If we use spec-only, don't need this
        "photometric_zeropoint_wide": StageParameter(bool, False,
            msg="If you want to add photometric zero point uncertainty due to the wide field zero point offsets " ),  #not implemented
        "photometric_skybackground_wide": StageParameter(bool, False,
            msg="If you want to add skybackground uncertainty on the wide field photometry " ), #not implemented       
        "num_lhc_points":StageParameter(int, 100,msg="number of lhc points we want to sample"),
        "num_3sdir":StageParameter(int, 100,msg="number of 3sdir sampling we want to do"),
    }


    def run(self):
        for inp, _ in self.inputs:
            filename = self.get_input(inp)
            print(f"    PreparePZRealizationsPipe {filename}")
            open(filename)

            




class PhotozDeepZeroPointPipe(PipelineStage):
    """
    Generates redshift realizations, where each is a possible redshift distribution given the uncertainty.
    Use SOM products and modeled uncertainty and saves to a HDFFile.
    """

    name = "PZRealizationsPipe"
    inputs = [("deep_balrog_file", ParquetFile), ("deep_file", ParquetFile), ("deep_som", PklFile), ("LHC_samples", NpyFile)]
    outputs = [("deep_cell_assignmemnt_balrog_files_withzp", NPZFile)]
    parallel = False
    
    config_options = {}


    def run(self):
        for inp, _ in self.inputs:
            filename = self.get_input(inp)
            print(f"    PZRealizationPipe reading from {filename}")
            open(filename)

            deep_data = 
            assign_som_zpu(deep_data, balrog_data, deep_som, LHC_samples, output_file):



class RunPZRealizationsPipe(PipelineStage):
    """
    Generates redshift realizations, where each is a possible redshift distribution given the uncertainty.
    Use SOM products and modeled uncertainty and saves to a HDFFile.
    """

    name = "RunPZRealizationsPipe"
    inputs = [("deep_balrog_file", ParquetFile), ("redshift_deep_balrog_file", ParquetFile), ("deep_som", PklFile), ("wide_som", PklFile), ("pchat", HDFFile), ("pcchat", HDFFile), ("tomo_bin_assignment", HDFFile), ("deep_cells_assignment_balrog_file", HDFFile)]
    outputs = [("photoz_realizations", ParquetFile)]
    parallel = False
    
    config_options = {   
    }


    def run(self):
        for inp, _ in self.inputs:
            filename = self.get_input(inp)
            print(f"    PZRealizationPipe reading from {filename}")
            open(filename)

            #open all the input files

            nz_realizations = get_realizations(input_files)


