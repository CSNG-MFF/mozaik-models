# LSV1M 

This repository contains the Cagnol2025 model implemented in the Mozaik framework, that was published by the [Computational Systems Neuroscience Group](http://csng.mff.cuni.cz/) at the Faculty of Mathematics and Physics, Charles University in [Cagnol, R., Antolík, J., Palmer L.A., Contreras D. (2025). Non-linearity of spatial integration varies across layers of primary visual cortex the unifying mechanism of spontaneous, visually and optogenetically driven V1 dynamics](https://www.biorxiv.org/content/10.1101/2025.04.10.648107v1.full). Setup used in the preprint: 16 processes on a EPYC 7302 CPU with 128 GB of RAM.

To run the models present in this repository one must first install the Mozaik package and its dependencies (see [here](https://github.com/CSNG-MFF/mozaik) for more information and detailed installation instructions).

## Instructions

    1 Running the different experiments:

        First, in run_parameter_search.py, replace the field [PATH_TO_ENV] by the path of the activate executable of your virtual environment (for example, $HOME/virt_env/mozaik/bin/activate)

            - Disks and Rings protocol as presented in Cagnol2025 preprint::

                python run_parameter_search.py run_disk_ring.py nest param/defaults (~7h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                    mpirun -n 16 python run_disk_ring.py nest 1 param/defaults SelfSustainedPushPull


            - Drifting grating and natural images protocol::

                python run_parameter_search.py run.py nest param/defaults (~10h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                    mpirun -n 16 python run.py nest 1 param/defaults SelfSustainedPushPull

            - Size tuning protocol::

                python run_parameter_search.py run_stc.py nest param/defaults (~10h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                    mpirun -n 16 python run_stc.py nest 1 param/defaults SelfSustainedPushPull

            - Spontaneous activity protocol::

                     python run_parameter_search.py run_spont.py nest param_spont/defaults (~1h30 of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                     mpirun -n 16 python run_spont.py nest 1 param_spont/defaults SelfSustainedPushPull



    2 Description of the files:
    
        - analysis_and_visualization.py: Contains the code for the different analysis and plotting. The function `perform_analysis_and_visualization` runs the analysis and the plots corresponding to the fullfield drifting grating and natural images protocol. The functions `perform_analysis_and_visualization_stc` and `perform_analysis_and_visualization_spont` do the same respectively for the size tuning and spontaneous activity protocol.
        - experiments.py: Defines for each protocol which experiments will be run as well as their parameters. `create_experiments` corresponds to the fullfield drifting grating protocol, `create_experiments_stc` corresponds to the size tuning protocol, and `create_experiments_spont` corresponds to the spontaneous activity protocol.
        - eye_path.pickle: Contains the coordinates of the eye path that is used by default in the natural image protocol. 
        - image_naturelle_HIGHC.bmp: The natural image that is used by default in the natural image protocol.
        - model.py: Contains the code which creates each layers of the model and build the connections based on the parameters used to run the model.
        - NewValuesLI.txt: Contains experimental data for Linearity Indices, as well as Non-Linearity Indices for firing rates.
        - NewValuesNLI.txt: Contains experimental data for Non-Linearity Indices for VmDC of complex cells.
        - nlivmf1a.txt: Contains experimental data for Non-Linearity Indices for VmF1 of simple cells.
        - or_map_new_16x16: Contains the precomputed orientation map. A specific central portion of it can be cropped based on the or_map_stretch parameters.
        - param: Defines the parameters used in the fullfield drifting gratings and natural images protocol as well as for the size tuning protocol. The 'default' file contains the basic parameters of the model as well as the path of the parameters corresponding to each sheets of the model. 'SpatioTemporalFilterRetinaLGN_defaults' contains the parameters of the input model as well as the parameters of the LGN neurons. `l4_cortex_exc', `l4_cortex_inh`, `l23_cortex_exc`, `l23_cortex_inh` contains the parameters of each cortical population as well as the parameters of the connections of the model. Each '_rec' file contains the recording parameters for each population of the model.
        - param_spont: Contains the parameters for the spontaneous activity protocol of the model. The only difference with the `param` directory resides in the recording parameters.
        - parameter_search_analysis.py: Runs the analysis one multiple simulation directories belonging to the same parameter search, distributed on multiple computational nodes and using by default Slurm for scheduling.
        - run.py: Runs the model. Defines which experimental protocol (as defined in experiments.py) and which analysis (as defined in analysis_and_visualization.py) will be run. By default runs the fullfield drifting grating and natural images protocol.
        - run_analysis.py: Runs only the analysis on the model on a mozaik datastore. Defines which analysis  (as defined in analysis_and_visualization.py) will be run. By default runs the fullfield drifting grating and natural images protocol analysis.
        - run_disk_ring.py: Same as `run.py`, but runs the disks and rings protocol shown in Cagnol2025 by default. 
        - run_parameter_search.py: Defines the parameters that will be used when running a search across multiple parameters. The parameter search will be distributed on different computational nodes, using Slurm as the scheduler by default. 
        - run_spont.py: Same as `run.py`, but runs the spontaneous activity protocol by default. 
        - run_stc.py: Same as `run.py`, but runs the size tuning protocol by default. 
        - visualization_functions.py: Contains the code specific to each figure. 
