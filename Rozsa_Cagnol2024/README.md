# Rozsa_Cagnol2024 

This repository contains the Rozsa_Cagnol2024 model implemented in the Mozaik framework, that was published by the [Computational Systems Neuroscience Group](http://csng.mff.cuni.cz/) at the Faculty of Mathematics and Physics, Charles University in [Rózsa, T., Cagnol, R., Antolík, J. (2024). Iso-orientation bias of layer 2/3 connections: the unifying mechanism of spontaneous, visually and optogenetically driven V1 dynamics.](https://www.biorxiv.org/content/10.1101/2024.11.19.624284v1). Setup used in the preprint: 16 threads on a EPYC 9384X CPU with 1536 GB of RAM.

To run the models present in this repository one must first install the Mozaik package and its dependencies (see [here](https://github.com/CSNG-MFF/mozaik) for more information and detailed installation instructions).

## Instructions
 
    1 Running the different experiments:

        First, in run_parameter_search.py, replace the field [PATH_TO_ENV] by the path of the activate executable of your virtual environment (for example, $HOME/virt_env/mozaik/bin/activate)

            - Traveling waves protocol:

                python run_parameter_search.py run_STW.py nest param_STW/defaults (~8h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                    python run_STW.py nest 16 param_STW/defaults SelfSustainedPushPull

            - Drifting grating and natural images protocol::

                python run_parameter_search.py run.py nest param/defaults (~26h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                    python run.py nest 16 param/defaults SelfSustainedPushPull

            - Size tuning protocol::

                python run_parameter_search.py run_stc.py nest param/defaults (~26h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                    python run_stc.py nest 16 param/defaults SelfSustainedPushPull

            - Spontaneous activity protocol::

                     python run_parameter_search.py run_spont.py nest param_spont/defaults (~2h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                     python run_spont.py nest 16 param_spont/defaults SelfSustainedPushPull

            - Spontaneous activity protocol for Modular Spontaneous Activity analysis::

                     python run_parameter_search.py run_MSA.py nest param_MSA/defaults (~10h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                     python run_MSA.py nest 16 param_MSA/defaults SelfSustainedPushPull

            - Full-field Optogenetic Stimulation protocol::

                     python run_parameter_search.py run_fullfield.py nest param_MSA/defaults (~73h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                     python run_fullfield.py nest 16 param_MSA/defaults SelfSustainedPushPull

            - Endogenous Optogenetic Stimulation protocol::

                     python run_parameter_search.py run_endogenous.py nest param_MSA/defaults (~40h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                     python run_endogenous.py nest 16 param_MSA/defaults SelfSustainedPushPull

            - Surrogate Optogenetic Stimulation protocol::

                     python run_parameter_search.py run_surrogate.py nest param_MSA/defaults (~40h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                     python run_surrogate.py nest 16 param_MSA/defaults SelfSustainedPushPull

            - 8-orientation Sinusoidal Drifting Grating protocol::

                     python run_parameter_search.py run_8_orientations.py nest param_MSA/defaults (~25h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                     python run_8_orientations.py nest 16 param_MSA/defaults SelfSustainedPushPull

            - 2-orientation Square Drifting Grating protocol::

                     python run_parameter_search.py run_chernov_visual.py nest param_MSA/defaults (~9h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                     python run_chernov_visual.py nest 16 param_MSA/defaults SelfSustainedPushPull

            - 2-orientation Square Drifting Grating with Central Optogenetic Stimulation protocol::

                     python run_parameter_search.py run_chernov_visual_opto.py nest param_MSA/defaults (~77h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                     python run_chernov_visual_opto.py nest 16 param_MSA/defaults SelfSustainedPushPull

            - Central Optogenetic Stimulation protocol::

                     python run_parameter_search.py run_central_stimulation.py nest param_MSA/defaults (~32h of runtime with our setup)

                - If not using slurm (results might slightly differ from the Preprint)::

                     python run_central_stimulation.py nest 16 param_MSA/defaults SelfSustainedPushPull
            - Analysis and plotting for Modular Spontaneous Activity protocol::

                    python run_analysis_msa.py path/to/spontaneous_activity_datastore path/to/grating_responses_datastore


            - Analysis and plotting for Patterned Optogenetic Stimulation protocol::

                    python run_analysis_patterned_stimulation.py path/to/spontaneous_activity_datastore path/to/fullfield_stimulation_datastore path/to/endogenous_stimulation_datastore path/to/surrogate_stimulation_datastore

            - Analysis and plotting for Central Optogenetic Stimulation protocol::

                    python run_analysis_center_stimulation.py path/to/spontaneous_activity_datastore path/to/center_stimulation_datastore

            - Analysis and plotting for simultaneous visual and optogenetic stimulation procotol::

                    python run_analysis_chernov.py path/to/spontaneous_activity_datastore path/to/drifting_square_grating_datastore path/to/drifting_square_grating_with_optogenetic_stimulation_datastore

    2 Description of the files:

        - analysis_and_visualization.py: Contains the code for the different analysis and plotting. The function `perform_analysis_and_visualization` runs the analysis and the plots corresponding to the fullfield drifting grating and natural images protocol. The functions `perform_analysis_and_visualization_stc` and `perform_analysis_and_visualization_spont` do the same respectively for the size tuning and spontaneous activity protocol.
        - calcium_light_spread_kernel.npy: Contains the spatial smoothing kernel for estimating the calcium imaging light spread in the cortex
        - data/exData.mat: Contains the monkey firing rate and coefficient of variation data for figures 2C and 2D.
        - data/ell_wolfie.mat: Contains the monkey wavelength data for figure 2F.
        - data/MonkeyT.txt: Contains the propagation speeds distribution data for Monkey 2 in figure 2G.
        - data/wolfie-speed_density.mat: Contains the propagation speeds distribution data for Monkey 1 in figure 2G.
        - experiments.py: Defines for each protocol which experiments will be run as well as their parameters. `create_experiments` corresponds to the fullfield drifting grating protocol, `create_experiments_stc` corresponds to the size tuning protocol, and `create_experiments_spont` corresponds to the spontaneous activity protocol.
        - eye_path.pickle: Contains the coordinates of the eye path that is used by default in the natural image protocol.
        - image_naturelle_HIGHC.bmp: The natural image that is used by default in the natural image protocol.
        - light_scattering_radial_profiles_lsd10.pickle: Contains the lookup table for optogenetic stimulation light spread in the cortex
        - model.py: Contains the code which creates each layers of the model and build the connections based on the parameters used to run the model.
        - msa_analysis.py: Contains code for the joint analysis and visualisation of the Modular Spontaneous Activity, Patterned Optogenetic Stimulation and Central Optogenetic Stimulation protocol results
        - or_map_new_16x16: Contains the precomputed orientation map. A specific central portion of it can be cropped based on the or_map_stretch parameters.
        - param: Defines the parameters used in the fullfield drifting gratings and natural images protocol as well as for the size tuning protocol. The 'default' file contains the basic parameters of the model as well as the path of the parameters corresponding to each sheets of the model. 'SpatioTemporalFilterRetinaLGN_defaults' contains the parameters of the input model as well as the parameters of the LGN neurons. `l4_cortex_exc', `l4_cortex_inh`, `l23_cortex_exc`, `l23_cortex_inh` contains the parameters of each cortical population as well as the parameters of the connections of the model. Each '_rec' file contains the recording parameters for each population of the model.
        - param_MSA/defaults: Contains the parameters for the Modular Spontaneous Activity and optogenetic stimulation protocols of the model. The only difference with the `param` directory resides in the recording parameters.
        - param_spont: Contains the parameters for the spontaneous activity protocol of the model. The only difference with the `param` directory resides in the recording parameters.
        - param_STW: Contains the parameters for the Spontaneous Traveling Waves protocol of the model. The only difference with the `param` directory resides in the recording parameters.
        - parameter_search_analysis.py: Runs the analysis one multiple simulation directories belonging to the same parameter search, distributed on multiple computational nodes and using by default Slurm for scheduling.
        - run.py: Runs the model. Defines which experimental protocol (as defined in experiments.py) and which analysis (as defined in analysis_and_visualization.py) will be run. By default runs the fullfield drifting grating and natural images protocol.
        - run_8_orientations.py: Same as `run.py` but runs the 8-orientation sinusoidal gratings protocol by default
        - run_analysis.py: Runs only the analysis on the model on a mozaik datastore. Defines which analysis  (as defined in analysis_and_visualization.py) will be run. By default runs the fullfield drifting grating and natural images protocol analysis.
        - run_analysis_center_stimulation.py: Runs the analysis on the mozaik datastore for the Central Optogenetic Stimulation protocol.
        - run_analysis_chernov.py: Runs the analysis on the mozaik datastore for the simultaneous optogenetic and visual stimulation protocol.
        - run_analysis_msa.py: Runs the analysis on the mozaik datastore for the Modular Spontaneous Activity protocol.
        - run_analysis_patterned_stimulation.py: Runs the analysis on the mozaik datastore for the Patterned Optogenetic Stimulation protocol.
        - run_central_stimulation.py: Same as `run.py` but runs the central optogenetic stimulation protocol by default
        - run_chernov_visual.py: Same as `run.py` but runs the visual-only part of the simultaneous optogenetic and visual stimulation protocol by default
        - run_chernov_visual_opto.py: Same as `run.py` but runs the combined visual-optogenetic part of the simultaneous optogenetic and visual stimulation protocol by default
        - run_endogenous.py: Same as `run.py`, but runs the endogenous optogenetic stimulation protocol
        - run_fullfield.py: Same as `run.py`, but runs the fullfield optogenetic stimulation protocol
        - run_MSA.py: Same as `run.py`, but runs the spontaneous activity protocol for Modular Spontaneous Activity analysis by default
        - run_parameter_search.py: Defines the parameters that will be used when running a search across multiple parameters. The parameter search will be distributed on different computational nodes, using Slurm as the scheduler by default.
        - run_spont.py: Same as `run.py`, but runs the spontaneous activity protocol by default.
        - run_stc.py: Same as `run.py`, but runs the size tuning protocol by default.
        - run_STW.py: Same as `run.py`, but runs the Spontaneous Traveling Waves protocol by default.
        - run_surrogate.py: Same as `run.py`, but runs the surrogate optogenetic stimulation protocol
        - visualization_functions.py: Contains the code specific to each figure.


