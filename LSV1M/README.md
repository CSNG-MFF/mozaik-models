# LSV1M 

This repository contains the LSV1M model implemented in the Mozaik framework, that was published by the [Computational Systems Neuroscience Group](http://csng.mff.cuni.cz/) at the Faculty of Mathematics and Physics, Charles University in [Antolík, J., Cagnol, R., Rózsa, T., Monier, C., Frégnac, Y., & Davison, A. P. (2024). A comprehensive data-driven model of cat primary visual cortex. PLOS Computational Biolology](https://pmc.ncbi.nlm.nih.gov/articles/PMC11371232/). Setup used in the article: 16 processes on a EPYC 7302 CPU with 128 GB of RAM. 

To run the models present in this repository one must first install the Mozaik package and its dependencies (see [here](https://github.com/CSNG-MFF/mozaik) for more information and detailed installation instructions).

## Instructions

    1 Running the different experiments:

        First, in run_parameter_search.py, replace the field [PATH_TO_ENV] by the path of the activate executable of your virtual environment (for example, $HOME/virt_env/mozaik/bin/activate)

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

            - Randomized Experanto protocol (images and videos from an Experanto dataset)::

                Each trial is split into chunks. Every simulation job saves and exports its chunk
                into <datastore>/experanto, then one dependent merge job produces shards/trial{T}.
                --help lists the launcher settings::

                    python run_parameter_search_experanto.py run_experanto.py nest param_experanto/defaults --data-root [PATH_TO_DATASET] --n-trials 2 --n-chunks 5 --mozaik-env [PATH_TO_ENV]

                Chunk lists are generated on the first run and reused afterwards. --chunks-dir
                selects their directory. --dry-run writes the lists and prints job submissions.
                Simulation allocations must cover both the simulation and chunk export phases.

                - A YAML config lists every setting, including the three positionals. With
                  --config, only --dry-run and --confirm-chunk-settings may also be passed::

                    python run_parameter_search_experanto.py --config exp1.yml

                    # exp1.yml
                    run_script: run_experanto.py
                    simulator: nest
                    parameters_url: param_experanto/defaults
                    data_root: "[PATH_TO_DATASET]"
                    n_trials: 2
                    n_chunks: 30
                    chunks: 0-9          # inclusive; null selects all chunks
                    results_dir: runs/exp1
                    chunks_dir: null    # use <results_dir>/chunks
                    num_mpi: 4
                    threads_per_rank: 1
                    mozaik_env: "[PATH_TO_ENV]"
                    chunk_seed: 42
                    export_settings:
                      sheet_names: null          # all recorded sheets
                      sampling_rate: 1000.0      # response metadata, in Hz
                      export_blank_spikes: true  # include spikes during explicit blanks
                      tier_reference_path: null  # keep source tier assignments
                    skip_merge: true    # intermediate round; false submits a full-trial merge

                  Replace the quoted path placeholders. Relative paths resolve against the
                  launcher's working directory, including tier_reference_path. Paths in YAML
                  do not expand ~ or environment variables; use relative paths or full paths.

                - The export_settings values above are the defaults; {} uses the same values.
                  sheet_names can select a list such as [X_ON, X_OFF]. sampling_rate is stored
                  in response metadata; spike times remain in seconds without resampling.
                  export_blank_spikes: false omits blank spikes while retaining blank timing
                  and screen metadata. Presentation durations come from the recording.
                  tier_reference_path optionally names an existing screen/combined_meta.json:
                  matching condition_hash values supply train/validation/test assignments.
                  With null, or for unmatched stimuli, source metadata supplies the assignments.

                - --skip-merge exports every chunk without scheduling a merge. It is a submission
                  option and does not change the dataset settings. To run in rounds, keep the same
                  --results-dir, use --skip-merge for intermediate rounds, and finish them before
                  submitting the final round with merging enabled::

                    ... --results-dir runs/exp1 --chunks 0-9 --skip-merge
                    ... --results-dir runs/exp1 --chunks 10-19 --skip-merge
                    ... --results-dir runs/exp1 --chunks 20-29

                - To merge after all simulations have finished, run::

                    python experanto_merge.py 0 --results-dir runs/exp1

                  The planned chunk count is read from run_config.json.
                  --output-dir selects a fresh destination. --chunk-start and --chunk-end select
                  an inclusive/exclusive range for a partial result. Merging retains individual
                  exports. A whole-trial merge requires every chunk's completed export.

                - run_config.json records the simulation and export settings and every chunk's
                  seed. Only chunks and skip_merge may vary between rounds. Each final shard's
                  meta.json preserves the provenance and shared export settings. Reused chunk
                  lists must match chunk_settings.json; --confirm-chunk-settings verifies and
                  records settings for lists without that record.

                - To run one chunk directly, set CHUNK, CHUNK_DIR and BASE_PATH and invoke
                  run_experanto.py with the usual Mozaik arguments, including the trial parameter.
                  Launcher jobs pass CHUNK and RUN_CONFIG; the run record supplies paths and settings.
                  Without RUN_CONFIG the export uses default settings,
                  with SHEET_NAMES optionally selecting comma-separated sheets.


    2 Description of the files:
    
        - analysis_and_visualization.py: Contains the code for the different analysis and plotting. The function `perform_analysis_and_visualization` runs the analysis and the plots corresponding to the fullfield drifting grating and natural images protocol. The functions `perform_analysis_and_visualization_stc` and `perform_analysis_and_visualization_spont` do the same respectively for the size tuning and spontaneous activity protocol.
        - experiments.py: Defines for each protocol which experiments will be run as well as their parameters. `create_experiments` corresponds to the fullfield drifting grating protocol, `create_experiments_stc` corresponds to the size tuning protocol, and `create_experiments_spont` corresponds to the spontaneous activity protocol.
        - eye_path.pickle: Contains the coordinates of the eye path that is used by default in the natural image protocol. 
        - image_naturelle_HIGHC.bmp: The natural image that is used by default in the natural image protocol.
        - model.py: Contains the code which creates each layers of the model and build the connections based on the parameters used to run the model.
        - or_map_new_16x16: Contains the precomputed orientation map. A specific central portion of it can be cropped based on the or_map_stretch parameters.
        - param: Defines the parameters used in the fullfield drifting gratings and natural images protocol as well as for the size tuning protocol. The 'default' file contains the basic parameters of the model as well as the path of the parameters corresponding to each sheets of the model. 'SpatioTemporalFilterRetinaLGN_defaults' contains the parameters of the input model as well as the parameters of the LGN neurons. `l4_cortex_exc', `l4_cortex_inh`, `l23_cortex_exc`, `l23_cortex_inh` contains the parameters of each cortical population as well as the parameters of the connections of the model. Each '_rec' file contains the recording parameters for each population of the model.
        - param_spont: Contains the parameters for the spontaneous activity protocol of the model. The only difference with the `param` directory resides in the recording parameters.
        - parameter_search_analysis.py: Runs the analysis one multiple simulation directories belonging to the same parameter search, distributed on multiple computational nodes and using by default Slurm for scheduling.
        - run.py: Runs the model. Defines which experimental protocol (as defined in experiments.py) and which analysis (as defined in analysis_and_visualization.py) will be run. By default runs the fullfield drifting grating and natural images protocol.
        - run_analysis.py: Runs only the analysis on the model on a mozaik datastore. Defines which analysis  (as defined in analysis_and_visualization.py) will be run. By default runs the fullfield drifting grating and natural images protocol analysis.
        - experanto_merge.py: Merges completed chunk exports into one Experanto shard per trial, without loading Mozaik datastores.
        - param_experanto: Contains the parameters for the Randomized Experanto protocol. The differences with the `param` directory are the recording parameters (record all), the spatial resolution of the input model, and the blank period between stimuli, which the Experanto stimuli bring with them.
        - run_experanto.py: Simulates and exports one Randomized Experanto chunk, selected by the model trial parameter and CHUNK, with paths from RUN_CONFIG or manual inputs.
        - run_parameter_search.py: Defines the parameters that will be used when running a search across multiple parameters. The parameter search will be distributed on different computational nodes, using Slurm as the scheduler by default. 
        - run_parameter_search_experanto.py: Generates chunk lists, submits simulation/export jobs and optional dependent merges, and records run provenance.
        - run_spont.py: Same as `run.py`, but runs the spontaneous activity protocol by default. 
        - run_stc.py: Same as `run.py`, but runs the size tuning protocol by default. 
        - visualization_functions.py: Contains the code specific to each figure. 
