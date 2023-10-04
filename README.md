# mozaik-models
This repository contains models implemented in the Mozaik framework, that were published by the [Computational Systems Neuroscience Group](http://csng.mff.cuni.cz/) at the Faculty of Mathematics and Physics, Charles University. 



## Model list

1. LSV1M  - Large scale model of cat primary visual cortex published in [this](https://www.biorxiv.org/content/biorxiv/early/2019/02/20/416156.full.pdf) BioRXiv pre-print.
        
        How to run the different experiments:

        First, in run_parameter_search.py, replace the field [PATH_TO_ENV] by then path of the activate executable of your virtual environment (for example, $HOME/virt_env/mozaik/bin/activate)

            - Drifting grating and natural images protocol::

                python run_parameter_search.py run.py nest param/defaults

                - If not using slurm (results might slightly differ from the Preprint)::

                    mpirun -n16 python run.py nest 1 param/defaults SelfSustainedPushPull

            - Size tuning protocol::

                python run_parameter_search.py run_stc.py nest param/defaults

                - If not using slurm (results might slightly differ from the Preprint)::

                    mpirun -n16 python run_stc.py nest 1 param/defaults SelfSustainedPushPull

            - Spontaneous activity protocol::

                     python run_parameter_search.py run_spont.py nest param_spont/defaults

                - If not using slurm (results might slightly differ from the Preprint)::

                     mpirun -n16 python run_spont.py nest 1 param_spont/defaults SelfSustainedPushPull


