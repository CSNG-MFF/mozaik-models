# -*- coding: utf-8 -*-
import sys
from mozaik.meta_workflow.parameter_search import CombinationParameterSearch, SlurmSequentialBackend, SlurmSequentialBackendMPI
import numpy
import time


if True:
    CombinationParameterSearch(SlurmSequentialBackend(num_threads=16, num_mpi=1,slurm_options=['--hint=nomultithread'],path_to_mozaik_env='/home/antolikjan/virt_env/mozaik/bin/activate'), {
      'trial' : [1],
    }).run_parameter_search()



