# -*- coding: utf-8 -*-
import sys
from mozaik.meta_workflow.parameter_search import CombinationParameterSearch, SlurmSequentialBackend
import numpy
import time


if True:
    CombinationParameterSearch(SlurmSequentialBackend(num_threads=1, num_mpi=16,slurm_options=['--hint=nomultithread'],path_to_mozaik_env='/home/antolikjan/virt_env/mozaik-python3/bin/activate'), {
      'trial' : [1],
    }).run_parameter_search()



