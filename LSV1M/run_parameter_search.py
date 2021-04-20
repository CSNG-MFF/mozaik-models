# -*- coding: utf-8 -*-
import sys
from mozaik.meta_workflow.parameter_search import CombinationParameterSearch, SlurmSequentialBackend
import numpy
import time


if True:
    CombinationParameterSearch(SlurmSequentialBackend(num_threads=16, num_mpi=1,slurm_options=['--hint=nomultithread'],path_to_mozaik_env='/home/antolikjan/virt_env/mozaik-python3/bin/activate'), {
      'trial' : [1],
      'pynn_seed' : [263,503,701,947],
    }).run_parameter_search()



