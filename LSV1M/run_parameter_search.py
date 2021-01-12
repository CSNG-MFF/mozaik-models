# -*- coding: utf-8 -*-
import sys
from mozaik.meta_workflow.parameter_search import CombinationParameterSearch, SlurmSequentialBackendUK
import numpy
import time


if True:
    CombinationParameterSearch(SlurmSequentialBackendUK(num_threads=16, num_mpi=1), {
      'trial' : [1],
    }).run_parameter_search()



