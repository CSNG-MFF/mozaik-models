# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

from mpi4py import MPI
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from parameters import ParameterSet
from model import SelfSustainedPushPull
from experiments import create_experiments_optogenetic_patterned_stimulation
import mozaik
from mozaik.controller import run_workflow, setup_logging
import mozaik.controller
import sys
from pyNN import nest

from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD

import nest
nest.Install("stepcurrentmodule")

data_store, model = run_workflow(
    'SelfSustainedPushPull', SelfSustainedPushPull, create_experiments_optogenetic_patterned_stimulation)
data_store.save()
