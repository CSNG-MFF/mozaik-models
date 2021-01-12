# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

from mpi4py import MPI
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from parameters import ParameterSet
from analysis_and_visualization import perform_analysis_and_visualization_stc
from model import SelfSustainedPushPull
from experiments import create_experiments_stc
import mozaik
from mozaik.controller import run_workflow, setup_logging
import mozaik.controller
import sys
from pyNN import nest


mpi_comm = MPI.COMM_WORLD


if True:
    data_store, model = run_workflow(
        'SelfSustainedPushPull', SelfSustainedPushPull, create_experiments_stc)
    data_store.save()


if mpi_comm.rank == 0:
    print "Starting visualization"
    perform_analysis_and_visualization_stc(data_store)
    data_store.save()
