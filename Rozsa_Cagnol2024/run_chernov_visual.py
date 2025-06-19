# -*- coding: utf-8 -*-
"""
This is the implementation of the model corresponding to the pre-print `Iso-orientation bias of layer 2/3 connections: the unifying mechanism of spontaneous, visually and optogenetically driven V1 dynamics`
Rózsa, T., Cagnol, R., Antolík, J. (2024).
https://www.biorxiv.org/ TODO: Update
"""
import matplotlib
matplotlib.use('Agg')

from mpi4py import MPI
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from parameters import ParameterSet
from model import SelfSustainedPushPull
from experiments import *
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
    'SelfSustainedPushPull', SelfSustainedPushPull, create_experiments_chernov_visual)
data_store.save()
