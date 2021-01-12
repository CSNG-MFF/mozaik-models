# -*- coding: utf-8 -*-
"""
This is implementation of model of self-sustained activitity in balanced networks from: 
Vogels, T. P., & Abbott, L. F. (2005). 
Signal propagation and logic gating in networks of integrate-and-fire neurons. 
The Journal of neuroscience : the official journal of the Society for Neuroscience, 25(46), 10786–95. 
"""
import matplotlib
matplotlib.use('Agg')

from mpi4py import MPI
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from parameters import ParameterSet
from analysis_and_visualization import perform_analysis_and_visualization
from model import SelfSustainedPushPull
from experiments import create_experiments
import mozaik
from mozaik.controller import run_workflow, setup_logging
import mozaik.controller
import sys
from pyNN import nest


mpi_comm = MPI.COMM_WORLD


if True:
    data_store, model = run_workflow(
        'SelfSustainedPushPull', SelfSustainedPushPull, create_experiments)
    if False:
        model.connectors['V1AffConnectionOn'].store_connections(data_store)
        model.connectors['V1AffConnectionOff'].store_connections(data_store)
        model.connectors['V1AffInhConnectionOn'].store_connections(data_store)
        model.connectors['V1AffInhConnectionOff'].store_connections(data_store)
        model.connectors['V1L4ExcL4ExcConnection'].store_connections(
            data_store)
        model.connectors['V1L4ExcL4InhConnection'].store_connections(
            data_store)
        model.connectors['V1L4InhL4ExcConnection'].store_connections(
            data_store)
        model.connectors['V1L4InhL4InhConnection'].store_connections(
            data_store)
        model.connectors['V1L4ExcL4ExcConnectionRand'].store_connections(
            data_store)
        model.connectors['V1L4ExcL4InhConnectionRand'].store_connections(
            data_store)
        model.connectors['V1L4InhL4ExcConnectionRand'].store_connections(
            data_store)
        model.connectors['V1L4InhL4InhConnectionRand'].store_connections(
            data_store)
        model.connectors['V1L23ExcL23ExcConnection'].store_connections(
            data_store)
        model.connectors['V1L23ExcL23InhConnection'].store_connections(
            data_store)
        model.connectors['V1L23InhL23ExcConnection'].store_connections(
            data_store)
        model.connectors['V1L23InhL23InhConnection'].store_connections(
            data_store)
        model.connectors['L4ExcL23ExcConnection'].store_connections(data_store)
        model.connectors['L4ExcL23InhConnection'].store_connections(data_store)
    data_store.save()
else:
    setup_logging()
    data_store = PickledDataStore(load=True, parameters=ParameterSet(
        {'root_directory': 'SelfSustainedPushPull_test____', 'store_stimuli': False}), replace=True)

if mpi_comm.rank == 0:
    print("Starting visualization")
    perform_analysis_and_visualization(data_store)
