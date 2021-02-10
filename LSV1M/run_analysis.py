# -*- coding: utf-8 -*-
"""
"""
import matplotlib
matplotlib.use('Agg')

from analysis_and_visualization import perform_analysis_and_visualization, perform_analysis_and_visualization_stc
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from mozaik.controller import Global
from parameters import ParameterSet
import mozaik
from mozaik.controller import setup_logging
import sys


Global.root_directory = sys.argv[1]+'/'

setup_logging()
data_store = PickledDataStore(load=True, parameters=ParameterSet(
    {'root_directory': sys.argv[1], 'store_stimuli': False}), replace=True)

perform_analysis_and_visualization(data_store)
data_store.save()
