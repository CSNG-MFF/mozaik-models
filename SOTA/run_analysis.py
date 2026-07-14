# -*- coding: utf-8 -*-
"""
"""
import matplotlib
matplotlib.use('Agg')

from analysis_and_visualization import perform_analysis_and_visualization, perform_analysis_and_visualization_stc, perform_analysis_and_visualization_spont
from mozaik.storage.datastore import Hdf5DataStore, PickledDataStore
from mozaik.controller import Global
from parameters import ParameterSet
import mozaik
from mozaik.controller import setup_logging
import sys
import os


Global.root_directory = sys.argv[1]+'/'

setup_logging()
data_store = PickledDataStore(load=True, parameters=ParameterSet(
    {'root_directory': sys.argv[1], 'store_stimuli': False}), replace=True)

run_analysis = os.environ.get('MOZAIK_SKIP_ANALYSIS', '0') not in ['1', 'true', 'True']
perform_analysis_and_visualization_stc(data_store, run_analysis=run_analysis)
#data_store.save()
