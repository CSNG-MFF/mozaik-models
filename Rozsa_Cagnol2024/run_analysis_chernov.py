# -*- coding: utf-8 -*-
from mozaik.storage.datastore import PickledDataStore
from mozaik.controller import Global
from parameters import ParameterSet
import mozaik
from mozaik.controller import setup_logging
import sys
from msa_analysis_plotting import *
from mozaik.storage.queries import *

# Usage: python run_analysis_chernov.py path/to/spontaneous_activity_datastore path/to/drifting_square_grating_datastore path/to/drifting_square_grating_with_optogenetic_stimulation_datastore

# Which analysis / plotting to run
spont_analysis = True
chernov_visual_analysis = True
chernov_visual_opto_analysis = True
plotting = True

Global.root_directory = sys.argv[1]+'/'

setup_logging()
data_store = PickledDataStore(load=True, parameters=ParameterSet(
    {'root_directory': sys.argv[1], 'store_stimuli': False}), replace=False)

data_store_chernov_visual = PickledDataStore(load=True, parameters=ParameterSet(
    {'root_directory': sys.argv[2], 'store_stimuli': False}), replace=False)

data_store_chernov_visual_opto = PickledDataStore(load=True, parameters=ParameterSet(
    {'root_directory': sys.argv[3], 'store_stimuli': False}), replace=False)

# Spontaneous activity
if spont_analysis:
    queries.param_filter_query(data_store,analysis_algorithm="RecordingArrayOrientationMap").remove_ads_from_datastore()
    RecordingArrayOrientationMap(param_filter_query(data_store,sheet_name=["V1_Exc_L2/3"]),
    ParameterSet(
        {
            "s_res": 40,
            "array_width": 4000,
        }
    ),
    ).analyse()
    queries.param_filter_query(data_store,analysis_algorithm="RecordingArrayTimecourse").remove_ads_from_datastore()
    RecordingArrayTimecourse(param_filter_query(data_store,sheet_name=["V1_Exc_L2/3"]),
    ParameterSet(
        {
            "s_res": 40,
            "t_res": 50,
            "array_width": 4000,
            "electrode_radius": 50,
        }
    ),
    ).analyse()
    data_store.save()

# Visual
if chernov_visual_analysis:
    queries.param_filter_query(data_store_chernov_visual,analysis_algorithm="RecordingArrayTimecourse").remove_ads_from_datastore()
    RecordingArrayTimecourse(param_filter_query(data_store_chernov_visual,sheet_name=["V1_Exc_L2/3"]),
        ParameterSet(
            {
                "s_res": 40,
                "t_res": 50,
                "array_width": 4000,
                "electrode_radius": 50,
            }
        ),
    ).analyse()
    data_store_chernov_visual.save()

# Visual-opto
if chernov_visual_opto_analysis:
    queries.param_filter_query(data_store_chernov_visual_opto,analysis_algorithm="RecordingArrayTimecourse").remove_ads_from_datastore()
    RecordingArrayTimecourse(param_filter_query(data_store_chernov_visual_opto,sheet_name=["V1_Exc_L2/3"]),
        ParameterSet(
            {
                "s_res": 40,
                "t_res": 50,
                "array_width": 4000,
                "electrode_radius": 50,
            }
        ),
    ).analyse()
    data_store_chernov_visual_opto.save()

if plotting:
    ChernovPlot(
        data_store,
        ParameterSet({
            "visual_dsv": queries.param_filter_query(data_store_chernov_visual),
            "visual_opto_dsv": queries.param_filter_query(data_store_chernov_visual_opto),
            "time_cutoff_ms": 600,
        }),
        fig_param={"dpi": 100, "figsize": (10, 4)},
        plot_file_name="ChernovPlot.png",
    ).plot()
