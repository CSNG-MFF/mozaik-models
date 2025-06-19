# -*- coding: utf-8 -*-
from mozaik.storage.datastore import PickledDataStore
from mozaik.controller import Global
from parameters import ParameterSet
import mozaik
from mozaik.controller import setup_logging
import sys
from msa_analysis_plotting import *
from mozaik.storage.queries import *

# Usage:
# If all radii in the same datastore: python run_analysis_center_stimulation.py path/to/spontaneous_activity_datastore path/to/center_stimulation_datastore
# If radii in separate datastores: python run_analysis_center_stimulation.py path/to/spontaneous_activity_datastore path/to/center_stimulation_datastore_radius_1 path/to/center_stimulation_datastore_radius_2 path/to/center_stimulation_datastore_radius_3 path/to/center_stimulation_datastore_radius_4 path/to/center_stimulation_datastore_radius_5 path/to/center_stimulation_datastore_radius_6

# Whether to run analysis or plotting
spont_analysis = False
center_stim_analysis = False
run_plotting = True

Global.root_directory = sys.argv[1]+'/'

setup_logging()
data_store = PickledDataStore(load=True, parameters=ParameterSet(
    {'root_directory': sys.argv[1], 'store_stimuli': False}), replace=False)

if len(sys.argv) > 3:
    ds_center = [PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": path, "store_stimuli": False}),
        replace=False,
    ) for path in sys.argv[2:8]
    ]
else:
    ds = PickledDataStore(
        load=True,
        parameters=ParameterSet({"root_directory": sys.argv[2], "store_stimuli": False}),
        replace=False,
        )
    ds_center = [ds for i in range(6)]

if spont_analysis:
    queries.param_filter_query(data_store,analysis_algorithm="RecordingArrayOrientationMap").remove_ads_from_datastore()
    RecordingArrayOrientationMap(param_filter_query(data_store,sheet_name=["V1_Exc_L2/3","V1_Inh_L2/3"]),
    ParameterSet(
        {
            "s_res": 40,
            "array_width": 4000,
        }
    ),
    ).analyse()
    queries.param_filter_query(data_store,analysis_algorithm="RecordingArrayTimecourse").remove_ads_from_datastore()
    RecordingArrayTimecourse(param_filter_query(data_store,sheet_name=["V1_Exc_L2/3","V1_Inh_L2/3"]),
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

if center_stim_analysis:
    for ds in ds_center:
        queries.param_filter_query(ds,analysis_algorithm="RecordingArrayTimecourse").remove_ads_from_datastore()
        RecordingArrayTimecourse(param_filter_query(ds,sheet_name=["V1_Exc_L2/3","V1_Inh_L2/3"]),
            ParameterSet(
                {
                    "s_res": 40,
                    "t_res": 50,
                    "array_width": 4000,
                    "electrode_radius": 50,
                }
            ),
        ).analyse()
        ds.save()

if run_plotting:
    CenterStimulationPlot(
        data_store,
        ParameterSet({
            "center_stim_dsv_list": ds_center,
        }),
        fig_param={"dpi": 400, "figsize": (9, 6)},
        plot_file_name="CenterStimulationPlot.png",
    ).plot()
