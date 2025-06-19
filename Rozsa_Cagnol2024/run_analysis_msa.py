# -*- coding: utf-8 -*-
from mozaik.storage.datastore import PickledDataStore
from mozaik.controller import Global
from parameters import ParameterSet
import mozaik
from mozaik.controller import setup_logging
import sys
from msa_analysis_plotting import *
from mozaik.storage.queries import *

# Usage: python run_analysis_msa.py path/to/spontaneous_activity_datastore path/to/grating_responses_datastore

spont_analysis = True
visual_stim_analysis = True
plotting = True

Global.root_directory = sys.argv[1]+'/'

setup_logging()
data_store = PickledDataStore(load=True, parameters=ParameterSet(
    {'root_directory': sys.argv[1], 'store_stimuli': False}), replace=False)

ds_ors = PickledDataStore(load=True, parameters=ParameterSet(
    {'root_directory': sys.argv[2], 'store_stimuli': False}), replace=False)

if visual_stim_analysis:
    queries.param_filter_query(ds_ors,analysis_algorithm="RecordingArrayTimecourse").remove_ads_from_datastore()
    RecordingArrayTimecourse(param_filter_query(ds_ors,sheet_name="V1_Exc_L2/3",st_contrast=100),
    ParameterSet(
        {
            "s_res": 40,
            "t_res": 50,
            "array_width": 4000,
            "electrode_radius": 50,
        }
    ),
    ).analyse()
    ds_ors.save()

# Spontaneous activity analysis
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

    analogsignal = queries.param_filter_query(data_store,analysis_algorithm="RecordingArrayTimecourse",sheet_name="V1_Exc_L2/3").get_analysis_result()[0]
    random_activity_1 = np.random.rand(*analogsignal.analog_signal.shape)
    random_activity_2 = np.random.rand(*analogsignal.analog_signal.shape)
    for k,v in {"random_1":random_activity_1, "random_2":random_activity_2}.items():
        data_store.full_datastore.add_analysis_result(
                AnalogSignal(
                NeoAnalogSignal(v, t_start=0, sampling_period=50*qt.ms,units=munits.spike / qt.s),
                y_axis_units=munits.spike / qt.s,
                tags=analogsignal.tags,
                sheet_name="V1_Exc_L2/3",
                stimulus_id=str({"module_path" :"mozaik.analysis.data_structures","name":'AnalogSignal',"identifier":k, "sheet_name":'V1_Exc_L2/3'}), # Hack because tags do not count into the signature of the analysis objects
                analysis_algorithm="RecordingArrayTimecourse",
            )
        )

    queries.param_filter_query(data_store,analysis_algorithm="SimulatedCalciumSignal").remove_ads_from_datastore()
    SimulatedCalciumSignal(data_store,ParameterSet({
        "reference_dsv": data_store,
        "spatial_profile_path": "calcium_light_spread_kernel.npy",
    })).analyse()
    queries.param_filter_query(data_store,analysis_algorithm="GaussianBandpassFilter").remove_ads_from_datastore()
    GaussianBandpassFilter(queries.param_filter_query(data_store,y_axis_name="Calcium imaging signal (normalized)"),ParameterSet({
        "highpass_sigma_um": 200,
        "lowpass_sigma_um": 26,
    })).analyse()
    queries.param_filter_query(data_store,analysis_algorithm="CorrelationMaps").remove_ads_from_datastore()
    CorrelationMaps(queries.param_filter_query(data_store,analysis_algorithm="GaussianBandpassFilter"),ParameterSet({})).analyse()
    queries.param_filter_query(data_store,analysis_algorithm="OrientationMapSimilarity").remove_ads_from_datastore()
    OrientationMapSimilarity(queries.param_filter_query(data_store,analysis_algorithm="CorrelationMaps",sheet_name="V1_Exc_L2/3",st_name='InternalStimulus'),ParameterSet({
        "or_map_dsv": queries.param_filter_query(data_store,sheet_name="V1_Exc_L2/3"),
    })).analyse()
    queries.param_filter_query(data_store,analysis_algorithm="CorrelationMapSimilarity").remove_ads_from_datastore()
    CorrelationMapSimilarity(queries.param_filter_query(data_store,st_name='InternalStimulus',sheet_name="V1_Exc_L2/3"),ParameterSet({
        "corr_map_dsv": queries.param_filter_query(data_store,st_name='InternalStimulus',sheet_name="V1_Inh_L2/3"),
        "exclusion_radius": 400,
    })).analyse()
    CorrelationMapSimilarity(queries.param_filter_query(data_store,st_identifier="random_1",analysis_algorithm="CorrelationMaps"),ParameterSet({
        "corr_map_dsv": queries.param_filter_query(data_store,st_identifier="random_2",analysis_algorithm="CorrelationMaps"),
        "exclusion_radius": 400,
    })).analyse()
    queries.param_filter_query(data_store,analysis_algorithm="Smith_2018_Mulholland_2021_2024_spont_analyses").remove_ads_from_datastore()
    Smith_2018_Mulholland_2021_2024_spont_analyses(data_store.full_datastore,ParameterSet({})).analyse()
    queries.param_filter_query(data_store,analysis_algorithm="Kenet_2003").remove_ads_from_datastore()
    data_store.save()

    Kenet_2003(queries.param_filter_query(data_store,sheet_name="V1_Exc_L2/3"),
               ParameterSet({"fullfield_gratings_dsv": ds_ors})).analyse()
    queries.param_filter_query(data_store,analysis_algorithm="Tsodyks_1999").remove_ads_from_datastore()
    Tsodyks_1999(queries.param_filter_query(data_store,sheet_name="V1_Exc_L2/3"),
               ParameterSet({"fullfield_gratings_dsv": ds_ors,"n_neurons":1000})).analyse()
    data_store.save()

# Plotting
if plotting:
    Smith2018Mulholland2024Plot(
        data_store,
        ParameterSet({}),
        fig_param={"dpi": 100, "figsize": (14, 6)},
        plot_file_name="Smith2018Mulholland2024Plot.png",
    ).plot()

    Mulholland2021Plot(
        data_store,
        ParameterSet({}),
        fig_param={"dpi": 100, "figsize": (11, 3)},
        plot_file_name="Mulholland2021Plot.png",
    ).plot()

    Kenet2003Tsodyks1999Plot(
        data_store,
        ParameterSet(
            {}
        ),
        fig_param={"dpi": 100, "figsize": (10, 3.5)},
        plot_file_name="Kenet2003Tsodyks1999Plot.png",
    ).plot()
