# -*- coding: utf-8 -*-
from mozaik.storage.datastore import PickledDataStore
from mozaik.controller import Global
from parameters import ParameterSet
import mozaik
from mozaik.controller import setup_logging
import sys
from msa_analysis_plotting import *
from mozaik.storage.queries import *

# Usage: python run_analysis_patterned_stimulation.py path/to/spontaneous_activity_datastore path/to/fullfield_stimulation_datastore path/to/endogenous_stimulation_datastore path/to/surrogate_stimulation_datastore

# Which analysis / plotting to run
spont_analysis = True
fullfield_opto_analysis = True
endogenous_opto_analysis = True
surrogate_opto_analysis = True
plotting = True

Global.root_directory = sys.argv[1]+'/'

setup_logging()
data_store = PickledDataStore(load=True, parameters=ParameterSet(
    {'root_directory': sys.argv[1], 'store_stimuli': False}), replace=False)

data_store_ff = PickledDataStore(load=True, parameters=ParameterSet(
    {'root_directory': sys.argv[2], 'store_stimuli': False}), replace=False)

data_store_endo = PickledDataStore(load=True, parameters=ParameterSet(
    {'root_directory': sys.argv[3], 'store_stimuli': False}), replace=False)

data_store_surr = PickledDataStore(load=True, parameters=ParameterSet(
    {'root_directory': sys.argv[4], 'store_stimuli': False}), replace=False)

# Spontaneous activity
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
    queries.param_filter_query(data_store,analysis_algorithm="DistributionOfActivityAcrossOrientationDomainAnalysis").remove_ads_from_datastore()
    DistributionOfActivityAcrossOrientationDomainAnalysis(queries.param_filter_query(data_store,sheet_name=["V1_Exc_L2/3","V1_Inh_L2/3"],st_name='InternalStimulus'),ParameterSet({
        "or_map_dsv": data_store,
        "use_stim_pattern_mask": False,
        "n_orientation_bins": 30,
    })).analyse()
    data_store.save()

# Fullfield stimulation
if fullfield_opto_analysis:
    queries.param_filter_query(data_store_ff,analysis_algorithm="RecordingArrayTimecourse").remove_ads_from_datastore()
    RecordingArrayTimecourse(param_filter_query(data_store_ff,sheet_name=["V1_Exc_L2/3"]),
        ParameterSet(
            {
                "s_res": 40,
                "t_res": 50,
                "array_width": 4000,
                "electrode_radius": 50,
            }
        ),
    ).analyse()
    queries.param_filter_query(data_store_ff,analysis_algorithm="SimulatedCalciumSignal").remove_ads_from_datastore()
    SimulatedCalciumSignal(data_store_ff,ParameterSet({
        "reference_dsv": queries.param_filter_query(data_store,analysis_algorithm="SimulatedCalciumSignal",st_name="InternalStimulus"),
        "spatial_profile_path": "calcium_light_spread_kernel.npy",
    })).analyse()
    queries.param_filter_query(data_store_ff,analysis_algorithm="GaussianBandpassFilter").remove_ads_from_datastore()
    GaussianBandpassFilter(queries.param_filter_query(data_store_ff,y_axis_name="Calcium imaging signal (normalized)"),ParameterSet({
        "highpass_sigma_um": 195,
        "lowpass_sigma_um": 26,
    })).analyse()

    from mozaik.tools.distribution_parametrization import load_parameters
    dsv = queries.param_filter_query(data_store_ff,st_direct_stimulation_name="OpticalStimulatorArrayChR")
    trials = max([load_parameters(s.replace("MozaikExtended",""))["trial"] for s in dsv.get_stimuli()]) + 1
    offset_time = load_parameters(dsv.get_stimuli()[0].replace("MozaikExtended",""))['direct_stimulation_parameters']['stimulating_signal_parameters']['offset_time']
    A_bandpass_ff = np.stack([np.array(queries.param_filter_query(data_store_ff,analysis_algorithm="GaussianBandpassFilter",st_trial=trial,st_direct_stimulation_name="OpticalStimulatorArrayChR",sheet_name="V1_Exc_L2/3").get_analysis_result()[0].analog_signal) for trial in range(trials)])
    if 1:
        # Only take last frame of optogenetic stimulation
        tags = queries.param_filter_query(data_store_ff,analysis_algorithm="GaussianBandpassFilter",st_direct_stimulation_name="OpticalStimulatorArrayChR").get_analysis_result()[0].tags
        t_res = int([t for t in tags if t[:5] == 't_res'][0].split(":")[-1])
        end_of_stim_index = int(offset_time / t_res)
        data_store_ff.full_datastore.add_analysis_result(
                AnalogSignal(
                NeoAnalogSignal(A_bandpass_ff[:,end_of_stim_index,:,:], t_start=0, sampling_period=50*qt.ms,units=munits.spike / qt.s),
                y_axis_units=munits.spike / qt.s,
                tags=tags,
                sheet_name="V1_Exc_L2/3",
                stimulus_id=None,
                analysis_algorithm="GaussianBandpassFilter",
            )
        )
    queries.param_filter_query(data_store_ff,analysis_algorithm="CorrelationMaps").remove_ads_from_datastore()
    CorrelationMaps(queries.param_filter_query(data_store_ff,analysis_algorithm="GaussianBandpassFilter",stimulus_id=None),ParameterSet({})).analyse()
    queries.param_filter_query(data_store_ff,analysis_algorithm="OrientationMapSimilarity").remove_ads_from_datastore()
    OrientationMapSimilarity(queries.param_filter_query(data_store_ff,analysis_algorithm="CorrelationMaps"),ParameterSet({
        "or_map_dsv": queries.param_filter_query(data_store,sheet_name="V1_Exc_L2/3"),
    })).analyse()
    queries.param_filter_query(data_store_ff,analysis_algorithm="CorrelationMapSimilarity").remove_ads_from_datastore()
    CorrelationMapSimilarity(queries.param_filter_query(data_store_ff),ParameterSet({
        "corr_map_dsv": queries.param_filter_query(data_store,st_name='InternalStimulus',sheet_name="V1_Exc_L2/3"),
        "exclusion_radius": 400,
    })).analyse()

    data_store_ff.save()

# Endogenous stimulation
if endogenous_opto_analysis:
    queries.param_filter_query(data_store_endo,analysis_algorithm="RecordingArrayTimecourse").remove_ads_from_datastore()
    RecordingArrayTimecourse(param_filter_query(data_store_endo,sheet_name=["V1_Exc_L2/3"]),
        ParameterSet(
            {
                "s_res": 40,
                "t_res": 50,
                "array_width": 4000,
                "electrode_radius": 50,
            }
        ),
    ).analyse()
    queries.param_filter_query(data_store_endo,analysis_algorithm="SimulatedCalciumSignal").remove_ads_from_datastore()
    SimulatedCalciumSignal(data_store_endo,ParameterSet({
        "reference_dsv": queries.param_filter_query(data_store, sheet_name="V1_Exc_L2/3",y_axis_name="Calcium imaging signal",st_name="InternalStimulus"),
        "spatial_profile_path": "calcium_light_spread_kernel.npy",
    })).analyse()

    queries.param_filter_query(data_store_endo,analysis_algorithm="GaussianBandpassFilter").remove_ads_from_datastore()
    GaussianBandpassFilter(queries.param_filter_query(data_store_endo,y_axis_name="Calcium imaging signal (normalized)"),ParameterSet({
        "highpass_sigma_um": 200,
        "lowpass_sigma_um": 26,
    })).analyse()

    queries.param_filter_query(data_store_endo,analysis_algorithm="SaveStimPatterns").remove_ads_from_datastore()
    SaveStimPatterns(queries.param_filter_query(data_store_endo,analysis_algorithm="RecordingArrayTimecourse"),ParameterSet({
    })).analyse()

    queries.param_filter_query(data_store_endo,analysis_algorithm="DistributionOfActivityAcrossOrientationDomainAnalysis").remove_ads_from_datastore()
    DistributionOfActivityAcrossOrientationDomainAnalysis(queries.param_filter_query(data_store_endo,sheet_name=["V1_Exc_L2/3","V1_Inh_L2/3"],analysis_algorithm='RecordingArrayTimecourse'),ParameterSet({
        "or_map_dsv": data_store,
        "use_stim_pattern_mask": True,
        "n_orientation_bins": 30,
    })).analyse()

    data_store_endo.save()

# Surrogate stimulation
if surrogate_opto_analysis:
    queries.param_filter_query(data_store_surr,analysis_algorithm="RecordingArrayTimecourse").remove_ads_from_datastore()
    RecordingArrayTimecourse(param_filter_query(data_store_surr,sheet_name=["V1_Exc_L2/3"]),
        ParameterSet(
            {
                "s_res": 40,
                "t_res": 50,
                "array_width": 4000,
                "electrode_radius": 50,
            }
        ),
    ).analyse()
    queries.param_filter_query(data_store_surr,analysis_algorithm="SimulatedCalciumSignal").remove_ads_from_datastore()
    SimulatedCalciumSignal(data_store_surr,ParameterSet({
        "reference_dsv": queries.param_filter_query(data_store, sheet_name="V1_Exc_L2/3",y_axis_name="Calcium imaging signal",st_name="InternalStimulus"),
        "spatial_profile_path": "calcium_light_spread_kernel.npy",
    })).analyse()

    queries.param_filter_query(data_store_surr,analysis_algorithm="GaussianBandpassFilter").remove_ads_from_datastore()
    GaussianBandpassFilter(queries.param_filter_query(data_store_surr,y_axis_name="Calcium imaging signal (normalized)"),ParameterSet({
        "highpass_sigma_um": 200,
        "lowpass_sigma_um": 26,
    })).analyse()

    queries.param_filter_query(data_store_surr,analysis_algorithm="SaveStimPatterns").remove_ads_from_datastore()
    SaveStimPatterns(queries.param_filter_query(data_store_surr,analysis_algorithm="RecordingArrayTimecourse"),ParameterSet({
    })).analyse()

    queries.param_filter_query(data_store_surr,analysis_algorithm="DistributionOfActivityAcrossOrientationDomainAnalysis").remove_ads_from_datastore()
    DistributionOfActivityAcrossOrientationDomainAnalysis(queries.param_filter_query(data_store_surr,sheet_name=["V1_Exc_L2/3","V1_Inh_L2/3"],analysis_algorithm='RecordingArrayTimecourse'),ParameterSet({
        "or_map_dsv": data_store,
        "use_stim_pattern_mask": True,
        "n_orientation_bins": 30,
    })).analyse()

    data_store_surr.save()

if plotting:
    PatternedOptogeneticStimulationPlot(
        data_store,
        ParameterSet({
            "fullfield_stim_dsv": data_store_ff,
            "endogenous_stim_dsv": data_store_endo,
            "surrogate_stim_dsv": data_store_surr,
        }),
        fig_param={"dpi": 100, "figsize": (11, 8)},
        plot_file_name="PatternedOptogeneticStimulationPlot.png",
    ).plot()
    IndividualDAODPlot(
        data_store,
        ParameterSet({
            "endogenous_stim_dsv": data_store_endo,
            "surrogate_stim_dsv": data_store_surr,
        }),
        fig_param={"dpi": 100, "figsize": (8, 11)},
        plot_file_name="IndividualDAODPlot.png",
    ).plot()
