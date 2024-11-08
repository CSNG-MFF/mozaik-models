#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.experiments.optogenetic import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet
from copy import deepcopy

def create_experiments(model):
    return [
        # Spontaneous Activity
        NoStimulation(model, ParameterSet(
            {'duration': 3*8*2*5*3*8*7})),

        # Measure orientation tuning with full-filed sinusoidal gratins
        MeasureOrientationTuningFullfield(model, ParameterSet(
        #    {'num_orientations': 1, 'spatial_frequency': 0.8, 'temporal_frequency': 2, 'grating_duration': 2*143*7, 'contrasts': [100], 'num_trials':10, 'shuffle_stimuli': True})),
            {'num_orientations': 10, 'spatial_frequency': 0.8, 'temporal_frequency': 2, 'grating_duration': 2*143*7, 'contrasts': [10,30,100], 'num_trials':10, 'shuffle_stimuli': True})),

        # Measure response to natural image with simulated eye movement
        MeasureNaturalImagesWithEyeMovement(model, ParameterSet(
            {'stimulus_duration': 2*143*7, 'num_trials': 10, 'size':30, 'shuffle_stimuli': False})),
    ]


def create_experiments_stc(model):
    return [
        # Spontaneous Activity
        NoStimulation(model, ParameterSet({'duration': 2*5*3*8*7})),

        # Size Tuning
        MeasureSizeTuning(model, ParameterSet({'num_sizes': 12, 'max_size': 5.0, 'log_spacing': True, 'orientations': [0], 'positions': [(0,0)],
                                               'spatial_frequency': 0.8, 'temporal_frequency': 2, 'grating_duration': 2*143*7, 'contrasts': [10, 100], 'num_trials': 10, 'shuffle_stimuli': True})),
    ]

def create_experiments_spont(model):
    return [
        # Spontaneous Activity
        NoStimulation(model, ParameterSet(
            {'duration': 3*8*2*5*3*8*7})),
    ]

def create_experiments_spont_STW(model):
    return [
        # Spontaneous Activity
        NoStimulation(model, ParameterSet(
            {'duration': 2*8*5*3*3*4*7})),
    ]

def create_experiments_spont_MSA(model):
    return [
        # Spontaneous Activity
        NoStimulation(model, ParameterSet(
            {'duration': 120 * 1001})),
    ]

def patterned_opt_stim(model, images_path, intensity, num_trials=5, sheet="V1_Exc_L2/3"):
    experiments = []
    mulholland_parameters = {
        "sheet_list": [sheet],
        "sheet_intensity_scaler": [1.0],
        "sheet_transfection_proportion": [1.0],
        "num_trials": num_trials,
        "stimulator_array_parameters": {
            "size": 4000,
            "spacing": 10.0,
            "depth_sampling_step": 10,
            "light_source_light_propagation_data": "light_scattering_radial_profiles_lsd10.pickle",
            "update_interval": 1,
        },
        "intensities": [intensity],
        "duration": 4000,
        "onset_time": 1000,
        "offset_time": 2000,
    }
    p_full = MozaikExtendedParameterSet(deepcopy(mulholland_parameters))
    p_full.images_path = images_path
    experiments.append(OptogeneticArrayImageStimulus(model, p_full))
    return experiments

# We match the stim intensity ratio of 7.6 mW/mm^2 and 10 mW/mm^2 based on
# H. N. Mulholland, H. Jayakumar, D. M. Farinella, G. B. Smith, All-optical interrogation of millimeter-scale networks and application to developing ferret cortex. J. Neurosci. Methods 403, 110051 (2024).
# and
# H. N. Mulholland, M. Kaschube, G. B. Smith, Self-organization of modular activity in immature cortical networks. Nat. Commun. 15, 4145 (2024).
def create_experiments_optogenetic_patterned_stimulation(model):
    patterns = [
        ("optogenetic_stimulation_patterns/fullfield.npy", 0.033, 40),
        ("optogenetic_stimulation_patterns/endogenous", 0.033 * 0.76, 5),
        ("optogenetic_stimulation_patterns/surrogate", 0.033 * 0.76, 5),
    ]
    return sum((patterned_opt_stim(model, pattern, intensity, n_trials) for pattern, intensity, n_trials in patterns), [])

def create_experiments_central_stimulation(model):
    intensities = [9,2.25,1,0.5625,0.36,0.25]
    radii = [50,100,150,200,250,300]
    experiments = []
    for i in range(len(intensities)):
        experiments.append(
            SingleOptogeneticArrayStimulus(
                model,
                MozaikExtendedParameterSet(
                    {
                        "sheet_list": ["V1_Exc_L2/3"],
                        'sheet_intensity_scaler': [1.0],
                        'sheet_transfection_proportion': [1.0],
                        "num_trials": 10,
                        "stimulator_array_parameters": MozaikExtendedParameterSet(
                            {
                                "size": 1000,
                                "spacing": 10,
                                "depth_sampling_step": 10,
                                "light_source_light_propagation_data": "light_scattering_radial_profiles_lsd10.pickle",
                                "update_interval": 1,
                            }
                        ),
                        "stimulating_signal": "mozaik.sheets.direct_stimulator.stimulating_pattern_flash",
                        "stimulating_signal_parameters": ParameterSet(
                            {
                                "shape": "circle",
                                "coords": [[0,0]],
                                "radius": radii[i],
                                "intensity": intensities[i],
                                "duration": 4000,
                                "onset_time": 1000,
                                "offset_time": 2000,
                            }
                        ),
                    }
                ),
            )
    )
    return experiments
