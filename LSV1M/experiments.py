#!/usr/local/bin/ipython -i
from mozaik.experiments import *
from mozaik.experiments.vision import *
from mozaik.sheets.population_selector import RCRandomPercentage
from parameters import ParameterSet


def create_experiments(model):

    return [
        # Lets kick the network up into activation

        # Spontaneous Activity
        NoStimulation(model, ParameterSet(
            {'duration': 3*8*2*5*3*8*7})),

        # Measure orientation tuning with full-filed sinusoidal gratins
        MeasureOrientationTuningFullfield(model, ParameterSet(
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


def create_randomized_experanto(model, chunk_path, base_path, width=11.0):
    """
    Present the stimuli listed in one chunk of an Experanto screen dataset.

    A chunk is one job's worth of an experiment that is far too long for a single job: the
    stimuli of a trial are shuffled and split into time-balanced chunks by
    mozaik.tools.experanto_chunks, each simulated separately, and the resulting datastores
    are concatenated back into one Experanto shard per trial by export.py.

    Parameters
    ----------
    chunk_path : str
        Absolute path of the chunk JSON listing the stimuli this job presents. It must be
        absolute: RandomizedExperanto joins it onto base_path, so a relative path would be
        silently resolved inside the dataset.

    base_path : str
        Root of the Experanto screen dataset the stimuli are read from (containing
        screen/meta and screen/data). It must be the dataset the chunk lists were built
        from, or they will name stimuli that do not exist here.

    width : float
        Angular size of the longest axis of the presented image, in degrees. The default
        matches the model's visual field (11 x 11 degrees), so the stimulus fills it; a
        dataset whose stimuli subtended a different angle needs its own value, and gets it
        by binding this argument rather than by editing this function.
    """
    return [
        RandomizedExperanto(
            model,
            ParameterSet(
                {
                    "base_path": base_path,
                    "chunk_dict_path": chunk_path,
                    "width": width,
                    "movie_frame_duration": 35,
                    "global_frame_offset": 0,
                    "images_per_trial": 300,
                    "shuffle_stimuli": False,
                    "video_max_value": 255.0,
                }
            ),
        ),
    ]
