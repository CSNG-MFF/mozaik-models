# -*- coding: utf-8 -*-
"""
Simulate one chunk of a Randomized Experanto experiment.

This is the LSV1M model presented with the stimuli of an Experanto screen dataset rather than
with gratings or natural images: it reuses model.py, or_map_new_16x16 and the connectivity of
param/ unchanged, and differs only in param_experanto/ (all spikes recorded, no implicit null
stimulus, LGN resolution matched to the screen) and in the experiment it runs.

One invocation simulates one (trial, chunk) pair. Which one is selected by the environment --
see "Environment" below -- because a chunk index is not something the model knows about and
so cannot be a mozaik parameter. Everything that is constant for a run is an ordinary
experiment parameter in experiments.py instead.

Usage
-----
    TRIAL=0 CHUNK=0 CHUNK_DIR=/path/to/chunks BASE_PATH=/path/to/dataset \
        mpirun -n 12 python run_experanto.py nest 12 param_experanto/defaults \
            results_dir "'/path/to/out/'" simulation_seed 1 trial0_chunk0 [--export]

Run the whole experiment with run_parameter_search_experanto.py instead, which builds the
chunk lists, submits one job per (trial, chunk), and queues the exports behind them.

Environment
-----------
TRIAL, CHUNK   Which chunk to simulate; selects {CHUNK_DIR}/{TRIAL}_{CHUNK}.json.
CHUNK_DIR      Directory holding the chunk lists.
BASE_PATH      Experanto screen dataset the stimuli are read from; defaults to
               DEFAULT_BASE_PATH below. Must be the dataset the chunk lists were built from.
SHEET_NAMES    Comma-separated sheets to export with --export; unset means all recorded ones.
"""
import os
import sys
from functools import partial

import matplotlib

matplotlib.use("Agg")

from model import SelfSustainedPushPull
from experiments import create_randomized_experanto
import mozaik
from mozaik.controller import run_workflow
from pyNN import nest

import nest

nest.Install("stepcurrentmodule")

# The Experanto screen dataset the stimuli are read from. Only used when BASE_PATH is unset,
# which is never the case under run_parameter_search_experanto.py: it passes the dataset it
# built the chunk lists from. Replace it to run this script by hand.
DEFAULT_BASE_PATH = "[PATH_TO_EXPERANTO_DATASET]"

# Optional inline export: simulate this chunk, then write it out as a full Experanto shard
# beside its datastore, instead of leaving it for a separate export job. Only useful for a
# single-chunk dataset -- a multi-chunk trial has to be concatenated in chunk order, which is
# what export.py does. Stripped from argv before the mozaik CLI parses the positionals.
EXPORT_INLINE = "--export" in sys.argv
if EXPORT_INLINE:
    sys.argv.remove("--export")


def selected_chunk():
    """Return the (trial, chunk, chunk_path) this job was asked to simulate."""
    trial = int(os.environ.get("TRIAL", 0))
    chunk = int(os.environ.get("CHUNK", 0))
    # Absolute, because RandomizedExperanto joins this onto base_path: an absolute path wins
    # that join, a relative one is looked for inside the dataset.
    chunk_dir = os.path.abspath(os.environ.get("CHUNK_DIR", "[PATH_TO_CHUNKS]"))
    return trial, chunk, os.path.join(chunk_dir, "%d_%d.json" % (trial, chunk))


def export_datastore_inline(data_store, trial, chunk_path):
    """
    Export the just-simulated, still in-memory datastore to a full Experanto shard beside it.

    Rank 0 only: PyNN gathers every rank's spikes onto MPI_ROOT, so only rank 0's store holds
    the whole network.
    """
    from mozaik.meta_workflow.experanto_export import export_dsvs_to_experanto
    from mozaik.storage.queries import param_filter_query

    experiment_dir = os.path.join(data_store.parameters.root_directory, "experanto")
    print(
        "[inline-export] trial=%d -> %s (chunk_json=%s)"
        % (trial, experiment_dir, chunk_path),
        flush=True,
    )

    # No st_name filter, so the blank (InternalStimulus) segments are kept and the spike
    # timeline stays aligned with the screen timeline; no sheet_name filter, so every recorded
    # sheet is folded into spikes.npy. SHEET_NAMES restricts that to a subset.
    dsv = param_filter_query(data_store)
    sheet_env = os.environ.get("SHEET_NAMES", "").strip()
    sheet_names = [s.strip() for s in sheet_env.split(",") if s.strip()] or None

    export_dsvs_to_experanto(
        [dsv],
        experiment_dir,
        trial_id=trial,
        chunk_paths=[chunk_path],
        sheet_names=sheet_names,
        frame_duration_ms=7.0,
        movie_frame_duration_ms=35.0,
    )
    print("[inline-export] done -> %s" % experiment_dir, flush=True)


trial, chunk, chunk_path = selected_chunk()
base_path = os.environ.get("BASE_PATH", DEFAULT_BASE_PATH)
print(
    "Simulating trial %d chunk %d from %s (dataset %s)"
    % (trial, chunk, chunk_path, base_path)
)

data_store, model = run_workflow(
    "SelfSustainedPushPull",
    SelfSustainedPushPull,
    partial(create_randomized_experanto, chunk_path=chunk_path, base_path=base_path),
)
data_store.save()

if EXPORT_INLINE and mozaik.mpi_comm.rank == mozaik.MPI_ROOT:
    export_datastore_inline(data_store, trial, chunk_path)
