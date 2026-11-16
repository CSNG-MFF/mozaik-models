# -*- coding: utf-8 -*-
"""Simulate and export one Randomized Experanto chunk.

CHUNK selects a chunk; the model's trial parameter selects its trial.
RUN_CONFIG names the launcher's run_config.json, containing input paths,
shared export settings and run provenance. Manual runs use CHUNK_DIR and BASE_PATH.
SHEET_NAMES selects comma-separated sheets for a manual run without RUN_CONFIG.

Run from the LSV1M directory with the usual Mozaik simulation arguments::

    CHUNK=0 CHUNK_DIR=chunks BASE_PATH=dataset \
        mpirun -n 4 python run_experanto.py nest 4 param_experanto/defaults \
            results_dir "'runs/manual/'" simulation_seed 1 trial 0 trial0_chunk0

MPI root exports into <datastore>/experanto after run_workflow saves the datastore.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")

from model import SelfSustainedPushPull
from experiments import create_randomized_experanto
import mozaik
from mozaik.controller import run_workflow
from mozaik.meta_workflow.experanto_export import (
    ExperantoExportSettings,
    export_experanto_chunk,
)
from pyNN import nest

import nest

nest.Install("stepcurrentmodule")

chunk = int(os.environ.get("CHUNK", 0))

provenance = None
if os.environ.get("RUN_CONFIG"):
    with open(os.environ["RUN_CONFIG"]) as stream:
        provenance = json.load(stream)
    chunk_dir = provenance["settings"]["chunks_dir"]
    base_path = provenance["settings"]["data_root"]
    export_options = provenance["settings"]["export_settings"]
else:
    chunk_dir = os.path.abspath(os.environ["CHUNK_DIR"])
    base_path = os.path.abspath(os.environ["BASE_PATH"])
    sheet_names = [
        name.strip()
        for name in os.environ.get("SHEET_NAMES", "").split(",")
        if name.strip()
    ] or None
    export_options = {"sheet_names": sheet_names}

print(f"Simulating chunk {chunk} (dataset {base_path})")
data_store, model = run_workflow(
    "SelfSustainedPushPull",
    SelfSustainedPushPull,
    lambda model: create_randomized_experanto(
        model,
        os.path.join(chunk_dir, f"{model.parameters.trial}_{chunk}.json"),
        base_path,
    ),
)

if mozaik.mpi_comm is None or mozaik.mpi_comm.rank == mozaik.MPI_ROOT:
    export_settings = ExperantoExportSettings(**export_options)
    export_experanto_chunk(
        data_store,
        trial_id=model.parameters.trial,
        chunk_id=chunk,
        chunk_path=os.path.join(chunk_dir, f"{model.parameters.trial}_{chunk}.json"),
        settings=export_settings,
        provenance=provenance,
    )
