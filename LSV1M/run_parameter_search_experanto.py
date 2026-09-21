# -*- coding: utf-8 -*-
"""
Run a whole Randomized Experanto experiment from one command.

    python run_parameter_search_experanto.py run_experanto.py nest param_experanto/defaults

builds the chunk lists, submits one simulation job per (trial, chunk), and queues one export
job per trial behind that trial's simulations. Nothing has to be run by hand in between.

Edit the settings below before running. --dry-run writes the chunk lists and prints the jobs
it would have submitted, without submitting them.

Layout of a run::

    <master>/chunks/{trial}_{chunk}.json                      the chunk lists
    <master>/SelfSustainedPushPull_trial{t}_chunk{c}_____*/    one datastore per chunk
    <master>/shards/trial{t}/{responses,screen}/               one Experanto shard per trial
"""

import json
import os
import sys

from mozaik.meta_workflow.parameter_search import (
    ParameterSearch,
    SlurmSequentialBackend,
)
from mozaik.tools.experanto_chunks import generate_chunks, scan_screen_metadata

# --- the experiment ---------------------------------------------------------------------
# Relative paths resolve against this directory, where the launcher and every job it submits
# run.
DATA_ROOT = "[PATH_TO_DATASET]"
N_TRIALS = 2
N_CHUNKS = 5  # chunks per trial, so also jobs per trial
CHUNK_SEED = 42  # fixes which stimuli land in which chunk

# --- what this run does -----------------------------------------------------------------
CHUNKS = None  # chunks to simulate now; None runs all of them
RESULTS_DIR = None  # None makes a fresh directory; set it to run in several goes
CHUNK_DIR = None  # None keeps them with the results; written if not there yet

# --- the cluster ------------------------------------------------------------------------
PATH_TO_MOZAIK_ENV = "[PATH_TO_ENV]"
NUM_MPI = 4  # ranks per simulation job
THREADS_PER_RANK = 1  # keep equal to the backend's num_threads
EXPORT_SCRIPT = "export.py"


def _thread_environment(threads):
    """
    The thread-pool caps every numerical library in the job should respect.

    Without them NumPy and its BLAS size their pools from the *machine's* core count, so each
    rank starts as many threads as the node has cores and they fight over the one core the
    rank was actually allocated.
    """
    return {
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "VECLIB_MAXIMUM_THREADS": threads,
    }


def _describe(chunks):
    """Render a chunk list compactly: contiguous runs as "a-b", anything else as a list."""
    if not chunks:
        return "none"
    if chunks == list(range(chunks[0], chunks[-1] + 1)):
        return "%d-%d" % (chunks[0], chunks[-1]) if len(chunks) > 1 else str(chunks[0])
    return ", ".join(str(c) for c in chunks)


class RandomizedExperantoSearch(ParameterSearch):
    """
    Fan a Randomized Experanto experiment out over its (trial, chunk) pairs.

    Unlike a CombinationParameterSearch this does not sweep a model parameter: the jobs differ
    in *which stimuli they present*, which is not something the model knows about. So the
    combinations vary only the two parameters that have to differ anyway -- ``trial``, and
    ``simulation_seed`` to give each trial its own noise -- while the chunk each job presents
    travels in its environment, and the run name carries (trial, chunk) so that the datastores
    can be told apart and found again at export time.
    """

    def __init__(
        self,
        backend,
        data_root,
        n_trials,
        n_chunks=None,
        chunk_dir=None,
        chunk_seed=42,
        chunks=None,
        results_dir=None,
        export_script="export.py",
        threads_per_rank=1,
    ):
        ParameterSearch.__init__(self, backend)
        self.data_root = data_root
        self.n_trials = n_trials
        # How many chunks to *generate*. The authoritative count is discovered from the chunk
        # directory in prepare(), so this is unused when chunk_dir points at existing lists.
        self.generate_n_chunks = n_chunks
        self.chunk_dir = chunk_dir
        self.chunk_seed = chunk_seed
        self.requested_chunks = chunks  # None -> every chunk
        self.results_dir = results_dir  # None -> a fresh timestamped directory
        self.export_script = export_script
        self.threads_per_rank = threads_per_rank
        self.master_dir = None
        self.n_chunks = None  # total, discovered in prepare()
        self.chunks = None  # the subset this invocation runs, resolved in prepare()

    # --- the search ---------------------------------------------------------------------

    def master_directory_name(self):
        return "RandomizedExperanto{trials:%d}/" % self.n_trials

    def master_directory(self, parameters_url):
        """
        A fixed directory when results_dir is set, so that successive invocations -- one per
        slice of the chunk range -- accumulate into it. Without that a later slice could
        neither append to the earlier slice's shard nor see its datastores.
        """
        if self.results_dir is None:
            return ParameterSearch.master_directory(self, parameters_url)
        return self.results_dir

    def prepare(self, master_directory):
        """
        Make sure the chunk lists exist, then take stock of them.

        chunk_dir is simply where they live; they are generated if they are not there yet.
        So the first invocation of a sliced run writes them and every later one reuses them,
        and pointing successive experiments at one directory cannot regenerate it underneath
        datastores that were simulated from the previous contents.
        """
        self.master_dir = os.path.abspath(master_directory)
        # Absolute, because RandomizedExperanto joins the chunk path onto base_path: an
        # absolute path wins that join, a relative one is looked for inside the dataset.
        self.chunk_dir = (
            os.path.join(self.master_dir, "chunks")
            if self.chunk_dir is None
            else os.path.abspath(self.chunk_dir)
        )

        if self._chunk_lists_present():
            print("Reusing chunk lists in %s" % self.chunk_dir)
            reused = True
        else:
            self._generate_chunk_lists()
            reused = False

        # The total always comes from disk rather than from a setting, so that it cannot
        # disagree with what is actually there. The export needs the true total: its screen
        # timeline is rebuilt from EVERY chunk list, even when only a slice is exported.
        self.n_chunks = self._discover_n_chunks()
        if reused:
            self._check_reused_chunk_lists()
        self.chunks = self._resolve_chunks()
        print(
            "Chunk set: %d chunks per trial; this invocation runs %d-%d."
            % (self.n_chunks, self.chunks[0], self.chunks[-1])
        )

    def _chunk_lists_present(self):
        """Whether chunk_dir already holds chunk lists. Emptiness means "generate here"."""
        if not os.path.isdir(self.chunk_dir):
            return False
        return any(n.endswith(".json") for n in os.listdir(self.chunk_dir))

    def _check_reused_chunk_lists(self):
        """
        Reuse is implicit, so the chunk lists have to be checked against what this run means
        to do. Silently running someone else's chunk lists is the one failure here that
        produces plausible-looking results from the wrong stimuli.
        """
        if (
            self.generate_n_chunks is not None
            and self.n_chunks != self.generate_n_chunks
        ):
            raise ValueError(
                "%s holds %d chunks per trial but n_chunks says %d -- point chunk_dir at a "
                "different directory, or set n_chunks to match"
                % (self.chunk_dir, self.n_chunks, self.generate_n_chunks)
            )

        available = {e["file"] for e in scan_screen_metadata(self.data_root)}
        for trial in range(self.n_trials):
            for chunk in range(self.n_chunks):
                path = os.path.join(self.chunk_dir, "%d_%d.json" % (trial, chunk))
                with open(path, "r") as f:
                    named = {item["file"] for item in json.load(f)}
                missing = named - available
                if missing:
                    raise ValueError(
                        "%s presents stimuli that %s does not contain (%s%s) -- these chunk "
                        "lists were built from a different dataset"
                        % (
                            path,
                            self.data_root,
                            ", ".join(sorted(missing)[:3]),
                            ", ..." if len(missing) > 3 else "",
                        )
                    )

    def _discover_n_chunks(self):
        """
        Count the chunks on disk, requiring 0..N-1 to be present for every trial.

        A gap is a half-generated or damaged chunk set rather than a short one, so it raises:
        clamping around it would silently drop stimuli from the experiment.
        """
        totals = set()
        for trial in range(self.n_trials):
            present = {
                c
                for c in range(len(os.listdir(self.chunk_dir)) + 1)
                if os.path.isfile(
                    os.path.join(self.chunk_dir, "%d_%d.json" % (trial, c))
                )
            }
            if not present:
                raise FileNotFoundError(
                    "no chunk lists for trial %d in %s" % (trial, self.chunk_dir)
                )
            expected = set(range(max(present) + 1))
            if present != expected:
                raise ValueError(
                    "chunk lists for trial %d are not contiguous in %s: missing %s"
                    % (trial, self.chunk_dir, sorted(expected - present))
                )
            totals.add(len(present))

        if len(totals) > 1:
            raise ValueError(
                "trials have differing chunk counts in %s: %s"
                % (self.chunk_dir, sorted(totals))
            )
        return totals.pop()

    def _resolve_chunks(self):
        """
        Resolve the requested chunk subset against what exists, clamping to the end.

        Clamping makes "the next ten, whatever is left" a valid way to ask for a final slice
        without knowing the total. An empty result is not a short slice but a mistake, and
        submitting nothing at all is the worst way to report it, so it raises.
        """
        if self.requested_chunks is None:
            return list(range(self.n_chunks))

        requested = list(self.requested_chunks)
        chunks = [c for c in requested if 0 <= c < self.n_chunks]
        dropped = [c for c in requested if c not in chunks]
        if dropped:
            print(
                "WARNING: chunks %s were requested but the chunk set has %d (0-%d); "
                "running %s instead."
                % (
                    _describe(dropped),
                    self.n_chunks,
                    self.n_chunks - 1,
                    _describe(chunks) if chunks else "nothing",
                )
            )
        if not chunks:
            raise ValueError(
                "no requested chunk exists: asked for %s, the chunk set has 0-%d"
                % (_describe(requested), self.n_chunks - 1)
            )
        return chunks

    def _generate_chunk_lists(self):
        if self.generate_n_chunks is None:
            raise ValueError(
                "n_chunks is required when the chunk lists have to be generated: it says "
                "how many to make"
            )
        print(
            "Generating chunk lists from %s into %s" % (self.data_root, self.chunk_dir)
        )
        summary = generate_chunks(
            self.data_root,
            self.chunk_dir,
            n_trials=self.n_trials,
            n_chunks=self.generate_n_chunks,
            seed=self.chunk_seed,
        )
        for entry in summary:
            print(
                "  trial %d chunk %d: %d stimuli (%d img, %d vid), est. %.1f h"
                % (
                    entry["trial"],
                    entry["chunk"],
                    entry["n_stimuli"],
                    entry["n_images"],
                    entry["n_videos"],
                    entry["est_seconds"] / 3600.0,
                )
            )
        longest = max(e["est_seconds"] for e in summary) / 3600.0
        print(
            "Wrote %d chunk lists to %s; longest chunk is an estimated %.1f h -- raise "
            "N_CHUNKS if that does not fit a job's time limit."
            % (len(summary), self.chunk_dir, longest)
        )

    def _trial_chunk(self, combination):
        """Recover (trial, chunk) from a combination this class built."""
        return combination["trial"], (combination["simulation_seed"] - 1) % 1000

    def generate_parameter_combinations(self):
        # simulation_seed must be nonzero (NEST rejects rng_seed=0) and unique per job, so
        # that no two jobs share a result directory name.
        return [
            {"trial": trial, "simulation_seed": trial * 1000 + chunk + 1}
            for trial in range(self.n_trials)
            for chunk in self.chunks
        ]

    def job_environment(self, combination):
        trial, chunk = self._trial_chunk(combination)
        return dict(
            _thread_environment(self.threads_per_rank),
            **{
                "TRIAL": trial,
                "CHUNK": chunk,
                "CHUNK_DIR": self.chunk_dir,
                # The dataset the chunk lists were built from: passing it explicitly is what keeps
                # the simulation from reading a different one than was scanned.
                "BASE_PATH": self.data_root,
            }
        )

    def simulation_run_name(self, combination):
        # export.py resolves a chunk's datastore by globbing
        # "SelfSustainedPushPull_trial{t}_chunk{c}_____*", so the run name has to spell that out.
        trial, chunk = self._trial_chunk(combination)
        return "trial%d_chunk%d" % (trial, chunk)

    # --- the export wave ----------------------------------------------------------------

    def submit_exports(self, job_ids):
        """
        Queue one export job per trial, each held until that trial's simulations succeed.

        Per-trial rather than one dependency on everything, so that a chunk failing in one
        trial does not hold up the export of the trials that finished.
        """
        if self.master_dir is None:
            raise RuntimeError(
                "submit_exports must be called after run_parameter_search"
            )

        output_prefix = os.path.join(self.master_dir, "shards", "trial")
        # The exported slice is contiguous, which is what the spike exporter's append mode
        # requires: chunk k's offset comes from the end_time chunk k-1 left behind.
        chunk_start, chunk_end = self.chunks[0], self.chunks[-1] + 1
        per_trial = len(self.chunks)
        for trial in range(self.n_trials):
            first = trial * per_trial
            depends_on = job_ids[first : first + per_trial]
            if any(j is None for j in depends_on):
                print(
                    "Trial %d: not every simulation reported a job id, so its export cannot "
                    "be made to wait for them. Run it by hand once they finish:\n"
                    "    CHUNK_DIR=%s DATASTORE_PREFIX=%s OUTPUT_PREFIX=%s "
                    "python -u %s %d --n-chunks %d --chunk-start %d --chunk-end %d"
                    % (
                        trial,
                        self.chunk_dir,
                        self.master_dir,
                        output_prefix,
                        self.export_script,
                        trial,
                        self.n_chunks,
                        chunk_start,
                        chunk_end,
                    )
                )
                continue

            self.backend.execute_script(
                [
                    "python",
                    "-u",
                    self.export_script,
                    str(trial),
                    # the TOTAL, so the screen timeline spans every chunk...
                    "--n-chunks",
                    str(self.n_chunks),
                    # ...while the spikes cover only what this invocation simulated, appending
                    # to the shard an earlier slice left behind when chunk_start > 0.
                    "--chunk-start",
                    str(chunk_start),
                    "--chunk-end",
                    str(chunk_end),
                ],
                env=dict(
                    _thread_environment(self.threads_per_rank),
                    CHUNK_DIR=self.chunk_dir,
                    DATASTORE_PREFIX=self.master_dir,
                    OUTPUT_PREFIX=output_prefix,
                ),
                depends_on=depends_on,
                # Alongside the simulations' slurm-%j.out, so a run's whole log trail is in
                # one directory; named apart so the two waves stay distinguishable.
                output=os.path.join(self.master_dir, "slurm-export-%j.out"),
            )
        print("Submitted %d export jobs." % self.n_trials)


if __name__ == "__main__":
    # Stripped before mozaik's CLI parses the positionals, which is why it can sit anywhere on
    # the command line (the same trick run_experanto.py uses for --export).
    DRY_RUN = "--dry-run" in sys.argv
    if DRY_RUN:
        sys.argv.remove("--dry-run")

    search = RandomizedExperantoSearch(
        SlurmSequentialBackend(
            num_threads=1,
            num_mpi=NUM_MPI,
            # --nodes=1 keeps a job's MPI ranks on a single node. Without it slurm is free to
            # scatter them, and NEST then exchanges spikes over the interconnect every
            # timestep, which is far slower than shared memory.
            slurm_options=["--hint=nomultithread", "--nodes=1"],
            path_to_mozaik_env=PATH_TO_MOZAIK_ENV,
            dry_run=DRY_RUN,
        ),
        data_root=DATA_ROOT,
        n_trials=N_TRIALS,
        n_chunks=N_CHUNKS,
        chunk_dir=CHUNK_DIR,
        chunk_seed=CHUNK_SEED,
        chunks=CHUNKS,
        results_dir=RESULTS_DIR,
        export_script=EXPORT_SCRIPT,
        threads_per_rank=THREADS_PER_RANK,
    )
    job_ids = search.run_parameter_search()
    search.submit_exports(job_ids)
