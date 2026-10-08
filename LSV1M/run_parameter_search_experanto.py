# -*- coding: utf-8 -*-
"""
Run a whole Randomized Experanto experiment from one command.

    python run_parameter_search_experanto.py run_experanto.py nest param_experanto/defaults \\
        --data-root /path/to/dataset --n-trials 2 --n-chunks 5 \\
        --mozaik-env /path/to/venv/bin/activate

builds the chunk lists, submits one simulation job per (trial, chunk), and queues one export
job per trial behind that trial's simulations. Nothing has to be run by hand in between.
--help lists every setting and its default.

The settings can be given in a YAML config file instead. It has to list every one of them,
the three positionals included, and nothing else may then be given on the command line, so
that there is never a question of which of the two wins::

    python run_parameter_search_experanto.py --config exp1.yml

    # exp1.yml
    run_script: run_experanto.py
    simulator: nest
    parameters_url: param_experanto/defaults
    data_root: /path/to/dataset
    n_trials: 2
    n_chunks: 30
    chunks: 0-9          # inclusive; also "a,b,c", one chunk, a list, or null for all
    results_dir: runs/exp1
    chunks_dir: null
    num_mpi: 4
    threads_per_rank: 1
    mozaik_env: /path/to/venv/bin/activate
    chunk_seed: 42

Relative paths resolve against the directory the launcher is run from, where every job it
submits runs too. --dry-run writes the chunk lists and prints the jobs it would have submitted,
without submitting them; it is allowed alongside --config.

Rounds
------
A run can be simulated in rounds -- chunks 0-9 now, 10-19 once they have finished -- by giving
every round the same results_dir. Rounds have to give exactly the result simulating every chunk
at once would, so nothing but `chunks` may differ between them: the first round records the
run's settings in <results_dir>/run_config.json, and a later round whose settings differ is
refused before anything is written or submitted.

Provenance
----------
    <chunks_dir>/chunk_settings.json    what the chunk lists were generated from
    <results_dir>/run_config.json       the run's settings, thread caps and simulation seeds
    <shard>/meta.json                   the same, for the chunks the shard holds -- what
                                        remains once the datastores are deleted

Chunk lists reused from elsewhere have to match the run's settings according to their
chunk_settings.json. Lists written before that file existed have none: --confirm-chunk-settings
then regenerates them from the run's settings, checks they come out identical, and records
the settings.

Layout of a run::

    <master>/chunks/{trial}_{chunk}.json                      the chunk lists
    <master>/SelfSustainedPushPull_trial{t}_chunk{c}_____*/    one datastore per chunk
    <master>/shards/trial{t}/{responses,screen}/               one Experanto shard per trial
"""

import argparse
import json
import os
import shlex
import sys

import yaml

from mozaik.meta_workflow.parameter_search import (
    ParameterSearch,
    SlurmSequentialBackend,
)
from mozaik.tools.experanto_chunks import (
    chunk_list_names,
    chunk_settings,
    generate_chunks,
    read_chunk_settings,
    scan_screen_metadata,
    verify_chunk_lists,
    write_chunk_settings,
)

EXPORT_SCRIPT = "export.py"

# Written into the results directory by a run's first round; see "Rounds" above.
RUN_CONFIG_FILE = "run_config.json"

# mozaik's own parameter-search command line: the script every job runs, the simulator, and
# the root parameter file.
POSITIONALS = ("run_script", "simulator", "parameters_url")

REQUIRED = object()

# Every other setting of a run, with its command-line default (REQUIRED for none) and help.
# A config file lists all of them instead of relying on these defaults.
SETTINGS = (
    ("data_root", REQUIRED, "the Experanto screen dataset the stimuli are read from"),
    ("n_trials", REQUIRED, "number of trials, i.e. presentations of the whole stimulus set"),
    (
        "n_chunks",
        None,
        "chunks per trial, so also jobs per trial; needed only when the chunk lists have "
        "to be generated",
    ),
    (
        "chunks",
        None,
        "the chunks to simulate now: 'a-b' (inclusive), 'a,b,c' or a single chunk, without "
        "gaps; all of them if not given",
    ),
    (
        "results_dir",
        None,
        "directory of the run; a fresh one if not given. Give it to run in rounds",
    ),
    (
        "chunks_dir",
        None,
        "where the chunk lists are kept, written if not there yet; inside the run's "
        "directory if not given",
    ),
    ("num_mpi", 4, "MPI ranks per simulation job"),
    (
        "threads_per_rank",
        1,
        "threads per MPI rank, which also caps every numerical library's thread pool",
    ),
    ("mozaik_env", REQUIRED, "activate script of the environment mozaik is installed in"),
    ("chunk_seed", 42, "fixes which stimuli land in which chunk"),
)
SETTING_NAMES = tuple(name for name, _, _ in SETTINGS)

_INTEGERS = ("n_trials", "n_chunks", "num_mpi", "threads_per_rank", "chunk_seed")
_AT_LEAST_ONE = ("n_trials", "n_chunks", "num_mpi", "threads_per_rank")
_PATHS = (
    "run_script",
    "parameters_url",
    "data_root",
    "results_dir",
    "chunks_dir",
    "mozaik_env",
)
_NULLABLE = ("n_chunks", "chunks", "results_dir", "chunks_dir")


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


def _simulation_seed(trial, chunk):
    """
    The simulation seed of a (trial, chunk) job: nonzero, since NEST rejects rng_seed=0, and
    unique per job, so that no two jobs share a result directory name.
    """
    return trial * 1000 + chunk + 1


def _describe(chunks):
    """Render a chunk list compactly: contiguous runs as "a-b", anything else as a list."""
    if not chunks:
        return "none"
    if chunks == list(range(chunks[0], chunks[-1] + 1)):
        return "%d-%d" % (chunks[0], chunks[-1]) if len(chunks) > 1 else str(chunks[0])
    return ", ".join(str(c) for c in chunks)


def parse_chunks(value):
    """
    Parse a chunk selection -- "a-b" (inclusive), "a,b,c", a single chunk, a list of chunks,
    or None for every chunk -- into a list of chunks, or None.

    A selection has to be contiguous: a round's chunks are appended to the shard in order, so
    a gap would be exported as if it had been simulated.
    """
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            raise ValueError
        if isinstance(value, int):
            chunks = [value]
        elif isinstance(value, str):
            text = value.replace(" ", "")
            if "-" in text:
                first, last = text.split("-")
                chunks = list(range(int(first), int(last) + 1))
            else:
                chunks = [int(c) for c in text.split(",")]
        elif isinstance(value, list) and all(
            isinstance(c, int) and not isinstance(c, bool) for c in value
        ):
            chunks = list(value)
        else:
            raise ValueError
    except ValueError:
        raise ValueError("chunks: %r is not a chunk selection" % (value,))

    if not chunks or chunks[0] < 0:
        raise ValueError("chunks: %r selects no chunk" % (value,))
    if chunks != list(range(chunks[0], chunks[-1] + 1)):
        raise ValueError(
            "chunks: %r is not a contiguous, ascending range of chunks" % (value,)
        )
    return chunks


def resolve_settings(raw):
    """
    Check the settings of a run, the positionals included, and resolve them: paths made
    absolute against the working directory, and chunks parsed with :func:`parse_chunks`.

    Absolute, because they are what is recorded, and a path relative to wherever the launcher
    happened to be run from would name a different file once read anywhere else.
    """
    settings = {}
    for name in POSITIONALS + SETTING_NAMES:
        value = raw[name]
        if value is None and name in _NULLABLE:
            settings[name] = None
        elif name == "chunks":
            settings[name] = parse_chunks(value)
        elif name in _INTEGERS:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError("%s: %r is not an integer" % (name, value))
            if name in _AT_LEAST_ONE and value < 1:
                raise ValueError("%s: has to be at least 1, not %d" % (name, value))
            settings[name] = value
        else:
            if not isinstance(value, str) or not value:
                raise ValueError("%s: %r is not a non-empty string" % (name, value))
            settings[name] = os.path.abspath(value) if name in _PATHS else value
    return settings


def load_config(path):
    """
    Read the raw settings of a run from a YAML config file, which has to list every setting,
    the positionals included, and nothing else.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("%s does not hold a mapping of settings" % path)

    expected = POSITIONALS + SETTING_NAMES
    missing = [name for name in expected if name not in raw]
    unknown = sorted(str(name) for name in raw if name not in expected)
    if missing or unknown:
        raise ValueError(
            "%s has to list every setting and nothing else:%s%s"
            % (
                path,
                " missing " + ", ".join(missing) + ";" if missing else "",
                " unknown " + ", ".join(unknown) if unknown else "",
            )
        )
    return raw


def parse_command_line(argv=None):
    """
    Read the settings of a run from the command line, or from the config file it names.

    Returns
    -------
    tuple
        ``(settings, command, dry_run, confirm_chunk_settings)``: the settings resolved by
        :func:`resolve_settings`, and ``command``, the three positionals as given -- what
        mozaik's own parameter-search command line is then made of.
    """
    parser = argparse.ArgumentParser(
        description="Run a whole Randomized Experanto experiment: chunk lists, one "
        "simulation job per (trial, chunk), and one export job per trial.",
    )
    for name in POSITIONALS:
        parser.add_argument(name, nargs="?", help="required unless --config is given")
    for name, default, help in SETTINGS:
        if default is REQUIRED:
            help += " (required)"
        elif default is not None:
            help += " (default: %s)" % default
        parser.add_argument(
            "--" + name.replace("_", "-"),
            dest=name,
            type=int if name in _INTEGERS else str,
            default=argparse.SUPPRESS,
            help=help,
        )
    parser.add_argument(
        "--config",
        help="YAML file listing every setting, the positionals included; excludes giving "
        "any of them on the command line",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the chunk lists and print the jobs instead of submitting them",
    )
    parser.add_argument(
        "--confirm-chunk-settings",
        action="store_true",
        help="for chunk lists written without chunk_settings.json: regenerate them from "
        "this run's settings, check they come out identical, and record the settings",
    )
    args = vars(parser.parse_args(argv))

    try:
        if args["config"] is not None:
            given = [name for name in POSITIONALS if args[name] is not None] + [
                "--" + name.replace("_", "-") for name in SETTING_NAMES if name in args
            ]
            if given:
                parser.error(
                    "--config cannot be combined with %s: the config file has to be the "
                    "only source of the settings" % ", ".join(given)
                )
            raw = load_config(args["config"])
        else:
            missing = [name for name in POSITIONALS if args[name] is None] + [
                "--" + name.replace("_", "-")
                for name, default, _ in SETTINGS
                if default is REQUIRED and name not in args
            ]
            if missing:
                parser.error("missing %s" % ", ".join(missing))
            raw = {name: args[name] for name in POSITIONALS}
            raw.update(
                (name, args.get(name, default)) for name, default, _ in SETTINGS
            )
        settings = resolve_settings(raw)
    except (OSError, ValueError, yaml.YAMLError) as e:
        parser.error(str(e))

    command = [str(raw[name]) for name in POSITIONALS]
    return settings, command, args["dry_run"], args["confirm_chunk_settings"]


def make_backend(settings, dry_run=False):
    """The slurm backend the jobs of a run with these *settings* are submitted through."""
    return SlurmSequentialBackend(
        num_threads=settings["threads_per_rank"],
        num_mpi=settings["num_mpi"],
        # --nodes=1 keeps a job's MPI ranks on a single node. Without it slurm is free to
        # scatter them, and NEST then exchanges spikes over the interconnect every
        # timestep, which is far slower than shared memory.
        slurm_options=["--hint=nomultithread", "--nodes=1"],
        path_to_mozaik_env=settings["mozaik_env"],
        dry_run=dry_run,
    )


class RandomizedExperantoSearch(ParameterSearch):
    """
    Fan a Randomized Experanto experiment out over its (trial, chunk) pairs.

    Unlike a CombinationParameterSearch this does not sweep a model parameter: the jobs differ
    in *which stimuli they present*, which is not something the model knows about. So the
    combinations vary only the two parameters that have to differ anyway -- ``trial``, and
    ``simulation_seed`` to give each trial its own noise -- while the chunk each job presents
    travels in its environment, and the run name carries (trial, chunk) so that the datastores
    can be told apart and found again at export time.

    *settings* are those of :func:`resolve_settings`; the backend has to be built from the same
    settings (see :func:`make_backend`), since they are what is recorded.
    """

    def __init__(
        self,
        backend,
        settings,
        export_script="export.py",
        confirm_chunk_settings=False,
    ):
        ParameterSearch.__init__(self, backend)
        self.settings = settings
        self.data_root = settings["data_root"]
        self.n_trials = settings["n_trials"]
        # How many chunks to *generate*. The authoritative count is discovered from the chunk
        # directory in prepare(), so this is unused when chunk_dir points at existing lists.
        self.generate_n_chunks = settings["n_chunks"]
        self.chunk_dir = settings["chunks_dir"]
        self.chunk_seed = settings["chunk_seed"]
        self.requested_chunks = settings["chunks"]  # None -> every chunk
        self.results_dir = settings["results_dir"]  # None -> a fresh timestamped directory
        self.threads_per_rank = settings["threads_per_rank"]
        self.export_script = export_script
        self.confirm_chunk_settings = confirm_chunk_settings
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

    @property
    def run_config_path(self):
        return os.path.join(self.master_dir, RUN_CONFIG_FILE)

    def prepare(self, master_directory):
        """
        Check this round against the run, make sure the chunk lists exist, then take stock of
        them, and record the run if this is its first round.

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
        # Before anything is written: a round that does not belong to this run must not
        # generate chunk lists into it either. The count of chunks is checked once known.
        self._check_run(self.run_record(self.generate_n_chunks))

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

        record = self.run_record(self.n_chunks)
        self._check_run(record)
        if not os.path.isfile(self.run_config_path) and not self.backend.dry_run:
            with open(self.run_config_path, "w") as f:
                json.dump(record, f, indent=4)
            print("Recorded the run's settings in %s" % self.run_config_path)
        print(
            "Chunk set: %d chunks per trial; this invocation runs %d-%d."
            % (self.n_chunks, self.chunks[0], self.chunks[-1])
        )

    def run_record(self, n_chunks):
        """
        What :data:`RUN_CONFIG_FILE` records for this run: every setting but ``chunks`` --
        the one thing that may differ between rounds -- resolved to what the jobs use, the
        thread caps they run under, and the simulation seed of every (trial, chunk).

        *n_chunks* may be None before the chunk lists are counted; the seeds are then left
        out, and neither is compared by :meth:`_check_run`.
        """
        settings = dict(self.settings)
        del settings["chunks"]
        settings.update(
            n_chunks=n_chunks, results_dir=self.master_dir, chunks_dir=self.chunk_dir
        )
        record = {
            "settings": settings,
            "environment_variables": _thread_environment(self.threads_per_rank),
        }
        if n_chunks is not None:
            record["simulation_seeds"] = {
                str(trial): [_simulation_seed(trial, c) for c in range(n_chunks)]
                for trial in range(self.n_trials)
            }
        return record

    def _check_run(self, record):
        """
        Refuse a round whose settings differ from those the run's first round recorded.

        Simulating a run in rounds has to give exactly what simulating it at once would, which
        only holds if nothing but the chunks changes between rounds. A directory holding
        results but no record was simulated without one being kept, so the settings of its
        earlier rounds are unknown and nothing could be checked against them.
        """
        if not os.path.isfile(self.run_config_path):
            if self._holds_results():
                raise ValueError(
                    "%s already holds results, but no %s recording the settings they were "
                    "simulated with, so this round cannot be checked against them -- give "
                    "the run a fresh results_dir"
                    % (self.master_dir, RUN_CONFIG_FILE)
                )
            return

        with open(self.run_config_path, "r") as f:
            recorded = json.load(f)
        differences = []
        for key, value in record["settings"].items():
            if value is None and key == "n_chunks":
                continue
            if recorded["settings"].get(key) != value:
                differences.append(
                    "%s: %r recorded, %r now" % (key, recorded["settings"].get(key), value)
                )
        for key in ("environment_variables", "simulation_seeds"):
            if key in record and record[key] != recorded[key]:
                differences.append("%s differ" % key)
        if differences:
            raise ValueError(
                "this round's settings differ from those %s records for the run, and only "
                "chunks may change between rounds -- %s"
                % (self.run_config_path, "; ".join(differences))
            )

    def _holds_results(self):
        """Whether the master directory holds datastores or shards."""
        if os.path.isdir(os.path.join(self.master_dir, "shards")):
            return True
        return any(
            "_____" in name and os.path.isdir(os.path.join(self.master_dir, name))
            for name in os.listdir(self.master_dir)
        )

    def _chunk_lists_present(self):
        """Whether chunk_dir already holds chunk lists. Emptiness means "generate here"."""
        return bool(chunk_list_names(self.chunk_dir))

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

        self._check_chunk_settings()

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

    def _check_chunk_settings(self):
        """
        Check the reused chunk lists were generated from this run's settings, as their
        chunk_settings.json records -- or, with --confirm-chunk-settings and no record, by
        regenerating them, recording the settings once they are confirmed.
        """
        expected = chunk_settings(
            self.data_root, self.n_trials, self.n_chunks, self.chunk_seed
        )
        recorded = read_chunk_settings(self.chunk_dir)
        if recorded is None:
            if not self.confirm_chunk_settings:
                raise ValueError(
                    "%s holds chunk lists but no chunk_settings.json, so the dataset, trial "
                    "count, chunk count and seed they were generated with are unknown. If "
                    "they were generated with this run's settings, rerun with "
                    "--confirm-chunk-settings to verify that and record it."
                    % self.chunk_dir
                )
            print("Confirming that %s was generated with %s" % (self.chunk_dir, expected))
            verify_chunk_lists(
                self.chunk_dir,
                self.data_root,
                self.n_trials,
                self.n_chunks,
                self.chunk_seed,
            )
            write_chunk_settings(self.chunk_dir, expected)
            print("Confirmed; recorded in chunk_settings.json")
            return

        differences = [
            "%s: %r there, %r here" % (key, recorded.get(key), value)
            for key, value in expected.items()
            if recorded.get(key) != value
        ]
        if differences:
            raise ValueError(
                "the chunk lists in %s were generated with other settings than this run's "
                "-- %s" % (self.chunk_dir, "; ".join(differences))
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
            "n_chunks if that does not fit a job's time limit."
            % (len(summary), self.chunk_dir, longest)
        )

    def _trial_chunk(self, combination):
        """Recover (trial, chunk) from a combination this class built."""
        return combination["trial"], (combination["simulation_seed"] - 1) % 1000

    def generate_parameter_combinations(self):
        return [
            {"trial": trial, "simulation_seed": _simulation_seed(trial, chunk)}
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
        env = dict(
            _thread_environment(self.threads_per_rank),
            CHUNK_DIR=self.chunk_dir,
            DATASTORE_PREFIX=self.master_dir,
            OUTPUT_PREFIX=output_prefix,
        )
        for trial in range(self.n_trials):
            command = [
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
                # recorded in the shard's meta.json, which outlives the datastores
                "--provenance",
                self.run_config_path,
            ]
            first = trial * per_trial
            depends_on = job_ids[first : first + per_trial]
            if any(j is None for j in depends_on):
                print(
                    "Trial %d: not every simulation reported a job id, so its export cannot "
                    "be made to wait for them. Run it by hand once they finish:\n    %s %s"
                    % (
                        trial,
                        " ".join(
                            "%s=%s" % (k, shlex.quote(str(v)))
                            for k, v in sorted(env.items())
                        ),
                        " ".join(shlex.quote(token) for token in command),
                    )
                )
                continue

            self.backend.execute_script(
                command,
                env=env,
                depends_on=depends_on,
                # Alongside the simulations' slurm-%j.out, so a run's whole log trail is in
                # one directory; named apart so the two waves stay distinguishable.
                output=os.path.join(self.master_dir, "slurm-export-%j.out"),
            )
        print("Submitted %d export jobs." % self.n_trials)


if __name__ == "__main__":
    settings, command, dry_run, confirm_chunk_settings = parse_command_line()
    # mozaik's own parameter-search command line is exactly the three positionals.
    sys.argv = sys.argv[:1] + command

    search = RandomizedExperantoSearch(
        make_backend(settings, dry_run),
        settings,
        export_script=EXPORT_SCRIPT,
        confirm_chunk_settings=confirm_chunk_settings,
    )
    job_ids = search.run_parameter_search()
    search.submit_exports(job_ids)
