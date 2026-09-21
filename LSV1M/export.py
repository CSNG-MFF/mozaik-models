"""
Concatenate the simulated chunks of a trial into one Experanto shard.

One invocation exports one trial: it walks that trial's chunks in order, resolves each one's
datastore, and folds them into a single spikes + screen shard. The fold is ordered -- each
chunk's spike times are offset by the total duration of the chunks before it -- so chunks
cannot be exported independently and merged afterwards.

The work itself lives in mozaik.meta_workflow.experanto_export; this script only supplies the
project's path conventions and command line.

Usage
-----
    # one trial, all 12 of its chunks
    CHUNK_DIR=... DATASTORE_PREFIX=... OUTPUT_PREFIX=... python -u export.py 0 --n-chunks 12

    # several trials
    python -u export.py 0 1 2 --n-chunks 12

    # split a long export across jobs; the second appends to the first
    python -u export.py 0 --n-chunks 12 --chunk-start 0 --chunk-end 6
    python -u export.py 0 --n-chunks 12 --chunk-start 6 --chunk-end 12

    # one half only
    python -u export.py 0 --n-chunks 12 --screen-only
    python -u export.py 0 --n-chunks 12 --spikes-only

Environment
-----------
CHUNK_DIR         Directory holding the chunk lists. The screen timeline is rebuilt from ALL
                  of a trial's chunks even when only some are exported for spikes.
DATASTORE_PREFIX  Directory the simulation wrote its datastores into.
OUTPUT_PREFIX     Shard directory per trial is {OUTPUT_PREFIX}{trial}.
SHEET_NAMES       Comma-separated sheets to fold into spikes.npy; unset means all recorded.
"""
import argparse
import logging
import os
import sys

# Imported for its side effect: registers the analysis classes needed to unpickle datastores.
from mozaik.analysis.analysis import *  # noqa: F401,F403

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

from mozaik.meta_workflow.experanto_export import run_experanto_export
from mozaik.tools.experanto_export import load_tier_reference

parser = argparse.ArgumentParser(
    description="Export Mozaik simulation data to Experanto format."
)
parser.add_argument("trials", nargs="+", type=int, help="Trial numbers to export")
parser.add_argument(
    "--n-chunks",
    type=int,
    required=True,
    help="Total number of chunks per trial (must match the chunk lists)",
)
parser.add_argument(
    "--chunk-start",
    type=int,
    default=None,
    help="First chunk index to process (inclusive, default: 0)",
)
parser.add_argument(
    "--chunk-end",
    type=int,
    default=None,
    help="Last chunk index to process (exclusive, default: --n-chunks)",
)
parser.add_argument(
    "--screen-only",
    action="store_true",
    help="Export screen data only (skip spike export)",
)
parser.add_argument(
    "--spikes-only",
    action="store_true",
    help="Export spike data only (skip screen export)",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=4,
    help="Number of chunks to hold in memory before flushing (default: 4)",
)
parser.add_argument(
    "--tier-reference",
    type=str,
    default=None,
    help="Path to a reference combined_meta.json whose tier assignments should be used "
    "(e.g. from the original mouse dataset)",
)
args = parser.parse_args()

chunk_dir = os.environ.get("CHUNK_DIR", "[PATH_TO_CHUNKS]")
output_prefix = os.environ.get("OUTPUT_PREFIX", "[PATH_TO_OUTPUT]/trial")
datastore_prefix = os.environ.get("DATASTORE_PREFIX", "")

# Which sheets to fold into spikes.npy; unset or empty means every recorded sheet.
sheet_env = os.environ.get("SHEET_NAMES", "").strip()
sheet_names = [s.strip() for s in sheet_env.split(",") if s.strip()] or None
print("Sheet selection: %s" % (sheet_names if sheet_names else "ALL recorded sheets"))

tier_reference = None
if args.tier_reference:
    print("Loading tier reference from %s" % args.tier_reference)
    tier_reference = load_tier_reference(args.tier_reference)
    print("Loaded %d condition_hash -> tier mappings" % len(tier_reference))

# The trial/chunk loop, datastore resolution, batching, resume and finalisation all live in
# the driver; what stays here are the two path conventions it cannot know:
#   output_dir_for_trial   where this project writes a trial's shard
#   chunk_paths_for_trial  ALL of a trial's chunk lists -- the screen timeline needs every one
#                          of them even when only a subset is processed for spikes
run_experanto_export(
    trials=args.trials,
    n_chunks=args.n_chunks,
    output_dir_for_trial=lambda t: "%s%d" % (output_prefix, t),
    chunk_paths_for_trial=lambda t: [
        os.path.join(chunk_dir, "%d_%d.json" % (t, c)) for c in range(args.n_chunks)
    ],
    datastore_prefix=datastore_prefix,
    sheet_names=sheet_names,
    chunk_start=args.chunk_start if args.chunk_start is not None else 0,
    chunk_end=args.chunk_end if args.chunk_end is not None else args.n_chunks,
    batch_size=args.batch_size,
    export_spikes=not args.screen_only,
    export_screen=not args.spikes_only,
    tier_reference=tier_reference,
)
