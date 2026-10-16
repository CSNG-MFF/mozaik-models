"""Merge completed Experanto chunk exports for one trial."""

import argparse
import json
from pathlib import Path

from mozaik.meta_workflow.experanto_export import merge_experanto_trial

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("trial", type=int)
parser.add_argument("--results-dir", required=True)
parser.add_argument("--output-dir")
parser.add_argument("--chunk-start", type=int, default=0)
parser.add_argument("--chunk-end", type=int)
args = parser.parse_args()
n_chunks = json.loads((Path(args.results_dir) / "run_config.json").read_text())[
    "settings"
]["n_chunks"]
end = n_chunks if args.chunk_end is None else args.chunk_end
if not 0 <= args.chunk_start < end <= n_chunks:
    parser.error("require 0 <= chunk-start < chunk-end <= n-chunks")

output = args.output_dir or Path(args.results_dir) / "shards" / f"trial{args.trial}"
merge_experanto_trial(
    trial_id=args.trial,
    datastore_prefix=args.results_dir,
    chunk_start=args.chunk_start,
    chunk_end=end,
    output_dir=output,
)
