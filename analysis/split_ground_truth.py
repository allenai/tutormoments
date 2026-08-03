#!/usr/bin/env python3
"""Split ground truth conversation IDs into train and test sets.

Loads the original train/test split from original_train_test.json, checks
coverage against data/transcripts/tutoring_provider_a.jsonl, and incorporates any ground
truth IDs present in tutoring_provider_a.jsonl but absent from the original split by
distributing them equally between train and test.

Output format:
  {
    "seed": <int>,
    "ground_truth_dir": "<str>",
    "train": ["<conv_id>", ...],
    "test":  ["<conv_id>", ...]
  }

Usage:
    python analysis/split_ground_truth.py
    python analysis/split_ground_truth.py --seed 123
    python analysis/split_ground_truth.py --ground-truth-dir data/ground_truth_hybrid
    python analysis/split_ground_truth.py --output analysis/my_split.json
"""

import argparse
import json
import random
import warnings
from pathlib import Path

# This script lives in analysis/; its build inputs live in the repo-root data/ dir,
# and its output (split.json) sits alongside it here in analysis/.
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
DEFAULT_GT_DIR = DATA_DIR / "ground_truth_hybrid"
DEFAULT_ORIG_SPLIT = DATA_DIR / "original_train_test.json"
DEFAULT_TRANSCRIPTS = DATA_DIR / "transcripts" / "tutoring_provider_a.jsonl"
DEFAULT_OUTPUT = HERE / "split.json"
DEFAULT_SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground-truth-dir",
        default=str(DEFAULT_GT_DIR),
        help=f"Ground truth directory (default: {DEFAULT_GT_DIR})",
    )
    parser.add_argument(
        "--original-split",
        default=str(DEFAULT_ORIG_SPLIT),
        help=f"Original train/test split JSON (default: {DEFAULT_ORIG_SPLIT})",
    )
    parser.add_argument(
        "--transcripts",
        default=str(DEFAULT_TRANSCRIPTS),
        help=f"Transcripts JSONL file (default: {DEFAULT_TRANSCRIPTS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for shuffling new IDs (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON file (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    gt_dir = Path(args.ground_truth_dir)
    if not gt_dir.exists():
        print(f"ERROR: ground truth directory not found: {gt_dir}")
        return

    orig_split_path = Path(args.original_split)
    if not orig_split_path.exists():
        print(f"ERROR: original split file not found: {orig_split_path}")
        return

    transcripts_path = Path(args.transcripts)
    if not transcripts_path.exists():
        print(f"ERROR: transcripts file not found: {transcripts_path}")
        return

    # Load original split
    with open(orig_split_path, encoding="utf-8") as f:
        orig = json.load(f)
    orig_train = orig["train_ids"]
    orig_test = orig["held_out_ids"]
    orig_all = set(orig_train) | set(orig_test)
    print(f"Loaded original split: {len(orig_train)} train, {len(orig_test)} held_out")

    # Load ground truth IDs early so we can filter the original split
    gt_ids = set(f.stem for f in gt_dir.glob("*.json"))
    if not gt_ids:
        print(f"ERROR: no JSON files found in {gt_dir}")
        return
    print(f"Ground truth conversations: {len(gt_ids)}")

    # Exclude original IDs that have no ground truth file
    orig_no_gt = orig_all - gt_ids
    if orig_no_gt:
        print(
            f"Excluding {len(orig_no_gt)} original ID(s) with no ground truth: {sorted(orig_no_gt)}"
        )
        orig_train = [x for x in orig_train if x in gt_ids]
        orig_test = [x for x in orig_test if x in gt_ids]
        orig_all = set(orig_train) | set(orig_test)
        print(f"  After exclusion: {len(orig_train)} train, {len(orig_test)} held_out")

    # Load transcript IDs from tutoring_provider_a.jsonl
    print(f"Loading transcript IDs from {transcripts_path} ...")
    transcript_ids = set()
    with open(transcripts_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            transcript_ids.add(record["transcript_id"])
    print(f"  {len(transcript_ids)} transcript IDs loaded")

    # Warn about original IDs missing from transcripts
    orig_missing = orig_all - transcript_ids
    if orig_missing:
        warnings.warn(
            f"{len(orig_missing)} ID(s) in original_train_test.json are NOT present in "
            f"{transcripts_path}:\n  " + "\n  ".join(sorted(orig_missing)),
            stacklevel=2,
        )
    else:
        print("  All original split IDs are present in transcripts. No warnings.")

    # New IDs: in transcripts + ground truth, but not in original split
    new_ids = sorted(gt_ids & transcript_ids - orig_all)
    print(
        f"New IDs (in transcripts & ground truth, not in original split): {len(new_ids)}"
    )

    # Shuffle new IDs and split equally between train and test
    rng = random.Random(args.seed)
    rng.shuffle(new_ids)
    mid = len(new_ids) // 2
    # odd N: test gets the extra ID
    new_train = new_ids[:mid]
    new_test = new_ids[mid:]

    train_ids = orig_train + new_train
    test_ids = orig_test + new_test

    out = {
        "seed": args.seed,
        "train": train_ids,
        "test": test_ids,
    }

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nSplit {len(train_ids) + len(test_ids)} conversations (seed={args.seed})")
    print(
        f"  train: {len(train_ids)} ({len(orig_train)} original + {len(new_train)} new)"
    )
    print(f"  test:  {len(test_ids)} ({len(orig_test)} original + {len(new_test)} new)")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
