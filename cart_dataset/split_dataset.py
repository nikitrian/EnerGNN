"""Create random or demand-generalization CART splits without resampling demand."""

import argparse
import json
import random
import shutil
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def split_instance_ids(metadata, *, mode="random", seed=123, val_frac=0.1, test_frac=0.1):
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("Validation/test fractions must be positive and sum to less than one.")
    ids = sorted(int(row["instance_id"]) for row in metadata)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate CART instance IDs.")
    rng = random.Random(seed)
    if mode == "random":
        rng.shuffle(ids)
        n_val, n_test = int(len(ids) * val_frac), int(len(ids) * test_frac)
        val, test = ids[:n_val], ids[n_val:n_val + n_test]
        train = ids[n_val + n_test:]
    elif mode == "demand_generalization":
        ranked = sorted(metadata, key=lambda row: (int(row["total_therapies"]), int(row["instance_id"])))
        n_test = int(len(ids) * test_frac)
        if n_test == 0:
            raise ValueError("Demand-generalization split needs at least one test instance.")
        test = [int(row["instance_id"]) for row in ranked[-n_test:]]
        pool = [int(row["instance_id"]) for row in ranked[:-n_test]]
        rng.shuffle(pool)
        n_val = int(len(pool) * val_frac)
        val, train = pool[:n_val], pool[n_val:]
    else:
        raise ValueError(f"Unsupported split mode: {mode}")
    if not train or not val or not test:
        raise ValueError("Requested fractions produce an empty train, validation or test split.")
    return {"train": set(train), "val": set(val), "test": set(test)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing samples, metadata and static_config.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["random", "demand_generalization"], default="random")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Fraction of all instances in random mode; of the non-test pool in generalization mode.")
    parser.add_argument("--test-frac", type=float, default=0.1, help="Use 0.2 for the 720/80/200 demand-generalization split.")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("Source and output must be separate, non-nested directories.")
    if output.exists():
        raise FileExistsError(f"{output} exists; choose a new output directory.")
    metadata = read_jsonl(source / "instances_meta.jsonl")
    samples = read_jsonl(source / "samples.jsonl")
    config = json.loads((source / "static_config.json").read_text(encoding="utf-8-sig"))
    splits = split_instance_ids(metadata, mode=args.mode, seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac)
    known = {int(row["instance_id"]) for row in metadata}
    if any(int(row["instance_id"]) not in known for row in samples):
        raise ValueError("Sample refers to an instance missing from metadata.")
    summary = {"problem": "cart", "source_dir": source.as_posix(), "split_mode": args.mode,
               "seed": args.seed, "val_frac": args.val_frac, "test_frac": args.test_frac, "splits": {}}
    for name, ids in splits.items():
        directory = output / name
        directory.mkdir(parents=True)
        selected_meta = [row for row in metadata if int(row["instance_id"]) in ids]
        selected_samples = [row for row in samples if int(row["instance_id"]) in ids]
        for filename, rows in (("instances_meta.jsonl", selected_meta), ("samples.jsonl", selected_samples)):
            (directory / filename).write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
        (directory / "static_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        summary["splits"][name] = {"instances": len(ids), "samples": len(selected_samples), "instance_ids": sorted(ids)}
    for filename in ("summary.json", "generation_config.json"):
        if (source / filename).exists():
            shutil.copy2(source / filename, output / f"source_{filename}")
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["splits"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
