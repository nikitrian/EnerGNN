"""Run an ordered dataset, training and evaluation recipe from YAML."""

import argparse
import importlib.util
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
MODULE_PREFIXES = ("cart_dataset.", "ca_dataset.", "qkp_dataset.", "gnn.")


def build_commands(recipe, stages=None):
    configured = recipe.get("stages") if isinstance(recipe, dict) else None
    if not isinstance(configured, dict) or not configured:
        raise ValueError("Recipe needs a nonempty stages mapping.")
    selected = list(configured) if stages is None else stages
    missing = [stage for stage in selected if stage not in configured]
    if missing:
        raise ValueError(f"Unknown recipe stages: {', '.join(missing)}")
    commands = []
    for stage in selected:
        jobs = configured[stage]
        if not isinstance(jobs, list) or not jobs:
            raise ValueError(f"Stage {stage} needs a nonempty list of module commands.")
        for job in jobs:
            if not isinstance(job, dict):
                raise ValueError(f"Invalid command in stage {stage}.")
            module, arguments = job.get("module"), job.get("args", {})
            if not isinstance(module, str) or not module.startswith(MODULE_PREFIXES):
                raise ValueError("Commands must use this repository's benchmark or GNN modules.")
            if not isinstance(arguments, dict):
                raise ValueError("Command args must be a mapping of CLI flags to values.")
            command = [sys.executable, "-m", module]
            for flag, value in arguments.items():
                if not isinstance(flag, str) or not flag.startswith("--") or any(char.isspace() for char in flag):
                    raise ValueError(f"Invalid CLI flag: {flag!r}")
                if value is False or value is None:
                    continue
                command.append(flag)
                if value is not True:
                    values = value if isinstance(value, list) else [value]
                    if not values or any(not isinstance(item, (str, int, float)) or isinstance(item, bool) for item in values):
                        raise ValueError(f"Invalid argument value for {flag}.")
                    if any(isinstance(item, float) and not math.isfinite(item) for item in values):
                        raise ValueError(f"Nonfinite argument value for {flag}.")
                    command.extend(str(item) for item in values)
            commands.append((stage, command))
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", nargs="+", help="Stages to run, in the supplied order; default is all recipe stages.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without generating data or starting training.")
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8-sig") as stream:
        recipe = yaml.safe_load(stream)
    try:
        commands = build_commands(recipe, args.stage)
        for _, command in commands:
            if importlib.util.find_spec(command[2]) is None:
                raise ValueError(f"Recipe module does not exist: {command[2]}")
    except (ValueError, ModuleNotFoundError) as error:
        parser.error(str(error))
    for stage, command in commands:
        display = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
        print(f"[{stage}] {display}", flush=True)
        if not args.dry_run:
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
