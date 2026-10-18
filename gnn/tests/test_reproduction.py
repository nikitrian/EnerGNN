from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from experiments import reproduce


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("filename", ["cart_detailed.yaml", "ca_100x175.yaml", "qkp150.yaml"])
def test_reproduction_recipes_have_connected_training_and_evaluation_paths(filename):
    path = ROOT / "configs/reproduction" / filename
    recipe = yaml.safe_load(path.read_text())
    commands = reproduce.build_commands(recipe)
    assert commands[0][0] == "generate" and commands[-1][0] == "evaluate"
    train = next(command for stage, command in commands if stage == "train")
    evaluate = commands[-1][1]
    output = train[train.index("--out_dir") + 1]
    assert evaluate[evaluate.index("--ckpt") + 1] == output + "/best.ckpt"
    assert "--val_dir" in train and "--test_dir" in train
    result = subprocess.run([sys.executable, "-m", "experiments.reproduce", "--config", str(path), "--dry-run"],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "[generate]" in result.stdout and "[evaluate]" in result.stdout


def test_reproduction_arguments_are_structured_and_optional_flags_are_omitted():
    recipe = {"stages": {"generate": [{"module": "cart_dataset.dataset_maker_masks", "args": {
        "--instance-id": [1, 2], "--collect-only": True, "--tee": False, "--references": None,
        "--out-dir": "path with spaces",
    }}]}}
    _, command = reproduce.build_commands(recipe)[0]
    assert command == [sys.executable, "-m", "cart_dataset.dataset_maker_masks", "--instance-id", "1", "2",
                       "--collect-only", "--out-dir", "path with spaces"]


def test_cart16_recipe_prepares_then_scores_and_collects_existing_patient_inputs():
    recipe = yaml.safe_load((ROOT / "configs/reproduction/cart_detailed.yaml").read_text())
    commands = reproduce.build_commands(recipe)
    assert [stage for stage, _ in commands] == ["generate"] * 3 + ["score"] * 3 + ["collect"] * 3 + ["train", "evaluate"]
    for stage, command in commands[:9]:
        assert command[command.index("--dat-dir") + 1].startswith("data/CART/m16/")
        assert ("--prepare-plans-only" in command) == (stage == "generate")
        assert ("--collect-only" in command) == (stage == "collect")


@pytest.mark.parametrize("filename", ["cart_detailed.yaml", "ca_100x175.yaml", "qkp150.yaml"])
def test_recipe_arguments_are_accepted_by_the_real_cli_parsers(filename):
    recipe = yaml.safe_load((ROOT / "configs/reproduction" / filename).read_text())
    program = """import argparse
import runpy
import sys
module = sys.argv[1]
sys.argv = [module, *sys.argv[2:]]
original = argparse.ArgumentParser.parse_args
def validate(parser, *args, **kwargs):
    original(parser, *args, **kwargs)
    raise SystemExit(0)
argparse.ArgumentParser.parse_args = validate
runpy.run_module(module, run_name='__main__')
"""
    for stage, command in reproduce.build_commands(recipe):
        result = subprocess.run([sys.executable, "-c", program, *command[2:]], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"{filename}/{stage}: {result.stderr}"


@pytest.mark.parametrize("recipe", [None, {}, {"stages": {}}, {"stages": {"generate": []}},
    {"stages": {"generate": [{"module": "subprocess", "args": {}}]}},
    {"stages": {"generate": [{"module": "cart_dataset.dataset_maker", "args": {"not-a-flag": 1}}]}},
])
def test_invalid_reproduction_recipes_fail_before_launching_commands(recipe):
    with pytest.raises(ValueError):
        reproduce.build_commands(recipe)


def test_reproduction_stage_selection_is_explicit():
    recipe = {"stages": {name: [{"module": "cart_dataset.dataset_maker"}] for name in ("generate", "split")}}
    assert [stage for stage, _ in reproduce.build_commands(recipe, ["split"])] == ["split"]
    with pytest.raises(ValueError, match="Unknown recipe stages"):
        reproduce.build_commands(recipe, ["missing"])


def test_reproduction_stops_when_a_stage_fails(tmp_path, monkeypatch):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({"stages": {name: [{"module": "cart_dataset.dataset_maker"}]
                                              for name in ("generate", "split")}}))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(reproduce.subprocess, "run", run)
    assert reproduce.main(["--config", str(path)]) == 2
    assert len(calls) == 1 and calls[0][1]["cwd"] == ROOT
