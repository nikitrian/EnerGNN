import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from experiments.qkp_example import solve_examples
from gnn.data.dataset import InstancePoolDataset
from gnn.data.reader import load_instances_meta_qkp
from gnn.inference.checkpoint import load_trained_model


ROOT = Path(__file__).resolve().parents[2]


def test_qkp_example_asset_checksums():
    example_dir = ROOT / "examples/qkp"
    manifest = json.loads((example_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        assert hashlib.sha256((example_dir / name).read_bytes()).hexdigest() == expected, name


def test_qkp_example_requires_gurobi(tmp_path):
    output = tmp_path / "example.json"
    program = "import runpy, sys; sys.modules['gurobipy'] = None; runpy.run_module('experiments.qkp_example', run_name='__main__')"
    result = subprocess.run(
        [sys.executable, "-c", program, "--output-json", str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert '.[solver]' in result.stderr and 'license' in result.stderr
    assert not output.exists()


def test_qkp_example_rejects_tampered_inputs(tmp_path):
    shutil.copytree(ROOT / "examples/qkp", tmp_path / "example")
    with (tmp_path / "example/data/instance_0005.npz").open("ab") as fh:
        fh.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        solve_examples(tmp_path / "example")


def test_qkp_gurobi_example_matches_exhaustive_learned_energy(tmp_path):
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    weights = np.array([1., 2., 3., 4.])
    capacity = 5.
    matrix = np.array([[0., 1., 2., 0.], [1., 0., 0., 1.],
                       [2., 0., 0., 3.], [0., 1., 3., 0.]])
    np.savez(data_dir / "instance_0000.npz", p=np.array([2., 4., 5., 7.]),
             Q=matrix, w=weights, W=capacity)
    manifest = json.loads((ROOT / "examples/qkp/manifest.json").read_text())
    shutil.copy2(ROOT / "examples/qkp/best.ckpt", tmp_path / "best.ckpt")
    manifest["sha256"] = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                          for name in ("best.ckpt", "data/instance_0000.npz")}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    results = solve_examples(tmp_path, time_limit=30)
    row = results[0]
    assert row["energy_proven_optimal"] and row["feasible"]
    assert row["weight"] <= capacity + 1e-6
    mask = np.zeros(4)
    mask[row["selected_items"]] = 1
    assert row["weight"] == pytest.approx(weights @ mask)
    assert row["objective"] == pytest.approx(np.array([2., 4., 5., 7.]) @ mask + 0.5 * mask @ matrix @ mask)

    metas = load_instances_meta_qkp(str(data_dir))
    graph = InstancePoolDataset(metas, {0: []}, TMFE=1)[0][0]
    model = load_trained_model("energnn", manifest["model_config"],
                               tmp_path / "best.ckpt", torch.device("cpu"))
    feasible_masks = [bits for bits in itertools.product((0, 1), repeat=4)
                      if weights @ bits <= capacity]
    with torch.inference_mode():
        minimum = model(graph, torch.tensor(feasible_masks, dtype=torch.float32)).min().item()
    assert row["energy"] == pytest.approx(minimum, abs=1e-5)
    assert row["milp_energy"] == pytest.approx(row["energy"], abs=1e-5)
