import json
from types import SimpleNamespace

import pytest

from zero_shot_constraint_injection import evaluate_cart_facility_outages as cart


def scenarios(**kwargs):
    return cart._generate_scenarios(
        [1, 0, 1, 0], outage_sizes=[], max_pairs_per_instance=None,
        max_triples_per_instance=None, active_off_inactive_on=False,
        seed=0, instance_id=0, **kwargs,
    )


def test_targeted_facility_constraints_are_combined_for_each_limit():
    rows = scenarios(force_off=[0], force_on=[1], max_facilities=[3, 2, 2])
    assert len(rows) == 2
    assert [row["max_facilities"] for row in rows] == [2, 3]
    assert all(row["off_indices"] == (0,) and row["on_indices"] == (1,) for row in rows)
    assert all(row["scenario_type"] == "forced_active_inactive" for row in rows)


def test_targeted_activation_and_deactivation_work_without_outage_enumeration():
    assert scenarios(force_on=[1])[0]["scenario_type"] == "forced_active"
    assert scenarios(force_off=[0])[0]["scenario_type"] == "forced_inactive"
    assert scenarios(max_facilities=[2])[0]["scenario_type"] == "max_facilities"


@pytest.mark.parametrize("kwargs", [
    {"force_off": [0], "force_on": [0]}, {"force_on": [-1]},
    {"force_off": [4]}, {"max_facilities": [0]}, {"max_facilities": [5]},
    {"force_on": [0, 1, 2], "max_facilities": [2]},
])
def test_invalid_facility_constraints_fail_before_solving(kwargs):
    with pytest.raises(ValueError):
        scenarios(**kwargs)


def test_prediction_must_satisfy_fixed_states_and_facility_count():
    assert cart._validate_mask_constraints([0, 1, 1, 0], {0: 0, 1: 1}, 1, 2)
    assert not cart._validate_mask_constraints([0, 1, 1, 1], {0: 0, 1: 1}, 1, 2)
    assert not cart._validate_mask_constraints([1, 1, 0, 0], {0: 0, 1: 1}, 1, 2)
    assert not cart._validate_mask_constraints([0, 0, 0, 0], {}, 1, 0)
    assert cart._scenario_max_k({"max_facilities": 3}, 2) == 2


def test_evaluator_applies_new_constraints_to_reference_inference_and_scoring(tmp_path, monkeypatch):
    meta = SimpleNamespace(M=[f"m{i + 1}" for i in range(16)])
    monkeypatch.setattr(cart, "_load_eval_data", lambda *args: ({0: meta}, {}, {}, 7))
    monkeypatch.setattr(cart, "_load_cart_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(cart.InstancePoolDataset, "_build_graph", lambda *args: ({}, {}))
    detail_calls, inference_calls = [], []

    def details(meta, **kwargs):
        detail_calls.append(kwargs)
        mask = [1, 0, 1] + [0] * 13 if not kwargs.get("fixed_assignments") else [0, 1, 1] + [0] * 13
        return {"configuration": mask, "objective": 100.0, "status": "optimal",
                "runtime": 0.1, "bound": 100.0, "solver_gap": 0.0, "proven_optimal": True}

    def inference(model, graph, meta, **kwargs):
        inference_calls.append(kwargs)
        return [0, 1, 1] + [0] * 13, 0.1, 0.1, "OPTIMAL", {}

    monkeypatch.setattr(cart, "_solve_cart_details", details)
    monkeypatch.setattr(cart, "solve_instance_simple", inference)
    cart.main(["--checkpoint", str(tmp_path / "model.ckpt"), "--output-dir", str(tmp_path / "results"),
               "--force-off", "0", "--force-on", "1",
               "--max-facilities", "2", "3", "--quiet"])

    assert detail_calls[0]["max_k"] == 0
    assert "fixed_assignments" not in detail_calls[0]
    assert [call["max_k"] for call in detail_calls[1:]] == [2, 2, 3, 3]
    assert all(call["fixed_assignments"] == {0: 0, 1: 1} for call in detail_calls[1:])
    assert [call["max_k"] for call in inference_calls] == [2, 3]
    assert all(call["fixed_assignments"] == {0: 0, 1: 1} for call in inference_calls)
    rows = [json.loads(line) for line in (tmp_path / "results/scenario_results.jsonl").read_text().splitlines()]
    assert [row["max_facilities"] for row in rows] == [2, 3]
    assert all(row["energnn_constraints_satisfied"] and row["energnn_feasible"] for row in rows)
    summary = json.loads((tmp_path / "results/summary.json").read_text())
    assert len(summary["summary_by_constraint"]) == 2
    assert summary["evaluation"]["force_on"] == [1]
