from types import SimpleNamespace

import pytest

from cart_dataset import candidate_scoring
from gnn.inference import run_milp


@pytest.mark.parametrize("formulation", ["auto", "detailed"])
@pytest.mark.parametrize("turnaround", [17, 18])
def test_backend_always_resolves_to_detailed(formulation, turnaround):
    assert candidate_scoring.resolve_formulation(formulation, turnaround) == "detailed"


@pytest.mark.parametrize("turnaround", [0, -1, float("nan"), float("inf")])
def test_invalid_turnaround_is_rejected(turnaround):
    with pytest.raises(ValueError, match="finite and positive"):
        candidate_scoring.resolve_formulation(turnaround_limit=turnaround)


def test_unsupported_formulation_is_rejected():
    with pytest.raises(ValueError, match="detailed Gurobi"):
        candidate_scoring.solve_mask({}, None, time_horizon=20, formulation="other")


@pytest.mark.parametrize("return_details", [False, True])
def test_exact_cart_uses_detailed_solver_and_preserves_constraints(monkeypatch, return_details):
    meta = SimpleNamespace(M=["m1", "m2"], T=20)
    data = object()
    monkeypatch.setattr(run_milp, "_cart_detailed_data_from_meta", lambda meta: data)
    captured = {}

    def solve(received, **kwargs):
        assert received is data
        captured.update(kwargs)
        return {
            "objective": 123.0, "opened_facilities": ["m2"],
            "status": "optimal", "proven_optimal": True,
            "solver_time_sec": 2.0, "best_bound": 123.0, "mip_gap_rel": 0.0,
        }

    monkeypatch.setattr(candidate_scoring, "solve_mask", solve)
    result = run_milp._solve_true_cart_milp(
        meta, min_k=1, max_k=1, fixed_mask=[0, 1], fixed_assignments={0: 0},
        time_limit=30, solver_threads=2, mip_gap=0, return_details=return_details,
    )
    assert captured["mask"] == [0, 1]
    assert captured["forced_e1_values"] == {"m1": 0}
    assert captured["min_open_facilities"] == captured["max_open_facilities"] == 1
    assert captured["formulation"] == "detailed"
    assert captured["time_limit"] == 30 and captured["threads"] == 2
    if return_details:
        assert result["configuration"] == [0, 1]
        assert result["proven_optimal"] and result["bound"] == 123.0
    else:
        assert result == ([0, 1], 123.0, "OPTIMAL", 2.0)


def test_conpas_cart_preserves_rank_neighbourhood_in_detailed_solver(monkeypatch):
    meta = SimpleNamespace(M=["m1", "m2", "m3"], T=20)
    monkeypatch.setattr(run_milp, "_cart_detailed_data_from_meta", lambda meta: {})
    captured = {}

    def solve(data, **kwargs):
        captured.update(kwargs)
        return {"objective": 123.0, "opened_facilities": ["m2"],
                "proven_optimal": True, "solver_time_sec": 2.0}

    monkeypatch.setattr(candidate_scoring, "solve_mask", solve)
    result = run_milp._solve_conpas_cart_milp(
        meta, x0_indices=[0, 2], x1_indices=[1], delta=1,
        min_k=1, max_k=2, fixed_assignments={0: 0},
    )
    assert result == ([0, 1, 0], 123.0, "OPTIMAL", 2.0)
    assert captured["conpas_x0_facilities"] == ["m1", "m3"]
    assert captured["conpas_x1_facilities"] == ["m2"]
    assert captured["conpas_delta"] == 1
    assert captured["forced_e1_values"] == {"m1": 0}
    assert captured["min_open_facilities"] == 1
    assert captured["max_open_facilities"] == 2


def test_detailed_builder_receives_rank_and_scenario_constraints(monkeypatch):
    pytest.importorskip("gurobipy")
    from cart_dataset import model_cart_detailed_gurobi as backend

    data = SimpleNamespace(m=["m1", "m2"])
    captured = {}
    disposed = []

    class Model:
        Status = 2
        SolCount = 1
        ObjVal = ObjBound = 123.0
        MIPGap = Runtime = 0.0
        NumVars = NumConstrs = NumBinVars = 2

        def optimize(self):
            pass

        def dispose(self):
            disposed.append("model")

    class Environment:
        def dispose(self):
            disposed.append("env")

    def build(received, **kwargs):
        assert received is data
        captured.update(kwargs)
        return Model(), SimpleNamespace(
            E1={"m1": SimpleNamespace(X=0), "m2": SimpleNamespace(X=1)},
            env=Environment(),
        )

    monkeypatch.setattr(backend, "detailed_data_from_mapping", lambda *args, **kwargs: data)
    monkeypatch.setattr(backend, "build_detailed_gurobi_model", build)
    monkeypatch.setattr(backend, "restrict_detailed_data", lambda *args:
                        pytest.fail("Scenario constraints require all facilities"))
    record = candidate_scoring.solve_mask(
        data, [0, 1], time_horizon=20, min_open_facilities=1, max_open_facilities=1,
        forced_e1_values={"m1": 0}, conpas_x0_facilities=["m1"],
        conpas_x1_facilities=["m2"], conpas_delta=0,
    )
    assert record["objective"] == 123.0 and record["formulation"] == "detailed"
    assert captured["fixed_e1_values"] == {"m1": 0, "m2": 1}
    assert captured["forced_e1_values"] == {"m1": 0}
    assert captured["conpas_x0_facilities"] == ["m1"]
    assert captured["conpas_x1_facilities"] == ["m2"]
    assert captured["conpas_delta"] == 0
    assert disposed == ["model", "env"]
