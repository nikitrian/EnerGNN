from gnn.inference.run_milp import resolve_fixing_inference_defaults
from gnn.models.model import resolve_ca_head_variant
from gnn.training.train import resolve_paper_training_defaults


def test_paper_training_defaults_are_benchmark_specific():
    cart = resolve_paper_training_defaults("cart")
    ca = resolve_paper_training_defaults("ca")
    qkp = resolve_paper_training_defaults("qkp")

    assert cart == {
        "epochs": 2000,
        "lr": 2e-6,
        "weight_decay": 1e-5,
        "alpha": 1.0,
        "temperature": 0.1,
        "lambda_inf_cal": 0.1,
        "rank_tau": 0.05,
    }
    assert ca == {
        "epochs": 2000,
        "lr": 1e-6,
        "weight_decay": 1e-5,
        "alpha": 0.01,
        "temperature": 0.2,
        "lambda_inf_cal": 0.1,
        "rank_tau": 0.005,
    }
    assert qkp == {
        "epochs": 2000,
        "lr": 1e-5,
        "weight_decay": 1e-5,
        "alpha": 0.05,
        "temperature": 0.5,
        "lambda_inf_cal": 0.1,
        "rank_tau": 0.05,
    }


def test_explicit_training_overrides_are_preserved():
    resolved = resolve_paper_training_defaults(
        "ca",
        epochs=12,
        lr=3e-4,
        alpha=0.7,
    )

    assert resolved["epochs"] == 12
    assert resolved["lr"] == 3e-4
    assert resolved["alpha"] == 0.7
    assert resolved["temperature"] == 0.2


def test_paper_head_defaults_to_shallow_width_32():
    assert resolve_ca_head_variant("cart", "auto") == "mlp32"
    assert resolve_ca_head_variant("ca", "auto") == "mlp32"
    assert resolve_ca_head_variant("qkp", "auto") == "mlp32"


def test_predict_and_search_paper_defaults():
    cart = resolve_fixing_inference_defaults("cart", "predict_and_search")
    ca = resolve_fixing_inference_defaults("ca", "predict_and_search")
    qkp = resolve_fixing_inference_defaults("qkp", "predict_and_search")

    assert (cart["fixing_threshold_low"], cart["fixing_threshold_high"]) == (0.4, 0.6)
    assert ca["fixing_ratio"] == 0.80
    assert qkp["fixing_ratio"] == 0.85
    assert cart["n_dives"] == ca["n_dives"] == qkp["n_dives"] == 1


def test_neural_diving_paper_defaults():
    cart = resolve_fixing_inference_defaults("cart", "gnn_fixing_mip")
    ca = resolve_fixing_inference_defaults("ca", "gnn_fixing_mip")
    qkp = resolve_fixing_inference_defaults("qkp", "gnn_fixing_mip")

    assert cart["n_dives"] == 8
    assert ca["n_dives"] == 100
    assert qkp["n_dives"] == 20
    assert ca["fixing_ratio"] == 0.50
    assert qkp["fixing_ratio"] == 0.85
