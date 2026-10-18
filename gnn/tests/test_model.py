# gnn/tests/test_model.py

from gnn.data.dataset import InstancePoolDataset
from gnn.data.schema import InstanceMeta, SampleRow
from gnn.models.model import (
    GraphEBM,
    LinearMIPBCEBaseline,
    LinearMIPContrastiveBaseline,
    model_display_name,
    resolve_qkp_readout_variant,
)
from gnn.models.energy_head import LinearEnergyHead, ShallowEnergyHead
from gnn.models.readout import (
    CADirectYReadout,
    CAPooledOnlyReadout,
    CASumOnlyPlusReadout,
    QKPDirectYReadout,
    QKPSizeAwareReadout,
)

def test_ca_defaults_to_sum_only_plus():
    model = GraphEBM(dc=1, dm=1, de=1, problem="ca")
    assert model.ca_readout_variant == "sum_only_plus"
    assert isinstance(model.readout, CASumOnlyPlusReadout)
    assert model.ca_head_variant == "mlp32"
    assert model.readout.summary_dim == 8
    assert isinstance(model.head, ShallowEnergyHead)






def test_ca_pooled_only_requires_explicit_request():
    model = GraphEBM(dc=1, dm=1, de=1, problem="ca", ca_readout_variant="pooled_only")
    assert model.ca_readout_variant == "pooled_only"
    assert isinstance(model.readout, CAPooledOnlyReadout)
    assert model.readout.out_dim == 4 * 128


def test_ca_sum_only_plus_requires_explicit_request():
    model = GraphEBM(dc=1, dm=1, de=1, problem="ca", ca_readout_variant="sum_only_plus")
    assert model.ca_readout_variant == "sum_only_plus"
    assert isinstance(model.readout, CASumOnlyPlusReadout)


def test_ca_direct_y_requires_explicit_request():
    model = GraphEBM(
        dc=1,
        dm=1,
        de=1,
        problem="ca",
        ca_readout_variant="direct_y",
        ca_direct_y_dim=6,
    )
    assert model.ca_readout_variant == "direct_y"
    assert isinstance(model.readout, CADirectYReadout)
    assert model.readout.out_dim == (4 * 128) + 6


def test_ca_mlp64_head_requires_explicit_request():
    model = GraphEBM(dc=1, dm=1, de=1, problem="ca", ca_head_variant="mlp64")
    assert model.ca_head_variant == "mlp64"
    assert getattr(model.head, "variant", None) == "mlp64"


def test_ca_mlp16_head_requires_explicit_request():
    model = GraphEBM(dc=1, dm=1, de=1, problem="ca", ca_head_variant="mlp16")
    assert model.ca_head_variant == "mlp16"
    assert getattr(model.head, "variant", None) == "mlp16"


def test_ca_linear_head_requires_explicit_request():
    model = GraphEBM(dc=1, dm=1, de=1, problem="ca", ca_head_variant="linear")
    assert model.ca_head_variant == "linear"
    assert isinstance(model.head, LinearEnergyHead)


def test_cart_can_use_shallow_mlp32_head_explicitly():
    model = GraphEBM(dc=1, dm=1, de=1, problem="cart", ca_head_variant="mlp32")
    assert model.ca_head_variant == "mlp32"
    assert isinstance(model.head, ShallowEnergyHead)


def test_linear_mip_bce_baseline_scores_variables_and_energy():
    import torch

    graph = {
        "Xc": torch.randn(3, 2),
        "Xm": torch.randn(4, 5),
        "edge_index": torch.tensor([[0, 1, 2], [3, 4, 6]], dtype=torch.long),
        "edge_attr": torch.randn(3, 1),
    }
    model = LinearMIPBCEBaseline(dc=2, dm=5, de=1, dh=8, nlayers=1, problem="cart")
    scores = model.score_variables(graph)
    assert tuple(scores.shape) == (4,)

    masks = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
    energies = model(graph, masks)
    assert tuple(energies.shape) == (2,)
    assert torch.allclose(energies, -(masks @ scores))


def test_linear_mip_names_identify_training_objective():
    assert model_display_name("linear_mip_bce") == "Linear+MIP(BCE)"
    assert model_display_name("linear_mip_contrastive") == "Linear+MIP(contrastive)"
    assert LinearMIPBCEBaseline.display_name == "Linear+MIP(BCE)"
    assert LinearMIPContrastiveBaseline.display_name == "Linear+MIP(contrastive)"


def test_qkp_uses_qkp_readout_and_synthetic_dataset_pipeline():
    p = [4.0, 7.0, 3.0, 5.0]
    w = [2.0, 4.0, 3.0, 5.0]
    Q = [
        [0.0, 1.5, 0.0, -0.5],
        [1.5, 0.0, 2.0, 0.0],
        [0.0, 2.0, 0.0, 1.0],
        [-0.5, 0.0, 1.0, 0.0],
    ]
    pair_indices = [(0, 1), (0, 3), (1, 2), (2, 3)]

    meta = InstanceMeta(
        instance_id=0,
        C=[f"{i}_{j}" for i, j in pair_indices],
        M=[str(i) for i in range(len(p))],
        CAPMAX={},
        U1={},
        CIM={},
        CVM={},
        INC={},
        T=1,
        globals={
            "problem": "qkp",
            "p": p,
            "Q": Q,
            "w": w,
            "W": 7.0,
            "n_items": len(p),
            "sparsity": 0.5,
            "opt_obj": 14.5,
            "pair_indices": pair_indices,
            "objective_sense": "max",
        },
    )
    sample = SampleRow(
        instance_id=0,
        label_type="positive",
        feasible=True,
        objective=14.5,
        status="OPTIMAL",
        e1={"0": 1, "1": 1, "2": 0, "3": 0},
        hamming_r=0,
        regret_rel=0.0,
        regret_abs=0.0,
        seed_demand=0,
        seed_solver=0,
        extras={"violation": 0.0},
    )

    ds = InstancePoolDataset({0: meta}, {0: [sample]}, TMFE=1)
    container, masks_t, _, _, meta_out = ds[0]
    assert meta_out["globals"]["problem"] == "qkp"
    assert tuple(container["Xc"].shape) == (4, 7)
    assert tuple(container["Xm"].shape) == (4, 7)
    assert tuple(container["edge_attr"].shape) == (8, 3)
    assert abs(float(meta_out["globals"]["n_items_norm"]) - (4.0 / 200.0)) < 1e-9

    model = GraphEBM(
        dc=container["Xc"].shape[1],
        dm=container["Xm"].shape[1],
        de=container["edge_attr"].shape[1],
        problem="qkp",
    )
    assert resolve_qkp_readout_variant("qkp", "auto") == "size_aware"
    assert model.qkp_readout_variant == "size_aware"
    assert isinstance(model.readout, QKPSizeAwareReadout)
    assert model.readout.out_dim == 4 * 128 + 2

    energies = model(container, masks_t)
    assert tuple(energies.shape) == (1,)


def test_qkp_size_aware_variant_constructs_and_runs():
    p = [4.0, 7.0, 3.0, 5.0]
    w = [2.0, 4.0, 3.0, 5.0]
    Q = [
        [0.0, 1.5, 0.0, -0.5],
        [1.5, 0.0, 2.0, 0.0],
        [0.0, 2.0, 0.0, 1.0],
        [-0.5, 0.0, 1.0, 0.0],
    ]
    pair_indices = [(0, 1), (0, 3), (1, 2), (2, 3)]

    meta = InstanceMeta(
        instance_id=0,
        C=[f"{i}_{j}" for i, j in pair_indices],
        M=[str(i) for i in range(len(p))],
        CAPMAX={},
        U1={},
        CIM={},
        CVM={},
        INC={},
        T=1,
        globals={
            "problem": "qkp",
            "p": p,
            "Q": Q,
            "w": w,
            "W": 7.0,
            "n_items": len(p),
            "sparsity": 0.5,
            "opt_obj": 14.5,
            "pair_indices": pair_indices,
            "objective_sense": "max",
        },
    )
    sample = SampleRow(
        instance_id=0,
        label_type="positive",
        feasible=True,
        objective=14.5,
        status="OPTIMAL",
        e1={"0": 1, "1": 1, "2": 0, "3": 0},
        hamming_r=0,
        regret_rel=0.0,
        regret_abs=0.0,
        seed_demand=0,
        seed_solver=0,
        extras={"violation": 0.0},
    )

    ds = InstancePoolDataset({0: meta}, {0: [sample]}, TMFE=1)
    container, masks_t, _, _, _ = ds[0]
    model = GraphEBM(
        dc=container["Xc"].shape[1],
        dm=container["Xm"].shape[1],
        de=container["edge_attr"].shape[1],
        problem="qkp",
        qkp_readout_variant="size_aware",
    )

    assert model.qkp_readout_variant == "size_aware"
    assert isinstance(model.readout, QKPSizeAwareReadout)
    assert model.readout.out_dim == (4 * 128) + 2

    energies = model(container, masks_t)
    assert tuple(energies.shape) == (1,)


def test_qkp_direct_y_variant_constructs_and_runs():
    p = [4.0, 7.0, 3.0, 5.0]
    w = [2.0, 4.0, 3.0, 5.0]
    Q = [
        [0.0, 1.5, 0.0, -0.5],
        [1.5, 0.0, 2.0, 0.0],
        [0.0, 2.0, 0.0, 1.0],
        [-0.5, 0.0, 1.0, 0.0],
    ]
    pair_indices = [(0, 1), (0, 3), (1, 2), (2, 3)]

    meta = InstanceMeta(
        instance_id=0,
        C=[f"{i}_{j}" for i, j in pair_indices],
        M=[str(i) for i in range(len(p))],
        CAPMAX={},
        U1={},
        CIM={},
        CVM={},
        INC={},
        T=1,
        globals={
            "problem": "qkp",
            "p": p,
            "Q": Q,
            "w": w,
            "W": 7.0,
            "n_items": len(p),
            "sparsity": 0.5,
            "opt_obj": 14.5,
            "pair_indices": pair_indices,
            "objective_sense": "max",
        },
    )
    sample = SampleRow(
        instance_id=0,
        label_type="positive",
        feasible=True,
        objective=14.5,
        status="OPTIMAL",
        e1={"0": 1, "1": 1, "2": 0, "3": 0},
        hamming_r=0,
        regret_rel=0.0,
        regret_abs=0.0,
        seed_demand=0,
        seed_solver=0,
        extras={"violation": 0.0},
    )

    ds = InstancePoolDataset({0: meta}, {0: [sample]}, TMFE=1)
    container, masks_t, _, _, _ = ds[0]
    model = GraphEBM(
        dc=container["Xc"].shape[1],
        dm=container["Xm"].shape[1],
        de=container["edge_attr"].shape[1],
        problem="qkp",
        qkp_readout_variant="direct_y",
        qkp_direct_y_dim=6,
    )

    assert model.qkp_readout_variant == "direct_y"
    assert isinstance(model.readout, QKPDirectYReadout)
    assert model.readout.out_dim == (3 * 128) + 6

    energies = model(container, masks_t)
    assert tuple(energies.shape) == (1,)


def test_qkp_size_aware_readout_matches_formula_order():
    import math
    import torch

    readout = QKPSizeAwareReadout(dh=2)
    hm = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 5.0],
            [7.0, 11.0],
        ]
    )
    hg = torch.tensor([13.0, 17.0, 19.0, 23.0])
    masks = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    graph = {
        "qkp_weights_t": torch.tensor([2.0, 4.0, 6.0]),
        "qkp_capacity_t": torch.tensor(8.0),
    }

    out = readout({"hm": hm, "hg": hg}, masks, graph=graph)
    inv_sqrt_3 = 1.0 / math.sqrt(3.0)
    expected = torch.tensor(
        [
            [
                8.0 * inv_sqrt_3,
                13.0 * inv_sqrt_3,
                3.0 * inv_sqrt_3,
                5.0 * inv_sqrt_3,
                2.0 / 3.0,
                1.0,
                13.0,
                17.0,
                19.0,
                23.0,
            ],
            [
                4.0 * inv_sqrt_3,
                7.0 * inv_sqrt_3,
                7.0 * inv_sqrt_3,
                11.0 * inv_sqrt_3,
                2.0 / 3.0,
                0.75,
                13.0,
                17.0,
                19.0,
                23.0,
            ],
        ]
    )
    assert torch.allclose(out, expected)



def test_qkp_direct_y_readout_matches_formula_order_and_padding():
    import torch

    readout = QKPDirectYReadout(dh=2, direct_y_dim=5)
    hm = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 5.0],
            [7.0, 11.0],
        ]
    )
    hg = torch.tensor([13.0, 17.0, 19.0, 23.0])
    masks = torch.tensor([[1.0, 0.0, 1.0]])

    out = readout({"hm": hm, "hg": hg}, masks)
    expected = torch.tensor([[8.0, 13.0, 1.0, 0.0, 1.0, 0.0, 0.0, 13.0, 17.0, 19.0, 23.0]])
    assert torch.allclose(out, expected)


def test_ca_pooled_only_readout_matches_formula_order():
    import torch

    readout = CAPooledOnlyReadout(dh=2)
    hm = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 5.0],
            [7.0, 11.0],
        ]
    )
    hc = torch.tensor(
        [
            [13.0, 17.0],
            [19.0, 23.0],
        ]
    )
    hg = torch.tensor([29.0, 31.0, 37.0, 41.0])
    masks = torch.tensor([[1.0, 0.0, 1.0]])
    incidence = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )

    out = readout({"hm": hm, "hc": hc, "hg": hg}, masks, graph={"incidence_t": incidence})
    expected = torch.tensor([[8.0, 13.0, 32.0, 40.0, 29.0, 31.0, 37.0, 41.0]])
    assert torch.allclose(out, expected)


def test_ca_direct_y_readout_matches_formula_order_and_padding():
    import torch

    readout = CADirectYReadout(dh=2, direct_y_dim=5)
    hm = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 5.0],
            [7.0, 11.0],
        ]
    )
    hc = torch.tensor(
        [
            [13.0, 17.0],
            [19.0, 23.0],
        ]
    )
    hg = torch.tensor([29.0, 31.0, 37.0, 41.0])
    masks = torch.tensor([[1.0, 0.0, 1.0]])
    incidence = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )

    out = readout({"hm": hm, "hc": hc, "hg": hg}, masks, graph={"incidence_t": incidence})
    expected = torch.tensor(
        [[8.0, 13.0, 32.0, 40.0, 1.0, 0.0, 1.0, 0.0, 0.0, 29.0, 31.0, 37.0, 41.0]]
    )
    assert torch.allclose(out, expected)
