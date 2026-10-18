# Paper Experiments

The [README](../README.md) is the starting point for downloading data, training
EnerGNN, and running Gurobi inference. This index maps the paper's experiments
to their implementations; commands below use the included datasets and your
trained checkpoints.

| Paper experiment | Implementation |
| --- | --- |
| Main results, Section 4.3 / Table 1 | `gnn.training.train`, `gnn.inference.run_milp` |
| Cross-scale inference, Section 4.4 / Table 2 | Same inference module with a different test directory |
| Zero-shot constraints, Section 4.5 / Table 3 and Appendix H | `zero_shot_constraint_injection/` |
| Loss ablations, Section 4.6 / Table 4 | Training flags `--alpha 0` or `--lambda_inf_cal 0` |
| Neural-MIP formulation and solve, Section 4.7 / Table 5 | `gnn.inference.run_milp` |
| Energy-head and readout ablations, Appendix J | `gnn.models.energy_head`, `gnn.models.readout`, training variant flags |
| Reported comparison methods, Section 4.2 / Appendix K | Model variants in `gnn.models.model` and the shared training/inference modules |
| Dataset generation, Appendix D | `cart_dataset/`, `ca_dataset/`, `qkp_dataset/` |

## EnerGNN Head Variants

The MLP variant uses `--model_type energnn --ca_head_variant mlp32`. The linear
EnerGNN variant uses `--model_type energnn --ca_head_variant linear`.
Despite the flag name, the energy-head selector applies to all three benchmarks.
Keep each variant in a separate run directory and pass its head selection to
`gnn.inference.run_milp` too. The linear EnerGNN head is distinct from the
factorized Linear+MIP comparison models.

Table 1 includes both heads. The paper uses the linear CA head for cross-scale
and constraint-injection results, and the MLP head for the full CART and QKP runs.
The README uses those variants. For the CA MLP result:

```powershell
python -m gnn.training.train --problem ca --model_type energnn --ca_head_variant mlp32 --data_dir data/CA/175 --out_dir runs/energnn_ca_mlp
python -m gnn.inference.run_milp --problem ca --model_type energnn --ca_head_variant mlp32 --ckpt runs/energnn_ca_mlp/best.ckpt --data_dir data/CA/175/test
```

## Cross-Scale Generalization

Reuse the CART-16 checkpoint on the existing smaller and larger test directories:

```powershell
python -m gnn.inference.run_milp --problem cart --model_type energnn --ckpt runs/energnn_cart/best.ckpt --data_dir data/CART/m12/m12/test
python -m gnn.inference.run_milp --problem cart --model_type energnn --ckpt runs/energnn_cart/best.ckpt --data_dir data/CART/m20/m20/test
```

For CA, evaluate the checkpoint trained on 175 bids directly on the included
250-bid test set. Only the test split is provided for 250 bids:

```powershell
python -m gnn.inference.run_milp --problem ca --model_type energnn --ca_head_variant linear --ckpt runs/energnn_ca_linear/best.ckpt --data_dir data/CA/250/test
```

## Zero-Shot Constraints

CART supports targeted activation/deactivation and maximum active facility
counts, as well as removal of facilities selected in the original optimum.
Use `cart_facility_constraints_m16.yaml` for targeted scenarios or
`cart_facility_outage_ood_m16.yaml` for outage enumeration.

`--force-on` and `--force-off` accept zero-based facility indices. Their forced
states are combined with every count supplied to `--max-facilities`. The
original reference uses the baseline `solver.min_k` and `solver.max_k`; each new
limit is applied to its constrained reference, inference, and prediction
scoring. Results are grouped in `summary_by_constraint.csv`.

CA supports winner removal, forced rejected bids, and `--ca-k-max` for an
allocation-cardinality cap. Use `ca_bid_outage_ood_100x175.yaml` with a linear-head
CA checkpoint. Scenario lists and sampling settings are configurable in YAML.
All injected constraints are evaluated without retraining.

## Ablations and Comparison Implementations

For the loss ablations, append `--alpha 0` to disable feasible calibration or
`--lambda_inf_cal 0` to disable infeasibility shaping. Store each run separately.
For energy-head/readout variants, inspect `python -m gnn.training.train --help`.

The comparison implementations reported in the paper are retained in the shared
modules: `exact_solver`, `predict_and_search`, `gnn_fixing_mip`,
`conpas_shared_encoder`, `linear_mip_bce`, and `linear_mip_contrastive`.
`exact_solver` runs through the inference module without a learned checkpoint.
They are retained for the paper comparisons; EnerGNN is the default user workflow.
