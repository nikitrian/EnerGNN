# EnerGNN

Code and datasets accompanying **EnerGNN: Learning Optimization-Compatible Energy Functions for Exact Constrained Combinatorial Inference** (NeurIPS 2026).

EnerGNN combines energy-based learning, graph neural networks, and mixed-integer optimisation to enable constrained inference over binary decisions. It learns expressive energy functions that admit an exact mixed-integer programming formulation, allowing optimisation over the learned energy while enforcing hard problem constraints. The framework is evaluated on quadratic knapsack (QKP), combinatorial auctions (CA), and cell therapy facility location and scheduling (CART).

## Get Started

Install [Git LFS](https://git-lfs.com/), then clone the repository and download
the datasets:

```powershell
git lfs install
git clone https://github.com/nikitrian/EnerGNN.git
cd EnerGNN
git lfs pull
```

Use Python 3.10 or newer. Install EnerGNN and its Gurobi dependency in your
preferred Python environment:

```bash
python -m pip install -e ".[solver]"
```

On macOS/Linux, use `python3` if `python` is unavailable.

Gurobi inference and dataset generation require a working license. Training on
existing datasets requires no solver; install with `python -m pip install -e .`
for training alone. `torch-scatter` is optional; the model has a PyTorch fallback.

## Run the QKP Example

Optimize the bundled pretrained energy on three QKP instances with Gurobi:

```powershell
python -m experiments.qkp_example --time-limit 60
```

The example reports solver status, QKP objective, capacity use, and learned
energy. Selected items and runtimes are saved to `outputs/qkp_example.json`.
See the [example guide](docs/qkp_example.md) for options.

## Train EnerGNN

Use the included `train/`, `val/`, and `test/` splits:

**CA**

```bash
python -m gnn.training.train --problem ca --model_type energnn --ca_head_variant linear --data_dir data/CA/175 --out_dir runs/energnn_ca_linear --train_seed 10 --tie_epsilon 0.005
```

**QKP**

```bash
python -m gnn.training.train --problem qkp --model_type energnn --data_dir data/QKP --out_dir runs/energnn_qkp --train_seed 10
```

**CART**

```bash
python -m gnn.training.train --problem cart --model_type energnn --data_dir data/CART/m16 --out_dir runs/energnn_cart --train_seed 10
```

The CA command uses the linear EnerGNN head used for the paper's cross-scale
and constraint-injection experiments. For the CA MLP variant, use
`--ca_head_variant mlp32` and a separate output directory.

The trainer detects the split directories, uses CUDA when available, and saves
`best.ckpt`, `run_config.json`, and metrics under the chosen output directory.
Defaults run for 2,000 epochs; append `--epochs 1` for one epoch. Validation
performance selects the checkpoint used for final test evaluation.
Training refuses a nonempty output directory to preserve existing runs. Choose
a new `--out_dir` for each run, including smoke tests. To continue training from
a checkpoint, pass `--ckpt` and save to a new `--out_dir`.

## Run Constrained Inference

Original EnerGNN checkpoints, configurations, metrics, and saved inference
results are included under `runs/pretrained/`, preserving the original
`ablations/` folders. Git LFS downloads the checkpoints during
cloning or with `git lfs pull`. These are separate from the new training outputs
above. See the [run inventory](runs/pretrained/README.md) for all six models.

Evaluate the original EnerGNN checkpoints on their matching test splits:

**CA**

```bash
python -m gnn.inference.run_milp --problem ca --model_type energnn --ca_head_variant linear --ckpt runs/pretrained/ablations/CA/energnn_full_inf_01_tau_0005_linear/best.ckpt --data_dir data/CA/175/test
```

**QKP**

```bash
python -m gnn.inference.run_milp --problem qkp --model_type energnn --ckpt runs/pretrained/ablations/QKP/full_lr1e5_tau005_t05_a005_linf01/best.ckpt --data_dir data/QKP/test
```

**CART**

```bash
python -m gnn.inference.run_milp --problem cart --model_type energnn --ckpt runs/pretrained/ablations/CART/energnn_m16_full/best.ckpt --data_dir data/CART/m16/test
```

For CA, QKP, and CART, standard learned-model evaluation reuses the best feasible
solution and objective stored in the dataset as the comparison reference.
Changing the constraints requires a new reference solve for the modified
problem. `--model_type exact_solver` explicitly solves the original optimization
problem instead of running a learned model.

For CA and QKP, a predicted mask's objective and feasibility are computed
directly from the instance data, without an additional optimization solve. If
no stored reference is available, the standard evaluation reports no objective
gap. CART also reuses stored detailed-model costs for predicted facility masks.
If a predicted mask has no stored cost, evaluation solves the detailed patient
scheduling model with that mask fixed to determine its cost and feasibility.
A missing CART reference or `--cart_exact_reference` triggers a detailed
reference solve as well. Before either additional solve, evaluation prints the
instance and reason, then streams Gurobi's solver output while it runs.

These additional CART solves can dominate evaluation time. `--time_limit`
defaults to 3,600 seconds per solve. For a quick check, append
`--limit 1 --time_limit 60` to the inference command; a shorter limit may leave
objectives or gaps unavailable or solutions without an optimality proof.

## Zero-Shot Constraint Injection

Try constraint injection on one test instance using the original pretrained
checkpoints. The CA example selects instance 0 and a reduced set of
perturbations. CART instance 6 is the smallest test instance by total therapies
(16).
Each solve has a 60-second time limit:

**CA: bid outages**

```bash
python -m zero_shot_constraint_injection.evaluate_ca_bid_outages --config zero_shot_constraint_injection/ca_bid_outage_ood_100x175.yaml --checkpoint runs/pretrained/ablations/CA/energnn_full_inf_01_tau_0005_linear/best.ckpt --instance-id 0 --time-limit 60 --removal-sizes 5 --random-scenarios-per-k 1 --force-rejected-sizes 1
```

**CART: facility constraints**

```bash
python -m zero_shot_constraint_injection.evaluate_cart_constraints --config zero_shot_constraint_injection/cart_facility_constraints_m16.yaml --checkpoint runs/pretrained/ablations/CART/energnn_m16_full/best.ckpt --instance-id 6 --time-limit 60 --force-on 1 --force-off 0 --max-facilities 6
```

**CART: facility outages**

```bash
python -m zero_shot_constraint_injection.evaluate_cart_facility_outages --config zero_shot_constraint_injection/cart_facility_outage_ood_m16.yaml --checkpoint runs/pretrained/ablations/CART/energnn_m16_full/best.ckpt --instance-id 6 --time-limit 60 --outage-sizes 1
```

For CART, indices start at zero: `--force-on 1` activates `m2`, `--force-off 0`
deactivates `m1`, and `--max-facilities 6` permits at most six active facilities.
These controls can be used separately or together. Each scenario is compared
with a Gurobi reference under the same constraints, without retraining.
`Using static config: ...` identifies the shared CART configuration file loaded
for the instance data; it is an informational message. These examples can still
perform several solves for the selected instance. A time-limited reference may
leave exact gaps unavailable or cause an instance to be skipped. For the full
evaluation, remove `--instance-id` and the example overrides for time and
scenario selection to use the YAML defaults.

## Repository Guide

| Purpose | Location |
| --- | --- |
| EnerGNN encoder, readout, and energy head | [`gnn/models/`](gnn/models/) |
| Training and losses | [`gnn/training/`](gnn/training/), [`gnn/losses/`](gnn/losses/) |
| Neural-MIP inference | [`gnn/inference/`](gnn/inference/) |
| Benchmark datasets | [`data/`](data/) |
| Dataset generation | [`qkp_dataset/`](qkp_dataset/), [`ca_dataset/`](ca_dataset/), [`cart_dataset/`](cart_dataset/) |
| Zero-shot experiments | [`zero_shot_constraint_injection/`](zero_shot_constraint_injection/) |
| Experiments by paper section | [Paper experiment guide](docs/paper_experiments.md) |
| Generate new data and run recipes | [Dataset generation guide](docs/reproduction.md) |

CA datasets use `data/CA/175/{train,val,test}` for the original benchmark and
`data/CA/250/test` for cross-scale evaluation. The 250-bid data is test-only.

Datasets occupy approximately 1,009 MiB. Large sample files use Git LFS; source
ZIPs may contain pointers instead of those files. For an existing clone, run
`git lfs pull`. Generated runs and results remain ignored by Git.

The original CART input is attributed in
[`cart_dataset/DATA_SOURCE.md`](cart_dataset/DATA_SOURCE.md).

## License and Citation

EnerGNN code is released under the [MIT license](LICENSE). Please cite the
paper when using this repository; citation metadata is provided in
[CITATION.cff](CITATION.cff):

> Triantafyllou, N., & Papathanasiou, M. M. (2026).
> [EnerGNN: Learning Optimization-Compatible Energy Functions for Exact Constrained Combinatorial Inference](https://neurips.cc/virtual/2026/loc/sydney/poster/148678).
> In *Advances in Neural Information Processing Systems (NeurIPS 2026)*.
