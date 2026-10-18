# Pretrained QKP Example with Gurobi

This example uses the bundled EnerGNN checkpoint and three 150-item QKP test
instances. It requires Gurobi and a working license that supports the encoded
neural MIP model. Training and dataset generation are unnecessary.

From the repository root, install the solver dependencies and run:

```powershell
python -m pip install -e ".[solver]"
python -m experiments.qkp_example --time-limit 60
```

For each instance, the example encodes the learned energy as a mixed-integer
model and uses Gurobi to minimize it over binary item selections subject to
the knapsack capacity constraint. It uses the same QKP inference implementation
as `gnn.inference.run_milp`. The instance graphs and pretrained checkpoint are
loaded on the CPU.

The default budget is 60 seconds per instance with one solver thread. Run only
one instance or increase the budget and thread count with:

```powershell
python -m experiments.qkp_example --limit 1 --time-limit 300 --solver-threads 8
```

## Results

The example prints each instance's solver status, QKP objective, total selected
weight, capacity, and learned energy. Results are written to
`outputs/qkp_example.json`, including the selected item indices and runtime.
Choose another output path with:

```powershell
python -m experiments.qkp_example --output-json outputs/qkp_gurobi_results.json
```

`OPTIMAL` means Gurobi proved optimality for the learned-energy MIP within its
solver tolerances; `SUBOPTIMAL` means it returned an incumbent without that
proof. Minimizing learned energy does not prove that the original QKP profit
is globally maximal. A time limit can end without a solution; such instances
are recorded as `NO_SOLUTION`, and the command returns a nonzero exit status.
Numerical objectives and solve times depend on solver settings and hardware.

## Included Assets

`examples/qkp/` contains `best.ckpt`, its training configuration, and the three
QKP input files for test instances 5, 15, and 35. The manifest records model
settings and asset checksums, which the example verifies before solving.

To use another compatible pretrained checkpoint:

```powershell
python -m experiments.qkp_example --ckpt runs/energnn_qkp/best.ckpt
```
