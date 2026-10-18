# Detailed CART Candidate Costs

Detailed CART candidate scoring is implemented in
`cart_dataset.dataset_maker_masks`. It can prepare patient-level inputs, score
candidate masks, and collect completed mask solves into training labels.

Prepare patient inputs without solving:

```powershell
python -m cart_dataset.dataset_maker_masks --dat-dir data/CART/m16/train_patient_dats --input-source data/CART/m16/train --split train --out-dir runs/CART16_train_detailed_labels --instance-index 0 --prepare-plans-only --time-horizon 130 --threads 8
```

Score masks by removing `--prepare-plans-only` and selecting the desired
instance or shard options. Use `--resume` when continuing an interrupted solve.
Use `--collect-only` after mask solves finish to gather labels for training.

The cluster array template is kept at `cart_dataset/jobs/cart16_labels.pbs`.
Submit it from the repository root with your Python environment activated and
use `qsub -V` to pass that environment to the job. If your cluster uses modules,
set `CART_GUROBI_MODULE` to its Gurobi module name before submitting. Configure
stdout and stderr locations using your scheduler's `qsub -o` and `qsub -e`
options; the template does not assume a particular cluster or home directory.
Generated mask outputs belong under ignored run directories rather than tracked
source.
