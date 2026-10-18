# Original EnerGNN trained runs

These files were copied from the original experiment runs. Each folder contains
the selected `best.ckpt`, original `run_config.json`, training metrics, and any
saved inference results. Checkpoints use Git LFS; run `git lfs pull` after cloning.

Keep these runs for inference and comparison. New training must use a separate
output directory; the trainer refuses nonempty output directories, including
when resuming. To resume, pass `--ckpt` and a new `--out_dir`.

Original configurations retain their historical data paths. Use `data/CA/175/test`
for CA, `data/CART/m16/test` for CART, and `data/QKP/test` for QKP in this repo.

Pass the head and readout settings recorded in each configuration when evaluating
EnerGNN.

All six EnerGNN checkpoints load with the current model code.

| Run folder | Model type | Energy head |
| --- | --- | --- |
| [ablations/CA/energnn_full_inf_01_tau_0005_linear](ablations/CA/energnn_full_inf_01_tau_0005_linear/run_config.json) | `energnn` | `linear` |
| [ablations/CA/energnn_full_mlp32_inf_01_tau_0005](ablations/CA/energnn_full_mlp32_inf_01_tau_0005/run_config.json) | `energnn` | `mlp32` |
| [ablations/CART/energnn_m16_full](ablations/CART/energnn_m16_full/run_config.json) | `energnn` | `mlp32` |
| [ablations/CART/energnn_m16_full_linear](ablations/CART/energnn_m16_full_linear/run_config.json) | `energnn` | `linear` |
| [ablations/QKP/full_lr1e5_tau005_t05_a005_linf01](ablations/QKP/full_lr1e5_tau005_t05_a005_linf01/run_config.json) | `energnn` | `mlp32` |
| [ablations/QKP/full_lr1e5_tau005_t05_a005_linf01_linear](ablations/QKP/full_lr1e5_tau005_t05_a005_linf01_linear/run_config.json) | `energnn` | `linear` |
