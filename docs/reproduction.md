# Dataset Generation and Experiment Recipes

For training on the included datasets, follow the [README](../README.md).
For experiment locations by paper section, see the [paper experiment guide](paper_experiments.md).
This guide covers generating new data; it is unnecessary for the pretrained example.
Gurobi and a working license are required for reference solving and candidate labeling.

## Entry Points

| Benchmark | Generate data | Additional preparation |
| --- | --- | --- |
| QKP | `qkp_dataset.dataset_maker` | `qkp_dataset.split_dataset` |
| CA | `ca_dataset.dataset_maker` | Splits are created by the generator |
| CART | `cart_dataset.dataset_maker` | `cart_dataset.patient_dats`, `cart_dataset.dataset_maker_masks` |

Run `python -m <module> --help` for each command's options.
CART's `dataset_maker` delegates to the sampled patient-level generator;
`dataset_maker_sampled` implements that workflow. `dataset_maker_masks` supports
scoring candidate masks for existing patient inputs. These share the detailed
CART formulation and candidate-scoring implementation.

## Recipes

The runner prints or executes the stages in `configs/reproduction/`:

```powershell
python -m experiments.reproduce --config configs/reproduction/qkp150.yaml --dry-run
python -m experiments.reproduce --config configs/reproduction/ca_100x175.yaml --dry-run
python -m experiments.reproduce --config configs/reproduction/cart_detailed.yaml --dry-run
```

Remove `--dry-run` to execute a recipe. Use `--stage train evaluate` to select
stages after their input datasets have been prepared. Recipes generate new
artifacts; the README commands use the already included datasets directly.

## QKP

The QKP recipe generates 1,000 150-item instances, solves references with a
3,600-second budget, and makes a seeded 800/100/100 split. Test instances are
sampled only from references marked `optimal: true`; the remaining instances
are randomly divided between training and validation. Solver objectives and
certification flags are preserved. If there are too few certified references
for the requested test size, splitting stops before writing output.

## CA

The CA recipe generates 100-item, 175-bid instances and creates the
800/100/100 train/validation/test splits under its `175/` directory. It trains the MLP-head EnerGNN variant.
The included datasets are `data/CA/175/{train,val,test}` and `data/CA/250/test`.
The generator accepts `--scale 175` or `--scale 250`; use `--splits test` to
generate only the cross-scale test split.
For linear-head CA training and cross-scale inference, use the commands in the
[paper experiment guide](paper_experiments.md).

## CART16

To generate sampled patient-level datasets from the supplied CART source input:

```powershell
python -m cart_dataset.dataset_maker --facility-counts 16 --train-count 800 --val-count 100 --test-count 100 --oversample-factor 2 --out-dir data/reproduced/CART_sampled_detailed
```

Inputs are saved before solving and completed solves are checkpointed. Use
`--prepare-only` to prepare inputs without solving and `--resume` to continue
an interrupted generation run. Patient `.dat` files are written beneath each
scale's split directories. Detailed solving uses a 17-day turnaround limit.

The `cart_detailed.yaml` recipe scores existing inputs under
`data/CART/m16/{train,val,test}` using the corresponding `{split}_patient_dats/`
directories. It prepares candidate plans, scores masks, collects labels, trains,
and evaluates. If those patient directories are absent, export them first:

```powershell
python -m cart_dataset.patient_dats --instances-meta-jsonl data/CART/m16/train/instances_meta.jsonl --static-config data/CART/m16/train/static_config.json --out-dir data/CART/m16/train_patient_dats --time-horizon 130
```

Repeat for `val` and `test`. When using newly generated inputs, update the recipe's
paths to the new dataset directories. See [detailed CART candidate costs](cart_detailed_masks.md)
for mask scoring, resuming, and the optional cluster job template.

## CART Input Attribution

`cart_dataset/Data_CART_og.dat` supplies the source CART network and patient inputs.
It comes from Triantafyllou et al., *A digital platform for the design of
patient-centric supply chains* (2022),
[doi:10.1038/s41598-022-21290-5](https://doi.org/10.1038/s41598-022-21290-5).
See the [data attribution](../cart_dataset/DATA_SOURCE.md).
