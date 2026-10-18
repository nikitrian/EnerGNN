# Combinatorial Auction Datasets

| Directory | Real items | Bids | Splits | Purpose |
| --- | ---: | ---: | --- | --- |
| `175/` | 100 | 175 | train (800), val (100), test (100) | Training and in-distribution evaluation |
| `250/` | 100 | 250 | test (100) | Cross-scale evaluation without retraining |

Use a checkpoint trained on `175/` when evaluating `250/test/`. Both datasets
include solver references and candidate samples. Large `samples.jsonl` files
are stored with Git LFS. The original 175-bid generation summary is retained in
`175/summary.json`; its historical scale label is `large`.
