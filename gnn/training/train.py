# gnn/training/train.py
import os
import math
import csv
import json
from pathlib import Path
import sys
from typing import Optional

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
import numpy as np
import torch
from torch.utils.data import DataLoader

from gnn.data.reader import (
    load_static_config,
    load_instances_meta_ca,
    load_instances_meta_qkp,
    load_instances_meta,
    load_samples,
    split_pools_for_training,
)
from gnn.data.dataset import InstancePoolDataset
from gnn.data.collate import instance_pool_collate
from gnn.models.model import (
    ConPaSSharedEncoder,
    GNNFixingMIPBaseline,
    GraphEBM,
    LinearMIPBCEBaseline,
    LinearMIPContrastiveBaseline,
    PredictAndSearchBaseline,
    model_display_name,
    resolve_ca_head_variant,
    resolve_ca_readout_variant,
    resolve_qkp_readout_variant,
)
from gnn.losses.combined import ebm_loss
from gnn.losses.conpas import conpas_contrastive_loss
from gnn.training.utils import (
    seed_everything, get_device, to_device_graph, to_device_tensor,
    save_checkpoint, load_checkpoint, split_instance_ids
)
from gnn.training.evaluator import eval_batch_metrics, reduce_epoch_metrics

TOPK_EVAL = (3, 5)
FEASIBLE_LABEL_TYPES = {"positive", "ham_neg_feas"}
MODEL_TYPES = {
    "energnn",
    "linear_mip_bce",
    "linear_mip_contrastive",
    "gnn_fixing_mip",
    "predict_and_search",
    "conpas_shared_encoder",
}

PAPER_TRAINING_DEFAULTS = {
    "cart": {
        "lr": 2e-6,
        "alpha": 1.0,
        "temperature": 0.1,
        "rank_tau": 0.05,
    },
    "ca": {
        "lr": 1e-6,
        "alpha": 0.01,
        "temperature": 0.2,
        "rank_tau": 0.005,
    },
    "qkp": {
        "lr": 1e-5,
        "alpha": 0.05,
        "temperature": 0.5,
        "rank_tau": 0.05,
    },
}
PAPER_EPOCHS = 2000
PAPER_WEIGHT_DECAY = 1e-5
PAPER_INFEASIBILITY_WEIGHT = 0.1


def resolve_paper_training_defaults(
    problem: str,
    *,
    epochs=None,
    lr=None,
    weight_decay=None,
    alpha=None,
    temperature=None,
    lambda_inf_cal=None,
    rank_tau=None,
) -> dict:
    """Resolve the paper's Table 7 settings while preserving overrides."""
    problem_key = str(problem).strip().lower()
    try:
        benchmark = PAPER_TRAINING_DEFAULTS[problem_key]
    except KeyError as exc:
        supported = ", ".join(PAPER_TRAINING_DEFAULTS)
        raise ValueError(f"Unsupported problem {problem!r}; expected one of: {supported}.") from exc

    return {
        "epochs": PAPER_EPOCHS if epochs is None else int(epochs),
        "lr": benchmark["lr"] if lr is None else float(lr),
        "weight_decay": PAPER_WEIGHT_DECAY if weight_decay is None else float(weight_decay),
        "alpha": benchmark["alpha"] if alpha is None else float(alpha),
        "temperature": benchmark["temperature"] if temperature is None else float(temperature),
        "lambda_inf_cal": (
            PAPER_INFEASIBILITY_WEIGHT
            if lambda_inf_cal is None
            else float(lambda_inf_cal)
        ),
        "rank_tau": benchmark["rank_tau"] if rank_tau is None else float(rank_tau),
    }


def _resolve_model_type(model_type: str) -> str:
    key = str(model_type).strip().lower()
    aliases = {
        "energnn": "energnn",
        "ebm": "energnn",
        "graph_ebm": "energnn",
        "linear_mip_contrastive": "linear_mip_contrastive",
        "linear_mip": "linear_mip_contrastive",
        "linear_additive": "linear_mip_contrastive",
        "linear_joint": "linear_mip_contrastive",
        "baseline": "linear_mip_contrastive",
        "linear_mip_bce": "linear_mip_bce",
        "linear_score_mip": "linear_mip_bce",
        "linear": "linear_mip_bce",
        "linear_score": "linear_mip_bce",
        "gnn_fixing_mip": "gnn_fixing_mip",
        "fixing": "gnn_fixing_mip",
        "gnn_fixing": "gnn_fixing_mip",
        "predict_and_search": "predict_and_search",
        "conpas_shared_encoder": "conpas_shared_encoder",
        "conpas": "conpas_shared_encoder",
    }
    if key not in aliases:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Expected one of: {', '.join(sorted(MODEL_TYPES))}."
        )
    return aliases[key]


def _positive_target_mask(masks_t: torch.Tensor, label_types) -> torch.Tensor:
    positive_indices = [i for i, label in enumerate(label_types) if label == "positive"]
    if not positive_indices:
        raise RuntimeError(
            "BCE baseline training requires one positive candidate per instance "
            "to provide the binary supervision target y*."
        )
    target = masks_t[int(positive_indices[0])].float()
    if target.dim() != 1:
        target = target.reshape(-1)
    return target


def _neural_diving_bce_loss(
    scores: torch.Tensor,
    masks_t: torch.Tensor,
    label_types,
) -> torch.Tensor:
    """
    Neural Diving training loss (Nair et al. 2020): average BCE across all
    positive (optimal/feasible) solutions in the candidate pool.  This trains
    the model to assign high probability to *any* good assignment rather than
    a single fixed one, matching the multi-solution imitation learning setup.
    Falls back to a single positive if only one is available.
    """
    positive_indices = [i for i, label in enumerate(label_types) if label == "positive"]
    if not positive_indices:
        raise RuntimeError(
            "Neural Diving BCE requires at least one positive candidate per instance."
        )
    loss_sum = scores.new_zeros(())
    for idx in positive_indices:
        target = masks_t[int(idx)].float().reshape(-1).to(device=scores.device)
        loss_sum = loss_sum + torch.nn.functional.binary_cross_entropy_with_logits(
            scores, target
        )
    return loss_sum / len(positive_indices)


def _safe_div(num, den):
    return float(num) / float(den) if den else float("nan")


def _epoch_metrics_from_agg(agg):
    cnt = max(int(agg.get("count", 0)), 1)
    out = {
        "top1_acc": _safe_div(agg.get("top1_acc", 0.0), cnt),
        "eps_top1_acc": _safe_div(agg.get("eps_top1_acc", 0.0), cnt),
        "mean_regret@1": _safe_div(agg.get("mean_regret@1", 0.0), cnt),
        "auc_feas": float("nan"),
    }
    auc_den = int(agg.get("auc_count", 0))
    if auc_den > 0:
        out["auc_feas"] = _safe_div(agg.get("auc_feas", 0.0), auc_den)

    for k in TOPK_EVAL:
        out[f"topk@{k}"] = _safe_div(agg.get(f"topk@{k}", 0.0), cnt)
        rk_sum = agg.get(f"regret_k_sum@{k}", 0.0)
        rk_count = int(agg.get(f"regret_k_count@{k}", 0))
        out[f"mean_regret@{k}"] = _safe_div(rk_sum, rk_count) if rk_count > 0 else float("nan")
    return out


def _append_epoch_metrics_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "model_type",
        "loss_variant",
        "lr",
        "train_loss",
        "train_rank_w",
        "train_cal_w",
        "train_inf_cal_w",
        "val_top1_acc",
        "val_eps_top1_acc",
        "val_mean_regret@1",
        "val_topk@3",
        "val_mean_regret@3",
        "val_topk@5",
        "val_mean_regret@5",
        "val_auc_feas",
        "val_ema_regret",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _append_summary_metrics_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "epoch",
        "model_type",
        "loss_variant",
        "lr",
        "top1_acc",
        "eps_top1_acc",
        "mean_regret@1",
        "topk@3",
        "mean_regret@3",
        "topk@5",
        "mean_regret@5",
        "auc_feas",
        "ema_regret",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def _build_dataset(metas, pools, TMFE, subset_ids, lambda_viol=1.0):
    sub_metas = {i: metas[i] for i in subset_ids}
    sub_pools = {i: pools[i] for i in subset_ids if i in pools}
    # Pass it to the Dataset constructor
    return InstancePoolDataset(
        sub_metas,
        sub_pools,
        TMFE=TMFE,
        lambda_viol=lambda_viol,
    )


def _load_problem_data(data_dir: str, problem: str):
    if problem == "ca":
        metas = load_instances_meta_ca(data_dir)
        pools_all = load_samples(f"{data_dir}/samples.jsonl", metas=metas)
        static = None
    elif problem == "qkp":
        metas = load_instances_meta_qkp(data_dir)
        pools_all = load_samples(f"{data_dir}/samples.jsonl", metas=metas)
        static = None
    else:
        static = load_static_config(f"{data_dir}/static_config.json")
        metas = load_instances_meta(f"{data_dir}/instances_meta.jsonl", static)
        pools_all = load_samples(f"{data_dir}/samples.jsonl", metas=metas)

    pools_listwise, _ = split_pools_for_training(pools_all)
    return metas, pools_listwise, static


def _resolve_data_dirs(
    data_dir: str,
    val_dir: Optional[str],
    test_dir: Optional[str],
    problem: str,
    scale: Optional[str],
):
    base = Path(data_dir)
    if scale:
        base = base / str(scale)

    if val_dir or test_dir:
        return base.as_posix(), val_dir, test_dir

    train_candidate = base / "train"
    val_candidate = base / "val"
    test_candidate = base / "test"
    if train_candidate.is_dir() and val_candidate.is_dir() and test_candidate.is_dir():
        print(
            f"[data] resolved split dirs from {base.as_posix()} -> "
            f"train={train_candidate.as_posix()} "
            f"val={val_candidate.as_posix()} "
            f"test={test_candidate.as_posix()}"
        )
        return (
            train_candidate.as_posix(),
            val_candidate.as_posix(),
            test_candidate.as_posix(),
        )

    if problem not in {"ca", "qkp"}:
        return base.as_posix(), val_dir, test_dir

    if any((base / name).is_dir() for name in ("175", "250", "small", "medium", "large")):
        raise ValueError(
            f"Dataset root {base.as_posix()} looks like a multi-scale directory. "
            f"Pass --scale 175|250, or point --data_dir directly at one scale."
        )

    return base.as_posix(), val_dir, test_dir


def _resolve_out_dir(out_dir: str, problem: str, scale: Optional[str], data_dir: str) -> str:
    default_outs = {"runs/exp1", "runs/exp5"}
    if str(out_dir) not in default_outs or problem != "ca":
        return str(out_dir)

    base_name = None
    if scale:
        base_name = str(scale)
    else:
        data_path = Path(data_dir)
        if data_path.name in {"train", "val", "test"} and data_path.parent.name:
            base_name = data_path.parent.name
        elif data_path.name:
            base_name = data_path.name

    if not base_name:
        base_name = "default"

    resolved = Path("cauctions_runs") / base_name
    print(f"[out] resolved CA output dir to {resolved.as_posix()}")
    return resolved.as_posix()


def run_train(
    data_dir="data/CART/m16",
    out_dir="runs/exp5",
    epochs=None,
    batch_size=1,             # listwise => 1 instance per batch
    lr=None,
    weight_decay=None,
    alpha=None,
    temperature=None,
    lambda_rank=1.0,
    lambda_inf_cal=None,
    rank_tau=None,
    tie_epsilon=0.015,
    inf_delta=0.1,
    use_amp=False,
    accum_steps=1,
    val_frac=0.1, test_frac=0.1, split_seed=123,
    ckpt=None, eval_only=False, save_every=1,
    early_stop_patience=None,
    energy_dump_dir=None,
    energy_gap_tol=0.02,
    lambda_viol=1.0,
    problem="cart",
    val_dir=None,
    test_dir=None,
    scale=None,
    loss_variant="full",
    loss_style="standard",
    infeasible_margin_anchor="best",
    model_type="energnn",
    mask_noise_std=0.0,
    ca_readout_variant="auto",
    ca_direct_y_dim=None,
    qkp_readout_variant="auto",
    qkp_direct_y_dim=None,
    ca_head_variant="mlp32",
    train_seed=10,
    ema_decay=0.90,
    checkpoint_metric="raw_regret",
    # ConPaS-specific training hyperparameters (used only when model_type == "conpas_shared_encoder")
    conpas_temperature=0.1,
    conpas_positive_selection="label",
    conpas_regret_pos_threshold=0.05,
    conpas_include_infeasible_negatives=True,
    conpas_include_suboptimal_negatives=True,
    conpas_positives_per_instance=None,
    conpas_negatives_per_instance=None,
):
    problem = str(problem).strip().lower()
    paper_defaults = resolve_paper_training_defaults(
        problem,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        alpha=alpha,
        temperature=temperature,
        lambda_inf_cal=lambda_inf_cal,
        rank_tau=rank_tau,
    )
    epochs = paper_defaults["epochs"]
    lr = paper_defaults["lr"]
    weight_decay = paper_defaults["weight_decay"]
    alpha = paper_defaults["alpha"]
    temperature = paper_defaults["temperature"]
    lambda_inf_cal = paper_defaults["lambda_inf_cal"]
    rank_tau = paper_defaults["rank_tau"]

    assert batch_size == 1, "Listwise training expects one instance per batch."
    seed_everything(int(train_seed))

    model_type = _resolve_model_type(model_type)
    model_name = model_display_name(model_type)
    device = get_device()
    use_amp = bool(use_amp and torch.cuda.is_available())
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"[model] type={model_type} name={model_name}")

    valid_loss_variants = {"full", "no_gap", "no_inf"}
    loss_variant = str(loss_variant).strip().lower()
    if loss_variant not in valid_loss_variants:
        raise ValueError(
            f"Unsupported loss_variant '{loss_variant}'. "
            f"Expected one of: {', '.join(sorted(valid_loss_variants))}."
        )
    valid_loss_styles = {"auto", "standard", "ca"}
    loss_style = str(loss_style).strip().lower()
    if loss_style not in valid_loss_styles:
        raise ValueError(
            f"Unsupported loss_style '{loss_style}'. "
            f"Expected one of: {', '.join(sorted(valid_loss_styles))}."
        )
    valid_checkpoint_metrics = {"raw_regret", "ema_regret"}
    checkpoint_metric = str(checkpoint_metric).strip().lower()
    if checkpoint_metric not in valid_checkpoint_metrics:
        raise ValueError(
            f"Unsupported checkpoint_metric '{checkpoint_metric}'. "
            f"Expected one of: {', '.join(sorted(valid_checkpoint_metrics))}."
        )

    gap_aware_ranking = loss_variant != "no_gap"
    effective_alpha = 0.0 if loss_variant == "no_gap" else float(alpha)
    effective_lambda_inf_cal = 0.0 if loss_variant == "no_inf" else float(lambda_inf_cal)
    effective_mask_noise_std = max(float(mask_noise_std or 0.0), 0.0)
    ca_readout_variant = resolve_ca_readout_variant(problem, ca_readout_variant)
    effective_ca_direct_y_dim = None
    qkp_readout_variant = resolve_qkp_readout_variant(problem, qkp_readout_variant)
    effective_qkp_direct_y_dim = None
    ca_head_variant = resolve_ca_head_variant(problem, ca_head_variant)
    if loss_style == "auto":
        rank_feasible_only = False
        infeasible_margin_loss = False
    elif loss_style == "ca":
        rank_feasible_only = True
        infeasible_margin_loss = True
    else:
        rank_feasible_only = False
        infeasible_margin_loss = False
    infeasible_margin_anchor = str(infeasible_margin_anchor).strip().lower()
    valid_margin_anchors = {"best", "worst"}
    if infeasible_margin_anchor not in valid_margin_anchors:
        raise ValueError(
            f"Unsupported infeasible_margin_anchor '{infeasible_margin_anchor}'. "
            f"Expected one of: {', '.join(sorted(valid_margin_anchors))}."
        )
    print(
        f"[loss] variant={loss_variant} "
        f"style={loss_style} "
        f"gap_aware_ranking={gap_aware_ranking} "
        f"alpha={effective_alpha:.4f} "
        f"lambda_inf_cal={effective_lambda_inf_cal:.4f}"
    )
    print(f"[masks] noise_std={effective_mask_noise_std:.4f} problem={problem}")
    if problem == "ca":
        print(f"[ca_readout] variant={ca_readout_variant}")
    if problem == "qkp":
        print(f"[qkp_readout] variant={qkp_readout_variant}")
    print(f"[head] variant={ca_head_variant}")
    print(
        f"[loss_mode] rank_feasible_only={rank_feasible_only} "
        f"infeasible_margin_loss={infeasible_margin_loss} "
        f"infeasible_margin_anchor={infeasible_margin_anchor}"
    )
    print(
        f"[seed] train_seed={int(train_seed)} | "
        f"[checkpoint] metric={checkpoint_metric} ema_decay={float(ema_decay):.4f}"
    )
    if model_type in {"gnn_fixing_mip", "predict_and_search"}:
        print(
            "[train_objective] BCE averaged across positive masks per instance; "
            "solution-level listwise loss settings are ignored for optimization."
        )
    elif model_type == "linear_mip_bce":
        print(
            "[train_objective] variable-wise BCE on one positive mask per instance; "
            "solution-level listwise loss settings are ignored for optimization."
        )
    elif model_type == "linear_mip_contrastive":
        print(
            "[train_objective] additive linear energy trained with the same "
            "solution-level contrastive/listwise loss as EnerGNN."
        )
    elif model_type == "conpas_shared_encoder":
        print(
            f"[train_objective] ConPaS InfoNCE contrastive loss on factorised "
            f"per-variable predictions (Huang et al., ICML 2024). "
            f"temperature={conpas_temperature:.4f} "
            f"positive_selection={conpas_positive_selection} "
            f"include_infeasible_neg={conpas_include_infeasible_negatives} "
            f"include_suboptimal_neg={conpas_include_suboptimal_negatives}"
        )

    data_dir, val_dir, test_dir = _resolve_data_dirs(
        data_dir=data_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        problem=problem,
        scale=scale,
    )

    data_path = Path(data_dir)
    if (
        not (data_path / "instances_meta.jsonl").is_file()
        and (data_path / "train").is_dir()
        and (data_path / "val").is_dir()
        and (data_path / "test").is_dir()
    ):
        train_candidate = data_path / "train"
        val_candidate = data_path / "val"
        test_candidate = data_path / "test"
        print(
            f"[data] resolved split dirs from {data_path.as_posix()} -> "
            f"train={train_candidate.as_posix()} "
            f"val={val_candidate.as_posix()} "
            f"test={test_candidate.as_posix()}"
        )
        data_dir = train_candidate.as_posix()
        val_dir = val_candidate.as_posix()
        test_dir = test_candidate.as_posix()

    out_dir = _resolve_out_dir(out_dir=out_dir, problem=problem, scale=scale, data_dir=data_dir)

    # Reject existing runs before loading data or writing configuration/metrics.
    out_dir_p = Path(out_dir)
    if out_dir_p.exists() and (not out_dir_p.is_dir() or any(out_dir_p.iterdir())):
        raise FileExistsError(
            f"Output directory is not empty: {out_dir_p}. "
            "Choose a new --out_dir to preserve existing checkpoints and results. "
            "To continue training, use --ckpt with a new --out_dir."
        )

    # ---- load data
    metas, pools_listwise, static = _load_problem_data(data_dir, problem)

    # ---- split by instance ids (random) or use fixed dirs
    use_fixed_split = bool(val_dir or test_dir)
    if use_fixed_split:
        tr_ids = sorted(list({*metas.keys()} & {*pools_listwise.keys()}))
        print(f"[split] train={len(tr_ids)} val=dir test=dir")
    else:
        all_ids = sorted(list({*metas.keys()} & {*pools_listwise.keys()}))
        tr_ids, va_ids, te_ids = split_instance_ids(
            all_ids, val_frac=val_frac, test_frac=test_frac, seed=split_seed
        )
        print(f"[split] train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)}")

    if problem in {"ca", "qkp"}:
        TMFE = 1
    else:
        TMFE = int(static.get("TMFE", 7))
    # Pass the param down
    train_ds = _build_dataset(
        metas,
        pools_listwise,
        TMFE,
        tr_ids,
        lambda_viol=lambda_viol,
    )

    if use_fixed_split:
        if val_dir:
            val_metas, val_pools, _ = _load_problem_data(val_dir, problem)
            val_ids = sorted(list({*val_metas.keys()} & {*val_pools.keys()}))
            val_ds = _build_dataset(val_metas, val_pools, TMFE, val_ids)
        else:
            val_ids = tr_ids[:max(1, len(tr_ids)//10 or 1)]
            val_ds = _build_dataset(metas, pools_listwise, TMFE, val_ids)

        if test_dir:
            test_metas, test_pools, _ = _load_problem_data(test_dir, problem)
            test_ids = sorted(list({*test_metas.keys()} & {*test_pools.keys()}))
            test_ds = _build_dataset(test_metas, test_pools, TMFE, test_ids)
        else:
            test_ds = None
    else:
        # if split produced empty val/test (e.g., small dataset), backfill minimally
        val_ids_fallback  = va_ids if va_ids else tr_ids[:max(1, len(tr_ids)//10 or 1)]
        test_ids_fallback = te_ids if te_ids else va_ids
        val_ds = _build_dataset(
            metas,
            pools_listwise,
            TMFE,
            val_ids_fallback,
        )
        test_ds = (
            _build_dataset(
                metas,
                pools_listwise,
                TMFE,
                test_ids_fallback,
            )
            if test_ids_fallback
            else None
        )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=instance_pool_collate, num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          collate_fn=instance_pool_collate, num_workers=0, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          collate_fn=instance_pool_collate, num_workers=0, pin_memory=True) if test_ds is not None else None

    # ---- model
    g0, _, _, _, _ = train_ds[0]
    dc, dm, de = g0["Xc"].shape[1], g0["Xm"].shape[1], g0["edge_attr"].shape[1]
    if problem == "qkp" and qkp_readout_variant == "direct_y":
        widths = []
        for ds_obj in (train_ds, val_ds, test_ds):
            if ds_obj is not None:
                widths.extend(len(ds_obj.metas[int(i)].M) for i in ds_obj.instance_ids)
        max_width = max(widths) if widths else int(g0["Xm"].shape[0])
        min_width = min(widths) if widths else max_width
        effective_qkp_direct_y_dim = (
            max_width if qkp_direct_y_dim is None else int(qkp_direct_y_dim)
        )
        if effective_qkp_direct_y_dim < max_width:
            raise ValueError(
                "--qkp_direct_y_dim is smaller than the largest QKP instance "
                f"({effective_qkp_direct_y_dim} < {max_width})."
            )
        print(
            f"[qkp_direct_y] dim={effective_qkp_direct_y_dim} "
            f"instance_width_range=[{min_width}, {max_width}]"
        )
    if problem == "ca" and ca_readout_variant == "direct_y":
        widths = []
        for ds_obj in (train_ds, val_ds, test_ds):
            if ds_obj is not None:
                widths.extend(len(ds_obj.metas[int(i)].M) for i in ds_obj.instance_ids)
        max_width = max(widths) if widths else int(g0["Xm"].shape[0])
        min_width = min(widths) if widths else max_width
        effective_ca_direct_y_dim = (
            max_width if ca_direct_y_dim is None else int(ca_direct_y_dim)
        )
        if effective_ca_direct_y_dim < max_width:
            raise ValueError(
                "--ca_direct_y_dim is smaller than the largest CA instance "
                f"({effective_ca_direct_y_dim} < {max_width})."
            )
        print(
            f"[ca_direct_y] dim={effective_ca_direct_y_dim} "
            f"instance_width_range=[{min_width}, {max_width}]"
        )
    if model_type == "linear_mip_contrastive":
        model = LinearMIPContrastiveBaseline(
            dc,
            dm,
            de,
            dh=128,
            nlayers=2,
            dropout=0.1,
            problem=problem,
        ).to(device)
    elif model_type == "gnn_fixing_mip":
        model = GNNFixingMIPBaseline(
            dc,
            dm,
            de,
            dh=128,
            nlayers=2,
            dropout=0.1,
            problem=problem,
        ).to(device)
    elif model_type == "predict_and_search":
        model = PredictAndSearchBaseline(
            dc,
            dm,
            de,
            dh=128,
            nlayers=2,
            dropout=0.1,
            problem=problem,
        ).to(device)
    elif model_type == "linear_mip_bce":
        model = LinearMIPBCEBaseline(
            dc,
            dm,
            de,
            dh=128,
            nlayers=2,
            dropout=0.1,
            problem=problem,
        ).to(device)
    elif model_type == "conpas_shared_encoder":
        model = ConPaSSharedEncoder(
            dc,
            dm,
            de,
            dh=128,
            nlayers=2,
            dropout=0.1,
            problem=problem,
        ).to(device)
    else:
        model = GraphEBM(
            dc,
            dm,
            de,
            dh=128,
            nlayers=2,
            dropout=0.1,
            problem=problem,
            ca_readout_variant=ca_readout_variant,
            ca_direct_y_dim=effective_ca_direct_y_dim,
            qkp_readout_variant=qkp_readout_variant,
            qkp_direct_y_dim=effective_qkp_direct_y_dim,
            ca_head_variant=ca_head_variant,
        ).to(device)
    
    # 1. Define Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # 2. Define Scheduler (Reduce LR when val plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=30,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-6,
    )

    # ---- resume / eval-only
    start_epoch = 1
    best_val = float("inf")
    no_improve = 0
    out_dir_p = Path(out_dir); out_dir_p.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = out_dir_p / "metrics.csv"
    summary_csv_path = out_dir_p / "summary_metrics.csv"
    with (out_dir_p / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "problem": problem,
                "model_type": model_type,
                "model_name": model_name,
                "data_dir": data_dir,
                "val_dir": val_dir,
                "test_dir": test_dir,
                "loss_variant": loss_variant,
                "loss_style": loss_style,
                "infeasible_margin_anchor": infeasible_margin_anchor,
                "ca_readout_variant": ca_readout_variant,
                "ca_direct_y_dim": effective_ca_direct_y_dim,
                "qkp_readout_variant": qkp_readout_variant,
                "qkp_direct_y_dim": effective_qkp_direct_y_dim,
                "ca_head_variant": ca_head_variant,
                "dh": 128,
                "nlayers": 2,
                "dropout": 0.1,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "batch_size": int(batch_size),
                "alpha": float(alpha),
                "temperature": float(temperature),
                "lambda_rank": float(lambda_rank),
                "lambda_inf_cal": float(lambda_inf_cal),
                "rank_tau": float(rank_tau),
                "tie_epsilon": float(tie_epsilon),
                "inf_delta": float(inf_delta),
                "adaptive_huber_kappa": 0.15,
                "adaptive_huber_min_delta": 0.01,
                "adaptive_huber_max_delta": 0.5,
                "lambda_viol": float(lambda_viol),
                "mask_noise_std": float(effective_mask_noise_std),
                "accum_steps": int(accum_steps),
                "early_stop_patience": early_stop_patience,
                "train_seed": int(train_seed),
                "ema_decay": float(ema_decay),
                "checkpoint_metric": checkpoint_metric,
                "epochs": int(epochs),
            },
            fh,
            indent=2,
        )
        fh.write("\n")

    if ckpt is not None and os.path.exists(ckpt):
        ck = load_checkpoint(model, opt if not eval_only else None, ckpt, device)
        start_epoch = int(ck.get("epoch", 0)) + 1
        print(f"[ckpt] loaded {ckpt} (epoch {ck.get('epoch', '?')})")

    energy_dump_dir = Path(energy_dump_dir) if energy_dump_dir else None
    energy_gap_tol = float(energy_gap_tol)

    def _make_energy_log(split: str, gap_tol: float):
        return {
            "split": split,
            "gap_tol": float(gap_tol),
            "energies": [],
            "feasible": [],
            "categories": [],
            "records": [],
        }

    def _jsonable_float(val):
        try:
            out = float(val)
        except Exception:
            return None
        return out if math.isfinite(out) else None

    def _candidate_records_path(out_path: Path) -> Path:
        out_path = Path(out_path)
        stem = out_path.stem.replace("_energy_dump", "_candidate_records")
        return out_path.with_name(f"{stem}.jsonl")

    def _append_energy_log(
        log,
        *,
        instance_id,
        energies_flat,
        feasible_flat,
        categories_flat,
        label_types_flat,
        objectives_flat,
        regret_rel_flat,
        reference_objective,
    ):
        if log is None:
            return
        log["energies"].extend(float(e) for e in energies_flat)
        log["feasible"].extend(bool(f) for f in feasible_flat)
        log["categories"].extend(str(c) for c in categories_flat)
        split = str(log.get("split", "unknown"))
        ref_obj = _jsonable_float(reference_objective)
        for cand_idx, energy_val in enumerate(energies_flat):
            rr_val = regret_rel_flat[cand_idx] if cand_idx < len(regret_rel_flat) else None
            rr_val = _jsonable_float(rr_val)
            log["records"].append(
                {
                    "split": split,
                    "instance_id": int(instance_id),
                    "candidate_id": int(cand_idx),
                    "energy": float(energy_val),
                    "objective": _jsonable_float(objectives_flat[cand_idx] if cand_idx < len(objectives_flat) else None),
                    "reference_objective": ref_obj,
                    "regret_rel": rr_val,
                    "gap_percent": (100.0 * rr_val) if rr_val is not None else None,
                    "feasible": bool(feasible_flat[cand_idx]) if cand_idx < len(feasible_flat) else False,
                    "label_type": str(label_types_flat[cand_idx]) if cand_idx < len(label_types_flat) else "",
                    "category": str(categories_flat[cand_idx]) if cand_idx < len(categories_flat) else "",
                }
            )

    def _save_energy_log(log, out_path: Path):
        if log is None or not log.get("energies"):
            return
        energies_arr = np.asarray(log["energies"], dtype=np.float32)
        feasible_arr = np.asarray(log["feasible"], dtype=np.bool_)
        categories_arr = np.asarray(log["categories"], dtype="<U32")
        split_arr = np.asarray([log.get("split", "unknown")], dtype="<U32")
        gap_tol_arr = np.asarray([log.get("gap_tol", float("nan"))], dtype=np.float32)
        count_arr = np.asarray([energies_arr.size], dtype=np.int64)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            energies=energies_arr,
            feasible_mask=feasible_arr,
            categories=categories_arr,
            split=split_arr,
            count=count_arr,
            optimal_gap_tol=gap_tol_arr,
        )
        records = log.get("records") or []
        if records:
            records_path = _candidate_records_path(out_path)
            with records_path.open("w", encoding="utf-8") as fh:
                for row in records:
                    fh.write(json.dumps(row))
                    fh.write("\n")

    def _eval_epoch(dloader, tag="[val]", energy_log=None):
        model.eval()
        agg = {"top1_acc": 0.0, "eps_top1_acc": 0.0, "mean_regret@1": 0.0, "auc_feas": 0.0, "auc_count": 0, "count": 0}
        for _k in TOPK_EVAL:
            agg[f"topk@{_k}"] = 0.0
            agg[f"regret_k_sum@{_k}"] = 0.0
            agg[f"regret_k_count@{_k}"] = 0
        printed = 0  # <--- add counter here
        with torch.no_grad():
            for container, masks_t, _, feasible_t, meta in dloader:
                container = to_device_graph(container, device)
                masks_t   = to_device_tensor(masks_t, device)
                feasible_t = to_device_tensor(feasible_t.to(torch.bool), device)

                rr_raw = meta.get("regret_rel")
                if rr_raw is None:
                    rr_list = [float("nan")] * masks_t.shape[0]
                else:
                    # convert each element; None -> NaN
                    rr_list = [float(x) if x is not None else float("nan") for x in rr_raw]

                # (optional) guard if lengths ever mismatch
                if len(rr_list) != masks_t.shape[0]:
                    rr_list = (rr_list + [float("nan")] * masks_t.shape[0])[:masks_t.shape[0]]

                regret_rel = torch.tensor(rr_list, dtype=torch.float32, device=device)

                energies = model(container, masks_t).squeeze(-1)
                if energy_log is not None:
                    energies_flat = energies.detach().cpu().view(-1).tolist()
                    feasible_raw = feasible_t.detach().cpu().view(-1).tolist()
                    label_types = meta.get("label_types") or []
                    objectives_raw = meta.get("objectives") or []
                    if len(objectives_raw) != len(energies_flat):
                        objectives_raw = (list(objectives_raw) + [None] * len(energies_flat))[:len(energies_flat)]
                    ref_idx = None
                    ref_rr = None
                    for idx, rr_val in enumerate(rr_list):
                        if not math.isfinite(rr_val):
                            continue
                        if ref_rr is None or rr_val < ref_rr:
                            ref_rr = rr_val
                            ref_idx = idx
                    reference_objective = (
                        objectives_raw[ref_idx]
                        if ref_idx is not None and ref_idx < len(objectives_raw)
                        else None
                    )
                    categories_flat = []
                    feasible_flags = []
                    for idx in range(len(energies_flat)):
                        lt = label_types[idx] if idx < len(label_types) else None
                        base_flag = bool(feasible_raw[idx])
                        rr_val = rr_list[idx] if idx < len(rr_list) else float("nan")
                        is_feasible = base_flag or (lt in FEASIBLE_LABEL_TYPES)
                        if is_feasible and math.isfinite(rr_val) and rr_val <= energy_gap_tol:
                            cat = "optimal"
                        elif is_feasible:
                            cat = "feasible_suboptimal"
                        else:
                            cat = "infeasible"
                            is_feasible = False
                        feasible_flags.append(is_feasible)
                        categories_flat.append(cat)
                    _append_energy_log(
                        energy_log,
                        instance_id=meta.get("instance_id", -1),
                        energies_flat=energies_flat,
                        feasible_flat=feasible_flags,
                        categories_flat=categories_flat,
                        label_types_flat=label_types,
                        objectives_flat=objectives_raw,
                        regret_rel_flat=rr_list,
                        reference_objective=reference_objective,
                    )

                metrics = eval_batch_metrics(
                    energies, meta["label_types"], feasible_t, regret_rel,
                    topk=TOPK_EVAL, tie_epsilon=tie_epsilon
                )
                for k, v in metrics.items():
                    if k == "auc_feas":
                        if math.isfinite(v):
                            agg["auc_feas"] += float(v)
                            agg["auc_count"] += 1
                        continue
                    if k.startswith("topk@"):
                        agg[k] += float(v)
                        continue
                    if k.startswith("mean_regret@") and k != "mean_regret@1":
                        suffix = k.split("@", 1)[1]
                        if math.isfinite(v):
                            agg[f"regret_k_sum@{suffix}"] += float(v)
                            agg[f"regret_k_count@{suffix}"] += 1
                        continue
                    agg[k] += float(v)
                agg["count"] += 1

        msg = reduce_epoch_metrics(agg, topk=TOPK_EVAL)
        print(msg.replace("[val]", tag))
        return _epoch_metrics_from_agg(agg)


    # ---- eval only
    if eval_only:
        val_log = _make_energy_log("val", energy_gap_tol) if energy_dump_dir is not None else None
        val_metrics = _eval_epoch(val_dl, tag="[val]", energy_log=val_log)
        _append_summary_metrics_row(
            summary_csv_path,
            {
                "epoch": "eval_only",
                "split": "val",
                "model_type": model_type,
                "loss_variant": loss_variant,
                "lr": opt.param_groups[0]["lr"],
                "top1_acc": val_metrics.get("top1_acc"),
                "eps_top1_acc": val_metrics.get("eps_top1_acc"),
                "mean_regret@1": val_metrics.get("mean_regret@1"),
                "topk@3": val_metrics.get("topk@3"),
                "mean_regret@3": val_metrics.get("mean_regret@3"),
                "topk@5": val_metrics.get("topk@5"),
                "mean_regret@5": val_metrics.get("mean_regret@5"),
                "auc_feas": val_metrics.get("auc_feas"),
            },
        )
        if val_log is not None:
            _save_energy_log(val_log, energy_dump_dir / "val_energy_dump.npz")
        if test_dl is not None:
            test_log = _make_energy_log("test", energy_gap_tol) if energy_dump_dir is not None else None
            test_metrics = _eval_epoch(test_dl, tag="[test]", energy_log=test_log)
            _append_summary_metrics_row(
                summary_csv_path,
                {
                    "epoch": "eval_only",
                    "split": "test",
                    "model_type": model_type,
                    "loss_variant": loss_variant,
                    "lr": opt.param_groups[0]["lr"],
                    "top1_acc": test_metrics.get("top1_acc"),
                    "eps_top1_acc": test_metrics.get("eps_top1_acc"),
                    "mean_regret@1": test_metrics.get("mean_regret@1"),
                    "topk@3": test_metrics.get("topk@3"),
                    "mean_regret@3": test_metrics.get("mean_regret@3"),
                    "topk@5": test_metrics.get("topk@5"),
                    "mean_regret@5": test_metrics.get("mean_regret@5"),
                    "auc_feas": test_metrics.get("auc_feas"),
                },
            )
            if test_log is not None:
                _save_energy_log(test_log, energy_dump_dir / "test_energy_dump.npz")
        return


    # ---- before the train loop
    ema_decay = float(ema_decay)  # EMA smoothing (0.8-0.95 are common)
    ema_regret = None             # lazy-init on first val
    best_score = float("inf")
    best_epoch = None
    no_improve = 0                # reset your patience counter
    best_metric_name = "EMA(regret@1)" if checkpoint_metric == "ema_regret" else "raw regret@1"
    print(
        f"[checkpoint] best.ckpt selection metric={best_metric_name}"
        + " | scheduler metric=EMA(regret@1)"
    )

    # ---- train loop
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        accum_steps_i = max(int(accum_steps), 1)
        opt.zero_grad(set_to_none=True)
        running = {
            "loss": 0.0,
            "rank_w": 0.0,
            "cal_w": 0.0,
            "inf_cal_w": 0.0,
        }

        for step_idx, (container, masks_t, targets_t, feasible_t, meta) in enumerate(train_dl):
            container   = to_device_graph(container, device)
            masks_t     = to_device_tensor(masks_t, device)
            targets_t   = to_device_tensor(targets_t, device)
            feasible_t  = to_device_tensor(feasible_t.to(torch.bool), device)

            if effective_mask_noise_std > 0.0:
                noise = torch.randn_like(masks_t) * effective_mask_noise_std
                masks_in = (masks_t + noise).clamp(0.0, 1.0)
            else:
                masks_in = masks_t

            rr_raw = meta.get("regret_rel")
            if rr_raw is None:
                rr_list = [float("nan")] * masks_t.shape[0]
            else:
                rr_list = [float(x) if x is not None else float("nan") for x in rr_raw]
            if len(rr_list) != masks_t.shape[0]:
                rr_list = (rr_list + [float("nan")] * masks_t.shape[0])[:masks_t.shape[0]]
            regret_rel = torch.tensor(rr_list, dtype=torch.float32, device=device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                if model_type == "conpas_shared_encoder":
                    # ConPaS InfoNCE contrastive loss on factorised per-variable predictions.
                    # Use clean binary masks_t (not noisy masks_in) since scores are log-probs.
                    logits_conpas, _ = model.predict_variables(container)
                    loss, _ = conpas_contrastive_loss(
                        logits=logits_conpas.view(-1),
                        masks=masks_t,
                        label_types=meta["label_types"],
                        feasible_mask=feasible_t,
                        regret_rel=regret_rel,
                        temperature=float(conpas_temperature),
                        positive_selection=str(conpas_positive_selection),
                        regret_pos_threshold=float(conpas_regret_pos_threshold),
                        include_infeasible_negatives=bool(conpas_include_infeasible_negatives),
                        include_suboptimal_negatives=bool(conpas_include_suboptimal_negatives),
                        positives_per_instance=conpas_positives_per_instance,
                        negatives_per_instance=conpas_negatives_per_instance,
                    )
                    parts = {"rank": 0.0, "cal": 0.0, "inf_cal": 0.0}
                elif model_type in {"gnn_fixing_mip", "predict_and_search", "linear_mip_bce"}:
                    scores = model.score_variables(container).view(-1)
                    if model_type in {"gnn_fixing_mip", "predict_and_search"}:
                        # BCE averaged over all positive solutions
                        loss = _neural_diving_bce_loss(scores, masks_t, meta["label_types"])
                    else:
                        target_mask = _positive_target_mask(masks_t, meta["label_types"]).to(device=device)
                        if scores.shape != target_mask.shape:
                            raise RuntimeError(
                                "BCE baseline score/target shape mismatch: "
                                f"{tuple(scores.shape)} vs {tuple(target_mask.shape)}"
                            )
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, target_mask)
                    parts = {"rank": 0.0, "cal": 0.0, "inf_cal": 0.0}
                else:
                    energies = model(container, masks_in).squeeze(-1)
                    loss, parts = ebm_loss(
                        energies=energies,
                        label_types=meta["label_types"],
                        feasible_mask=feasible_t,
                        regret_rel=regret_rel,
                        infeasible_target=targets_t,
                        alpha=effective_alpha, temperature=temperature,
                        lambda_rank=lambda_rank,
                        rank_tau=rank_tau,
                        tie_epsilon=tie_epsilon,
                        lambda_inf_cal=effective_lambda_inf_cal,
                        inf_delta=inf_delta,
                        gap_aware_ranking=gap_aware_ranking,
                        rank_feasible_only=rank_feasible_only,
                        infeasible_margin_loss=infeasible_margin_loss,
                        infeasible_margin_anchor=infeasible_margin_anchor,
                    )

            loss_scaled = loss / accum_steps_i
            scaler.scale(loss_scaled).backward()
            do_step = ((step_idx + 1) % accum_steps_i == 0) or (step_idx + 1 == len(train_dl))
            if do_step:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            running["loss"] += float(loss.item())
            running["rank_w"] += float(lambda_rank * parts["rank"])
            running["cal_w"]  += float(effective_alpha * parts["cal"])
            running["inf_cal_w"] += float(effective_lambda_inf_cal * parts["inf_cal"])

        n_batches = len(train_dl)
        print(f"[epoch {epoch}] train:"
            f" loss={(running['loss']/n_batches):.6f}"
            f" rank_w={(running['rank_w']/n_batches):.6f}"
            f" cal_w={(running['cal_w']/n_batches):.6f}"
            f" inf_cal_w={(running['inf_cal_w']/n_batches):.6f}")

        # ---- validation
        val_out = _eval_epoch(val_dl, tag="[val]")
        raw_regret = float(val_out["mean_regret@1"])
        top1 = float(val_out.get("top1_acc", float("nan")))
        auc_feas = float(val_out.get("auc_feas", float("nan")))

        # ---- EMA update
        if ema_regret is None:
            ema_regret = raw_regret  # init with first value
        else:
            ema_regret = ema_decay * ema_regret + (1.0 - ema_decay) * raw_regret

        print(f"[val] epoch={epoch} regret@1={raw_regret:.6f} | EMA={ema_regret:.6f}"
            + ("" if (top1 != top1) else f" | top1={top1:.3f}")
            + ("" if (auc_feas != auc_feas) else f" | auc_feas={auc_feas:.3f}"))

        _append_epoch_metrics_row(
            metrics_csv_path,
            {
                "epoch": epoch,
                "model_type": model_type,
                "loss_variant": loss_variant,
                "lr": opt.param_groups[0]["lr"],
                "train_loss": running["loss"] / n_batches,
                "train_rank_w": running["rank_w"] / n_batches,
                "train_cal_w": running["cal_w"] / n_batches,
                "train_inf_cal_w": running["inf_cal_w"] / n_batches,
                "val_top1_acc": val_out.get("top1_acc"),
                "val_eps_top1_acc": val_out.get("eps_top1_acc"),
                "val_mean_regret@1": val_out.get("mean_regret@1"),
                "val_topk@3": val_out.get("topk@3"),
                "val_mean_regret@3": val_out.get("mean_regret@3"),
                "val_topk@5": val_out.get("topk@5"),
                "val_mean_regret@5": val_out.get("mean_regret@5"),
                "val_auc_feas": val_out.get("auc_feas"),
                "val_ema_regret": ema_regret,
            },
        )

        # ---- step scheduler on validation metric
        prev_lr = opt.param_groups[0]["lr"]
        scheduler.step(ema_regret)
        new_lr = opt.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"[lr] reduced {prev_lr:.2e} -> {new_lr:.2e}")

        # ---- checkpoints (optional: keep periodic snapshots)
        if save_every and (epoch % int(save_every) == 0):
            ckpt_path = out_dir_p / f"epoch_{epoch:03d}.ckpt"
            save_checkpoint(model, opt, ckpt_path.as_posix(), epoch, best=False)
            save_checkpoint(model, opt, (out_dir_p / "latest.ckpt").as_posix(), epoch, best=False)

        # ---- checkpoint selection / early stopping
        selection_score = float(ema_regret) if checkpoint_metric == "ema_regret" else raw_regret
        if not math.isfinite(selection_score):
            selection_score = float("inf")
        improved = selection_score < best_score - 1e-8
        if improved:
            best_score = selection_score
            best_epoch = epoch
            save_checkpoint(model, opt, (out_dir_p / "best.ckpt").as_posix(), epoch, best=True)
            no_improve = 0
        else:
            no_improve += 1
            if (early_stop_patience is not None) and (no_improve >= early_stop_patience):
                print(f"[early-stop] {best_metric_name} did not improve for {no_improve} epochs; "
                    f"best at epoch {best_epoch} with {best_metric_name}={best_score:.6f}.")
                break
            
    # ---- test at the end
    if test_dl is not None:
        best_ckpt_path = out_dir_p / "best.ckpt"
        if best_epoch is not None and best_ckpt_path.exists():
            ck = load_checkpoint(model, None, best_ckpt_path.as_posix(), device)
            print(f"[test] loaded best.ckpt (epoch {ck.get('epoch', '?')}) for final evaluation")
        test_log = _make_energy_log("test", energy_gap_tol) if energy_dump_dir is not None else None
        test_out = _eval_epoch(test_dl, tag="[test]", energy_log=test_log)
        _append_summary_metrics_row(
            summary_csv_path,
            {
                "epoch": best_epoch if best_epoch is not None else "final",
                "split": "test",
                "model_type": model_type,
                "loss_variant": loss_variant,
                "lr": opt.param_groups[0]["lr"],
                "top1_acc": test_out.get("top1_acc"),
                "eps_top1_acc": test_out.get("eps_top1_acc"),
                "mean_regret@1": test_out.get("mean_regret@1"),
                "topk@3": test_out.get("topk@3"),
                "mean_regret@3": test_out.get("mean_regret@3"),
                "topk@5": test_out.get("topk@5"),
                "mean_regret@5": test_out.get("mean_regret@5"),
                "auc_feas": test_out.get("auc_feas"),
                "ema_regret": ema_regret,
            },
        )
        if test_log is not None:
            _save_energy_log(test_log, energy_dump_dir / "test_energy_dump.npz")



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/CART/m16")
    ap.add_argument("--out_dir",  type=str, default="runs/exp5")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Training epochs. Paper default: 2000.")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None,
                    help="Learning rate. Paper defaults: CART 2e-6, CA 1e-6, QKP 1e-5.")
    ap.add_argument("--weight_decay", type=float, default=None,
                    help="Adam weight decay. Paper default: 1e-5.")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Feasible calibration weight. Paper defaults: CART 1.0, CA 0.01, QKP 0.05.")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Prediction softmax temperature. Paper defaults: CART 0.1, CA 0.2, QKP 0.5.")
    ap.add_argument("--lambda_rank", type=float, default=1.0)
    ap.add_argument("--lambda_inf_cal", type=float, default=None,
                    help="Infeasibility shaping weight. Paper default: 0.1.")
    ap.add_argument("--rank_tau", type=float, default=None,
                    help="Target-ranking temperature. Paper defaults: CART 0.05, CA 0.005, QKP 0.05.")
    ap.add_argument("--tie_epsilon", type=float, default=0.015)
    ap.add_argument("--inf_delta", type=float, default=0.1,
                    help="Huber delta for infeasible calibration.")
    ap.add_argument(
        "--mask_noise_std",
        type=float,
        default=0.0,
        help="Stddev of Gaussian mask noise. Paper default: 0.0.",
    )
    ap.add_argument("--accum_steps", type=int, default=1,
                    help="Gradient accumulation steps to simulate a larger batch size.")

    ap.add_argument("--use_amp", action="store_true")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=123)
    ap.add_argument("--val_dir", type=str, default=None,
                    help="Optional fixed validation directory (overrides val_frac).")
    ap.add_argument("--test_dir", type=str, default=None,
                    help="Optional fixed test directory (overrides test_frac).")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--early_stop_patience", type=int, default=None)
    ap.add_argument("--train_seed", type=int, default=10,
                    help="Random seed for model initialization, shuffling, and training noise.")
    ap.add_argument("--ema_decay", type=float, default=0.90,
                    help="EMA smoothing for validation regret used by the LR scheduler.")
    ap.add_argument("--checkpoint_metric", type=str, default="raw_regret",
                    choices=["raw_regret", "ema_regret"],
                    help="Metric used for best.ckpt selection and early stopping.")
    ap.add_argument("--energy_dump_dir", type=str, default=None,
                    help="Optional directory to save evaluation energy dumps (.npz) and candidate-level records (.jsonl).")
    ap.add_argument("--energy_gap_tol", type=float, default=0.02,
                    help="Relative regret tolerance (e.g., 0.02) used to tag 'optimal' candidates in energy dumps.")
    ap.add_argument("--lambda_viol", type=float, default=1.0, 
                    help="Slope penalty for infeasible samples (target = lambda * viol)")
    ap.add_argument("--problem", type=str, default="cart", choices=["cart", "ca", "qkp"])
    ap.add_argument("--ca_readout_variant", type=str, default="sum_only_plus",
                    choices=[
                        "auto",
                        "pooled_only",
                        "sum_only_plus",
                        "direct_y",
                    ],
                    help="CA only. 'pooled_only' removes scalar z_lin; "
                         "'sum_only_plus' uses the paper's structural summaries; "
                         "'direct_y' appends a fixed-width raw mask for ablation.")
    ap.add_argument("--ca_direct_y_dim", type=int, default=None,
                    help="CA direct_y only: fixed raw-mask width. Defaults to the largest CA instance in the selected splits.")
    ap.add_argument("--qkp_readout_variant", type=str, default="auto",
                    choices=[
                        "auto",
                        "sum_only",
                        "size_aware",
                        "direct_y",
                    ],
                    help="QKP only. 'sum_only' removes structural summaries; "
                         "'size_aware' uses the paper's structural summaries; "
                         "'direct_y' appends a fixed-width raw mask for ablation.")
    ap.add_argument("--qkp_direct_y_dim", type=int, default=None,
                    help="QKP direct_y only: fixed raw-mask width. Defaults to the largest QKP instance in the selected splits.")
    ap.add_argument("--ca_head_variant", type=str, default="mlp32", choices=["auto", "mlp16", "mlp16_deep", "mlp32", "mlp32_deep", "mlp64", "mlp64_shallow", "linear"],
                    help="Energy head variant. Paper default: one hidden layer of width 32.")
    ap.add_argument("--scale", type=str, default=None, choices=["175", "250", "small", "medium", "large"],
                    help="Optional scale subdirectory under --data_dir for split-aware datasets.")
    ap.add_argument("--loss_variant", type=str, default="full", choices=["full", "no_gap", "no_inf"])
    ap.add_argument("--loss_style", type=str, default="standard", choices=["auto", "standard", "ca"],
                    help="Loss wiring profile. 'standard' is the paper loss; 'ca' enables the strict-margin ablation.")
    ap.add_argument("--infeasible_margin_anchor", type=str, default="best", choices=["best", "worst"],
                    help="When infeasible_margin_loss is active, anchor infeasible energies above the best feasible candidate or above the worst feasible candidate.")
    ap.add_argument("--model_type", type=str, default="energnn",
                    choices=["energnn", "linear_mip_bce", "linear_mip_contrastive", "gnn_fixing_mip", "predict_and_search", "conpas_shared_encoder"],
                    help="Model family. 'linear_mip_contrastive' is Linear+MIP(contrastive); "
                         "'gnn_fixing_mip' is Neural Diving with benchmark-specific dive counts; "
                         "'predict_and_search' is the original deterministic fixing baseline (linear head, K=1); "
                         "'linear_mip_bce' is Linear+MIP(BCE); "
                         "'conpas_shared_encoder' uses the ConPaS InfoNCE contrastive loss (Huang et al., ICML 2024).")
    # ConPaS hyperparameters
    ap.add_argument("--conpas_temperature", type=float, default=0.1,
                    help="ConPaS: softmax temperature tau for the InfoNCE loss. "
                         "Nair et al. suggest 0.05-0.1; lower = sharper contrast.")
    ap.add_argument("--conpas_positive_selection", type=str, default="label",
                    choices=["label", "regret_threshold"],
                    help="ConPaS: how to identify positives. 'label' uses label_type=='positive'; "
                         "'regret_threshold' uses feasible candidates with regret <= threshold.")
    ap.add_argument("--conpas_regret_pos_threshold", type=float, default=0.05,
                    help="ConPaS: regret threshold for positive_selection='regret_threshold'.")
    ap.add_argument("--conpas_include_infeasible_negatives", action="store_true", default=True,
                    help="ConPaS: include infeasible candidates as negatives.")
    ap.add_argument("--conpas_no_infeasible_negatives", dest="conpas_include_infeasible_negatives",
                    action="store_false",
                    help="ConPaS: exclude infeasible candidates from the negative set.")
    ap.add_argument("--conpas_include_suboptimal_negatives", action="store_true", default=True,
                    help="ConPaS: include suboptimal feasible candidates as negatives.")
    ap.add_argument("--conpas_no_suboptimal_negatives", dest="conpas_include_suboptimal_negatives",
                    action="store_false",
                    help="ConPaS: exclude suboptimal feasible candidates from the negative set.")
    ap.add_argument("--conpas_positives_per_instance", type=int, default=None,
                    help="ConPaS: cap the number of positives used per instance. None = use all.")
    ap.add_argument("--conpas_negatives_per_instance", type=int, default=None,
                    help="ConPaS: cap the number of negatives used per instance. None = use all.")
    args = ap.parse_args()

    run_train(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha, temperature=args.temperature,
        lambda_rank=args.lambda_rank,
        lambda_inf_cal=args.lambda_inf_cal,
        rank_tau=args.rank_tau,
        tie_epsilon=args.tie_epsilon,
        inf_delta=args.inf_delta,
        use_amp=args.use_amp,
        accum_steps=args.accum_steps,
        val_frac=args.val_frac, test_frac=args.test_frac, split_seed=args.split_seed,
        ckpt=args.ckpt, eval_only=args.eval_only, save_every=args.save_every,
        early_stop_patience=args.early_stop_patience,
        energy_dump_dir=args.energy_dump_dir,
        energy_gap_tol=args.energy_gap_tol,
        lambda_viol=args.lambda_viol,
        problem=args.problem,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
        scale=args.scale,
        loss_variant=args.loss_variant,
        loss_style=args.loss_style,
        infeasible_margin_anchor=args.infeasible_margin_anchor,
        model_type=args.model_type,
        mask_noise_std=args.mask_noise_std,
        ca_readout_variant=args.ca_readout_variant,
        ca_direct_y_dim=args.ca_direct_y_dim,
        qkp_readout_variant=args.qkp_readout_variant,
        qkp_direct_y_dim=args.qkp_direct_y_dim,
        ca_head_variant=args.ca_head_variant,
        train_seed=args.train_seed,
        ema_decay=args.ema_decay,
        checkpoint_metric=args.checkpoint_metric,
        conpas_temperature=args.conpas_temperature,
        conpas_positive_selection=args.conpas_positive_selection,
        conpas_regret_pos_threshold=args.conpas_regret_pos_threshold,
        conpas_include_infeasible_negatives=args.conpas_include_infeasible_negatives,
        conpas_include_suboptimal_negatives=args.conpas_include_suboptimal_negatives,
        conpas_positives_per_instance=args.conpas_positives_per_instance,
        conpas_negatives_per_instance=args.conpas_negatives_per_instance,
    )
