# gnn/tests/test_losses.py
import torch

from gnn.losses.listwise import (
    listwise_softmax_ce,
    listwise_softmax_ce_soft,
)
from gnn.losses.calibration import feasible_huber
from gnn.losses.combined import ebm_loss


def test_listwise_basic():
    # lower energy should help the positive -> lower loss
    energies_good = torch.tensor([0.0, 1.0, 5.0])  # positive is best
    energies_bad  = torch.tensor([5.0, 1.0, 0.0])  # positive is worst
    labels = ["positive", "ham_neg_feas", "ham_neg_feas"]

    L_good = listwise_softmax_ce(energies_good, labels).item()
    L_bad  = listwise_softmax_ce(energies_bad,  labels).item()

    print("[listwise] good:", L_good, "bad:", L_bad)
    assert L_good < L_bad, "Listwise loss should be lower when positive has lowest energy."

def test_listwise_multi_positive():
    # two positives; target mass is split uniformly
    energies = torch.tensor([0.1, 0.2, 3.0, 4.0])
    labels   = ["positive", "positive", "ham_neg_feas", "ham_neg_feas"]
    L = listwise_softmax_ce(energies, labels).item()
    print("[listwise] multi-positive loss:", L)
    # sanity: if we worsen the two positives, loss should go up
    L_worse = listwise_softmax_ce(energies + torch.tensor([2.0, 2.0, 0.0, 0.0]), labels).item()
    assert L < L_worse, "Loss should increase when positives get worse (higher energies)."


def test_gap_aware_listwise_scale_and_ordering():
    n = 200
    feasible = torch.ones(n, dtype=torch.bool)
    regret = torch.full((n,), 0.10)
    regret[0] = 0.0

    equal_energies = torch.zeros(n)
    L_equal = listwise_softmax_ce_soft(
        equal_energies,
        regret,
        feasible,
        temperature=1.0,
        tau=0.01,
    )
    print("[gap_aware_listwise] equal-energy loss:", L_equal.item())
    assert L_equal.item() > 4.0, "Soft listwise CE should stay on log(N) scale."

    energies_good = regret.clone()
    energies_bad = -regret.clone()
    L_good = listwise_softmax_ce_soft(
        energies_good,
        regret,
        feasible,
        temperature=0.1,
        tau=0.02,
    )
    L_bad = listwise_softmax_ce_soft(
        energies_bad,
        regret,
        feasible,
        temperature=0.1,
        tau=0.02,
    )
    assert L_good.item() < L_bad.item(), "Lower energy should be assigned to lower-regret masks."


def test_calibration():
    energies = torch.tensor([0.0, 0.5, 1.0,  2.0])
    regret   = torch.tensor([0.0, 0.5, float("nan"), float("nan")])  # last two infeasible → ignored
    L_h = feasible_huber(energies, regret, delta=0.1).item()
    print("[calibration] huber:", L_h)

    # Perfect match on feasible entries -> near zero
    L_zero = feasible_huber(torch.tensor([0.0, 0.5]), torch.tensor([0.0, 0.5]), delta=0.1).item()
    assert L_zero < 1e-6, "Calibration should be ~0 when energies == regret on feasible subset."


def test_combined():
    # One instance with 3 masks: pos, feas-neg, infeas-neg
    energies = torch.tensor([0.1, 1.0, -0.5])  # argmin is the infeasible one (bad)
    labels   = ["positive", "ham_neg_feas", "ham_neg_infeas"]
    feasible = torch.tensor([ True, True, False], dtype=torch.bool)
    regret   = torch.tensor([0.0, 1.0, float("nan")])  # only feasible have regret

    loss, parts = ebm_loss(
        energies=energies,
        label_types=labels,
        feasible_mask=feasible,
        regret_rel=regret,
        alpha=0.2, temperature=1.0,
    )
    numeric_parts = {k: float(v) for k, v in parts.items() if isinstance(v, (int, float))}
    print("[combined] total:", loss.item(), "| parts:", numeric_parts)
    assert loss.item() > 0, "Combined loss should be positive."
    # If we push the infeasible energy up and align feasible targets, loss should drop
    energies_better = torch.tensor([0.0, 1.0, 0.8])
    loss_better, _ = ebm_loss(
        energies=energies_better,
        label_types=labels,
        feasible_mask=feasible,
        regret_rel=regret,
        alpha=0.2, temperature=1.0,
    )
    assert loss_better.item() < loss.item(), "Loss should decrease when ranking/calibration improve."


def test_combined_variants():
    energies = torch.tensor([0.2, 0.8, 1.5])
    labels   = ["positive", "ham_neg_feas", "ham_neg_infeas"]
    feasible = torch.tensor([True, True, False], dtype=torch.bool)
    regret   = torch.tensor([0.0, 1.0, float("nan")])
    infeasible_target = torch.tensor([0.0, 1.0, 2.0])

    _, parts_full = ebm_loss(
        energies=energies,
        label_types=labels,
        feasible_mask=feasible,
        regret_rel=regret,
        infeasible_target=infeasible_target,
        alpha=0.5,
        lambda_inf_cal=1.0,
        gap_aware_ranking=True,
    )
    _, parts_no_gap = ebm_loss(
        energies=energies,
        label_types=labels,
        feasible_mask=feasible,
        regret_rel=regret,
        infeasible_target=infeasible_target,
        alpha=0.0,
        lambda_inf_cal=1.0,
        gap_aware_ranking=False,
    )
    _, parts_no_inf = ebm_loss(
        energies=energies,
        label_types=labels,
        feasible_mask=feasible,
        regret_rel=regret,
        infeasible_target=infeasible_target,
        alpha=0.5,
        lambda_inf_cal=0.0,
        gap_aware_ranking=True,
    )

    assert parts_full["gap_aware_ranking"] is True
    assert parts_no_gap["gap_aware_ranking"] is False
    assert parts_no_inf["inf_cal"] == 0.0


def test_rank_feasible_only_ignores_infeasible_denominator():
    labels = ["positive", "ham_neg_feas", "ham_neg_infeas"]
    feasible = torch.tensor([True, True, False], dtype=torch.bool)
    regret = torch.tensor([0.0, 1.0, float("nan")])

    loss_low_inf, _ = ebm_loss(
        energies=torch.tensor([0.1, 0.8, -10.0]),
        label_types=labels,
        feasible_mask=feasible,
        regret_rel=regret,
        gap_aware_ranking=True,
        rank_feasible_only=True,
        lambda_inf_cal=0.0,
    )
    loss_high_inf, _ = ebm_loss(
        energies=torch.tensor([0.1, 0.8, 10.0]),
        label_types=labels,
        feasible_mask=feasible,
        regret_rel=regret,
        gap_aware_ranking=True,
        rank_feasible_only=True,
        lambda_inf_cal=0.0,
    )

    assert abs(loss_low_inf.item() - loss_high_inf.item()) < 1e-8


def test_infeasible_margin_loss_uses_best_feasible_anchor():
    labels = ["positive", "ham_neg_feas", "ham_neg_infeas"]
    feasible = torch.tensor([True, True, False], dtype=torch.bool)
    regret = torch.tensor([0.0, 1.0, float("nan")])
    infeasible_target = torch.tensor([0.0, 0.0, 2.0])

    loss_bad, parts_bad = ebm_loss(
        energies=torch.tensor([0.5, 1.5, 1.0]),
        label_types=labels,
        feasible_mask=feasible,
        regret_rel=regret,
        infeasible_target=infeasible_target,
        lambda_inf_cal=1.0,
        rank_feasible_only=True,
        infeasible_margin_loss=True,
    )
    loss_good, parts_good = ebm_loss(
        energies=torch.tensor([0.5, 1.5, 3.0]),
        label_types=labels,
        feasible_mask=feasible,
        regret_rel=regret,
        infeasible_target=infeasible_target,
        lambda_inf_cal=1.0,
        rank_feasible_only=True,
        infeasible_margin_loss=True,
    )

    assert parts_bad["infeasible_margin_loss"] is True
    assert parts_good["infeasible_margin_loss"] is True
    assert loss_good.item() < loss_bad.item()


def test_infeasible_margin_loss_can_use_worst_feasible_anchor():
    labels = ["positive", "ham_neg_feas", "ham_neg_infeas"]
    feasible = torch.tensor([True, True, False], dtype=torch.bool)
    regret = torch.tensor([0.0, 1.0, float("nan")])
    infeasible_target = torch.tensor([0.0, 0.0, 0.2])

    _, parts_best = ebm_loss(
        energies=torch.tensor([0.0, 1.0, 0.5]),
        label_types=labels,
        feasible_mask=feasible,
        regret_rel=regret,
        infeasible_target=infeasible_target,
        lambda_inf_cal=1.0,
        rank_feasible_only=True,
        infeasible_margin_loss=True,
        infeasible_margin_anchor="best",
    )
    _, parts_worst = ebm_loss(
        energies=torch.tensor([0.0, 1.0, 0.5]),
        label_types=labels,
        feasible_mask=feasible,
        regret_rel=regret,
        infeasible_target=infeasible_target,
        lambda_inf_cal=1.0,
        rank_feasible_only=True,
        infeasible_margin_loss=True,
        infeasible_margin_anchor="worst",
    )

    assert parts_best["infeasible_margin_anchor"] == "best"
    assert parts_worst["infeasible_margin_anchor"] == "worst"
    assert parts_best["inf_cal"] == 0.0
    assert parts_worst["inf_cal"] > 0.0
