# demand.py
import numpy as np
from typing import Dict, Tuple, Sequence

def generate_demand_profiles(
    c_sites,
    t_range,
    *,
    total_therapies=None,          # if provided, enforce exact total (feasible if ≤ 8*|c|*t_range)
    per_day_cap=8,
    distribution="poisson",        # "poisson", "uniform", "seasonal_poisson", "bursty", "sparse_total"
    lambda_demand=0.3,             # base λ for Poisson-like modes
    season_amp=0.5,                # amplitude for seasonal modulation (0..1), used in seasonal_poisson
    season_period=7,               # period for seasonal modulation (e.g., weekly)
    n_bursts=3,                    # for "bursty"
    burst_len=5,                   # for "bursty"
    burst_mult=3.0,                # λ multiplier inside bursts
    site_weights=None,             # optional weights per site (dict {c: weight} or list aligned with c_sites)
    rng_seed=42
):
    """
    Returns dict {(c, t): int demand} with 0 <= demand <= per_day_cap and no front-loading.

    Modes:
      - distribution="poisson": independent Poisson(λ) at each (c,t), capped (approximate total).
      - distribution="uniform": iid Uniform{0..per_day_cap} at each (c,t) (approximate total).
      - distribution="seasonal_poisson": λ_t = λ*(1 + season_amp*sin(2π t/period)), capped (approx total).
      - distribution="bursty": Poisson(λ) baseline + n random burst windows (length burst_len, multiplier burst_mult).
      - distribution="sparse_total": ignores Poisson; places exactly total_therapies as 1-by-1 balls into bins uniformly (or weighted), respecting per_day_cap.

    If total_therapies is not None with a Poisson-like distribution, we rescale to match exactly and
    redistribute any rounding diff randomly across the horizon (no front-loading).
    """
    rng = np.random.default_rng(rng_seed)
    C = list(c_sites)
    T = int(t_range)
    assert T >= 1 and len(C) >= 1

    # Site probs
    if site_weights is None:
        site_probs = np.full(len(C), 1.0 / len(C))
    else:
        if isinstance(site_weights, dict):
            w = np.array([site_weights.get(c, 1.0) for c in C], dtype=float)
        else:
            w = np.array(site_weights, dtype=float)
            assert len(w) == len(C), "site_weights must match c_sites length"
        w = np.clip(w, 1e-12, None)
        site_probs = w / w.sum()

    keys = [(c, t) for c in C for t in range(1, T + 1)]
    n_bins = len(keys)
    cap_vec = np.full(n_bins, per_day_cap, dtype=int)

    # helpers
    def seasonal_lambda(t):
        return lambda_demand * (1.0 + season_amp * np.sin(2.0 * np.pi * (t / season_period)))

    def burst_mask():
        mask = np.ones(T, dtype=float)
        starts = rng.integers(low=1, high=max(2, T - burst_len + 2), size=n_bursts)
        for s in starts:
            e = min(T, s + burst_len - 1)
            mask[s-1:e] *= burst_mult
        return mask

    # sparse_total: exact placement under caps
    if distribution == "sparse_total":
        assert total_therapies is not None, "sparse_total requires total_therapies"
        max_total = cap_vec.sum()
        if total_therapies > max_total:
            raise ValueError(f"total_therapies={total_therapies} exceeds capacity={max_total}")

        time_probs = np.full(T, 1.0 / T)
        probs = np.outer(site_probs, time_probs).ravel()
        probs = probs / probs.sum()

        demand = np.zeros(n_bins, dtype=int)
        remaining = total_therapies
        available = cap_vec.copy()
        while remaining > 0:
            mask = available > 0
            if not np.any(mask):
                raise RuntimeError("No remaining capacity to place therapies, but remaining > 0.")
            p = probs.copy()
            p[~mask] = 0.0
            p_sum = p.sum()
            if p_sum <= 0:
                p = mask.astype(float)
                p /= p.sum()
            else:
                p /= p_sum

            chunk = min(remaining, int(5_000))
            alloc = rng.multinomial(chunk, p)
            alloc = np.minimum(alloc, available)
            demand += alloc
            available -= alloc
            remaining -= int(alloc.sum())

        return {k: int(v) for k, v in zip(keys, demand)}

    # Poisson-like generators
    lam_t = np.full(T, lambda_demand, dtype=float)
    if distribution == "seasonal_poisson":
        lam_t = np.array([max(0.0, seasonal_lambda(t)) for t in range(1, T + 1)], dtype=float)
    elif distribution == "bursty":
        lam_t = lambda_demand * burst_mask()
    elif distribution == "poisson":
        pass
    elif distribution == "uniform":
        pass
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    if distribution in ("poisson", "seasonal_poisson", "bursty"):
        site_weight_norm = site_probs * len(C)
        lam_mat = np.outer(site_weight_norm, lam_t)  # (|C|, T)
        draws = rng.poisson(lam=lam_mat).astype(int)
    elif distribution == "uniform":
        draws = rng.integers(low=0, high=per_day_cap + 1, size=(len(C), T), dtype=int)

    draws = np.minimum(draws, per_day_cap)

    # Optional exact total with rescaling + random redistribution
    if total_therapies is not None:
        current_total = int(draws.sum())
        max_total = per_day_cap * len(C) * T
        if total_therapies > max_total:
            raise ValueError(f"total_therapies={total_therapies} exceeds capacity={max_total}")

        if current_total == 0 and total_therapies > 0:
            return generate_demand_profiles(
                c_sites=C, t_range=T, total_therapies=total_therapies, per_day_cap=per_day_cap,
                distribution="sparse_total", rng_seed=rng_seed
            )

        if current_total != total_therapies:
            v = draws.ravel().astype(int)
            cap = np.full_like(v, per_day_cap, dtype=int)

            if current_total > 0:
                scale = total_therapies / current_total
                v = np.minimum((v.astype(float) * scale + rng.uniform(0, 1, size=v.size)).astype(int), cap)
            else:
                v[:] = 0

            diff = int(total_therapies - v.sum())
            if diff != 0:
                idx = np.arange(v.size)
                if diff > 0:
                    for _ in range(diff):
                        mask = v < cap
                        if not np.any(mask): break
                        choices = idx[mask]
                        v[rng.choice(choices)] += 1
                else:
                    diff = -diff
                    for _ in range(diff):
                        mask = v > 0
                        if not np.any(mask): break
                        choices = idx[mask]
                        v[rng.choice(choices)] -= 1

            draws = v.reshape(len(C), T)

    out = {}
    for i, c in enumerate(C):
        for t in range(1, T + 1):
            out[(c, t)] = int(draws[i, t - 1])
    return out


def draw_inc_for_instance(
    c_sites: Sequence,
    t_range: int,
    total_therapies: int,
    seed: int,
    **kwargs
) -> Dict[Tuple[str, int], int]:
    """
    Convenience wrapper used by dataset_maker.py:
    returns the INC dict for one instance, given a seed.
    """
    return generate_demand_profiles(
        c_sites=c_sites,
        t_range=t_range,
        total_therapies=total_therapies,
        rng_seed=seed,
        **kwargs
    )

