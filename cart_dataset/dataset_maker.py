"""Shared helpers for detailed CART dataset generation."""

import logging
import math

import numpy as np
from scipy.stats.qmc import Sobol

from .config import FACILITIES_BOUND, PER_DAY_CAP, TIME_HORIZON, TMFE
from .demand import draw_inc_for_instance
from .model_cart_detailed import (
    build_detailed_data_from_aggregate,
    build_detailed_model,
    load_static_network,
)


logging.getLogger("pyomo").setLevel(logging.ERROR)


def sobol_totals(N: int, low: int, high: int, seed: int = 12345) -> np.ndarray:
    """Generate deterministic, space-filling integer demand totals."""
    count = high - low + 1
    sampler = Sobol(d=1, scramble=True, seed=seed)
    exponent = math.ceil(math.log2(max(1, N)))
    values = sampler.random_base2(exponent).ravel()[:N]
    return (low + np.floor(values * count)).astype(int)


def sample_demand_kwargs(rng: np.random.Generator):
    """Sample demand-generation hyperparameters."""
    distribution = rng.choice(["poisson", "seasonal_poisson", "bursty"])

    if distribution == "poisson":
        return {
            "distribution": "poisson",
            "lambda_demand": float(rng.uniform(2.0, 3.5)),
            "per_day_cap": PER_DAY_CAP,
        }

    if distribution == "seasonal_poisson":
        return {
            "distribution": "seasonal_poisson",
            "lambda_demand": float(rng.uniform(2.0, 3.5)),
            "season_amp": float(rng.uniform(0.3, 1.0)),
            "season_period": int(rng.choice([7, 14, 30, 60, 90])),
            "per_day_cap": PER_DAY_CAP,
        }

    return {
        "distribution": "bursty",
        "lambda_demand": float(rng.uniform(2.0, 3.5)),
        "n_bursts": int(rng.integers(1, 4)),
        "burst_len": int(rng.choice([3, 5, 7])),
        "burst_mult": float(rng.uniform(2.0, 4.0)),
        "per_day_cap": PER_DAY_CAP,
    }


def load_static_from_dat(dat_path: str):
    """Load the static network data used to build detailed CART instances."""
    static_sets, static_params = load_static_network(dat_path)
    static_params["TMFE"] = int(TMFE)
    return static_sets, static_params


def make_instance_data(
    static_sets,
    static_params,
    total_therapies,
    seed_demand,
    demand_kwargs,
    *,
    time_horizon=TIME_HORIZON,
    demand_horizon=None,
):
    """Generate patient parameters before any facility variables are built."""
    demand_horizon = time_horizon if demand_horizon is None else int(demand_horizon)
    if not 0 < demand_horizon <= int(time_horizon):
        raise ValueError("Demand horizon must be positive and within the model horizon.")
    inc = draw_inc_for_instance(
        static_sets["c"],
        demand_horizon,
        total_therapies,
        seed_demand,
        **demand_kwargs,
    )
    return build_detailed_data_from_aggregate(
        static_sets,
        static_params,
        inc,
        time_horizon=int(time_horizon),
    )


def make_instance(
    static_sets,
    static_params,
    total_therapies,
    seed_demand,
    demand_kwargs,
    facilities_bound=FACILITIES_BOUND,
    *,
    time_horizon=TIME_HORIZON,
    demand_horizon=None,
):
    """Generate demand and construct its detailed patient-level CART model."""
    data = make_instance_data(
        static_sets, static_params, total_therapies, seed_demand, demand_kwargs,
        time_horizon=time_horizon, demand_horizon=demand_horizon,
    )
    bound = None if facilities_bound is None else int(facilities_bound)
    model = build_detailed_model(
        time_horizon=int(time_horizon),
        max_open_facilities=bound,
    )
    return model.create_instance({None: data})


def main(argv=None):
    """Run the sampled CART16 generator through the main entry point."""
    from .dataset_maker_sampled import main as generate

    return generate(argv)


if __name__ == "__main__":
    raise SystemExit(main())
