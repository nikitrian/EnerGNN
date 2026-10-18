# Fixed knobs you stated
TIME_HORIZON = 110
TMFE = 7
FACILITIES_BOUND = 3
FACILITY_UTIL_CAP = 1.0
PER_DAY_CAP = 8

# Costs (fixed)
C_MATERIAL = 10476.0
C_QC = 9312.0

# Demand sampling defaults (only demand is uncertain)
DEFAULT_DEMAND = dict(
    distribution="seasonal_poisson",  # or "poisson","bursty","uniform","sparse_total"
    lambda_demand=3.0,
    season_amp=1.0,
    season_period=90,
    n_bursts=2,
    burst_len=6,
    burst_mult=3.0,
)
