from __future__ import annotations

import math

import numpy as np
import pandas as pd


def logistic(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(x))))


def generate_panel_data(config: dict, heterogeneous_trend: bool) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["seed_population"]))
    n_units = int(config["n_units"])
    t0 = int(config["pre_periods"])
    t_total = t0 + int(config["post_periods"])
    times = np.arange(1, t_total + 1)

    x_draw = rng.uniform(0.0, 1.0, size=n_units)
    x1 = (x_draw >= 0.3).astype(int)
    x2 = (x_draw >= 0.7).astype(int)
    u = rng.uniform(0.0, 1.0, size=n_units)

    alpha0 = float(config["adoption_intercept"])
    alpha1 = float(config["adoption_slope_middle"])
    alpha2 = float(config["adoption_slope_late"])
    p1 = logistic(alpha0)
    p2 = np.array([logistic(alpha0 + alpha1 * value) for value in x1])
    p3 = np.array([logistic(alpha0 + alpha1 * value) for value in x1 + x2])
    p4 = np.array([logistic(alpha0 + alpha2 * value) for value in x1 + x2])

    cohort = np.zeros(n_units, dtype=int)
    cohort[u <= p1] = t0 + 1
    cohort[(u > p1) & (u <= p2)] = t0 + 2
    cohort[(u > p2) & (u <= p3)] = t0 + 3
    cohort[(u > p3) & (u <= p4)] = t0 + 4

    individual_effect = rng.normal(0.0, float(config["individual_sd"]), size=n_units)
    tau_i = rng.normal(float(config["tau_mean"]), float(config["tau_sd"]), size=n_units)
    time_shocks = rng.uniform(float(config["tau_time_low"]), float(config["tau_time_high"]), size=t_total + 1)

    rows = []
    for time in times:
        if heterogeneous_trend:
            trend_cfg = config["heterogeneous_trend"]
            trend = (
                (time / t_total) * float(trend_cfg["baseline_slope"]) * (1 - x1 - x2)
                + (time / t_total) * float(trend_cfg["x1_slope"]) * x1
                + (time / t_total) * float(trend_cfg["x2_slope"]) * x2
            )
        else:
            trend = np.full(n_units, time / t_total)

        error = rng.normal(0.0, float(config["error_sd"]), size=n_units)
        treated_ever = (cohort > 0).astype(int)
        y0 = float(config["base_level"]) + treated_ever * (-individual_effect) + (1 - treated_ever) * individual_effect + trend + error

        d = ((cohort > 0) & (time >= cohort)).astype(int)
        multiplier = np.zeros(n_units)
        multiplier[cohort == t0 + 1] = 1.0
        multiplier[cohort == t0 + 2] = -2.5
        multiplier[cohort == t0 + 3] = -1.75
        multiplier[cohort == t0 + 4] = -1.0
        tau_it = time_shocks[time] * np.abs(tau_i) * multiplier
        y = y0 + d * tau_it
        relative_time = np.where(cohort > 0, time - cohort, 0)

        for idx in range(n_units):
            rows.append(
                {
                    "id": idx + 1,
                    "x1": int(x1[idx]),
                    "x2": int(x2[idx]),
                    "cohort": int(cohort[idx]),
                    "time": int(time),
                    "relative_time": int(relative_time[idx]),
                    "d": int(d[idx]),
                    "y0": float(y0[idx]),
                    "tau_it": float(tau_it[idx]),
                    "y": float(y[idx]),
                }
            )

    return pd.DataFrame(rows, columns=["id", "x1", "x2", "cohort", "time", "relative_time", "d", "y0", "tau_it", "y"])


def summarize_group_shares_and_att(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per treated cohort and one row for all treated observations.
    """
    n_units = data["id"].nunique()
    total_obs = len(data)
    
    # Identify treated cohorts, skipping the never-treated (0) group
    treated_cohorts = sorted([c for c in data["cohort"].unique() if c > 0])
    
    rows = []
    for g in treated_cohorts:
        # Cohort share: Number of unique units in this cohort / total unique units
        cohort_units = data[data["cohort"] == g]["id"].nunique()
        fraction = cohort_units / n_units
        
        # ATT: Average tau_it for treated observations in this cohort
        treated_obs = data[(data["cohort"] == g) & (data["d"] == 1)]
        att = treated_obs["tau_it"].mean()
        
        rows.append({
            "group": f"cohort_{g}",
            "fraction": fraction,
            "att": att
        })
        
    # 'all_treated' row metrics
    all_treated_obs = data[data["d"] == 1]
    fraction_all = len(all_treated_obs) / total_obs
    att_all = all_treated_obs["tau_it"].mean()
    
    rows.append({
        "group": "all_treated",
        "fraction": fraction_all,
        "att": att_all
    })
    
    return pd.DataFrame(rows)


def estimate_cohort_did(data: pd.DataFrame, cohort: int, event_time: int, control_group: str) -> float:
    """
    Return a two-period DID estimate for one treatment cohort and event time.
    """
    target_period = cohort + event_time
    baseline_period = cohort - 1
    
    # 1. Treated group outcomes
    treated_mask = (data["cohort"] == cohort)
    y_treat_target = data[treated_mask & (data["time"] == target_period)]["y"].mean()
    y_treat_base = data[treated_mask & (data["time"] == baseline_period)]["y"].mean()
    
    # 2. Control group outcomes
    if control_group == "never":
        control_mask = (data["cohort"] == 0)
    elif control_group == "notyet":
        control_mask = (data["cohort"] == 0) | (data["cohort"] > target_period)
    else:
        raise ValueError("control_group must be 'never' or 'notyet'")
        
    y_control_target = data[control_mask & (data["time"] == target_period)]["y"].mean()
    y_control_base = data[control_mask & (data["time"] == baseline_period)]["y"].mean()
    
    # 3. DID Estimate
    treat_diff = y_treat_target - y_treat_base
    control_diff = y_control_target - y_control_base
    
    return float(treat_diff - control_diff)


def estimate_event_study(data: pd.DataFrame, event_times: list[int], control_group: str) -> pd.DataFrame:
    """
    Return cohort-event DID estimates.
    """
    treated_cohorts = sorted([c for c in data["cohort"].unique() if c > 0])
    valid_times = set(data["time"].unique())
    
    rows = []
    for g in treated_cohorts:
        for e in event_times:
            target_period = g + e
            baseline_period = g - 1
            
            # Ensure both baseline and target periods exist in the observed panel
            if target_period in valid_times and baseline_period in valid_times:
                estimate = estimate_cohort_did(data, g, e, control_group)
                rows.append({
                    "cohort": g,
                    "event_time": e,
                    "estimate": estimate
                })
                
    event_study_df = pd.DataFrame(rows)
    if not event_study_df.empty:
        event_study_df = event_study_df.sort_values(by=["cohort", "event_time"]).reset_index(drop=True)
        
    return event_study_df


def aggregate_post_treatment_effects(event_study: pd.DataFrame) -> float:
    """
    Return the average estimate over post-treatment event times.
    """
    post_treatment_data = event_study[event_study["event_time"] >= 0]
    return float(post_treatment_data["estimate"].mean())


def estimate_twfe_coefficient(data: pd.DataFrame) -> float:
    """
    Return the coefficient from a residualized two-way fixed effects regression of y on d.
    """
    # Unit and time averages, plus grand average
    y_bar_i = data.groupby("id")["y"].transform("mean")
    y_bar_t = data.groupby("time")["y"].transform("mean")
    y_bar = data["y"].mean()
    
    d_bar_i = data.groupby("id")["d"].transform("mean")
    d_bar_t = data.groupby("time")["d"].transform("mean")
    d_bar = data["d"].mean()
    
    # Residualize Y and D
    y_ddot = data["y"] - y_bar_i - y_bar_t + y_bar
    d_ddot = data["d"] - d_bar_i - d_bar_t + d_bar
    
    # Compute the pooled TWFE coefficient
    numerator = (d_ddot * y_ddot).sum()
    denominator = (d_ddot ** 2).sum()
    
    return float(numerator / denominator)
