"""Relative growth rate (RGR) and growth inhibition per dish, following OECD TG 221."""

import numpy as np
import pandas as pd


def _rgr_endpoint(days, values):
    """RGR = (ln N_t - ln N_0) / t using the first and last available observation."""
    ok = np.isfinite(values)
    days, values = days[ok], values[ok]
    if len(days) < 2 or values[0] <= 0 or values[-1] <= 0 or days[-1] == days[0]:
        return np.nan
    return (np.log(values[-1]) - np.log(values[0])) / (days[-1] - days[0])


def _rgr_slope(days, values):
    """Slope of ln(N) against day using all observations."""
    ok = np.isfinite(values) & (values > 0)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(days[ok], np.log(values[ok]), 1)[0])


def growth_table(df, metrics, control_group):
    """One row per dish with RGR (endpoint and slope) for each metric and % inhibition."""
    rows = []
    for dish_id, sub in df.sort_values("day").groupby("dish_id"):
        days = sub["day"].to_numpy(float)
        first = sub.iloc[0]
        row = {
            "dish_id": dish_id,
            "group": first["group"],
            "replicate": first["replicate"],
            "group_label": first["group_label"],
            "plastic": first["plastic"],
            "detergent": first["detergent"],
            "factorial": bool(first.get("factorial", True)),
            "series": int(first.get("series", 1)),
            "day_start": int(days[0]),
            "day_end": int(days[-1]),
            "n_photos": len(sub),
        }
        for m in metrics:
            vals = sub[m].to_numpy(float)
            fin = np.where(np.isfinite(vals))[0]
            row[f"{m}_start"] = vals[fin[0]] if len(fin) else np.nan
            row[f"{m}_end"] = vals[fin[-1]] if len(fin) else np.nan
            row[f"rgr_{m}"] = _rgr_endpoint(days, vals)
            row[f"rgr_slope_{m}"] = _rgr_slope(days, vals)
        if "yellow_pct_of_plant" in sub.columns:
            row["yellow_pct_end"] = sub["yellow_pct_of_plant"].iloc[-1]
        notes = []
        for m in metrics:
            if not np.isfinite(row[f"rgr_{m}"]):
                if row[f"{m}_end"] == 0:
                    notes.append(f"{m}: zero at end, RGR undefined")
                elif row[f"{m}_start"] == 0:
                    notes.append(f"{m}: zero at start, RGR undefined")
                else:
                    notes.append(f"{m}: fewer than two values")
        row["warnings"] = "; ".join(notes)
        rows.append(row)
    g = pd.DataFrame(rows)

    for m in metrics:
        col = f"rgr_{m}"
        ctrl = g.loc[g["group"] == control_group, col].mean()
        if np.isfinite(ctrl) and ctrl != 0:
            g[f"inhibition_pct_{m}"] = (ctrl - g[col]) / ctrl * 100.0
        else:
            g[f"inhibition_pct_{m}"] = np.nan
    return g.sort_values(["group", "replicate"]).reset_index(drop=True)


def group_summary(growth, metrics):
    cols = []
    for m in metrics:
        cols += [f"rgr_{m}", f"inhibition_pct_{m}"]
    if "yellow_pct_end" in growth.columns:
        cols.append("yellow_pct_end")
    agg = growth.groupby(["group", "group_label"])[cols].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def daily_summary(df, columns):
    agg = df.groupby(["group", "group_label", "day"])[columns].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()
