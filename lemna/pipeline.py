"""Complete workflow: photos -> measurements -> growth rates -> statistics -> figures.

Every run is archived in results/runs/<run_id>/ (run_id = analysis date and time).
The files results/history_runs.csv, history_measurements.csv and
history_growth_per_dish.csv accumulate the output of all runs with their run_id.
"""

import json
import shutil
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .growth import daily_summary, group_summary, growth_table
from .measure import (analyse_folder, area_column, calibrate_area_count, dishes_without_manual_endpoints,
                      frond_column, merge_manual_counts)
from .plots import make_all
from .stats import full_report


def list_runs(results_root):
    """Archived run ids, newest first."""
    runs_dir = Path(results_root) / "runs"
    if not runs_dir.exists():
        return []
    return sorted([p.name for p in runs_dir.iterdir() if (p / "measurements.csv").exists()], reverse=True)


def _append_csv(path, frame):
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True, sort=False)
    frame.to_csv(path, index=False)


def _run_summary(run_id, analysed_at, note, photo_dir, df, growth, tables, metrics, cfg, per_frond):
    row = {
        "run_id": run_id,
        "analysed_at": analysed_at,
        "note": note,
        "photo_folder": str(photo_dir),
        "n_photos": len(df),
        "n_dishes": df["dish_id"].nunique(),
        "days": " ".join(str(d) for d in sorted(df["day"].unique())),
        "first_photo": df["photo_datetime"].min(),
        "last_photo": df["photo_datetime"].max(),
        "dish_mode": cfg["dish"]["mode"],
        "hsv_green": f"{cfg['hsv']['green_low']}-{cfg['hsv']['green_high']}",
        "area_per_frond": per_frond,
    }
    for m in metrics:
        col = f"rgr_{m}"
        for g, v in growth.groupby("group")[col].mean().items():
            row[f"mean_{col}_{g}"] = v
        anova = tables.get(f"anova_{m}")
        if anova is not None:
            for _, a in anova.dropna(subset=["PR(>F)"]).iterrows():
                row[f"p_{m}_{a['source']}"] = a["PR(>F)"]
        rob = tables.get(f"robust_{m}")
        if rob is not None:
            inter = rob.set_index("source").loc["C(plastic):C(detergent)"]
            row[f"p_perm_{m}_interaction"] = inter["p_permutation"]
            row[f"p_logrgr_{m}_interaction"] = inter["p_anova_log_rgr"]
    if "agreement" in tables:
        for _, a in tables["agreement"].iterrows():
            key = "area" if "area" in a["method"] else "watershed"
            row[f"agreement_{key}_pearson_r"] = a["pearson_r"]
            row[f"agreement_{key}_mae"] = a["mae"]
    return pd.DataFrame([row])


def quality_checks(df, cfg):
    """Warnings about photos or data that need a look before the results are used."""
    warnings = []
    failed = df.loc[df["dish_method"].str.contains("failed", na=False), "file"].tolist()
    if failed:
        warnings.append(f"Dish circle not found, default circle used ({len(failed)} photos): " + ", ".join(failed))
    if cfg["dish"]["mode"] == "auto" and df["mm_per_px"].notna().sum() >= 3:
        med = df["mm_per_px"].median()
        dev = (df["mm_per_px"] / med - 1).abs()
        odd = df.loc[dev > 0.05, "file"].tolist()
        if odd:
            warnings.append("Dish size differs by more than 5 % from the median, check the dish circle "
                            f"({len(odd)} photos): " + ", ".join(odd))
    empty = df.loc[df["plant_area_px"] == 0, "file"].tolist()
    if empty:
        warnings.append(f"No plant tissue detected ({len(empty)} photos): " + ", ".join(empty))
    unknown = sorted(set(df["group"]) - set(cfg["groups"]))
    if unknown:
        warnings.append("Groups not defined in config.json (left out of the factorial tests): " + ", ".join(unknown))
    days = set(df["day"])
    incomplete = [f"{d} (missing days {sorted(days - set(sub['day']))})"
                  for d, sub in df.groupby("dish_id") if set(sub["day"]) != days]
    if incomplete:
        warnings.append("Dishes without a photo on every day: " + ", ".join(incomplete))
    if "frond_count_manual" in df.columns:
        missing = dishes_without_manual_endpoints(df)
        if missing:
            warnings.append("Manual counts are missing on the first or last day for "
                            f"{len(missing)} dishes, so growth rates use the automatic count: " + ", ".join(missing))
    return warnings


def run(photo_dir, cfg, results_root, manual_csv=None, save_overlays=True, progress=None,
        note="", make_figures=True, stage=None, endpoints_csv=None, log_csv=None):
    """stage(name) is called when a step starts: images, statistics, saving, figures.

    endpoints.csv and daily_log.csv are taken from the folder of the manual-count
    sheet unless given explicitly.
    """
    ws = Path(manual_csv).parent if manual_csv is not None else Path(photo_dir).parent
    endpoints_csv = endpoints_csv or ws / "endpoints.csv"
    log_csv = log_csv or ws / "daily_log.csv"
    stage = stage or (lambda name: None)
    results_root = Path(results_root)
    now = datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H-%M-%S")
    analysed_at = now.isoformat(sep=" ", timespec="seconds")
    base, k = run_id, 1
    while (results_root / "runs" / run_id).exists():
        k += 1
        run_id = f"{base}_{k}"
    out_dir = results_root / "runs" / run_id
    out_dir.mkdir(parents=True)

    stage("images")
    try:
        df, skipped = analyse_folder(photo_dir, cfg, out_dir, save_overlays, progress)
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    if df.empty:
        shutil.rmtree(out_dir, ignore_errors=True)
        return {"measurements": df, "skipped": skipped, "report": "No correctly named photos were found."}

    df = merge_manual_counts(df, manual_csv)
    df, per_frond, cal_day = calibrate_area_count(df, cfg)
    if cal_day is not None:
        df.attrs["calibration_day"] = cal_day
    metrics = [area_column(df), frond_column(df)]
    if "frond_count_manual" in df.columns and frond_column(df) == "frond_count_manual":
        metrics.append("frond_count_auto")

    growth = growth_table(df, metrics, cfg["control_group"])
    ends = read_endpoints(endpoints_csv)
    chem, chem_daily = dish_water_chemistry(log_csv)
    for extra_table in (ends, chem):
        if len(extra_table):
            growth = growth.merge(extra_table, on=["group", "replicate"], how="left")
    extra = [(c, d) for c, d in EXTRA_RESPONSES if c in growth.columns and growth[c].notna().sum() >= 4]
    planned = [tuple(p) for p in cfg.get("planned_comparisons", [])]
    stage("statistics")
    report, tables = full_report(df, growth, metrics, extra=extra, planned=planned)
    if len(chem_daily) and chem_daily.filter(like="_count").to_numpy().sum() > 0:
        tables["water_chemistry_daily"] = chem_daily
    qc = quality_checks(df, cfg) + tables.pop("_area_per_frond_warnings", [])
    status = tables.get("mixed_model_status")
    if status is not None:
        for r in status.itertuples():
            if not r.converged:
                qc.append(f"Mixed model for {r.response} did not converge; do not report its estimates")
    if qc:
        report = "QUALITY CHECKS (look at these before using the results)\n" + \
                 "\n".join(f"  ! {w}" for w in qc) + "\n\n" + report
    if per_frond is not None:
        unit = "mm2" if area_column(df) == "plant_area_mm2" else "px"
        report += f"\n\nArea per frond used for the area-based count: {per_frond:.4g} {unit}"
    if skipped:
        report += "\n\nSkipped files (name not recognised or unreadable):\n  " + "\n  ".join(skipped)

    daily_cols = [c for c in metrics + ["green_pct_of_roi", "yellow_pct_of_plant", "brown_pct_of_plant", "exg_index"]
                  if c in df.columns]
    daily = daily_summary(df, daily_cols)
    summary = group_summary(growth, metrics)

    df.insert(0, "run_id", run_id)
    growth.insert(0, "run_id", run_id)
    stage("saving")
    df.to_csv(out_dir / "measurements.csv", index=False)
    growth.to_csv(out_dir / "growth_per_dish.csv", index=False)
    summary.to_csv(out_dir / "growth_per_group.csv", index=False)
    daily.to_csv(out_dir / "daily_summary.csv", index=False)
    (out_dir / "statistics_report.txt").write_text(report, encoding="utf-8")

    with pd.ExcelWriter(out_dir / "results.xlsx") as xw:
        df.to_excel(xw, sheet_name="measurements", index=False)
        growth.to_excel(xw, sheet_name="growth_per_dish", index=False)
        summary.to_excel(xw, sheet_name="growth_per_group", index=False)
        daily.to_excel(xw, sheet_name="daily_summary", index=False)
        for name, t in tables.items():
            t.to_excel(xw, sheet_name=name[:31], index=False)

    if make_figures:
        stage("figures")
    figures = make_all(df, growth, metrics, out_dir) if make_figures else []

    with open(out_dir / "config_used.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    with open(out_dir / "run_info.json", "w", encoding="utf-8") as fh:
        json.dump({"run_id": run_id, "analysed_at": analysed_at, "note": note, "metrics": metrics,
                   "photo_folder": str(photo_dir), "skipped": skipped, "calibration_day": cal_day,
                   "area_per_frond": per_frond, "quality_warnings": qc,
                   "extra_responses": [c for c, _ in extra], "planned_comparisons": planned},
                  fh, indent=2, ensure_ascii=False)

    summary_row = _run_summary(run_id, analysed_at, note, photo_dir, df, growth, tables, metrics, cfg, per_frond)
    _append_csv(results_root / "history_runs.csv", summary_row)
    _append_csv(results_root / "history_measurements.csv", df.assign(analysed_at=analysed_at))
    _append_csv(results_root / "history_growth_per_dish.csv", growth.assign(analysed_at=analysed_at))

    return {
        "run_id": run_id,
        "out_dir": out_dir,
        "measurements": df,
        "growth": growth,
        "summary": summary,
        "daily": daily,
        "report": report,
        "tables": tables,
        "figures": figures,
        "skipped": skipped,
        "metrics": metrics,
        "area_per_frond": per_frond,
        "calibration_day": cal_day,
        "quality_warnings": qc,
    }


def design_keys(cfg, extra=None):
    """(group, replicate, day) keys of the planned design plus any extra keys (e.g. from photos)."""
    d = cfg.get("design", {})
    reps = range(1, int(d.get("replicates_per_group", 5)) + 1)
    days = range(int(d.get("first_day", 0)), int(d.get("last_day", 7)) + 1)
    keys = {(g, r, day) for g in cfg["groups"] for r in reps for day in days}
    if extra is not None:
        keys |= {(str(g), int(r), int(day)) for g, r, day in extra}
    return sorted(keys, key=lambda k: (int(k[0][1:]) if k[0][1:].isdigit() else 0, k[0], k[1], k[2]))


_SHEETS = {
    "manual": ("manual_frond_counts.csv", ["frond_count_manual"]),
    "log": ("daily_log.csv", ["date", "temperature_C", "pH", "light_lux", "notes"]),
}
ENDPOINT_COLUMNS = ["root_length_mm", "fresh_weight_mg", "notes"]


def ensure_endpoints(folder, cfg, extra=None):
    """Sheet for the end-of-test measurements of each dish (root length, fresh weight)."""
    path = Path(folder) / "endpoints.csv"
    keys = sorted({(g, r) for g, r, _ in design_keys(cfg, extra)},
                  key=lambda k: (int(k[0][1:]) if k[0][1:].isdigit() else 0, k[0], k[1]))
    if path.exists():
        sheet = pd.read_csv(path, dtype=str)
        have = set()
        if {"group", "replicate"} <= set(sheet.columns):
            r = pd.to_numeric(sheet["replicate"], errors="coerce")
            have = {(g.strip().upper(), int(a)) for g, a in zip(sheet["group"].astype(str), r) if pd.notna(a)}
        new = [k for k in keys if k not in have]
        if not new:
            return path
        sheet = pd.concat([sheet, pd.DataFrame([{"group": g, "replicate": r} for g, r in new]).astype(str)],
                          ignore_index=True)
    else:
        sheet = pd.DataFrame([{"group": g, "replicate": r, **{c: "" for c in ENDPOINT_COLUMNS}} for g, r in keys])
    sheet.to_csv(path, index=False)
    return path


def _numbers(series):
    return pd.to_numeric(series.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def read_endpoints(path):
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["group", "replicate"])
    e = pd.read_csv(path, dtype=str)
    if not {"group", "replicate"} <= set(e.columns):
        return pd.DataFrame(columns=["group", "replicate"])
    out = pd.DataFrame({"group": e["group"].astype(str).str.strip().str.upper(), "replicate": _numbers(e["replicate"])})
    for col in ["root_length_mm", "fresh_weight_mg"]:
        if col in e.columns:
            out[col] = _numbers(e[col])
    out = out.dropna(subset=["replicate"])
    out["replicate"] = out["replicate"].astype(int)
    return out.drop_duplicates(["group", "replicate"], keep="last")


def dish_water_chemistry(path):
    """Per-dish pH at the start and end of the test from the daily log."""
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=["group", "replicate"]), pd.DataFrame()
    log = pd.read_csv(path, dtype=str)
    if not {"group", "replicate", "day"} <= set(log.columns):
        return pd.DataFrame(columns=["group", "replicate"]), pd.DataFrame()
    d = pd.DataFrame({"group": log["group"].astype(str).str.strip().str.upper(),
                      "replicate": _numbers(log["replicate"]), "day": _numbers(log["day"])})
    d["pH"] = _numbers(log["pH"]) if "pH" in log.columns else np.nan
    d = d.dropna(subset=["replicate", "day"])
    d["replicate"] = d["replicate"].astype(int)
    rows = []
    for (g, r), sub in d.sort_values("day").groupby(["group", "replicate"]):
        row = {"group": g, "replicate": r}
        v = sub.dropna(subset=["pH"])
        if len(v):
            row["pH_end"] = v["pH"].iloc[-1]
            row["pH_change"] = v["pH"].iloc[-1] - v["pH"].iloc[0] if len(v) > 1 else np.nan
        rows.append(row)
    per_dish = pd.DataFrame(rows)
    daily = d.groupby(["group", "day"])[["pH"]].agg(["mean", "std", "count"])
    daily.columns = [f"{a}_{b}" for a, b in daily.columns]
    return per_dish, daily.reset_index()


_SAVE_LOCK = threading.Lock()


def _key_frame(df, keys):
    out = df.copy()
    for k in keys:
        if k == "group":
            out[k] = out[k].astype(str).str.strip().str.upper()
        else:
            out[k] = pd.to_numeric(out[k], errors="coerce").astype("Int64")
    return out.dropna(subset=keys).drop_duplicates(keys, keep="last").set_index(keys)


def _same(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def merge_save(path, keys, before, after):
    """Write only the cells changed between `before` and `after` into the CSV at `path`.

    The file is read again just before writing, so changes saved by other users in the
    meantime (other rows or other columns) are kept.
    """
    with _SAVE_LOCK:
        path = Path(path)
        current = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=list(after.columns))
        cur, b, a = _key_frame(current, keys), _key_frame(before, keys), _key_frame(after, keys)
        cur = cur.astype(object)
        for idx, row in a.iterrows():
            for col, val in row.items():
                old = b.at[idx, col] if idx in b.index and col in b.columns else np.nan
                if not _same(val, old):
                    cur.loc[idx, col] = val
        cur.reset_index().to_csv(path, index=False)


EXTRA_RESPONSES = [
    ("root_length_mm", "root length at the end of the test, mm"),
    ("fresh_weight_mg", "fresh weight at the end of the test, mg"),
    ("pH_end", "pH on the last day"),
    ("pH_change", "pH change from the first to the last day"),
]


def ensure_sheet(folder, kind, cfg, extra=None):
    """Create the manual-count or daily-log sheet, or add rows for missing keys.

    Values already entered are never changed. Returns the path of the sheet.
    """
    name, value_cols = _SHEETS[kind]
    path = Path(folder) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = design_keys(cfg, extra)
    if path.exists():
        sheet = pd.read_csv(path, dtype=str)
        have = set()
        if {"group", "replicate", "day"} <= set(sheet.columns):
            r = pd.to_numeric(sheet["replicate"], errors="coerce")
            dd = pd.to_numeric(sheet["day"], errors="coerce")
            have = {(g.strip().upper(), int(a), int(b)) for g, a, b in zip(sheet["group"].astype(str), r, dd)
                    if pd.notna(a) and pd.notna(b)}
        new = [k for k in keys if k not in have]
        missing_cols = [c for c in value_cols if c not in sheet.columns]
        if not new and not missing_cols:
            return path
        for col in missing_cols:
            sheet[col] = ""
        if new:
            add = pd.DataFrame([{"group": g, "replicate": r, "day": d, **{c: "" for c in value_cols}}
                                for g, r, d in new])
            sheet = pd.concat([sheet, add.astype(str)], ignore_index=True)
    else:
        sheet = pd.DataFrame([{"group": g, "replicate": r, "day": d, **{c: "" for c in value_cols}} for g, r, d in keys])
        if kind == "log":
            sheet = sheet[["date", "day", "group", "replicate"] + [c for c in value_cols if c != "date"]]
    sheet.to_csv(path, index=False)
    return path


def write_templates(folder, cfg, n_reps=None, days=None):
    """Blank sheets for manual frond counts and the daily environment log.

    Existing sheets are kept; only rows missing from the planned design are added.
    """
    cfg = dict(cfg)
    design = dict(cfg.get("design", {}))
    if n_reps is not None:
        design["replicates_per_group"] = n_reps
    if days is not None:
        days = list(days)
        design["first_day"], design["last_day"] = min(days), max(days)
    cfg["design"] = design
    return ensure_sheet(folder, "manual", cfg), ensure_sheet(folder, "log", cfg)
