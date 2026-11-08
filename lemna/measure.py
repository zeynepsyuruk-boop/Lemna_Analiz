"""Batch analysis of a photo folder into a measurement table."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .imaging import analyse_image, read_image
from .naming import list_images, parse_name


def capture_time(path):
    """Photo date from EXIF (DateTimeOriginal), otherwise the file modification time."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            exif = im.getexif()
            value = exif.get_ifd(0x8769).get(36867) or exif.get(306)
        if value:
            return datetime.strptime(str(value).strip(), "%Y:%m:%d %H:%M:%S").isoformat(sep=" "), "exif"
    except Exception:
        pass
    return datetime.fromtimestamp(Path(path).stat().st_mtime).isoformat(sep=" ", timespec="seconds"), "file_time"


def group_info(cfg, group):
    info = cfg["groups"].get(group, {})
    plastic, detergent = info.get("plastic", "unknown"), info.get("detergent", "unknown")
    return {
        "group_label": info.get("label", group),
        "plastic": plastic,
        "detergent": detergent,
        # leachate controls and other extra groups stay out of the plastic x detergent design
        "factorial": bool(not info.get("leachate", False) and plastic in ("none", "PLA", "PETG")
                          and detergent in ("no", "yes")),
    }


def series_of(cfg, replicate):
    """Independent experimental series of a dish (replicates 1-4 -> 1, 5-8 -> 2 ... when 4 per series)."""
    per = int(cfg.get("design", {}).get("replicates_per_series") or 0)
    return (int(replicate) - 1) // per + 1 if per > 0 else 1


def _analyse_one(path, cfg, overlay_dir):
    image = read_image(path, cfg["work_max_side_px"])
    result, overlay, _ = analyse_image(image, cfg)
    if overlay_dir is not None:
        ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            buf.tofile(str(overlay_dir / f"{path.stem}_qc.jpg"))
    return result, capture_time(path)


def analyse_folder(photo_dir, cfg, out_dir, save_overlays=True, progress=None, workers=None):
    """Analyse every correctly named photo in photo_dir, several photos in parallel.

    Returns (measurements DataFrame, list of skipped file names).
    """
    out_dir = Path(out_dir)
    overlay_dir = out_dir / "qc_overlays" if save_overlays else None
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    jobs, skipped = [], []
    for path in list_images(photo_dir):
        parsed = parse_name(path)
        if parsed is None:
            skipped.append(path.name)
        else:
            jobs.append((path, parsed))

    keys = pd.Series([f"{g}_{r}_day{d}" for _, (g, r, d) in jobs])
    dup = keys.duplicated(keep=False)
    if dup.any():
        names = ", ".join(p.name for (p, _), d in zip(jobs, dup) if d)
        raise ValueError(f"More than one photo for the same dish and day: {names}")

    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_analyse_one, path, cfg, overlay_dir): (path, parsed) for path, parsed in jobs}
        for done, fut in enumerate(as_completed(futures), start=1):
            path, (group, rep, day) = futures[fut]
            try:
                result, (taken, source) = fut.result()
            except Exception as exc:  # unreadable or corrupt file
                skipped.append(f"{path.name} ({exc})")
            else:
                row = {"file": path.name, "group": group, "replicate": rep, "dish_id": f"{group}_{rep}",
                       "series": series_of(cfg, rep), "day": day, "photo_datetime": taken,
                       "photo_datetime_source": source}
                row.update(group_info(cfg, group))
                row.update(result)
                rows.append(row)
            if progress is not None:
                progress(done, len(jobs), path.name)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["group", "replicate", "day"]).reset_index(drop=True)
    return df, skipped


def read_manual_counts(manual_csv):
    """Manual count sheet with clean keys and numbers.

    Group names are upper-cased, decimal commas are accepted, rows without a valid
    group/replicate/day are dropped and, for repeated keys, the last row is kept.
    """
    cols = ["group", "replicate", "day", "frond_count_manual"]
    if manual_csv is None or not Path(manual_csv).exists():
        return pd.DataFrame(columns=cols)
    manual = pd.read_csv(manual_csv, dtype=str)
    manual.columns = [c.strip() for c in manual.columns]
    if not {"group", "replicate", "day"} <= set(manual.columns):
        return pd.DataFrame(columns=cols)
    if "frond_count_manual" not in manual.columns:
        manual["frond_count_manual"] = np.nan

    def number(col):
        return pd.to_numeric(manual[col].astype(str).str.strip().str.replace(",", ".", regex=False),
                             errors="coerce")

    out = pd.DataFrame({
        "group": manual["group"].astype(str).str.strip().str.upper(),
        "replicate": number("replicate"),
        "day": number("day"),
        "frond_count_manual": number("frond_count_manual"),
    })
    out = out.dropna(subset=["replicate", "day"])
    out = out[out["group"].str.match(r"^G\d+$")]
    out["replicate"] = out["replicate"].astype(int)
    out["day"] = out["day"].astype(int)
    out.loc[out["frond_count_manual"] < 0, "frond_count_manual"] = np.nan
    return out.drop_duplicates(["group", "replicate", "day"], keep="last").reset_index(drop=True)


def merge_manual_counts(df, manual_csv):
    """Add manual frond counts (columns group, replicate, day, frond_count_manual)."""
    manual = read_manual_counts(manual_csv).dropna(subset=["frond_count_manual"])
    if manual.empty:
        return df
    df = df.drop(columns=["frond_count_manual"], errors="ignore")
    return df.merge(manual, on=["group", "replicate", "day"], how="left", validate="many_to_one")


def calibrate_area_count(df, cfg):
    """Frond count estimated from plant area divided by the mean area of one frond.

    The area per frond is the median of plant area / manual count over the photos of
    the calibration day (the first day by default), or reference_frond_area_mm2 from
    the config when no manual counts are available.
    Returns the DataFrame, the area per frond used (None if not available) and the
    calibration day (None unless manual counts were used).
    """
    area_col = area_column(df)
    per_frond, used_day = None, None
    if "frond_count_manual" in df.columns:
        cal_day = cfg["frond_count"].get("calibration_day")
        if cal_day is None:
            cal_day = int(df["day"].min())
        d = df[(df["day"] == cal_day) & (df["frond_count_manual"] > 0) & (df[area_col] > 0)]
        if len(d) >= 3:
            per_frond = float(np.median(d[area_col] / d["frond_count_manual"]))
            used_day = int(cal_day)
    ref = cfg["frond_count"].get("reference_frond_area_mm2")
    if per_frond is None and ref and area_col == "plant_area_mm2":
        per_frond = float(ref)
    if per_frond:
        df["frond_count_from_area"] = df[area_col] / per_frond
    return df, per_frond, used_day


def dishes_without_manual_endpoints(df):
    """Dishes lacking a manual count on their first or last photographed day."""
    if "frond_count_manual" not in df.columns:
        return sorted(df["dish_id"].unique())
    missing = []
    for dish, sub in df.sort_values("day").groupby("dish_id"):
        if pd.isna(sub["frond_count_manual"].iloc[0]) or pd.isna(sub["frond_count_manual"].iloc[-1]):
            missing.append(dish)
    return missing


def frond_column(df):
    """Manual counts are used for growth rates when every dish has a manual count
    on its first and last photographed day."""
    if "frond_count_manual" in df.columns and not dishes_without_manual_endpoints(df):
        return "frond_count_manual"
    return "frond_count_auto"


def area_column(df):
    if "plant_area_mm2" in df.columns and np.isfinite(df["plant_area_mm2"]).all():
        return "plant_area_mm2"
    return "plant_area_px"
