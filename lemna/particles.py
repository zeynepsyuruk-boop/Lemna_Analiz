"""Particle size from microscope photos (plastic particles on a plain background)."""

import cv2
import numpy as np
import pandas as pd


def _max_feret(contour):
    """Largest distance between two points of the convex hull (px)."""
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(float)
    if len(hull) < 2:
        return 0.0
    d = np.sqrt(((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1))
    return float(d.max())


def measure_particles(image, um_per_px, dark_particles=True, min_area_px=20, exclude_border=True):
    """Detect particles and measure their size.

    Returns (table with one row per particle, overlay image). Sizes are in micrometres:
    max_length_um is the largest Feret diameter, equivalent_diameter_um the diameter of
    a circle with the same area.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mode = cv2.THRESH_BINARY_INV if dark_particles else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blur, 0, 255, mode + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    rows = []
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if exclude_border and (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1):
            continue
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(cnt)
        rows.append({
            "x_px": round(cx, 1), "y_px": round(cy, 1),
            "area_um2": area * um_per_px ** 2,
            "equivalent_diameter_um": 2 * np.sqrt(area / np.pi) * um_per_px,
            "max_length_um": _max_feret(cnt) * um_per_px,
            "min_width_um": min(rw, rh) * um_per_px,
        })
        cv2.drawContours(overlay, [cnt], -1, (0, 0, 255), 2)
        cv2.putText(overlay, str(len(rows)), (int(cx) + 4, int(cy) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1, cv2.LINE_AA)
    return pd.DataFrame(rows), overlay


def summarise(table, low=200.0, high=500.0):
    """Count, mean, SD, median and the share of particles whose largest dimension is in [low, high] µm."""
    if table.empty:
        return {"n": 0}
    L = table["max_length_um"]
    return {
        "n": int(len(L)),
        "mean_max_length_um": float(L.mean()),
        "sd_max_length_um": float(L.std(ddof=1)) if len(L) > 1 else float("nan"),
        "median_max_length_um": float(L.median()),
        "mean_equivalent_diameter_um": float(table["equivalent_diameter_um"].mean()),
        "pct_in_range": float(((L >= low) & (L <= high)).mean() * 100),
        "range_um": f"{low:g}-{high:g}",
    }
