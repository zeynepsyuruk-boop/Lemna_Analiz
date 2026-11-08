"""Loading and saving of the analysis settings (config.json)."""

import copy
import json
from pathlib import Path

DEFAULT_CONFIG = {
    "groups": {
        "G1": {"label": "Control", "plastic": "none", "detergent": "no"},
        "G2": {"label": "Detergent", "plastic": "none", "detergent": "yes"},
        "G3": {"label": "PLA", "plastic": "PLA", "detergent": "no"},
        "G4": {"label": "PETG", "plastic": "PETG", "detergent": "no"},
        "G5": {"label": "PLA + detergent", "plastic": "PLA", "detergent": "yes"},
        "G6": {"label": "PETG + detergent", "plastic": "PETG", "detergent": "yes"},
        # leachate controls: plants grow in filtered water in which particles were kept, no particle contact
        "G7": {"label": "PLA leachate + detergent", "plastic": "PLA", "detergent": "yes", "leachate": True},
        "G8": {"label": "PETG leachate + detergent", "plastic": "PETG", "detergent": "yes", "leachate": True},
    },
    "control_group": "G1",
    # planned design, used for the blank manual-count and daily-log sheets
    # 12 dishes per group run as 3 independent series of 4 (replicates 1-4, 5-8, 9-12)
    "design": {"replicates_per_group": 12, "replicates_per_series": 4, "first_day": 0, "last_day": 7},
    # (group1, group2) pairs compared with Holm correction; difference = group2 - group1
    "planned_comparisons": [["G2", "G7"], ["G5", "G7"], ["G2", "G8"], ["G6", "G8"]],
    "work_max_side_px": 1600,
    "dish": {
        # auto: Hough circle detection, fixed: circle given as image fractions,
        # full: whole image is used as the region of interest
        "mode": "auto",
        "inner_diameter_mm": 90.0,
        "margin_fraction": 0.94,
        "fixed_cx": 0.5,
        "fixed_cy": 0.5,
        "fixed_r": 0.45,
    },
    # none or grayworld
    "white_balance": "none",
    # OpenCV HSV ranges: H 0-179, S 0-255, V 0-255
    "hsv": {
        "green_low": [33, 40, 35],
        "green_high": [90, 255, 255],
        "yellow_low": [18, 50, 70],
        "yellow_high": [32, 255, 255],
        "brown_low": [5, 60, 30],
        "brown_high": [17, 255, 200],
    },
    "morphology": {
        "open_kernel_px": 3,
        "close_kernel_px": 5,
        "min_object_area_px": 40,
        # reclassify edge pixels with a 50 % plant/background mixing rule
        "refine_edges": True,
    },
    "frond_count": {
        # seed window (x typical frond half-width): smaller splits more, larger merges more
        "seed_window": 1.2,
        "peak_ratio": 0.5,
        "min_frond_area_px": 60,
        "reference_frond_area_mm2": None,
        # day whose manual counts calibrate the area per frond (None = first day)
        "calibration_day": None,
    },
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != "groups":
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path=None):
    """Return the default settings updated with the values in a JSON file."""
    if path is None or not Path(path).exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as fh:
        user = json.load(fh)
    return _merge(DEFAULT_CONFIG, user)


def with_workspace_config(cfg, path):
    """Settings of a workspace (e.g. the detergent pre-test) override the general settings."""
    if path is None or not Path(path).exists():
        return cfg
    with open(path, encoding="utf-8") as fh:
        return _merge(cfg, json.load(fh))


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
