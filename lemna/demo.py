"""Synthetic dish photos for testing the software before real photos are available.

Each dish is simulated as a time series: colonies of fronds bud, split and drift on
the medium, fronds of stressed groups stay smaller and turn yellow or brown starting
with the oldest ones, and plastic fragments float in the plastic groups. Photos vary
slightly in brightness, colour temperature and camera position. Manual counts carry
a small counting error and the true values are written to demo_truth.csv.

The treatment effects are assumed values chosen to resemble a plausible outcome
(detergent toxic, plastics alone weakly inhibitory, PLA with detergent synergistic,
PETG with detergent close to additive, the PLA effect in detergent mainly caused by
substances released into the water and the PETG effect by the particles). Root
length and pH follow the same scenario. They are not experimental
results.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Assumed true relative growth rates (1/day)
DEMO_RGR = {"G1": 0.300, "G2": 0.225, "G3": 0.276, "G4": 0.267, "G5": 0.165, "G6": 0.193,
            "G7": 0.178, "G8": 0.222}
# Fraction of fronds that are yellow (chlorotic) and brown (necrotic) on the last day
DEMO_YELLOW = {"G1": 0.01, "G2": 0.12, "G3": 0.03, "G4": 0.04, "G5": 0.24, "G6": 0.15, "G7": 0.20, "G8": 0.12}
DEMO_BROWN = {"G1": 0.0, "G2": 0.02, "G3": 0.0, "G4": 0.0, "G5": 0.06, "G6": 0.03, "G7": 0.05, "G8": 0.02}
# Mature frond size relative to the control (stress reduces frond size)
DEMO_SIZE = {"G1": 1.00, "G2": 0.93, "G3": 0.99, "G4": 0.98, "G5": 0.89, "G6": 0.92, "G7": 0.90, "G8": 0.93}
# Particles are visible only in the particle groups (the leachate groups are filtered)
PLASTIC = {"G3": "PLA", "G4": "PETG", "G5": "PLA", "G6": "PETG"}
# Mean root length on the last day (mm), dish-to-dish SD 1.0 mm
DEMO_ROOT = {"G1": 14.0, "G2": 10.5, "G3": 12.0, "G4": 12.5, "G5": 7.5, "G6": 9.5, "G7": 8.5, "G8": 10.3}
# Water chemistry: start value and daily change of pH
DEMO_PH = {"G1": (6.50, 0.02), "G2": (7.30, -0.01), "G3": (6.50, -0.01), "G4": (6.50, 0.01),
           "G5": (7.30, -0.06), "G6": (7.30, -0.01), "G7": (7.00, -0.03), "G8": (7.30, -0.01)}
SERIES_SD_RGR = 0.03   # relative difference between independent series

SIZE = 1200            # image size (px)
DISH_R = 520           # dish radius in the canonical image (px), inner diameter 90 mm
FROND_A = 19.0         # half-length of a mature control frond (px), about 1.65 mm
DISH_SD_RGR = 0.05     # relative variation of the growth rate between dishes
DAILY_SD = 0.03        # day-to-day variation of ln(frond number)
START_FRONDS = 12      # four colonies of three fronds, as in OECD TG 221

GREEN = np.array([45, 150, 55])
YELLOW = np.array([60, 195, 215])
BROWN = np.array([45, 85, 140])


class Dish:
    """State of one dish: colonies with fronds, plastic fragments."""

    def __init__(self, group, rng):
        self.group, self.rng = group, rng
        self.size = DEMO_SIZE[group]
        self.colonies = []
        for _ in range(START_FRONDS // 3):
            col = self._new_colony(self._free_position(60))
            # a growing population has many young fronds: one mother and two daughters of different age
            for age in (rng.uniform(2.5, 5), rng.uniform(0.5, 1.5), rng.uniform(0, 0.7)):
                self._add_frond(col, age=age)
        self.fragments = self._fragments() if group in PLASTIC else []

    # -------------------------------------------------- structure
    def _free_position(self, clearance):
        c = SIZE / 2
        for _ in range(400):
            r = DISH_R * 0.78 * np.sqrt(self.rng.uniform())
            ang = self.rng.uniform(0, 2 * np.pi)
            p = np.array([c + r * np.cos(ang), c + r * np.sin(ang)])
            if all(np.hypot(*(p - col["pos"])) > clearance for col in self.colonies):
                return p
        return np.array([c, c]) + self.rng.normal(0, DISH_R * 0.4, 2)

    def _new_colony(self, pos):
        col = {"pos": np.array(pos, float), "angle": self.rng.uniform(0, 360), "fronds": []}
        self.colonies.append(col)
        return col

    def _add_frond(self, col, age=0.0):
        col["fronds"].append({"age": age, "a": FROND_A * self.rng.normal(1, 0.06), "state": "green"})

    def n_fronds(self):
        return sum(len(c["fronds"]) for c in self.colonies)

    def grow_to(self, target):
        """Add daughter fronds until the dish has `target` fronds."""
        while self.n_fronds() < target:
            col = self.colonies[self.rng.integers(len(self.colonies))]
            if len(col["fronds"]) >= 4:
                # a daughter colony detaches next to its parent
                ang = self.rng.uniform(0, 2 * np.pi)
                pos = col["pos"] + 3.8 * FROND_A * np.array([np.cos(ang), np.sin(ang)])
                new = self._new_colony(self._inside(pos))
                moved = col["fronds"].pop()
                new["fronds"].append(moved)
                self._add_frond(new)
            else:
                self._add_frond(col)

    def _inside(self, pos):
        c = np.array([SIZE / 2, SIZE / 2])
        v = pos - c
        r = np.hypot(*v)
        limit = DISH_R * 0.82
        return c + v * (limit / r) if r > limit else pos

    def next_day(self):
        for col in self.colonies:
            col["pos"] = self._inside(col["pos"] + self.rng.normal(0, 5, 2))
            col["angle"] += self.rng.normal(0, 6)
            for f in col["fronds"]:
                f["age"] += 1
        self._separate()
        for frag in self.fragments:
            frag["pos"] = self._inside(frag["pos"] + self.rng.normal(0, 3, 2))

    def _separate(self):
        """Push overlapping colonies apart a little, as floating colonies do."""
        for _ in range(3):
            for i, a in enumerate(self.colonies):
                for b in self.colonies[i + 1:]:
                    d = b["pos"] - a["pos"]
                    dist = np.hypot(*d) + 1e-6
                    need = 3.4 * FROND_A
                    if dist < need:
                        shift = d / dist * (need - dist) / 2
                        a["pos"] = self._inside(a["pos"] - shift)
                        b["pos"] = self._inside(b["pos"] + shift)

    def colour(self, day, last_day):
        """Turn the oldest fronds yellow and then brown up to the group's target fractions."""
        t = (day / last_day) ** 1.5 if last_day else 0
        fronds = sorted((f for c in self.colonies for f in c["fronds"]), key=lambda f: -f["age"])
        n = len(fronds)
        n_brown = int(round(DEMO_BROWN[self.group] * t * n))
        n_yellow = int(round(DEMO_YELLOW[self.group] * t * n))
        for i, f in enumerate(fronds):
            if i < n_brown:
                f["state"] = "brown"
            elif i < n_brown + n_yellow and f["state"] == "green":
                f["state"] = "yellow"

    def _fragments(self):
        kind = PLASTIC[self.group]
        frags = []
        for _ in range(int(self.rng.integers(14, 22))):
            n = int(self.rng.integers(4, 7))
            r = self.rng.uniform(4, 11)
            angles = np.sort(self.rng.uniform(0, 2 * np.pi, n))
            shape = np.stack([np.cos(angles), np.sin(angles)], 1) * r * self.rng.uniform(0.5, 1.0, (n, 1))
            colour = (np.array([226, 232, 233]) if kind == "PLA" else np.array([214, 206, 196]))
            frags.append({"pos": self._free_position(0), "shape": shape,
                          "colour": colour + self.rng.integers(-6, 6, 3)})
        return frags

    # -------------------------------------------------- drawing
    def render(self):
        """Canonical image and plant mask (before the camera transform)."""
        c = SIZE // 2
        img = np.full((SIZE, SIZE, 3), (236, 238, 240), np.float32)
        cv2.circle(img, (c, c), DISH_R + 14, (205, 208, 210), -1, cv2.LINE_AA)
        cv2.circle(img, (c, c), DISH_R, (196, 200, 201), -1, cv2.LINE_AA)
        cv2.circle(img, (c, c), DISH_R + 7, (140, 142, 145), 5, cv2.LINE_AA)
        cv2.ellipse(img, (c, c), (DISH_R - 4, DISH_R - 4), 0, 200, 250, (225, 228, 230), 3, cv2.LINE_AA)
        mask = np.zeros((SIZE, SIZE), np.uint8)
        for frag in self.fragments:
            pts = (frag["pos"] + frag["shape"]).astype(np.int32)
            cv2.fillPoly(img, [pts], frag["colour"].tolist(), cv2.LINE_AA)
            cv2.polylines(img, [pts], True, (frag["colour"] * 0.85).tolist(), 1, cv2.LINE_AA)
        for col in self.colonies:
            k = len(col["fronds"])
            for j, f in enumerate(col["fronds"]):
                grow = min(1.0, 0.6 + 0.25 * f["age"])
                a = f["a"] * self.size * grow
                b = a * 0.68
                theta = col["angle"] + j * 360.0 / max(k, 1)
                off = 1.0 * a if k > 1 else 0.0
                cx = col["pos"][0] + off * np.cos(np.deg2rad(theta))
                cy = col["pos"][1] + off * np.sin(np.deg2rad(theta))
                base = {"green": GREEN, "yellow": YELLOW, "brown": BROWN}[f["state"]]
                shade = base * self.rng.normal(1, 0.04) + self.rng.normal(0, 4, 3)
                centre, axes = (int(round(cx)), int(round(cy))), (max(2, int(round(a))), max(2, int(round(b))))
                cv2.ellipse(img, centre, axes, theta, 0, 360, np.clip(shade, 0, 255).tolist(), -1, cv2.LINE_AA)
                cv2.ellipse(img, centre, (max(1, axes[0] // 2), max(1, axes[1] // 2)), theta, 0, 360,
                            np.clip(shade * 1.12, 0, 255).tolist(), -1, cv2.LINE_AA)
                cv2.ellipse(mask, centre, axes, theta, 0, 360, 255, -1)
        return img, mask


def _camera(img, mask, rng):
    """Small changes in camera position, light and colour temperature between photos."""
    scale = rng.normal(1.0, 0.012)
    angle = rng.normal(0, 1.5)
    shift = rng.normal(0, 12, 2)
    m = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), angle, scale)
    m[:, 2] += shift
    img = cv2.warpAffine(img, m, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(mask, m, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    r2 = ((xx - SIZE / 2) ** 2 + (yy - SIZE / 2) ** 2) / (SIZE / 2) ** 2
    light = rng.normal(1.0, 0.03) * (1 - 0.07 * r2)
    balance = 1 + rng.normal(0, 0.015, 3)
    img = img * light[..., None] * balance + rng.normal(0, 3, img.shape)
    img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (3, 3), 0)
    return img, mask, scale


def _manual_count(n, day, rng):
    """Manual count with a small human counting error that grows with the number of fronds."""
    if day == 0 or n <= 20:
        return n
    return int(max(1, round(n + rng.normal(0, 0.02 * n))))


def make_demo(out_dir, n_reps=12, days=range(0, 8), seed=2026, reps_per_series=4):
    out_dir = Path(out_dir)
    photo_dir = out_dir / "photos"
    photo_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    days = list(days)
    manual, truth, endpoints, log = [], [], [], []
    n_series = (n_reps - 1) // reps_per_series + 1
    series_factor = rng.normal(1.0, SERIES_SD_RGR, n_series)
    for g, rgr in DEMO_RGR.items():
        for rep in range(1, n_reps + 1):
            series = (rep - 1) // reps_per_series
            dish = Dish(g, rng)
            dish_rgr = rgr * series_factor[series] * rng.normal(1.0, DISH_SD_RGR)
            root = DEMO_ROOT[g] * series_factor[series] + rng.normal(0, 1.0)
            endpoints.append({"group": g, "replicate": rep, "root_length_mm": round(root, 1),
                              "fresh_weight_mg": "", "notes": ""})
            ph0, dph = DEMO_PH[g]
            for d in days:
                log.append({"date": "", "day": d, "group": g, "replicate": rep,
                            "temperature_C": round(24 + rng.normal(0, 0.4), 1),
                            "pH": round(ph0 + dph * d + rng.normal(0, 0.03), 2),
                            "light_lux": int(8000 + rng.normal(0, 150)), "notes": ""})
            log_n = np.log(START_FRONDS)
            for i, d in enumerate(days):
                if i > 0:
                    dish.next_day()
                    log_n += dish_rgr * (d - days[i - 1]) + rng.normal(0, DAILY_SD)
                    dish.grow_to(max(dish.n_fronds(), int(round(np.exp(log_n)))))
                dish.colour(d, days[-1])
                img, mask = dish.render()
                img, mask, scale = _camera(img, mask, rng)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                buf.tofile(str(photo_dir / f"{g}_{rep}_Gun{d}.jpg"))
                n = dish.n_fronds()
                mm_per_px = 90.0 / (2 * DISH_R * scale)
                manual.append({"group": g, "replicate": rep, "day": d,
                               "frond_count_manual": _manual_count(n, d, rng)})
                truth.append({"group": g, "replicate": rep, "day": d, "series": series + 1,
                              "frond_count_true": n,
                              "plant_area_mm2_true": round(np.count_nonzero(mask) * mm_per_px ** 2, 2),
                              "dish_rgr_true": round(dish_rgr, 5), "root_length_true": round(root, 2)})
    pd.DataFrame(manual).to_csv(out_dir / "manual_frond_counts.csv", index=False)
    pd.DataFrame(truth).to_csv(out_dir / "demo_truth.csv", index=False)
    pd.DataFrame(endpoints).to_csv(out_dir / "endpoints.csv", index=False)
    pd.DataFrame(log).to_csv(out_dir / "daily_log.csv", index=False)
    return photo_dir, out_dir / "manual_frond_counts.csv"
