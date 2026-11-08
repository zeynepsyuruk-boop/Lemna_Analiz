"""Figures: growth curves, RGR per group, interaction plot, colour change and agreement."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"G1": "#4d4d4d", "G2": "#1f77b4", "G3": "#2ca02c", "G4": "#d62728", "G5": "#98df8a", "G6": "#ff9896"}

AXIS_LABELS = {
    "plant_area_mm2": "Plant area (mm²)",
    "plant_area_px": "Plant area (px)",
    "green_pct_of_roi": "Green area (% of dish)",
    "frond_count_auto": "Frond count (automatic)",
    "frond_count_manual": "Frond count (manual)",
    "frond_count_from_area": "Frond count (area-based)",
    "brown_pct_of_plant": "Browning (% of plant area)",
    "plant_pct_of_roi": "Plant cover (% of dish)",
    "mean_hue": "Mean hue (H)",
    "yellow_pct_of_plant": "Yellowing (% of plant area)",
    "exg_index": "Excess green index",
}


def _color(group, i):
    return COLORS.get(group, plt.cm.tab10(i % 10))


def growth_curves(df, column, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (group, sub) in enumerate(df.groupby("group")):
        s = sub.groupby("day")[column].agg(["mean", "std"]).reset_index()
        label = f"{group} – {sub['group_label'].iloc[0]}"
        ax.errorbar(s["day"], s["mean"], yerr=s["std"], marker="o", ms=4, capsize=3,
                    color=_color(group, i), label=label)
    ax.set_xlabel("Day")
    ax.set_ylabel(AXIS_LABELS.get(column, column))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def rgr_boxplot(growth, response, path, ylabel):
    groups = sorted(growth["group"].unique())
    data = [growth.loc[growth["group"] == g, response].dropna().to_numpy() for g in groups]
    labels = [f"{g}\n{growth.loc[growth['group'] == g, 'group_label'].iloc[0]}" for g in groups]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, widths=0.5, showfliers=False)
    rng = np.random.default_rng(1)
    for i, (g, y) in enumerate(zip(groups, data), start=1):
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(y)), y, color=_color(g, i), zorder=3, s=22)
    ax.set_xticks(range(1, len(groups) + 1), labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def interaction_plot(growth, response, path, ylabel):
    order = ["none", "PLA", "PETG"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for det, style in [("no", "-o"), ("yes", "--s")]:
        sub = growth[growth["detergent"] == det]
        s = sub.groupby("plastic")[response].agg(["mean", "std", "count"]).reindex(order)
        se = s["std"] / np.sqrt(s["count"])
        ax.errorbar(range(len(order)), s["mean"], yerr=se, fmt=style, capsize=4,
                    label=f"detergent: {det}")
    ax.set_xticks(range(len(order)), ["No plastic", "PLA", "PETG"])
    ax.set_ylabel(ylabel)
    ax.set_title("Interaction plot (mean ± SE)", fontsize=10)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def bland_altman(df, path, auto_col="frond_count_auto", manual_col="frond_count_manual"):
    if auto_col not in df.columns or manual_col not in df.columns:
        return False
    d = df[[auto_col, manual_col]].dropna()
    if len(d) < 3:
        return False
    a, m = d[auto_col].to_numpy(float), d[manual_col].to_numpy(float)
    mean, diff = (a + m) / 2, a - m
    bias, sd = diff.mean(), diff.std(ddof=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    lim = [0, max(a.max(), m.max()) * 1.05]
    axes[0].scatter(m, a, s=18, alpha=0.7)
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("Manual frond count")
    axes[0].set_ylabel("Automatic frond count")
    axes[0].grid(alpha=0.3)
    axes[1].scatter(mean, diff, s=18, alpha=0.7)
    for y, ls in [(bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")]:
        axes[1].axhline(y, color="k", ls=ls, lw=1)
    axes[1].set_xlabel("Mean of automatic and manual count")
    axes[1].set_ylabel("Automatic − manual")
    axes[1].set_title("Bland–Altman", fontsize=10)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def make_all(df, growth, metrics, out_dir):
    out = Path(out_dir) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for col in list(dict.fromkeys(metrics + ["green_pct_of_roi", "yellow_pct_of_plant", "exg_index"])):
        if col in df.columns and df[col].notna().any():
            p = out / f"curve_{col}.png"
            growth_curves(df, col, p)
            made.append(p)
    for m in metrics:
        r = f"rgr_{m}"
        if r in growth.columns and growth[r].notna().sum() >= 2:
            label = f"RGR from {AXIS_LABELS.get(m, m).lower()} (1/day)"
            p = out / f"rgr_box_{m}.png"
            rgr_boxplot(growth, r, p, label)
            made.append(p)
            p = out / f"interaction_{m}.png"
            interaction_plot(growth, r, p, label)
            made.append(p)
    if "frond_count_manual" in df.columns:
        p = out / "agreement_manual_vs_auto.png"
        if bland_altman(df, p):
            made.append(p)
    return made


CHART_KINDS = {
    "curve": "Growth curve (group mean ± SD)",
    "curve_dishes": "Growth curve (each dish)",
    "day_box": "Box plot on a selected day",
    "rgr_box": "RGR per group",
    "inhibition": "Growth inhibition vs control",
    "interaction": "Interaction plot (plastic x detergent)",
    "heatmap": "Dish x day heat map",
    "agreement": "Manual vs automatic count",
}


def _labels(df):
    return {g: f"{g} – {lab}" for g, lab in df.groupby("group")["group_label"].first().items()}


def make_chart(kind, df, growth, metric, path, groups=None, day_range=None, day=None, logy=False,
               exclude_day=None):
    """Draw one figure chosen in the interface and save it as PNG.

    exclude_day removes one day (the calibration day of the area-based count) from the
    agreement plot. Returns the path, or None when there is not enough data.
    """
    if groups:
        df = df[df["group"].isin(groups)]
        growth = growth[growth["group"].isin(groups)] if growth is not None else growth
    if day_range:
        df = df[(df["day"] >= day_range[0]) & (df["day"] <= day_range[1])]
    labels = _labels(df)
    ylabel = AXIS_LABELS.get(metric, metric)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if kind == "curve":
        if logy:
            _log_curves(df, metric, path)
        else:
            growth_curves(df, metric, path)
        return path
    if kind == "rgr_box":
        rgr_boxplot(growth, f"rgr_{metric}", path, f"RGR from {ylabel.lower()} (1/day)")
        return path
    if kind == "interaction":
        interaction_plot(growth, f"rgr_{metric}", path, f"RGR from {ylabel.lower()} (1/day)")
        return path
    if kind == "agreement":
        if exclude_day is not None:
            df = df[df["day"] != exclude_day]
        return path if bland_altman(df, path, auto_col=metric) else None

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if kind == "curve_dishes":
        for i, (dish, sub) in enumerate(df.sort_values("day").groupby("dish_id")):
            g = sub["group"].iloc[0]
            ax.plot(sub["day"], sub[metric], marker="o", ms=3, lw=1, color=_color(g, i), alpha=0.8)
        for i, g in enumerate(sorted(df["group"].unique())):
            ax.plot([], [], color=_color(g, i), label=labels[g])
        ax.set_xlabel("Day")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, frameon=False)
        if logy:
            ax.set_yscale("log")
    elif kind == "day_box":
        sel = df[df["day"] == day]
        groups_ = sorted(sel["group"].unique())
        data = [sel.loc[sel["group"] == g, metric].dropna().to_numpy() for g in groups_]
        ax.boxplot(data, widths=0.5, showfliers=False)
        for i, (g, y) in enumerate(zip(groups_, data), start=1):
            ax.scatter(np.full(len(y), i), y, color=_color(g, i), zorder=3, s=22)
        ax.set_xticks(range(1, len(groups_) + 1), [labels[g].replace(" – ", "\n") for g in groups_], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Day {day}", fontsize=10)
    elif kind == "inhibition":
        col = f"inhibition_pct_{metric}"
        s = growth.groupby("group")[col].agg(["mean", "std"])
        ax.bar(range(len(s)), s["mean"], yerr=s["std"], capsize=4,
               color=[_color(g, i) for i, g in enumerate(s.index)])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(s)), [labels.get(g, g).replace(" – ", "\n") for g in s.index], fontsize=8)
        ax.set_ylabel(f"Growth inhibition of {ylabel.lower()} (%)")
    elif kind == "heatmap":
        pv = df.pivot_table(index="dish_id", columns="day", values=metric)
        order = sorted(pv.index, key=lambda s: (int(s.split("_")[0][1:]), int(s.split("_")[1])))
        pv = pv.reindex(order)
        im = ax.imshow(pv.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(pv.shape[1]), pv.columns)
        ax.set_yticks(range(pv.shape[0]), pv.index, fontsize=6)
        ax.set_xlabel("Day")
        fig.colorbar(im, ax=ax, label=ylabel)
        fig.set_size_inches(7.5, max(4.5, 0.18 * pv.shape[0]))
    if kind != "heatmap":
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _log_curves(df, metric, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (group, sub) in enumerate(df.groupby("group")):
        s = sub.groupby("day")[metric].agg(["mean", "std"]).reset_index()
        ax.errorbar(s["day"], s["mean"], yerr=s["std"], marker="o", ms=4, capsize=3,
                    color=_color(group, i), label=f"{group} – {sub['group_label'].iloc[0]}")
    ax.set_yscale("log")
    ax.set_xlabel("Day")
    ax.set_ylabel(AXIS_LABELS.get(metric, metric))
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
