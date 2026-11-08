"""Statistical tests: descriptive statistics, assumption checks, two-way ANOVA,
interaction contrasts, a mixed model for daily data and manual/automatic agreement.

The dish is the experimental unit. Photos of the same dish on different days are
not treated as independent replicates.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

ALPHA = 0.05
N_PERMUTATIONS = 9999
PERMUTATION_SEED = 12345


def factorial_part(growth):
    """Dishes that belong to the plastic x detergent design (leachate and other extra groups excluded)."""
    if "factorial" in growth.columns:
        growth = growth[growth["factorial"].astype(bool)]
    return growth[growth["plastic"].isin(["none", "PLA", "PETG"]) & growth["detergent"].isin(["no", "yes"])]


def has_series(d):
    return "series" in d.columns and d["series"].nunique() > 1


def _prepare(growth):
    d = growth.copy()
    d["plastic"] = pd.Categorical(d["plastic"], categories=["none", "PLA", "PETG"])
    d["detergent"] = pd.Categorical(d["detergent"], categories=["no", "yes"])
    return d


def design_cells(growth, response):
    """Number of usable dishes in each plastic x detergent cell."""
    d = factorial_part(growth).dropna(subset=[response])
    cells = d.groupby(["plastic", "detergent"]).size()
    full = pd.MultiIndex.from_product([["none", "PLA", "PETG"], ["no", "yes"]], names=["plastic", "detergent"])
    return cells.reindex(full, fill_value=0)


def is_full_factorial(growth, response, min_n=2):
    return bool((design_cells(growth, response) >= min_n).all())


def assumption_checks(growth, response):
    d = growth.dropna(subset=[response])
    out = {}
    # residuals from the group means (identical to the full factorial model when all cells exist)
    resid = d[response] - d.groupby("group")[response].transform("mean")
    if len(resid) >= 3:
        w, p = stats.shapiro(resid)
        out["shapiro_W"], out["shapiro_p"] = float(w), float(p)
    samples = [g[response].to_numpy() for _, g in d.groupby("group") if len(g) >= 2]
    if len(samples) >= 2:
        lev, p = stats.levene(*samples, center="median")
        out["levene_stat"], out["levene_p"] = float(lev), float(p)
    if len(samples) >= 2:
        kw, p = stats.kruskal(*samples)
        out["kruskal_H"], out["kruskal_p"] = float(kw), float(p)
    return out


def two_way_anova(growth, response):
    """Plastic (none/PLA/PETG) x detergent (no/yes), type II sums of squares.

    When the experiment was run in several independent series, the series is added
    as a block factor.
    """
    d = _prepare(factorial_part(growth).dropna(subset=[response]))
    block = " + C(series)" if has_series(d) else ""
    model = smf.ols(f"{response} ~ C(plastic) * C(detergent){block}", data=d).fit()
    table = anova_lm(model, typ=2)
    ss_res = table.loc["Residual", "sum_sq"]
    table["partial_eta_sq"] = table["sum_sq"] / (table["sum_sq"] + ss_res)
    table.loc["Residual", "partial_eta_sq"] = np.nan
    return table, model


def interaction_contrasts(model):
    """Deviation of each combined treatment from additivity.

    For PLA: (PLA+det) - PLA - det + control. Zero means the combined effect equals
    the sum of the single effects; the same contrast is the interaction coefficient
    of the treatment-coded model.
    """
    rows = []
    params, ci, pvals = model.params, model.conf_int(), model.pvalues
    for plastic in ["PLA", "PETG"]:
        name = f"C(plastic)[T.{plastic}]:C(detergent)[T.yes]"
        if name in params.index:
            rows.append({
                "contrast": f"{plastic} x detergent (deviation from additivity)",
                "estimate": params[name],
                "ci_low": ci.loc[name, 0],
                "ci_high": ci.loc[name, 1],
                "p_value": pvals[name],
            })
    return pd.DataFrame(rows)


def tukey(growth, response):
    """Tukey HSD between all groups (exact p values and 95 % confidence intervals)."""
    d = growth.dropna(subset=[response])
    counts = d.groupby("group").size()
    d = d[d["group"].isin(counts[counts >= 2].index)]
    groups = sorted(d["group"].unique(), key=lambda g: int(g[1:]) if g[1:].isdigit() else g)
    samples = [d.loc[d["group"] == g, response].to_numpy() for g in groups]
    res = stats.tukey_hsd(*samples)
    ci = res.confidence_interval(confidence_level=1 - ALPHA)
    rows = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            rows.append({
                "group1": groups[i],
                "group2": groups[j],
                "meandiff": samples[j].mean() - samples[i].mean(),
                "p_adj": res.pvalue[j, i],
                "ci_low": ci.low[j, i],
                "ci_high": ci.high[j, i],
                "reject": bool(res.pvalue[j, i] < ALPHA),
            })
    return pd.DataFrame(rows)


def planned_contrasts(growth, response, pairs):
    """Differences b - a between selected groups with Holm-adjusted p values.

    Uses a one-way model over all groups (with the series as block when present), so
    groups outside the factorial design such as leachate controls can be compared.
    """
    d = growth.dropna(subset=[response])
    pairs = [(a, b) for a, b in pairs if a in set(d["group"]) and b in set(d["group"])]
    if not pairs:
        return pd.DataFrame()
    block = " + C(series)" if has_series(d) else ""
    model = smf.ols(f"{response} ~ C(group){block}", data=d).fit()
    names = model.params.index
    rows = []
    for a, b in pairs:
        v = np.zeros(len(names))
        for g, sign in [(b, 1), (a, -1)]:
            name = f"C(group)[T.{g}]"
            if name in names:
                v[list(names).index(name)] += sign
        t = model.t_test(v)
        ci = np.asarray(t.conf_int()).ravel()
        rows.append({"group1": a, "group2": b, "difference": float(np.asarray(t.effect).ravel()[0]),
                     "ci_low": ci[0], "ci_high": ci[1], "p_raw": float(np.asarray(t.pvalue).ravel()[0])})
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p_raw"], method="holm")[1]
    out["reject"] = out["p_holm"] < ALPHA
    return out


def descriptive(growth, response):
    return (growth.groupby(["group", "group_label"])[response]
            .agg(n="count", mean="mean", sd="std", median="median", min="min", max="max")
            .reset_index())


def mixed_model(df, response):
    """ln(response) ~ day * plastic * detergent with random effects per dish.

    A random intercept and slope is tried first and a random intercept only when the
    first model does not converge. Returns (fit, status) where status records the
    random-effects structure, convergence, a boundary fit and the warnings raised.
    """
    d = factorial_part(df[df[response] > 0]).copy()
    d["log_y"] = np.log(d[response])
    d = _prepare(d)
    formula = "log_y ~ day * C(plastic) * C(detergent)" + (" + C(series)" if has_series(d) else "")
    first, error = None, None
    for label, re_formula in [("random intercept and slope", "~day"), ("random intercept", None)]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fit = smf.mixedlm(formula, d, groups=d["dish_id"], re_formula=re_formula).fit(reml=True)
            except Exception as exc:
                error = exc
                continue
        messages = sorted({str(w.message).split("\n")[0] for w in caught})
        variances = np.diag(np.atleast_2d(np.asarray(fit.cov_re, float)))
        boundary = bool(np.any(variances < 1e-8)) or any("boundary" in m.lower() for m in messages)
        status = {"response": response, "structure": label, "converged": bool(fit.converged),
                  "boundary": boundary, "warnings": "; ".join(messages)}
        if fit.converged:
            return fit, status
        first = first or (fit, status)
    if first is None:
        raise error
    return first


def _design(d, terms):
    """Treatment-coded design matrix for the given model terms (plus series dummies when present)."""
    cols = [np.ones(len(d))]
    if has_series(d):
        for level in sorted(d["series"].unique())[1:]:
            cols.append((d["series"] == level).to_numpy(float))
    plastic = {k: (d["plastic"] == k).to_numpy(float) for k in ["PLA", "PETG"]}
    det = (d["detergent"] == "yes").to_numpy(float)
    if "plastic" in terms:
        cols += [plastic["PLA"], plastic["PETG"]]
    if "detergent" in terms:
        cols.append(det)
    if "interaction" in terms:
        cols += [plastic["PLA"] * det, plastic["PETG"] * det]
    return np.column_stack(cols)


def permutation_tests(growth, response, n_perm=N_PERMUTATIONS, seed=PERMUTATION_SEED):
    """Freedman-Lane permutation p values for the type II ANOVA terms.

    For each term the residuals of the model without that term are permuted and the
    F statistic of the term is recomputed, which does not assume normal residuals.
    """
    d = factorial_part(growth).dropna(subset=[response])
    y = d[response].to_numpy(float)
    rng = np.random.default_rng(seed)
    perms = np.array([rng.permutation(len(y)) for _ in range(n_perm)]).T
    comparisons = {
        "C(plastic)": (("plastic", "detergent"), ("detergent",), 2),
        "C(detergent)": (("plastic", "detergent"), ("plastic",), 1),
        "C(plastic):C(detergent)": (("plastic", "detergent", "interaction"), ("plastic", "detergent"), 2),
    }
    out = {}
    for term, (full_terms, reduced_terms, df_term) in comparisons.items():
        xf, xr = _design(d, full_terms), _design(d, reduced_terms)
        mf = np.eye(len(y)) - xf @ np.linalg.pinv(xf)
        mr = np.eye(len(y)) - xr @ np.linalg.pinv(xr)
        df_res = len(y) - np.linalg.matrix_rank(xf)

        def f_stat(yy):
            ssr_f = ((mf @ yy) ** 2).sum(axis=0)
            ssr_r = ((mr @ yy) ** 2).sum(axis=0)
            return ((ssr_r - ssr_f) / df_term) / (ssr_f / df_res)

        f0 = float(f_stat(y[:, None])[0])
        resid = mr @ y
        fitted = y - resid
        f_perm = f_stat(fitted[:, None] + resid[perms])
        out[term] = (f0, (np.sum(f_perm >= f0 - 1e-12) + 1) / (n_perm + 1))
    return out


def robustness(growth, response):
    """Interaction and main effects checked without the normality assumption.

    Returns a table with the ANOVA p value, the permutation p value and the ANOVA
    p value on log(RGR), plus the Shapiro-Wilk p value of the log(RGR) residuals.
    """
    anova, _ = two_way_anova(growth, response)
    perm = permutation_tests(growth, response)
    d = factorial_part(growth).dropna(subset=[response])
    log_p, shapiro_log = {}, np.nan
    if (d[response] > 0).all():
        dl = d.assign(log_rgr=np.log(d[response]))
        log_anova, _ = two_way_anova(dl, "log_rgr")
        log_p = log_anova["PR(>F)"].to_dict()
        resid = dl["log_rgr"] - dl.groupby("group")["log_rgr"].transform("mean")
        if len(resid) >= 3:
            shapiro_log = float(stats.shapiro(resid).pvalue)
    rows = []
    for term in ["C(plastic)", "C(detergent)", "C(plastic):C(detergent)"]:
        rows.append({
            "source": term,
            "F": anova.loc[term, "F"],
            "p_anova": anova.loc[term, "PR(>F)"],
            "p_permutation": perm[term][1],
            "p_anova_log_rgr": log_p.get(term, np.nan),
        })
    return pd.DataFrame(rows), shapiro_log


def area_per_frond_check(df, area_col, tolerance=0.10):
    """Is the area of one frond similar across groups and days?

    The area-based frond count assumes it is. Uses photos with a manual count.
    Returns (group x day table of median area per frond, summary dict, warnings).
    """
    if "frond_count_manual" not in df.columns:
        return None, {}, []
    d = df[(df["frond_count_manual"] > 0) & (df[area_col] > 0)].copy()
    if len(d) < 6:
        return None, {}, []
    d["area_per_frond"] = d[area_col] / d["frond_count_manual"]
    table = d.pivot_table(index="group", columns="day", values="area_per_frond", aggfunc="median")
    overall = float(d["area_per_frond"].median())
    group_median = d.groupby("group")["area_per_frond"].median()
    group_dev = (group_median / overall - 1) * 100
    table["all_days"] = group_median
    table["deviation_from_overall_pct"] = group_dev
    summary = {"overall_median": overall, "unit": "mm2" if area_col == "plant_area_mm2" else "px",
               "max_group_deviation_pct": float(group_dev.abs().max())}

    # the dish is the unit: one mean value and one time slope per dish
    dish = d.groupby(["dish_id", "group"])["area_per_frond"].mean().reset_index()
    samples = [v.to_numpy() for _, v in dish.groupby("group")["area_per_frond"] if len(v) >= 2]
    if len(samples) >= 2:
        summary["kruskal_p_groups"] = float(stats.kruskal(*samples).pvalue)
    slopes = []
    for _, sub in d.groupby("dish_id"):
        if sub["day"].nunique() >= 2:
            slopes.append(np.polyfit(sub["day"], sub["area_per_frond"] / overall, 1)[0])
    if len(slopes) >= 3:
        span = d["day"].max() - d["day"].min()
        summary["mean_change_over_experiment_pct"] = float(np.mean(slopes) * span * 100)
        summary["t_test_p_time_trend"] = float(stats.ttest_1samp(slopes, 0).pvalue)

    warnings_ = []
    # small but significant differences are reported in the text; only differences above the tolerance
    # make the area-based count unreliable
    if summary["max_group_deviation_pct"] > tolerance * 100:
        warnings_.append("Area per frond differs between groups (largest deviation "
                         f"{summary['max_group_deviation_pct']:.1f} %, Kruskal-Wallis p = "
                         f"{summary.get('kruskal_p_groups', np.nan):.3g}); the area-based count is not reliable, "
                         "use manual counts")
    change = summary.get("mean_change_over_experiment_pct", 0.0)
    if abs(change) > tolerance * 100 and summary.get("t_test_p_time_trend", 1) < ALPHA:
        warnings_.append(f"Area per frond changes over the experiment ({change:+.1f} %, p = "
                         f"{summary['t_test_p_time_trend']:.3g}); the area-based count is not reliable, use manual counts")
    return table, summary, warnings_


def agreement(df, auto_col="frond_count_auto", manual_col="frond_count_manual"):
    """Agreement between automatic and manual frond counts (H6)."""
    if manual_col not in df.columns:
        return None
    d = df[[auto_col, manual_col]].dropna()
    if len(d) < 3:
        return None
    a = d[auto_col].to_numpy(float)
    m = d[manual_col].to_numpy(float)
    diff = a - m
    bias = diff.mean()
    sd = diff.std(ddof=1)
    r, p = stats.pearsonr(a, m)
    # Lin's concordance correlation coefficient (moments with denominator n, as in Lin 1989)
    ccc = 2 * np.cov(a, m, ddof=0)[0, 1] / (a.var() + m.var() + (a.mean() - m.mean()) ** 2)
    rel = np.abs(diff)[m > 0] / m[m > 0]
    mape = rel.mean() * 100 if len(rel) else np.nan
    return {
        "n_pairs": int(len(d)),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "lin_ccc": float(ccc),
        "mean_bias_auto_minus_manual": float(bias),
        "loa_low": float(bias - 1.96 * sd),
        "loa_high": float(bias + 1.96 * sd),
        "mae": float(np.mean(np.abs(diff))),
        "mape_pct": float(mape),
        "n_pairs_mape": int(len(rel)),
    }


def full_report(df, growth, metrics, extra=None, planned=None):
    """Run all tests and return (text report, dict of DataFrames for export).

    metrics: measurement columns whose growth rates (rgr_<metric>) are tested.
    extra: further per-dish responses as (column, description), e.g. root length.
    planned: (group_a, group_b) pairs compared with Holm correction.
    """
    lines = []
    tables = {}
    n_dish = growth["dish_id"].nunique()
    lines.append("LEMNA MINOR GROWTH ANALYSIS - STATISTICAL REPORT")
    lines.append("=" * 60)
    lines.append(f"Dishes (experimental units): {n_dish}")
    lines.append(f"Photos: {len(df)}")
    lines.append(f"Days: {sorted(df['day'].unique().tolist())}")
    lines.append(f"Significance level: {ALPHA}")
    lines.append("")
    if "warnings" in growth.columns:
        flagged = growth[growth["warnings"].fillna("") != ""]
        if len(flagged):
            lines.append("Dishes left out of the growth-rate tests for the listed metric:")
            for _, r in flagged.iterrows():
                lines.append(f"  {r['dish_id']}: {r['warnings']}")
            lines.append("")

    responses = [(f"rgr_{m}", m, "relative growth rate, 1/day") for m in metrics]
    responses += [(col, col, desc) for col, desc in (extra or []) if col in growth.columns]
    if has_series(growth):
        lines.append(f"Independent series: {sorted(growth['series'].unique().tolist())} "
                     "(included as a block factor in the ANOVA and the mixed model)")
        lines.append("")
    for response, m, what in responses:
        if growth[response].notna().sum() < 4:
            lines.append(f"[{response}] not enough data.")
            continue
        lines.append("-" * 60)
        lines.append(f"RESPONSE: {response}  ({what})")
        lines.append("-" * 60)

        desc = descriptive(growth, response)
        tables[f"desc_{m}"] = desc
        lines.append("Descriptive statistics per group:")
        lines.append(desc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        lines.append("")

        checks = assumption_checks(growth, response)
        lines.append("Assumption checks:")
        if "shapiro_p" in checks:
            lines.append(f"  Shapiro-Wilk on ANOVA residuals: W = {checks['shapiro_W']:.3f}, p = {checks['shapiro_p']:.4f}")
        if "levene_p" in checks:
            lines.append(f"  Levene (median) across groups:   F = {checks['levene_stat']:.3f}, p = {checks['levene_p']:.4f}")
        if "kruskal_p" in checks:
            lines.append(f"  Kruskal-Wallis (non-parametric): H = {checks['kruskal_H']:.3f}, p = {checks['kruskal_p']:.4f}")
        if checks.get("shapiro_p", 1) < ALPHA or checks.get("levene_p", 1) < ALPHA:
            lines.append("  Note: an ANOVA assumption is not met; use the robustness check below (permutation test and "
                         "log(RGR)). Kruskal-Wallis compares the six groups but does not test the interaction.")
        lines.append("")

        cells = design_cells(growth, response)
        if not is_full_factorial(growth, response):
            missing = ", ".join(f"{p}/{d} (n={n})" for (p, d), n in cells.items() if n < 2)
            lines.append("Two-way ANOVA skipped: the plastic x detergent design is incomplete, "
                         f"cells with fewer than 2 dishes: {missing}.")
            samples = [g[response].dropna().to_numpy() for _, g in growth.groupby("group")]
            samples = [x for x in samples if len(x) >= 2]
            if len(samples) >= 2:
                f, p = stats.f_oneway(*samples)
                lines.append(f"One-way ANOVA across the available groups: F = {f:.4g}, p = {p:.4g}")
            lines.append("")
        else:
            try:
                anova, model = two_way_anova(growth, response)
                tables[f"anova_{m}"] = anova.reset_index().rename(columns={"index": "source"})
                lines.append("Two-way ANOVA (plastic x detergent, type II SS):")
                lines.append(anova.to_string(float_format=lambda x: f"{x:.5g}"))
                lines.append("")
                ic = interaction_contrasts(model)
                if not ic.empty:
                    tables[f"interaction_{m}"] = ic
                    lines.append("Interaction contrasts (combined effect minus sum of single effects):")
                    lines.append(ic.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
                    lines.append("")
                rob, shapiro_log = robustness(growth, response)
                tables[f"robust_{m}"] = rob
                lines.append(f"Robustness check ({N_PERMUTATIONS} Freedman-Lane permutations, seed {PERMUTATION_SEED}; "
                             "ANOVA on log(RGR)):")
                lines.append(rob.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
                if np.isfinite(shapiro_log):
                    lines.append(f"  Shapiro-Wilk on log(RGR) residuals: p = {shapiro_log:.4f}")
                agree = ((rob["p_anova"] < ALPHA) == (rob["p_permutation"] < ALPHA))
                if rob["p_anova_log_rgr"].notna().all():
                    agree &= (rob["p_anova"] < ALPHA) == (rob["p_anova_log_rgr"] < ALPHA)
                if agree.all():
                    lines.append("  All three tests lead to the same conclusion at alpha = 0.05.")
                else:
                    terms = ", ".join(rob.loc[~agree, "source"])
                    lines.append(f"  The conclusion for {terms} depends on the test; interpret it with caution.")
                lines.append("")
            except Exception as exc:
                lines.append(f"Two-way ANOVA could not be computed: {exc}")
                lines.append("")

        try:
            tk = tukey(growth, response)
            tables[f"tukey_{m}"] = tk
            lines.append("Tukey HSD pairwise comparisons between groups:")
            lines.append(tk.to_string(index=False))
            lines.append("")
        except Exception as exc:
            lines.append(f"Tukey HSD could not be computed: {exc}")
            lines.append("")
        if planned:
            pc = planned_contrasts(growth, response, planned)
            if not pc.empty:
                tables[f"planned_{m}"] = pc
                lines.append("Planned comparisons (difference group2 - group1, Holm-adjusted p):")
                lines.append(pc.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
                lines.append("")

    mixed_status = []
    for m in metrics:
        if df["day"].nunique() < 3:
            break
        if not is_full_factorial(growth, f"rgr_{m}"):
            lines.append(f"Mixed model for {m} skipped: the plastic x detergent design is incomplete.")
            continue
        try:
            fit, status = mixed_model(df, m)
            mixed_status.append(status)
            lines.append("-" * 60)
            lines.append(f"MIXED MODEL for daily ln({m}), {status['structure']} per dish")
            lines.append("-" * 60)
            if not status["converged"]:
                lines.append("  WARNING: the model did not converge. Its estimates and p values are not reliable "
                             "and should not be reported.")
            elif status["boundary"]:
                lines.append("  Note: a random-effect variance is estimated at zero (boundary fit). The fixed effects "
                             "can be used, but the random-effects part is effectively simpler than specified.")
            if status["structure"] == "random intercept":
                lines.append("  Note: the model with a random slope did not converge; a random intercept was used.")
            lines.append(str(fit.summary()))
            lines.append("")
        except Exception as exc:
            mixed_status.append({"response": m, "structure": "-", "converged": False, "boundary": False,
                                 "warnings": str(exc)})
            lines.append(f"Mixed model for {m} could not be fitted: {exc}")
    if mixed_status:
        tables["mixed_model_status"] = pd.DataFrame(mixed_status)

    area_col = next((c for c in metrics if c.startswith("plant_area")), None)
    if area_col:
        apf, apf_summary, apf_warn = area_per_frond_check(df, area_col)
        if apf is not None:
            tables["area_per_frond"] = apf.reset_index()
            lines.append("-" * 60)
            lines.append(f"AREA PER FROND ({apf_summary['unit']}, median of {area_col} / manual count), "
                         "assumption of the area-based count")
            lines.append("-" * 60)
            lines.append(apf.to_string(float_format=lambda x: f"{x:.3g}"))
            lines.append(f"  Largest group deviation from the overall median: {apf_summary['max_group_deviation_pct']:.1f} %")
            if "kruskal_p_groups" in apf_summary:
                lines.append(f"  Kruskal-Wallis across groups (dish means): p = {apf_summary['kruskal_p_groups']:.4f}")
            if "mean_change_over_experiment_pct" in apf_summary:
                lines.append(f"  Mean change over the experiment: {apf_summary['mean_change_over_experiment_pct']:+.1f} % "
                             f"(t-test of dish slopes, p = {apf_summary['t_test_p_time_trend']:.4f})")
            for w in apf_warn:
                lines.append(f"  WARNING: {w}.")
            if not apf_warn and apf_summary.get("kruskal_p_groups", 1) < ALPHA:
                lines.append("  Area per frond differs slightly but significantly between groups (within 10 %); the "
                             "area-based count may be biased by about this amount.")
            elif not apf_warn:
                lines.append("  Area per frond is similar across groups and days (within 10 %).")
            lines.append("")
            tables["_area_per_frond_warnings"] = apf_warn

    methods = [("frond_count_auto", "watershed count", df)]
    if "frond_count_from_area" in df.columns:
        cal_day = df.attrs.get("calibration_day")
        held_out = df[df["day"] != cal_day] if cal_day is not None else df
        methods.append(("frond_count_from_area", "area-based count (calibration day excluded)", held_out))
    rows = []
    for col, name, data in methods:
        agr = agreement(data, auto_col=col)
        if agr is None:
            continue
        rows.append({"method": name, **agr})
        lines.append("-" * 60)
        lines.append(f"AGREEMENT: {name} vs manual frond count (H6)")
        lines.append("-" * 60)
        for k, v in agr.items():
            lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        lines.append("")
    if rows:
        tables["agreement"] = pd.DataFrame(rows)
    else:
        lines.append("Manual frond counts not available; agreement analysis (H6) skipped.")

    return "\n".join(lines), tables
