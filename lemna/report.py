"""Word report with tables, figures, hypothesis decisions and a biological interpretation.

The report is written in Turkish for the TUBITAK 2204-A project. All statistics are
recomputed from the CSV files of a run, so reports can also be made for older runs.
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor
from scipy import stats
from statsmodels.stats.power import TTestIndPower

from . import literature as LIT
from . import stats as LS
from .config import load_config

ALPHA = LS.ALPHA
# OECD TG 221 validity criterion: frond-number doubling time in the control below 2.5 days
OECD_MIN_CONTROL_RGR = np.log(2) / 2.5
CCC_HIGH = 0.90
CELL_ORDER = [("none", "no"), ("none", "yes"), ("PLA", "no"), ("PLA", "yes"), ("PETG", "no"), ("PETG", "yes")]
CELL_NAME = {
    ("none", "no"): "Kontrol", ("none", "yes"): "Deterjan", ("PLA", "no"): "PLA",
    ("PLA", "yes"): "PLA + deterjan", ("PETG", "no"): "PETG", ("PETG", "yes"): "PETG + deterjan",
}
METRIC_NAME = {
    "frond_count_manual": "manuel yaprak sayısı",
    "frond_count_auto": "otomatik yaprak sayısı (watershed)",
    "frond_count_from_area": "alan tabanlı yaprak sayısı",
    "plant_area_mm2": "bitki alanı (mm²)",
    "plant_area_px": "bitki alanı (piksel)",
}
COLORS = {"G1": "#4d4d4d", "G2": "#1f77b4", "G3": "#2ca02c", "G4": "#d62728", "G5": "#98df8a", "G6": "#ff9896",
          "G7": "#ff7f0e", "G8": "#9467bd"}


# ------------------------------------------------------------------ formatting

def num(x, nd=3):
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def pval(p):
    if p is None or not np.isfinite(p):
        return "–"
    return "p < 0,001" if p < 0.001 else f"p = {num(p, 3)}"


def pct(x, nd=1):
    return "–" if x is None or not np.isfinite(x) else f"%{num(x, nd)}"


# ------------------------------------------------------------------ data

class RunData:
    """Measurements, per-dish growth rates and derived statistics of one run."""

    def __init__(self, run_dir, cfg=None):
        self.run_dir = Path(run_dir)
        self.m = pd.read_csv(self.run_dir / "measurements.csv")
        self.g = pd.read_csv(self.run_dir / "growth_per_dish.csv")
        info_path = self.run_dir / "run_info.json"
        self.info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        cfg_path = self.run_dir / "config_used.json"
        self.cfg = cfg or (load_config(cfg_path) if cfg_path.exists() else load_config(None))
        self.g["warnings"] = self.g.get("warnings", pd.Series("", index=self.g.index)).fillna("")

        self.area_col = "plant_area_mm2" if ("rgr_plant_area_mm2" in self.g.columns) else "plant_area_px"
        manual_ok = ("rgr_frond_count_manual" in self.g.columns
                     and self.g["rgr_frond_count_manual"].notna().sum() >= 4)
        self.primary = "frond_count_manual" if manual_ok else self.area_col
        self.secondary = [c for c in [self.area_col, "frond_count_auto"]
                          if c != self.primary and f"rgr_{c}" in self.g.columns]
        if "factorial" not in self.g.columns:
            self.g["factorial"] = [not self.cfg["groups"].get(g, {}).get("leachate", False) for g in self.g["group"]]
        self.g["factorial"] = self.g["factorial"].astype(str).str.lower().isin(["true", "1"])
        self.cells = {}
        for cell in CELL_ORDER:
            sub = self.g[(self.g["plastic"] == cell[0]) & (self.g["detergent"] == cell[1]) & self.g["factorial"]]
            if len(sub):
                self.cells[cell] = sub["group"].iloc[0]
        self.control = self.cells.get(("none", "no"), self.cfg.get("control_group", "G1"))
        self.days = sorted(self.m["day"].unique())
        xlsx = self.run_dir / "results.xlsx"
        try:
            self.sheets = pd.read_excel(xlsx, sheet_name=None) if xlsx.exists() else {}
        except Exception:
            self.sheets = {}
        self.planned = [tuple(p) for p in self.info.get("planned_comparisons", self.cfg.get("planned_comparisons", []))]
        self.synthetic = (Path(self.info.get("photo_folder", "")).parent.name.lower().startswith("demo")
                          or str(self.info.get("note", "")).lower().startswith("demo"))

    def name(self, group):
        for cell, g in self.cells.items():
            if g == group:
                return CELL_NAME[cell]
        info = self.cfg["groups"].get(group, {})
        if info.get("leachate"):
            return f"{info.get('plastic', '')} süzüntüsü" + (" + deterjan" if info.get("detergent") == "yes" else "")
        return info.get("label", group)

    def leachate_groups(self):
        return [g for g, info in self.cfg["groups"].items() if info.get("leachate") and g in set(self.g["group"])]

    def response(self, metric):
        return f"rgr_{metric}"

    def group_table(self, metric):
        col = self.response(metric)
        t = self.g.groupby("group").agg(n=(col, "count"), mean=(col, "mean"), sd=(col, "std"))
        ctrl = t.loc[self.control, "mean"] if self.control in t.index else np.nan
        # inhibition relative to the group without plastic and detergent
        t["inhibition"] = (ctrl - t["mean"]) / ctrl * 100 if np.isfinite(ctrl) and ctrl != 0 else np.nan
        t["doubling"] = np.where(t["mean"] > 0, np.log(2) / t["mean"], np.nan)
        order = sorted(t.index, key=lambda g: (int(g[1:]) if g[1:].isdigit() else 999, g))
        return t.reindex(order)

    def full_factorial(self, metric):
        return LS.is_full_factorial(self.g, self.response(metric))


# ------------------------------------------------------------------ figures (Turkish labels)

def _style(ax):
    ax.grid(alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_growth(d, metric, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for g in d.group_table(d.primary).index:
        sub = d.m[d.m["group"] == g]
        if metric not in sub.columns or sub[metric].isna().all():
            continue
        s = sub.groupby("day")[metric].agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", ms=4, capsize=3,
                    color=COLORS.get(g), label=f"{g} – {d.name(g)}")
    ax.set_xlabel("Gün")
    ax.set_ylabel(METRIC_NAME.get(metric, metric).capitalize())
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_rgr(d, metric, path):
    col = d.response(metric)
    groups = list(d.group_table(metric).index)
    data = [d.g.loc[d.g["group"] == g, col].dropna().to_numpy() for g in groups]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.boxplot(data, widths=0.5, showfliers=False)
    rng = np.random.default_rng(1)
    for i, (g, y) in enumerate(zip(groups, data), start=1):
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(y)), y, s=22, color=COLORS.get(g), zorder=3)
    ax.set_xticks(range(1, len(groups) + 1), [f"{g}\n{d.name(g)}" for g in groups], fontsize=8)
    ax.set_ylabel("Bağıl büyüme hızı, RGR (1/gün)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_interaction(d, metric, path):
    col = d.response(metric)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for det, label, style in [("no", "Deterjan yok", "-o"), ("yes", "Deterjan var", "--s")]:
        sub = d.g[d.g["detergent"] == det]
        s = sub.groupby("plastic")[col].agg(["mean", "std", "count"]).reindex(["none", "PLA", "PETG"])
        ax.errorbar(range(3), s["mean"], yerr=s["std"] / np.sqrt(s["count"]), fmt=style, capsize=4, label=label)
    ax.set_xticks(range(3), ["Plastik yok", "PLA", "PETG"])
    ax.set_ylabel("RGR (1/gün), ortalama ± SH")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_additivity(rows, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r["expected"] for r in rows], 0.36, color="#bdbdbd", label="Toplamsal beklenti")
    ax.bar(x + 0.18, [r["observed"] for r in rows], 0.36, color="#2b8cbe", label="Gözlenen")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [r["name"] for r in rows])
    ax.set_ylabel("Büyüme inhibisyonu (%)")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_agreement(d, auto_col, path, exclude_day=None):
    a = d.m[[auto_col, "frond_count_manual", "day"]].dropna()
    if exclude_day is not None:
        a = a[a["day"] != exclude_day]
    x, y = a["frond_count_manual"].to_numpy(float), a[auto_col].to_numpy(float)
    diff, mean = y - x, (x + y) / 2
    bias, sd = diff.mean(), diff.std(ddof=1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    lim = [0, max(x.max(), y.max()) * 1.05]
    axes[0].scatter(x, y, s=12, alpha=0.6)
    axes[0].plot(lim, lim, "k--", lw=1)
    axes[0].set_xlabel("Manuel sayım")
    axes[0].set_ylabel("Otomatik sayım")
    axes[1].scatter(mean, diff, s=12, alpha=0.6)
    for v, ls in [(bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")]:
        axes[1].axhline(v, color="black", ls=ls, lw=1)
    axes[1].set_xlabel("İki sayımın ortalaması")
    axes[1].set_ylabel("Otomatik − manuel")
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ analysis helpers

def pairwise(tk, a, b):
    """Tukey result for groups a and b: (difference b - a, adjusted p)."""
    if tk is None or a is None or b is None:
        return np.nan, np.nan
    row = tk[(tk["group1"] == a) & (tk["group2"] == b)]
    if len(row):
        return float(row["meandiff"].iloc[0]), float(row["p_adj"].iloc[0])
    row = tk[(tk["group1"] == b) & (tk["group2"] == a)]
    if len(row):
        return -float(row["meandiff"].iloc[0]), float(row["p_adj"].iloc[0])
    return np.nan, np.nan


def detectable_difference(d, metric):
    col = d.response(metric)
    y = d.g.dropna(subset=[col])
    resid = y[col] - y.groupby("group")[col].transform("mean")
    dof = len(y) - y["group"].nunique()
    if dof < 2:
        return np.nan, np.nan, np.nan
    sd = float(np.sqrt((resid ** 2).sum() / dof))
    n = int(np.median(y.groupby("group").size()))
    if n < 2:
        return sd, np.nan, n
    eff = TTestIndPower().solve_power(nobs1=n, alpha=ALPHA, power=0.8)
    return sd, float(eff * sd), n


def additivity_rows(d, metric):
    """Observed and expected (response addition) inhibition of the combined treatments."""
    t = d.group_table(metric)
    det = d.cells.get(("none", "yes"))
    rows = []
    for plastic in ["PLA", "PETG"]:
        alone, combo = d.cells.get((plastic, "no")), d.cells.get((plastic, "yes"))
        if None in (det, alone, combo):
            continue
        i_p, i_d, i_c = t.loc[alone, "inhibition"], t.loc[det, "inhibition"], t.loc[combo, "inhibition"]
        rows.append({"plastic": plastic, "name": CELL_NAME[(plastic, "yes")], "alone": i_p, "detergent": i_d,
                     "expected": i_p + i_d, "observed": i_c, "difference": i_c - (i_p + i_d)})
    return rows


# ------------------------------------------------------------------ document helpers

class Doc:
    def __init__(self):
        self.doc = Document()
        st = self.doc.styles["Normal"]
        st.font.name = "Calibri"
        st.font.size = Pt(11)
        for sec in self.doc.sections:
            sec.left_margin = sec.right_margin = Cm(2.2)
            sec.top_margin = sec.bottom_margin = Cm(2.0)
        self.n_table = 0
        self.n_fig = 0

    def h(self, text, level=1):
        self.doc.add_heading(text, level=level)

    def section(self, title):
        """Numbered main section; returns its number."""
        self.n_sec = getattr(self, "n_sec", 0) + 1
        self.n_sub = 0
        self.h(f"{self.n_sec}. {title}")
        return str(self.n_sec)

    def subsection(self, title):
        """Numbered subsection of the current section; returns its number (e.g. '4.2')."""
        self.n_sub += 1
        no = f"{self.n_sec}.{self.n_sub}"
        self.h(f"{no}. {title}", 2)
        return no

    def p(self, text, italic=False, bold=False, size=None, color=None, align=None):
        par = self.doc.add_paragraph()
        run = par.add_run(text)
        run.italic, run.bold = italic, bold
        if size:
            run.font.size = Pt(size)
        if color:
            run.font.color.rgb = RGBColor(*color)
        if align:
            par.alignment = align
        par.paragraph_format.space_after = Pt(6)
        return par

    def bullets(self, items):
        for it in items:
            self.doc.add_paragraph(it, style="List Bullet")

    def table(self, caption, header, rows, widths=None):
        self.n_table += 1
        cap = self.p(f"Tablo {self.n_table}. {caption}", bold=True, size=10)
        cap.paragraph_format.keep_with_next = True
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, htext in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            r = cell.paragraphs[0].add_run(htext)
            r.bold = True
            r.font.size = Pt(9)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                r = cells[i].paragraphs[0].add_run(str(v))
                r.font.size = Pt(9)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Cm(w)
        self.doc.add_paragraph()
        return self.n_table

    def figure(self, path, caption, width_cm=15):
        self.n_fig += 1
        self.doc.add_picture(str(path), width=Cm(width_cm))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.p(f"Şekil {self.n_fig}. {caption}", italic=True, size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER)
        return self.n_fig

    def save(self, path):
        self.doc.save(str(path))


class Citations:
    """Collects literature keys cited in the report text."""

    def __init__(self):
        self.keys = set()

    def __call__(self, *keys):
        self.keys.update(keys)
        return LIT.c(*keys)


def group_values(d, col):
    order = sorted(d.g["group"].unique(), key=lambda g: (int(g[1:]) if g[1:].isdigit() else 999, g))
    t = d.g.groupby("group")[col].agg(["count", "mean", "std"])
    return t.reindex(order).dropna(subset=["mean"])


def fig_box(d, col, ylabel, path):
    t = group_values(d, col)
    groups = list(t.index)
    data = [d.g.loc[d.g["group"] == g, col].dropna().to_numpy() for g in groups]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.boxplot(data, widths=0.5, showfliers=False)
    rng = np.random.default_rng(2)
    for i, (g, y) in enumerate(zip(groups, data), start=1):
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(y)), y, s=16, color=COLORS.get(g), zorder=3)
    ax.set_xticks(range(1, len(groups) + 1), [f"{g}\n{d.name(g)}" for g in groups], fontsize=7)
    ax.set_ylabel(ylabel)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_chemistry(d, daily, path):
    cols = [(c, lab) for c, lab in [("pH_mean", "pH")] if c in daily.columns and daily[c].notna().any()]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.6 * len(cols), 3.8), squeeze=False)
    for ax, (c, lab) in zip(axes[0], cols):
        for g, sub in daily.groupby("group"):
            ax.plot(sub["day"], sub[c], marker="o", ms=3, color=COLORS.get(g), label=f"{g} – {d.name(g)}")
        ax.set_xlabel("Gün")
        ax.set_ylabel(lab)
        _style(ax)
    axes[0][-1].legend(fontsize=6.5, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def anova_phrase(d, col):
    """Short Turkish summary of the two-way ANOVA for a per-dish response, or '' if not possible."""
    if not LS.is_full_factorial(d.g, col):
        return ""
    a, _ = LS.two_way_anova(d.g, col)
    parts = [f"{lab} {pval(a.loc[t, 'PR(>F)'])}" for t, lab in
             [("C(detergent)", "deterjan"), ("C(plastic)", "plastik türü"), ("C(plastic):C(detergent)", "etkileşim")]]
    return "İki faktörlü varyans analizinde " + ", ".join(parts) + " bulunmuştur."


# ------------------------------------------------------------------ report

def build_report(run_dir, out_path=None, title=None, author="", cfg=None):
    """Write the Word report of one run and return its path."""
    d = RunData(run_dir, cfg)
    run_id = d.info.get("run_id", d.run_dir.name)
    out_path = Path(out_path) if out_path else d.run_dir / f"rapor_{run_id}.docx"
    fig_dir = d.run_dir / "report_figures"
    fig_dir.mkdir(exist_ok=True)
    P = d.primary
    resp = d.response(P)
    gt = d.group_table(P)
    if d.g[resp].notna().sum() < 2:
        return _short_report(d, out_path, title, author)
    counts = d.g.groupby("group")[resp].count()
    tk_groups = counts[counts >= 2].index
    tk = LS.tukey(d.g[d.g["group"].isin(tk_groups)], resp) if len(tk_groups) >= 2 else None
    full = d.full_factorial(P)
    anova = contrasts = rob = None
    if full:
        anova, model = LS.two_way_anova(d.g, resp)
        contrasts = LS.interaction_contrasts(model).set_index("contrast") if model is not None else None
        rob, _ = LS.robustness(d.g, resp)
        rob = rob.set_index("source")
    shapiro = LS.assumption_checks(d.g, resp)
    sd, mdd, n_rep = detectable_difference(d, P)
    ctrl_rgr = gt.loc[d.control, "mean"] if d.control in gt.index else np.nan
    add_rows = additivity_rows(d, P)
    cal_day = d.info.get("calibration_day")

    doc = Doc()
    doc.p(title or "3D Yazıcı Kaynaklı PLA ve PETG Parçacıklarının Deterjan Varlığında Lemna minor "
          "Gelişimine Etkisi", bold=True, size=16)
    doc.p("Görüntü analizi sonuç raporu", size=12, color=(90, 90, 90))
    meta = [f"Analiz: {run_id}", f"Rapor tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}"]
    if author:
        meta.insert(0, f"Hazırlayan: {author}")
    doc.p("   ·   ".join(meta), size=9.5, color=(90, 90, 90))
    if d.synthetic:
        doc.p("Bu rapor yazılımı denemek için bilgisayarla üretilmiş yapay fotoğraflardan hazırlanmıştır. "
              "Gruplar arasındaki farklar programa önceden verilmiş değerlerden kaynaklanır ve biyolojik bir "
              "bulgu olarak yorumlanmamalıdır.", bold=True, color=(180, 40, 40))
    doc.p("Bu rapor yazılım tarafından ölçüm ve test sonuçlarından otomatik olarak üretilmiştir. Biyolojik "
          "yorumlar olası açıklamalar niteliğindedir ve deney koşulları, literatür ve öğretmen değerlendirmesiyle "
          "birlikte ele alınmalıdır.", italic=True, size=9.5, color=(90, 90, 90))

    cite = Citations()
    doc.section("Literatür özeti")
    for text, keys in LIT.REPORT_SUMMARY:
        cite.keys.update(keys)
        doc.p(text)

    # ---------------- 1. data and methods
    doc.section("Veri ve yöntem özeti")
    n_dish = d.g["dish_id"].nunique()
    doc.p(f"Analizde {len(d.m)} fotoğraf ve {n_dish} deney kabı kullanılmıştır. Fotoğraflar "
          f"{d.days[0]}. ile {d.days[-1]}. günler arasında çekilmiştir. Deney birimi kaptır ve aynı kabın farklı "
          "günlerdeki fotoğrafları bağımsız tekrar olarak sayılmamıştır.")
    doc.p("Her fotoğrafta kap otomatik olarak bulunmuş, bitki dokusu HSV renk eşikleriyle sağlıklı (yeşil), "
          "sararmış ve kahverengileşmiş doku olarak sınıflandırılmıştır. Kap çapı kullanılarak alanlar mm² "
          "cinsine çevrilmiştir. Her kap için bağıl büyüme hızı (RGR) OECD 221 rehberindeki formülle "
          f"hesaplanmıştır {cite('oecd2006')}: RGR = [ln N(t) − ln N(0)] / t. Burada N(0) ve N(t) ilk ve son gündeki değerleri, t ise "
          "gün cinsinden süreyi gösterir. Büyüme inhibisyonu kontrol grubunun ortalama RGR değerine göre "
          "I = [RGR(kontrol) − RGR(uygulama)] / RGR(kontrol) × 100 olarak hesaplanmıştır.")
    doc.p(f"Ana ölçüt olarak {METRIC_NAME[P]} temelinde hesaplanan RGR kullanılmıştır"
          + (" (OECD 221'in temel ölçütü)." if P == "frond_count_manual" else
             ". Manuel sayım her kapta ilk ve son gün için bulunmadığından alan temelli RGR esas alınmıştır.")
          + " Gruplar iki faktörlü varyans analiziyle (plastik türü × deterjan) karşılaştırılmış, etkileşim "
          f"normallik varsayımı gerektirmeyen bir permütasyon testi {cite('freedman1983')} ve log(RGR) üzerinde ANOVA ile de "
          f"sınanmıştır. İkili karşılaştırmalarda Tukey HSD testi uygulanmıştır. Anlamlılık düzeyi {num(ALPHA, 2)} "
          "olarak alınmıştır."
          + (f" Deney {d.g['series'].nunique()} bağımsız seri halinde yürütüldüğünden seri, varyans analizine ve karma "
             "modele blok etkeni olarak eklenmiştir." if "series" in d.g.columns and d.g["series"].nunique() > 1 else "")
          + (" Süzüntü grupları faktöriyel analize alınmamış, planlanmış karşılaştırmalarla (Holm düzeltmesi) "
             "değerlendirilmiştir." if d.leachate_groups() else ""))

    # ---------------- 2. quality
    doc.section("Veri kalitesi")
    qc = list(d.info.get("quality_warnings", []))
    flagged = d.g[d.g["warnings"] != ""]
    if not qc and flagged.empty:
        doc.p("Kalite kontrolünde uyarı bulunmamıştır. Tüm fotoğraflarda kap bulunmuş, her kabın her gün "
              "fotoğrafı mevcuttur ve kap ölçeği fotoğraflar arasında tutarlıdır.")
    else:
        doc.p("Kalite kontrolünde aşağıdaki durumlar saptanmıştır. Sonuçlar bu durumlar göz önünde tutularak "
              "yorumlanmalıdır.")
        items = [_tr_quality(w) for w in qc]
        items += [f"{r.dish_id}: büyüme hızı hesaplanamadı ({_tr_quality(r.warnings)})" for r in flagged.itertuples()]
        doc.bullets(items)
    if P == "frond_count_manual" and np.isfinite(ctrl_rgr):
        dt = np.log(2) / ctrl_rgr
        if ctrl_rgr >= OECD_MIN_CONTROL_RGR:
            doc.p(f"Kontrol grubunda yaprak sayısının ikiye katlanma süresi {num(dt, 2)} gündür. Bu değer OECD 221 "
                  "rehberinin geçerlilik ölçütünü (2,5 günden kısa) karşılamaktadır ve kontrol bitkilerinin deney "
                  "koşullarında beklenen hızda büyüdüğünü gösterir.")
        else:
            doc.p(f"Kontrol grubunda yaprak sayısının ikiye katlanma süresi {num(dt, 2)} gündür ve OECD 221 "
                  "rehberinin geçerlilik ölçütünü (2,5 günden kısa) karşılamamaktadır. Kontrol bitkilerinin "
                  "yavaş büyümesi ışık, sıcaklık veya besin çözeltisiyle ilgili olabilir ve gruplar arasındaki "
                  "farkların saptanmasını zorlaştırır.", bold=True)

    # ---------------- 3. results
    doc.section("Bulgular")
    doc.subsection("Büyüme ve büyüme hızı")
    rows = [[g, d.name(g), int(r.n), f"{num(r['mean'])} ± {num(r.sd)}", num(r.doubling, 2), pct(r.inhibition)]
            for g, r in gt.iterrows()]
    t_growth = doc.table(f"Gruplara göre bağıl büyüme hızı ({METRIC_NAME[P]} temelinde), ikiye katlanma süresi "
                         "ve kontrole göre büyüme inhibisyonu. Değerler ortalama ± standart sapmadır.",
                         ["Grup", "Uygulama", "n (kap)", "RGR (1/gün)", "İkiye katlanma (gün)", "İnhibisyon"], rows,
                         widths=[1.4, 3.6, 1.6, 3.2, 2.8, 2.2])
    p1 = fig_dir / "sekil_buyume.png"
    fig_growth(d, P, p1)
    f_growth = doc.figure(p1, f"Gruplara göre günlük {METRIC_NAME[P]} (ortalama ± standart sapma).")
    p2 = fig_dir / "sekil_rgr.png"
    fig_rgr(d, P, p2)
    f_rgr = doc.figure(p2, "Kaplara göre bağıl büyüme hızı. Her nokta bir deney kabını gösterir.")
    order = gt.dropna(subset=["mean"]).sort_values("mean")
    slow, fast = order.index[0], order.index[-1]
    doc.p(f"En yüksek büyüme hızı {d.name(fast)} grubunda ({num(gt.loc[fast, 'mean'])} 1/gün), en düşük büyüme hızı "
          f"{d.name(slow)} grubunda ({num(gt.loc[slow, 'mean'])} 1/gün) ölçülmüştür (Tablo {t_growth}, Şekil "
          f"{f_growth} ve Şekil {f_rgr}).")

    doc.subsection("Plastik türü ve deterjanın etkisi")
    if full:
        rows = []
        for term, label in [("C(plastic)", "Plastik türü"), ("C(detergent)", "Deterjan"),
                            ("C(plastic):C(detergent)", "Plastik × deterjan")]:
            a = anova.loc[term]
            rows.append([label, int(a["df"]), num(a["F"], 2), pval(a["PR(>F)"]), num(a["partial_eta_sq"], 2),
                         pval(rob.loc[term, "p_permutation"]), pval(rob.loc[term, "p_anova_log_rgr"])])
        t_anova = doc.table("İki faktörlü varyans analizi ve sağlamlık kontrolü (Tip II kareler toplamı). "
                            "Kısmi η² etki büyüklüğünü gösterir.",
                            ["Etki", "sd", "F", "p (ANOVA)", "Kısmi η²", "p (permütasyon)", "p (log RGR)"], rows,
                            widths=[3.4, 1.0, 1.6, 2.4, 1.8, 2.6, 2.4])
        doc.p(_anova_sentence(anova, rob, shapiro, t_anova))
    else:
        cells = LS.design_cells(d.g, resp)
        missing = ", ".join(CELL_NAME[c] for c, n in cells.items() if n < 2)
        doc.p(f"Deney düzeninde en az iki kabı olmayan uygulamalar bulunduğundan ({missing}) iki faktörlü varyans "
              "analizi ve etkileşim testi yapılamamıştır. Bu bölümde yalnızca ikili karşılaştırmalar verilmiştir.")

    if tk is not None:
        rows = []
        for (a, b, label) in _key_pairs(d):
            diff, p = pairwise(tk, a, b)
            if np.isfinite(p):
                wrap = lambda g: f"({d.name(g)})" if "+" in d.name(g) else d.name(g)
                rows.append([label, f"{wrap(b)} − {wrap(a)}", num(diff), pval(p),
                             "anlamlı" if p < ALPHA else "anlamlı değil"])
        if rows:
            t_tukey = doc.table("Hipotezlerle ilgili ikili karşılaştırmalar (Tukey HSD, düzeltilmiş p).",
                                ["Karşılaştırma", "Fark (RGR, 1/gün)", "Değer", "p", "Sonuç"],
                                [[r[0], r[1], r[2], r[3], r[4]] for r in rows], widths=[4.2, 5.0, 1.8, 2.2, 2.4])
            doc.p(f"Tukey düzeltmesi tüm grup çiftleri için yapılmış, hipotezlerle ilgili karşılaştırmalar Tablo "
                  f"{t_tukey} içinde verilmiştir.")

    if add_rows:
        doc.subsection("Birlikte maruziyet ve etkileşim")
        rows = [[r["name"], pct(r["alone"]), pct(r["detergent"]), pct(r["expected"]), pct(r["observed"]),
                 f"{'+' if r['difference'] >= 0 else '−'}{num(abs(r['difference']), 1)} puan"] for r in add_rows]
        t_add = doc.table("Birlikte uygulamada gözlenen inhibisyon ile tekil etkilerin toplamından beklenen "
                          "(toplamsal) inhibisyon.",
                          ["Uygulama", "Plastik tek başına", "Deterjan tek başına", "Beklenen", "Gözlenen", "Fark"],
                          rows, widths=[3.2, 2.6, 2.6, 2.2, 2.2, 2.4])
        p3 = fig_dir / "sekil_etkilesim.png"
        fig_interaction(d, P, p3)
        f_int = doc.figure(p3, "Etkileşim grafiği. Paralel çizgiler plastik ve deterjanın etkilerinin toplandığını, "
                               "paralellikten belirgin sapma ise etkileşimi gösterir.", width_cm=12)
        p4 = fig_dir / "sekil_toplamsallik.png"
        fig_additivity(add_rows, p4)
        f_add = doc.figure(p4, "Birlikte uygulamalarda gözlenen ve toplamsal olarak beklenen büyüme inhibisyonu.",
                           width_cm=12)
        for r in add_rows:
            doc.p(_additivity_sentence(r, contrasts, t_add, f"{f_int} ve Şekil {f_add}"))

    pc = LS.planned_contrasts(d.g, resp, d.planned) if d.planned else pd.DataFrame()
    if d.leachate_groups() and len(pc):
        doc.subsection("Süzüntü kontrolleri (H6)")
        rows = [[f"{d.name(r.group2)} − {d.name(r.group1)}", num(r.difference), f"{num(r.ci_low)} – {num(r.ci_high)}",
                 pval(r.p_holm), "anlamlı" if r.p_holm < ALPHA else "anlamlı değil"] for r in pc.itertuples()]
        t_pc = doc.table("Süzüntü gruplarıyla yapılan planlanmış karşılaştırmalar (büyüme hızı farkı, 1/gün; p değerleri "
                         "Holm yöntemiyle düzeltilmiştir).", ["Karşılaştırma", "Fark", "%95 güven aralığı", "p", "Sonuç"],
                         rows, widths=[6.0, 1.8, 3.2, 2.0, 2.4])
        for par in _leachate_text(d, pc, cite, t_pc):
            doc.p(par)

    doc.subsection("Renk değişimi (sararma)")
    for par in _yellowing(d, P):
        doc.p(par)

    root_tk = None
    if "root_length_mm" in d.g.columns and d.g["root_length_mm"].notna().sum() >= 4:
        doc.subsection("Kök uzunluğu")
        rt = group_values(d, "root_length_mm")
        root_tk = LS.tukey(d.g, "root_length_mm")
        letters = compact_letters(list(rt.index), rt["mean"].to_dict(), root_tk)
        rows = [[g, d.name(g), int(r["count"]), f"{num(r['mean'], 1)} ± {num(r['std'], 1)}", letters.get(g, "")]
                for g, r in rt.iterrows()]
        t_root = doc.table("Yedinci gündeki kök uzunluğu (mm, ortalama ± standart sapma). Aynı harfi paylaşan gruplar "
                           "arasında anlamlı fark yoktur (Tukey HSD).", ["Grup", "Uygulama", "n", "Kök uzunluğu (mm)",
                                                                       "Harf"], rows, widths=[1.5, 5.2, 1.2, 3.6, 1.6])
        pr = fig_dir / "sekil_kok.png"
        fig_box(d, "root_length_mm", "Kök uzunluğu (mm)", pr)
        f_root = doc.figure(pr, "Kaplara göre kök uzunluğu. Her nokta bir deney kabını gösterir.", width_cm=13)
        doc.p(_root_text(d, rt, root_tk, P, tk, cite, t_root, f_root))

    chem_cols = [c for c in ["pH_end", "pH_change"] if c in d.g.columns
                 and d.g[c].notna().sum() >= 4]
    if chem_cols:
        doc.subsection("Su kimyası (pH)")
        labels = {"pH_end": "pH (son gün)", "pH_change": "pH değişimi"}
        tabs = {c: group_values(d, c) for c in chem_cols}
        groups = list(next(iter(tabs.values())).index)
        rows = [[g, d.name(g)] + [f"{num(tabs[c].loc[g, 'mean'], 2)} ± "
                                  f"{num(tabs[c].loc[g, 'std'], 2)}" if g in tabs[c].index else "–"
                                  for c in chem_cols] for g in groups]
        t_chem = doc.table("Gruplara göre pH (ortalama ± standart sapma). Değişim, son gün ile "
                           "ilk gün arasındaki farktır.", ["Grup", "Uygulama"] + [labels[c] for c in chem_cols], rows)
        daily = d.sheets.get("water_chemistry_daily")
        f_chem = None
        if daily is not None and len(daily):
            pch = fig_dir / "sekil_su_kimyasi.png"
            fig_chemistry(d, daily, pch)
            f_chem = doc.figure(pch, "Gruplara göre günlük ortalama pH.")
        for par in _chemistry_text(d, chem_cols, cite, t_chem, f_chem):
            doc.p(par)

    doc.subsection("Otomatik ölçümün doğruluğu (H7)")
    agr_rows, h6 = [], None
    if "frond_count_manual" in d.m.columns and d.m["frond_count_manual"].notna().sum() >= 3:
        for col, label, sub in [("frond_count_auto", "Watershed sayımı", d.m),
                                ("frond_count_from_area", "Alan tabanlı sayım", d.m if cal_day is None
                                 else d.m[d.m["day"] != cal_day])]:
            if col in d.m.columns:
                a = LS.agreement(sub, auto_col=col)
                if a:
                    agr_rows.append((label, col, a))
    if agr_rows:
        rows = [[lab, a["n_pairs"], num(a["lin_ccc"], 3), num(a["mean_bias_auto_minus_manual"], 2),
                 f"{num(a['loa_low'], 1)} / {num(a['loa_high'], 1)}", num(a["mae"], 2), pct(a["mape_pct"])]
                for lab, _, a in agr_rows]
        t_agr = doc.table("Otomatik ve manuel yaprak sayımlarının uyumu. "
                          f"CCC: Lin uyum korelasyon katsayısı {cite('lin1989')}, "
                          "fark: otomatik − manuel.",
                          ["Yöntem", "n", "CCC", "Ortalama fark", "%95 uyum sınırları", "MAE", "MAPE"], rows,
                          widths=[3.4, 1.2, 1.6, 2.4, 3.2, 1.6, 1.8])
        best = max(agr_rows, key=lambda x: x[2]["lin_ccc"])
        p5 = fig_dir / "sekil_uyum.png"
        fig_agreement(d, best[1], p5, cal_day if best[1] == "frond_count_from_area" else None)
        f_agr = doc.figure(p5, f"{best[0]} ile manuel sayımın karşılaştırılması (solda) ve Bland–Altman grafiği "
                               f"{cite('bland1986')} (sağda).")
        h6 = agr_rows
        doc.p(_agreement_sentence(agr_rows, t_agr, f_agr, cal_day))
        apf, apf_sum, apf_warn = LS.area_per_frond_check(d.m, d.area_col)
        if apf is not None and any(lab == "Alan tabanlı sayım" for lab, _, _ in agr_rows):
            if apf_warn:
                doc.p("Yaprak başına düşen alan gruplar veya günler arasında belirgin biçimde değişmiştir "
                      f"(en büyük grup sapması {pct(apf_sum['max_group_deviation_pct'])}). Bu durumda alan tabanlı "
                      "sayım yaprak sayısını sistematik olarak hatalı tahmin eder ve manuel sayım esas alınmalıdır.")
            elif apf_sum.get("kruskal_p_groups", 1) < ALPHA:
                doc.p("Yaprak başına düşen alan gruplar arasında küçük ama anlamlı biçimde farklıdır (en büyük grup "
                      f"sapması {pct(apf_sum['max_group_deviation_pct'])}, Kruskal–Wallis "
                      f"{pval(apf_sum['kruskal_p_groups'])}). Stres altındaki bitkilerde yaprakların daha küçük "
                      "kalması beklenen bir durumdur. Fark %10'un altında kaldığından alan tabanlı sayım kullanılabilir, "
                      "ancak bu gruplarda yaprak sayısını yaklaşık bu oranda yanlı tahmin edebilir. Büyüme hızı "
                      "karşılaştırmalarında manuel sayım esas alınmalıdır.")
            else:
                doc.p("Yaprak başına düşen alan gruplar ve günler arasında benzer kalmıştır (en büyük grup sapması "
                      f"{pct(apf_sum['max_group_deviation_pct'])}). Alan tabanlı sayımın temel varsayımı bu veride "
                      "sağlanmaktadır.")
    else:
        doc.p("Manuel sayım bulunmadığından otomatik ölçümün doğruluğu değerlendirilememiştir.")

    # ---------------- 4. statistical summary
    used_refs, power_no = _statistics_section(doc, d, P, resp, gt, tk, anova, rob, contrasts, shapiro, sd, mdd, n_rep,
                                    ctrl_rgr)

    # ---------------- 5. hypotheses
    doc.section("Hipotezlerin değerlendirilmesi")
    hyp = _hypotheses(d, tk, anova, rob, h6, mdd, root_tk, pc)
    doc.table("Araştırma hipotezlerinin test sonuçlarına göre değerlendirilmesi.",
              ["Hipotez", "Kullanılan test", "Sonuç", "Karar"], hyp, widths=[4.4, 4.0, 5.0, 2.8])
    if power_no:
        doc.p("“Desteklenmedi” kararı bu deney düzeninin duyarlılığıyla birlikte okunmalıdır. Saptanabilir en küçük "
              f"fark ({num(mdd)} 1/gün) ve gereken örneklem büyüklüğü Bölüm {power_no} içinde verilmiştir.")

    # ---------------- 6. interpretation
    doc.section("Biyolojik yorum ve literatürle karşılaştırma")
    for par in _interpretation(d, P, gt, tk, anova, rob, contrasts, add_rows, ctrl_rgr, cite):
        doc.p(par)

    # ---------------- 6. limitations and recommendations
    doc.section("Sınırlılıklar")
    lim = [
        "Yeşil alan ve renk ölçümleri bitkinin kapladığı alanı ve rengini gösterir. Biyokütle, klorofil içeriği "
        "veya fotosentez doğrudan ölçülmemiştir.",
        "Deterjan tek bir konsantrasyonda uygulandığından konsantrasyona bağlı bir ilişki (doz-yanıt) "
        "değerlendirilememiştir.",
        "Kullanılan filamentlerin katkı maddeleri ve parçacıkların hazırlanma biçimi çevrede bulunan "
        "mikroplastikleri tam olarak temsil etmeyebilir.",
        "Deney yedi günlük kısa süreli etkileri kapsar ve laboratuvar sonuçları doğal su ortamlarına doğrudan "
        "genellenemez.",
        "Bu düzen karşılaştırmalı bir araştırma tasarımıdır ve OECD 221'in en az beş konsantrasyonlu standart "
        "toksisite testinin yerine geçmez.",
    ]
    if shapiro.get("shapiro_p", 1) < ALPHA:
        lim.append("ANOVA artıklarında normallik varsayımı sağlanmamıştır. Bu nedenle sonuçlar permütasyon testi ve "
                   "log(RGR) analiziyle birlikte değerlendirilmiştir.")
    doc.bullets(lim)
    doc.section("Öneriler")
    doc.bullets(_recommendations(d, add_rows, contrasts, mdd, ctrl_rgr))

    doc.h("Kaynaklar")
    cite.keys.update(used_refs)
    entries = [LIT.apa(k) for k in cite.keys]
    for ref in sorted(set(entries), key=str.lower):
        doc.p(ref, size=10)
    doc.save(out_path)
    return out_path


def _short_report(d, out_path, title, author):
    """Report for runs in which no growth rate could be computed (e.g. a single day)."""
    doc = Doc()
    doc.p(title or "Lemna minor görüntü analizi", bold=True, size=16)
    doc.p("Görüntü analizi sonuç raporu", size=12, color=(90, 90, 90))
    if author:
        doc.p(f"Hazırlayan: {author}", size=9.5, color=(90, 90, 90))
    doc.h("Rapor oluşturulamadı")
    doc.p(f"Bu analizde {len(d.m)} fotoğraf bulunmaktadır ve fotoğraflar yalnızca şu günlere aittir: "
          f"{', '.join(str(x) for x in d.days)}. Büyüme hızı en az iki farklı günün ölçümünü gerektirdiğinden "
          "hesaplanamamış, bu nedenle istatistiksel karşılaştırma ve biyolojik yorum yapılamamıştır. Son gün "
          "fotoğrafları eklendikten sonra analiz yeniden çalıştırılmalıdır.")
    rows = [[g, d.name(g), len(sub), num(sub[d.area_col].mean(), 1)] for g, sub in d.m.groupby("group")]
    doc.table("Mevcut fotoğraflardaki ortalama bitki alanı.", ["Grup", "Uygulama", "Fotoğraf", "Alan"], rows)
    doc.save(out_path)
    return out_path


def lower_first(text):
    """Lower-case the first letter unless the first word is an abbreviation such as PLA."""
    word = text.split()[0] if text else ""
    if len(word) > 1 and word[:2].isupper():
        return text
    return text[:1].replace("I", "ı").replace("İ", "i").lower() + text[1:]


def eta_label(eta):
    """Verbal size of partial eta squared (Cohen 1988 benchmarks 0.01, 0.06, 0.14)."""
    if not np.isfinite(eta):
        return "–"
    if eta >= 0.14:
        return "büyük"
    if eta >= 0.06:
        return "orta"
    if eta >= 0.01:
        return "küçük"
    return "ihmal edilebilir"


def compact_letters(groups, means, tk):
    """Compact letter display: groups sharing a letter do not differ (Tukey, alpha 0.05).

    Letters are the maximal sets of groups whose pairs are all non-significant,
    lettered from the highest to the lowest mean.
    """
    groups = list(groups)
    n = len(groups)

    def same(a, b):
        _, p = pairwise(tk, a, b)
        return not (np.isfinite(p) and p < ALPHA)

    cliques = []
    for mask in range(1, 2 ** n):
        members = [groups[i] for i in range(n) if mask >> i & 1]
        if all(same(a, b) for i, a in enumerate(members) for b in members[i + 1:]):
            cliques.append(set(members))
    maximal = [c for c in cliques if not any(c < o for o in cliques)]
    maximal.sort(key=lambda c: -max(means[g] for g in c))
    letters = {g: "" for g in groups}
    for i, c in enumerate(maximal):
        for g in sorted(c):
            letters[g] += "abcdefghijklmnopqrstuvwxyz"[i]
    return letters


def mixed_slopes(d, metric):
    """Daily ln-growth rate of each treatment from the mixed model with 95 % confidence intervals."""
    fit, status = LS.mixed_model(d.m, metric)
    fe = fit.fe_params
    cov = fit.cov_params().loc[fe.index, fe.index].to_numpy()
    rows = []
    for cell in CELL_ORDER:
        if cell not in d.cells:
            continue
        plastic, det = cell
        v = pd.Series(0.0, index=fe.index)
        v["day"] = 1
        if plastic != "none":
            v[f"day:C(plastic)[T.{plastic}]"] = 1
        if det == "yes":
            v["day:C(detergent)[T.yes]"] = 1
        if plastic != "none" and det == "yes":
            v[f"day:C(plastic)[T.{plastic}]:C(detergent)[T.yes]"] = 1
        est = float(v @ fe)
        se = float(np.sqrt(v.to_numpy() @ cov @ v.to_numpy()))
        rows.append({"group": d.cells[cell], "slope": est, "low": est - 1.96 * se, "high": est + 1.96 * se})
    rows.sort(key=lambda r: (int(r["group"][1:]) if r["group"][1:].isdigit() else 999, r["group"]))
    inter = {pl: (float(fit.params.get(f"day:C(plastic)[T.{pl}]:C(detergent)[T.yes]", np.nan)),
                  float(fit.pvalues.get(f"day:C(plastic)[T.{pl}]:C(detergent)[T.yes]", np.nan)))
             for pl in ["PLA", "PETG"]}
    return pd.DataFrame(rows), status, inter


def evidence(p, robust_ps=None):
    """Evidence level from the main p value and, when available, the robustness p values."""
    if not np.isfinite(p):
        return "–"
    robust_ps = [x for x in (robust_ps or []) if np.isfinite(x)]
    agree = all((x < ALPHA) == (p < ALPHA) for x in robust_ps)
    if p < ALPHA and not agree:
        return "Zayıf (teste bağlı)"
    if p >= ALPHA:
        return "Zayıf (teste bağlı)" if not agree else "Kanıt yok"
    return "Güçlü" if p < 0.01 else "Orta"


def _statistics_section(doc, d, P, resp, gt, tk, anova, rob, contrasts, shapiro, sd, mdd, n_rep, ctrl_rgr):
    """Section 4: assumptions, effect sizes, pairwise summary, daily model, consistency, power, overview."""
    refs = []
    power_no = None
    sub = doc.subsection
    doc.section("İstatistiksel analizin özeti ve yorumu")
    doc.p("Bu bölümde kullanılan istatistiksel yöntemler, varsayımları ve sonuçların ne ölçüde güvenilir olduğu "
          f"özetlenmiştir. Tüm testlerde deney birimi kap, ana ölçüt {METRIC_NAME[P]} temelinde hesaplanan bağıl "
          "büyüme hızıdır.")

    # 4.1 assumptions
    sub("Varsayımların kontrolü")
    parts = []
    if "shapiro_p" in shapiro:
        normal = shapiro["shapiro_p"] >= ALPHA
        parts.append(f"Varyans analizi, grup içi sapmaların (artıkların) normal dağıldığını ve grupların "
                     f"varyanslarının benzer olduğunu varsayar. Shapiro–Wilk testine göre artıklar "
                     f"{'normal dağılıma uygundur' if normal else 'normal dağılmamaktadır'} "
                     f"(W = {num(shapiro['shapiro_W'], 3)}, {pval(shapiro['shapiro_p'])}).")
    if "levene_p" in shapiro:
        homo = shapiro["levene_p"] >= ALPHA
        parts.append(f"Levene testine göre grupların varyansları "
                     f"{'benzerdir' if homo else 'birbirinden farklıdır'} ({pval(shapiro['levene_p'])}).")
    if shapiro.get("shapiro_p", 1) < ALPHA or shapiro.get("levene_p", 1) < ALPHA:
        parts.append("Varsayımlardan biri sağlanmadığı için sonuçlar iki ek yolla sınanmıştır. Permütasyon testi "
                     "veriyi dağılım varsaymadan rastgele yeniden düzenleyerek p değerini hesaplar. Logaritma "
                     "dönüşümü ise çarpık dağılımları simetrikleştirir.")
        refs.append("freedman1983")
    else:
        parts.append("Varsayımlar sağlandığından varyans analizi sonuçları doğrudan kullanılabilir. Permütasyon "
                     "testi ve log(RGR) analizi ek güvence olarak raporlanmıştır.")
        refs.append("freedman1983")
    if "kruskal_p" in shapiro:
        parts.append(f"Altı grubu sıralara dayalı olarak karşılaştıran Kruskal–Wallis testi de gruplar arasında "
                     f"{'anlamlı fark bulmuştur' if shapiro['kruskal_p'] < ALPHA else 'anlamlı fark bulmamıştır'} "
                     f"({pval(shapiro['kruskal_p'])}). Bu test etkileşimi sınamadığından yalnızca destekleyici "
                     "bilgi olarak kullanılmıştır.")
    doc.p(" ".join(parts))

    # 4.2 effect sizes
    if anova is not None:
        sub("Etki büyüklükleri")
        refs.append("cohen1988")
        labels = {"C(plastic)": "Plastik türü", "C(detergent)": "Deterjan", "C(plastic):C(detergent)": "Etkileşim"}
        rows = [[labels[t], num(anova.loc[t, "partial_eta_sq"], 2), eta_label(anova.loc[t, "partial_eta_sq"]),
                 pct(anova.loc[t, "partial_eta_sq"] * 100, 0)] for t in labels]
        t_eta = doc.table("Kısmi η² değerleri ve yorumları (sınırlar: 0,01 küçük, 0,06 orta, 0,14 büyük).",
                          ["Etki", "Kısmi η²", "Büyüklük", "Kısmi açıklanan varyans"], rows, widths=[4, 2.5, 3, 3.5])
        big = max(labels, key=lambda t: anova.loc[t, "partial_eta_sq"])
        doc.p(f"Bir etkinin p değeri, etkinin rastlantıyla açıklanıp açıklanamayacağını gösterir. Etki büyüklüğü ise "
              f"etkinin ne kadar güçlü olduğunu ifade eder. Büyüme hızındaki değişkenliğin en büyük kısmını "
              f"{labels[big].lower()} açıklamaktadır (kısmi η² = {num(anova.loc[big, 'partial_eta_sq'], 2)}, "
              f"Tablo {t_eta}). Kısmi η², diğer etkiler hesaba katıldıktan sonra kalan değişkenliğin ilgili etkiyle "
              "açıklanan oranıdır ve Cohen (1988) sınırlarına göre yorumlanmıştır.")

    # 4.3 pairwise summary with compact letters
    if tk is not None:
        sub("Grupların karşılaştırılması")
        groups = [g for g in gt.index if g in set(tk["group1"]) | set(tk["group2"])]
        means = gt["mean"].to_dict()
        letters = compact_letters(groups, means, tk)
        rows = [[g, d.name(g), num(means[g]), letters[g]] for g in sorted(groups, key=lambda g: -means[g])]
        t_cld = doc.table("Gruplar büyüme hızına göre büyükten küçüğe sıralanmıştır. Aynı harfi paylaşan gruplar "
                          "arasında anlamlı fark yoktur (Tukey HSD, α = 0,05).",
                          ["Grup", "Uygulama", "Ortalama RGR (1/gün)", "Harf"], rows, widths=[1.6, 4.4, 4, 2])
        n_pairs = len(tk)
        n_sig = int((tk["p_adj"] < ALPHA).sum())
        distinct = len(set("".join(letters.values())))
        doc.p(f"Altı grup arasındaki {n_pairs} ikili karşılaştırmanın {n_sig} tanesinde anlamlı fark bulunmuştur. "
              f"Gruplar {distinct} farklı harfle gösterilen, birbirinden ayrışan kümelere ayrılmaktadır (Tablo "
              f"{t_cld}). Tukey testi çoklu karşılaştırmadan kaynaklanan yanlış pozitif riskini düzelttiğinden tek "
              "tek t testlerine göre daha temkinlidir.")

    # 4.4 daily data, mixed model
    if anova is not None and d.m["day"].nunique() >= 3:
        try:
            sl, status, inter = mixed_slopes(d, P)
        except Exception:
            sl = None
        if sl is not None and len(sl):
            sub("Günlük ölçümlerin analizi")
            rows = [[r.group, d.name(r.group), num(r.slope), f"{num(r.low)} – {num(r.high)}",
                     num(gt.loc[r.group, "mean"])] for r in sl.itertuples()]
            t_mix = doc.table("Karma etkili modelden tahmin edilen günlük büyüme hızı (ln ölçütün güne göre eğimi) "
                              "ve uç noktalardan hesaplanan RGR.",
                              ["Grup", "Uygulama", "Model eğimi (1/gün)", "%95 güven aralığı", "Uç nokta RGR"],
                              rows, widths=[1.5, 3.8, 3.2, 3.6, 2.6])
            diff = np.abs(sl.set_index("group")["slope"] - gt.loc[sl["group"], "mean"]).max()
            text = (f"Her kabın tüm günlerdeki ölçümleri, kaplar arası farkları hesaba katan karma etkili bir modelle "
                    f"de analiz edilmiştir. Modelin verdiği büyüme hızları uç nokta RGR değerleriyle en fazla "
                    f"{num(diff)} 1/gün farklıdır (Tablo {t_mix}). ")
            text += ("Bu yakınlık, büyümenin deney boyunca düzenli (üstel) seyrettiğini ve sonuçların yalnızca ilk ve "
                     "son günün ölçümüne bağlı olmadığını gösterir. " if diff < 0.02 else
                     "Bu fark, büyümenin deney boyunca düzenli seyretmediğini ve ara günlerin sonuçları etkilediğini "
                     "düşündürür. ")
            ps = [f"{pl} için {pval(p)}" for pl, (_, p) in inter.items() if np.isfinite(p)]
            if ps:
                text += ("Plastik ve deterjanın büyüme hızı üzerindeki ortak etkisine ait model terimleri: "
                         + ", ".join(ps) + ".")
            if not status["converged"]:
                text += " Model yakınsamadığından bu tablodaki değerler raporlanmamalıdır."
            elif status["boundary"]:
                text += (" Modelde kaplar arası rastgele farkların bir bileşeni sıfır tahmin edilmiştir. Bu durum "
                         "kaplar arasında ek değişkenlik olmadığını gösterir ve sabit etkilerin yorumunu etkilemez.")
            doc.p(text)

    # 4.5 consistency with secondary metrics
    sec = [m for m in d.secondary if d.full_factorial(m)]
    if anova is not None and sec:
        sub("Farklı ölçütlerle tutarlılık")
        labels = {"C(plastic)": "Plastik türü", "C(detergent)": "Deterjan", "C(plastic):C(detergent)": "Etkileşim"}
        tables = {m: LS.two_way_anova(d.g, d.response(m))[0] for m in sec}
        rows = []
        consistent = []
        for t in labels:
            row = [labels[t], pval(anova.loc[t, "PR(>F)"])]
            for m in sec:
                ps = tables[m].loc[t, "PR(>F)"]
                row.append(pval(ps))
                consistent.append((ps < ALPHA) == (anova.loc[t, "PR(>F)"] < ALPHA))
            rows.append(row)
        t_con = doc.table("Ana ölçüt ile diğer ölçütlerden hesaplanan büyüme hızlarının varyans analizi p değerleri.",
                          ["Etki", METRIC_NAME[P]] + [METRIC_NAME[m] for m in sec], rows)
        if all(consistent):
            doc.p(f"Farklı ölçütlerle hesaplanan büyüme hızları aynı sonuçlara ulaşmaktadır (Tablo {t_con}). Bulguların "
                  "ölçüm yönteminden bağımsız olması sonuçların güvenilirliğini artırır.")
        else:
            doc.p(f"Bazı etkiler için ölçütler farklı sonuç vermektedir (Tablo {t_con}). Ana ölçüt manuel sayıma "
                  "dayandığından esas alınmalı, diğer ölçütlerdeki farklılık ölçüm hatası veya daha düşük "
                  "duyarlılıkla açıklanabilir. Otomatik sayımın hata payı büyüdükçe grup içi değişkenlik artar ve "
                  "etkileri saptamak zorlaşır.")

    # 4.6 power and sample size
    if np.isfinite(mdd):
        power_no = sub("Örneklem büyüklüğü ve testlerin gücü")
        doc.p(f"Grup içi standart sapma {num(sd, 4)} 1/gün olarak tahmin edilmiştir. Grup başına {n_rep} kapla, %80 "
              f"olasılıkla saptanabilecek en küçük RGR farkı yaklaşık {num(mdd)} 1/gündür. Bu fark, kontrol grubunun "
              f"büyüme hızına oranla {pct(mdd / ctrl_rgr * 100, 0)} düzeyindedir. Bundan küçük etkiler bu deney "
              "düzeniyle güvenilir biçimde saptanamaz ve “anlamlı fark saptanmadı” sonucu etkinin olmadığı anlamına "
              "gelmez.")
        rows = []
        for a, b, label in _key_pairs(d):
            diff, p = pairwise(tk, a, b)
            if np.isfinite(p) and p >= ALPHA and np.isfinite(diff) and sd > 0:
                eff = abs(diff) / sd
                need = TTestIndPower().solve_power(effect_size=eff, power=0.8, alpha=ALPHA) if eff > 0.05 else np.inf
                rows.append([label, num(diff), "> 200" if not np.isfinite(need) or need > 200 else str(int(np.ceil(need)))])
        if rows:
            t_pow = doc.table("Anlamlı bulunmayan karşılaştırmalarda gözlenen farkı %80 güçle saptamak için gereken "
                              "kap sayısı (grup başına).", ["Karşılaştırma", "Gözlenen fark (1/gün)", "Gereken kap"],
                              rows, widths=[6, 4, 3])
            doc.p(f"Gözlenen farklar gerçekse, bunları saptamak için gereken kap sayısı Tablo {t_pow} içinde "
                  "verilmiştir. Gereken sayı çok büyükse fark biyolojik olarak da önemsiz olabilir.")

    # 4.7 overview of evidence
    sub("Sonuçların güvenilirlik özeti")
    rows = []
    if anova is not None:
        for t, label in [("C(detergent)", "Deterjanın büyümeye etkisi"), ("C(plastic)", "Plastik türünün etkisi"),
                         ("C(plastic):C(detergent)", "Plastik × deterjan etkileşimi")]:
            p = anova.loc[t, "PR(>F)"]
            rows.append([label, f"ANOVA {pval(p)}", evidence(p, [rob.loc[t, "p_permutation"],
                                                                 rob.loc[t, "p_anova_log_rgr"]])])
        if contrasts is not None:
            for pl in ["PLA", "PETG"]:
                key = f"{pl} x detergent (deviation from additivity)"
                if key in contrasts.index:
                    p = contrasts.loc[key, "p_value"]
                    rows.append([f"{pl} ile deterjanın toplamsallıktan sapması", f"kontrast {pval(p)}", evidence(p)])
    if tk is not None:
        for a, b, label in _key_pairs(d):
            _, p = pairwise(tk, a, b)
            if np.isfinite(p):
                rows.append([label, f"Tukey {pval(p)}", evidence(p)])
    if rows:
        t_ev = doc.table("Sonuçların kanıt düzeyi. Güçlü: p < 0,01 ve tüm testler uyumlu. Orta: 0,01 ≤ p < 0,05. "
                         "Zayıf: sonuç kullanılan teste göre değişiyor. Kanıt yok: p ≥ 0,05.",
                         ["Sonuç", "Dayanak", "Kanıt düzeyi"], rows, widths=[7, 4.5, 3.5])
        strong = [lower_first(r[0]) for r in rows if r[2] == "Güçlü"]
        moderate = [lower_first(r[0]) for r in rows if r[2] == "Orta"]
        weak = [lower_first(r[0]) for r in rows if r[2].startswith("Zayıf")]
        text = f"Tablo {t_ev} sonuçların ne kadar sağlam olduğunu özetlemektedir. "
        if strong:
            text += "Güçlü kanıt bulunan sonuçlar: " + ", ".join(strong) + ". "
        if moderate:
            text += ("Orta düzeyde kanıt bulunan sonuçlar: " + ", ".join(moderate)
                     + ". Bunlar anlamlı olmakla birlikte tekrar deneyle doğrulanmalıdır. ")
        if weak:
            text += "Kullanılan teste bağlı, zayıf kanıt bulunan sonuçlar: " + ", ".join(weak) + ". "
        text += ("Kanıt bulunmayan karşılaştırmalar etkinin olmadığını kanıtlamaz, yalnızca bu deneyde "
                 "saptanamadığını gösterir.")
        doc.p(text)
    return refs, power_no


# ------------------------------------------------------------------ text builders

def _tr_quality(text):
    rules = [
        ("Dish circle not found, default circle used", "Kap dairesi bulunamadı, varsayılan daire kullanıldı"),
        ("Dish size differs by more than 5 % from the median, check the dish circle",
         "Kap boyutu ortancadan %5'ten fazla farklı, kap dairesi kontrol edilmeli"),
        ("No plant tissue detected", "Bitki dokusu bulunamadı"),
        ("Groups not defined in config.json (left out of the factorial tests)",
         "Ayarlarda tanımlı olmayan gruplar (iki faktörlü testlere alınmadı)"),
        ("Dishes without a photo on every day", "Her gün fotoğrafı olmayan kaplar"),
        ("missing days", "eksik günler"), ("photos", "fotoğraf"),
        ("Manual counts are missing on the first or last day for", "İlk veya son gün manuel sayımı eksik kap sayısı:"),
        ("dishes, so growth rates use the automatic count", "– büyüme hızlarında otomatik sayım kullanıldı"),
        ("zero at end, RGR undefined", "son gün sıfır, RGR tanımsız"),
        ("zero at start, RGR undefined", "ilk gün sıfır, RGR tanımsız"),
        ("fewer than two values", "ikiden az değer"),
        ("did not converge; do not report its estimates", "için karma model yakınsamadı, tahminleri raporlanmamalı"),
        ("Mixed model for", ""),
        ("Area per frond differs between groups", "Yaprak başına alan gruplar arasında farklı"),
        ("Area per frond changes over the experiment", "Yaprak başına alan deney boyunca değişiyor"),
        ("the area-based count is not reliable, use manual counts",
         "alan tabanlı sayım güvenilir değil, manuel sayım kullanılmalı"),
        ("largest deviation", "en büyük sapma"),
    ]
    for en, tr in rules:
        text = text.replace(en, tr)
    for metric, name in METRIC_NAME.items():
        text = text.replace(metric, name)
    return text.strip()


def _key_pairs(d):
    c = d.cells
    pairs = [
        (c.get(("none", "no")), c.get(("PLA", "no")), "PLA etkisi (deterjansız)"),
        (c.get(("none", "yes")), c.get(("PLA", "yes")), "PLA etkisi (deterjanlı)"),
        (c.get(("none", "no")), c.get(("PETG", "no")), "PETG etkisi (deterjansız)"),
        (c.get(("none", "yes")), c.get(("PETG", "yes")), "PETG etkisi (deterjanlı)"),
        (c.get(("none", "no")), c.get(("none", "yes")), "Deterjan etkisi"),
        (c.get(("PLA", "no")), c.get(("PETG", "no")), "PETG − PLA (deterjansız)"),
        (c.get(("PLA", "yes")), c.get(("PETG", "yes")), "PETG − PLA (deterjanlı)"),
    ]
    return [p for p in pairs if p[0] and p[1]]


def _anova_sentence(anova, rob, shapiro, t_no):
    parts = []
    det_p = anova.loc["C(detergent)", "PR(>F)"]
    pla_p = anova.loc["C(plastic)", "PR(>F)"]
    int_p = anova.loc["C(plastic):C(detergent)", "PR(>F)"]
    parts.append(f"Varyans analizinde deterjanın büyüme hızı üzerindeki etkisi "
                 f"{'anlamlı bulunmuştur' if det_p < ALPHA else 'anlamlı bulunmamıştır'} ({pval(det_p)}, Tablo {t_no}).")
    parts.append(f"Plastik türünün ana etkisi {'anlamlıdır' if pla_p < ALPHA else 'anlamlı değildir'} ({pval(pla_p)}).")
    if int_p < ALPHA:
        parts.append(f"Plastik türü ile deterjan arasındaki etkileşim anlamlıdır ({pval(int_p)}). Plastiğin "
                     "etkisi deterjanın varlığına göre değiştiğinden ana etkiler tek başına "
                     "yorumlanmamalıdır.")
    else:
        parts.append(f"Plastik türü ile deterjan arasında anlamlı bir etkileşim bulunmamıştır ({pval(int_p)}).")
    agree = all(((rob.loc[t, "p_anova"] < ALPHA) == (rob.loc[t, "p_permutation"] < ALPHA)) and
                (not np.isfinite(rob.loc[t, "p_anova_log_rgr"]) or
                 (rob.loc[t, "p_anova"] < ALPHA) == (rob.loc[t, "p_anova_log_rgr"] < ALPHA))
                for t in rob.index)
    if shapiro.get("shapiro_p", 1) < ALPHA:
        parts.append("Artıkların normal dağılmadığı görüldüğünden sonuçlar permütasyon testi ve log(RGR) "
                     "analiziyle de sınanmıştır.")
    if agree:
        parts.append("Üç test aynı sonuca vardığından bulgular normallik varsayımına bağlı değildir.")
    else:
        diff = [{"C(plastic)": "plastik türü", "C(detergent)": "deterjan",
                 "C(plastic):C(detergent)": "etkileşim"}[t] for t in rob.index
                if (rob.loc[t, "p_anova"] < ALPHA) != (rob.loc[t, "p_permutation"] < ALPHA)
                or (np.isfinite(rob.loc[t, "p_anova_log_rgr"])
                    and (rob.loc[t, "p_anova"] < ALPHA) != (rob.loc[t, "p_anova_log_rgr"] < ALPHA))]
        parts.append(f"Testler {', '.join(diff)} için farklı sonuç verdiğinden bu etki temkinli yorumlanmalıdır.")
    return " ".join(parts)


def _additivity_sentence(r, contrasts, t_no, f_no):
    key = f"{r['plastic']} x detergent (deviation from additivity)"
    c = contrasts.loc[key] if contrasts is not None and key in contrasts.index else None
    p = c["p_value"] if c is not None else np.nan
    observed, expected = pct(r["observed"]), pct(r["expected"])
    s = (f"{r['name']} uygulamasında gözlenen inhibisyon {observed}, tekil etkilerin toplamından beklenen değer "
         f"{expected} olmuştur (Tablo {t_no}, Şekil {f_no}).")
    if not np.isfinite(p):
        return s + " Etkileşim kontrastı hesaplanamamıştır."
    caution = (" Elde edilen p değeri 0,05 sınırına yakın olduğundan bu sonuç bağımsız bir tekrar deneyle doğrulanmalıdır."
               if 0.01 <= p < ALPHA else "")
    if p < ALPHA and r["difference"] > 0:
        return s + (f" Etkileşim kontrastı anlamlıdır ({pval(p)}). Birlikte etki tekil etkilerin toplamından büyüktür "
                    "ve bu durum sinerjik bir etkileşime işaret eder." + caution)
    if p < ALPHA and r["difference"] < 0:
        return s + (f" Etkileşim kontrastı anlamlıdır ({pval(p)}). Birlikte etki tekil etkilerin toplamından küçüktür "
                    "ve bu durum antagonistik bir etkileşime işaret eder." + caution)
    tendency = "beklenenden yüksek" if r["difference"] > 0 else "beklenenden düşük"
    return s + (f" İnhibisyon {tendency} olmakla birlikte fark istatistiksel olarak anlamlı değildir ({pval(p)}). "
                "Bu nedenle birlikte etkinin toplamsal etkiden farklı olduğu söylenemez.")


def _yellowing(d, P):
    if "yellow_pct_of_plant" not in d.m.columns:
        return ["Renk ölçümü bulunmamaktadır."]
    last = d.m[d.m["day"] == d.days[-1]].dropna(subset=["yellow_pct_of_plant"])
    if last.empty:
        return ["Son gün için renk ölçümü bulunmamaktadır."]
    t = last.groupby("group")["yellow_pct_of_plant"].mean().reindex(d.group_table(P).index).dropna()
    most, least = t.idxmax(), t.idxmin()
    samples = [v.to_numpy() for _, v in last.groupby("group")["yellow_pct_of_plant"] if len(v) >= 2]
    kw = stats.kruskal(*samples).pvalue if len(samples) >= 2 and np.ptp(np.concatenate(samples)) > 0 else np.nan
    out = [f"Son günde sararmış doku oranı en yüksek {d.name(most)} grubunda ({pct(t[most])}), en düşük "
           f"{d.name(least)} grubunda ({pct(t[least])}) ölçülmüştür. Gruplar arasındaki fark "
           f"{'anlamlıdır' if np.isfinite(kw) and kw < ALPHA else 'anlamlı değildir'} (Kruskal–Wallis, {pval(kw)})."]
    col = d.response(P)
    merged = last.groupby("dish_id")["yellow_pct_of_plant"].mean().to_frame().join(d.g.set_index("dish_id")[col]).dropna()
    if len(merged) >= 6 and merged["yellow_pct_of_plant"].nunique() > 1:
        rho, p = stats.spearmanr(merged["yellow_pct_of_plant"], merged[col])
        if p < ALPHA and rho < 0:
            out.append(f"Sararma oranı yüksek olan kaplarda büyüme hızı daha düşüktür (Spearman ρ = {num(rho, 2)}, "
                       f"{pval(p)}). Sararma (kloroz) klorofil kaybının görünür bir işaretidir ve büyüme "
                       "inhibisyonuyla birlikte görülmesi bitkilerin strese girdiğini destekler. Renk oranı klorofil "
                       "miktarının doğrudan ölçüsü olmadığından bu ilişki bir gösterge olarak değerlendirilmelidir.")
        else:
            out.append(f"Kap düzeyinde sararma oranı ile büyüme hızı arasında anlamlı bir ilişki bulunmamıştır "
                       f"(Spearman ρ = {num(rho, 2)}, {pval(p)}).")
    return out


def _agreement_sentence(rows, t_no, f_no, cal_day):
    parts = []
    for label, col, a in rows:
        level = "yüksek" if a["lin_ccc"] >= CCC_HIGH else "yetersiz"
        direction = "fazla" if a["mean_bias_auto_minus_manual"] > 0 else "eksik"
        parts.append(f"{label} manuel sayımla {level} uyum göstermiştir (CCC = {num(a['lin_ccc'], 3)}). Yöntem, "
                     f"yaprak sayısını ortalama {num(abs(a['mean_bias_auto_minus_manual']), 1)} yaprak {direction} "
                     f"tahmin etmiştir ve ortalama yüzde hata {pct(a['mape_pct'])} olmuştur.")
    if cal_day is not None and any(c == "frond_count_from_area" for _, c, _ in rows):
        parts.append(f"Alan tabanlı sayım {cal_day}. günün manuel sayımlarıyla kalibre edildiğinden o gün "
                     "karşılaştırmaya alınmamıştır.")
    parts.append(f"Uyum için CCC ≥ {num(CCC_HIGH, 2)} ölçütü kullanılmıştır (Tablo {t_no}, Şekil {f_no}).")
    return " ".join(parts)


def _decision(ps, mdd):
    """Supported if at least one comparison is significant (all or some)."""
    ps = [p for p in ps if np.isfinite(p)]
    if not ps:
        return "Değerlendirilemedi"
    sig = sum(p < ALPHA for p in ps)
    if sig == len(ps):
        return "Desteklendi"
    if sig:
        return "Kısmen desteklendi"
    return "Desteklenmedi*"


def _pc_lookup(pc, a, b):
    if pc is None or not len(pc):
        return np.nan, np.nan
    row = pc[(pc["group1"] == a) & (pc["group2"] == b)]
    return (float(row["difference"].iloc[0]), float(row["p_holm"].iloc[0])) if len(row) else (np.nan, np.nan)


def _leachate_text(d, pc, cite, t_no):
    """Interpretation of the leachate controls for each plastic."""
    out = []
    for leach in d.leachate_groups():
        info = d.cfg["groups"][leach]
        plastic = info.get("plastic")
        det, combo = d.cells.get(("none", "yes")), d.cells.get((plastic, "yes"))
        _, p_det = _pc_lookup(pc, det, leach)
        _, p_combo = _pc_lookup(pc, combo, leach)
        if not (np.isfinite(p_det) and np.isfinite(p_combo)):
            continue
        name = d.name(leach)
        if p_det < ALPHA and p_combo >= ALPHA:
            text = (f"{name} grubunda büyüme yalnızca deterjan uygulanan gruptan anlamlı biçimde farklı "
                    f"({pval(p_det)}), parçacıklı gruptan ise farklı değildir ({pval(p_combo)}). {plastic}'nın deterjanlı "
                    "ortamdaki etkisinin büyük kısmı parçacıktan suya geçen maddelerden kaynaklanmaktadır.")
            if plastic == "PLA":
                text += (" Bu sonuç, Spirodela polyrhiza'da PLA'nın etkisinin büyük ölçüde süzüntüden geldiğini bildiren "
                         f"çalışmayla uyumludur {cite('wang2026')}.")
        elif p_det >= ALPHA and p_combo < ALPHA:
            text = (f"{name} grubunda büyüme deterjan grubundan farklı değildir ({pval(p_det)}), ancak parçacıklı gruptan "
                    f"anlamlı biçimde farklıdır ({pval(p_combo)}). {plastic}'nın etkisi suya geçen maddelerden çok "
                    "parçacığın kendisinden, örneğin köklere ve yapraklara tutunmasından kaynaklanmaktadır. Parçacıkların "
                    f"su bitkilerinin yüzeyine tutunduğu daha önce gösterilmiştir {cite('mateoscardenas2019', 'dovidat2020')}.")
        elif p_det < ALPHA and p_combo < ALPHA:
            text = (f"{name} grubu hem deterjan grubundan ({pval(p_det)}) hem de parçacıklı gruptan ({pval(p_combo)}) "
                    f"anlamlı biçimde farklıdır. {plastic}'nın etkisinin bir kısmı suya geçen maddelerden, bir kısmı "
                    "parçacığın kendisinden kaynaklanıyor olabilir.")
        else:
            text = (f"{name} grubu ne deterjan grubundan ({pval(p_det)}) ne de parçacıklı gruptan ({pval(p_combo)}) "
                    "anlamlı biçimde ayrışmaktadır. Bu nedenle etkinin kaynağı konusunda bir sonuca varılamamaktadır.")
        text = text.replace(f"{name} grubunda", f"{name} grubunda (Tablo {t_no})", 1)
        out.append(text.replace("PETG'nın", "PETG'nin"))
    return out


def _root_text(d, rt, root_tk, P, tk, cite, t_no, f_no):
    longest, shortest = rt["mean"].idxmax(), rt["mean"].idxmin()
    text = (f"Kök uzunluğu en fazla {d.name(longest)} ({num(rt.loc[longest, 'mean'], 1)} mm), en az "
            f"{d.name(shortest)} ({num(rt.loc[shortest, 'mean'], 1)} mm) grubunda ölçülmüştür (Tablo {t_no}, Şekil "
            f"{f_no}). ")
    text += anova_phrase(d, "root_length_mm") + " "
    c = d.cells
    extra = []
    for plastic in ["PLA", "PETG"]:
        a, b = c.get(("none", "no")), c.get((plastic, "no"))
        _, p_root = pairwise(root_tk, a, b)
        _, p_rgr = pairwise(tk, a, b)
        if np.isfinite(p_root) and np.isfinite(p_rgr) and p_root < ALPHA <= p_rgr:
            extra.append(plastic)
    if extra:
        text += (f"{' ve '.join(extra)} tek başına büyüme hızını anlamlı biçimde değiştirmezken kök uzunluğunu "
                 "azaltmıştır. Kök uzunluğunun büyüme hızından daha duyarlı bir ölçüt olduğu daha önce de "
                 f"bildirilmiştir {cite('kalcikova2017', 'rozman2022a')}.")
    return text.strip()


def _chemistry_text(d, cols, cite, t_no, f_no):
    out = []
    ref = f"(Tablo {t_no}" + (f", Şekil {f_no})" if f_no else ")")
    c = d.cells
    det, pla_det = c.get(("none", "yes")), c.get(("PLA", "yes"))
    if "pH_change" in cols:
        tkp = LS.tukey(d.g, "pH_change")
        diff, p = pairwise(tkp, det, pla_det)
        tab = group_values(d, "pH_change")
        text = f"Deney süresince pH değişimi gruplara göre farklılık göstermiştir {ref}. " + anova_phrase(d, "pH_change")
        if np.isfinite(p) and p < ALPHA and diff < 0:
            text += (f" PLA + deterjan grubunda pH, yalnızca deterjan uygulanan gruba göre anlamlı biçimde daha fazla "
                     f"düşmüştür ({num(tab.loc[pla_det, 'mean'], 2)} ve {num(tab.loc[det, 'mean'], 2)}, {pval(p)}). Bu "
                     "bulgu, PLA'nın bazik ortamda hidrolize olarak ortama asidik ürünler (laktik asit) bıraktığı "
                     f"hipoteziyle uyumludur {cite('vaid2021')}.")
        elif np.isfinite(p):
            text += (f" PLA + deterjan ile yalnızca deterjan grubu arasında pH değişimi bakımından anlamlı fark "
                     f"bulunmamıştır ({pval(p)}). pH verisi PLA'nın hidrolizine ilişkin bir kanıt sunmamaktadır.")
        out.append(text.strip())
    return out


def _hypotheses(d, tk, anova, rob, h7, mdd, root_tk=None, pc=None):
    c = d.cells
    rows = []

    def comp(pairs, table, unit, label):
        ps, txt = [], []
        for a, b, lab in pairs:
            diff, p = pairwise(table, a, b)
            if np.isfinite(p):
                ps.append(p)
                txt.append(f"{label}{lab}: {num(diff, 3 if unit == '1/gün' else 1)} {unit}, {pval(p)}")
        return ps, txt

    defs = [
        ("H1: PLA parçacıkları büyüme hızını veya kök uzunluğunu etkiler.",
         [(c.get(("none", "no")), c.get(("PLA", "no")), "deterjansız"),
          (c.get(("none", "yes")), c.get(("PLA", "yes")), "deterjanlı")]),
        ("H2: PETG parçacıkları büyüme hızını veya kök uzunluğunu etkiler.",
         [(c.get(("none", "no")), c.get(("PETG", "no")), "deterjansız"),
          (c.get(("none", "yes")), c.get(("PETG", "yes")), "deterjanlı")]),
        ("H3: Deterjan büyüme hızını azaltır.",
         [(c.get(("none", "no")), c.get(("none", "yes")), "kontrole göre")]),
        ("H5: PLA ve PETG'nin etkileri farklıdır.",
         [(c.get(("PLA", "no")), c.get(("PETG", "no")), "deterjansız"),
          (c.get(("PLA", "yes")), c.get(("PETG", "yes")), "deterjanlı")]),
    ]
    for hyp, pairs in defs:
        pairs = [p for p in pairs if p[0] and p[1]]
        ps, txt = comp(pairs, tk, "1/gün", "RGR, ")
        test = "Tukey HSD"
        if root_tk is not None and not hyp.startswith("H3"):
            ps_r, txt_r = comp(pairs, root_tk, "mm", "kök, ")
            ps, txt = ps + ps_r, txt + txt_r
            test = "Tukey HSD (RGR ve kök uzunluğu)"
        rows.append([hyp, test, "; ".join(txt) if txt else "–", _decision(ps, mdd)])
    if anova is not None:
        p_a = anova.loc["C(plastic):C(detergent)", "PR(>F)"]
        p_p = rob.loc["C(plastic):C(detergent)", "p_permutation"]
        if p_a < ALPHA and p_p < ALPHA:
            dec = "Desteklendi"
        elif p_a < ALPHA or p_p < ALPHA:
            dec = "Belirsiz (teste bağlı)"
        else:
            dec = "Desteklenmedi*"
        rows.insert(3, ["H4: Plastik ile deterjanın birlikte etkisi, tekil etkilerin toplamından farklıdır.",
                        "ANOVA etkileşim terimi, permütasyon", f"ANOVA {pval(p_a)}; permütasyon {pval(p_p)}", dec])
    else:
        rows.insert(3, ["H4: Plastik ile deterjanın birlikte etkisi, tekil etkilerin toplamından farklıdır.",
                        "ANOVA etkileşim terimi", "Deney düzeni eksik olduğundan test edilemedi", "Değerlendirilemedi"])
    leach = d.leachate_groups()
    if leach and pc is not None and len(pc):
        det = c.get(("none", "yes"))
        ps, txt = [], []
        for g in leach:
            diff, p = _pc_lookup(pc, det, g)
            if np.isfinite(p):
                ps.append(p)
                txt.append(f"{d.name(g)} − deterjan: {num(diff)} 1/gün, {pval(p)}")
        rows.append(["H6: Deterjanlı ortamda plastiğin etkisinin bir kısmı suya geçen maddelerden kaynaklanır.",
                     "Planlanmış karşılaştırmalar (Holm)", "; ".join(txt) if txt else "–", _decision(ps, mdd)])
    else:
        rows.append(["H6: Deterjanlı ortamda plastiğin etkisinin bir kısmı suya geçen maddelerden kaynaklanır.",
                     "Planlanmış karşılaştırmalar", "Süzüntü grubu yok", "Değerlendirilemedi"])
    if h7:
        txt = "; ".join(f"{lab}: CCC = {num(a['lin_ccc'], 3)}, MAPE {pct(a['mape_pct'])}" for lab, _, a in h7)
        ok = [a["lin_ccc"] >= CCC_HIGH for _, _, a in h7]
        dec = "Desteklendi" if all(ok) else ("Kısmen desteklendi" if any(ok) else "Desteklenmedi")
        rows.append(["H7: Görüntü analizi manuel sayımla yüksek uyum gösterir.", f"Lin CCC (ölçüt ≥ {num(CCC_HIGH, 2)})",
                     txt, dec])
    else:
        rows.append(["H7: Görüntü analizi manuel sayımla yüksek uyum gösterir.", "Lin CCC",
                     "Manuel sayım yok", "Değerlendirilemedi"])
    rows.append(["* Anlamlı fark saptanmadı. Küçük etkiler bu örneklem büyüklüğüyle dışlanamaz.", "", "", ""])
    return rows


def _interpretation(d, P, gt, tk, anova, rob, contrasts, add_rows, ctrl_rgr, cite=None):
    cite = cite or Citations()
    c = d.cells
    out = []
    if d.synthetic:
        out.append("Bu bölümdeki yorumlar yapay veriye uygulanmış örnek yorumlardır. Gerçek deney verisiyle "
                   "üretilen raporda aynı kurallar gerçek ölçümlere uygulanır.")
    det_diff, det_p = pairwise(tk, c.get(("none", "no")), c.get(("none", "yes")))
    if np.isfinite(det_p):
        inh = gt.loc[c[("none", "yes")], "inhibition"]
        if det_p < ALPHA and det_diff < 0:
            out.append(f"Deterjan tek başına Lemna minor büyümesini belirgin biçimde yavaşlatmıştır (inhibisyon "
                       f"{pct(inh)}). Olası bir açıklama, deterjandaki yüzey aktif maddelerin bitki hücre zarlarıyla "
                       "etkileşerek zarın geçirgenliğini artırması ve besin alımını bozmasıdır. Lineer alkilbenzen sülfonatın "
                       "Lemna minor'da büyüme hızını ve pigmentleri azaltıp zar hasarı oluşturduğu, sodyum dodesil sülfatın "
                       f"stres enzimlerini değiştirdiği bildirilmiştir {cite('masoudian2022', 'forni2012')}. Sonuç, "
                       "evsel atık sularla ulaşan deterjanların sucul bitkiler için tek başına da bir stres kaynağı "
                       "olabileceğine işaret etmektedir.")
        elif det_p < ALPHA:
            out.append(f"Deterjan uygulanan kaplarda büyüme kontrolden hızlı olmuştur ({pct(inh)} inhibisyon). "
                       "Düşük konsantrasyonlarda bazı deterjan bileşenlerinin (ör. fosfat içeren yardımcı maddeler) "
                       "besin etkisi yapabileceği akla gelmektedir. Bu sonuç kendiliğinden olumlu bir etki olarak "
                       "yorumlanmamalı ve deterjanın bileşimiyle birlikte değerlendirilmelidir.")
        else:
            out.append("Deterjan tek başına büyüme hızını anlamlı biçimde değiştirmemiştir. Uygulanan "
                       "konsantrasyonun ölçülebilir bir etki için düşük kalmış olması mümkündür.")

    for plastic in ["PLA", "PETG"]:
        diff, p = pairwise(tk, c.get(("none", "no")), c.get((plastic, "no")))
        if not np.isfinite(p):
            continue
        inh = gt.loc[c[(plastic, "no")], "inhibition"]
        if p < ALPHA and diff < 0:
            out.append(f"{plastic} parçacıkları tek başına büyümeyi anlamlı biçimde yavaşlatmıştır ({pct(inh)} "
                       "inhibisyon). Parçacıkların yaprak ve köklere fiziksel olarak tutunması, ışığı kısmen "
                       "engellemesi veya filamentten suya geçen katkı maddeleri olası nedenler arasındadır. PET şişe "
                       "parçacıklarının ve başka polimerlerin su mercimeğinde büyümeyi azalttığı da gösterilmiştir "
                       f"{cite('cui2023', 'ceschin2023')}.")
        elif p < ALPHA:
            out.append(f"{plastic} uygulanan kaplarda büyüme kontrolden hızlı olmuştur. Bu farkın nedeni ayrıca "
                       "araştırılmalıdır.")
        else:
            out.append(f"{plastic} parçacıkları tek başına büyüme hızında anlamlı bir değişikliğe yol açmamıştır "
                       f"(gözlenen inhibisyon {pct(inh)}). Uygulanan konsantrasyonda ve yedi günlük sürede {plastic} "
                       "parçacıklarının güçlü bir etkisi olmadığı düşünülebilir. Polietilen parçacıkların Lemna minor ve "
                       "Spirodela polyrhiza'da yedi gün içinde büyümeyi değiştirmediği önceki çalışmalarla uyumludur "
                       f"{cite('mateoscardenas2019', 'dovidat2020')}. Ancak küçük etkiler bu örneklem büyüklüğüyle "
                       "dışlanamaz ve etkilerin uzun maruziyette ortaya çıkabildiği bildirilmiştir "
                       f"{cite('rozman2022a', 'cui2023')}.")

    if anova is not None:
        p_int = anova.loc["C(plastic):C(detergent)", "PR(>F)"]
        p_perm = rob.loc["C(plastic):C(detergent)", "p_permutation"]
        if p_int < ALPHA and p_perm < ALPHA:
            out.append("Plastik türü ile deterjan arasındaki etkileşim, iki kirleticinin bir arada bulunmasının "
                       "tek başlarına oluşturdukları etkilerden farklı bir sonuç doğurduğunu göstermektedir. Çevresel risk "
                       "değerlendirmesinde kirleticilerin yalnızca ayrı ayrı incelenmesi bu nedenle yetersiz "
                       "kalabilir.")
    for r in add_rows:
        key = f"{r['plastic']} x detergent (deviation from additivity)"
        p = contrasts.loc[key, "p_value"] if contrasts is not None and key in contrasts.index else np.nan
        if not np.isfinite(p):
            continue
        if p < ALPHA and r["difference"] > 0:
            if r["plastic"] == "PLA":
                out.append("PLA ile deterjanın birlikte etkisi tekil etkilerin toplamını aşmıştır. PLA, ester "
                           "bağları hidrolizle parçalanabilen, biyolojik olarak bozunabilen bir polyesterdir. "
                           "Deterjanların çoğunun bazik olması PLA yüzeyinin hidrolizini hızlandırarak ortama "
                           "laktik asit ve küçük PLA parçaları (oligomerler) geçmesine yol açabilir. Yüzey aktif "
                           "maddeler ayrıca parçacıkların suda dağılmasını ve filamentteki katkı maddelerinin suya "
                           "geçişini kolaylaştırabilir. PLA'nın bazik ortamda daha hızlı hidrolize olduğu "
                           f"{cite('vaid2021')} ve yüzey aktif maddelerin mikroplastiklerin yüzey özelliklerini değiştirdiği "
                           f"{cite('shen2021', 'xue2022')} bilinmektedir. Biyobozunur mikroplastiklerin başka kirleticilerle "
                           f"sinerjik etkiler oluşturabildiği de bildirilmiştir {cite('ali2024')}. Bu açıklamalar birer "
                           "hipotezdir ve süzüntü grupları ile pH ölçümleri bu hipotezi sınamak için kullanılmıştır.")
            else:
                out.append(f"{r['plastic']} ile deterjanın birlikte etkisi tekil etkilerin toplamını aşmıştır. "
                           "Yüzey aktif maddelerin parçacıkların suda dağılmasını ve katkı maddelerinin suya geçişini "
                           "artırması olası bir açıklamadır ve deneysel olarak sınanmalıdır.")
        elif p < ALPHA and r["difference"] < 0:
            out.append(f"{r['plastic']} ile deterjanın birlikte etkisi tekil etkilerin toplamından küçük kalmıştır. "
                       f"Yüzey aktif maddelerin {r['plastic']} parçacıklarının yüzeyine tutunarak sudaki serbest "
                       "deterjan miktarını azaltması olası bir açıklamadır. Bu hipotez deterjanın parçacıklara "
                       "tutunma oranı ölçülerek sınanabilir.")
        else:
            word = "yüksek" if r["difference"] > 0 else "düşük"
            out.append(f"{r['name']} grubunda inhibisyon toplamsal beklentiden {word} olsa da bu fark anlamlı "
                       f"değildir ({pval(p)}). Bu nedenle {r['plastic']} için sinerjik veya antagonistik bir "
                       "etkileşimden söz edilemez.")
    if P == "frond_count_manual" and np.isfinite(ctrl_rgr) and ctrl_rgr < OECD_MIN_CONTROL_RGR:
        out.append("Kontrol bitkileri OECD 221'in öngördüğü hızda büyümediğinden gruplar arasındaki farklar "
                   "olduğundan küçük görünmüş olabilir.")
    return out


def _recommendations(d, add_rows, contrasts, mdd, ctrl_rgr):
    rec = []
    sig = []
    for r in add_rows:
        key = f"{r['plastic']} x detergent (deviation from additivity)"
        if contrasts is not None and key in contrasts.index and contrasts.loc[key, "p_value"] < ALPHA:
            sig.append(r["plastic"])
    if sig:
        rec.append("Etkileşim bulunan gruplarda pH her gün ölçülerek kimyasal bir "
                   "değişimin büyüme farkına eşlik edip etmediği incelenmelidir.")
        rec.append("Parçacıkların deterjanlı suda bekletilip süzülmesiyle elde edilen suda yetiştirme (süzüntü "
                   "kontrolü) yapılarak etkinin parçacığın kendisinden mi suya geçen maddelerden mi kaynaklandığı "
                   "ayrılmalıdır.")
    rec.append("Deterjan için birkaç konsantrasyon denenerek etkinin ve etkileşimin konsantrasyona bağlı olup "
               "olmadığı belirlenmelidir.")
    if np.isfinite(mdd) and np.isfinite(ctrl_rgr) and mdd / ctrl_rgr > 0.10:
        rec.append(f"Saptanabilir en küçük fark, kontrol büyüme hızına oranla {pct(mdd / ctrl_rgr * 100, 0)} "
                   "düzeyindedir. "
                   "Daha küçük etkileri saptamak için kap sayısı artırılmalı veya deney tekrarlanmalıdır.")
    rec.append("Sınırda anlamlı (p değeri 0,05'e yakın) sonuçlar bağımsız bir tekrar deneyle doğrulanmadan kesin "
               "bulgu olarak sunulmamalıdır.")
    rec.append("Literatürde PLA, PETG ve deterjanı birlikte inceleyen çalışmalar ayrıntılı bir taramayla "
               "araştırılmalı ve bulgular bu çalışmalarla karşılaştırılmalıdır.")
    return rec
