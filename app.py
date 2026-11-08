"""Web interface for the Lemna minor image analysis.

Start with:  streamlit run app.py
"""

import os
import sys
from pathlib import Path

# ==============================================================================
# 🚀 KRİTİK DÜZELTME: 'lemna' klasörünü Python'un arama yoluna (sys.path) ekler
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
# ==============================================================================

import hashlib
import io
import json
import time
import zipfile

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lemna.auth import login_gate, logout
from lemna.config import load_config, save_config, with_workspace_config
from lemna.imaging import analyse_image, read_image, resize_to_work, find_dish, roi_mask, gray_world
from lemna.naming import IMAGE_EXTENSIONS, list_images, make_name, parse_name
from lemna.measure import read_manual_counts
from lemna.pipeline import ensure_endpoints, ensure_sheet, list_runs, merge_save, run
from lemna.plots import CHART_KINDS, make_chart
from lemna.report import build_report, compact_letters

# CONFIG_PATH tanımı (HERE yukarı taşındığı için doğrudan kullanılabilir)
CONFIG_PATH = HERE / "config.json"

st.set_page_config(page_title="Lemna Analiz", page_icon="🌿", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 3.6rem; max-width: 1450px;}
    div[data-testid="stMetricValue"] {font-size: 1.35rem;}
    div[data-testid="stMetricLabel"] p {font-size: 0.82rem; color: #5f6b5f;}
    .small-note {color: #6b7280; font-size: 0.85rem;}
    .step-title {font-weight: 600; font-size: 1.02rem; margin-bottom: 0.1rem;}
    .step-detail {color: #5f6b5f; font-size: 0.86rem;}
    .hero {background: linear-gradient(90deg, #e8f5e9 0%, #f1f8e9 100%); border-radius: 12px;
           padding: 1.1rem 1.4rem; margin-bottom: 1rem; border: 1px solid #c8e6c9;}
    .hero h2 {margin: 0 0 0.2rem 0; color: #1b5e20;}
    .hero p {margin: 0; color: #33691e;}
    </style>
    """,
    unsafe_allow_html=True,
)

USER = login_gate(st)

GROUP_COLORS = {"G1": "#4d4d4d", "G2": "#1f77b4", "G3": "#2ca02c", "G4": "#d62728", "G5": "#98df8a", "G6": "#ff9896",
                "G7": "#ff7f0e", "G8": "#9467bd"}

METRIC_LABELS = {
    "plant_area_mm2": "Bitki alanı (mm²)",
    "plant_area_px": "Bitki alanı (piksel)",
    "green_pct_of_roi": "Yeşil alan (% kap)",
    "plant_pct_of_roi": "Bitki örtüsü (% kap)",
    "frond_count_auto": "Yaprak sayısı (otomatik, watershed)",
    "frond_count_from_area": "Yaprak sayısı (alan tabanlı)",
    "frond_count_manual": "Yaprak sayısı (manuel)",
    "yellow_pct_of_plant": "Sararma (% bitki alanı)",
    "brown_pct_of_plant": "Kahverengileşme (% bitki alanı)",
    "exg_index": "Fazla yeşil indeksi (ExG)",
    "mean_hue": "Ortalama ton (H)",
    "root_length_mm": "Kök uzunluğu (mm)",
    "fresh_weight_mg": "Yaş ağırlık (mg)",
    "pH_end": "pH (son gün)",
    "pH_change": "pH değişimi",
}


# ---------------------------------------------------------------- helpers

def workspace_paths(ws):
    ws = Path(ws)
    return {
        "ws": ws,
        "photos": ws / "photos",
        "results": ws / "results",
        "manual": ws / "manual_frond_counts.csv",
        "log": ws / "daily_log.csv",
    }


def get_cfg():
    """Settings from config.json, reloaded whenever the file changes."""
    stamp = mtime(CONFIG_PATH)
    if "cfg" not in st.session_state or st.session_state.get("cfg_mtime") != stamp:
        st.session_state.cfg = load_config(CONFIG_PATH)
        st.session_state.cfg_mtime = stamp
    return st.session_state.cfg


def clamp(value, low, high):
    return min(max(value, low), high)


def bump(name):
    """Change the key of an editor so that it shows the saved file instead of earlier edits."""
    st.session_state[name] = st.session_state.get(name, 0) + 1


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False, max_entries=200, hash_funcs={bytes: lambda b: len(b)})
def upload_thumbnail(file_id, data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
    return None if img is None else bgr_to_rgb(resize_to_work(img, 300)[0])


def mtime(path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False, max_entries=40)
def load_image_cached(path, mtime, max_side):
    """Photo decoded and reduced to the working size, as in the batch analysis."""
    return resize_to_work(read_image(path, max_side), max_side)[0]


@st.cache_data(show_spinner=False, max_entries=600)
def thumbnail(path, mtime, size):
    img = resize_to_work(read_image(path, size), size)[0]
    return bgr_to_rgb(img)


@st.cache_data(show_spinner=False, max_entries=60)
def analyse_cached(path, mtime, cfg_json):
    """Single-photo analysis, repeated only when the photo or the settings change."""
    c = json.loads(cfg_json)
    img = load_image_cached(path, mtime, c["work_max_side_px"])
    res, overlay, masks = analyse_image(img, c)
    return res, bgr_to_rgb(overlay), masks["plant"]


def cfg_key(c):
    return json.dumps(c, sort_keys=True)


def folder_stamp(folder):
    """Latest modification time of a folder and its subfolders (changes when files are added)."""
    stamp = mtime(folder)
    for root, dirs, _ in os.walk(folder):
        for d in dirs:
            stamp = max(stamp, mtime(Path(root) / d))
    return stamp


def photo_inventory(photo_dir):
    return _photo_inventory(str(photo_dir), folder_stamp(photo_dir))


@st.cache_data(show_spinner=False, max_entries=20)
def _photo_inventory(photo_dir, dir_mtime):
    rows = []
    for f in list_images(photo_dir):
        p = parse_name(f)
        if p:
            rows.append({"group": p[0], "replicate": p[1], "day": p[2], "file": f.name, "path": str(f)})
    return pd.DataFrame(rows, columns=["group", "replicate", "day", "file", "path"])


@st.cache_data(show_spinner=False, max_entries=20)
def count_images(photo_dir, dir_mtime):
    return len(list_images(photo_dir))


def dish_order(cfg, n_reps):
    return [(g, r) for g in cfg["groups"] for r in range(1, n_reps + 1)]


def load_results(results_dir):
    results_dir = Path(results_dir)
    return _load_results(str(results_dir), mtime(results_dir / "results.xlsx"), mtime(results_dir / "measurements.csv"))


@st.cache_data(show_spinner="⏳ Sonuçlar yükleniyor…", max_entries=10)
def _load_results(results_dir, xlsx_mtime, csv_mtime):
    results_dir = Path(results_dir)
    out = {}
    for name in ["measurements", "growth_per_dish", "growth_per_group", "daily_summary"]:
        p = results_dir / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
    rp = results_dir / "statistics_report.txt"
    if rp.exists():
        out["report"] = rp.read_text(encoding="utf-8")
    xp = results_dir / "results.xlsx"
    if xp.exists():
        try:
            out["sheets"] = pd.read_excel(xp, sheet_name=None)
        except Exception:
            out["sheets"] = {}
    info = results_dir / "run_info.json"
    if info.exists():
        try:
            out["info"] = json.loads(info.read_text(encoding="utf-8"))
        except ValueError:
            out["info"] = {}
    return out


def zip_folder(folder):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in Path(folder).rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(folder))
    return buf.getvalue()


def fmt_time(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60} dk {seconds % 60:02d} sn" if seconds >= 60 else f"{seconds} sn"


_WARNING_TR = [
    (r"Dish circle not found, default circle used \((\d+) photos\)", r"Kap dairesi bulunamadı, varsayılan daire kullanıldı (\1 fotoğraf)"),
    (r"Dish size differs by more than 5 % from the median, check the dish circle \((\d+) photos\)",
     r"Kap boyutu ortancadan %5'ten fazla farklı, kap dairesini kontrol edin (\1 fotoğraf)"),
    (r"No plant tissue detected \((\d+) photos\)", r"Bitki dokusu bulunamadı (\1 fotoğraf)"),
    (r"Groups not defined in config.json \(left out of the factorial tests\)",
     "config.json'da tanımlı olmayan gruplar (iki faktörlü testlere alınmadı)"),
    (r"Dishes without a photo on every day", "Her gün fotoğrafı olmayan kaplar"),
    (r"missing days", "eksik günler"),
    (r"Manual counts are missing on the first or last day for (\d+) dishes, so growth rates use the automatic count",
     r"\1 kapta ilk veya son gün manuel sayımı eksik; büyüme hızlarında otomatik sayım kullanıldı"),
    (r"Area per frond differs between groups \(largest deviation ([\d.]+) %, Kruskal-Wallis p = ([^)]+)\); "
     r"the area-based count is not reliable, use manual counts",
     r"Yaprak başına alan gruplar arasında farklı (en büyük sapma %\1, Kruskal-Wallis p = \2); "
     r"alan tabanlı sayım güvenilir değil, manuel sayımı kullanın"),
    (r"Area per frond changes over the experiment \(([^,]+) %, p = ([^)]+)\); "
     r"the area-based count is not reliable, use manual counts",
     r"Yaprak başına alan deney boyunca değişiyor (%\1, p = \2); alan tabanlı sayım güvenilir değil, "
     r"manuel sayımı kullanın"),
    (r"Mixed model for (\S+) did not converge; do not report its estimates",
     r"\1 için karma model yakınsamadı; tahminleri raporlanmamalıdır"),
    (r"zero at end, RGR undefined", "son gün sıfır, RGR tanımsız"),
    (r"zero at start, RGR undefined", "ilk gün sıfır, RGR tanımsız"),
    (r"fewer than two values", "ikiden az değer"),
]


TERM_TR = {"C(plastic)": "Plastik türü", "C(detergent)": "Deterjan",
           "C(plastic):C(detergent)": "Plastik × deterjan etkileşimi", "C(series)": "Deney serisi"}
STRUCTURE_TR = {"random intercept and slope": "rastgele kesişim ve eğim", "random intercept": "rastgele kesişim"}


def tr_warning(text):
    """Turkish wording of the warnings produced by the analysis code."""
    import re
    for pattern, repl in _WARNING_TR:
        text = re.sub(pattern, repl, text)
    return text


LABEL_TR = [("Control", "Kontrol"), ("leachate", "süzüntüsü"), ("Detergent", "Deterjan"), ("detergent", "deterjan")]


def group_label(cfg, g):
    """Group name shown in the interface (Turkish wording of the standard English labels)."""
    label = cfg["groups"].get(g, {}).get("label", g)
    for en, tr in LABEL_TR:
        label = label.replace(en, tr)
    return label


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "–"
    return "p < 0,001" if p < 0.001 else f"p = {p:.3f}".replace(".", ",")


def render_summary(cfg, R, df, growth, info):
    """Plain-language overview of one analysis run."""
    sheets = R.get("sheets", {})
    metrics = info.get("metrics") or []
    primary = "frond_count_manual" if "frond_count_manual" in metrics else (metrics[0] if metrics else None)
    col = f"rgr_{primary}" if primary else None
    k = st.columns(5)
    k[0].metric("Fotoğraf", len(df))
    k[1].metric("Kap (deney birimi)", df["dish_id"].nunique())
    k[2].metric("Günler", f"{df['day'].min()}–{df['day'].max()}")
    k[3].metric("Deney serisi", int(growth["series"].nunique()) if "series" in growth.columns else 1)
    if col is None or col not in growth.columns:
        st.info("Büyüme hızı hesaplanamadı (en az iki gün gerekir).")
        return
    ctrl = growth.loc[growth["group"] == cfg["control_group"], col].mean()
    if np.isfinite(ctrl) and ctrl > 0:
        k[4].metric("Kontrolde ikiye katlanma", f"{np.log(2) / ctrl:.2f} gün",
                    help="OECD 221 geçerlilik ölçütü: 2,5 günden kısa olmalıdır.")

    t = growth.groupby("group")[col].agg(["count", "mean", "std"])
    order = sorted(t.index, key=lambda g: (int(g[1:]) if g[1:].isdigit() else 999, g))
    t = t.reindex(order)
    tk = sheets.get(f"tukey_{primary}")
    letters = compact_letters(order, t["mean"].to_dict(), tk.rename(columns={"p-adj": "p_adj"})) \
        if tk is not None and "p_adj" in tk.columns else {g: "" for g in order}
    t["inhibition"] = (ctrl - t["mean"]) / ctrl * 100 if ctrl else np.nan

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(f"**Gruplara göre büyüme hızı** ({METRIC_LABELS.get(primary, primary).lower()} temelinde)")
        st.dataframe(pd.DataFrame({
            "Grup": order,
            "Uygulama": [group_label(cfg, g) for g in order],
            "n": t["count"].astype(int).values,
            "RGR (1/gün)": [f"{m:.3f} ± {s:.3f}" for m, s in zip(t["mean"], t["std"])],
            "İnhibisyon": [f"%{v:.1f}" for v in t["inhibition"]],
            "Harf": [letters.get(g, "") for g in order],
        }), hide_index=True, width="stretch")
        st.caption("Aynı harfi paylaşan gruplar arasında anlamlı fark yoktur (Tukey HSD, α = 0,05).")
    with right:
        fig = go.Figure(go.Bar(x=order, y=t["mean"], error_y=dict(type="data", array=t["std"]),
                               marker_color=[GROUP_COLORS.get(g, "#888888") for g in order],
                               text=[letters.get(g, "") for g in order], textposition="outside"))
        fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="RGR (1/gün)",
                          showlegend=False)
        st.plotly_chart(fig, width="stretch")

    findings = []
    anova = sheets.get(f"anova_{primary}")
    if anova is not None and "PR(>F)" in anova.columns:
        a = anova.set_index("source")
        for term, text in [("C(detergent)", "Deterjanın etkisi"), ("C(plastic)", "Plastik türünün etkisi"),
                           ("C(plastic):C(detergent)", "Plastik × deterjan etkileşimi (birlikte etki ≠ toplam)")]:
            if term in a.index:
                p = a.loc[term, "PR(>F)"]
                findings.append(f"{'✅' if p < 0.05 else '➖'} {text}: "
                                f"{'anlamlı' if p < 0.05 else 'anlamlı değil'} ({fmt_p(p)})")
    pl = sheets.get(f"planned_{primary}")
    if pl is not None and len(pl):
        for r in pl.itertuples():
            findings.append(f"{'✅' if r.p_holm < 0.05 else '➖'} {group_label(cfg, r.group2)} ile "
                            f"{group_label(cfg, r.group1)}: {'farklı' if r.p_holm < 0.05 else 'fark yok'} "
                            f"({fmt_p(r.p_holm)}, Holm)")
    agr = sheets.get("agreement")
    if agr is not None and len(agr):
        best = agr.sort_values("lin_ccc").iloc[-1]
        findings.append(f"{'✅' if best.lin_ccc >= 0.9 else '⚠️'} Otomatik ölçüm ile manuel sayımın uyumu: "
                        f"CCC = {best.lin_ccc:.3f} ({'yüksek' if best.lin_ccc >= 0.9 else 'yetersiz'})")
    if findings:
        with st.container(border=True):
            st.markdown("**Başlıca bulgular**")
            st.markdown("\n".join(f"- {f}" for f in findings))
            st.caption("p < 0,05 anlamlı kabul edilmiştir. Anlamlı bulunmayan bir fark, etkinin bu deneyle "
                       "saptanamadığını gösterir.")
    c1, c2 = st.columns(2)
    c1.button("📈 Ayrıntılı istatistiğe git", on_click=go_to, args=(PG_ADV, ADV_STATS), width="stretch")
    c2.button("📄 Word raporu oluştur", on_click=lambda: st.session_state.update(res_view="📄 Rapor ve indirme"),
              width="stretch", type="primary")


# ---------------------------------------------------------------- sidebar

cfg = get_cfg()

with st.sidebar:
    st.markdown("## 🌿 Lemna Analiz")
    st.caption("PLA / PETG parçacıkları ve deterjanın *Lemna minor* büyümesine etkisi")
    if USER:
        u1, u2 = st.columns([3, 2])
        u1.caption(f"👤 {USER}")
        if u2.button("Çıkış", key="logout", width="stretch"):
            logout(st)
            st.rerun()

    ws_choice = st.radio(
        "Çalışma klasörü",
        ["data", "demo", "Diğer…"], horizontal=True,
        help="data: gerçek deney fotoğrafları. demo: yazılımı denemek için yapay fotoğraflar. "
             "Diğer…: ör. ön deney için ayrı bir klasör.",
    )
    if ws_choice == "Diğer…":
        if USER:
            # On a shared server only folders inside the application folder can be used
            name = st.text_input("Klasör adı", value="on_deney",
                                 help="Uygulama klasörünün içinde bu adla bir çalışma klasörü kullanılır.")
            name = "".join(ch for ch in name.strip() if ch.isalnum() or ch in "_-") or "on_deney"
            ws_choice = name
        else:
            ws_choice = st.text_input("Klasör yolu", value=str(HERE / "data"))
    P = workspace_paths(HERE / ws_choice if not Path(ws_choice).is_absolute() else ws_choice)
    P["photos"].mkdir(parents=True, exist_ok=True)
    P["results"].mkdir(parents=True, exist_ok=True)
    cfg = with_workspace_config(cfg, P["ws"] / "config.json")

    inv = photo_inventory(P["photos"])
    n_files = count_images(str(P["photos"]), folder_stamp(P["photos"]))
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Fotoğraf", len(inv))
        c2.metric("Kap", inv[["group", "replicate"]].drop_duplicates().shape[0] if len(inv) else 0)
        c3.metric("Gün", inv["day"].nunique() if len(inv) else 0)
        if n_files - len(inv):
            st.caption(f"⚠️ Adı tanınmayan dosya: {n_files - len(inv)}")
        if P["manual"].exists():
            man_sb = read_manual_counts(P["manual"])
            filled = int(man_sb["frond_count_manual"].notna().sum())
            st.progress(min(1.0, filled / max(len(man_sb), 1)), text=f"Manuel sayım: {filled}/{len(man_sb)}")
    if ws_choice == "demo" and len(inv) == 0:
        if st.button("Demo fotoğraflarını oluştur", width="stretch"):
            from lemna.demo import make_demo
            with st.spinner("⏳ 768 yapay fotoğraf oluşturuluyor (birkaç dakika)…", show_time=True):
                make_demo(P["ws"])
            st.rerun()
    with st.expander("Ayar bilgisi"):
        st.caption(f"Ayar dosyası: `{CONFIG_PATH.name}`" + (" + bu klasörün `config.json` dosyası"
                                                           if (P["ws"] / "config.json").exists() else ""))
        st.caption("Gruplar: " + ", ".join(f"{g} ({group_label(cfg, g)})" for g in cfg["groups"]))
        d_ = cfg.get("design", {})
        st.caption(f"Plan: grup başına {d_.get('replicates_per_group')} kap"
                   + (f", seri başına {d_.get('replicates_per_series')} kap" if d_.get("replicates_per_series") else "")
                   + f", gün {d_.get('first_day')}–{d_.get('last_day')}")

MAIN_PAGES = ["🏠 Başlangıç", "📤 Fotoğraflar", "🎯 Kalibrasyon", "✍️ Sayım", "📝 Kayıtlar", "⚙️ Analiz",
              "📊 Sonuçlar", "🧪 Gelişmiş", "❓ Yardım"]
(PG_HOME, PG_UPLOAD, PG_CAL, PG_COUNT, PG_REC, PG_RUN, PG_RES, PG_ADV, PG_HELP) = MAIN_PAGES
ADV_PAGES = ["📈 Ayrıntılı istatistik", "🔀 Etkileşim ve renk", "🎯 Ölçüm doğruluğu", "🖼️ Grafik oluşturucu",
             "📋 Tablolar", "🔍 Görüntü inceleme", "🔬 Parçacık boyutu", "🕘 Geçmiş"]
(ADV_STATS, ADV_INTER, ADV_AGREE, ADV_CHARTS, ADV_TABLES, ADV_IMAGES, ADV_PARTICLES, ADV_HISTORY) = ADV_PAGES


def go_to(target, sub=None):
    st.session_state.page = target
    if sub:
        st.session_state.adv = sub


# Only the selected page is computed on each interaction, which keeps the interface fast.
if st.session_state.get("page") not in MAIN_PAGES:
    st.session_state.page = st.session_state.get("last_page") if st.session_state.get("last_page") in MAIN_PAGES \
        else PG_HOME
page = st.segmented_control("Bölüm", MAIN_PAGES, key="page", label_visibility="collapsed")
st.session_state.last_page = page
adv = None
if page == PG_ADV:
    if st.session_state.get("adv") not in ADV_PAGES:
        st.session_state.adv = st.session_state.get("last_adv") if st.session_state.get("last_adv") in ADV_PAGES \
            else ADV_PAGES[0]
    st.caption("Ayrıntılı hesaplamalar, ek analizler ve yardımcı araçlar")
    adv = st.segmented_control("Gelişmiş", ADV_PAGES, key="adv", label_visibility="collapsed")
    st.session_state.last_adv = adv
st.divider()

# ---------------------------------------------------------------- home

if page == PG_HOME:
    st.markdown("<div class='hero'><h2>🌿 Lemna Analiz</h2><p>Fotoğraftan büyüme hızına, istatistikten Word "
                "raporuna kadar su mercimeği deneyinin tüm adımları. Aşağıdaki listeden sıradaki adıma geçin."
                "</p></div>", unsafe_allow_html=True)
    inv_h = photo_inventory(P["photos"])
    dishes_h = inv_h[["group", "replicate"]].drop_duplicates() if len(inv_h) else pd.DataFrame()
    man_h = read_manual_counts(P["manual"]) if P["manual"].exists() else pd.DataFrame(columns=["frond_count_manual"])
    counted = man_h.dropna(subset=["frond_count_manual"]) if len(man_h) else man_h
    endpoint_ok = 0
    if len(inv_h) and len(counted):
        first, last = inv_h["day"].min(), inv_h["day"].max()
        have = {(g, r, d) for g, r, d in zip(counted["group"], counted["replicate"], counted["day"])}
        endpoint_ok = sum(((g, r, first) in have) and ((g, r, last) in have)
                          for g, r in zip(dishes_h["group"], dishes_h["replicate"]))
    log_n = 0
    if P["log"].exists():
        lg = pd.read_csv(P["log"], dtype=str)
        log_n = int(pd.to_numeric(lg.get("pH", pd.Series(dtype=str)), errors="coerce").notna().sum())
    ep_path = P["ws"] / "endpoints.csv"
    root_n = 0
    if ep_path.exists():
        ep = pd.read_csv(ep_path, dtype=str)
        root_n = int(pd.to_numeric(ep.get("root_length_mm", pd.Series(dtype=str)), errors="coerce").notna().sum())
    runs_h = list_runs(P["results"])
    latest = runs_h[0] if runs_h else None
    has_report = bool(latest and list((P["results"] / "runs" / latest).glob("rapor_*.docx")))
    stale = bool(latest and len(inv_h)) and folder_stamp(P["photos"]) > mtime(P["results"] / "runs" / latest)
    n_d = len(dishes_h)
    steps = [
        ("Fotoğrafları yükle", len(inv_h) > 0, f"{len(inv_h)} fotoğraf, {n_d} kap, "
         f"{inv_h['day'].nunique() if len(inv_h) else 0} gün", PG_UPLOAD),
        ("Kalibrasyonu kontrol et", None, "Kap dairesi ve yeşil/sarı eşikler birkaç fotoğrafta doğru mu? "
         f"Ayarlar son kaydedildi: {pd.Timestamp(mtime(CONFIG_PATH), unit='s'):%d.%m.%Y %H:%M}", PG_CAL),
        ("Manuel sayım (kör sayım önerilir)", n_d > 0 and endpoint_ok == n_d,
         f"İlk ve son gün sayımı tamamlanan kap: {endpoint_ok}/{n_d}", PG_COUNT),
        ("Günlük kayıt ve kök uzunluğu", log_n > 0 and root_n > 0,
         f"pH girilen satır: {log_n} · kök uzunluğu girilen kap: {root_n}", PG_REC),
        ("Analizi çalıştır", bool(latest) and not stale,
         (f"Son analiz: {latest}" + (" · fotoğraflar sonradan değişti, yeniden çalıştırın" if stale else ""))
         if latest else "Henüz analiz yok", PG_RUN),
        ("Sonuçları incele ve Word raporu oluştur", has_report,
         "Rapor hazır" if has_report else "Sonuçlar → 📄 Rapor ve indirme", PG_RES),
    ]
    nxt = next((i for i, (_, ok, _, _) in enumerate(steps) if ok is False), None)
    for i, (title, ok, detail, target) in enumerate(steps):
        with st.container(border=True):
            c1, c2, c3 = st.columns([0.6, 6, 1.4], vertical_alignment="center")
            c1.markdown(f"### {'✅' if ok else ('⏳' if ok is False else 'ℹ️')}")
            c2.markdown(f"<div class='step-title'>{i + 1}. {title}"
                        f"{' &nbsp;·&nbsp; <span style=\'color:#2e7d32\'>sıradaki adım</span>' if i == nxt else ''}"
                        f"</div><div class='step-detail'>{detail}</div>", unsafe_allow_html=True)
            c3.button("Git →", key=f"home_go_{i}", on_click=go_to, args=(target,), width="stretch",
                      type="primary" if i == nxt else "secondary")
    with st.expander("Gelişmiş araçlar"):
        st.markdown("Ayrıntılı istatistik, etkileşim ve renk analizleri, ölçüm doğruluğu, grafik oluşturucu, "
                    "tablolar, görüntü inceleme, parçacık boyutu ve analiz geçmişi **🧪 Gelişmiş** bölümündedir.")
        st.button("🧪 Gelişmiş bölüme git", on_click=go_to, args=(PG_ADV,))

# ---------------------------------------------------------------- 1. upload

if page == PG_UPLOAD:
    st.header("Fotoğrafları yükle")
    st.markdown(
        "Fotoğraflar `GRUP_TEKRAR_GunGÜN.jpg` biçiminde adlandırılır, örneğin **G3_2_Gun5.jpg** "
        "(PLA grubu, 2. kap, 5. gün). Telefonun verdiği adlarla (IMG_1234.jpg) yüklenen dosyalar "
        "aşağıdaki tabloda gruba, kaba ve güne atanabilir."
    )
    mode = st.radio(
        "Yükleme yöntemi",
        ["Dosya adları zaten doğru", "Bir günün fotoğraflarını sırayla ata", "Tek tek ata"],
        horizontal=True,
    )
    uploads = st.file_uploader(
        "Fotoğrafları sürükleyip bırakın",
        type=[e.strip(".") for e in IMAGE_EXTENSIONS],
        accept_multiple_files=True,
    )

    if uploads:
        uploads = sorted(uploads, key=lambda u: u.name)
        if mode == "Dosya adları zaten doğru":
            rows = []
            for u in uploads:
                p = parse_name(u.name)
                rows.append({
                    "yüklenen dosya": u.name,
                    "grup": p[0] if p else "",
                    "kap": p[1] if p else None,
                    "gün": p[2] if p else None,
                })
            table = pd.DataFrame(rows)
        elif mode == "Bir günün fotoğraflarını sırayla ata":
            design = cfg.get("design", {})
            per_series = int(design.get("replicates_per_series") or design.get("replicates_per_group", 5))
            c1, c2, c3, c4 = st.columns(4)
            day = c1.number_input("Gün", 0, 60, 0, 1, key="seq_day")
            n_reps = c2.number_input("Bu yüklemede gruptaki kap sayısı", 1, 40, per_series, 1, key="seq_reps")
            start = c3.number_input("İlk kap numarası", 1, 200, 1, 1, key="seq_start",
                                    help="Seri halinde çalışılıyorsa 2. seri için 5, 3. seri için 9 (4'er kap).")
            c4.markdown(
                f"<div class='small-note'>Fotoğraflar dosya adına göre sıralanır ve "
                f"{list(cfg['groups'])[0]}_{int(start)}, {list(cfg['groups'])[0]}_{int(start) + 1} … sırasıyla "
                "atanır. Çekimi her gün aynı sırayla yapın.</div>",
                unsafe_allow_html=True,
            )
            order = [(g, r) for g in cfg["groups"] for r in range(int(start), int(start) + int(n_reps))]
            rows = []
            for i, u in enumerate(uploads):
                g, r = order[i] if i < len(order) else ("", None)
                rows.append({"yüklenen dosya": u.name, "grup": g, "kap": r, "gün": int(day)})
            table = pd.DataFrame(rows)
            if len(uploads) != len(order):
                st.warning(f"{len(uploads)} fotoğraf yüklendi, beklenen kap sayısı {len(order)}. Tabloyu kontrol edin.")
        else:
            table = pd.DataFrame([{"yüklenen dosya": u.name, "grup": "", "kap": None, "gün": None} for u in uploads])

        table = st.data_editor(
            table,
            column_config={
                "yüklenen dosya": st.column_config.TextColumn(disabled=True),
                "grup": st.column_config.SelectboxColumn(options=list(cfg["groups"]), required=True),
                "kap": st.column_config.NumberColumn(min_value=1, max_value=50, step=1),
                "gün": st.column_config.NumberColumn(min_value=0, max_value=60, step=1),
            },
            width="stretch",
            hide_index=True,
            key="upload_editor_" + hashlib.md5(
                (mode + str(st.session_state.get("seq_day")) + str(st.session_state.get("seq_reps"))
                 + "".join(u.file_id for u in uploads)).encode()).hexdigest(),
        )

        if st.toggle("Önizlemeyi göster (ilk 12 fotoğraf)", value=False):
            cols = st.columns(6)
            with st.spinner("⏳ Önizlemeler hazırlanıyor…", show_time=True):
                for i, u in enumerate(uploads[:12]):
                    small = upload_thumbnail(u.file_id, u.getvalue())
                    if small is not None:
                        row = table.iloc[i]
                        cap = f"{row['grup']}_{row['kap']}_Gun{row['gün']}" if row["grup"] else u.name
                        cols[i % 6].image(small, caption=cap, width="stretch")

        overwrite = st.checkbox("Aynı adlı dosyaların üzerine yaz", value=False)
        if st.button("💾 Fotoğrafları kaydet", type="primary"):
            saved, problems, seen = 0, [], {}
            existing = {}
            for f in list_images(P["photos"]):
                k = parse_name(f)
                if k:
                    existing.setdefault(k, []).append(f)
            for u, (_, row) in zip(uploads, table.iterrows()):
                if pd.isna(row["grup"]) or not row["grup"] or pd.isna(row["kap"]) or pd.isna(row["gün"]):
                    problems.append(f"{u.name}: grup, kap veya gün eksik")
                    continue
                key = (str(row["grup"]), int(row["kap"]), int(row["gün"]))
                if key in seen:
                    problems.append(f"{u.name}: {seen[key]} ile aynı kap ve güne atanmış, kaydedilmedi")
                    continue
                seen[key] = u.name
                ext = Path(u.name).suffix.lower()
                target = P["photos"] / make_name(*key, ext)
                old = existing.get(key, [])
                if old and not overwrite:
                    problems.append(f"{u.name}: bu kap ve gün için zaten fotoğraf var ({old[0].name})")
                    continue
                for f in old:  # replace, also when the old file had another extension
                    f.unlink()
                target.write_bytes(u.getvalue())
                saved += 1
            if saved:
                st.success(f"{saved} fotoğraf `{P['photos']}` klasörüne kaydedildi.")
            for p in problems:
                st.warning(p)
            _photo_inventory.clear()

    st.subheader("Fotoğraf envanteri")
    inv = photo_inventory(P["photos"])
    if inv.empty:
        st.info("Henüz doğru adlandırılmış fotoğraf yok.")
    else:
        inv["kap_id"] = inv["group"] + "_" + inv["replicate"].astype(str)
        pivot = inv.pivot_table(index="kap_id", columns="day", values="file", aggfunc="count").fillna(0)
        pivot = pivot.reindex(sorted(pivot.index, key=lambda s: (int(s.split("_")[0][1:]), int(s.split("_")[1]))))
        fig = px.imshow(
            pivot.values, x=[f"Gün {d}" for d in pivot.columns], y=pivot.index,
            color_continuous_scale=[[0, "#fee2e2"], [0.5, "#bbf7d0"], [1, "#f59e0b"]], zmin=0, zmax=2,
            aspect="auto",
        )
        fig.update_traces(text=np.where(pivot.values == 0, "✗", np.where(pivot.values > 1, "2+", "✓")),
                          texttemplate="%{text}", hovertemplate="%{y}, %{x}<extra></extra>")
        fig.update_layout(coloraxis_showscale=False, height=max(300, 22 * len(pivot)), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        missing = int((pivot.values == 0).sum())
        if missing:
            st.warning(f"{missing} kap-gün fotoğrafı eksik (kırmızı hücreler).")
        unknown = sorted(set(inv["group"]) - set(cfg["groups"]))
        if unknown:
            st.error(f"Ayarlarda tanımlı olmayan gruplar: {unknown}")
        bad = [f.name for f in list_images(P["photos"]) if not parse_name(f)]
        if bad:
            st.error("Adı tanınmayan dosyalar (analize alınmaz): " + ", ".join(bad))

# ---------------------------------------------------------------- 2. calibration

if page == PG_CAL:
    st.header("Kalibrasyon")
    st.caption("Birkaç farklı gün ve gruptan fotoğraf seçip kap dairesinin ve yeşil doku eşiklerinin doğru olduğunu "
               "kontrol edin. Kontrol görüntüsünde yeşil = sağlıklı doku, sarı = sararmış doku, kahverengi = "
               "kahverengileşmiş doku, mor çizgi = yaprak sınırıdır. Ayarları kaydettikten sonra analizi yeniden "
               "çalıştırın.")
    inv = photo_inventory(P["photos"])
    if inv.empty:
        st.info("Kalibrasyon için önce fotoğraf yükleyin.")
    else:
        work = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
        work["hsv"] = {k: list(v) for k, v in cfg["hsv"].items()}

        left, right = st.columns([1, 2.2])
        with left:
            pick = st.selectbox("Fotoğraf", inv["file"].tolist(), index=len(inv) - 1)
            path = inv.loc[inv["file"] == pick, "path"].iloc[0]

            with st.expander("Kap (ilgi alanı)", expanded=True):
                modes = {"auto": "Otomatik daire bulma", "fixed": "Sabit daire", "full": "Tüm görüntü"}
                work["dish"]["mode"] = st.radio("Yöntem", list(modes), format_func=modes.get,
                                                index=list(modes).index(cfg["dish"]["mode"]))
                work["dish"]["inner_diameter_mm"] = st.number_input(
                    "Kabın iç çapı (mm)", 10.0, 300.0, clamp(float(cfg["dish"]["inner_diameter_mm"] or 90.0), 10.0, 300.0), 0.5,
                    help="Piksel alanını mm²'ye çevirmek için kullanılır.")
                work["dish"]["margin_fraction"] = st.slider(
                    "Kenar payı (yarıçap oranı)", 0.50, 1.00, clamp(float(cfg["dish"]["margin_fraction"]), 0.5, 1.0), 0.01,
                    help="Kap kenarındaki yansımaları dışarıda bırakır.")
                if work["dish"]["mode"] == "fixed":
                    work["dish"]["fixed_cx"] = st.slider("Merkez x", 0.0, 1.0, clamp(float(cfg["dish"]["fixed_cx"]), 0.0, 1.0), 0.005)
                    work["dish"]["fixed_cy"] = st.slider("Merkez y", 0.0, 1.0, clamp(float(cfg["dish"]["fixed_cy"]), 0.0, 1.0), 0.005)
                    work["dish"]["fixed_r"] = st.slider("Yarıçap", 0.05, 0.75, clamp(float(cfg["dish"]["fixed_r"]), 0.05, 0.75), 0.005)

            cal_adv = st.toggle("⚙️ Gelişmiş parametreler", value=False, key="cal_adv",
                                help="Sarı ve kahverengi doku eşikleri, gürültü temizleme, yaprak ayırma, çalışma "
                                     "çözünürlüğü ve beyaz dengesi. Çoğu deneyde varsayılan değerler yeterlidir.")
            colour_sets = [("green", "Yeşil doku")]
            if cal_adv:
                colour_sets += [("yellow", "Sarı doku"), ("brown", "Kahverengi doku")]
            for key, title in colour_sets:
                with st.expander(f"{title} HSV aralığı", expanded=(key == "green")):
                    lo, hi = cfg["hsv"][f"{key}_low"], cfg["hsv"][f"{key}_high"]
                    h = st.slider("Ton (H)", 0, 179, (int(lo[0]), int(hi[0])), key=f"{key}_h")
                    s = st.slider("Doygunluk (S)", 0, 255, (int(lo[1]), int(hi[1])), key=f"{key}_s")
                    v = st.slider("Parlaklık (V)", 0, 255, (int(lo[2]), int(hi[2])), key=f"{key}_v")
                    work["hsv"][f"{key}_low"] = [h[0], s[0], v[0]]
                    work["hsv"][f"{key}_high"] = [h[1], s[1], v[1]]

            if cal_adv:
                work["white_balance"] = "grayworld" if st.checkbox(
                    "Beyaz dengesi düzelt (gray-world)", cfg.get("white_balance") == "grayworld") else "none"
                with st.expander("Gürültü temizleme ve yaprak sayımı"):
                    work["morphology"]["open_kernel_px"] = st.slider("Açma çekirdeği (px)", 1, 15, clamp(int(cfg["morphology"]["open_kernel_px"]), 1, 15))
                    work["morphology"]["close_kernel_px"] = st.slider("Kapama çekirdeği (px)", 1, 21, clamp(int(cfg["morphology"]["close_kernel_px"]), 1, 21))
                    work["morphology"]["min_object_area_px"] = st.slider("En küçük nesne (px)", 0, 5000, clamp(int(cfg["morphology"]["min_object_area_px"]), 0, 5000))
                    work["morphology"]["refine_edges"] = st.checkbox(
                        "Kenar piksellerini düzelt (önerilir)", bool(cfg["morphology"].get("refine_edges", True)),
                        help="Yaprak kenarındaki yarı yaprak/yarı zemin pikselleri %50 kuralıyla sınıflandırır; "
                             "JPEG renk taşmasından kaynaklanan alan fazlalığını giderir.")
                    work["frond_count"]["seed_window"] = st.slider(
                        "Yaprak ayırma penceresi", 0.6, 2.5, clamp(float(cfg["frond_count"].get("seed_window", 1.2)), 0.6, 2.5), 0.05,
                        help="Küçük değer bitişik yaprakları daha çok ayırır, büyük değer birleştirir. "
                             "Manuel sayımla karşılaştırarak ayarlayın.")
                    work["frond_count"]["peak_ratio"] = st.slider(
                        "Tepe eşiği", 0.2, 1.0, clamp(float(cfg["frond_count"]["peak_ratio"]), 0.2, 1.0), 0.05,
                        help="Yaprak merkezi sayılacak en düşük kalınlık (tipik yaprak yarı genişliğine oranla).")
                    work["frond_count"]["min_frond_area_px"] = st.slider(
                        "En küçük yaprak alanı (px)", 10, 5000, clamp(int(cfg["frond_count"]["min_frond_area_px"]), 10, 5000))
                work["work_max_side_px"] = st.select_slider(
                    "Çalışma çözünürlüğü (uzun kenar, px)",
                    sorted({800, 1200, 1600, 2000, 2400, 3000, int(cfg["work_max_side_px"])}),
                    value=int(cfg["work_max_side_px"]))

            b1, b2 = st.columns(2)
            if b1.button("💾 Ayarları kaydet", type="primary", width="stretch"):
                st.session_state.cfg = work
                save_config(work, CONFIG_PATH)
                st.success("config.json kaydedildi.")
            if b2.button("↺ Varsayılanlar", width="stretch"):
                st.session_state.cfg = load_config(None)
                save_config(st.session_state.cfg, CONFIG_PATH)
                for k in list(st.session_state.keys()):
                    if k.endswith(("_h", "_s", "_v")):
                        del st.session_state[k]
                st.rerun()

        with right:
            with st.spinner("⏳ Fotoğraf işleniyor…", show_time=True):
                res, overlay, plant_mask = analyse_cached(path, mtime(path), cfg_key(work))
                work_img = load_image_cached(path, mtime(path), work["work_max_side_px"])
            m1, m2, m3, m4, m5 = st.columns(5)
            area_txt = f"{res['plant_area_mm2']:.1f} mm²" if np.isfinite(res["plant_area_mm2"]) else f"{res['plant_area_px']} px"
            m1.metric("Bitki alanı", area_txt)
            m2.metric("Yeşil alan", f"{res['green_pct_of_roi']:.2f} %")
            m3.metric("Sararma", f"{res['yellow_pct_of_plant']:.1f} %" if np.isfinite(res["yellow_pct_of_plant"]) else "–")
            m4.metric("Yaprak (oto.)", res["frond_count_auto"])
            m5.metric("Kap bulma", res["dish_method"])
            view = st.radio("Görünüm", ["Yan yana", "Kontrol görüntüsü", "Maske", "Orijinal"], horizontal=True)
            if view == "Yan yana":
                c1, c2 = st.columns(2)
                c1.image(bgr_to_rgb(work_img), caption="Orijinal", width="stretch")
                c2.image(overlay, caption="Kontrol", width="stretch")
            elif view == "Kontrol görüntüsü":
                st.image(overlay, width="stretch")
            elif view == "Maske":
                st.image(plant_mask, width="stretch", clamp=True)
            else:
                st.image(bgr_to_rgb(work_img), width="stretch")

            if st.toggle("Kap içindeki renk dağılımını göster (eşik seçimine yardımcı)", value=False):
                im2 = gray_world(work_img) if work["white_balance"] == "grayworld" else work_img
                circle, _ = find_dish(im2, work["dish"])
                roi = roi_mask(im2.shape, circle, work["dish"]["margin_fraction"]) > 0
                hsv = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
                sel = hsv[roi]
                sel = sel[sel[:, 1] > 30]  # ignore grey background
                if len(sel):
                    sample = sel[np.random.default_rng(0).choice(len(sel), min(20000, len(sel)), replace=False)]
                    fig = px.density_heatmap(x=sample[:, 0], y=sample[:, 1], nbinsx=90, nbinsy=64,
                                             labels={"x": "Ton (H)", "y": "Doygunluk (S)"},
                                             color_continuous_scale="Viridis")
                    for key, col in [("green", "lime"), ("yellow", "yellow"), ("brown", "orange")]:
                        lo, hi = work["hsv"][f"{key}_low"], work["hsv"][f"{key}_high"]
                        fig.add_shape(type="rect", x0=lo[0], x1=hi[0], y0=lo[1], y1=hi[1],
                                      line=dict(color=col, width=2))
                    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, width="stretch")
                    st.caption("Kutular seçili eşik aralıklarını gösterir. Bitki pikselleri kutuların içinde kalmalıdır.")

        st.subheader("Toplu kontrol")
        n_check = st.slider("Rastgele seçilecek fotoğraf sayısı", 2, 12, 6)
        if st.button("Seçili ayarlarla rastgele fotoğrafları dene"):
            rng = np.random.default_rng()
            idx = rng.choice(len(inv), min(n_check, len(inv)), replace=False)
            cols = st.columns(3)
            with st.spinner(f"⏳ {len(idx)} fotoğraf işleniyor…", show_time=True):
                for j, i in enumerate(idx):
                    p = inv.iloc[i]
                    r, ov, _ = analyse_cached(p["path"], mtime(p["path"]), cfg_key(work))
                    cols[j % 3].image(ov, caption=f"{p['file']} – {r['frond_count_auto']} yaprak", width="stretch")

# ---------------------------------------------------------------- 3. manual counts

if page == PG_COUNT:
    st.header("Manuel yaprak sayımı")
    st.markdown(
        "OECD 221'e göre en az başlangıç ve son gün sayımı gerekir. Manuel sayımlar büyüme hızında "
        "otomatik sayımın yerine kullanılır, ilk günün sayımları alan tabanlı sayımı kalibre eder ve "
        "diğer günlerin sayımları otomatik yöntemin doğruluğunu (H6) test eder."
    )
    inv = photo_inventory(P["photos"])
    ensure_sheet(P["ws"], "manual", cfg, zip(inv["group"], inv["replicate"], inv["day"]))
    man = read_manual_counts(P["manual"])
    man_loaded = man.copy()
    ver = st.session_state.get("manual_ver", 0)

    def save_manual(frame):
        merge_save(P["manual"], ["group", "replicate", "day"], man_loaded, frame)
        bump("manual_ver")

    helper, table_col = st.columns([1.3, 1])
    with helper:
        st.subheader("Sayım yardımcısı")
        blind = st.toggle("🙈 Kör sayım modu", value=False,
                          help="Fotoğraflar rastgele sırayla ve yalnızca bir kodla gösterilir. Sayan kişi hangi grubu "
                               "saydığını bilmez, bu da sayımdaki bilinçsiz yanlılığı önler.")
        if inv.empty:
            st.info("Fotoğraf yok.")
        elif blind:
            all_days = sorted(inv["day"].unique())
            default_days = [d for d in (0, 3, 5, 7) if d in all_days] or all_days[:1]
            bdays = st.multiselect("Sayılacak günler", all_days, default=default_days, key="blind_days")
            if "blind_seed" not in st.session_state:
                st.session_state.blind_seed = int(np.random.default_rng().integers(1, 10**6))
            items = inv[inv["day"].isin(bdays)].sort_values(["group", "replicate", "day"]).reset_index(drop=True)
            order = np.random.default_rng(st.session_state.blind_seed).permutation(len(items))
            counted = {(g, r, d) for g, r, d, v in zip(man["group"], man["replicate"], man["day"],
                                                       man["frond_count_manual"]) if pd.notna(v)}
            skipped = st.session_state.setdefault("blind_skipped", [])
            queue = [i for i in order if (items.loc[i, "group"], items.loc[i, "replicate"], items.loc[i, "day"])
                     not in counted]
            queue = [i for i in queue if i not in skipped] + [i for i in queue if i in skipped]
            done = len(items) - len(queue)
            st.progress(done / max(len(items), 1), text=f"Sayılan: {done} / {len(items)}")
            if not queue:
                st.success("Seçilen günlerdeki tüm fotoğraflar sayıldı.")
            else:
                i = int(queue[0])
                row = items.loc[i]
                code = f"K{int(np.where(order == i)[0][0]) + 1:03d}"
                st.markdown(f"**Fotoğraf kodu: {code}**")
                st.image(thumbnail(row["path"], mtime(row["path"]), 1600), width="stretch")
                val = st.number_input("Sayılan yaprak", min_value=0, max_value=10000, value=None, step=1,
                                      placeholder="sayıyı girin", key=f"blind_{code}_{ver}")
                b1, b2 = st.columns(2)
                if b1.button("💾 Kaydet ve sonraki", type="primary", disabled=val is None, width="stretch"):
                    mask = (man["group"] == row["group"]) & (man["replicate"] == row["replicate"]) & \
                           (man["day"] == row["day"])
                    if mask.any():
                        man.loc[mask, "frond_count_manual"] = val
                    else:
                        man = pd.concat([man, pd.DataFrame([{"group": row["group"], "replicate": row["replicate"],
                                                             "day": row["day"], "frond_count_manual": val}])],
                                        ignore_index=True)
                    save_manual(man)
                    if i in skipped:
                        skipped.remove(i)
                    st.rerun()
                if b2.button("Atla", width="stretch"):
                    if i in skipped:
                        skipped.remove(i)
                    skipped.append(i)
                    st.rerun()
        else:
            dishes = sorted({(g, r) for g, r in zip(inv["group"], inv["replicate"])}, key=lambda x: (int(x[0][1:]), x[1]))
            c1, c2 = st.columns(2)
            dsel = c1.selectbox("Kap", dishes, format_func=lambda x: f"{x[0]}_{x[1]} ({group_label(cfg, x[0])})")
            days = sorted(inv.loc[(inv["group"] == dsel[0]) & (inv["replicate"] == dsel[1]), "day"])
            day = c2.selectbox("Gün", days)
            p = inv[(inv["group"] == dsel[0]) & (inv["replicate"] == dsel[1]) & (inv["day"] == day)].iloc[0]
            show_overlay = st.toggle("Otomatik bölütlemeyi göster", value=False)
            with st.spinner("⏳ Fotoğraf yükleniyor…", show_time=True):
                if show_overlay:
                    r, ov, _ = analyse_cached(p["path"], mtime(p["path"]), cfg_key(cfg))
                    st.image(ov, width="stretch")
                    st.caption(f"Otomatik sayım: {r['frond_count_auto']}")
                else:
                    st.image(thumbnail(p["path"], mtime(p["path"]), 1600), width="stretch")
            mask = (man["group"] == dsel[0]) & (man["replicate"] == dsel[1]) & (man["day"] == day)
            current = man.loc[mask, "frond_count_manual"]
            cur = int(current.iloc[0]) if len(current) and pd.notna(current.iloc[0]) else None
            val = st.number_input("Sayılan yaprak", min_value=0, max_value=10000, value=cur, step=1,
                                  placeholder="henüz sayılmadı", key=f"cnt_{dsel}_{day}_{ver}")
            if st.button("Bu sayımı kaydet", type="primary", disabled=val is None):
                if mask.any():
                    man.loc[mask, "frond_count_manual"] = val
                else:
                    man = pd.concat([man, pd.DataFrame([{"group": dsel[0], "replicate": dsel[1], "day": day,
                                                         "frond_count_manual": val}])], ignore_index=True)
                save_manual(man)
                st.session_state.flash = f"{dsel[0]}_{dsel[1]}, gün {day}: {val} yaprak kaydedildi."
                st.rerun()
            if st.session_state.get("flash"):
                st.success(st.session_state.pop("flash"))

    with table_col:
        if blind:
            st.subheader("Sayım tablosu")
            st.info("Kör sayım sırasında tablo gizlenir. Sayım bitince kör sayım modunu kapatın.")
        else:
            st.subheader("Sayım tablosu")
            all_days = sorted(man["day"].unique())
            day_filter = st.multiselect("Günler", all_days, default=all_days[:1])
            view = man[man["day"].isin(day_filter)] if day_filter else man
            edited = st.data_editor(
                view, hide_index=True, width="stretch", height=520,
                column_config={
                    "group": st.column_config.TextColumn("grup", disabled=True),
                    "replicate": st.column_config.NumberColumn("kap", disabled=True),
                    "day": st.column_config.NumberColumn("gün", disabled=True),
                    "frond_count_manual": st.column_config.NumberColumn("yaprak sayısı", min_value=0, step=1),
                },
                key=f"manual_editor_{ver}_{'-'.join(map(str, day_filter))}",
            )
            if st.button("💾 Tabloyu kaydet"):
                man.loc[edited.index, "frond_count_manual"] = edited["frond_count_manual"]
                save_manual(man)
                st.session_state.flash_table = "Manuel sayımlar kaydedildi."
                st.rerun()
            if st.session_state.get("flash_table"):
                st.success(st.session_state.pop("flash_table"))
            st.download_button("CSV indir", man.to_csv(index=False).encode("utf-8"),
                               "manual_frond_counts.csv", "text/csv")

# ---------------------------------------------------------------- 4. analysis

if page == PG_RUN:
    st.header("Analizi çalıştır")
    inv = photo_inventory(P["photos"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Analiz edilecek fotoğraf", len(inv))
    c2.metric("Kontrol grubu", f"{cfg['control_group']} ({group_label(cfg, cfg['control_group'])})")
    c3.metric("Kap bulma", cfg["dish"]["mode"])
    with st.expander("Deney grupları (config.json)"):
        st.dataframe(pd.DataFrame(cfg["groups"]).T.rename_axis("grup"), width="stretch")
    c1, c2 = st.columns([2, 1])
    note = c1.text_input("Bu çalıştırma için not (isteğe bağlı)",
                         placeholder="örn. 7. gün sonrası tüm veri, yeşil eşik H 33–90")
    save_ov = c2.checkbox("Kontrol görüntülerini kaydet", value=True)
    make_figs = c2.checkbox("Standart grafikleri de oluştur", value=False,
                            help="Kapalıyken grafikler Sonuçlar > Grafik oluşturucu'da seçilerek üretilir.")
    st.caption("Her çalıştırma `results/runs/<tarih_saat>/` klasörüne ayrı kaydedilir; önceki sonuçlar silinmez. "
               "Tüm çalıştırmalar `results/history_*.csv` dosyalarında birikir.")
    if st.button("▶️ Analizi başlat", type="primary", disabled=inv.empty):
        STAGES = {
            "images": "Fotoğraflar analiz ediliyor",
            "statistics": "Büyüme hızları ve istatistikler hesaplanıyor",
            "saving": "Sonuçlar kaydediliyor (CSV, Excel, geçmiş)",
            "figures": "Standart grafikler çiziliyor",
        }
        t0 = time.time()
        with st.status("⏳ Analiz yapılıyor, lütfen bekleyin. Bu sırada sayfayı kapatmayın.", expanded=True) as status:
            steps = st.empty()
            bar = st.progress(0.0, text="Başlıyor…")
            order = list(STAGES)

            def stage(name):
                lines = [f"✅ {STAGES[k]}" for k in order[:order.index(name)]]
                lines.append(f"⏳ **{STAGES[name]}…**")
                steps.markdown("  \n".join(lines))
                if name != "images":
                    bar.progress(1.0, text=f"Fotoğraflar tamamlandı · geçen süre {fmt_time(time.time() - t0)}")

            def progress(i, n, name):
                elapsed = time.time() - t0
                remaining = elapsed / i * (n - i)
                bar.progress(i / n, text=f"⌛ {i}/{n} fotoğraf · geçen {fmt_time(elapsed)} · "
                                         f"kalan yaklaşık {fmt_time(remaining)}")

            try:
                res = run(P["photos"], cfg, P["results"], P["manual"], save_ov, progress,
                          note=note, make_figures=make_figs, stage=stage)
            except Exception as exc:
                status.update(label=f"❌ Analiz hata verdi: {exc}", state="error")
                res = None
            else:
                total = fmt_time(time.time() - t0)
                if res.get("run_id"):
                    steps.markdown("  \n".join(f"✅ {STAGES[k]}" for k in STAGES if make_figs or k != "figures"))
                    status.update(label=f"✅ Analiz tamamlandı ({total})", state="complete", expanded=False)
                else:
                    status.update(label="⚠️ Analiz edilecek fotoğraf bulunamadı", state="error")
        if res and res.get("run_id"):
            st.session_state.selected_run = res["run_id"]
            _photo_inventory.clear()
            st.success(f"Çalıştırma **{res['run_id']}** kaydedildi → `{res['out_dir']}`")
            for w in res.get("quality_warnings", []):
                st.warning(tr_warning(w))
            st.button("📊 Sonuçlara git", on_click=go_to, args=(PG_RES,))
            st.caption("Word raporu Sonuçlar bölümündeki **📄 Rapor oluştur** düğmesiyle hazırlanır.")
        elif res:
            st.error(res["report"])
        if res and res.get("skipped"):
            st.warning("Atlanan dosyalar: " + ", ".join(res["skipped"]))

# ---------------------------------------------------------------- 5. results

if page == PG_RES or (page == PG_ADV and adv in ADV_PAGES[:5]):
    st.header("Sonuçlar")
    runs = list_runs(P["results"])
    if runs:
        default = runs.index(st.session_state.get("selected_run")) if st.session_state.get("selected_run") in runs else 0
        hist = P["results"] / "history_runs.csv"
        notes = (pd.read_csv(hist, dtype={"note": str, "run_id": str}).set_index("run_id")["note"].fillna("").to_dict()
                 if hist.exists() else {})
        run_id = st.selectbox(
            "Çalıştırma (tarih_saat)", runs, index=default,
            format_func=lambda r: f"{r}  {'· ' + str(notes.get(r)) if notes.get(r) else ''}"
                                  f"{'  (en son)' if r == runs[0] else ''}")
        st.session_state.selected_run = run_id
        RUN_DIR = P["results"] / "runs" / run_id
    else:
        RUN_DIR = P["results"] / "runs" / "-"
    R = load_results(RUN_DIR)
    if "measurements" not in R:
        st.info("Henüz sonuç yok. Önce **Analiz** bölümünden analizi çalıştırın.")
    else:
        df, growth = R["measurements"], R.get("growth_per_dish", pd.DataFrame())
        df["group_name"] = df["group"].map(lambda gg: f"{gg} – {group_label(cfg, gg)}")
        groups_all = sorted(df["group"].unique())
        with st.popover(f"🎛️ Gruplar ({len(st.session_state.get('sel_groups', groups_all))}/{len(groups_all)})"):
            sel_groups = st.multiselect("Gösterilecek gruplar", groups_all, default=groups_all, key="sel_groups",
                                        format_func=lambda g: f"{g} – {group_label(cfg, g)}")
        if not sel_groups:
            st.warning("En az bir grup seçin; şimdilik tüm gruplar gösteriliyor.")
            sel_groups = groups_all
        d = df[df["group"].isin(sel_groups)]
        g = growth[growth["group"].isin(sel_groups)] if len(growth) else growth
        info = R.get("info", {})
        cal_day = info.get("calibration_day")
        if info.get("quality_warnings"):
            with st.expander(f"⚠️ Kalite uyarıları ({len(info['quality_warnings'])}) – sonuçları kullanmadan önce bakın",
                             expanded=(page == PG_RES)):
                for w in info["quality_warnings"]:
                    st.markdown(f"- {tr_warning(w)}")
        if "warnings" in growth.columns and growth["warnings"].fillna("").ne("").any():
            bad = growth[growth["warnings"].fillna("").ne("")]
            st.warning("Büyüme hızı hesaplanamayan kaplar (sıfır ya da eksik değer) testlere alınmadı: "
                       + "; ".join(f"{r.dish_id}: {tr_warning(r.warnings)}" for r in bad.itertuples()))

        if page == PG_RES:
            BASIC_VIEWS = {"📋 Özet": "summary", "📈 Büyüme eğrileri": "curves", "📊 Büyüme hızı": "rgr",
                           "📄 Rapor ve indirme": "report"}
            if st.session_state.get("res_view") not in BASIC_VIEWS:
                st.session_state.res_view = st.session_state.get("last_res_view") \
                    if st.session_state.get("last_res_view") in BASIC_VIEWS else "📋 Özet"
            res_view = st.segmented_control("Görünüm", list(BASIC_VIEWS), key="res_view",
                                            label_visibility="collapsed")
            st.session_state.last_res_view = res_view
            show = {BASIC_VIEWS.get(res_view, "summary")}
        else:
            show = {ADV_STATS: {"stats"}, ADV_INTER: {"interaction", "colour"}, ADV_AGREE: {"agreement"},
                    ADV_CHARTS: {"charts"}, ADV_TABLES: {"tables"}}[adv]

        rgr_cols = [c for c in growth.columns if c.startswith("rgr_") and not c.startswith("rgr_slope")]
        if "summary" in show:
            render_summary(cfg, R, df, growth, info)

        if "report" in show:
            with st.container(border=True):
                st.markdown("**📄 Word raporu** – tablolar, grafikler, hipotez değerlendirmesi ve biyolojik yorum")
                c1, c2, c3 = st.columns([2, 2, 1.4])
                author = c1.text_input("Hazırlayan (isteğe bağlı)", key="report_author")
                title = c2.text_input("Rapor başlığı", key="report_title",
                                      value="3D Yazıcı Kaynaklı PLA ve PETG Parçacıklarının Deterjan Varlığında "
                                            "Lemna minor Gelişimine Etkisi")
                c3.write("")
                if c3.button("📄 Rapor oluştur", type="primary", width="stretch"):
                    try:
                        with st.spinner("⏳ Rapor hazırlanıyor (istatistikler ve grafikler)…", show_time=True):
                            path = build_report(RUN_DIR, title=title or None, author=author)
                        st.session_state.report = (RUN_DIR.name, path.name, path.read_bytes())
                    except Exception as exc:
                        st.error(f"Rapor oluşturulamadı: {exc}")
                rep_state = st.session_state.get("report")
                if rep_state and rep_state[0] == RUN_DIR.name:
                    st.download_button("⬇️ Raporu indir (.docx)", rep_state[2], rep_state[1],
                                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                       type="primary")
                    st.caption(f"Rapor bu çalıştırmanın klasörüne de kaydedildi: `{RUN_DIR / rep_state[1]}`. "
                               "Biyolojik yorumlar kurallara dayalı olası açıklamalardır; rapor teslim edilmeden önce "
                               "okunup öğretmenle birlikte değerlendirilmelidir.")
            with st.container(border=True):
                st.markdown("**⬇️ Veri dosyaları**")
                c1, c2, c3 = st.columns(3)
                xp_ = RUN_DIR / "results.xlsx"
                if xp_.exists():
                    c1.download_button("📗 Excel (tüm tablolar)", xp_.read_bytes(), "results.xlsx", width="stretch",
                                       key="dl_xlsx_basic")
                c2.download_button("📄 Ölçümler (CSV)", R["measurements"].to_csv(index=False).encode("utf-8"),
                                   "measurements.csv", "text/csv", width="stretch", key="dl_csv_basic")
                c3.download_button("📄 Kap başına büyüme (CSV)", growth.to_csv(index=False).encode("utf-8"),
                                   "growth_per_dish.csv", "text/csv", width="stretch", key="dl_growth_basic")

        if "curves" in show:
            metrics = [c for c in METRIC_LABELS if c in d.columns and d[c].notna().any()]
            metric = st.selectbox("Ölçüt", metrics, format_func=METRIC_LABELS.get, key="curve_metric")
            c1, c2 = st.columns([1, 3])
            show = c1.radio("Gösterim", ["Grup ortalaması ± SS", "Her kap ayrı"], key="curve_show")
            logy = c1.checkbox("Logaritmik y ekseni", value=False,
                               help="Üstel büyümede doğrusal görünür, eğim büyüme hızıdır.")
            if show == "Grup ortalaması ± SS":
                s = d.groupby(["group", "group_name", "day"])[metric].agg(["mean", "std"]).reset_index()
                fig = px.line(s, x="day", y="mean", color="group_name", error_y="std", markers=True,
                              color_discrete_sequence=[GROUP_COLORS.get(x, None) for x in sorted(s["group"].unique())])
            else:
                fig = px.line(d.sort_values("day"), x="day", y=metric, color="group_name", line_group="dish_id",
                              markers=True, hover_data=["dish_id", "file"],
                              color_discrete_sequence=[GROUP_COLORS.get(x, None) for x in sorted(d["group"].unique())])
            fig.update_layout(xaxis_title="Gün", yaxis_title=METRIC_LABELS.get(metric, metric),
                              legend_title="Grup", height=480, yaxis_type="log" if logy else "linear")
            c2.plotly_chart(fig, width="stretch")

        if "rgr" in show:
            if not rgr_cols:
                st.info("Büyüme hızı hesaplanamadı (en az iki gün gerekir).")
            else:
                rc = st.selectbox("Büyüme hızı ölçütü", rgr_cols,
                                  format_func=lambda c: "RGR – " + METRIC_LABELS.get(c[4:], c[4:]))
                base = rc[4:]
                c1, c2 = st.columns(2)
                fig = px.box(g, x="group", y=rc, color="group", points="all", hover_data=["dish_id"],
                             color_discrete_map=GROUP_COLORS,
                             category_orders={"group": sorted(g["group"].unique())})
                fig.update_layout(showlegend=False, yaxis_title="RGR (1/gün)", xaxis_title="Grup", height=420)
                c1.plotly_chart(fig, width="stretch")
                inh = f"inhibition_pct_{base}"
                if inh in g.columns:
                    s = g.groupby("group")[inh].agg(["mean", "std"]).reset_index()
                    s["label"] = s["group"].map(lambda x: f"{x} – {group_label(cfg, x)}")
                    fig = px.bar(s, x="label", y="mean", error_y="std", color="group",
                                 color_discrete_map=GROUP_COLORS)
                    fig.add_hline(y=0, line_color="black", line_width=1)
                    fig.update_layout(showlegend=False, yaxis_title="Büyüme inhibisyonu (%)", xaxis_title="",
                                      height=420)
                    c2.plotly_chart(fig, width="stretch")
                    c2.caption("Negatif inhibisyon kontrole göre daha hızlı büyüme demektir.")
                st.dataframe(
                    g[["dish_id", "group_label", f"{base}_start", f"{base}_end", rc, f"rgr_slope_{base}", inh]]
                    .rename(columns={"dish_id": "kap", "group_label": "grup"}),
                    width="stretch", hide_index=True)

        if "interaction" in show:
            if rgr_cols:
                rc = st.selectbox("Ölçüt", rgr_cols, key="int_metric",
                                  format_func=lambda c: "RGR – " + METRIC_LABELS.get(c[4:], c[4:]))
                s = growth.groupby(["plastic", "detergent"])[rc].agg(["mean", "std", "count"]).reset_index()
                s["se"] = s["std"] / np.sqrt(s["count"])
                s["plastic"] = pd.Categorical(s["plastic"], ["none", "PLA", "PETG"])
                s = s.sort_values("plastic")
                fig = go.Figure()
                for det, name, dash in [("no", "Deterjan yok", "solid"), ("yes", "Deterjan var", "dash")]:
                    t = s[s["detergent"] == det]
                    fig.add_trace(go.Scatter(x=t["plastic"].astype(str).replace({"none": "Plastik yok"}), y=t["mean"],
                                             error_y=dict(type="data", array=t["se"]), mode="lines+markers",
                                             name=name, line=dict(dash=dash, width=3), marker=dict(size=10)))
                fig.update_layout(yaxis_title="RGR (1/gün), ortalama ± SH", xaxis_title="Plastik türü", height=450)
                st.plotly_chart(fig, width="stretch")
                st.markdown(
                    "Çizgiler paralelse plastik ve deterjanın etkileri toplanabilir (etkileşim yok). "
                    "Çizgilerin eğimi farklıysa birlikte maruziyet, tekil etkilerin toplamından farklıdır. "
                    "Anlamlılık için **İstatistik** görünümündeki etkileşim terimine bakın."
                )

        if "colour" in show:
            cc = [c for c in ["yellow_pct_of_plant", "brown_pct_of_plant", "exg_index", "mean_hue"] if c in d.columns]
            col = st.selectbox("Renk ölçütü", cc, format_func=METRIC_LABELS.get)
            s = d.groupby(["group", "group_name", "day"])[col].agg(["mean", "std"]).reset_index()
            fig = px.line(s, x="day", y="mean", error_y="std", color="group_name", markers=True,
                          color_discrete_sequence=[GROUP_COLORS.get(x, None) for x in sorted(s["group"].unique())])
            fig.update_layout(xaxis_title="Gün", yaxis_title=METRIC_LABELS[col], height=450, legend_title="Grup")
            st.plotly_chart(fig, width="stretch")
            last = d[d["day"] == d["day"].max()]
            fig = px.box(last, x="group", y=col, color="group", points="all", color_discrete_map=GROUP_COLORS)
            fig.update_layout(showlegend=False, title=f"Son gün (gün {d['day'].max()})", height=380,
                              yaxis_title=METRIC_LABELS[col])
            st.plotly_chart(fig, width="stretch")

        if "agreement" in show:
            if "frond_count_manual" not in df.columns or df["frond_count_manual"].notna().sum() < 3:
                st.info("Uyum analizi için **Manuel sayım** bölümünden sayım girin ve analizi yeniden çalıştırın.")
            else:
                options = [c for c in ["frond_count_auto", "frond_count_from_area"] if c in df.columns]
                meth = st.radio("Otomatik yöntem", options, format_func=METRIC_LABELS.get, horizontal=True)
                a = df[[meth, "frond_count_manual", "group", "day", "dish_id"]].dropna()
                if meth == "frond_count_from_area" and cal_day is not None:
                    a = a[a["day"] != cal_day]
                    st.caption(f"Alan tabanlı sayım {cal_day}. günün manuel sayımlarıyla kalibre edildiği için "
                               "o gün bu karşılaştırmaya alınmadı.")
                if meth == "frond_count_from_area" and "area_per_frond" in R.get("sheets", {}):
                    with st.expander("Yaprak başına alan (alan tabanlı sayımın varsayımı)", expanded=True):
                        st.dataframe(R["sheets"]["area_per_frond"].rename(
                            columns={"group": "grup", "all_days": "tüm günler",
                                     "deviation_from_overall_pct": "genel ortancadan sapma (%)"}),
                            width="stretch", hide_index=True)
                        st.caption("Ortanca alan / manuel yaprak sayısı. Gruplar ve günler arasında yaprak büyüklüğü "
                                   "%10'dan fazla değişiyorsa alan tabanlı sayım güvenilir değildir; manuel sayım esas alınır.")
                if len(a) < 3:
                    st.info("Karşılaştırma için en az 3 eşleşen sayım gerekir.")
                    st.stop()
                diff = a[meth] - a["frond_count_manual"]
                mean = (a[meth] + a["frond_count_manual"]) / 2
                bias, sd = diff.mean(), diff.std(ddof=1)
                r = np.corrcoef(a[meth], a["frond_count_manual"])[0, 1]
                k = st.columns(4)
                k[0].metric("Pearson r", f"{r:.3f}")
                k[1].metric("Ortalama fark (oto − manuel)", f"{bias:.2f}")
                k[2].metric("Uyum sınırları", f"{bias - 1.96 * sd:.1f} / {bias + 1.96 * sd:.1f}")
                k[3].metric("Ortalama mutlak hata", f"{diff.abs().mean():.2f}")
                c1, c2 = st.columns(2)
                mx = float(max(a[meth].max(), a["frond_count_manual"].max())) * 1.05
                fig = px.scatter(a, x="frond_count_manual", y=meth, color="day", hover_data=["dish_id"],
                                 labels={"frond_count_manual": "Manuel sayım", meth: "Otomatik sayım"})
                fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dash", color="grey"))
                fig.update_layout(height=420)
                c1.plotly_chart(fig, width="stretch")
                fig = px.scatter(x=mean, y=diff, color=a["day"], labels={"x": "Ortalama", "y": "Otomatik − manuel", "color": "gün"})
                for y, dash in [(bias, "solid"), (bias + 1.96 * sd, "dash"), (bias - 1.96 * sd, "dash")]:
                    fig.add_hline(y=y, line_dash=dash, line_color="black")
                fig.update_layout(title="Bland–Altman", height=420)
                c2.plotly_chart(fig, width="stretch")

        if "stats" in show:
            sheets = R.get("sheets", {})
            if not any(n.startswith("anova_") for n in sheets):
                st.info("İki faktörlü ANOVA bu çalıştırmada yapılamadı (ör. deney düzeninde eksik grup). "
                        "Nedeni ve yapılan testler aşağıdaki tam raporda yazılıdır.")
            for name in [n for n in sheets if n.startswith("anova_")]:
                key = name[6:]
                rate = key in ("plant_area_mm2", "plant_area_px", "frond_count_manual", "frond_count_auto",
                               "frond_count_from_area")
                st.subheader(f"İki faktörlü ANOVA – {'büyüme hızı (RGR): ' if rate else ''}"
                             f"{METRIC_LABELS.get(key, key)}")
                t = sheets[name]
                st.dataframe(t, width="stretch", hide_index=True)
                if "PR(>F)" in t.columns:
                    for _, row in t.dropna(subset=["PR(>F)"]).iterrows():
                        src = TERM_TR.get(row["source"], row["source"])
                        sig = "anlamlı" if row["PR(>F)"] < 0.05 else "anlamlı değil"
                        st.markdown(f"- **{src}**: F = {row['F']:.2f}, p = {row['PR(>F)']:.4f} → {sig} (α = 0,05)")
                inter = f"interaction_{name[6:]}"
                if inter in sheets:
                    st.markdown("**Toplanabilirlikten sapma (birlikte etki − tekil etkilerin toplamı)**")
                    st.dataframe(sheets[inter], width="stretch", hide_index=True)
                rob = f"robust_{name[6:]}"
                if rob in sheets:
                    rt = sheets[rob]
                    st.markdown("**Sağlamlık kontrolü** – normallik varsayımı gerektirmeyen permütasyon testi ve "
                                "log(RGR) üzerinde ANOVA")
                    st.dataframe(rt.assign(source=rt["source"].map(TERM_TR).fillna(rt["source"])).rename(
                                 columns={"source": "etki", "p_anova": "p (ANOVA)",
                                                    "p_permutation": "p (permütasyon)",
                                                    "p_anova_log_rgr": "p (log RGR)"}),
                                 width="stretch", hide_index=True)
                    cols = [c for c in ["p_anova", "p_permutation", "p_anova_log_rgr"] if rt[c].notna().all()]
                    same = (rt[cols] < 0.05).nunique(axis=1).eq(1)
                    if same.all():
                        st.success("Üç test de aynı sonuca varıyor; sonuç normallik varsayımına bağlı değil.")
                    else:
                        bad = ", ".join(rt.loc[~same, "source"].map(TERM_TR).fillna(rt.loc[~same, "source"]))
                        st.warning(f"{bad} için sonuç kullanılan teste göre değişiyor; dikkatle yorumlayın.")
                tk = f"tukey_{name[6:]}"
                if tk in sheets:
                    with st.expander("Tukey HSD ikili karşılaştırmalar"):
                        st.dataframe(sheets[tk], width="stretch", hide_index=True)
                pl = f"planned_{name[6:]}"
                if pl in sheets:
                    st.markdown("**Planlanmış karşılaştırmalar (süzüntü grupları, Holm düzeltmesi)**")
                    st.dataframe(sheets[pl].rename(columns={"group1": "grup 1", "group2": "grup 2",
                                                            "difference": "fark (grup 2 − grup 1)",
                                                            "p_holm": "p (Holm)", "p_raw": "p (düzeltmesiz)"}),
                                 width="stretch", hide_index=True)
            if "mixed_model_status" in sheets:
                ms = sheets["mixed_model_status"]
                st.subheader("Günlük veriler için karma model")
                for r in ms.itertuples():
                    label = METRIC_LABELS.get(r.response, r.response)
                    if not bool(r.converged):
                        st.error(f"{label}: model yakınsamadı; tahminleri ve p değerleri raporlanmamalıdır.")
                    elif bool(r.boundary):
                        st.info(f"{label}: yakınsadı ({STRUCTURE_TR.get(r.structure, r.structure)}); "
                                "bir rastgele etki varyansı sıfır tahmin edildi, "
                                "sabit etkiler kullanılabilir.")
                    else:
                        st.success(f"{label}: model yakınsadı ({STRUCTURE_TR.get(r.structure, r.structure)}).")
            with st.expander("Tam istatistik raporu (metin)"):
                st.code(R.get("report", ""), language=None)

        if "charts" in show:
            st.markdown("Grafik türünü ve seçenekleri belirleyin; **Grafiği oluştur** düğmesi grafiği rapora uygun "
                        "PNG (200 dpi) olarak bu çalıştırmanın `figures/` klasörüne kaydeder.")
            kinds_tr = {
                "curve": "Büyüme eğrisi (grup ortalaması ± SS)",
                "curve_dishes": "Büyüme eğrisi (her kap ayrı)",
                "day_box": "Seçilen günde gruplar (kutu grafiği)",
                "rgr_box": "Gruplara göre büyüme hızı (RGR)",
                "inhibition": "Kontrole göre büyüme inhibisyonu",
                "interaction": "Etkileşim grafiği (plastik × deterjan)",
                "heatmap": "Kap × gün ısı haritası",
                "agreement": "Manuel ve otomatik sayım uyumu",
            }
            c1, c2 = st.columns([1, 1])
            kind = c1.selectbox("Grafik türü", list(CHART_KINDS), format_func=kinds_tr.get)
            if kind in ("rgr_box", "inhibition", "interaction"):
                opts = [c[4:] for c in rgr_cols]
            elif kind == "agreement":
                opts = [c for c in ["frond_count_auto", "frond_count_from_area"]
                        if c in df.columns and "frond_count_manual" in df.columns]
            else:
                opts = [c for c in METRIC_LABELS if c in df.columns and df[c].notna().any()]
            if not opts:
                st.info("Bu grafik için gerekli veri yok (ör. manuel sayım veya en az iki gün).")
            else:
                metric = c2.selectbox("Ölçüt", opts, format_func=lambda c: METRIC_LABELS.get(c, c), key="gb_metric")
                c3, c4, c5 = st.columns(3)
                gsel = c3.multiselect("Gruplar", groups_all, default=groups_all, key="gb_groups",
                                      format_func=lambda x: f"{x} – {group_label(cfg, x)}") or groups_all
                dmin, dmax = int(df["day"].min()), int(df["day"].max())
                drange, dsel, logy = None, None, False
                if kind in ("curve", "curve_dishes", "heatmap") and dmax > dmin:
                    drange = c4.slider("Gün aralığı", dmin, dmax, (dmin, dmax), key="gb_range")
                if kind == "day_box":
                    dsel = c4.select_slider("Gün", sorted(df["day"].unique()), value=dmax, key="gb_day")
                if kind in ("curve", "curve_dishes"):
                    logy = c5.checkbox("Logaritmik y ekseni", key="gb_log")
                if st.button("🖼️ Grafiği oluştur", type="primary"):
                    stamp = pd.Timestamp.now().strftime("%H%M%S")
                    out = RUN_DIR / "figures" / f"{kind}_{metric}_{stamp}.png"
                    try:
                        with st.spinner("⏳ Grafik çiziliyor…", show_time=True):
                            made = make_chart(kind, df, growth, metric, out, groups=gsel, day_range=drange,
                                              day=dsel, logy=logy,
                                              exclude_day=cal_day if metric == "frond_count_from_area" else None)
                        if made is None:
                            st.warning("Bu grafik için yeterli veri yok (en az 3 eşleşen sayım gerekir).")
                        else:
                            st.session_state.last_chart = str(out)
                    except Exception as exc:
                        st.error(f"Grafik oluşturulamadı: {exc}")
                last = st.session_state.get("last_chart")
                if last and Path(last).exists() and Path(last).parent == RUN_DIR / "figures":
                    st.image(last, width="stretch")
                    st.download_button("PNG indir", Path(last).read_bytes(), Path(last).name, "image/png")
            figs = sorted((RUN_DIR / "figures").glob("*.png"), key=lambda f: f.stat().st_mtime, reverse=True)
            if figs:
                if st.toggle(f"Bu çalıştırmada oluşturulan grafikleri göster ({len(figs)})", value=False):
                    cols = st.columns(3)
                    for i, f in enumerate(figs):
                        cols[i % 3].image(str(f), caption=f.name, width="stretch")

        if "tables" in show:
            for name in ["growth_per_group", "growth_per_dish", "daily_summary", "measurements"]:
                if name in R:
                    with st.expander(name, expanded=(name == "growth_per_group")):
                        st.dataframe(R[name], width="stretch", hide_index=True)
            c1, c2, c3 = st.columns(3)
            xp = RUN_DIR / "results.xlsx"
            if xp.exists():
                c1.download_button("📗 Excel (tüm tablolar)", xp.read_bytes(), "results.xlsx", width="stretch")
            c2.download_button("📄 measurements.csv", R["measurements"].to_csv(index=False).encode("utf-8"),
                               "measurements.csv", "text/csv", width="stretch")
            if c3.button("🗜️ Tüm sonuçları ZIP hazırla", width="stretch"):
                with st.spinner("⏳ ZIP hazırlanıyor…", show_time=True):
                    st.session_state.zip = (RUN_DIR.name, zip_folder(RUN_DIR))
            if st.session_state.get("zip", (None,))[0] == RUN_DIR.name:
                c3.download_button("ZIP indir", st.session_state.zip[1], f"results_{RUN_DIR.name}.zip",
                                   width="stretch")

# ---------------------------------------------------------------- 6. history

if page == PG_ADV and adv == ADV_HISTORY:
    st.header("Analiz geçmişi")
    hist_runs = P["results"] / "history_runs.csv"
    hist_meas = P["results"] / "history_measurements.csv"
    hist_growth = P["results"] / "history_growth_per_dish.csv"
    if not hist_runs.exists():
        st.info("Henüz kayıtlı çalıştırma yok.")
    else:
        H = pd.read_csv(hist_runs)
        st.markdown("Her satır bir analiz çalıştırmasıdır. Ayrıntılar `results/runs/<run_id>/` klasöründe, "
                    "tüm ölçümler ise `history_measurements.csv` dosyasında `run_id` sütunuyla birlikte tutulur.")
        base_cols = ["run_id", "analysed_at", "note", "n_photos", "n_dishes", "days",
                     "first_photo", "last_photo", "dish_mode", "hsv_green"]
        st.dataframe(H[[c for c in base_cols if c in H.columns]].iloc[::-1], width="stretch", hide_index=True)
        with st.expander("Tüm özet sütunları (grup ortalama RGR, ANOVA p değerleri, uyum)"):
            st.dataframe(H.iloc[::-1], width="stretch", hide_index=True)

        st.subheader("Çalıştırmaları karşılaştır")
        if hist_growth.exists():
            HG = pd.read_csv(hist_growth)
            sel_runs = st.multiselect("Çalıştırmalar", H["run_id"].tolist()[::-1],
                                      default=H["run_id"].tolist()[::-1][:2])
            rcols = [c for c in HG.columns if c.startswith("rgr_") and not c.startswith("rgr_slope")]
            if sel_runs and rcols:
                rc = st.selectbox("Ölçüt", rcols, key="hist_metric",
                                  format_func=lambda c: "RGR – " + METRIC_LABELS.get(c[4:], c[4:]))
                hg = HG[HG["run_id"].isin(sel_runs)].dropna(subset=[rc])
                s = hg.groupby(["run_id", "group"])[rc].agg(["mean", "std"]).reset_index()
                fig = px.bar(s, x="group", y="mean", error_y="std", color="run_id", barmode="group")
                fig.update_layout(yaxis_title="RGR (1/gün)", xaxis_title="Grup", height=420)
                st.plotly_chart(fig, width="stretch")
                st.caption("Aynı fotoğraflar farklı ayarlarla analiz edildiyse farklar ayarların etkisini gösterir.")

        st.subheader("Geçmiş dosyalarını indir")
        c1, c2, c3 = st.columns(3)
        for col, f in zip((c1, c2, c3), (hist_runs, hist_meas, hist_growth)):
            if f.exists():
                col.download_button(f.name, f.read_bytes(), f.name, "text/csv", width="stretch")

# ---------------------------------------------------------------- 7. image review

if page == PG_ADV and adv == ADV_IMAGES:
    st.header("Görüntü inceleme")
    runs = list_runs(P["results"])
    rv_run = st.session_state.get("selected_run") if st.session_state.get("selected_run") in runs else (runs[0] if runs else "-")
    qc_dir = P["results"] / "runs" / rv_run / "qc_overlays"
    inv = photo_inventory(P["photos"])
    if inv.empty:
        st.info("Fotoğraf yok.")
    else:
        mode = st.radio("Görünüm", ["Bir kabın zaman serisi", "Bir günün tüm kapları"], horizontal=True)
        use_qc = st.toggle("Kontrol görüntülerini göster", value=qc_dir.exists())
        st.caption(f"Kontrol görüntüleri çalıştırma **{rv_run}** içinden gösterilir (Sonuçlar bölümünde seçilen).")
        R2 = load_results(P["results"] / "runs" / rv_run)
        meas = R2.get("measurements")

        def show(row, col, caption=None):
            qc = qc_dir / f"{Path(row['file']).stem}_qc.jpg"
            if use_qc and qc.exists():
                col.image(str(qc), caption=caption or row["file"], width="stretch")
            else:
                col.image(thumbnail(row["path"], mtime(row["path"]), 700), caption=caption or row["file"],
                          width="stretch")

        if mode == "Bir kabın zaman serisi":
            dishes = sorted({(g_, r_) for g_, r_ in zip(inv["group"], inv["replicate"])}, key=lambda x: (int(x[0][1:]), x[1]))
            dsel = st.selectbox("Kap", dishes, format_func=lambda x: f"{x[0]}_{x[1]} ({group_label(cfg, x[0])})", key="rev_dish")
            rows = inv[(inv["group"] == dsel[0]) & (inv["replicate"] == dsel[1])].sort_values("day")
            layout = st.radio("Düzen", ["Kaydırıcı", "Izgara"], horizontal=True)
            if layout == "Kaydırıcı":
                days = rows["day"].tolist()
                dsl = st.select_slider("Gün", days, value=days[0])
                c1, c2 = st.columns([2, 1])
                show(rows[rows["day"] == dsl].iloc[0], c1)
                if meas is not None:
                    m = meas[meas["dish_id"] == f"{dsel[0]}_{dsel[1]}"].sort_values("day")
                    if len(m):
                        ycol = "plant_area_mm2" if m["plant_area_mm2"].notna().all() else "plant_area_px"
                        fig = px.line(m, x="day", y=ycol, markers=True)
                        fig.add_vline(x=dsl, line_dash="dash")
                        fig.update_layout(height=300, yaxis_title=METRIC_LABELS[ycol], xaxis_title="Gün")
                        c2.plotly_chart(fig, width="stretch")
                        c2.dataframe(m[m["day"] == dsl][["frond_count_auto", "green_pct_of_roi",
                                                          "yellow_pct_of_plant"]].T, width="stretch")
            else:
                cols = st.columns(4)
                for i, (_, row) in enumerate(rows.iterrows()):
                    show(row, cols[i % 4], f"Gün {row['day']}")
        else:
            day = st.selectbox("Gün", sorted(inv["day"].unique()), key="rev_day")
            rows = inv[inv["day"] == day].sort_values(["group", "replicate"])
            for grp, gr in rows.groupby("group"):
                st.markdown(f"**{grp} – {group_label(cfg, grp)}**")
                cols = st.columns(max(5, len(gr)))
                for i, (_, row) in enumerate(gr.iterrows()):
                    show(row, cols[i], f"{grp}_{row['replicate']}")

# ---------------------------------------------------------------- 8. daily log

if page == PG_REC:
    st.header("Deney kayıtları")
    log_tabs = st.tabs(["📅 Günlük kayıt (sıcaklık, pH, ışık)", "📏 Son gün ölçümleri (kök uzunluğu)"])
    inv = photo_inventory(P["photos"])
    keys = list(zip(inv["group"], inv["replicate"], inv["day"]))

    with log_tabs[0]:
        st.markdown("pH her kapta her gün ölçülür. Program bu değerlerden her kap için son gün değerini ve deney "
                    "boyunca değişimi hesaplar ve gruplar arasında karşılaştırır. Sıcaklık ve ışık şiddeti, ortam "
                    "koşullarının OECD 221 sınırları içinde kaldığını göstermek için kaydedilir.")
        ensure_sheet(P["ws"], "log", cfg, keys)
        log = pd.read_csv(P["log"], dtype={"date": str, "notes": str, "group": str})
        log["day"] = pd.to_numeric(log["day"], errors="coerce")
        log = log.dropna(subset=["day"])
        log["day"] = log["day"].astype(int)
        numeric = ["temperature_C", "pH", "light_lux"]
        for c in numeric:
            if c not in log.columns:
                log[c] = np.nan
            log[c] = pd.to_numeric(log[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        log_loaded = log.copy()
        dsel = st.selectbox("Gün", sorted(log["day"].unique()), key="log_day")
        view = log[log["day"] == dsel]
        edited = st.data_editor(
            view, hide_index=True, width="stretch",
            column_config={
                "date": st.column_config.TextColumn("tarih (GG.AA.YYYY)"),
                "day": st.column_config.NumberColumn("gün", disabled=True),
                "group": st.column_config.TextColumn("grup", disabled=True),
                "replicate": st.column_config.NumberColumn("kap", disabled=True),
                "temperature_C": st.column_config.NumberColumn("sıcaklık (°C)", format="%.1f"),
                "pH": st.column_config.NumberColumn("pH", format="%.2f"),
                "light_lux": st.column_config.NumberColumn("ışık (lüks)"),
                "notes": st.column_config.TextColumn("gözlem"),
            },
            key=f"log_editor_{dsel}_{st.session_state.get('log_ver', 0)}",
        )
        if st.button("💾 Günlük kaydı kaydet"):
            for c in edited.columns:
                log.loc[edited.index, c] = edited[c]
            merge_save(P["log"], ["group", "replicate", "day"], log_loaded, log)
            bump("log_ver")
            st.session_state.flash_log = "Kaydedildi."
            st.rerun()
        if st.session_state.get("flash_log"):
            st.success(st.session_state.pop("flash_log"))
        if log["pH"].notna().any():
            summary = log.groupby(["day", "group"])[["pH"]].mean().reset_index()
            fig = px.line(summary, x="day", y="pH", color="group", markers=True,
                          color_discrete_map=GROUP_COLORS, labels={"day": "Gün", "group": "Grup"})
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), title="Ortalama pH")
            st.plotly_chart(fig, width="stretch")

    with log_tabs[1]:
        st.markdown("Yedinci günde her kaptan rastgele seçilen 5 koloninin en uzun kökü ölçülür ve ortalaması "
                    "girilir. Yaş ağırlık ölçülmüşse ayrıca girilebilir. Analiz bu değerleri de gruplar arasında "
                    "karşılaştırır.")
        ensure_endpoints(P["ws"], cfg, keys)
        ends = pd.read_csv(P["ws"] / "endpoints.csv", dtype={"group": str, "notes": str})
        for c in ["root_length_mm", "fresh_weight_mg"]:
            if c not in ends.columns:
                ends[c] = np.nan
            ends[c] = pd.to_numeric(ends[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        if "notes" not in ends.columns:
            ends["notes"] = ""
        ends_loaded = ends[["group", "replicate", "root_length_mm", "fresh_weight_mg", "notes"]].copy()
        edited_e = st.data_editor(
            ends_loaded.copy(), hide_index=True,
            width="stretch", height=480,
            column_config={
                "group": st.column_config.TextColumn("grup", disabled=True),
                "replicate": st.column_config.NumberColumn("kap", disabled=True),
                "root_length_mm": st.column_config.NumberColumn("kök uzunluğu (mm)", min_value=0.0, format="%.1f"),
                "fresh_weight_mg": st.column_config.NumberColumn("yaş ağırlık (mg)", min_value=0.0, format="%.1f"),
                "notes": st.column_config.TextColumn("not"),
            },
            key=f"endpoint_editor_{st.session_state.get('end_ver', 0)}",
        )
        if st.button("💾 Son gün ölçümlerini kaydet"):
            merge_save(P["ws"] / "endpoints.csv", ["group", "replicate"], ends_loaded, edited_e)
            bump("end_ver")
            st.session_state.flash_end = "Kaydedildi."
            st.rerun()
        if st.session_state.get("flash_end"):
            st.success(st.session_state.pop("flash_end"))

# ---------------------------------------------------------------- particle size

if page == PG_ADV and adv == ADV_PARTICLES:
    st.header("Parçacık boyutu")
    st.markdown(
        "Plastik parçacıklarının mikroskop fotoğrafından boyutunu ölçer. Parçacıkları lam üzerinde **birbirine "
        "değmeyecek şekilde** dağıtın, düz ve açık renkli bir zeminde fotoğraflayın. Ölçeği bulmak için aynı "
        "büyütmede mikrometrik cetveli fotoğraflayıp kaç pikselin kaç mikrometreye karşılık geldiğini hesaplayın.")
    c1, c2, c3 = st.columns(3)
    um_per_px = c1.number_input("Ölçek (µm / piksel)", 0.01, 1000.0, 2.0, 0.01, format="%.3f")
    dark = c2.radio("Parçacık rengi", ["Zeminden koyu", "Zeminden açık"], horizontal=True) == "Zeminden koyu"
    min_area = c3.number_input("En küçük parçacık (piksel)", 1, 10000, 20, 1)
    low, high = st.slider("Hedef boyut aralığı (µm)", 0, 2000, (200, 500), 10)
    files = st.file_uploader("Mikroskop fotoğrafları", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
                             accept_multiple_files=True, key="particle_files")
    if files:
        from lemna.particles import measure_particles, summarise
        tables = []
        with st.spinner("⏳ Parçacıklar ölçülüyor…", show_time=True):
            for f in files:
                img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    st.warning(f"{f.name} okunamadı.")
                    continue
                t, ov = measure_particles(img, um_per_px, dark_particles=dark, min_area_px=min_area)
                t.insert(0, "file", f.name)
                tables.append(t)
                with st.expander(f"{f.name}: {len(t)} parçacık"):
                    st.image(bgr_to_rgb(ov), width="stretch")
        allp = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        if allp.empty:
            st.info("Parçacık bulunamadı. Parçacık rengini ve ölçeği kontrol edin.")
        else:
            sm = summarise(allp, low, high)
            k = st.columns(4)
            k[0].metric("Parçacık sayısı", sm["n"])
            k[1].metric("Ortalama en uzun boy (µm)", f"{sm['mean_max_length_um']:.0f}")
            k[2].metric("Ortanca (µm)", f"{sm['median_max_length_um']:.0f}")
            k[3].metric(f"{low}–{high} µm aralığında", f"%{sm['pct_in_range']:.0f}")
            fig = px.histogram(allp, x="max_length_um", nbins=30, labels={"max_length_um": "En uzun boy (µm)"})
            fig.add_vrect(x0=low, x1=high, fillcolor="green", opacity=0.08, line_width=0)
            fig.update_layout(height=320, yaxis_title="Parçacık sayısı")
            st.plotly_chart(fig, width="stretch")
            st.download_button("CSV indir", allp.to_csv(index=False).encode("utf-8"), "particle_sizes.csv",
                               "text/csv")
            if st.button("💾 Sonuçları çalışma klasörüne kaydet"):
                allp.to_csv(P["ws"] / "particle_sizes.csv", index=False)
                st.success(f"Kaydedildi: `{P['ws'] / 'particle_sizes.csv'}`")

# ---------------------------------------------------------------- 9. help

if page == PG_HELP:
    st.header("Yardım")
    help_tabs = st.tabs(["Kısa kullanım", "📖 Terimler sözlüğü", "Sonuçları okuma"])

    with help_tabs[0]:
        st.markdown(
            """
1. **🏠 Başlangıç:** her adımın durumunu ve sıradaki adımı gösterir. “Git →” ile ilgili sayfaya geçilir.
2. **📤 Fotoğraflar:** her kap her gün aynı yükseklik, ışık ve mat zeminle tam tepeden fotoğraflanır. Adı
   `G1_1_Gun0.jpg` biçimindeki dosyalar doğrudan tanınır. Telefondan gelen adlar için “Bir günün fotoğraflarını
   sırayla ata” seçeneği kullanılır (2. seri için ilk kap numarası 5, 3. seri için 9).
3. **🎯 Kalibrasyon:** birkaç fotoğrafta kap dairesi ve yeşil eşikler kontrol edilip kaydedilir. Ayrıntılı
   ayarlar “⚙️ Gelişmiş parametreler” anahtarının arkasındadır.
4. **✍️ Sayım:** 0., 3., 5. ve 7. günlerde manuel yaprak sayımı, tercihen kör sayım moduyla.
5. **📝 Kayıtlar:** günlük sıcaklık, pH, ışık ve 7. gündeki kök uzunluğu.
6. **⚙️ Analiz:** “Analizi başlat”. Her analiz tarihli ayrı bir klasöre kaydedilir.
7. **📊 Sonuçlar:** özet tablo ve bulgular, büyüme eğrileri, büyüme hızı ve Word raporu.
8. **🧪 Gelişmiş:** ayrıntılı istatistik, etkileşim ve renk, ölçüm doğruluğu, grafik oluşturucu, tablolar,
   görüntü inceleme, parçacık boyutu ve analiz geçmişi.
            """
        )

    with help_tabs[1]:
        GLOSSARY = {
            "Deney ve ölçüm": [
                ("Deney birimi (kap)", "İstatistikte bağımsız tekrar sayılan birim kaptır. Aynı kabın farklı günlerdeki "
                 "fotoğrafları bağımsız tekrar değildir. Her grupta 5 kap varsa n = 5'tir."),
                ("Gün 0", "Bitkilerin deney ortamına konduğu gün. Başlangıç ölçümü (N₀) bu gün yapılır. Gün 7, "
                 "7 gündür maruz kalındığını gösterir."),
                ("Manuel yaprak sayısı", "Fotoğraftaki yaprakların (frond) elle sayılması. OECD 221'in temel ölçütüdür "
                 "ve programda büyüme hızının ana ölçütü olarak kullanılır."),
                ("Otomatik sayım (watershed)", "Bitişik yaprakları, kalınlık haritasındaki tepe noktalarına göre "
                 "birbirinden ayırarak sayar. Üst üste binen yapraklarda hata yapabilir."),
                ("Alan tabanlı sayım", "Bitki alanı ÷ tek yaprak alanı. Tek yaprak alanı ilk günün manuel "
                 "sayımlarından hesaplanır (kalibrasyon günü). Yapraklar küçülürse yanlı sonuç verir."),
                ("Bitki alanı (mm²)", "Kap içindeki bitki dokusunun kapladığı alan. Kap çapı kullanılarak pikselden "
                 "mm²'ye çevrilir."),
                ("Yeşil alan yüzdesi", "Yeşil piksellerin kap alanına oranı. Biyokütle veya fotosentez ölçümü değildir."),
                ("Sararma (kloroz)", "Sarı dokunun bitki alanına oranı. Klorofil kaybının görünür bir işaretidir ve "
                 "strese bağlı olabilir."),
                ("Kahverengileşme (nekroz)", "Kahverengi dokunun bitki alanına oranı. Ölmekte olan dokuyu gösterebilir."),
                ("ExG indeksi", "Fazla yeşil indeksi: 2·yeşil − kırmızı − mavi (oranlarla). Değer düştükçe bitki daha "
                 "az yeşildir."),
                ("HSV", "Rengi ton (H), doygunluk (S) ve parlaklık (V) olarak ifade eden renk uzayı. Yeşil, sarı ve "
                 "kahverengi doku bu değerlerin aralıklarıyla ayrılır."),
                ("Kenar payı", "Kap kenarındaki yansımaları dışarıda bırakmak için kap yarıçapının kullanılan kısmı "
                 "(ör. 0,94)."),
                ("Kenar pikseli düzeltmesi", "Yaprak kenarındaki yarı yaprak/yarı zemin piksellerini %50 kuralıyla "
                 "sınıflandırır. JPEG renk taşmasından doğan alan fazlalığını giderir."),
                ("Kontrol görüntüsü", "Programın neyi bitki saydığını gösteren görüntü: yeşil, sarı, kahverengi doku ve "
                 "mor yaprak sınırları. Hatalı ölçümleri bulmak için kullanılır."),
                ("Kör sayım", "Manuel sayımda fotoğrafların rastgele sırayla ve yalnızca bir kodla gösterilmesi. Sayan "
                 "kişi grubu bilmediği için beklentisine göre sayma (yanlılık) önlenir."),
                ("Kök uzunluğu", "7. günde her kaptan rastgele seçilen 5 koloninin en uzun kökünün ortalaması (mm). "
                 "Mikroplastiklere karşı büyüme hızından daha duyarlı olabilir."),
                ("pH", "Suyun asitliği. PLA suda parçalanırsa laktik asit oluşacağı için pH düşebilir. Her kapta her "
                 "gün ölçülür."),
                ("Feret çapı (en uzun boy)", "Bir parçacığın iki ucu arasındaki en büyük uzaklık. Parçacık boyutu "
                 "aracı bu değeri mikrometre olarak verir."),
                ("Ölçek (µm/piksel)", "Mikroskop fotoğrafında bir pikselin kaç mikrometre olduğu. Aynı büyütmede "
                 "mikrometrik cetvel fotoğraflanarak bulunur."),
            ],
            "Büyüme": [
                ("Bağıl büyüme hızı (RGR)", "Günlük büyüme oranı: RGR = [ln(Nₜ) − ln(N₀)] / t. Örneğin 0,30 1/gün, "
                 "yaprak sayısının günde yaklaşık %35 arttığı anlamına gelir."),
                ("RGR (eğim)", "Tüm günlerin ln(değer) – gün doğrusunun eğimi. Tek bir hatalı fotoğraftan daha az "
                 "etkilenir."),
                ("Büyüme inhibisyonu (%)", "Büyümenin kontrole göre ne kadar yavaşladığı: "
                 "I = (RGRkontrol − RGRuygulama) / RGRkontrol × 100. %25, büyüme hızının dörtte bir azaldığı anlamına gelir."),
                ("Negatif inhibisyon", "Uygulama grubu kontrolden hızlı büyümüştür. Kendiliğinden “yararlı etki” olarak "
                 "yorumlanmaz."),
                ("İkiye katlanma süresi", "Yaprak sayısının iki katına çıkması için gereken süre: ln 2 / RGR. "
                 "RGR = 0,30 ise yaklaşık 2,3 gündür."),
                ("OECD 221 geçerlilik ölçütü", "Kontrol grubunda yaprak sayısının ikiye katlanma süresi 2,5 günden kısa "
                 "olmalıdır. Sağlanmazsa deney koşulları sorgulanmalıdır."),
                ("Yaprak başına alan", "Bitki alanı ÷ manuel yaprak sayısı. Stres altında yapraklar küçük kalabilir; "
                 "bu durumda alan tabanlı sayım güvenilmez olur."),
            ],
            "Birlikte etki (etkileşim)": [
                ("Toplamsal beklenti", "Plastik ve deterjanın tek başına inhibisyonlarının toplamı. Örneğin %5 + %25 = %30."),
                ("Etkileşim", "Birlikte etkinin toplamsal beklentiden farklı olması. Etkileşim grafiğinde çizgilerin "
                 "paralel olmamasıyla görülür."),
                ("Sinerji", "Birlikte etkinin toplamdan büyük olması (ör. beklenen %30, gözlenen %45)."),
                ("Antagonizma", "Birlikte etkinin toplamdan küçük olması (ör. beklenen %30, gözlenen %20)."),
                ("Etkileşim kontrastı", "(Birlikte) − (plastik) − (deterjan) + (kontrol) büyüme hızı farkı. Sıfırdan "
                 "anlamlı biçimde farklıysa etkileşim vardır."),
                ("Süzüntü kontrolü (G7, G8)", "Parçacıklar deterjanlı suda bekletilip süzülür, bitki yalnızca bu suda "
                 "yetiştirilir. Süzüntü grubu parçacıklı gruba benziyorsa etki suya geçen maddelerden, deterjan grubuna "
                 "benziyorsa parçacığın kendisinden kaynaklanır."),
            ],
            "İstatistik": [
                ("p değeri", "Gerçekte fark yokken bu kadar büyük bir farkın yalnızca rastlantıyla görülme olasılığı. "
                 "Küçük p, farkın rastlantıyla açıklanmasının zor olduğunu gösterir. Etkinin büyüklüğünü göstermez."),
                ("Anlamlılık düzeyi (α)", "Sonucun “anlamlı” sayılacağı sınır. Bu projede 0,05'tir (p < 0,05)."),
                ("Anlamlı fark saptanmadı", "Etkinin olmadığı anlamına gelmez. Etki küçükse veya kap sayısı azsa "
                 "saptanamamış olabilir."),
                ("İki faktörlü ANOVA", "Plastik türü ve deterjanın etkisini ve ikisinin etkileşimini aynı anda test eder."),
                ("Ana etki", "Bir faktörün (ör. deterjan) diğer faktörün düzeylerinin ortalamasındaki etkisi. Etkileşim "
                 "varsa tek başına yorumlanmamalıdır."),
                ("Kısmi η² (etki büyüklüğü)", "Bir etkinin açıkladığı değişkenlik oranı. 0,01 küçük, 0,06 orta, "
                 "0,14 ve üzeri büyük kabul edilir."),
                ("Tukey HSD", "Tüm grup çiftlerini karşılaştırır ve çok sayıda karşılaştırmadan doğan yanlış pozitif "
                 "riskini düzeltir."),
                ("Harf gösterimi (a, b, c)", "Aynı harfi paylaşan gruplar arasında anlamlı fark yoktur. Ortak harfi "
                 "olmayan gruplar birbirinden farklıdır."),
                ("Shapiro–Wilk testi", "Grup içi sapmaların (artıkların) normal dağılıp dağılmadığını sınar. p < 0,05 "
                 "ise normal dağılım varsayımı sağlanmaz."),
                ("Levene testi", "Grupların değişkenliğinin (varyansının) benzer olup olmadığını sınar."),
                ("Kruskal–Wallis testi", "Grupları sıralara göre karşılaştıran, normal dağılım gerektirmeyen test. "
                 "Etkileşimi sınamaz."),
                ("Permütasyon testi", "Veriyi binlerce kez rastgele yeniden düzenleyerek p değerini hesaplar. Normal "
                 "dağılım varsaymaz ve etkileşimi de sınar."),
                ("log(RGR) analizi", "Büyüme hızlarının logaritması alınarak yapılan ANOVA. Çarpık dağılımlarda "
                 "sonucun sağlamlığını kontrol eder."),
                ("Karma etkili model", "Her kabın tüm günlerdeki ölçümlerini birlikte analiz eder ve kaplar arası "
                 "farkları hesaba katar."),
                ("Yakınsama / sınır çözümü", "Model çözüme ulaşamazsa (yakınsamazsa) sonuçları raporlanmaz. Sınır "
                 "çözümü, kaplar arası ek değişkenliğin sıfır tahmin edildiğini gösterir."),
                ("Güven aralığı (%95)", "Gerçek değerin büyük olasılıkla içinde bulunduğu aralık. Aralık sıfırı "
                 "içeriyorsa fark anlamlı değildir."),
                ("SS ve SH", "Standart sapma (SS) kaplar arasındaki değişkenliği, standart hata (SH) ortalamanın ne kadar "
                 "kesin tahmin edildiğini gösterir."),
                ("Güç / saptanabilir en küçük fark", "Mevcut kap sayısıyla %80 olasılıkla yakalanabilecek en küçük fark. "
                 "Daha küçük etkiler için daha çok kap gerekir."),
                ("Kanıt düzeyi", "Güçlü: p < 0,01 ve testler uyumlu. Orta: 0,01 ≤ p < 0,05. Zayıf: sonuç teste göre "
                 "değişiyor. Kanıt yok: p ≥ 0,05."),
                ("Deney serisi (blok)", "Deneyin bağımsız olarak tekrarlanan her turu. Seriler arasındaki küçük farklar "
                 "(ör. farklı haftalarda ışık) analizde blok etkeni olarak hesaba katılır. Kap numarası 1–4 1. seri, "
                 "5–8 2. seri, 9–12 3. seridir."),
                ("Planlanmış karşılaştırma ve Holm düzeltmesi", "Deneyden önce belirlenen birkaç özel karşılaştırma "
                 "(ör. süzüntü grubu – deterjan grubu). Holm düzeltmesi, birden çok test yapıldığı için artan yanlış "
                 "pozitif riskini düzeltir."),
                ("Ön deney", "Deterjanın birkaç derişimde denendiği kısa deney. Büyümeyi %20–30 azaltan derişim ana "
                 "deney için seçilir. Ön deney ayrı bir çalışma klasöründe, kendi config.json dosyasıyla analiz "
                 "edilir."),
            ],
            "Ölçüm doğruluğu (H6)": [
                ("Lin CCC", "Otomatik ve manuel sayımın ne kadar örtüştüğü (1 = tam uyum). Bu projede 0,90 ve üzeri "
                 "yüksek uyum kabul edilir."),
                ("Bland–Altman grafiği", "İki yöntemin farkını ortalamalarına karşı gösterir. Ortadaki çizgi ortalama "
                 "farkı, kesikli çizgiler %95 uyum sınırlarını verir."),
                ("Ortalama fark (yanlılık)", "Otomatik − manuel sayımın ortalaması. Pozitifse otomatik yöntem fazla sayar."),
                ("MAE", "Ortalama mutlak hata: iki sayım arasındaki farkın ortalama büyüklüğü (yaprak)."),
                ("MAPE", "Ortalama mutlak yüzde hata: farkın manuel sayıma oranının ortalaması (%)."),
                ("Pearson r", "İki ölçümün birlikte artıp azalma derecesi. Uyumu tek başına göstermez, CCC ile "
                 "birlikte okunur."),
            ],
            "Program": [
                ("Çalıştırma (run)", "Bir analizin tüm sonuçları. Tarih-saat adlı ayrı klasöre kaydedilir ve önceki "
                 "sonuçlar silinmez."),
                ("Kalite uyarıları", "Sonuçlar kullanılmadan önce bakılması gereken durumlar: kap bulunamaması, eksik "
                 "fotoğraf, eksik manuel sayım vb."),
                ("Çalışma klasörü", "data: gerçek deney, demo: yazılımı denemek için yapay fotoğraflar."),
            ],
        }
        def tr_lower(text):
            return text.replace("İ", "i").replace("I", "ı").lower()

        query = tr_lower(st.text_input("Terim ara", placeholder="ör. inhibisyon, p değeri, CCC").strip())
        shown = 0
        for category, items in GLOSSARY.items():
            hits = [(t, e) for t, e in items if not query or query in tr_lower(t) or query in tr_lower(e)]
            if not hits:
                continue
            with st.expander(f"{category} ({len(hits)})", expanded=bool(query)):
                for term, text in hits:
                    st.markdown(f"**{term}** – {text}")
            shown += len(hits)
        if query and not shown:
            st.info("Bu kelimeyle eşleşen terim bulunamadı.")

    with help_tabs[2]:
        st.markdown(
            """
**Sonuçlar şu sırayla okunmalıdır:**

1. **Kalite uyarıları:** kap bulunamayan, eksik ya da hatalı fotoğraf varsa önce bunlar düzeltilir.
2. **Kontrol grubunun büyümesi:** ikiye katlanma süresi 2,5 günden kısa olmalıdır (OECD 221).
3. **Ana ölçüt:** manuel yaprak sayısından hesaplanan RGR. Alan ve otomatik sayım destekleyici bilgidir.
4. **İki faktörlü ANOVA:** deterjan, plastik türü ve etkileşim anlamlı mı? Etkileşim anlamlıysa ana etkiler
   tek başına yorumlanmaz.
5. **Sağlamlık kontrolü:** ANOVA, permütasyon testi ve log(RGR) aynı sonuca varıyor mu?
6. **Etkileşim kontrastları ve toplamsal beklenti:** birlikte etki sinerjik mi, antagonistik mi?
7. **Tukey ve harf gösterimi:** hangi gruplar birbirinden farklı?
8. **Güç:** anlamlı bulunmayan farklar için “etki yok” değil “saptanamadı” denir.
9. **H6:** otomatik ölçümler manuel sayımla ne kadar uyumlu?

**Yaygın hatalar:**
- Aynı kabın günlük fotoğraflarını bağımsız tekrar saymak (deney birimi kaptır).
- p > 0,05 sonucunu “etki yok” olarak yazmak.
- Yeşil alan yüzdesini biyokütle veya fotosentez ölçümü gibi yorumlamak.
- Sınırda anlamlı (p ≈ 0,04) bir sonucu tekrar deney yapmadan kesin bulgu olarak sunmak.
            """
        )
