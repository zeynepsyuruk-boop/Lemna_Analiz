"""Command-line interface.

Examples:
    python lemna_cli.py init
    python lemna_cli.py check
    python lemna_cli.py preview --n 6
    python lemna_cli.py analyse
    python lemna_cli.py demo

All commands work on a workspace folder (default: data) that contains
photos/, manual_frond_counts.csv, daily_log.csv and results/.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from lemna.config import load_config, with_workspace_config
from lemna.imaging import analyse_image, read_image, resize_to_work
from lemna.naming import list_images, parse_name
from lemna.pipeline import run, write_templates

HERE = Path(__file__).resolve().parent


def cmd_init(args, cfg):
    ws = Path(args.workspace)
    for sub in ["photos", "results"]:
        (ws / sub).mkdir(parents=True, exist_ok=True)
    existed = [p.name for p in (ws / "manual_frond_counts.csv", ws / "daily_log.csv") if p.exists()]
    m, e = write_templates(ws, cfg, args.reps, range(args.first_day, args.last_day + 1))
    print(f"Folders ready: {ws / 'photos'}, {ws / 'results'}")
    print(f"Manual count sheet: {m}")
    print(f"Daily log sheet:    {e}")
    if existed:
        print("Kept existing sheets (not overwritten): " + ", ".join(existed))


def cmd_check(args, cfg):
    files = list_images(Path(args.workspace) / "photos")
    ok, bad = [], []
    for f in files:
        p = parse_name(f)
        (ok if p else bad).append((f, p))
    print(f"{len(files)} image files, {len(ok)} correctly named, {len(bad)} not recognised.")
    for f, _ in bad:
        print(f"  NOT RECOGNISED: {f.name}")
    unknown = sorted({p[0] for _, p in ok} - set(cfg["groups"]))
    if unknown:
        print(f"  Groups missing from config.json: {unknown}")
    dishes = Counter((p[0], p[1]) for _, p in ok)
    days = sorted({p[2] for _, p in ok})
    print(f"Dishes: {len(dishes)}   Days: {days}")
    missing = [f"{g}_{r} (missing days: {sorted(set(days) - {p[2] for _, p in ok if p[0] == g and p[1] == r})})"
               for g, r in sorted(dishes) if dishes[(g, r)] < len(days)]
    for m in missing:
        print(f"  INCOMPLETE: {m}")
    keys = Counter((p[0], p[1], p[2]) for _, p in ok)
    for k, n in keys.items():
        if n > 1:
            print(f"  DUPLICATE: {k[0]}_{k[1]}_Gun{k[2]} appears {n} times")


def cmd_preview(args, cfg):
    ws = Path(args.workspace)
    files = [f for f in list_images(ws / "photos") if parse_name(f)]
    if not files:
        sys.exit("No correctly named photos found.")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(files), size=min(args.n, len(files)), replace=False)
    tiles = []
    for i in sorted(pick):
        img = read_image(files[i])
        res, overlay, masks = analyse_image(img, cfg)
        work, _ = resize_to_work(img, cfg["work_max_side_px"])
        mask_rgb = cv2.cvtColor(masks["plant"], cv2.COLOR_GRAY2BGR)
        row = np.hstack([work, overlay, mask_rgb])
        h = 360
        row = cv2.resize(row, (int(row.shape[1] * h / row.shape[0]), h))
        cv2.putText(row, files[i].name, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(row)
    width = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 4, 0, width - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    out = ws / "results" / "preview.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", np.vstack(tiles))
    buf.tofile(str(out))
    print(f"Preview written to {out} (original | overlay | plant mask)")


def cmd_analyse(args, cfg):
    def progress(i, n, name):
        print(f"\r[{i}/{n}] {name}            ", end="", flush=True)

    ws = Path(args.workspace)
    res = run(ws / "photos", cfg, ws / "results", ws / "manual_frond_counts.csv", not args.no_overlays, progress,
              note=args.note)
    print()
    if res["measurements"].empty:
        sys.exit(res["report"])
    print(res["report"])
    print(f"\nRun {res['run_id']} written to {res['out_dir'].resolve()}")
    print(f"History files: {(ws / 'results').resolve()}/history_*.csv")


def cmd_report(args, cfg):
    from lemna.pipeline import list_runs
    from lemna.report import build_report
    results = Path(args.workspace) / "results"
    runs = list_runs(results)
    if not runs:
        sys.exit("No analysis runs found. Run 'analyse' first.")
    run_id = args.run or runs[0]
    if run_id not in runs:
        sys.exit(f"Run {run_id} not found. Available: {', '.join(runs)}")
    path = build_report(results / "runs" / run_id, author=args.author)
    print(f"Report written to {path}")


def cmd_demo(args, cfg):
    from lemna.demo import make_demo
    photo_dir, manual = make_demo(HERE / "demo")
    print(f"Synthetic photos: {photo_dir}")
    res = run(photo_dir, cfg, HERE / "demo" / "results", manual, note="demo")
    print(res["report"])


def main():
    ap = argparse.ArgumentParser(description="Lemna minor image analysis")
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--workspace", default=str(HERE / "data"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create folders and blank data sheets")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--first-day", type=int, default=0)
    p.add_argument("--last-day", type=int, default=7)

    sub.add_parser("check", help="check photo file names")

    p = sub.add_parser("preview", help="save segmentation previews for a few photos")
    p.add_argument("--n", type=int, default=6)

    p = sub.add_parser("analyse", help="analyse all photos")
    p.add_argument("--no-overlays", action="store_true")
    p.add_argument("--note", default="", help="short note stored with this run")

    p = sub.add_parser("report", help="write the Word report of an analysis run")
    p.add_argument("--run", default=None, help="run id (default: the latest run)")
    p.add_argument("--author", default="")

    sub.add_parser("demo", help="generate synthetic photos and run the full analysis")

    args = ap.parse_args()
    ws = Path(args.workspace)
    if not ws.is_absolute() and not ws.exists() and (HERE / ws).exists():
        args.workspace = str(HERE / ws)
    cfg = with_workspace_config(load_config(args.config), Path(args.workspace) / "config.json")
    {"init": cmd_init, "check": cmd_check, "preview": cmd_preview,
     "analyse": cmd_analyse, "report": cmd_report, "demo": cmd_demo}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
