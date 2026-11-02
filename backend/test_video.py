"""
test_clip.py -- full-clip test for your trained baseball detector.

What it does:
  1. Runs your trained model on EVERY frame of a clip at a very low confidence
     threshold, keeping every box it finds (real ball AND false alarms).
  2. Looks for smooth arcs (a thrown ball's path) that many detections agree on,
     using RANSAC. It reports up to 3 separate tracks (e.g. the ball AND a hand). Detections that don't fit the arc are treated as
     false alarms. Stationary things (logos, gloves) are not allowed to count
     as an arc, because a ball in flight has to actually travel.
  3. Saves an annotated video, CSVs, a plot, and prints a summary.

Usage (Colab or local):
  python test_clip.py pitch.mp4 --weights path/to/best.pt
  python test_clip.py pitch.mp4 --weights best.pt --conf 0.03 --tol 20

IMPORTANT: use a clip that was NOT in your training data. Testing on a clip the
model trained on makes the results look better than they really are.

Outputs (in --out-dir, default "clip_test"):
  annotated.mp4       all detections (gray = rejected, green = on the arc),
                      the fitted arc in yellow, confidence next to each box
  raw_detections.csv  every box the model produced
  arc_detections.csv  the detections kept on the arc (one per frame), with fit
  arc_plot.png        x/y positions and the fitted arc (needs matplotlib)
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def to_np(t):
    """Ultralytics gives torch tensors; accept plain arrays too."""
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


# ---------------------------------------------------------------------------
# Step 1: run the detector on every frame
# ---------------------------------------------------------------------------
def run_detection(model, video, conf, imgsz, device):
    """Returns (dets, n_frames). dets rows: frame, x, y, w, h, conf (pixels)."""
    kwargs = dict(source=video, stream=True, conf=conf, imgsz=imgsz, verbose=False)
    if device is not None:
        kwargs["device"] = device

    rows, n_frames = [], 0
    for i, r in enumerate(model.predict(**kwargs)):
        n_frames = i + 1
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xywh, cs = to_np(r.boxes.xywh), to_np(r.boxes.conf)
        for (x, y, w, h), c in zip(xywh, cs):
            rows.append((i, float(x), float(y), float(w), float(h), float(c)))
    return np.array(rows, dtype=float).reshape(-1, 6), n_frames


# ---------------------------------------------------------------------------
# Step 2: find the one arc that most detections agree on (RANSAC)
# ---------------------------------------------------------------------------
def travel(xs, ys):
    """How far apart the points are (diagonal of their bounding box), in pixels."""
    return float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))


def find_arc(dets, fps, tol=15.0, min_travel=60.0, min_frames=8,
             x_deg=1, iters=4000, seed=0):
    """
    Model of a thrown ball in the image:
        x(t) = polynomial of degree x_deg (1 = straight line sideways)
        y(t) = quadratic (gravity makes a parabola)

    RANSAC idea: repeat thousands of times -- pick 3 random detections, fit a
    curve through them, count how many OTHER detections lie within `tol` pixels
    of that curve (counting each frame once). Keep the curve with the most
    frames supporting it. Real ball hits all lie on one curve; false alarms
    are scattered, so they can't agree on one.

    A curve only counts if its supporters move at least `min_travel` pixels.
    Otherwise a logo that fires on every frame would "win".
    """
    if len(dets) < 3:
        return None
    rng = np.random.default_rng(seed)
    frame, x, y = dets[:, 0], dets[:, 1], dets[:, 2]
    tt = frame / fps - (frame.min() / fps)       # seconds, starting at 0
    N = len(dets)

    best_mask, best_count = None, 0
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        f = frame[idx]
        if len(np.unique(f)) < 3 or f.max() - f.min() < 3:
            continue                              # need 3 different, spread-out frames
        px = np.polyfit(tt[idx], x[idx], x_deg)
        py = np.polyfit(tt[idx], y[idx], 2)
        res = np.hypot(np.polyval(px, tt) - x, np.polyval(py, tt) - y)
        mask = res < tol
        count = len(np.unique(frame[mask]))       # frames supported, not boxes
        if count <= best_count:
            continue
        if travel(x[mask], y[mask]) < min_travel:
            continue                              # stationary -> not a trajectory
        best_mask, best_count = mask, count

    if best_mask is None or best_count < min_frames:
        return None

    # Refine: refit on all supporters (least squares), then re-check who fits.
    mask = best_mask
    for _ in range(3):
        px = np.polyfit(tt[mask], x[mask], x_deg)
        py = np.polyfit(tt[mask], y[mask], 2)
        res = np.hypot(np.polyval(px, tt) - x, np.polyval(py, tt) - y)
        new = res < tol
        if new.sum() < 3 or travel(x[new], y[new]) < min_travel:
            break
        mask = new
    px = np.polyfit(tt[mask], x[mask], x_deg)
    py = np.polyfit(tt[mask], y[mask], 2)
    res = np.hypot(np.polyval(px, tt) - x, np.polyval(py, tt) - y)

    # Keep ONE detection per frame: the one closest to the curve.
    chosen = {}
    for i in np.where(mask)[0]:
        f = int(frame[i])
        if f not in chosen or res[i] < res[chosen[f]]:
            chosen[f] = i
    sel = np.array(sorted(chosen.values(), key=lambda i: frame[i]), dtype=int)

    return dict(px=px, py=py, t_offset=frame.min() / fps, fps=fps, x_deg=x_deg,
                sel=sel, members=np.where(mask)[0],
                first=int(frame[sel[0]]), last=int(frame[sel[-1]]))


def arc_point(arc, f):
    """Fitted (x, y) of the arc at frame f."""
    t = f / arc["fps"] - arc["t_offset"]
    return float(np.polyval(arc["px"], t)), float(np.polyval(arc["py"], t))


def drop_static(dets, radius=6.0, min_frames=10):
    """
    Marks detections that sit in (almost) the same spot in many different frames
    -- a logo, a banner, a glove on a still player. A ball in flight moves, so it
    never stays within `radius` px for `min_frames` frames. Returns a boolean
    mask of detections to KEEP.
    (If you film at a very high frame rate, raise --static-frames.)
    """
    n = len(dets)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    frame, x, y = dets[:, 0], dets[:, 1], dets[:, 2]
    for i in range(n):
        near = np.hypot(x - x[i], y - y[i]) < radius
        if len(np.unique(frame[near])) >= min_frames:
            keep[i] = False
    return keep


def arc_stats(arc, dets, fps):
    """Adds size/density/speed numbers to an arc."""
    sel = arc["sel"]
    n = len(sel)
    span = arc["last"] - arc["first"] + 1
    dist = float(np.hypot(dets[sel[-1], 1] - dets[sel[0], 1], dets[sel[-1], 2] - dets[sel[0], 2]))
    frames_elapsed = max(arc["last"] - arc["first"], 1)
    arc.update(n=n, span=span, coverage=n / span,
               quality=n * n / span,                 # rewards BOTH many frames and a dense track
               travel=dist, speed_pf=dist / frames_elapsed)   # pixels per frame
    return arc


def locate_arcs(dets, fps, static_radius=6.0, static_frames=10, chance_trials=5,
                max_arcs=3, min_speed=0.0, **arc_kwargs):
    """
    1. Drop stationary detections (logos etc.).
    2. Find the best arc, remove its detections, repeat -> up to `max_arcs` tracks.
       (A clip can hold a ball AND other smooth movers, like a hand or glove.)
    3. Estimate the chance level: scramble the frame order of the same detections
       and see how good an arc that produces. Quality = frames * density, so a
       thin, spread-out chance arc scores low and a tight real track scores high.
    4. An arc is CONVINCING if it has enough frames and its quality is at least
       twice the chance quality.
    Returns (arcs, info).
    """
    keep = drop_static(dets, static_radius, static_frames)
    pool = np.where(keep)[0]

    arcs = []
    for _ in range(max_arcs):
        if len(pool) < 3:
            break
        arc = find_arc(dets[pool], fps, **arc_kwargs)
        if arc is None:
            break
        arc["sel"] = pool[arc["sel"]]
        arc["members"] = pool[arc["members"]]
        pool = np.setdiff1d(pool, arc["members"])     # these detections are now used up
        arc_stats(arc, dets, fps)
        if arc["speed_pf"] >= min_speed:              # optionally hide slow movers
            arcs.append(arc)

    kept = dets[keep]
    chance_q = 0.0
    kw = {k: v for k, v in arc_kwargs.items() if k != "seed"}
    for s_ in range(chance_trials):
        shuffled = kept.copy()
        shuffled[:, 0] = np.random.default_rng(100 + s_).permutation(shuffled[:, 0])
        a = find_arc(shuffled, fps, seed=100 + s_, **kw)
        if a is not None:
            n = len(a["sel"])
            span = a["last"] - a["first"] + 1
            chance_q = max(chance_q, n * n / span)

    min_frames = arc_kwargs.get("min_frames", 8)
    for a in arcs:
        a["convincing"] = a["n"] >= min_frames and a["quality"] >= 2 * chance_q
    return arcs, dict(n_static=int((~keep).sum()), chance_q=chance_q)


# ---------------------------------------------------------------------------
# Step 3: summary, files, overlay video
# ---------------------------------------------------------------------------
def summarize(dets, n_frames, arcs, info, min_frames):
    lines = []
    n_det = len(dets)
    frames_with = len(np.unique(dets[:, 0])) if n_det else 0
    lines.append(f"Frames in clip:            {n_frames}")
    lines.append(f"Frames with any detection: {frames_with} ({100 * frames_with / max(n_frames, 1):.0f}%)")
    lines.append(f"Total detections:          {n_det}  (avg {n_det / max(n_frames, 1):.2f} per frame)")
    if n_det:
        c = dets[:, 5]
        lines.append("Confidence spread:         "
                     f"<0.10: {(c < 0.10).sum()}   0.10-0.25: {((c >= 0.10) & (c < 0.25)).sum()}   "
                     f"0.25-0.50: {((c >= 0.25) & (c < 0.50)).sum()}   >=0.50: {(c >= 0.50).sum()}")
    lines.append(f"Dropped as stationary (logos, etc.): {info['n_static']} detections")
    lines.append("")
    if not arcs:
        lines.append("TRACKS: none found. The model saw the ball in too few frames, or false alarms")
        lines.append("        drowned it out. Try a lower --conf or larger --tol, and watch annotated.mp4.")
        return "\n".join(lines)

    lines.append(f"TRACKS FOUND: {len(arcs)}   (chance level, same measure: {info['chance_q']:.1f})")
    for k, a in enumerate(arcs, 1):
        verdict = "CONVINCING" if a["convincing"] else "not convincing"
        lines.append(f"  #{k}  frames {a['first']}-{a['last']}   {a['n']} of {a['span']} frames "
                     f"({100 * a['coverage']:.0f}% dense)   moves {a['travel']:.0f} px "
                     f"at {a['speed_pf']:.1f} px/frame   quality {a['quality']:.1f}   -> {verdict}")
    lines.append("")
    lines.append("How to read this:")
    lines.append("  - Dense + convincing means a smooth track the model followed. It is NOT proof it is")
    lines.append("    the ball: hands, gloves and bats can also give smooth tracks.")
    lines.append("  - The ball is usually the FASTEST track. Slow ones (a few px/frame) are often a")
    lines.append("    body part. Hide them with --min-speed (px/frame).")
    lines.append("  - Always confirm in annotated.mp4 (each track has its own colour).")
    return "\n".join(lines)


def save_csvs(out_dir, dets, arcs, fps):
    with open(out_dir / "raw_detections.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "x_px", "y_px", "w_px", "h_px", "conf"])
        for fr, x, y, bw, bh, c in dets:
            w.writerow([int(fr), round(fr / fps, 5), round(x, 1), round(y, 1),
                        round(bw, 1), round(bh, 1), round(c, 3)])
    if arcs:
        with open(out_dir / "arc_detections.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["arc_id", "convincing", "frame", "time_s", "x_px", "y_px", "conf",
                        "fit_x_px", "fit_y_px"])
            for k, a in enumerate(arcs, 1):
                for i in a["sel"]:
                    fr, x, y, _, _, c = dets[i]
                    fx, fy = arc_point(a, int(fr))
                    w.writerow([k, int(a["convincing"]), int(fr), round(fr / fps, 5),
                                round(x, 1), round(y, 1), round(c, 3), round(fx, 1), round(fy, 1)])


PALETTE = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]        # BGR: yellow, magenta, cyan
PLOT_COLORS = ["#d4a000", "#c000c0", "#0090c0"]
COLOR_NAMES = ["yellow", "magenta", "cyan"]


def render_video(video, out_path, dets, arcs, fps):
    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    by_frame = defaultdict(list)
    for di, row in enumerate(dets):
        by_frame[int(row[0])].append(di)
    owner = {}                                   # detection index -> arc number
    curves = []
    for k, a in enumerate(arcs):
        for di in a["sel"]:
            owner[int(di)] = k
        curves.append({f: arc_point(a, f) for f in range(a["first"], a["last"] + 1)})

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for di in by_frame.get(i, []):
            _, x, y, _, _, c = dets[di]
            k = owner.get(di)
            color = PALETTE[k % len(PALETTE)] if k is not None else (170, 170, 170)
            r = 11 if k is not None else 7
            cv2.circle(frame, (int(x), int(y)), r, color, 2 if k is not None else 1)
            cv2.putText(frame, f"{c:.2f}", (int(x) + r + 2, int(y) - r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        for k, a in enumerate(arcs):
            if i >= a["first"]:
                upto = min(i, a["last"])
                pts = np.array([curves[k][f] for f in range(a["first"], upto + 1)], dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False,
                                  PALETTE[k % len(PALETTE)], 3 if a["convincing"] else 1)
        cv2.putText(frame, f"frame {i}  boxes {len(by_frame.get(i, []))}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()


def save_plot(out_path, dets, arcs, fps):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed, skipping arc_plot.png: pip install matplotlib)")
        return
    if len(dets) == 0:
        return
    fr, x, y = dets[:, 0], dets[:, 1], dets[:, 2]
    owner = np.full(len(dets), -1)
    for k, a in enumerate(arcs):
        owner[a["sel"]] = k

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    views = [(x, y, "x (px)", "y (px)"), (fr, x, "frame", "x (px)"), (fr, y, "frame", "y (px)")]
    for a_, (u, v, ul, vl) in zip(ax, views):
        m = owner == -1
        a_.scatter(u[m], v[m], s=12, c="gray", alpha=0.5, label="rejected")
        for k, a in enumerate(arcs):
            mk = owner == k
            a_.scatter(u[mk], v[mk], s=24, c=PLOT_COLORS[k % 3],
                       label=f"track #{k + 1} ({COLOR_NAMES[k % 3]})")
        a_.set_xlabel(ul)
        a_.set_ylabel(vl)
    ax[0].invert_yaxis()
    ax[2].invert_yaxis()
    for k, a in enumerate(arcs):
        fs = np.arange(a["first"], a["last"] + 1)
        pts = np.array([arc_point(a, int(f)) for f in fs])
        ax[0].plot(pts[:, 0], pts[:, 1], "-", c=PLOT_COLORS[k % 3], lw=2)
        ax[1].plot(fs, pts[:, 0], "-", c=PLOT_COLORS[k % 3], lw=2)
        ax[2].plot(fs, pts[:, 1], "-", c=PLOT_COLORS[k % 3], lw=2)
    ax[0].legend(fontsize=8)
    ax[0].set_title("Paths in the image")
    ax[1].set_title("x over time")
    ax[2].set_title("y over time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Full-clip test of a trained baseball detector.")
    ap.add_argument("video")
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--conf", type=float, default=0.05, help="detection threshold (low on purpose)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None, help="e.g. 0 for the first GPU, or cpu")
    ap.add_argument("--tol", type=float, default=15.0, help="max pixels a detection may be from the arc")
    ap.add_argument("--min-travel", type=float, default=60.0, help="min pixels the ball must move")
    ap.add_argument("--min-frames", type=int, default=8, help="min frames needed to call it an arc")
    ap.add_argument("--x-deg", type=int, default=1, choices=[1, 2],
                    help="horizontal motion model: 1 = straight, 2 = curved (try 2 if the arc is cut short)")
    ap.add_argument("--static-radius", type=float, default=6.0,
                    help="detections within this many px across many frames count as stationary")
    ap.add_argument("--static-frames", type=int, default=10,
                    help="how many frames in one spot makes a detection 'stationary'")
    ap.add_argument("--max-arcs", type=int, default=3, help="how many separate tracks to report")
    ap.add_argument("--min-speed", type=float, default=0.0,
                    help="hide tracks slower than this many pixels per frame (e.g. 8 hides hands/gloves)")
    ap.add_argument("--out-dir", default="clip_test")
    args = ap.parse_args()

    from ultralytics import YOLO          # imported here so the helpers can be tested without it

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS) or 30.0
    device = int(args.device) if args.device and args.device.isdigit() else args.device

    print(f"Running {args.weights} on {args.video} at conf={args.conf} ...")
    model = YOLO(args.weights)
    dets, n_frames = run_detection(model, args.video, args.conf, args.imgsz, device)

    arcs, info = locate_arcs(dets, fps, static_radius=args.static_radius,
                             static_frames=args.static_frames, max_arcs=args.max_arcs,
                             min_speed=args.min_speed, tol=args.tol,
                             min_travel=args.min_travel, min_frames=args.min_frames,
                             x_deg=args.x_deg)

    print("\n" + summarize(dets, n_frames, arcs, info, args.min_frames) + "\n")
    save_csvs(out_dir, dets, arcs, fps)
    render_video(args.video, out_dir / "annotated.mp4", dets, arcs, fps)
    save_plot(out_dir / "arc_plot.png", dets, arcs, fps)
    print(f"Saved to {out_dir}/: annotated.mp4, raw_detections.csv"
          + (", arc_detections.csv" if arcs else "") + ", arc_plot.png")


if __name__ == "__main__":
    main()