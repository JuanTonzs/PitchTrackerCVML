import argparse

import difflib
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

PLATE_FRON_FT = 17 / 22 # The distance from edge to home plate

"""
THE DATA FROM HERE IS TAKEN FROM THE DATA SET I AM USING

"""
FEATURE_SETS = {
    "minimal": ["avg_speed", "vaa", "px", "pz"],
    "video": ["avg_speed", "vaa", "haa", "px", "pz", "pfx_x", "pfx_z", "x0", "z0"],
    "side": ["avg_speed", "vaa", "pz", "pfx_z", "z0"],
    "full": ["avg_speed", "start_speed", "end_speed", "vaa", "haa", "px", "pz", "pfx_x", "pfx_z",
             "x0", "z0", "spin_rate", "spin_dir", "break_angle", "break_length", "break_y"],
}



# ASSUMED measurement errors (1 standard deviation) for a decent phone setup.
# These are GUESSES. Replace them with real numbers once we compare videos
# estimates against known pitches (e.g. a radar gun reading).

NOISE_STD = {
    "avg_speed": 3.0, "start_speed": 3.0, "end_speed": 3.0,
    "vaa": 0.7, "haa": 0.7, # degrees
    "px": 0.25, "pz": 0.25, # feet
    "pfx_x": 3.0, "pfx_z": 3.0, # inches
    "x0": 0.5, "z0": 0.5, # feet
    "spin_rate": 300.0, "spin_dir": 30.0, # rpm, degrees
    "break_angle": 5.0, "break_length": 0.8, "break_y": 1.0 
}

# Labels
MERGE = {
    # Fast Balls
    "FF": "FB",
    "FT": "FB",
    "SI": "FB",

    # Cutter
    "FC": "CT",

    # Slider
    "SL": "SL",

    # Curveball
    "CU": "CU",
    "KC": "CU",

    # Off Speed Pitches
    "CH": "CH",
    "FS": "CH"
}

NOISE_LEVELS = [0, 0.5, 1, 2, 3] # multiples of NOISE_STD


def add_physics_features(df):
    """
    Since we need to figure out the physics of the ball via a guess we use
    the equation for displacement

    y(t) = y0 + vy0*t + 1/2 ay*t^2

    If vaa is less than 0 the ball is dropping
    haa is the sideways angle
    """

    a = 0.5 * df["ay"]
    b = df["vy0"]
    c = df["y0"] - PLATE_FRON_FT
    disc = b * b - 4 * a * c # Quadratic formula
    t = (-b - np.sqrt(disc.where(disc >= 0))) / (2 * a) # earlier roo = time to plate
    vx = df["vx0"] + df["ax"] * t
    vy = df["vy0"] + df["ay"] * t
    vz = df["vz0"] + df["az"] * t
    df["t_plate"] = t


    # Average speed along the pitch line from 50 ft to the plate. This is what a camera
    # watching release -> plate actually measures (it falls between start_speed and
    # end_speed, because the ball slows down in flight).
    df["avg_speed"] = (df["y0"] - PLATE_FRON_FT) / t * (3600 / 5280)
    df["vaa"] = np.degrees(np.arctan2(vz, vy.abs()))
    df["haa"] = np.degrees(np.arctan2(vx, vy.abs()))

    return df

def make_noisy(X, features, mult, rng):
    out = X.copy()
    for j, f in enumerate(features):
        out[:, j] += rng.normal(0, NOISE_STD.get(f, 0.0) * mult, size=len(out))


    return out


def fit(X, y):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--feature-set", choices=list(FEATURE_SETS), default="video",
                    help="set used for the detailed report, noise test and saved model")
    ap.add_argument("--features", default=None,
                    help="your own comma-separated feature list, e.g. avg_speed,vaa,pz,pfx_z,x0 "
                         "(overrides --feature-set; names are checked for typos)")
    ap.add_argument("--no-merge", action="store_true",
                    help="keep raw pitch codes (FF, FT, SI...) instead of merged groups")
    ap.add_argument("--min-count", type=int, default=2000, help="drop pitch types rarer than this")
    ap.add_argument("--max-rows", type=int, default=150000, help="subsample for speed")
    ap.add_argument("--out", default="pitch_model.joblib")
    args = ap.parse_args()
 
    if args.features:
        FEATURE_SETS["custom"] = [f.strip() for f in args.features.split(",") if f.strip()]
        args.feature_set = "custom"
    df = pd.read_csv(args.csv, low_memory=False)
    need = ["vx0", "vy0", "vz0", "ax", "ay", "az", "y0", "pitch_type"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing columns: {missing}")
    for c in set(sum(FEATURE_SETS.values(), [])) | set(need[:-1]):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = add_physics_features(df)
 
    if args.features:
        for f in FEATURE_SETS["custom"]:
            if f not in df.columns:
                close = difflib.get_close_matches(f, df.columns, n=3)
                hint = f"  Did you mean: {', '.join(close)}?" if close else ""
                raise SystemExit(f"Unknown feature '{f}'.{hint}\nAvailable columns: {', '.join(df.columns)}")
 
    df["label"] = df["pitch_type"] if args.no_merge else df["pitch_type"].map(MERGE)
    df = df.dropna(subset=["label"])
    counts = df["label"].value_counts()
    df = df[df["label"].isin(counts[counts >= args.min_count].index)]
    all_feats = sorted(set(sum(FEATURE_SETS.values(), [])))
    df = df.dropna(subset=[f for f in all_feats if f in df.columns])
    if len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=0)
    print(f"{len(df)} pitches, classes: {df['label'].value_counts().to_dict()}")
 
    # Split by at-bat (not by pitch) when possible, so related pitches don't leak.
    if "ab_id" in df.columns and df["ab_id"].notna().all():
        tr, te = next(GroupShuffleSplit(test_size=0.25, random_state=0).split(df, groups=df["ab_id"]))
    else:
        tr, te = train_test_split(np.arange(len(df)), test_size=0.25, random_state=0)
    df_tr, df_te = df.iloc[tr], df.iloc[te]
    y_tr, y_te = df_tr["label"].values, df_te["label"].values
 
    # ---- 1) what does each kind of information buy you? ----
    print("\n=== Accuracy by feature set (clean test data) ===")
    for name, feats in FEATURE_SETS.items():
        absent = [f for f in feats if f not in df.columns]
        if absent:
            print(f"{name:8s} skipped (CSV has no column: {', '.join(absent)})")
            continue
        m = fit(df_tr[feats].values, y_tr)
        p = m.predict(df_te[feats].values)
        print(f"{name:8s} accuracy {accuracy_score(y_te, p):.3f}   macro-F1 {f1_score(y_te, p, average='macro'):.3f}"
              f"   ({len(feats)} features)")
    print("(Majority-class baseline: "
          f"{pd.Series(y_te).value_counts(normalize=True).iloc[0]:.3f})")
 
    # ---- 2) detailed report on the chosen set ----
    feats = FEATURE_SETS[args.feature_set]
    X_tr, X_te = df_tr[feats].values, df_te[feats].values
    clean_model = fit(X_tr, y_tr)
    pred = clean_model.predict(X_te)
    print(f"\n=== Detailed report: '{args.feature_set}' features, clean data ===")
    print(classification_report(y_te, pred, digits=3))
    labels = sorted(set(y_te))
    cm = pd.DataFrame(confusion_matrix(y_te, pred, labels=labels), index=labels, columns=labels)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(cm.to_string())
 
    # ---- 3) the real question: noisy measurements ----
    rng = np.random.default_rng(0)
    X_aug = np.vstack([X_tr, make_noisy(X_tr, feats, 1.0, rng)])
    y_aug = np.concatenate([y_tr, y_tr])
    robust_model = fit(X_aug, y_aug)
 
    no_noise = [f for f in feats if f not in NOISE_STD]
    if no_noise:
        print(f"\nNOTE: no noise level is defined for {no_noise}, so they stay perfect in the noise test.")
        print("      Add them to NOISE_STD in the script to include them.")
    print(f"\n=== Accuracy as measurement noise grows ('{args.feature_set}' features) ===")
    print("noise level = multiple of the assumed video error (see NOISE_STD in the script)")
    print(f"{'noise':>6s}  {'trained on clean':>17s}  {'trained with noise':>19s}  {'macro-F1 (noise-trained)':>25s}")
    for m in NOISE_LEVELS:
        Xn = make_noisy(X_te, feats, m, np.random.default_rng(1)) if m else X_te
        a1 = accuracy_score(y_te, clean_model.predict(Xn))
        p2 = robust_model.predict(Xn)
        a2 = accuracy_score(y_te, p2)
        f2 = f1_score(y_te, p2, average="macro")
        print(f"{m:>5}x  {a1:>17.3f}  {a2:>19.3f}  {f2:>25.3f}")
 
    joblib.dump(dict(model=robust_model, features=feats, classes=list(robust_model.classes_)), args.out)
    print(f"\nSaved noise-tolerant model -> {args.out}  (expects features: {feats})")
 
 
if __name__ == "__main__":
    main()
