"""v2 step 3: label every hit with the near player's stroke, using stroke_model.pt.

Usage:  python classify.py VIDEO [--data data] [--model stroke_model.pt]
        python highlighter.py VIDEO --out DIR --strokes   (same, plus v1 point clips)
Runs poses.py's extraction first if it hasn't been done (about 5 s per hit, resumable),
then writes data/<video>/strokes.csv: hit, time_s, strength, swing, stroke, confidence.

stroke is one of the model's labels (e.g. forehand / backhand / not_a_hit), or
  opponent  the near player isn't swinging (swing < NEAR_SWING), so the other player hit it
  unknown   the near player wasn't detected around the hit
The model only knows the classes it was trained on: a serve gets forced into one of them.
"""
import argparse, csv, os
import numpy as np
import highlighter as hl
import poses, strokes

FIELDS = ["hit", "time_s", "strength", "swing", "stroke", "confidence"]
SHORT = {"forehand": "F", "backhand": "B", "serve": "S", "volley": "V", "overhead": "O"}


def classify(video, data="data", model_path=hl.STROKE_MODEL, progress=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found: train it with train_strokes.ipynb first")
    predict = strokes.load_classifier(model_path)
    sd, n = poses.extract(video, data, progress=progress)
    with open(os.path.join(sd, "hits.csv"), newline="") as fh:
        hits = list(csv.DictReader(fh))
    rows, X, todo = [], [], []
    for h in hits:
        npz = os.path.join(sd, "poses", f"hit_{int(h['hit']):04d}.npz")
        swing = strokes.swing_score(npz)
        row = {"hit": int(h["hit"]), "time_s": float(h["time_s"]), "strength": float(h["strength"]),
               "swing": round(swing, 1), "stroke": "opponent", "confidence": ""}
        if swing >= hl.NEAR_SWING:
            f = strokes.hit_features(npz, "near")
            if (np.abs(f).sum(1) > 0).mean() < 0.5:          # near player missing in most frames
                row["stroke"] = "unknown"
            else:
                X.append(f); todo.append(row)
        rows.append(row)
    if X:
        labels, probs = predict(np.stack(X))
        for row, p in zip(todo, probs):
            row["stroke"], row["confidence"] = labels[int(p.argmax())], round(float(p.max()), 2)
    with open(os.path.join(sd, "strokes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS); w.writeheader(); w.writerows(rows)
    return rows


def point_of(t, points):
    """1-based point number containing time t, or "" if the hit is outside every point."""
    return next((i for i, p in enumerate(points, 1) if p["start"] <= t <= p["end"]), "")


def per_point(rows, points):
    """Columns for highlighter's points.csv: near-player stroke counts and sequence per point."""
    kinds = [k for k in SHORT if any(r["stroke"] == k for r in rows)]
    cols = {f"near_{k}s": [0] * len(points) for k in kinds}
    seqs = [[] for _ in points]
    for r in rows:
        i = point_of(r["time_s"], points)
        if i and r["stroke"] in SHORT:
            cols[f"near_{r['stroke']}s"][i - 1] += 1
            seqs[i - 1].append(SHORT[r["stroke"]])
    cols["near_strokes"] = [" ".join(s) for s in seqs]
    return cols


def write_hits(rows, path, points):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["point"] + FIELDS); w.writeheader()
        w.writerows(dict(r, point=point_of(r["time_s"], points)) for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--data", default="data")
    ap.add_argument("--model", default=hl.STROKE_MODEL)
    a = ap.parse_args()
    rows = classify(a.video, a.data, a.model)
    counts = {}
    for r in rows:
        counts[r["stroke"]] = counts.get(r["stroke"], 0) + 1
    print(f"{len(rows)} hits: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    print("Saved", os.path.join(poses.session_dir(a.video, a.data), "strokes.csv"))


if __name__ == "__main__":
    main()
