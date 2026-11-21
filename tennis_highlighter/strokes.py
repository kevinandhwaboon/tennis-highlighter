"""v2 shared stroke code: labels, picking the hitter's skeleton, features, dataset export.

Used by label_app.py (export) and, later, by the local classifier, so training in
Colab and inference on the Mac see exactly the same features.

Usage:  python strokes.py [--data data]   -> data/dataset.npz
"""
import argparse, csv, glob, os
import numpy as np
import highlighter as hl

LABELS = ["forehand", "backhand", "serve", "volley", "overhead", "not_a_hit"]
HITTERS = ["near", "far"]
POSE_FPS = 30.0 / hl.POSE_FRAME_STEP
GRID = np.round(np.arange(-hl.POSE_WINDOW, hl.POSE_WINDOW + 1e-6, 1 / POSE_FPS), 4)  # e.g. 31 steps
# COCO keypoints: 5/6 shoulders, 11/12 hips
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # left<->right keypoint swap
N_FEATURES = 17 * 3 + 2


def select_track(kpts, boxes, t, hitter):
    """Follow one person through the clip. At the frame nearest the hit pick the near
    player (box bottom lowest in the image) or far player (box bottom highest), then
    in other frames take whoever's box centre is closest to the previous pick.
    Returns (F, 17, 3) and (F, 5); frames with nobody are zeros."""
    F = len(t)
    trk_k = np.zeros((F, 17, 3), np.float32); trk_b = np.zeros((F, 5), np.float32)
    present = boxes[..., 4] > 0
    cand = [f for f in np.argsort(np.abs(t)) if present[f].any()]
    if not cand:
        return trk_k, trk_b
    f0 = cand[0]
    idx = np.where(present[f0])[0]
    bottoms = boxes[f0, idx, 3]
    p0 = idx[np.argmax(bottoms)] if hitter == "near" else idx[np.argmin(bottoms)]
    trk_k[f0], trk_b[f0] = kpts[f0, p0], boxes[f0, p0]

    def centre(b): return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

    for direction in (1, -1):
        prev = trk_b[f0]
        f = f0 + direction
        while 0 <= f < F:
            idx = np.where(present[f])[0]
            if len(idx):
                d = [np.linalg.norm(centre(boxes[f, i]) - centre(prev)) for i in idx]
                j = idx[int(np.argmin(d))]
                # reject jumps bigger than 1.5 box heights (switched to someone else)
                if min(d) < 1.5 * max(prev[3] - prev[1], 1):
                    trk_k[f], trk_b[f] = kpts[f, j], boxes[f, j]
                    prev = trk_b[f]
            f += direction
    return trk_k, trk_b


def features(t, trk_k, trk_b, hitter):
    """Resample the track to GRID and normalise to body size.
    Per frame: 17 x (x, y) relative to mid-hip, / torso length, + 17 confidences,
    + mid-hip position relative to the hit frame (captures serve jump / running).
    The far player faces the camera, so their image is mirrored: flip x and swap
    left/right keypoints so a forehand looks the same from either end.
    Returns (len(GRID), N_FEATURES) float32."""
    out = np.zeros((len(GRID), N_FEATURES), np.float32)
    ok = trk_b[:, 4] > 0
    if not ok.any():
        return out
    k = trk_k.copy()
    if hitter == "far":
        k[..., 0] = -k[..., 0]
        k = k[:, FLIP]
    hip = (k[:, L_HIP, :2] + k[:, R_HIP, :2]) / 2
    sh = (k[:, L_SH, :2] + k[:, R_SH, :2]) / 2
    good = ok & (k[:, [L_HIP, R_HIP, L_SH, R_SH], 2].min(1) > 0.3)
    if good.any():
        scale = float(np.median(np.linalg.norm(sh[good] - hip[good], axis=1)))
    else:
        scale = float(np.median(trk_b[ok, 3] - trk_b[ok, 1])) / 3
    scale = max(scale, 1.0)
    f_hit = int(np.argmin(np.where(ok, np.abs(t), np.inf)))
    for g, gt in enumerate(GRID):
        f = int(np.argmin(np.abs(t - gt)))
        if abs(t[f] - gt) > 0.6 / POSE_FPS or not ok[f]:
            continue
        rel = (k[f, :, :2] - hip[f]) / scale
        out[g, :34] = rel.reshape(-1)
        out[g, 34:51] = k[f, :, 2]
        out[g, 51:53] = (hip[f] - hip[f_hit]) / scale
    return out


def hit_features(npz_path, hitter):
    d = np.load(npz_path)
    trk_k, trk_b = select_track(d["kpts"], d["boxes"], d["t"], hitter)
    return features(d["t"], trk_k, trk_b, hitter)


def read_labels(session):
    path = os.path.join(session, "labels.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as fh:
        return {int(r["hit"]): r for r in csv.DictReader(fh)}


def export_dataset(data="data", out=None):
    """Collect every labelled hit across all sessions into one .npz for Colab."""
    out = out or os.path.join(data, "dataset.npz")
    X, y, hitter, video, hit, time_s = [], [], [], [], [], []
    for sd in sorted(glob.glob(os.path.join(data, "*", ""))):
        for h, r in sorted(read_labels(sd).items()):
            npz = os.path.join(sd, "poses", f"hit_{h:04d}.npz")
            if r.get("label") not in LABELS or not os.path.exists(npz):
                continue
            hh = r.get("hitter") or "near"
            X.append(hit_features(npz, hh)); y.append(LABELS.index(r["label"]))
            hitter.append(hh); video.append(os.path.basename(os.path.normpath(sd)))
            hit.append(h); time_s.append(float(r["time_s"]))
    if not X:
        raise ValueError("No labelled hits found yet.")
    np.savez_compressed(out, X=np.stack(X), y=np.array(y), labels=np.array(LABELS),
                        hitter=np.array(hitter), video=np.array(video), hit=np.array(hit),
                        time_s=np.array(time_s), grid=GRID)
    counts = {l: int((np.array(y) == i).sum()) for i, l in enumerate(LABELS)}
    return out, len(X), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    out, n, counts = export_dataset(a.data)
    print(f"{n} labelled hits -> {out}")
    for l, c in counts.items():
        print(f"  {l:10s} {c}")


if __name__ == "__main__":
    main()
