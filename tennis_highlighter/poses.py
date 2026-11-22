"""v2 step 1: cut a short clip around every audio-detected hit and run YOLO pose on it.

Usage:  python poses.py INPUT_VIDEO [--data data]
Writes data/<video name>/:
  hits.csv            hit, time_s, strength, hitter_guess
  clips/hit_0001.mp4  640px clip, CLIP_BEFORE..CLIP_AFTER around the hit (for labelling)
  frames/hit_0001.jpg frame at the hit with detected skeletons drawn
  poses/hit_0001.npz  t (F,) seconds relative to hit,
                      kpts (F, MAX_PEOPLE, 17, 3) x_px, y_px, conf (zeros = nobody),
                      boxes (F, MAX_PEOPLE, 5) x1, y1, x2, y2, conf, frame_wh (2,), hit_t
Re-running skips hits that are already done, so it can be stopped and resumed.
"""
import argparse, csv, os, subprocess, time
import numpy as np
import highlighter as hl

_model = None


def session_dir(video, data="data"):
    return os.path.join(data, os.path.splitext(os.path.basename(video))[0])


def load_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(hl.POSE_MODEL)
    return _model


def cut_clip(video, t, out):
    start = max(0.0, t - hl.CLIP_BEFORE)
    subprocess.run([hl.FFMPEG, "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", video,
                    "-t", f"{t + hl.CLIP_AFTER - start:.3f}", "-vf", f"scale={hl.CLIP_WIDTH}:-2",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-movflags", "+faststart", out], check=True)
    return start


def pose_clip(clip, clip_start, hit_t, npz_out, jpg_out):
    import cv2
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, times, i = [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % hl.POSE_FRAME_STEP == 0:
            frames.append(fr); times.append(clip_start + i / fps - hit_t)
        i += 1
    cap.release()
    F, P = len(frames), hl.MAX_PEOPLE
    kpts = np.zeros((F, P, 17, 3), np.float32)
    boxes = np.zeros((F, P, 5), np.float32)
    if F == 0:
        raise RuntimeError(f"could not read frames from {clip}")
    results = load_model().predict(frames, imgsz=hl.CLIP_WIDTH, conf=0.25, verbose=False)
    for f, r in enumerate(results):
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            continue
        b = r.boxes.data.cpu().numpy()[:, :5]                      # x1 y1 x2 y2 conf
        k = r.keypoints.data.cpu().numpy()                          # (n, 17, 3)
        if k.shape[-1] == 2:
            k = np.concatenate([k, np.ones_like(k[..., :1])], -1)
        order = np.argsort(-(b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[:P]  # largest first
        boxes[f, :len(order)] = b[order]
        kpts[f, :len(order)] = k[order]
    times = np.array(times, np.float32)
    h, w = frames[0].shape[:2]
    np.savez_compressed(npz_out, t=times, kpts=kpts, boxes=boxes, frame_wh=np.array([w, h]),
                        hit_t=np.float32(hit_t))
    cv2.imwrite(jpg_out, results[int(np.argmin(np.abs(times)))].plot(), [cv2.IMWRITE_JPEG_QUALITY, 85])


def _done(npz, clip, jpg, t):
    """True if this hit was already extracted at the same time (settings unchanged)."""
    if not all(os.path.exists(f) for f in (npz, clip, jpg)):
        return False
    try:
        return abs(float(np.load(npz)["hit_t"]) - t) < 0.01
    except Exception:
        return False


def extract(video, data="data", progress=None):
    """Detect hits and extract clip + pose for each. progress(frac, desc) is optional."""
    sd = session_dir(video, data)
    for sub in ("clips", "frames", "poses"):
        os.makedirs(os.path.join(sd, sub), exist_ok=True)
    if progress: progress(0, "Listening for hits...")
    hits, strengths, dur = hl.detect_hits(video, return_strength=True)
    with open(os.path.join(sd, "hits.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["hit", "time_s", "strength", "hitter_guess"])
        for i, (t, s) in enumerate(zip(hits, strengths), 1):
            w.writerow([i, round(float(t), 3), round(float(s), 1),
                        "near" if s >= hl.NEAR_STRENGTH else "far"])
    t0 = time.time()
    for i, t in enumerate(hits, 1):
        name = f"hit_{i:04d}"
        clip = os.path.join(sd, "clips", name + ".mp4")
        npz = os.path.join(sd, "poses", name + ".npz")
        jpg = os.path.join(sd, "frames", name + ".jpg")
        if not _done(npz, clip, jpg, t):
            start = cut_clip(video, float(t), clip)
            pose_clip(clip, start, float(t), npz, jpg)
        msg = f"Hit {i}/{len(hits)} ({time.time() - t0:.0f}s)"
        if progress: progress(i / len(hits), msg)
        else: print("\r" + msg, end="", flush=True)
    if not progress: print()
    return sd, len(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--data", default="data")
    a = ap.parse_args()
    sd, n = extract(a.video, a.data)
    print(f"{n} hits extracted to {os.path.abspath(sd)}")


if __name__ == "__main__":
    main()
