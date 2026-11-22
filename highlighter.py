"""Tennis Highlighter v1: cut dead time from a match video using racket-hit sounds.

Usage:  python highlighter.py INPUT_VIDEO [--out OUTDIR]
Outputs: OUTDIR/points/point_XX.mp4, OUTDIR/highlights.mp4, OUTDIR/points.csv
"""
import argparse, csv, os, shutil, subprocess, tempfile
import numpy as np, librosa, scipy.signal as ss


def _find_ffmpeg():
    """System ffmpeg if installed, else the static binary from the imageio-ffmpeg pip package."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("ffmpeg not found: `pip install imageio-ffmpeg` or `brew install ffmpeg`")


FFMPEG = _find_ffmpeg()

# ---- tunable settings ----
BAND = (1500, 6000)     # Hz range where the racket "pop" lives
HIT_THRESHOLD = 6.0     # peak height (x median loudness) to count as a hit
MIN_HIT_SPACING = 0.5   # s, ignore peaks closer than this
MAX_GAP = 4.5           # s, hits further apart than this start a new point
MIN_HITS = 2            # a point needs at least this many hits
PAD_BEFORE = 2.0        # s of lead-in before first hit (serve toss)
PAD_AFTER = 2.0         # s after last hit (ball lands / point ends)

# ---- v2: stroke labelling (used by poses.py / strokes.py / label_app.py) ----
POSE_MODEL = "yolo11n-pose.pt"  # smallest Ultralytics pose model, auto-downloads (~6 MB)
CLIP_BEFORE = 1.5       # s of video before each hit in the labelling clip (serve toss)
CLIP_AFTER = 1.0        # s after
CLIP_WIDTH = 640        # px, clips + pose inference run at this width (CPU budget)
POSE_FRAME_STEP = 2     # run pose on every Nth frame (30 fps -> 15 pose fps)
POSE_WINDOW = 1.0       # s either side of the hit used for the classifier input
MAX_PEOPLE = 4          # keep up to this many people per frame (largest boxes)
NEAR_STRENGTH = 9.0     # audio peak >= this (x median) -> guess near player hit it
NEAR_ONLY = True        # label_app only asks about the near (visible) player's hits
NEAR_SWING = 2.0        # near player's peak wrist speed (body heights/s) near the hit to count as their swing
SWING_WINDOW = 0.35     # s either side of the hit to look for that swing


def detect_hits(video, return_strength=False):
    """Return (hit_times, duration). With return_strength=True, return
    (hit_times, strengths, duration), strength = peak height in x median."""
    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "a.wav")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", video, "-vn", "-ac", "1",
                        "-ar", "22050", wav], check=True)
        y, sr = librosa.load(wav, sr=22050)
    b, a = ss.butter(4, BAND, btype="band", fs=sr)
    hop = 256
    env = librosa.onset.onset_strength(y=ss.filtfilt(b, a, y), sr=sr, hop_length=hop)
    env = env / np.median(env)
    pk, props = ss.find_peaks(env, height=HIT_THRESHOLD, distance=int(MIN_HIT_SPACING * sr / hop))
    times = librosa.frames_to_time(pk, sr=sr, hop_length=hop)
    if return_strength:
        return times, props["peak_heights"], len(y) / sr
    return times, len(y) / sr


def group_points(hits, duration):
    points, cur = [], []
    for t in hits:
        if cur and t - cur[-1] > MAX_GAP:
            points.append(cur); cur = []
        cur.append(t)
    if cur: points.append(cur)
    out = []
    for p in points:
        if len(p) < MIN_HITS: continue
        s, e = max(0, p[0] - PAD_BEFORE), min(duration, p[-1] + PAD_AFTER)
        if out and s <= out[-1]["end"]:          # merge overlapping
            out[-1]["end"] = e; out[-1]["hits"] += len(p)
        else:
            out.append({"start": s, "end": e, "hits": len(p)})
    return out


def export(video, points, outdir):
    pdir = os.path.join(outdir, "points"); os.makedirs(pdir, exist_ok=True)
    files = []
    for i, p in enumerate(points, 1):
        f = os.path.join(pdir, f"point_{i:02d}.mp4")
        subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{p['start']:.2f}", "-to", f"{p['end']:.2f}",
                        "-i", video, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-c:a", "aac", "-movflags", "+faststart", f], check=True)
        files.append(f)
    lst = os.path.join(outdir, "list.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{os.path.abspath(f)}'\n" for f in files)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", os.path.join(outdir, "highlights.mp4")], check=True)
    os.remove(lst)
    with open(os.path.join(outdir, "points.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["point", "start_s", "end_s", "length_s", "hits"])
        for i, p in enumerate(points, 1):
            w.writerow([i, round(p["start"], 2), round(p["end"], 2), round(p["end"] - p["start"], 1), p["hits"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--out", default="output")
    a = ap.parse_args()
    hits, dur = detect_hits(a.video)
    pts = group_points(hits, dur)
    kept = sum(p["end"] - p["start"] for p in pts)
    print(f"{len(hits)} hits -> {len(pts)} points; kept {kept:.0f}s of {dur:.0f}s ({100*kept/dur:.0f}%)")
    for i, p in enumerate(pts, 1):
        print(f"  point {i:2d}: {p['start']:6.1f}-{p['end']:6.1f}s  ({p['hits']} hits)")
    export(a.video, pts, a.out)
    print("Saved to", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
