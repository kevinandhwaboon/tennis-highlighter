# Tennis Highlighter

Free, local, SwingVision-style tool: upload a raw tennis match video, get back
clips of each point with dead time cut out, later labelled by stroke and in/out.

Repo: https://github.com/kevinandhwaboon/tennis-highlighter

## Constraints
- **Users:** recreational singles players.
- **Footage:** iPhone, 30 fps, 1080p `.MOV`, H.264 + AAC audio. Full matches are about 3 GB.
- **Hardware:** Intel MacBook with Iris Plus 645, so **no CUDA and no MPS**. Everything runs on CPU.
  Keep inference light (downscale to about 640px, skip frames). Do any training in Google Colab and bring the weights back.
- **Cost:** must stay free. No paid APIs or cloud hosting.
- **UI:** Gradio app running locally in the browser (`python app.py`).

## Layout
- `highlighter.py`: core pipeline plus a command-line interface (`python highlighter.py VIDEO --out DIR`)
- `app.py`: Gradio UI wrapping highlighter.py
- `poses.py`: v2, cuts a 640px clip around each hit and runs YOLO pose. Writes `data/<video>/` (`python poses.py VIDEO`)
- `strokes.py`: v2, label list, hitter tracking, feature normalisation, dataset export (`python strokes.py`, which writes `data/dataset.npz`)
- `label_app.py`: v2 Gradio labelling tool (`python label_app.py`, port 7861)
- `train_strokes.ipynb`: v2 training notebook (Colab or local). Reads `dataset.npz` and writes `stroke_model.pt`
- `classify.py`: v2, labels every hit with the near player's stroke using `stroke_model.pt`, and writes `data/<video>/strokes.csv` (`python classify.py VIDEO`).
  `python highlighter.py VIDEO --out DIR --strokes` does the same and adds stroke counts and a sequence per point to `points.csv`, plus `DIR/hits.csv`.
- `docs/PLAN.md`: roadmap and findings

## v1 (done): cut dead time, audio only
1. ffmpeg extracts mono 22.05 kHz audio.
2. Bandpass to 1.5–6 kHz, where the racket "pop" lives, then librosa onset strength, normalised by its median.
3. Peaks above 6× median, at least 0.5 s apart, count as hits.
4. Hits less than 4.5 s apart belong to the same point. A point needs 2 or more hits.
5. Pad 2 s before and after the point, then cut with ffmpeg (libx264 veryfast, crf 23) and concat into highlights.mp4.

Results on the 2:08 sample: 36 hits gave 6 points, keeping 60% of the footage. All boundaries were checked visually and are correct, with about 2 s of extra tail.
Near-player hits peak at about 10–14× median and far-player hits at about 5–7×.

## v2 (in progress): stroke classification
Labelling tool is built; next is labelling a few hundred hits, then training.
- **Hits:** every audio hit (not only those inside points), so isolated false positives get labelled `not_a_hit`.
- **Pose:** `yolo11n-pose` on CPU, 640px clip from 1.5 s before to 1.0 s after each hit, every 2nd frame (15 pose fps).
  Keeps up to 4 people per frame (largest boxes) so the hitter can be re-picked later.
- **Labels:** `data/<video>/labels.csv` with columns hit, time_s, strength, label, hitter (near/far), labelled_at. Saved after every click.
  Labels are matched to hits by time, so changing the detection settings doesn't attach labels to the wrong hit.
- **Hitter guess:** audio peak of 9× median or more means near player, otherwise far. The labeller corrects it with keys 1/2.
- **First real-footage run** (`trimmed.mov`, 36 hits): about 5.3 s per hit on the Intel Mac (190 s for 36 hits, plus audio and a one-time model download).
  Exactly one person is detected per frame, and it is the **near** player (back to the camera, makes the loud hits). The far player is off-screen.
- **Near player only** (`NEAR_ONLY`): we only classify the near player's strokes. Loudness is a poor guide to who hit it (many near swings are only 6–8×).
  Instead `strokes.swing_score` measures the near player's peak wrist speed within ±0.35 s of the hit, in body heights per second.
  Near swings score about 2.5–9 and non-swings under 1.7, so label_app only shows hits scoring at least `NEAR_SWING` = 2.0 (244 of 393 on IMG_0222).
  The near/far control is hidden. Key U = `opponent` (the near player didn't hit it), which `export_dataset` skips but could later become an "opponent hit" class.
- **Features** (`strokes.features`): follow the hitter (near = box bottom lowest in the frame at the hit, far = highest, then nearest box centre frame to frame).
  Resample to ±1.0 s at 15 fps (31 steps). Per step, use keypoints relative to mid-hip divided by torso length, plus 17 confidences, plus hip displacement from the hit frame: 53 features in total.
  **The far player is mirrored** (flip x and swap left/right keypoints), because they face the camera.
  Left-handers aren't handled yet. They'll need a per-player flip.
- **Dataset:** `data/dataset.npz` with X (N, 31, 53), y, labels, hitter, video, hit, time_s.
- **Training** (`train_strokes.ipynb`): a small 1D CNN (2 conv layers of 64, about 40k params) with class weights and time-shift/noise augmentation.
  Classes with no examples are dropped. It is evaluated with 5-fold cross-validation grouped by 30 s video blocks, so neighbouring hits never sit on both sides of a split.
  Then it trains on everything and saves `stroke_model.pt` (state_dict, labels, mean/std; loadable with `weights_only=True`). Training takes about 40 s on the Mac CPU.
- **First result** (IMG_0222, practice rallying, 233 near-player hits: 168 forehand, 49 backhand, 16 not_a_hit, no serves or volleys): 95% cross-validated accuracy, against 72% for always guessing forehand.
  Forehand and backhand are 97–98% F1. not_a_hit is 0.71 F1, but with only 16 examples that score is noisy.
- **Inference** (`classify.py`): swing score < `NEAR_SWING` gives `opponent`, the near player missing in most frames gives `unknown`, and otherwise the model predicts with a confidence.
  `strokes.make_model` must stay identical to the notebook's `make_model` (checked through the `arch` string in the .pt).
  With cached poses it takes about 50 s for 393 hits. On a new video, pose extraction (about 5 s per hit) dominates.
- **Next:** label match footage to add serves and volleys, then retrain. Until then, serves get forced into forehand, backhand or not_a_hit.

## Later: v3, ball tracking and line calls
- Fine-tune TrackNet on our footage, detect court keypoints, compute a homography, and project bounces onto the court.
- Report "clearly in / clearly out / too close to call" based on estimated error.
- **Needs a raised camera behind the baseline with the court lines visible.**
  The current sample was filmed from ground level into the sun: silhouettes, no lines, and only the near player in frame. Line calls won't work from that setup.

## Conventions
- Never commit video files (`*.mov`, `*.mp4`) or model weights over 50 MB. See `.gitignore`.
- Keep tunable settings as constants at the top of `highlighter.py`.
- Test on `~/Desktop/tennis/trimmed.mov` (2 min sample) before running full matches.
