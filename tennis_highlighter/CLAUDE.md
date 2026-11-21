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
- `docs/PLAN.md`: roadmap and findings

## v1 (done): cut dead time, audio only
1. ffmpeg extracts mono 22.05 kHz audio.
2. Bandpass to 1.5–6 kHz, where the racket "pop" lives, then librosa onset strength, normalised by its median.
3. Peaks above 6× median, at least 0.5 s apart, count as hits.
4. Hits less than 4.5 s apart belong to the same point. A point needs 2 or more hits.
5. Pad 2 s before and after the point, then cut with ffmpeg (libx264 veryfast, crf 23) and concat into highlights.mp4.

Results on the 2:08 sample: 36 hits gave 6 points, keeping 60% of the footage. All boundaries were checked visually and are correct, with about 2 s of extra tail.
Near-player hits peak at about 10–14× median and far-player hits at about 5–7×.

## Next: v2, stroke classification
- Run Ultralytics YOLO pose (CPU, small model) in a window of about ±1 s around each detected hit.
- Build a labelling tool (Gradio) that shows each hit clip with buttons for
  forehand / backhand / serve / volley / overhead / not-a-hit. Save the labels to CSV.
- Train a small sequence classifier (1D CNN or GRU on keypoints) in Colab, then run it on CPU locally.
- Label "not-a-hit" too, so audio false positives get filtered out.

## Later: v3, ball tracking and line calls
- Fine-tune TrackNet on our footage, detect court keypoints, compute a homography, and project bounces onto the court.
- Report "clearly in / clearly out / too close to call" based on estimated error.
- **Needs a raised camera behind the baseline with the court lines visible.**
  The current sample was filmed from ground level into the sun: silhouettes, no lines, and only the far player in frame. Line calls won't work from that setup.

## Conventions
- Never commit video files (`*.mov`, `*.mp4`) or model weights over 50 MB. See `.gitignore`.
- Keep tunable settings as constants at the top of `highlighter.py`.
- Test on `~/Desktop/tennis/trimmed.mov` (2 min sample) before running full matches.
