# Tennis Highlighter — plan & status

**Goal:** free SwingVision-style tool for recreational singles. Phone video, 30 fps. Runs locally on an Intel Mac (Iris Plus 645, no usable GPU) as a Python + Gradio browser app.

## Roadmap
- **v1 (done):** cut dead time. Audio-only: racket "pop" in the 1.5–6 kHz band → onset peaks (>6× median) → hits <4.5 s apart grouped into a point (2+ hits) → 2 s padding → ffmpeg clips + highlight reel. No training.
- **v2 (next):** stroke type (forehand/backhand/serve/volley/overhead, plus not-a-hit) from YOLO-pose sequences around each detected hit. Gradio labelling tool, then a small classifier trained in Colab on a few hundred labelled hits.
- **v3:** ball tracking (TrackNet fine-tune) + in/out calls with "clear / too close" confidence. Needs an elevated camera behind the baseline that shows the court lines.

## Findings from sample (trimmed.mov, 2:08)
- 36 hits → 6 points, kept 60%. All boundaries checked visually and correct (about 2 s of extra tail).
- Processing: about 45 s for a 2 min clip, mostly re-encoding.
- Camera was at ground level, into the sun: silhouettes, court lines not visible, only the far player in frame. That's fine for v1, but v3 line calls are impossible from this angle.

## Repo
https://github.com/kevinandhwaboon/tennis-highlighter
