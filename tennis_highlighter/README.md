# Tennis Highlighter

Cuts the dead time out of a tennis match video by listening for racket hits,
then saves each point as its own clip plus one combined highlight reel.

## One-time setup (Mac)

Open Terminal and run:

```bash
# 1. Install Homebrew if you don't have it (https://brew.sh), then:
brew install python ffmpeg

# 2. Set up the app
cd ~/Desktop/tennis/tennis_highlighter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run it

```bash
cd ~/Desktop/tennis/tennis_highlighter
source .venv/bin/activate
python app.py
```

A browser tab opens. Drop in a video, click **Make highlights**.

Or from the command line, no browser:

```bash
python highlighter.py ../IMG_0222.MOV --out ../IMG_0222_highlights
```

## v2: label hits by stroke

This builds training data for a stroke classifier (forehand / backhand / serve / volley / overhead / not a hit).
If you set up before v2, run `pip install -r requirements.txt` again first.

```bash
source .venv/bin/activate
python label_app.py
```

1. Paste the video's path (drag the file into Terminal to copy it) and click **Find hits & poses**.
   It takes about 2 s per hit on the Intel Mac, and progress is saved, so you can stop and resume.
   The first run downloads a small 6 MB pose model.
2. Each hit loops. Press **F / B / S / V / O / X** to label it, **1 / 2** for the near or far player,
   **← →** to move, and **Space** for slow motion. Every label saves immediately to `data/<video>/labels.csv`.
3. When you've labelled a few hundred hits, click **Build training set**. That writes `data/dataset.npz` for Colab.

Clips and poses can also be extracted without the browser: `python poses.py ../trimmed.mov`.

## Tuning

- **Missing points / quiet hits not caught** → lower the hit sensitivity threshold.
- **Dead time sneaking in (people talking, ball bouncing)** → raise it.
- **One point split in two** → raise "max seconds between hits".
- **Two points merged** → lower it.

## How it works

1. Pulls the audio, keeps only the 1.5–6 kHz band where the racket "pop" lives.
2. Finds sharp loudness spikes = hits.
3. Hits less than 4.5 s apart belong to the same point; a point needs 2+ hits.
4. Adds 2 s before (serve) and after (ball dies), then cuts with ffmpeg.
