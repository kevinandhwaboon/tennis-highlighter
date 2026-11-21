# Tennis Highlighter (v1)

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
