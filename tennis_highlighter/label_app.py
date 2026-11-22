"""v2 labelling tool: label each audio-detected hit by stroke type.

Run:  python label_app.py   (opens http://127.0.0.1:7861)
1. Paste a video path, click "Find hits & poses" (or pick a video you started before).
2. Each hit plays on a loop. Press a key or click a button; it saves and moves on.
3. Click "Build training set" -> data/dataset.npz for the Colab classifier.
Labels are saved to data/<video>/labels.csv after every click.
"""
import csv, datetime, glob, inspect, os
import gradio as gr
import highlighter as hl
import poses, strokes

DATA = os.path.abspath("data")
BUTTONS = [("forehand", "Forehand", "F"), ("backhand", "Backhand", "B"), ("serve", "Serve", "S"),
           ("volley", "Volley", "V"), ("overhead", "Overhead", "O"), ("not_a_hit", "Not a hit", "X"),
           ("opponent", "Opponent hit it", "U")]  # excluded from training
FIELDS = ["hit", "time_s", "strength", "label", "hitter", "labelled_at"]
TOTALS = {}  # session dir -> number of hits before the near-only filter

KEYS_JS = """
<script>
(() => {
  const keys = {f:'forehand', b:'backhand', s:'serve', v:'volley', o:'overhead', x:'not_a_hit', u:'opponent'};
  const click = id => { const el = document.getElementById(id); if (el) el.click(); };
  window.tennisSlow = false;
  document.addEventListener('keydown', e => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (keys[k]) { click('btn-' + keys[k]); e.preventDefault(); }
    else if (k === 'arrowleft') { click('btn-prev'); e.preventDefault(); }
    else if (k === 'arrowright') { click('btn-next'); e.preventDefault(); }
    else if (k === '1' || k === '2') {
      const el = document.querySelector('#hitter input[value="' + (k === '1' ? 'near' : 'far') + '"]');
      if (el) el.click();
    }
    else if (k === ' ') { window.tennisSlow = !window.tennisSlow; e.preventDefault(); }
  });
  // keep slow-motion applied when a new clip loads
  setInterval(() => {
    const v = document.querySelector('#clip video');
    if (v) v.playbackRate = window.tennisSlow ? 0.4 : 1.0;
  }, 250);
})();
</script>
"""


# ---------- data helpers ----------
def sessions():
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(DATA, "*", "hits.csv")))


def load_hits(sd):
    with open(os.path.join(sd, "hits.csv"), newline="") as fh:
        hits = [dict(r, hit=int(r["hit"])) for r in csv.DictReader(fh)]
    labels = strokes.read_labels(sd)
    TOTALS[sd] = len(hits)
    for h in hits:
        lab = labels.get(h["hit"])
        # only reuse a label if it belongs to the same moment (settings may have changed)
        if lab and abs(float(lab["time_s"]) - float(h["time_s"])) < 0.05:
            h["label"], h["hitter"] = lab["label"], lab["hitter"] or h["hitter_guess"]
        else:
            h["label"], h["hitter"] = "", h["hitter_guess"]
    if hl.NEAR_ONLY:
        # only ask about hits where the near player is swinging (keep anything already labelled)
        for h in hits:
            h["swing"] = round(strokes.swing_score(os.path.join(sd, "poses", f"hit_{h['hit']:04d}.npz")), 1)
            h["hitter"] = "near"
        hits = [h for h in hits if h["swing"] >= hl.NEAR_SWING or h["label"]]
    return hits


def save_labels(sd, hits):
    # merge with what's on disk so labels of hits not shown (filtered out) are kept
    rows = strokes.read_labels(sd)
    rows.update({h["hit"]: h for h in hits if h["label"]})
    tmp = os.path.join(sd, "labels.csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore"); w.writeheader()
        w.writerows(rows[k] for k in sorted(rows))
    os.replace(tmp, os.path.join(sd, "labels.csv"))


def first_unlabelled(hits):
    return next((i for i, h in enumerate(hits) if not h["label"]), 0)


def counts_md(hits):
    done = sum(1 for h in hits if h["label"])
    per = {k: sum(1 for h in hits if h["label"] == k) for k, _, _ in BUTTONS}
    return f"**{done}/{len(hits)} labelled** · " + " · ".join(
        f"{name} {per[k]}" for k, name, _ in BUTTONS)


# ---------- UI callbacks ----------
def show(sd, hits, i):
    if not hits:
        return "No video loaded.", None, None, "near"
    i = max(0, min(i, len(hits) - 1))
    h = hits[i]
    name = f"hit_{h['hit']:04d}"
    cur = next((n for k, n, _ in BUTTONS if k == h["label"]), "—")
    if hl.NEAR_ONLY:
        who = (f"near player's arm speed {h['swing']} · showing {len(hits)} of "
               f"{TOTALS.get(sd, len(hits))} hits (near player swinging)")
    else:
        who = f"loudness {h['strength']}× (guess: {h['hitter_guess']} player)"
    info = (f"### Hit {i + 1} of {len(hits)} · at {float(h['time_s']):.1f}s · {who}\n"
            f"Current label: **{cur}**\n\n{counts_md(hits)}")
    return (info, os.path.join(sd, "clips", name + ".mp4"),
            os.path.join(sd, "frames", name + ".jpg"), h["hitter"])


def open_session(name):
    if not name:
        raise gr.Error("Pick a video first.")
    sd = os.path.join(DATA, name)
    hits = load_hits(sd)
    i = first_unlabelled(hits)
    return (sd, hits, i) + show(sd, hits, i)


def extract(path, progress=gr.Progress()):
    path = (path or "").strip().strip("'\"")
    if not os.path.isfile(path):
        raise gr.Error(f"File not found: {path}")
    sd, n = poses.extract(path, DATA, progress=lambda f, d: progress(f, desc=d))
    if n == 0:
        raise gr.Error("No hits found in this video.")
    name = os.path.basename(sd)
    return (gr.update(choices=sessions(), value=name),) + open_session(name)


def make_label(key):
    def fn(sd, hits, i, hitter):
        if not hits:
            raise gr.Error("Load a video first.")
        hits[i].update(label=key, hitter=hitter,
                       labelled_at=datetime.datetime.now().isoformat(timespec="seconds"))
        save_labels(sd, hits)
        j = min(i + 1, len(hits) - 1)
        return (hits, j) + show(sd, hits, j)
    return fn


def set_hitter(sd, hits, i, hitter):
    # changing near/far on an already-labelled hit should stick
    if hits and hits[i]["label"] and hits[i]["hitter"] != hitter:
        hits[i]["hitter"] = hitter
        save_labels(sd, hits)
    elif hits:
        hits[i]["hitter"] = hitter
    return hits


def step(delta):
    def fn(sd, hits, i):
        j = max(0, min(i + delta, len(hits) - 1)) if hits else 0
        return (j,) + show(sd, hits, j)
    return fn


def jump_unlabelled(sd, hits, i):
    j = first_unlabelled(hits) if hits else 0
    return (j,) + show(sd, hits, j)


def export():
    try:
        out, n, counts = strokes.export_dataset(DATA)
    except ValueError as e:
        raise gr.Error(str(e))
    rows = "\n".join(f"| {k} | {v} |" for k, v in counts.items())
    return (f"Saved **{n}** labelled hits to `{out}`. Upload it to Colab to train.\n\n"
            f"| label | count |\n|---|---|\n{rows}")


# ---------- layout ----------
_blocks_kw = {"title": "Tennis Hit Labeller"}
_launch_kw = {"inbrowser": True, "server_port": 7861, "allowed_paths": [DATA]}
# Gradio 5 takes `head` on Blocks, Gradio 6 on launch()
(_blocks_kw if "head" in inspect.signature(gr.Blocks.__init__).parameters else _launch_kw)["head"] = KEYS_JS

with gr.Blocks(**_blocks_kw) as demo:
    sd_state, hits_state, idx_state = gr.State(""), gr.State([]), gr.State(0)
    gr.Markdown("# 🎾 Tennis Hit Labeller\nLabel each detected hit so we can train a stroke classifier.")
    with gr.Accordion("1 · Choose a video", open=True):
        with gr.Row():
            path = gr.Textbox(label="Video file path (tip: drag the file into Terminal to copy its path)",
                              placeholder="/Users/you/Desktop/tennis/trimmed.mov", scale=4)
            extract_btn = gr.Button("Find hits & poses", variant="primary", scale=1)
        with gr.Row():
            session_dd = gr.Dropdown(sessions(), label="…or continue a video you started", scale=4)
            open_btn = gr.Button("Open", scale=1)

    info = gr.Markdown("No video loaded.")
    with gr.Row():
        clip = gr.Video(label="Hit clip (loops) · Space = slow motion", autoplay=True, loop=True,
                        elem_id="clip", height=360, scale=3)
        frame = gr.Image(label="Moment of the hit, with detected skeletons", height=360, scale=2)
    hitter = gr.Radio(strokes.HITTERS, value="near", label="Who hit it? (keys 1 / 2)", elem_id="hitter",
                      visible=not hl.NEAR_ONLY)
    with gr.Row():
        label_btns = [gr.Button(f"{name} ({k})", elem_id=f"btn-{key}",
                                variant="stop" if key == "not_a_hit" else "secondary")
                      for key, name, k in BUTTONS]
    with gr.Row():
        prev_btn = gr.Button("← Previous", elem_id="btn-prev")
        unl_btn = gr.Button("Next unlabelled")
        next_btn = gr.Button("Next / skip →", elem_id="btn-next")
    gr.Markdown("Keys: **F** forehand · **B** backhand · **S** serve · **V** volley · **O** overhead · "
                "**X** not a hit (no ball struck) · **U** opponent hit it, not the near player (skipped in training) · **← →** move · **Space** slow motion. "
                "Each label saves immediately.")
    with gr.Accordion("3 · Build training set", open=False):
        export_btn = gr.Button("Build training set (data/dataset.npz)")
        export_md = gr.Markdown()

    view = [info, clip, frame, hitter]
    extract_btn.click(extract, [path], [session_dd, sd_state, hits_state, idx_state] + view)
    open_btn.click(open_session, [session_dd], [sd_state, hits_state, idx_state] + view)
    for (key, _, _), b in zip(BUTTONS, label_btns):
        b.click(make_label(key), [sd_state, hits_state, idx_state, hitter], [hits_state, idx_state] + view)
    hitter.input(set_hitter, [sd_state, hits_state, idx_state, hitter], [hits_state])
    prev_btn.click(step(-1), [sd_state, hits_state, idx_state], [idx_state] + view)
    next_btn.click(step(1), [sd_state, hits_state, idx_state], [idx_state] + view)
    unl_btn.click(jump_unlabelled, [sd_state, hits_state, idx_state], [idx_state] + view)
    export_btn.click(export, None, [export_md])

if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    demo.launch(**_launch_kw)
