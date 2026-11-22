"""Browser UI for Tennis Highlighter. Run: python app.py  (opens http://127.0.0.1:7860)"""
import os, tempfile
import gradio as gr
import highlighter as hl


def process(video, threshold, max_gap):
    if not video:
        raise gr.Error("Upload a video first.")
    hl.HIT_THRESHOLD, hl.MAX_GAP = threshold, max_gap
    out = tempfile.mkdtemp(prefix="tennis_")
    hits, dur = hl.detect_hits(video)
    pts = hl.group_points(hits, dur)
    if not pts:
        raise gr.Error("No points found. Try lowering the hit sensitivity threshold.")
    hl.export(video, pts, out)
    kept = sum(p["end"] - p["start"] for p in pts)
    summary = (f"Found {len(pts)} points from {len(hits)} hits. "
               f"Kept {kept/60:.1f} of {dur/60:.1f} min ({100*kept/dur:.0f}%).")
    table = [[i, f"{p['start']:.1f}s", f"{p['end']:.1f}s", f"{p['end']-p['start']:.0f}s", p["hits"]]
             for i, p in enumerate(pts, 1)]
    clips = sorted(os.path.join(out, "points", f) for f in os.listdir(os.path.join(out, "points")))
    return summary, os.path.join(out, "highlights.mp4"), table, clips


with gr.Blocks(title="Tennis Highlighter") as demo:
    gr.Markdown("# 🎾 Tennis Highlighter\nUpload a match video. Dead time is cut out and each point is saved as its own clip.")
    with gr.Row():
        with gr.Column():
            vid = gr.Video(label="Match video")
            thr = gr.Slider(3, 12, value=hl.HIT_THRESHOLD, step=0.5,
                            label="Hit sensitivity threshold (lower = catches quieter hits)")
            gap = gr.Slider(2, 8, value=hl.MAX_GAP, step=0.5,
                            label="Max seconds between hits within one point")
            btn = gr.Button("Make highlights", variant="primary")
        with gr.Column():
            summary = gr.Markdown()
            reel = gr.Video(label="Highlight reel")
    table = gr.Dataframe(headers=["Point", "Start", "End", "Length", "Hits"], label="Points")
    clips = gr.File(label="Individual point clips", file_count="multiple")
    btn.click(process, [vid, thr, gap], [summary, reel, table, clips])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
